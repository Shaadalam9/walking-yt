"""Video download, temporal sampling, and InternVL verification."""

from __future__ import annotations

import gc
import math
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from . import settings
from .cut_detection import build_segments, run_ffmpeg_scene_detection
from .model_loading import load_model_with_fallback
from .shared import (
    clamp_float,
    clean_text,
    log,
    normalise_bool,
    normalise_string_list,
    recover_json,
    require_binary,
    run_command,
    save_state,
    unload_model,
)


@dataclass
class VisualSample:
    sample_index: int
    start_time: float
    midpoint: float
    include: bool
    confidence: float
    is_walking_video: bool
    camera_motion: str
    time_of_day: str
    quality_issues: List[str]
    short_reason: str
    raw_response: str


def find_downloaded_video(video_id: str) -> Optional[Path]:
    if not settings.VIDEO_DIR.exists():
        return None
    candidates = sorted(
        path
        for path in settings.VIDEO_DIR.glob(f"{video_id}.*")
        if path.suffix.lower() in settings.VIDEO_EXTENSIONS
    )
    return candidates[0] if candidates else None


def download_video(video_id: str) -> Optional[Path]:
    require_binary("yt-dlp")
    existing = find_downloaded_video(video_id)
    if existing:
        return existing

    settings.VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    output_template = str(settings.VIDEO_DIR / f"{video_id}.%(ext)s")
    url = f"https://www.youtube.com/watch?v={video_id}"
    result = run_command(
        [
            "yt-dlp",
            "--no-playlist",
            "--force-overwrites",
            "--no-warnings",
            "-f",
            settings.VIDEO_FORMAT,
            "--merge-output-format",
            "mp4",
            "-o",
            output_template,
            url,
        ],
        timeout=1800,
    )
    if result.returncode != 0:
        log(f"Download failed for {video_id}: {result.stderr.strip()}")
        return None
    return find_downloaded_video(video_id)


def get_video_duration(video_path: Path) -> Optional[float]:
    require_binary("ffprobe")
    result = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        timeout=60,
    )
    if result.returncode != 0:
        return None
    try:
        duration = float(result.stdout.strip())
    except ValueError:
        return None
    return duration if duration > 0 and math.isfinite(duration) else None


def temporal_sample_start_times(duration: float) -> List[float]:
    usable_end = max(0.0, duration - settings.SAMPLE_SECONDS)
    if settings.TEMPORAL_SAMPLES <= 1:
        return [usable_end / 2.0]

    fractions = [
        0.05 + (0.90 * index / (settings.TEMPORAL_SAMPLES - 1))
        for index in range(settings.TEMPORAL_SAMPLES)
    ]
    starts = [min(usable_end, max(0.0, usable_end * f)) for f in fractions]

    unique: List[float] = []
    seen: set[float] = set()
    for start in starts:
        rounded = round(start, 3)
        if rounded not in seen:
            unique.append(start)
            seen.add(rounded)
    return unique


def create_sample_clip(
    video_path: Path, start_time: float, output_path: Path
) -> bool:
    require_binary("ffmpeg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = run_command(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start_time:.3f}",
            "-t",
            str(settings.SAMPLE_SECONDS),
            "-i",
            str(video_path),
            "-an",
            "-vf",
            "scale='min(480,iw)':-2",
            "-y",
            str(output_path),
        ],
        timeout=240,
    )
    return (
        result.returncode == 0
        and output_path.exists()
        and output_path.stat().st_size > 0
    )


class InternVLVisualJudge:
    def __init__(self, model_name: str, device: str | None = None) -> None:
        from transformers import (  # type: ignore
            AutoModelForImageTextToText,
            AutoProcessor,
        )

        selected_device = device or settings.SEQUENTIAL_DEVICE
        log(f"Loading visual model: {model_name}")
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = load_model_with_fallback(
            AutoModelForImageTextToText.from_pretrained,
            model_name,
            device=selected_device,
            load_in_4bit=settings.VLM_LOAD_IN_4BIT,
            model_label="visual model",
        ).eval()
        log("Visual model loaded")

    def judge(
        self,
        sample_path: Path,
        metadata: Dict[str, Any],
        sample_index: int,
        start_time: float,
    ) -> VisualSample:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "url": str(sample_path)},
                    {"type": "text", "text": self._build_prompt(metadata)},
                ],
            }
        ]
        try:
            answer = self._generate(messages)
        except Exception as exc:
            return self._model_error(sample_index, start_time, exc)

        data = recover_json(answer) or {}
        time_label = clean_text(data.get("time_of_day")).lower()
        if time_label in {"dawn", "dusk", "twilight", "dawn or dusk"}:
            time_label = "dawn_dusk"
        if time_label not in settings.TIME_OF_DAY_CODES:
            time_label = "unknown"

        issues = normalise_string_list(data.get("quality_issues"))
        return VisualSample(
            sample_index=sample_index,
            start_time=start_time,
            midpoint=start_time + settings.SAMPLE_SECONDS / 2.0,
            include=normalise_bool(data.get("include")),
            confidence=clamp_float(data.get("confidence"), 0.0, 1.0),
            is_walking_video=normalise_bool(
                data.get("is_walking_video")
            ),
            camera_motion=clean_text(
                data.get("camera_motion") or "unclear"
            ).lower(),
            time_of_day=time_label,
            quality_issues=issues or ["none"],
            short_reason=clean_text(data.get("short_reason")),
            raw_response=answer,
        )

    def _generate(self, messages: List[Dict[str, Any]]) -> str:
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            num_frames=settings.FRAMES_PER_SAMPLE,
        ).to(self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=settings.VLM_MAX_NEW_TOKENS,
                do_sample=False,
            )
        prompt_length = inputs["input_ids"].shape[1]
        return self.processor.decode(
            generated[0, prompt_length:], skip_special_tokens=True
        )

    @staticmethod
    def _model_error(
        sample_index: int, start_time: float, exc: Exception
    ) -> VisualSample:
        return VisualSample(
            sample_index=sample_index,
            start_time=start_time,
            midpoint=start_time + settings.SAMPLE_SECONDS / 2.0,
            include=False,
            confidence=0.0,
            is_walking_video=False,
            camera_motion="unclear",
            time_of_day="unknown",
            quality_issues=["model_error"],
            short_reason=f"Visual model failed: {exc}",
            raw_response="",
        )

    @staticmethod
    def _build_prompt(metadata: Dict[str, Any]) -> str:
        title = clean_text(metadata.get("title"))
        description = clean_text(metadata.get("description"))[:1200]
        return f"""
You are visually verifying a research dataset candidate.

Include only when the sample shows a camera moving through a real physical
location as a pedestrian walking video. Reject vehicles, bicycles, drones,
boats, trains, buses, static scenes, talking heads, games, animation,
slideshows, maps, screen recordings, intros, and heavily edited montages.

Classify time_of_day as day, night, dawn_dusk, or unknown. Indoor footage with
no reliable external time evidence must be unknown.

Title: {title}
Description excerpt: {description}

Return valid JSON only:
{{
  "include": true,
  "confidence": 0.0,
  "is_walking_video": true,
  "camera_motion": "walking",
  "time_of_day": "day",
  "quality_issues": ["none"],
  "short_reason": "one short sentence"
}}
""".strip()


def aggregate_visual_samples(samples: List[VisualSample]) -> Dict[str, Any]:
    if not samples:
        return {
            "include": False,
            "confidence": 0.0,
            "included_sample_ratio": 0.0,
            "short_reason": "No temporal samples were available.",
            "samples": [],
        }

    valid_samples = [
        sample
        for sample in samples
        if "model_error" not in sample.quality_issues
    ]
    if not valid_samples:
        return {
            "include": False,
            "confidence": 0.0,
            "included_sample_ratio": 0.0,
            "short_reason": "Every visual model call failed.",
            "samples": [asdict(sample) for sample in samples],
            "error": "all_visual_calls_failed",
        }

    included_count = sum(
        1
        for sample in valid_samples
        if sample.include and sample.is_walking_video
    )
    included_ratio = included_count / len(valid_samples)
    mean_confidence = sum(
        sample.confidence for sample in valid_samples
    ) / len(valid_samples)
    include = (
        included_ratio >= settings.MIN_INCLUDED_SAMPLE_RATIO
        and mean_confidence >= settings.MIN_VISUAL_CONFIDENCE
    )

    if include:
        reason = "Accepted as a pedestrian walking video."
    elif included_ratio < settings.MIN_INCLUDED_SAMPLE_RATIO:
        reason = "Too few temporal samples show pedestrian walking."
    else:
        reason = "The mean visual confidence is too low."

    return {
        "include": include,
        "confidence": round(mean_confidence, 4),
        "included_samples": included_count,
        "total_samples": len(valid_samples),
        "included_sample_ratio": round(included_ratio, 4),
        "short_reason": reason,
        "samples": [asdict(sample) for sample in samples],
        "error": None,
    }


def process_visual_candidate(
    video_id: str,
    record: Dict[str, Any],
    judge: InternVLVisualJudge,
) -> None:
    metadata = record.get("metadata", {})
    log(f"Downloading accepted metadata candidate {video_id}")
    video_path = download_video(video_id)
    if video_path is None:
        record["status"] = "download_failed"
        record["error"] = "yt-dlp could not download the video"
        return

    record["local_video_path"] = str(video_path)
    duration = get_video_duration(video_path)
    if duration is None:
        record["status"] = "duration_failed"
        record["error"] = "ffprobe could not determine the duration"
        remove_video_if_required(video_path)
        return

    record["duration_seconds"] = round(duration, 3)
    samples = _judge_temporal_samples(
        video_id, video_path, duration, metadata, judge
    )
    visual_decision = aggregate_visual_samples(samples)
    record["visual_decision"] = visual_decision
    if not visual_decision.get("include"):
        record["status"] = (
            "visual_error"
            if visual_decision.get("error")
            else "visual_rejected"
        )
        record["error"] = visual_decision.get("error")
        remove_video_if_required(video_path)
        return

    log(f"Detecting cuts in {video_id}")
    cut_result = run_ffmpeg_scene_detection(video_path, duration)
    record["cuts"] = {
        "threshold": settings.SCENE_THRESHOLD,
        "merge_nearby_seconds": settings.MERGE_NEARBY_SEC,
        "cut_times": [
            round(value, 3) for value in cut_result.cut_times
        ],
        "message": cut_result.message,
    }
    if cut_result.message:
        record["status"] = "cut_detection_failed"
        record["error"] = cut_result.message
        return

    record["segments"] = build_segments(
        duration=duration,
        cut_times=cut_result.cut_times,
        visual_samples=visual_decision.get("samples", []),
    )
    record["status"] = "complete"
    record["error"] = None


def _judge_temporal_samples(
    video_id: str,
    video_path: Path,
    duration: float,
    metadata: Dict[str, Any],
    judge: InternVLVisualJudge,
) -> List[VisualSample]:
    samples: List[VisualSample] = []
    with tempfile.TemporaryDirectory(
        prefix=f"walk_samples_{video_id}_"
    ) as temporary_name:
        temporary_dir = Path(temporary_name)
        for sample_index, start_time in enumerate(
            temporal_sample_start_times(duration)
        ):
            sample_path = temporary_dir / f"sample_{sample_index:02d}.mp4"
            if not create_sample_clip(video_path, start_time, sample_path):
                log(
                    f"Could not create temporal sample {sample_index} "
                    f"for {video_id}"
                )
                continue

            sample = judge.judge(
                sample_path, metadata, sample_index, start_time
            )
            samples.append(sample)
            log(
                f"{video_id} sample {sample_index}: "
                f"include={sample.include}, "
                f"confidence={sample.confidence:.2f}, "
                f"time={sample.time_of_day}"
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return samples


def remove_video_if_required(video_path: Optional[Path]) -> None:
    if settings.KEEP_REJECTED_VIDEOS or video_path is None:
        return
    try:
        video_path.unlink()
    except OSError:
        pass


def run_visual_stage(
    state: Dict[str, Any],
    after_video: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> int:
    pending: List[Tuple[str, Dict[str, Any]]] = []
    for video_id, record in state.get("videos", {}).items():
        text_decision = record.get("text_decision")
        if not isinstance(text_decision, dict):
            continue
        if not text_decision.get("include"):
            continue
        if record.get("status") in {"complete", "visual_rejected"}:
            continue
        pending.append((video_id, record))

    if settings.MAX_VIDEOS_PER_RUN is not None:
        pending = pending[: settings.MAX_VIDEOS_PER_RUN]
    if not pending:
        return 0

    judge = InternVLVisualJudge(settings.VLM_MODEL_NAME)
    processed = 0
    try:
        for video_id, record in pending:
            try:
                process_visual_candidate(video_id, record, judge)
            except KeyboardInterrupt:
                save_state(state)
                raise
            except Exception as exc:
                record["status"] = "pipeline_error"
                record["error"] = str(exc)
                log(f"Visual pipeline failed for {video_id}: {exc}")

            save_state(state)
            if after_video:
                after_video(state)
            processed += 1
    finally:
        unload_model(judge)
    return processed
