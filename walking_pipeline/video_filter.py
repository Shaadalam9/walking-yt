"""Video download, cut verification, and segment level VLM review."""

from __future__ import annotations

import gc
import math
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from . import settings
from .cut_detection import (
    build_segments,
    run_ffmpeg_scene_detection,
    segment_duration_seconds,
    split_long_segments,
)
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


_SEGMENT_CONTENT_TYPES = {
    "walking",
    "advertisement",
    "channel_promotion",
    "intro_highlights",
    "nonwalking",
    "unclear",
}


@dataclass
class CutVerification:
    cut_time: float
    clip_start_time: float
    boundary_time_in_clip: float
    is_real_cut: bool
    confidence: float
    transition_type: str
    short_reason: str
    raw_response: str
    error: Optional[str] = None


@dataclass
class ContentStartDecision:
    main_content_start_time: Optional[float]
    confidence: float
    short_reason: str
    raw_response: str
    error: Optional[str] = None


@dataclass
class SegmentReview:
    segment_index: int
    start_time: int
    end_time: int
    include: bool
    confidence: float
    is_walking_video: bool
    content_type: str
    is_advertisement: bool
    is_intro_highlights: bool
    time_of_day: str
    quality_issues: List[str]
    short_reason: str
    raw_response: str
    walking_fraction: float = 0.0
    promotion_fraction: float = 0.0
    decision_method: str = "single_label"
    error: Optional[str] = None


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
        "--continue",
        "--no-warnings",
        "--cookies-from-browser",
        "chrome",
        "--remote-components",
        "ejs:github",
        "-f",
        settings.VIDEO_FORMAT,
        "-S",
        "res:720,fps",
        "--merge-output-format",
        "mp4",
        "--remux-video",
        "mp4",
        "-o",
        output_template,
        url,
    ],
    timeout=7200,
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


def get_video_frame_count(video_path: Path) -> Optional[int]:
    """Return the number of decodable video frames."""
    require_binary("ffprobe")
    result = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        timeout=60,
    )
    if result.returncode != 0:
        return None
    try:
        frame_count = int(result.stdout.strip())
    except ValueError:
        return None
    return frame_count if frame_count > 0 else None


def create_sample_clip(
    video_path: Path,
    start_time: float,
    output_path: Path,
    clip_seconds: Optional[float] = None,
) -> bool:
    """Create a continuous clip used to inspect an FFmpeg cut boundary."""
    require_binary("ffmpeg")
    duration = clip_seconds or float(settings.CUT_VERIFICATION_SECONDS)
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
            f"{duration:.3f}",
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


def create_content_start_clip(
    video_path: Path,
    output_path: Path,
    clip_seconds: float,
) -> bool:
    """Create a continuous low resolution opening clip for temporal grounding."""
    require_binary("ffmpeg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = run_command(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            "0",
            "-t",
            f"{clip_seconds:.3f}",
            "-i",
            str(video_path),
            "-an",
            "-vf",
            (
                f"fps={settings.COSMOS3_INTRO_FPS:.10f},"
                f"scale='min({settings.COSMOS3_VIDEO_WIDTH},iw)':-2,"
                f"setpts=N/({settings.COSMOS3_INTRO_FPS:.10f}*TB)"
            ),
            "-preset",
            "veryfast",
            "-crf",
            "28",
            "-r",
            f"{settings.COSMOS3_INTRO_FPS:.10f}",
            "-y",
            str(output_path),
        ],
        timeout=600,
    )
    return (
        result.returncode == 0
        and output_path.exists()
        and output_path.stat().st_size > 0
    )


def create_segment_review_clip(
    video_path: Path,
    start_time: int,
    end_time: int,
    output_path: Path,
    target_frame_count: Optional[int] = None,
    maximum_width: int = 480,
) -> bool:
    """Create a compact clip sampling frames across one entire segment."""
    require_binary("ffmpeg")
    segment_duration = max(1.0, float(end_time - start_time + 1))
    requested_frame_count = max(
        1,
        target_frame_count or settings.FRAMES_PER_SAMPLE,
    )
    encoded_frame_count = requested_frame_count
    sample_rate = encoded_frame_count / segment_duration
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = run_command(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{float(start_time):.3f}",
            "-t",
            f"{segment_duration:.3f}",
            "-skip_frame",
            "nokey",
            "-i",
            str(video_path),
            "-an",
            "-vf",
            (
                f"fps={sample_rate:.10f},"
                f"scale='min({maximum_width},iw)':-2,"
                "setpts=N/(1*TB),"
                f"tpad=stop_mode=clone:stop_duration={encoded_frame_count}"
            ),
            "-frames:v",
            str(encoded_frame_count),
            "-r",
            "1",
            "-y",
            str(output_path),
        ],
        timeout=600,
    )
    return (
        result.returncode == 0
        and output_path.exists()
        and output_path.stat().st_size > 0
    )


def cut_verification_window(
    cut_time: float, duration: float
) -> tuple[float, float, float]:
    """Return clip start, duration, and boundary offset for a candidate cut."""
    clip_duration = min(float(settings.CUT_VERIFICATION_SECONDS), duration)
    latest_start = max(0.0, duration - clip_duration)
    clip_start = min(max(0.0, cut_time - clip_duration / 2.0), latest_start)
    boundary_offset = min(clip_duration, max(0.0, cut_time - clip_start))
    return clip_start, clip_duration, boundary_offset


class InternVLVisualJudge:
    def __init__(self, model_name: str, device: str | None = None) -> None:
        from transformers import (  # type: ignore
            AutoModelForImageTextToText,
            AutoProcessor,
        )

        selected_device = device or settings.SEQUENTIAL_DEVICE
        self.frames_per_sample = settings.FRAMES_PER_SAMPLE
        self.maximum_width = 480
        log(f"Loading visual model: {model_name}")
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.processor.video_processor.size = {
            "height": 448,
            "width": 448,
        }
        self.model = load_model_with_fallback(
            AutoModelForImageTextToText.from_pretrained,
            model_name,
            device=selected_device,
            load_in_4bit=settings.VLM_LOAD_IN_4BIT,
            model_label="visual model",
        ).eval()
        log("Visual model loaded")

    def verify_cut(
        self,
        sample_path: Path,
        cut_time: float,
        clip_start_time: float,
        boundary_time_in_clip: float,
    ) -> CutVerification:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": str(sample_path)},
                    {
                        "type": "text",
                        "text": self._build_cut_prompt(boundary_time_in_clip),
                    },
                ],
            }
        ]
        try:
            answer = self._generate(messages, sample_path)
        except Exception as exc:
            return self._cut_error(
                cut_time,
                clip_start_time,
                boundary_time_in_clip,
                "model_error",
                str(exc),
            )

        data = recover_json(answer)
        if data is None:
            return self._cut_error(
                cut_time,
                clip_start_time,
                boundary_time_in_clip,
                "json_recovery_failed",
                "The visual model did not return valid JSON.",
                answer,
            )

        return CutVerification(
            cut_time=cut_time,
            clip_start_time=clip_start_time,
            boundary_time_in_clip=boundary_time_in_clip,
            is_real_cut=normalise_bool(data.get("is_real_cut")),
            confidence=clamp_float(data.get("confidence"), 0.0, 1.0),
            transition_type=clean_text(
                data.get("transition_type") or "unclear"
            ).lower(),
            short_reason=clean_text(data.get("short_reason")),
            raw_response=answer,
            error=None,
        )

    def judge_segment(
        self,
        sample_path: Path,
        metadata: Dict[str, Any],
        segment_index: int,
        segment_count: int,
        start_time: int,
        end_time: int,
        video_duration: float,
    ) -> SegmentReview:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": str(sample_path)},
                    {
                        "type": "text",
                        "text": self._build_segment_prompt(
                            metadata,
                            segment_index,
                            segment_count,
                            start_time,
                            end_time,
                            video_duration,
                        ),
                    },
                ],
            }
        ]
        try:
            answer = self._generate(messages, sample_path)
        except Exception as exc:
            return self._segment_error(
                segment_index,
                start_time,
                end_time,
                "model_error",
                str(exc),
            )

        data = recover_json(answer)
        if data is None:
            return self._segment_error(
                segment_index,
                start_time,
                end_time,
                "json_recovery_failed",
                "The visual model did not return valid JSON.",
                answer,
            )

        content_type = clean_text(
            data.get("content_type") or "unclear"
        ).lower()
        if content_type not in _SEGMENT_CONTENT_TYPES:
            content_type = "unclear"

        is_walking = normalise_bool(data.get("is_walking_video"))
        is_advertisement = normalise_bool(
            data.get("is_advertisement")
        ) or content_type in {"advertisement", "channel_promotion"}
        is_intro_highlights = normalise_bool(
            data.get("is_intro_highlights")
        ) or content_type == "intro_highlights"
        confidence = clamp_float(data.get("confidence"), 0.0, 1.0)
        requested_include = normalise_bool(data.get("include"))
        include = (
            requested_include
            and is_walking
            and content_type == "walking"
            and not is_advertisement
            and not is_intro_highlights
            and confidence >= settings.MIN_SEGMENT_CONFIDENCE
        )

        return SegmentReview(
            segment_index=segment_index,
            start_time=start_time,
            end_time=end_time,
            include=include,
            confidence=confidence,
            is_walking_video=is_walking,
            content_type=content_type,
            is_advertisement=is_advertisement,
            is_intro_highlights=is_intro_highlights,
            time_of_day=self._normalise_time_of_day(
                data.get("time_of_day")
            ),
            quality_issues=(
                normalise_string_list(data.get("quality_issues")) or ["none"]
            ),
            short_reason=clean_text(data.get("short_reason")),
            raw_response=answer,
            error=None,
        )

    def _generate(
        self,
        messages: List[Dict[str, Any]],
        sample_path: Path,
    ) -> str:
        available_frames = get_video_frame_count(sample_path)
        if available_frames is None:
            raise RuntimeError(
                f"Could not determine frame count for {sample_path}"
            )
        requested_frames = min(
            settings.FRAMES_PER_SAMPLE,
            available_frames,
        )
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={
                "num_frames": requested_frames,
            },
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
    def _normalise_time_of_day(value: Any) -> str:
        label = clean_text(value).lower()
        if label in {"dawn", "dusk", "twilight", "dawn or dusk"}:
            label = "dawn_dusk"
        if label not in settings.TIME_OF_DAY_CODES:
            label = "unknown"
        return label

    @staticmethod
    def _cut_error(
        cut_time: float,
        clip_start_time: float,
        boundary_time_in_clip: float,
        error: str,
        reason: str,
        raw_response: str = "",
    ) -> CutVerification:
        return CutVerification(
            cut_time=cut_time,
            clip_start_time=clip_start_time,
            boundary_time_in_clip=boundary_time_in_clip,
            is_real_cut=False,
            confidence=0.0,
            transition_type="unclear",
            short_reason=reason,
            raw_response=raw_response,
            error=error,
        )

    @staticmethod
    def _segment_error(
        segment_index: int,
        start_time: int,
        end_time: int,
        error: str,
        reason: str,
        raw_response: str = "",
    ) -> SegmentReview:
        return SegmentReview(
            segment_index=segment_index,
            start_time=start_time,
            end_time=end_time,
            include=False,
            confidence=0.0,
            is_walking_video=False,
            content_type="unclear",
            is_advertisement=False,
            is_intro_highlights=False,
            time_of_day="unknown",
            quality_issues=[error],
            short_reason=reason,
            raw_response=raw_response,
            error=error,
        )

    @staticmethod
    def _build_cut_prompt(boundary_time_in_clip: float) -> str:
        return f"""
FFmpeg proposed a scene cut at approximately {boundary_time_in_clip:.2f}
seconds from the start of this short clip. Decide whether an actual visual
transition occurs at that position.

A real cut changes to a different shot, scene, camera source, title card,
advertisement, or highlight. Camera shake, walking motion, motion blur,
compression artefacts, lighting fluctuations, exposure changes, or gradual
movement are not cuts.

Return valid JSON only:
{{
  "is_real_cut": true,
  "confidence": 0.0,
  "transition_type": "hard_cut",
  "short_reason": "one short sentence"
}}
""".strip()

    @staticmethod
    def _build_segment_prompt(
        metadata: Dict[str, Any],
        segment_index: int,
        segment_count: int,
        start_time: int,
        end_time: int,
        video_duration: float,
    ) -> str:
        title = clean_text(metadata.get("title"))
        description = clean_text(metadata.get("description"))[:1200]
        return f"""
You are reviewing segment {segment_index + 1} of {segment_count} from a
pedestrian walking video candidate. This segment covers {start_time} to
{end_time} seconds of a {video_duration:.1f} second video.

The supplied review clip contains frames sampled across this entire segment.
The jumps between supplied frames were created by sampling and must not be
treated as source video cuts or as evidence of a montage.

Include only genuine pedestrian walking footage through a real physical
location. Reject advertisements, sponsorship material, channel promotion,
logos or title cards, requests to subscribe, intros, outros, and other creator
material. Also reject opening highlight reels, previews, recaps, and teaser
sequences that summarise footage shown later in the video. Use the segment's
position and duration as context when judging opening highlights.

Reject vehicle, bicycle, drone, boat, train, bus, static, talking head, game,
animation, slideshow, or screen recording content. Classify time_of_day from
the visual evidence in this segment only as day, night, dawn_dusk, or unknown.
Indoor or ambiguous footage must be unknown.

Title: {title}
Description excerpt: {description}

Return valid JSON only:
{{
  "include": true,
  "confidence": 0.0,
  "is_walking_video": true,
  "content_type": "walking",
  "is_advertisement": false,
  "is_intro_highlights": false,
  "time_of_day": "day",
  "quality_issues": ["none"],
  "short_reason": "one short sentence"
}}
""".strip()


class Cosmos3VisualJudge:
    """Cosmos 3 Nano Reasoner for temporal grounding and segment review."""

    def __init__(self, model_name: str, device: str | None = None) -> None:
        from transformers import (  # type: ignore
            AutoProcessor,
            Cosmos3OmniForConditionalGeneration,
        )

        selected_device = device or settings.SEQUENTIAL_DEVICE
        self.frames_per_sample = settings.COSMOS3_FRAMES_PER_SAMPLE
        self.maximum_width = settings.COSMOS3_VIDEO_WIDTH
        log(f"Loading Cosmos 3 Reasoner: {model_name}")
        self.processor = AutoProcessor.from_pretrained(model_name)

        def load_cosmos_reasoner(
            checkpoint: str,
            **model_kwargs: Any,
        ) -> Any:
            model_dtype = model_kwargs.pop("torch_dtype", None)
            if model_dtype is not None:
                model_kwargs["dtype"] = model_dtype
            return Cosmos3OmniForConditionalGeneration.from_pretrained(
                checkpoint,
                **model_kwargs,
            )

        self.model = load_model_with_fallback(
            load_cosmos_reasoner,
            model_name,
            device=selected_device,
            load_in_4bit=settings.COSMOS3_LOAD_IN_4BIT,
            model_label="Cosmos 3 Nano Reasoner",
        ).eval()
        log("Cosmos 3 Reasoner loaded")

    def locate_main_content_start(
        self,
        sample_path: Path,
        metadata: Dict[str, Any],
        clip_duration: float,
    ) -> ContentStartDecision:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": str(sample_path)},
                    {
                        "type": "text",
                        "text": self._build_content_start_prompt(
                            metadata,
                            clip_duration,
                        ),
                    },
                ],
            }
        ]
        try:
            answer = self._generate(
                messages,
                fps=settings.COSMOS3_INTRO_FPS,
            )
        except Exception as exc:
            return ContentStartDecision(
                main_content_start_time=None,
                confidence=0.0,
                short_reason=str(exc),
                raw_response="",
                error="model_error",
            )

        data = recover_json(answer)
        if data is None:
            return ContentStartDecision(
                main_content_start_time=None,
                confidence=0.0,
                short_reason="Cosmos 3 did not return valid JSON.",
                raw_response=answer,
                error="json_recovery_failed",
            )

        start_time = self._optional_timestamp(
            data.get("main_content_start_time"),
            clip_duration,
        )
        return ContentStartDecision(
            main_content_start_time=start_time,
            confidence=clamp_float(data.get("confidence"), 0.0, 1.0),
            short_reason=clean_text(data.get("short_reason")),
            raw_response=answer,
            error=None,
        )

    def verify_cut(
        self,
        sample_path: Path,
        cut_time: float,
        clip_start_time: float,
        boundary_time_in_clip: float,
    ) -> CutVerification:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": str(sample_path)},
                    {
                        "type": "text",
                        "text": InternVLVisualJudge._build_cut_prompt(
                            boundary_time_in_clip
                        ),
                    },
                ],
            }
        ]
        try:
            answer = self._generate(messages, fps=2.0)
        except Exception as exc:
            return InternVLVisualJudge._cut_error(
                cut_time,
                clip_start_time,
                boundary_time_in_clip,
                "model_error",
                str(exc),
            )

        data = recover_json(answer)
        if data is None:
            return InternVLVisualJudge._cut_error(
                cut_time,
                clip_start_time,
                boundary_time_in_clip,
                "json_recovery_failed",
                "Cosmos 3 did not return valid JSON.",
                answer,
            )

        return CutVerification(
            cut_time=cut_time,
            clip_start_time=clip_start_time,
            boundary_time_in_clip=boundary_time_in_clip,
            is_real_cut=normalise_bool(data.get("is_real_cut")),
            confidence=clamp_float(data.get("confidence"), 0.0, 1.0),
            transition_type=clean_text(
                data.get("transition_type") or "unclear"
            ).lower(),
            short_reason=clean_text(data.get("short_reason")),
            raw_response=answer,
            error=None,
        )

    def judge_segment(
        self,
        sample_path: Path,
        metadata: Dict[str, Any],
        segment_index: int,
        segment_count: int,
        start_time: int,
        end_time: int,
        video_duration: float,
    ) -> SegmentReview:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": str(sample_path)},
                    {
                        "type": "text",
                        "text": self._build_segment_prompt(
                            metadata,
                            segment_index,
                            segment_count,
                            start_time,
                            end_time,
                            video_duration,
                        ),
                    },
                ],
            }
        ]
        try:
            answer = self._generate(
                messages,
                fps=settings.COSMOS3_REVIEW_FPS,
            )
        except Exception as exc:
            return InternVLVisualJudge._segment_error(
                segment_index,
                start_time,
                end_time,
                "model_error",
                str(exc),
            )

        data = recover_json(answer)
        if data is None:
            return InternVLVisualJudge._segment_error(
                segment_index,
                start_time,
                end_time,
                "json_recovery_failed",
                "Cosmos 3 did not return valid JSON.",
                answer,
            )

        content_type = clean_text(
            data.get("content_type") or "unclear"
        ).lower()
        if content_type not in _SEGMENT_CONTENT_TYPES:
            content_type = "unclear"

        reported_walking = normalise_bool(data.get("is_walking_video"))
        walking_fraction = self._fraction_or_default(
            data,
            "walking_fraction",
            1.0 if reported_walking else 0.0,
        )
        reported_promotion = normalise_bool(
            data.get("is_inserted_promotion")
        )
        promotion_fraction = self._fraction_or_default(
            data,
            "inserted_promotion_fraction",
            1.0 if reported_promotion else 0.0,
        )
        is_intro_highlights = normalise_bool(
            data.get("is_intro_highlights")
        ) or content_type == "intro_highlights"
        confidence = clamp_float(data.get("confidence"), 0.0, 1.0)
        include = (
            walking_fraction >= settings.COSMOS3_MIN_WALKING_FRACTION
            and promotion_fraction <= settings.COSMOS3_MAX_PROMOTION_FRACTION
            and not is_intro_highlights
            and confidence >= settings.MIN_SEGMENT_CONFIDENCE
        )
        is_walking = (
            reported_walking
            or walking_fraction >= settings.COSMOS3_MIN_WALKING_FRACTION
        )
        is_advertisement = (
            promotion_fraction > settings.COSMOS3_MAX_PROMOTION_FRACTION
        )
        if include:
            content_type = "walking"

        return SegmentReview(
            segment_index=segment_index,
            start_time=start_time,
            end_time=end_time,
            include=include,
            confidence=confidence,
            is_walking_video=is_walking,
            content_type=content_type,
            is_advertisement=is_advertisement,
            is_intro_highlights=is_intro_highlights,
            time_of_day=InternVLVisualJudge._normalise_time_of_day(
                data.get("time_of_day")
            ),
            quality_issues=(
                normalise_string_list(data.get("quality_issues")) or ["none"]
            ),
            short_reason=clean_text(data.get("short_reason")),
            raw_response=answer,
            walking_fraction=round(walking_fraction, 4),
            promotion_fraction=round(promotion_fraction, 4),
            decision_method="temporal_fraction",
            error=None,
        )

    def _generate(
        self,
        messages: List[Dict[str, Any]],
        *,
        fps: float,
    ) -> str:
        inputs = self.processor.apply_chat_template(
            messages,
            fps=fps,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, torch.bfloat16)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=settings.COSMOS3_MAX_NEW_TOKENS,
                do_sample=False,
            )
        prompt_length = inputs["input_ids"].shape[1]
        generated_ids = generated[:, prompt_length:]
        decoded = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0] if decoded else ""

    @staticmethod
    def _optional_timestamp(value: Any, upper_bound: float) -> Optional[float]:
        if value is None:
            return None
        try:
            timestamp = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(timestamp):
            return None
        return round(min(max(0.0, timestamp), upper_bound), 3)

    @staticmethod
    def _fraction_or_default(
        data: Dict[str, Any],
        key: str,
        default: float,
    ) -> float:
        if key not in data:
            return default
        return clamp_float(data.get(key), 0.0, 1.0)

    @staticmethod
    def _build_content_start_prompt(
        metadata: Dict[str, Any],
        clip_duration: float,
    ) -> str:
        title = clean_text(metadata.get("title"))
        description = clean_text(metadata.get("description"))[:1200]
        return f"""
This is the opening {clip_duration:.1f} seconds of a pedestrian walking tour
candidate. Find the earliest timestamp where the actual main continuous
walking tour begins.

Do not select walking shots that appear only inside an opening preview or
highlight montage. Opening aerial views, title cards, logos, channel material,
teasers, rapid shot changes, and previews are introduction content. The main
tour begins when the video settles into sustained first person or pedestrian
movement through the location. Valid main footage may begin indoors, inside a
shop, market, station, mall, or other real place. The camera wearer does not
need to be visible.

Return null when the main tour does not begin within this clip.

Title: {title}
Description excerpt: {description}

Return valid JSON only:
{{
  "main_content_start_time": 43.0,
  "confidence": 0.0,
  "short_reason": "one short sentence"
}}
""".strip()

    @staticmethod
    def _build_segment_prompt(
        metadata: Dict[str, Any],
        segment_index: int,
        segment_count: int,
        start_time: int,
        end_time: int,
        video_duration: float,
    ) -> str:
        title = clean_text(metadata.get("title"))
        description = clean_text(metadata.get("description"))[:1200]
        return f"""
Review window {segment_index + 1} of {segment_count}, covering source video
time {start_time} to {end_time} seconds in a {video_duration:.1f} second
pedestrian walking tour candidate.

The review clip contains evenly spaced frames from the whole source window.
The jumps between frames are sampling artefacts, not source cuts or a montage.
Judge the dominant content across all sampled moments. Never reject the entire
window because of one frame or a short portion.

Valid walking includes first person walking indoors or outdoors, walking
through shops, markets, malls, stations, parks, gardens, forests, temples,
streets, and other real locations. The camera wearer need not be visible.
Brief pauses, static views, captions, visible bicycles, shop interiors, product
displays, billboards, storefronts, and real world signs remain valid walking
context.

Inserted promotion means creator supplied sponsor cards, sales pitches,
affiliate overlays, channel promotion, subscribe requests, or full screen
advertising unrelated to the physical surroundings. Products and advertising
signs naturally visible in the filmed environment are not inserted promotion.

Estimate the fraction of this entire source window containing valid pedestrian
walking and the fraction containing inserted promotion. Reject vehicle, cycle,
drone, boat, train, bus, game, animation, slideshow, and screen recording
content when it dominates the window. Classify time of day as day, night,
dawn_dusk, or unknown. Indoor or ambiguous footage is unknown.

Title: {title}
Description excerpt: {description}

Return valid JSON only:
{{
  "confidence": 0.0,
  "is_walking_video": true,
  "walking_fraction": 0.0,
  "content_type": "walking",
  "is_inserted_promotion": false,
  "inserted_promotion_fraction": 0.0,
  "is_intro_highlights": false,
  "time_of_day": "day",
  "quality_issues": ["none"],
  "short_reason": "one short sentence"
}}
""".strip()


def create_visual_judge(device: str | None = None) -> Any:
    """Create the configured visual model without loading both backends."""
    if settings.VISUAL_MODEL_BACKEND == "cosmos3":
        return Cosmos3VisualJudge(settings.COSMOS3_MODEL_NAME, device=device)
    return InternVLVisualJudge(settings.VLM_MODEL_NAME, device=device)


def verify_cut_candidates(
    video_id: str,
    video_path: Path,
    duration: float,
    cut_times: List[float],
    judge: Any,
) -> List[CutVerification]:
    verifications: List[CutVerification] = []
    with tempfile.TemporaryDirectory(
        prefix=f"walk_cuts_{video_id}_"
    ) as temporary_name:
        temporary_dir = Path(temporary_name)
        for cut_index, cut_time in enumerate(cut_times):
            clip_start, clip_duration, boundary_offset = (
                cut_verification_window(cut_time, duration)
            )
            sample_path = temporary_dir / f"cut_{cut_index:04d}.mp4"
            if not create_sample_clip(
                video_path,
                clip_start,
                sample_path,
                clip_duration,
            ):
                verification = InternVLVisualJudge._cut_error(
                    cut_time,
                    clip_start,
                    boundary_offset,
                    "clip_creation_failed",
                    "FFmpeg could not create the cut verification clip.",
                )
            else:
                verification = judge.verify_cut(
                    sample_path,
                    cut_time,
                    clip_start,
                    boundary_offset,
                )
            verifications.append(verification)
            log(
                f"{video_id} cut {cut_index}: "
                f"time={cut_time:.2f}, real={verification.is_real_cut}, "
                f"confidence={verification.confidence:.2f}"
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return verifications


def verified_cut_times(
    verifications: List[CutVerification],
) -> List[float]:
    return [
        verification.cut_time
        for verification in verifications
        if verification.error is None
        and verification.is_real_cut
        and verification.confidence >= settings.MIN_CUT_CONFIDENCE
    ]


def review_segments(
    video_id: str,
    video_path: Path,
    duration: float,
    metadata: Dict[str, Any],
    segments: List[Dict[str, Any]],
    judge: Any,
) -> List[SegmentReview]:
    reviews: List[SegmentReview] = []
    segment_count = len(segments)
    with tempfile.TemporaryDirectory(
        prefix=f"walk_segments_{video_id}_"
    ) as temporary_name:
        temporary_dir = Path(temporary_name)
        for segment_index, segment in enumerate(segments):
            start_time = int(segment["start_time"])
            end_time = int(segment["end_time"])
            sample_path = temporary_dir / f"segment_{segment_index:04d}.mp4"
            segment_duration = segment_duration_seconds(segment)
            forced_rejection = segment.get("_forced_rejection")
            if isinstance(forced_rejection, dict):
                review = SegmentReview(
                    segment_index=segment_index,
                    start_time=start_time,
                    end_time=end_time,
                    include=False,
                    confidence=clamp_float(
                        forced_rejection.get("confidence"),
                        0.0,
                        1.0,
                    ),
                    is_walking_video=False,
                    content_type=clean_text(
                        forced_rejection.get("content_type")
                        or "intro_highlights"
                    ),
                    is_advertisement=False,
                    is_intro_highlights=True,
                    time_of_day="unknown",
                    quality_issues=[
                        clean_text(
                            forced_rejection.get("quality_issue")
                            or "intro_before_main_content"
                        )
                    ],
                    short_reason=clean_text(
                        forced_rejection.get("short_reason")
                        or "This footage precedes the main walking tour."
                    ),
                    raw_response=clean_text(
                        forced_rejection.get("raw_response")
                    ),
                    decision_method="semantic_content_start",
                    error=None,
                )
            elif segment_duration < settings.MIN_SEGMENT_DURATION_SECONDS:
                review = SegmentReview(
                    segment_index=segment_index,
                    start_time=start_time,
                    end_time=end_time,
                    include=False,
                    confidence=1.0,
                    is_walking_video=False,
                    content_type="unclear",
                    is_advertisement=False,
                    is_intro_highlights=False,
                    time_of_day="unknown",
                    quality_issues=["segment_too_short"],
                    short_reason=(
                        f"The segment is {segment_duration} seconds long, "
                        "which is below the configured minimum of "
                        f"{settings.MIN_SEGMENT_DURATION_SECONDS} seconds."
                    ),
                    raw_response="",
                    error=None,
                )
            elif not create_segment_review_clip(
                video_path,
                start_time,
                end_time,
                sample_path,
                target_frame_count=getattr(
                    judge,
                    "frames_per_sample",
                    settings.FRAMES_PER_SAMPLE,
                ),
                maximum_width=getattr(judge, "maximum_width", 480),
            ):
                review = InternVLVisualJudge._segment_error(
                    segment_index,
                    start_time,
                    end_time,
                    "clip_creation_failed",
                    "FFmpeg could not create the segment review clip.",
                )
            else:
                review = judge.judge_segment(
                    sample_path,
                    metadata,
                    segment_index,
                    segment_count,
                    start_time,
                    end_time,
                    duration,
                )
            reviews.append(review)
            log(
                f"{video_id} segment {segment_index}: "
                f"include={review.include}, type={review.content_type}, "
                f"confidence={review.confidence:.2f}, "
                f"time={review.time_of_day}"
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return reviews


def locate_main_content_start(
    video_id: str,
    video_path: Path,
    duration: float,
    metadata: Dict[str, Any],
    judge: Any,
) -> tuple[ContentStartDecision, float]:
    """Use one compact opening clip to locate the sustained walking tour."""
    clip_duration = min(
        duration,
        float(settings.COSMOS3_INTRO_SEARCH_SECONDS),
    )
    with tempfile.TemporaryDirectory(
        prefix=f"walk_intro_{video_id}_"
    ) as temporary_name:
        sample_path = Path(temporary_name) / "opening.mp4"
        if not create_content_start_clip(
            video_path,
            sample_path,
            clip_duration,
        ):
            return (
                ContentStartDecision(
                    main_content_start_time=None,
                    confidence=0.0,
                    short_reason=(
                        "FFmpeg could not create the opening review clip."
                    ),
                    raw_response="",
                    error="clip_creation_failed",
                ),
                clip_duration,
            )
        return (
            judge.locate_main_content_start(
                sample_path,
                metadata,
                clip_duration,
            ),
            clip_duration,
        )


def build_cosmos_review_segments(
    duration: float,
    decision: ContentStartDecision,
    searched_seconds: float,
) -> tuple[List[Dict[str, Any]], int]:
    """Build length independent windows after a semantic opening boundary."""
    if duration <= 0:
        return [], 0

    final_second = max(0, int(math.floor(duration)))
    reliable = (
        decision.error is None
        and decision.confidence
        >= settings.COSMOS3_MIN_CONTENT_START_CONFIDENCE
    )
    if not reliable:
        content_start_second = 0
    elif decision.main_content_start_time is None:
        content_start_second = int(math.ceil(searched_seconds))
    else:
        content_start_second = int(
            math.ceil(decision.main_content_start_time)
        )
    content_start_second = min(
        max(0, content_start_second),
        final_second + 1,
    )

    base_segments: List[Dict[str, Any]] = []
    if content_start_second > 0:
        base_segments.append(
            {
                "start_time": 0,
                "end_time": content_start_second - 1,
                "_forced_rejection": {
                    "confidence": decision.confidence,
                    "content_type": "intro_highlights",
                    "quality_issue": "intro_before_main_content",
                    "short_reason": (
                        decision.short_reason
                        or "This footage precedes the main walking tour."
                    ),
                    "raw_response": decision.raw_response,
                },
            }
        )
    if content_start_second <= final_second:
        base_segments.append(
            {
                "start_time": content_start_second,
                "end_time": final_second,
            }
        )

    return (
        split_long_segments(
            base_segments,
            settings.MAX_SEGMENT_REVIEW_SECONDS,
            settings.MIN_SEGMENT_DURATION_SECONDS,
        ),
        content_start_second,
    )


def accepted_segments_from_reviews(
    reviews: List[SegmentReview],
) -> List[Dict[str, Any]]:
    return [
        {
            "start_time": review.start_time,
            "end_time": review.end_time,
            "time_of_day": settings.TIME_OF_DAY_CODES.get(
                review.time_of_day,
                settings.TIME_OF_DAY_CODES["unknown"],
            ),
            "visual_confidence": round(review.confidence, 4),
        }
        for review in reviews
        if review.include and review.error is None
    ]


def aggregate_segment_reviews(
    reviews: List[SegmentReview],
) -> Dict[str, Any]:
    if not reviews:
        return {
            "include": False,
            "confidence": 0.0,
            "included_segments": 0,
            "total_segments": 0,
            "short_reason": "No segments were available for visual review.",
            "segments": [],
            "error": "no_segments",
        }

    valid_reviews = [review for review in reviews if review.error is None]
    if not valid_reviews:
        return {
            "include": False,
            "confidence": 0.0,
            "included_segments": 0,
            "total_segments": len(reviews),
            "short_reason": "Every segment review failed.",
            "segments": [asdict(review) for review in reviews],
            "error": "all_segment_reviews_failed",
        }

    accepted = [review for review in valid_reviews if review.include]
    confidence_source = accepted or valid_reviews
    mean_confidence = sum(
        review.confidence for review in confidence_source
    ) / len(confidence_source)
    include = bool(accepted)
    if include:
        reason = (
            "Accepted pedestrian walking segments after removing unusable "
            "content."
        )
    else:
        reason = (
            "No segment passed the pedestrian walking and content checks."
        )

    warning = None
    failed_count = len(reviews) - len(valid_reviews)
    if failed_count:
        warning = f"{failed_count} segment review(s) failed and were omitted."

    return {
        "include": include,
        "confidence": round(mean_confidence, 4),
        "included_segments": len(accepted),
        "total_segments": len(reviews),
        "short_reason": reason,
        "segments": [asdict(review) for review in reviews],
        "warning": warning,
        "error": None,
    }


def process_visual_candidate(
    video_id: str,
    record: Dict[str, Any],
    judge: Any,
) -> None:
    record["visual_review_version"] = settings.VISUAL_REVIEW_VERSION
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
    if duration < settings.MIN_VIDEO_DURATION_SECONDS:
        record["status"] = "visual_rejected"
        record["error"] = (
            "downloaded video is shorter than "
            f"{settings.MIN_VIDEO_DURATION_SECONDS} seconds"
        )
        remove_video_if_required(video_path)
        return

    use_cosmos_fast_path = (
        settings.VISUAL_MODEL_BACKEND == "cosmos3"
        and settings.COSMOS3_FAST_MODE
    )
    if use_cosmos_fast_path:
        log(f"Locating the main walking content in {video_id}")
        content_start, searched_seconds = locate_main_content_start(
            video_id,
            video_path,
            duration,
            metadata,
            judge,
        )
        candidate_segments, content_start_second = (
            build_cosmos_review_segments(
                duration,
                content_start,
                searched_seconds,
            )
        )
        record["content_start_decision"] = asdict(content_start)
        record["cuts"] = {
            "method": "cosmos3_semantic_content_start",
            "threshold": None,
            "merge_nearby_seconds": None,
            "detected_cut_times": [],
            "cut_times": (
                [content_start_second] if content_start_second > 0 else []
            ),
            "verification": [],
            "message": (
                "Full video scene detection skipped in Cosmos 3 fast mode."
            ),
        }
        log(
            f"Prepared {len(candidate_segments)} review windows for "
            f"{video_id}; semantic content start={content_start_second}s"
        )
    else:
        log(f"Detecting candidate cuts in {video_id}")
        cut_result = run_ffmpeg_scene_detection(video_path, duration)
        record["cuts"] = {
            "method": "ffmpeg_scene_detection",
            "threshold": settings.SCENE_THRESHOLD,
            "merge_nearby_seconds": settings.MERGE_NEARBY_SEC,
            "detected_cut_times": [
                round(value, 3) for value in cut_result.cut_times
            ],
            "cut_times": [],
            "verification": [],
            "message": cut_result.message,
        }
        if cut_result.message:
            record["status"] = "cut_detection_failed"
            record["error"] = cut_result.message
            return

        verifications = verify_cut_candidates(
            video_id,
            video_path,
            duration,
            cut_result.cut_times,
            judge,
        )
        record["cuts"]["verification"] = [
            asdict(verification) for verification in verifications
        ]
        verification_errors = [
            verification.error
            for verification in verifications
            if verification.error
        ]
        if verification_errors:
            record["status"] = "cut_verification_failed"
            record["error"] = (
                f"{len(verification_errors)} cut verification call(s) failed"
            )
            return

        confirmed_cut_times = verified_cut_times(verifications)
        record["cuts"]["cut_times"] = [
            round(value, 3) for value in confirmed_cut_times
        ]
        cut_segments = build_segments(duration, confirmed_cut_times)
        candidate_segments = split_long_segments(
            cut_segments,
            settings.MAX_SEGMENT_REVIEW_SECONDS,
            settings.MIN_SEGMENT_DURATION_SECONDS,
        )
        log(
            f"Prepared {len(candidate_segments)} fixed duration review "
            f"windows from {len(cut_segments)} cut based segments for "
            f"{video_id}"
        )
    record["cuts"]["minimum_segment_seconds"] = (
        settings.MIN_SEGMENT_DURATION_SECONDS
    )
    record["cuts"]["maximum_review_window_seconds"] = (
        settings.MAX_SEGMENT_REVIEW_SECONDS
    )
    reviews = review_segments(
        video_id,
        video_path,
        duration,
        metadata,
        candidate_segments,
        judge,
    )
    failed_reviews = [
        review for review in reviews if review.error is not None
    ]
    if failed_reviews:
        record["visual_decision"] = aggregate_segment_reviews(reviews)
        record["segments"] = []
        record["status"] = "segment_review_failed"
        record["error"] = (
            f"{len(failed_reviews)} segment review call(s) failed"
        )
        return

    visual_decision = aggregate_segment_reviews(reviews)
    record["visual_decision"] = visual_decision
    record["segments"] = accepted_segments_from_reviews(reviews)

    if not visual_decision.get("include"):
        record["status"] = (
            "visual_error"
            if visual_decision.get("error")
            else "visual_rejected"
        )
        record["error"] = visual_decision.get("error")
        remove_video_if_required(video_path)
        return

    record["status"] = "complete"
    record["error"] = None


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
        if (
            record.get("status") in {"complete", "visual_rejected"}
            and record.get("visual_review_version")
            == settings.VISUAL_REVIEW_VERSION
        ):
            continue
        pending.append((video_id, record))

    if settings.MAX_VIDEOS_PER_RUN is not None:
        pending = pending[: settings.MAX_VIDEOS_PER_RUN]
    if not pending:
        return 0

    judge = create_visual_judge()
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