"""Video download, cut verification, and segment level VLM review."""

from __future__ import annotations

import gc
import math
import queue
import re
import tempfile
import threading
from dataclasses import asdict, dataclass, field
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


VIDEO_FILTER_SCHEMA_VERSION = "walking_download_queue_retention_v8"

_SEGMENT_CONTENT_TYPES = {
    "walking",
    "advertisement",
    "channel_promotion",
    "intro_highlights",
    "nonwalking",
    "unclear",
}

_BURST_CONTENT_TYPES = {
    "walking",
    "vehicle",
    "static",
    "highlight",
    "map_title",
    "promotion",
    "other",
}

_WALKING_ENVIRONMENTS = {
    "street",
    "indoor",
    "beach",
    "park_nature",
    "trail",
    "market",
    "waterfront",
    "square_plaza",
    "transport_hub",
    "mixed",
    "other",
    "unknown",
    "not_applicable",
}

_CUT_BOUNDARY_EVIDENCE = {
    "edit",
    "continuous_motion",
    "occlusion_or_blur",
    "uncertain",
}

_TIMESTAMP_PATTERN = re.compile(
    r"(?<!\d)(?P<timestamp>(?:\d{1,2}:)?\d{1,3}:\d{2})(?!\d)"
)

_CHAPTER_HEADING_PATTERN = re.compile(
    r"\b(?:tour\s+timeline|chapters?|timeline)\b\s*:?",
    re.IGNORECASE,
)

_DESCRIPTION_SECTION_BREAK_PATTERN = re.compile(r"(?:━{5,}|-{5,})")

_COOKIE_EXTRACTION_ERROR_MARKERS = (
    "dpapi",
    "failed to decrypt",
    "failed to load cookies",
    "failed to copy",
    "cookie database",
)

_AMBIGUOUS_CUT_REASON_MARKERS = (
    "change in perspective",
    "change in subject focus",
    "change in framing",
    "camera reposition",
    "foreground subject disappearing",
    "different group of people",
    "camera abruptly shifts",
    "camera shifts abruptly",
    "abrupt shift",
)

_DIRECT_EDIT_REASON_MARKERS = (
    "black screen",
    "title card",
    "graphic overlay",
    "map overlay",
    "fade to",
    "dissolve",
    "different location",
    "unrelated scene",
    "jump in time",
)


def normalise_cut_boundary_evidence(value: Any) -> str:
    evidence = clean_text(value).lower()
    if evidence not in _CUT_BOUNDARY_EVIDENCE:
        return "uncertain"
    return evidence


def normalise_confidence(value: Any) -> float:
    text = clean_text(value).lower().replace("_", " ")
    qualitative = {
        "very high": 0.98,
        "high": 0.90,
        "medium": 0.60,
        "moderate": 0.60,
        "low": 0.30,
        "very low": 0.10,
    }
    if text in qualitative:
        return qualitative[text]
    return clamp_float(value, 0.0, 1.0)


def cut_reason_needs_motion_retry(value: Any) -> bool:
    reason = clean_text(value).lower()
    if any(marker in reason for marker in _DIRECT_EDIT_REASON_MARKERS):
        return False
    return any(marker in reason for marker in _AMBIGUOUS_CUT_REASON_MARKERS)


def normalise_walking_environment(value: Any, is_walking: bool) -> str:
    if not is_walking:
        return "not_applicable"
    environment = clean_text(value).lower()
    if (
        environment not in _WALKING_ENVIRONMENTS
        or environment == "not_applicable"
    ):
        return "unknown"
    return environment


def timestamp_text_to_seconds(value: str) -> Optional[int]:
    parts = value.split(":")
    try:
        numbers = [int(part) for part in parts]
    except ValueError:
        return None
    if len(numbers) == 2:
        minutes, seconds = numbers
        if seconds >= 60:
            return None
        return minutes * 60 + seconds
    if len(numbers) == 3:
        hours, minutes, seconds = numbers
        if minutes >= 60 or seconds >= 60:
            return None
        return hours * 3600 + minutes * 60 + seconds
    return None


def extract_description_timestamp_labels(
    metadata: Dict[str, Any],
) -> List[Dict[str, Any]]:
    description = metadata.get("description")
    if not isinstance(description, str):
        return []

    chapter_regions: List[str] = []
    for heading in _CHAPTER_HEADING_PATTERN.finditer(description):
        remainder = description[heading.end() :]
        first_timestamp = _TIMESTAMP_PATTERN.search(remainder)
        if first_timestamp is None or first_timestamp.start() > 160:
            continue
        next_section = _DESCRIPTION_SECTION_BREAK_PATTERN.search(
            remainder,
            first_timestamp.end(),
        )
        chapter_regions.append(
            remainder[: next_section.start()]
            if next_section is not None
            else remainder
        )
    if chapter_regions:
        description = chapter_regions[-1]

    matches = list(_TIMESTAMP_PATTERN.finditer(description))
    labels: List[Dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for index, match in enumerate(matches):
        prefix = description[max(0, match.start() - 48) : match.start()]
        prefix = prefix.strip().lower()
        if any(
            marker in prefix
            for marker in (
                "http",
                "duration",
                "shooting time",
                "weather",
                "published",
                "uploaded",
            )
        ):
            continue
        timestamp_seconds = timestamp_text_to_seconds(
            match.group("timestamp")
        )
        next_start = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(description)
        )
        raw_label = description[match.end() : next_start]
        raw_label = re.split(
            r"(?:-{5,}|https?://|video duration\s*:|shooting time\s*:|"
            r"weather\s*:|watch also\s*:)",
            raw_label,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        label = clean_text(
            raw_label.strip(" \t\r\n-–—|:•")
        )
        if (
            timestamp_seconds is None
            or not label
            or label.lower().startswith(("http://", "https://"))
        ):
            continue
        if len(label) > 200:
            label = label[:200].rstrip()
        if labels and timestamp_seconds < int(
            labels[-1]["timestamp_seconds"]
        ):
            continue
        key = (timestamp_seconds, label)
        if key in seen:
            continue
        seen.add(key)
        labels.append(
            {
                "timestamp_seconds": timestamp_seconds,
                "timestamp_text": match.group("timestamp"),
                "label": label,
            }
        )
    labels.sort(key=lambda item: int(item["timestamp_seconds"]))
    return labels


def timestamp_labels_for_segment(
    metadata: Dict[str, Any],
    start_time: int,
    end_time: int,
) -> List[Dict[str, Any]]:
    candidate_labels = metadata.get("_timestamp_labels")
    if not isinstance(candidate_labels, list):
        candidate_labels = extract_description_timestamp_labels(metadata)

    valid_labels = [
        item
        for item in candidate_labels
        if isinstance(item, dict)
        and isinstance(item.get("timestamp_seconds"), int)
        and clean_text(item.get("label"))
    ]
    active_label: Optional[Dict[str, Any]] = None
    for item in valid_labels:
        if int(item["timestamp_seconds"]) <= start_time:
            active_label = item
        else:
            break

    selected: List[Dict[str, Any]] = []
    if active_label is not None:
        selected.append(dict(active_label))
    for item in valid_labels:
        timestamp_seconds = int(item["timestamp_seconds"])
        if start_time < timestamp_seconds <= end_time:
            selected.append(dict(item))
    return selected


def normalise_embedded_location_text(value: Any) -> List[str]:
    labels: List[str] = []
    for item in normalise_string_list(value):
        label = clean_text(item)
        if not label or label.lower() in {"none", "unknown", "null"}:
            continue
        if label not in labels:
            labels.append(label)
    return labels[:5]


def segment_location_source(
    timestamp_labels: List[Dict[str, Any]],
    embedded_location_text: List[str],
) -> str:
    if timestamp_labels and embedded_location_text:
        return "both"
    if timestamp_labels:
        return "timestamp_description"
    if embedded_location_text:
        return "embedded_video"
    return "none"


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
    boundary_evidence: str = "uncertain"
    camera_motion_possible: bool = True
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
    burst_content: List[str] = field(default_factory=list)
    walking_environment: str = "unknown"
    timestamp_labels: List[Dict[str, Any]] = field(default_factory=list)
    embedded_location_text: List[str] = field(default_factory=list)
    location_source: str = "none"
    error: Optional[str] = None


def find_downloaded_video(video_id: str) -> Optional[Path]:
    if not settings.VIDEO_DIR.exists():
        return None
    candidates = sorted(
        path
        for path in settings.VIDEO_DIR.glob(f"{video_id}.*")
        if path.suffix.lower() in settings.VIDEO_EXTENSIONS
    )
    candidates.sort(key=lambda path: (path.stem != video_id, path.name))
    for candidate in candidates:
        if has_video_stream(candidate):
            return candidate
        log(
            "Ignoring downloaded file without a video stream: "
            f"{candidate.name}"
        )
    return None


def has_video_stream(media_path: Path) -> bool:
    """Return whether FFprobe can find a decodable video stream."""
    require_binary("ffprobe")
    result = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=index",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(media_path),
        ],
        timeout=60,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def _yt_dlp_cookie_args() -> List[str]:
    """Build explicit cookie arguments without assuming a browser exists."""
    if settings.YT_DLP_COOKIE_FILE:
        return ["--cookies", settings.YT_DLP_COOKIE_FILE]
    if settings.YT_DLP_COOKIES_FROM_BROWSER:
        return [
            "--cookies-from-browser",
            settings.YT_DLP_COOKIES_FROM_BROWSER,
        ]
    return ["--no-cookies-from-browser"]


def _yt_dlp_download_command(
    video_id: str,
    output_template: str,
    cookie_args: List[str],
) -> List[str]:
    url = f"https://www.youtube.com/watch?v={video_id}"
    return [
        "yt-dlp",
        "--no-playlist",
        "--continue",
        "--no-warnings",
        *cookie_args,
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
    ]


def _is_cookie_extraction_error(result: Any) -> bool:
    output = f"{result.stdout}\n{result.stderr}".lower()
    return any(marker in output for marker in _COOKIE_EXTRACTION_ERROR_MARKERS)


def download_video(video_id: str) -> Optional[Path]:
    require_binary("yt-dlp")
    existing = find_downloaded_video(video_id)
    if existing:
        return existing

    settings.VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    output_template = str(settings.VIDEO_DIR / f"{video_id}.%(ext)s")
    cookie_args = _yt_dlp_cookie_args()
    result = run_command(
        _yt_dlp_download_command(
            video_id,
            output_template,
            cookie_args,
        ),
        timeout=7200,
    )
    if (
        result.returncode != 0
        and settings.YT_DLP_COOKIES_FROM_BROWSER
        and _is_cookie_extraction_error(result)
    ):
        log(
            "Browser cookie extraction failed; retrying this public video "
            "once without browser cookies"
        )
        result = run_command(
            _yt_dlp_download_command(
                video_id,
                output_template,
                ["--no-cookies-from-browser"],
            ),
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
    created = (
        result.returncode == 0
        and output_path.exists()
        and output_path.stat().st_size > 0
    )
    if not created:
        reason = clean_text(result.stderr) or "FFmpeg returned no output."
        log(
            f"Opening clip creation failed for {video_path.name}: "
            f"{reason[-1200:]}"
        )
    return created


def segment_review_burst_plan(
    start_time: int,
    end_time: int,
    requested_frame_count: int,
    output_fps: float,
) -> Tuple[List[int], List[float], float]:
    """Distribute one fixed frame budget without favouring short videos."""
    segment_duration = max(1, end_time - start_time + 1)
    use_long_window_plan = (
        settings.VISUAL_MODEL_BACKEND == "cosmos3"
        and segment_duration >= settings.COSMOS3_LONG_WINDOW_SECONDS
        and requested_frame_count >= 4
    )
    maximum_bursts = (
        settings.COSMOS3_LONG_WINDOW_BURSTS
        if use_long_window_plan
        else 3
    )
    burst_count = min(
        maximum_bursts,
        max(1, requested_frame_count // 2),
    )
    frames_per_burst = [requested_frame_count // burst_count] * burst_count
    for index in range(requested_frame_count % burst_count):
        frames_per_burst[index] += 1

    if burst_count == 1:
        positions = [0.0]
    elif use_long_window_plan:
        # Two late positions cover transitions near both 80 and 95 percent of
        # a balanced long window while keeping three contiguous frames per
        # burst with the default 12 frame budget.
        if burst_count == 4:
            positions = [0.0, 1.0 / 3.0, 0.8, 0.95]
        else:
            positions = [
                0.95 * index / (burst_count - 1)
                for index in range(burst_count)
            ]
    else:
        positions = [
            index / (burst_count - 1)
            for index in range(burst_count)
        ]

    source_sampling_fps = output_fps
    if use_long_window_plan:
        source_sampling_fps = min(
            output_fps,
            settings.COSMOS3_LONG_WINDOW_SOURCE_FPS,
        )
    return frames_per_burst, positions, source_sampling_fps


def create_segment_review_clip(
    video_path: Path,
    start_time: int,
    end_time: int,
    output_path: Path,
    target_frame_count: Optional[int] = None,
    maximum_width: int = 480,
) -> bool:
    """Create fixed-budget motion bursts across one entire source segment."""
    require_binary("ffmpeg")
    segment_duration = max(1.0, float(end_time - start_time + 1))
    requested_frame_count = max(
        1,
        target_frame_count or settings.FRAMES_PER_SAMPLE,
    )
    output_fps = (
        settings.COSMOS3_REVIEW_FPS
        if settings.VISUAL_MODEL_BACKEND == "cosmos3"
        else 1.0
    )
    frames_per_burst, positions, source_sampling_fps = (
        segment_review_burst_plan(
            start_time,
            end_time,
            requested_frame_count,
            output_fps,
        )
    )
    burst_count = len(frames_per_burst)
    input_arguments: List[str] = []
    filter_chains: List[str] = []
    labels: List[str] = []
    for index, frame_count in enumerate(frames_per_burst):
        burst_duration = min(
            segment_duration,
            frame_count / source_sampling_fps,
        )
        position = positions[index]
        available_span = max(0.0, segment_duration - burst_duration)
        burst_start = float(start_time) + available_span * position
        input_arguments.extend(
            [
                "-ss",
                f"{burst_start:.3f}",
                "-t",
                f"{burst_duration:.3f}",
                "-i",
                str(video_path),
            ]
        )
        label = f"v{index}"
        labels.append(f"[{label}]")
        filter_chains.append(
            f"[{index}:v]fps={source_sampling_fps:.10f},"
            f"scale='min({maximum_width},iw)':-2,"
            "format=yuv420p,setpts=PTS-STARTPTS"
            f"[{label}]"
        )

    filter_chains.append(
        "".join(labels)
        + f"concat=n={burst_count}:v=1:a=0[joined]"
    )
    filter_chains.append(
        f"[joined]setpts=N/({output_fps:.10f}*TB)[outv]"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = run_command(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            *input_arguments,
            "-an",
            "-filter_complex",
            ";".join(filter_chains),
            "-map",
            "[outv]",
            "-preset",
            "veryfast",
            "-crf",
            "28",
            "-frames:v",
            str(requested_frame_count),
            "-r",
            f"{output_fps:.10f}",
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

        boundary_evidence = normalise_cut_boundary_evidence(
            data.get("boundary_evidence")
        )
        camera_motion_possible = (
            True
            if "camera_motion_possible" not in data
            else normalise_bool(data.get("camera_motion_possible"))
        )
        return CutVerification(
            cut_time=cut_time,
            clip_start_time=clip_start_time,
            boundary_time_in_clip=boundary_time_in_clip,
            is_real_cut=(
                normalise_bool(data.get("is_real_cut"))
                and boundary_evidence == "edit"
                and not camera_motion_possible
            ),
            confidence=normalise_confidence(data.get("confidence")),
            transition_type=clean_text(
                data.get("transition_type") or "unclear"
            ).lower(),
            short_reason=clean_text(data.get("short_reason")),
            raw_response=answer,
            boundary_evidence=boundary_evidence,
            camera_motion_possible=camera_motion_possible,
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
        segment_prompt = self._build_segment_prompt(
            metadata,
            segment_index,
            segment_count,
            start_time,
            end_time,
            video_duration,
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": str(sample_path)},
                    {
                        "type": "text",
                        "text": segment_prompt,
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
        confidence = normalise_confidence(data.get("confidence"))
        requested_include = normalise_bool(data.get("include"))
        include = (
            requested_include
            and is_walking
            and content_type == "walking"
            and not is_advertisement
            and not is_intro_highlights
            and confidence >= settings.MIN_SEGMENT_CONFIDENCE
        )
        walking_environment = normalise_walking_environment(
            data.get("walking_environment"),
            is_walking,
        )
        embedded_location_text = normalise_embedded_location_text(
            data.get("embedded_location_text")
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
            walking_environment=walking_environment,
            embedded_location_text=embedded_location_text,
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
            walking_environment="not_applicable",
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

A real cut is an edit that creates an abrupt discontinuity between the adjacent
frames immediately before and immediately after that boundary. Track stable
objects, geometry, camera direction, and motion across those adjacent frames.
Choose edit only when their change cannot be explained by continuous camera or
subject motion.

A fast turn, entering or leaving a doorway, temporary occlusion by a person or
object, autofocus, motion blur, camera shake, exposure change, lighting change,
compression artefact, or sudden but physically possible viewpoint change is
not a cut. A large difference between frames farther away from the boundary is
also not sufficient. It is still possible for an edit to have walking on both
sides, but there must be direct discontinuity evidence at the stated boundary.

Judge only the last frame immediately before the stated boundary and the first
frame immediately after it. Do not compare the beginning and end of the clip.
A rapid pan or camera rotation may replace every visible object while remaining
continuous motion. In that case camera_motion_possible must be true and the
decision must be no_cut. Set camera_motion_possible to false only when adjacent
frame geometry proves that continuous physical camera movement is impossible.

Confidence means certainty in the classification, not the probability that a
cut exists. A confident no-cut decision should therefore have high confidence.

Return exactly one valid JSON object and nothing else. It must contain:
is_real_cut as a Boolean, confidence as a number from zero to one,
transition_type as hard_cut, transition, or no_cut, and short_reason as one
short sentence. It must also contain boundary_evidence as exactly one of edit,
continuous_motion, occlusion_or_blur, or uncertain, and
camera_motion_possible as a Boolean. Use edit only when an actual video edit is
directly visible at the proposed boundary.
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
        return f"""
You are reviewing segment {segment_index + 1} of {segment_count} from a
pedestrian walking video candidate. This segment covers {start_time} to
{end_time} seconds of a {video_duration:.1f} second video.

The supplied review clip contains short continuous motion bursts from the
beginning, middle, and end of this entire source segment. Motion inside each
burst is real source motion. The jumps between bursts were created by sampling
and must not be treated as source video cuts or as evidence of a montage.

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

For genuine walking, classify walking_environment as exactly one of street,
indoor, beach, park_nature, trail, market, waterfront, square_plaza,
transport_hub, mixed, other, or unknown. Use mixed when no single environment
dominates. Use not_applicable when the segment is not genuine walking.

Extract embedded_location_text as one JSON string containing the most specific
exact readable place name or location caption visibly written in this clip.
Exclude channel names, creator branding, generic titles, and inferred places.
Return an empty string when no such location text is clearly readable.

Confidence means certainty in the classification, not the probability that the
segment should be included. A confident rejection should have high confidence.

Return valid JSON only:
{{
  "include": true,
  "confidence": 0.85,
  "is_walking_video": true,
  "content_type": "walking",
  "is_advertisement": false,
  "is_intro_highlights": false,
  "time_of_day": "day",
  "walking_environment": "street",
  "embedded_location_text": "",
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
            confidence=normalise_confidence(data.get("confidence")),
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
            answer = self._generate(
                messages,
                fps=settings.COSMOS3_CUT_FPS,
            )
        except Exception as exc:
            return InternVLVisualJudge._cut_error(
                cut_time,
                clip_start_time,
                boundary_time_in_clip,
                "model_error",
                str(exc),
            )

        data = recover_json(answer)
        if (
            data is None
            or "boundary_evidence" not in data
            or "camera_motion_possible" not in data
        ):
            log(
                f"Cosmos 3 returned an incomplete cut decision at "
                f"{cut_time:.2f}s; retrying once"
            )
            retry_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "path": str(sample_path)},
                        {
                            "type": "text",
                            "text": (
                                InternVLVisualJudge._build_cut_prompt(
                                    boundary_time_in_clip
                                )
                                + "\n\nReturn every required property, "
                                "including boundary_evidence and "
                                "camera_motion_possible."
                            ),
                        },
                    ],
                }
            ]
            try:
                answer = self._generate(
                    retry_messages,
                    fps=settings.COSMOS3_CUT_FPS,
                )
            except Exception as exc:
                return InternVLVisualJudge._cut_error(
                    cut_time,
                    clip_start_time,
                    boundary_time_in_clip,
                    "model_error",
                    f"Cosmos 3 cut retry failed: {exc}",
                    answer,
                )
            data = recover_json(answer)
        if (
            data is not None
            and normalise_bool(data.get("is_real_cut"))
            and cut_reason_needs_motion_retry(data.get("short_reason"))
        ):
            log(
                f"Cosmos 3 cited only perspective or subject motion at "
                f"{cut_time:.2f}s; running a conservative cut retry"
            )
            motion_retry_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "path": str(sample_path)},
                        {
                            "type": "text",
                            "text": (
                                InternVLVisualJudge._build_cut_prompt(
                                    boundary_time_in_clip
                                )
                                + "\n\nThe first answer relied only on a "
                                "change in perspective, subject focus, framing, "
                                "or camera repositioning. Those observations "
                                "can result from a fast continuous pan and are "
                                "not direct edit evidence. Recheck the adjacent "
                                "frames. Return edit only if stable background "
                                "geometry or location changes discontinuously "
                                "in a way that physical camera motion cannot "
                                "explain. Otherwise return no_cut with "
                                "camera_motion_possible true."
                            ),
                        },
                    ],
                }
            ]
            try:
                motion_retry_answer = self._generate(
                    motion_retry_messages,
                    fps=settings.COSMOS3_CUT_FPS,
                )
                motion_retry_data = recover_json(motion_retry_answer)
                if (
                    motion_retry_data is not None
                    and "boundary_evidence" in motion_retry_data
                    and "camera_motion_possible" in motion_retry_data
                ):
                    answer = motion_retry_answer
                    data = motion_retry_data
            except Exception as exc:
                log(
                    f"Cosmos 3 conservative cut retry failed at "
                    f"{cut_time:.2f}s: {exc}"
                )
        if data is None:
            return InternVLVisualJudge._cut_error(
                cut_time,
                clip_start_time,
                boundary_time_in_clip,
                "json_recovery_failed",
                "Cosmos 3 did not return valid JSON.",
                answer,
            )

        boundary_evidence = normalise_cut_boundary_evidence(
            data.get("boundary_evidence")
        )
        camera_motion_possible = (
            True
            if "camera_motion_possible" not in data
            else normalise_bool(data.get("camera_motion_possible"))
        )
        short_reason = clean_text(data.get("short_reason"))
        transition_type = clean_text(
            data.get("transition_type") or "unclear"
        ).lower()
        if cut_reason_needs_motion_retry(short_reason):
            boundary_evidence = "continuous_motion"
            camera_motion_possible = True
            transition_type = "no_cut"
            short_reason = (
                "Rejected as a cut because perspective, framing, or subject "
                "changes alone do not prove an edit across adjacent frames."
            )
        return CutVerification(
            cut_time=cut_time,
            clip_start_time=clip_start_time,
            boundary_time_in_clip=boundary_time_in_clip,
            is_real_cut=(
                normalise_bool(data.get("is_real_cut"))
                and boundary_evidence == "edit"
                and not camera_motion_possible
            ),
            confidence=normalise_confidence(data.get("confidence")),
            transition_type=transition_type,
            short_reason=short_reason,
            raw_response=answer,
            boundary_evidence=boundary_evidence,
            camera_motion_possible=camera_motion_possible,
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
        segment_prompt = self._build_segment_prompt(
            metadata,
            segment_index,
            segment_count,
            start_time,
            end_time,
            video_duration,
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": str(sample_path)},
                    {
                        "type": "text",
                        "text": segment_prompt,
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
        burst_content = self._normalise_burst_content(
            data.get("burst_content") if data else None
        )
        expected_burst_count = len(
            segment_review_burst_plan(
                start_time,
                end_time,
                settings.COSMOS3_FRAMES_PER_SAMPLE,
                settings.COSMOS3_REVIEW_FPS,
            )[0]
        )
        if (
            data is None
            or len(burst_content) != expected_burst_count
            or "confidence" not in data
            or "walking_environment" not in data
            or "embedded_location_text" not in data
        ):
            log(
                f"Cosmos 3 returned an incomplete classification for "
                f"segment {segment_index}; retrying once"
            )
            retry_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "path": str(sample_path)},
                        {
                            "type": "text",
                            "text": (
                                segment_prompt
                                + "\n\nYour previous attempt was incomplete. "
                                "Return exactly one complete JSON object with "
                                f"exactly {expected_burst_count} "
                                "burst_content labels in chronological order. Set "
                                "embedded_location_text to an empty string. "
                                "Keep short_reason to one sentence. "
                                "Do not use Markdown fences."
                            ),
                        },
                    ],
                }
            ]
            try:
                answer = self._generate(
                    retry_messages,
                    fps=settings.COSMOS3_REVIEW_FPS,
                )
            except Exception as exc:
                return InternVLVisualJudge._segment_error(
                    segment_index,
                    start_time,
                    end_time,
                    "model_error",
                    f"Cosmos 3 JSON retry failed: {exc}",
                    answer,
                )
            data = recover_json(answer)
            burst_content = self._normalise_burst_content(
                data.get("burst_content") if data else None
            )
        if data is None:
            return InternVLVisualJudge._segment_error(
                segment_index,
                start_time,
                end_time,
                "json_recovery_failed",
                "Cosmos 3 did not return valid JSON.",
                answer,
            )
        if len(burst_content) != expected_burst_count:
            return InternVLVisualJudge._segment_error(
                segment_index,
                start_time,
                end_time,
                "invalid_burst_content",
                "Cosmos 3 did not classify every motion burst.",
                answer,
            )

        counts = {
            label: burst_content.count(label)
            for label in _BURST_CONTENT_TYPES
        }
        walking_count = counts["walking"]
        vehicle_count = counts["vehicle"]
        promotion_count = counts["promotion"]
        intro_count = counts["highlight"] + counts["map_title"]
        walking_fraction = walking_count / len(burst_content)
        promotion_fraction = promotion_count / len(burst_content)
        is_intro_highlights = intro_count > len(burst_content) / 2
        is_advertisement = (
            promotion_fraction > settings.COSMOS3_MAX_PROMOTION_FRACTION
        )
        required_walking_bursts = max(
            1,
            math.ceil(
                settings.COSMOS3_MIN_WALKING_FRACTION * 3 - 1e-9
            ),
        )
        is_walking = (
            walking_count >= required_walking_bursts
            and vehicle_count == 0
        )
        walking_environment = normalise_walking_environment(
            data.get("walking_environment"),
            is_walking,
        )
        embedded_location_text = normalise_embedded_location_text(
            data.get("embedded_location_text")
        )
        confidence = normalise_confidence(data.get("confidence"))
        include = (
            is_walking
            and not is_advertisement
            and not is_intro_highlights
            and confidence >= settings.MIN_SEGMENT_CONFIDENCE
        )

        if include:
            content_type = "walking"
        elif is_intro_highlights:
            content_type = "intro_highlights"
        elif is_advertisement:
            content_type = "advertisement"
        else:
            content_type = "nonwalking"

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
            decision_method="fixed_budget_burst_recall",
            burst_content=burst_content,
            walking_environment=walking_environment,
            embedded_location_text=embedded_location_text,
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
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"fps": fps},
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
    def _normalise_burst_content(value: Any) -> List[str]:
        if not isinstance(value, list) or not value:
            return []
        labels = [clean_text(item).lower() for item in value]
        if any(label not in _BURST_CONTENT_TYPES for label in labels):
            return []
        return labels

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
Confidence means certainty in either a timestamp or the null decision. A
confident null decision should therefore have high confidence.

Title: {title}
Description excerpt: {description}

Return exactly one valid JSON object and nothing else. It must contain
main_content_start_time as a timestamp in seconds or null, confidence as a
number from zero to one, and short_reason as one short sentence. Determine the
timestamp from this video rather than using a typical intro duration.
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
        expected_burst_count = len(
            segment_review_burst_plan(
                start_time,
                end_time,
                settings.COSMOS3_FRAMES_PER_SAMPLE,
                settings.COSMOS3_REVIEW_FPS,
            )[0]
        )
        return f"""
Review window {segment_index + 1} of {segment_count}, covering source video
time {start_time} to {end_time} seconds in a {video_duration:.1f} second
pedestrian walking tour candidate.

The review clip contains {expected_burst_count} short continuous motion bursts
distributed chronologically across the whole source window. Motion inside each
burst is real source motion and shows whether the camera is moving through the
location. The jumps between bursts are sampling artefacts, not source cuts or
a montage. Never reject the entire window because of one burst or a short
portion.

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

Classify each of the {expected_burst_count} bursts independently in
chronological order, using exactly one of these labels:

walking: the camera wearer is moving on foot through a real location, indoors
or outdoors, or is briefly paused during an otherwise pedestrian continuation
vehicle: driving, riding, cycling, boat, train, bus, drone, or other transport
static: a genuinely fixed tripod, aerial, scenic, or staged viewpoint with no
pedestrian progression; do not use static merely because motion is slow,
stabilised, or interrupted by a short pause
highlight: an opening preview, montage, recap, or teaser
map_title: a map, route graphic, logo, title card, or similar screen
promotion: inserted sponsor or channel promotion unrelated to the surroundings
other: game, animation, slideshow, screen recording, or unclear content

Compare the frames inside each burst for changes in nearby geometry, parallax,
footstep sway, and forward progression. A head height first person view along
a street, path, market, shop, bridge, or waterfront should be walking when
those changes show pedestrian movement. One genuine walking burst is enough
for the whole window when the remaining bursts are only static pauses. Never
apply this exception when any burst is vehicle content.

Standing temporarily and rotating or panning the camera to look around during
an ongoing first person pedestrian tour is valid walking context. Label that
burst walking rather than static. A brief channel introduction or title in one
burst does not invalidate a long window when another burst clearly shows real
walking, but a highlight reel or map dominated window must still be rejected.

Do not call road motion or scenic travel walking when the camera is moving in
or on a vehicle. Classify time of day as day, night, dawn_dusk, or unknown.
Indoor or ambiguous footage is unknown.

For genuine walking, classify the dominant walking_environment as exactly one
of street, indoor, beach, park_nature, trail, market, waterfront,
square_plaza, transport_hub, mixed, other, or unknown. Base this only on bursts
labelled walking, ignoring the environment of static bursts. Use beach for
walking on sand beside the sea. Use mixed only when the walking bursts cover
multiple environments with similar duration. Use not_applicable when valid
walking does not dominate the window.

Extract embedded_location_text as one JSON string containing the most specific
exact readable place name or location caption visibly written in the sampled
video bursts. Exclude channel names, creator branding, generic titles, and
inferred places. Return an empty string when no such location text is clearly
readable.

Confidence means certainty in the classification, not the probability that the
window should be included. A confident rejection should have high confidence.

Return exactly one valid JSON object and nothing else. It must contain these
keys: confidence, burst_content, time_of_day, walking_environment,
embedded_location_text, quality_issues, and short_reason. burst_content must be
a JSON array containing exactly {expected_burst_count} allowed labels in
chronological order. embedded_location_text must be a JSON string and
quality_issues must be a JSON array.
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
                f"confidence={verification.confidence:.2f}, "
                f"evidence={verification.boundary_evidence}"
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
    prior_reviews: Optional[List[Dict[str, Any]]] = None,
) -> List[SegmentReview]:
    reviews: List[SegmentReview] = []
    segment_count = len(segments)
    reusable_reviews: Dict[tuple[int, int, int], SegmentReview] = {}
    allowed_fields = set(SegmentReview.__dataclass_fields__)
    for value in prior_reviews or []:
        if not isinstance(value, dict) or value.get("error") is not None:
            continue
        try:
            cached_review = SegmentReview(
                **{
                    key: item
                    for key, item in value.items()
                    if key in allowed_fields
                }
            )
        except (TypeError, ValueError):
            continue
        reusable_reviews[
            (
                cached_review.segment_index,
                cached_review.start_time,
                cached_review.end_time,
            )
        ] = cached_review
    if reusable_reviews:
        log(
            f"Reusing {len(reusable_reviews)} completed segment review(s) "
            f"for {video_id}"
        )

    with tempfile.TemporaryDirectory(
        prefix=f"walk_segments_{video_id}_"
    ) as temporary_name:
        temporary_dir = Path(temporary_name)
        for segment_index, segment in enumerate(segments):
            start_time = int(segment["start_time"])
            end_time = int(segment["end_time"])
            timestamp_labels = timestamp_labels_for_segment(
                metadata,
                start_time,
                end_time,
            )
            sample_path = temporary_dir / f"segment_{segment_index:04d}.mp4"
            segment_duration = segment_duration_seconds(segment)
            forced_rejection = segment.get("_forced_rejection")
            cache_key = (segment_index, start_time, end_time)
            if cache_key in reusable_reviews:
                review = reusable_reviews[cache_key]
            elif isinstance(forced_rejection, dict):
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
                    walking_environment="not_applicable",
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
                    walking_environment="not_applicable",
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
            review.timestamp_labels = timestamp_labels
            review.location_source = segment_location_source(
                review.timestamp_labels,
                review.embedded_location_text,
            )
            reviews.append(review)
            log(
                f"{video_id} segment {segment_index}: "
                f"include={review.include}, type={review.content_type}, "
                f"confidence={review.confidence:.2f}, "
                f"time={review.time_of_day}, "
                f"environment={review.walking_environment}, "
                f"location_source={review.location_source}"
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if review.error is not None:
                log(
                    f"Stopping segment review for {video_id} after "
                    f"segment {segment_index} failed: {review.error}"
                )
                break
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
    content_start_second = semantic_content_start_second(
        duration,
        decision,
        searched_seconds,
    )

    base_segments: List[Dict[str, Any]] = []
    if content_start_second > 0:
        base_segments.append(
            {
                "start_time": 0,
                "end_time": content_start_second - 1,
                "_forced_rejection": content_start_rejection(decision),
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


def semantic_content_start_second(
    duration: float,
    decision: ContentStartDecision,
    searched_seconds: float,
) -> int:
    """Convert a reliable semantic start decision to an integer boundary."""
    if duration <= 0:
        return 0

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
    return content_start_second


def content_start_rejection(
    decision: ContentStartDecision,
) -> Dict[str, Any]:
    return {
        "confidence": decision.confidence,
        "content_type": "intro_highlights",
        "quality_issue": "intro_before_main_content",
        "short_reason": (
            decision.short_reason
            or "This footage precedes the main walking tour."
        ),
        "raw_response": decision.raw_response,
    }


def build_hybrid_review_segments(
    duration: float,
    decision: ContentStartDecision,
    searched_seconds: float,
    cut_times: List[float],
) -> tuple[List[Dict[str, Any]], int, int]:
    """Combine semantic opening detection with verified scene boundaries."""
    if duration <= 0:
        return [], 0, 0

    content_start_second = semantic_content_start_second(
        duration,
        decision,
        searched_seconds,
    )
    boundaries = list(cut_times)
    if content_start_second > 0:
        boundaries.append(float(content_start_second - 1))

    cut_segments = build_segments(duration, boundaries)
    for segment in cut_segments:
        if int(segment["end_time"]) < content_start_second:
            segment["_forced_rejection"] = content_start_rejection(decision)

    return (
        split_long_segments(
            cut_segments,
            settings.MAX_SEGMENT_REVIEW_SECONDS,
            settings.MIN_SEGMENT_DURATION_SECONDS,
        ),
        content_start_second,
        len(cut_segments),
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
            "walking_environment": review.walking_environment,
            "timestamp_labels": review.timestamp_labels,
            "embedded_location_text": review.embedded_location_text,
            "location_source": review.location_source,
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


def _process_visual_candidate(
    video_id: str,
    record: Dict[str, Any],
    judge: Any,
    video_path: Optional[Path] = None,
) -> None:
    previous_review_version = record.get("visual_review_version")
    previous_visual_decision = record.get("visual_decision")
    prior_reviews: List[Dict[str, Any]] = []
    if (
        previous_review_version == settings.VISUAL_REVIEW_VERSION
        and isinstance(previous_visual_decision, dict)
        and isinstance(previous_visual_decision.get("segments"), list)
    ):
        prior_reviews = previous_visual_decision["segments"]
    record["visual_review_version"] = settings.VISUAL_REVIEW_VERSION
    metadata = record.get("metadata", {})
    timestamp_labels = extract_description_timestamp_labels(metadata)
    record["timestamp_labels"] = timestamp_labels
    if timestamp_labels:
        log(
            f"Found {len(timestamp_labels)} description timestamp "
            f"annotation(s) for {video_id}"
        )
    review_metadata = dict(metadata)
    review_metadata["_timestamp_labels"] = timestamp_labels
    if video_path is None:
        log(f"Downloading accepted metadata candidate {video_id}")
        video_path = download_video(video_id)
    else:
        log(f"Using prefetched video {video_id}: {video_path.name}")
    if video_path is None:
        record["status"] = "download_failed"
        record["error"] = "yt-dlp could not download the video"
        return

    record["local_video_path"] = str(video_path)
    duration = get_video_duration(video_path)
    if duration is None:
        record["status"] = "duration_failed"
        record["error"] = "ffprobe could not determine the duration"
        return

    record["duration_seconds"] = round(duration, 3)
    if duration < settings.MIN_VIDEO_DURATION_SECONDS:
        record["status"] = "visual_rejected"
        record["error"] = (
            "downloaded video is shorter than "
            f"{settings.MIN_VIDEO_DURATION_SECONDS} seconds"
        )
        return

    use_cosmos = settings.VISUAL_MODEL_BACKEND == "cosmos3"
    use_cosmos_fast_path = use_cosmos and settings.COSMOS3_FAST_MODE
    content_start: Optional[ContentStartDecision] = None
    searched_seconds = 0.0
    if use_cosmos:
        log(f"Locating the main walking content in {video_id}")
        content_start, searched_seconds = locate_main_content_start(
            video_id,
            video_path,
            duration,
            review_metadata,
            judge,
        )
        record["content_start_decision"] = asdict(content_start)
        if content_start.error is not None:
            record["status"] = "content_start_failed"
            record["error"] = (
                "Cosmos 3 content start inference failed: "
                f"{content_start.short_reason}"
            )
            return

    if use_cosmos_fast_path:
        assert content_start is not None
        candidate_segments, content_start_second = (
            build_cosmos_review_segments(
                duration,
                content_start,
                searched_seconds,
            )
        )
        record["cuts"] = {
            "method": "cosmos3_semantic_content_start",
            "threshold": None,
            "merge_nearby_seconds": None,
            "detected_cut_times": [],
            "cut_times": [],
            "semantic_content_start": content_start_second,
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
            "method": (
                "ffmpeg_scene_detection_with_semantic_start"
                if use_cosmos
                else "ffmpeg_scene_detection"
            ),
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
        log(
            f"Cut summary for {video_id}: FFmpeg candidates="
            f"{len(cut_result.cut_times)}, verified actual="
            f"{len(confirmed_cut_times)}, rejected="
            f"{len(cut_result.cut_times) - len(confirmed_cut_times)}"
        )
        if use_cosmos:
            assert content_start is not None
            (
                candidate_segments,
                content_start_second,
                cut_segment_count,
            ) = build_hybrid_review_segments(
                duration,
                content_start,
                searched_seconds,
                confirmed_cut_times,
            )
            record["cuts"]["semantic_content_start"] = (
                content_start_second
            )
        else:
            cut_segments = build_segments(duration, confirmed_cut_times)
            candidate_segments = split_long_segments(
                cut_segments,
                settings.MAX_SEGMENT_REVIEW_SECONDS,
                settings.MIN_SEGMENT_DURATION_SECONDS,
            )
            cut_segment_count = len(cut_segments)
        log(
            f"Prepared {len(candidate_segments)} fixed duration review "
            f"windows from {cut_segment_count} cut based segments for "
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
        review_metadata,
        candidate_segments,
        judge,
        prior_reviews,
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
        return

    record["status"] = "complete"
    record["error"] = None


_FINISHED_ANALYSIS_STATUSES = {"complete", "visual_rejected"}


def _record_video_path(
    record: Dict[str, Any],
    prefetched_path: Optional[Path],
) -> Optional[Path]:
    path_value = record.get("local_video_path")
    if isinstance(path_value, str) and path_value.strip():
        return Path(path_value)
    return prefetched_path


def _finalise_analysed_video_file(
    record: Dict[str, Any],
    prefetched_path: Optional[Path],
) -> None:
    if record.get("status") not in _FINISHED_ANALYSIS_STATUSES:
        return

    video_path = _record_video_path(record, prefetched_path)
    if video_path is None:
        record["video_file_kept"] = False
        return

    if settings.KEEP_VIDEO_FILES_AFTER_ANALYSIS:
        record["video_file_kept"] = video_path.exists()
        return

    try:
        video_path.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        record["video_file_kept"] = video_path.exists()
        record["video_file_cleanup_error"] = str(exc)
        log(f"Could not delete analysed video {video_path}: {exc}")
        return

    record["local_video_path"] = None
    record["video_file_kept"] = False
    record.pop("video_file_cleanup_error", None)
    log(f"Deleted analysed video file: {video_path}")


def process_visual_candidate(
    video_id: str,
    record: Dict[str, Any],
    judge: Any,
    video_path: Optional[Path] = None,
) -> None:
    """Analyse one video and apply the configured file retention policy."""
    record["status"] = "visual_processing"
    try:
        _process_visual_candidate(video_id, record, judge, video_path)
    finally:
        _finalise_analysed_video_file(record, video_path)


DownloadQueueItem = Tuple[
    str,
    Dict[str, Any],
    Optional[Path],
    Optional[str],
]


def _fill_download_queue(
    pending: List[Tuple[str, Dict[str, Any]]],
    ready_queue: "queue.Queue[DownloadQueueItem]",
    stop_event: threading.Event,
    producer_finished: threading.Event,
) -> None:
    try:
        for video_id, record in pending:
            if stop_event.is_set():
                break

            video_path: Optional[Path] = None
            download_error: Optional[str] = None
            try:
                log(f"Prefetching accepted metadata candidate {video_id}")
                video_path = download_video(video_id)
                if video_path is None:
                    download_error = "yt-dlp could not download the video"
            except Exception as exc:
                download_error = str(exc)
                log(f"Video prefetch failed for {video_id}: {exc}")

            item = (video_id, record, video_path, download_error)
            while not stop_event.is_set():
                try:
                    ready_queue.put(item, timeout=0.25)
                    break
                except queue.Full:
                    continue
    finally:
        producer_finished.set()


def _next_downloaded_video(
    ready_queue: "queue.Queue[DownloadQueueItem]",
    producer_finished: threading.Event,
) -> DownloadQueueItem:
    while True:
        try:
            return ready_queue.get(timeout=0.50)
        except queue.Empty:
            if producer_finished.is_set():
                raise RuntimeError(
                    "The video download worker stopped before every pending "
                    "video produced a result."
                )


def _wait_for_initial_download_buffer(
    ready_queue: "queue.Queue[DownloadQueueItem]",
    producer_finished: threading.Event,
    target: int,
) -> None:
    while ready_queue.qsize() < target:
        if producer_finished.wait(timeout=0.10):
            break

    log(
        f"Initial download queue ready: {ready_queue.qsize()}/{target} "
        "video(s)"
    )


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

    # One extra slot ensures that, while one video is being analysed, the
    # configured number can remain downloaded and waiting whenever enough
    # pending videos exist.
    queue_capacity = settings.VIDEO_DOWNLOAD_QUEUE_SIZE + 1
    ready_queue: "queue.Queue[DownloadQueueItem]" = queue.Queue(
        maxsize=queue_capacity
    )
    stop_event = threading.Event()
    producer_finished = threading.Event()
    producer = threading.Thread(
        target=_fill_download_queue,
        args=(pending, ready_queue, stop_event, producer_finished),
        name="walking-video-prefetch",
        daemon=True,
    )
    producer.start()

    log(
        "Video download queue target: "
        f"{settings.VIDEO_DOWNLOAD_QUEUE_SIZE} waiting video(s); "
        "keep analysed files="
        f"{settings.KEEP_VIDEO_FILES_AFTER_ANALYSIS}"
    )

    judge: Any = None
    processed = 0
    try:
        judge = create_visual_judge()
        initial_target = min(queue_capacity, len(pending))
        _wait_for_initial_download_buffer(
            ready_queue,
            producer_finished,
            initial_target,
        )

        for _ in pending:
            (
                video_id,
                record,
                video_path,
                download_error,
            ) = _next_downloaded_video(ready_queue, producer_finished)
            try:
                if video_path is None:
                    record["visual_review_version"] = (
                        settings.VISUAL_REVIEW_VERSION
                    )
                    record["status"] = "download_failed"
                    record["error"] = (
                        download_error
                        or "yt-dlp could not download the video"
                    )
                    log(f"Download failed for {video_id}: {record['error']}")
                else:
                    process_visual_candidate(
                        video_id,
                        record,
                        judge,
                        video_path,
                    )
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
        stop_event.set()
        producer.join(timeout=1.0)
        if judge is not None:
            unload_model(judge)
    return processed