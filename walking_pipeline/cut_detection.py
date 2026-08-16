"""FFmpeg scene cut detection and segment construction."""

from __future__ import annotations

import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from . import settings
from .shared import log, require_binary, run_command


CUT_DETECTION_SCHEMA_VERSION = "walking_ffmpeg_cuda_v1"

_HIGH_BIT_DEPTH_PIXEL_FORMAT_HINTS = (
    "10",
    "12",
    "14",
    "16",
    "p010",
    "p012",
    "p016",
)


@dataclass
class CutDetectionResult:
    cut_times: List[float]
    duration_seconds: float
    message: str


def _number_text(value: float) -> str:
    """Return a compact locale independent FFmpeg numeric value."""
    return format(float(value), ".8g")


def _probe_source_pixel_format(video_path: Path) -> str:
    """Return the first video stream pixel format when FFprobe can read it."""
    try:
        result = run_command(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=pix_fmt",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""

    if result.returncode != 0:
        return ""
    for line in result.stdout.splitlines():
        value = line.strip().lower()
        if value:
            return value
    return ""


def _cuda_download_formats(pixel_format: str) -> Tuple[str, str]:
    """Order CUDA download formats using the encoded source bit depth."""
    is_high_bit_depth = any(
        hint in pixel_format
        for hint in _HIGH_BIT_DEPTH_PIXEL_FORMAT_HINTS
    )
    if is_high_bit_depth:
        return ("p010le", "nv12")
    return ("nv12", "p010le")


def _filter_graph(*, cuda_download_format: str | None) -> str:
    fps = _number_text(settings.CUT_DETECTION_FPS)
    width = settings.CUT_DETECTION_WIDTH

    if cuda_download_format:
        preprocessing = (
            f"[0:v]scale_cuda={width}:-2,"
            "hwdownload,"
            f"format={cuda_download_format},"
            "format=yuv420p,"
            f"fps={fps},"
        )
    else:
        preprocessing = (
            f"[0:v]fps={fps},"
            f"scale={width}:-2:flags=fast_bilinear,"
            "format=yuv420p,"
        )

    return (
        f"{preprocessing}setpts=PTS-STARTPTS,split=2"
        "[scene_input][adaptive_input];"
        "[scene_input]"
        f"select='gt(scene,{settings.SCENE_THRESHOLD})',"
        "showinfo[scene_output];"
        "[adaptive_input]"
        f"scdet=t={settings.SCDET_THRESHOLD}:s=1,"
        "metadata=mode=print:key=lavfi.scd.time[adaptive_output]"
    )


def _ffmpeg_detection_command(
    video_path: Path,
    *,
    cuda_download_format: str | None,
) -> List[str]:
    command = ["ffmpeg", "-hide_banner", "-nostdin"]
    if cuda_download_format:
        command.extend(
            [
                "-hwaccel",
                "cuda",
                "-hwaccel_output_format",
                "cuda",
            ]
        )
    command.extend(
        [
            "-i",
            str(video_path),
            "-filter_complex",
            _filter_graph(cuda_download_format=cuda_download_format),
            "-map",
            "[scene_output]",
            "-map",
            "[adaptive_output]",
            "-an",
            "-f",
            "null",
            "-",
        ]
    )
    return command


def _result_error(result: subprocess.CompletedProcess[str]) -> str:
    lines = [line.strip() for line in result.stderr.splitlines() if line.strip()]
    return lines[-1] if lines else "ffmpeg_failed"


def _extract_cut_times(stderr: str, duration: float) -> List[float]:
    cut_times: List[float] = []
    for line in stderr.splitlines():
        values: List[str] = []
        if "showinfo" in line:
            match = re.search(
                r"pts_time:([+-]?(?:\d+(?:\.\d*)?|\.\d+))",
                line,
            )
            if match:
                values.append(match.group(1))
        adaptive_match = re.search(
            r"lavfi\.scd\.time=([+-]?(?:\d+(?:\.\d*)?|\.\d+))",
            line,
        )
        if adaptive_match:
            values.append(adaptive_match.group(1))
        for text_value in values:
            value = float(text_value)
            if math.isfinite(value):
                cut_times.append(value)

    filtered = filter_cut_edges(cut_times, duration)
    return merge_nearby_times(filtered)


def _completed_detection_result(
    result: subprocess.CompletedProcess[str],
    duration: float,
) -> CutDetectionResult:
    return CutDetectionResult(
        _extract_cut_times(result.stderr, duration),
        duration,
        "",
    )


def run_ffmpeg_scene_detection(
    video_path: Path, duration: float
) -> CutDetectionResult:
    require_binary("ffmpeg")
    if duration <= 0:
        return CutDetectionResult([], duration, "invalid_duration")

    backend = settings.CUT_DETECTION_BACKEND
    failure_messages: List[str] = []

    if backend in {"auto", "ffmpeg_cuda"}:
        pixel_format = _probe_source_pixel_format(video_path)
        for download_format in _cuda_download_formats(pixel_format):
            log(
                "Running CUDA cut detection at "
                f"{_number_text(settings.CUT_DETECTION_FPS)} FPS and "
                f"width {settings.CUT_DETECTION_WIDTH}; "
                f"download format={download_format}"
            )
            command = _ffmpeg_detection_command(
                video_path,
                cuda_download_format=download_format,
            )
            try:
                result = run_command(
                    command,
                    timeout=settings.CUT_DETECTION_TIMEOUT,
                )
            except subprocess.TimeoutExpired:
                failure_messages.append("ffmpeg_cuda_timeout")
                break

            if result.returncode == 0:
                log("CUDA cut detection completed successfully")
                return _completed_detection_result(result, duration)

            error = _result_error(result)
            failure_messages.append(
                f"ffmpeg_cuda_{download_format}: {error}"
            )
            format_error = (
                "invalid output format" in result.stderr.lower()
                or "failed to configure output pad"
                in result.stderr.lower()
            )
            if not format_error:
                break

        if not settings.CUT_DETECTION_CPU_FALLBACK:
            return CutDetectionResult(
                [],
                duration,
                " | ".join(failure_messages) or "ffmpeg_cuda_failed",
            )
        log("CUDA cut detection failed; using the CPU fallback")

    cpu_command = _ffmpeg_detection_command(
        video_path,
        cuda_download_format=None,
    )
    try:
        cpu_result = run_command(
            cpu_command,
            timeout=settings.CUT_DETECTION_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        failure_messages.append("ffmpeg_cpu_timeout")
        return CutDetectionResult(
            [],
            duration,
            " | ".join(failure_messages),
        )

    if cpu_result.returncode == 0:
        log("CPU cut detection completed successfully")
        return _completed_detection_result(cpu_result, duration)

    failure_messages.append(
        f"ffmpeg_cpu: {_result_error(cpu_result)}"
    )
    return CutDetectionResult(
        [],
        duration,
        " | ".join(failure_messages),
    )


def filter_cut_edges(
    times: Sequence[float], duration: float
) -> List[float]:
    result: List[float] = []
    lower = max(0.0, settings.IGNORE_FIRST_SEC)
    upper = max(lower, duration - settings.IGNORE_LAST_SEC)
    for value in sorted(times):
        if value <= lower or value >= upper:
            continue
        result.append(value)
    return result


def merge_nearby_times(times: Sequence[float]) -> List[float]:
    if not times:
        return []

    ordered = sorted(float(value) for value in times)
    merged: List[float] = []
    group: List[float] = [ordered[0]]
    for value in ordered[1:]:
        if value - group[-1] <= settings.MERGE_NEARBY_SEC:
            group.append(value)
        else:
            merged.append(group[0])
            group = [value]
    merged.append(group[0])
    return merged


def build_segments(
    duration: float,
    cut_times: Sequence[float],
) -> List[Dict[str, Any]]:
    if duration <= 0:
        return []

    final_second = max(0, int(math.floor(duration)))
    integer_cuts: List[int] = []
    for cut_time in sorted(cut_times):
        cut_second = int(math.floor(cut_time))
        if 0 <= cut_second < final_second:
            if not integer_cuts or cut_second > integer_cuts[-1]:
                integer_cuts.append(cut_second)

    segments: List[Dict[str, Any]] = []
    start_second = 0
    for end_second in [*integer_cuts, final_second]:
        if end_second < start_second:
            continue
        segments.append(
            {
                "start_time": start_second,
                "end_time": end_second,
            }
        )
        start_second = end_second + 1
    return segments


def segment_duration_seconds(segment: Dict[str, Any]) -> int:
    """Return the inclusive duration of an integer second segment."""
    try:
        start_time = int(segment["start_time"])
        end_time = int(segment["end_time"])
    except (KeyError, TypeError, ValueError):
        return 0
    return max(0, end_time - start_time + 1)


def split_long_segments(
    segments: Sequence[Dict[str, Any]],
    max_duration_seconds: int,
    min_duration_seconds: int = 1,
) -> List[Dict[str, Any]]:
    """Split long segments into balanced, length independent review windows.

    Balancing prevents an artificial final window shorter than the configured
    minimum. If the requested maximum and minimum cannot both be satisfied,
    the original segment remains intact so that footage is not discarded only
    because of an artificial boundary.
    """
    if max_duration_seconds < 1:
        raise ValueError("max_duration_seconds must be at least 1")
    if min_duration_seconds < 1:
        raise ValueError("min_duration_seconds must be at least 1")

    review_segments: List[Dict[str, Any]] = []
    for segment in segments:
        duration = segment_duration_seconds(segment)
        if duration <= 0:
            continue

        start_time = int(segment["start_time"])
        end_time = int(segment["end_time"])
        window_count = max(1, math.ceil(duration / max_duration_seconds))
        if duration // window_count < min_duration_seconds:
            window_count = max(1, duration // min_duration_seconds)

        base_duration, extra_seconds = divmod(duration, window_count)
        window_start = start_time
        for window_index in range(window_count):
            window_duration = base_duration + (
                1 if window_index < extra_seconds else 0
            )
            window_end = min(
                end_time,
                window_start + window_duration - 1,
            )
            review_segment = dict(segment)
            review_segment["start_time"] = window_start
            review_segment["end_time"] = window_end
            review_segments.append(review_segment)
            window_start = window_end + 1

    return review_segments