"""FFmpeg scene cut detection and segment construction."""

from __future__ import annotations

import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

from . import settings
from .shared import require_binary, run_command


@dataclass
class CutDetectionResult:
    cut_times: List[float]
    duration_seconds: float
    message: str


def run_ffmpeg_scene_detection(
    video_path: Path, duration: float
) -> CutDetectionResult:
    require_binary("ffmpeg")
    if duration <= 0:
        return CutDetectionResult([], duration, "invalid_duration")

    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-i",
        str(video_path),
        "-vf",
        (
            "setpts=PTS-STARTPTS,"
            f"select='gt(scene,{settings.SCENE_THRESHOLD})',showinfo"
        ),
        "-an",
        "-f",
        "null",
        "-",
    ]
    try:
        result = run_command(
            command, timeout=settings.CUT_DETECTION_TIMEOUT
        )
    except subprocess.TimeoutExpired:
        return CutDetectionResult([], duration, "ffmpeg_timeout")

    if result.returncode != 0:
        message = (
            result.stderr.splitlines()[-1]
            if result.stderr
            else "ffmpeg_failed"
        )
        return CutDetectionResult([], duration, message)

    cut_times: List[float] = []
    for line in result.stderr.splitlines():
        match = re.search(
            r"pts_time:([+-]?(?:\d+(?:\.\d*)?|\.\d+))", line
        )
        if match:
            value = float(match.group(1))
            if math.isfinite(value):
                cut_times.append(value)

    filtered = filter_cut_edges(cut_times, duration)
    merged = merge_nearby_times(filtered)
    return CutDetectionResult(merged, duration, "")


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