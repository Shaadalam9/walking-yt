#!/usr/bin/env python3
"""
Interactive frame selector for the walking dataset.

What it does
------------
1. Reads walking mapping.csv only.
2. Expands every video/segment pair from the nested mapping fields.
3. Randomly selects an unseen walking segment.
4. Shows its locality, country, environment, start/end time and midpoint.
5. Asks before downloading anything.
6. Downloads only a short section around the segment midpoint.
7. Opens a frame browser so you can move backwards and forwards.
8. Saves exactly one selected frame as:

       {video_id}_{time}.png

The script keeps a separate history file so the same walking segment is not
selected again across runs unless you explicitly reset the history.

Requirements
------------
yt-dlp
ffmpeg
ffprobe
Python tkinter

This version includes automatic YouTube 403 recovery. It tries multiple
YouTube player/format strategies instead of relying on the single format
chosen by yt-dlp.

Usage
-----
python3 extract_walking_frame_browser.py

Optional
--------
python3 extract_walking_frame_browser.py --frame-step 10
python3 extract_walking_frame_browser.py --reset-history
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DOWNLOAD_CONTEXT_SECONDS = 2.5
BROWSE_RADIUS_SECONDS = 1.5
PREVIEW_MAX_WIDTH = 1280
PREVIEW_MAX_HEIGHT = 800

DEFAULT_HISTORY_FILE = Path(".walking_frame_history.txt")


@dataclass(frozen=True)
class WalkingSegment:
    mapping_row_id: str
    locality: str
    state: str
    country: str
    continent: str
    video_id: str
    start_time: float
    end_time: float
    time_of_day: str
    walking_environment: str

    @property
    def midpoint(self) -> float:
        return (self.start_time + self.end_time) / 2.0

    @property
    def history_key(self) -> str:
        return (
            f"{self.video_id}|"
            f"{self.start_time:.3f}|"
            f"{self.end_time:.3f}"
        )


@dataclass(frozen=True)
class PreviewFrame:
    path: Path
    source_time: float
    source_frame_index: int


class PreviewDownloadError(RuntimeError):
    """Raised after every safe preview download strategy has failed."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly choose an unseen walking segment from mapping.csv, "
            "browse nearby frames and save exactly one selected frame."
        )
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("mapping.csv"),
        help="Path to the walking mapping.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("walking_frames"),
        help="Directory for selected frames",
    )
    parser.add_argument(
        "--history-file",
        type=Path,
        default=DEFAULT_HISTORY_FILE,
        help="Persistent history of already shown walking segments",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=5,
        help=(
            "Number of frames moved by normal Previous/Next buttons "
            "(default: 5)"
        ),
    )
    parser.add_argument(
        "--reset-history",
        action="store_true",
        help="Forget previously shown walking segments before starting",
    )
    parser.add_argument(
        "--cookies-from-browser",
        default=None,
        help=(
            "Optional yt-dlp browser cookie source, for example "
            "'safari', 'chrome' or 'firefox'. Normally leave this unset."
        ),
    )
    parser.add_argument(
        "--cookies",
        type=Path,
        default=None,
        help="Optional Netscape-format cookie file passed to yt-dlp",
    )
    return parser.parse_args()


def require_program(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(
            f"Required command '{name}' was not found on PATH."
        )


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]


def parse_simple_bracket_cell(raw: Any) -> Any:
    """
    Parse the nested bracket format used by the walking mapping file.

    This parser is intentionally used only for fields whose elements do not
    themselves contain commas, namely videos, start_time, end_time,
    time_of_day and walking_environment.

    Examples:
        [abc]                  -> ["abc"]
        [abc,def]              -> ["abc", "def"]
        [[1,2,3]]              -> [["1", "2", "3"]]
        [[1,2],[3,4]]          -> [["1", "2"], ["3", "4"]]
    """
    text = str(raw or "").strip()
    if not text:
        return []

    index = 0

    def skip_space() -> None:
        nonlocal index
        while index < len(text) and text[index].isspace():
            index += 1

    def parse_value() -> Any:
        nonlocal index
        skip_space()

        if index < len(text) and text[index] == "[":
            index += 1
            items: list[Any] = []
            skip_space()

            if index < len(text) and text[index] == "]":
                index += 1
                return items

            while index < len(text):
                items.append(parse_value())
                skip_space()

                if index >= len(text):
                    break

                if text[index] == ",":
                    index += 1
                    continue

                if text[index] == "]":
                    index += 1
                    break

                raise ValueError(
                    "Unexpected character while parsing bracket cell "
                    f"at position {index}: {text[index]!r}"
                )

            return items

        start = index

        while index < len(text) and text[index] not in ",]":
            index += 1

        return text[start:index].strip().strip("'").strip('"')

    result = parse_value()
    skip_space()

    if index != len(text):
        raise ValueError(
            f"Could not completely parse bracket cell: {text!r}"
        )

    return result


def normalise_per_video(
    raw: Any,
    video_count: int,
) -> list[list[Any]]:
    """
    Normalise a mapping field to one inner list per video.

    Current walking mapping rows normally use nested lists even for a single
    video, e.g. [[45,254,463]]. This also tolerates a single flattened list.
    """
    parsed = parse_simple_bracket_cell(raw)

    if video_count == 0:
        return []

    if (
        isinstance(parsed, list)
        and len(parsed) == video_count
        and all(isinstance(item, list) for item in parsed)
    ):
        return parsed

    if video_count == 1:
        return [as_list(parsed)]

    return [[] for _ in range(video_count)]


def scalar_at(values: list[Any], index: int) -> str:
    if index >= len(values):
        return ""

    value = values[index]

    if isinstance(value, list):
        return ""

    return str(value or "").strip()


def read_mapping_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Mapping file not found: {path}")

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    required = {
        "id",
        "locality",
        "state",
        "country",
        "continent",
        "videos",
        "time_of_day",
        "walking_environment",
        "start_time",
        "end_time",
    }

    present = set(reader.fieldnames or [])
    missing = sorted(required - present)

    if missing:
        raise ValueError(
            "mapping.csv is missing required walking columns: "
            + ", ".join(missing)
        )

    return rows


def expand_mapping_row(
    row: dict[str, str],
) -> list[WalkingSegment]:
    videos = [
        str(value).strip()
        for value in as_list(
            parse_simple_bracket_cell(
                row.get("videos", "")
            )
        )
        if str(value).strip()
    ]

    if not videos:
        return []

    starts = normalise_per_video(
        row.get("start_time", ""),
        len(videos),
    )
    ends = normalise_per_video(
        row.get("end_time", ""),
        len(videos),
    )
    times_of_day = normalise_per_video(
        row.get("time_of_day", ""),
        len(videos),
    )
    environments = normalise_per_video(
        row.get("walking_environment", ""),
        len(videos),
    )

    segments: list[WalkingSegment] = []

    for video_index, video_id in enumerate(videos):
        video_starts = starts[video_index]
        video_ends = ends[video_index]
        video_times = times_of_day[video_index]
        video_environments = environments[video_index]

        pair_count = min(
            len(video_starts),
            len(video_ends),
        )

        for segment_index in range(pair_count):
            start = as_float(
                video_starts[segment_index]
            )
            end = as_float(
                video_ends[segment_index]
            )

            if (
                start is None
                or end is None
                or start < 0
                or end <= start
            ):
                continue

            segments.append(
                WalkingSegment(
                    mapping_row_id=str(
                        row.get("id", "")
                    ).strip(),
                    locality=str(
                        row.get("locality", "")
                    ).strip(),
                    state=str(
                        row.get("state", "")
                    ).strip(),
                    country=str(
                        row.get("country", "")
                    ).strip(),
                    continent=str(
                        row.get("continent", "")
                    ).strip(),
                    video_id=video_id,
                    start_time=start,
                    end_time=end,
                    time_of_day=scalar_at(
                        video_times,
                        segment_index,
                    ),
                    walking_environment=scalar_at(
                        video_environments,
                        segment_index,
                    ),
                )
            )

    return segments


def build_segment_pool(
    mapping_rows: list[dict[str, str]],
) -> list[WalkingSegment]:
    segments: list[WalkingSegment] = []

    for row in mapping_rows:
        segments.extend(
            expand_mapping_row(row)
        )

    return segments


def load_history(path: Path) -> set[str]:
    if not path.exists():
        return set()

    return {
        line.strip()
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    }


def append_history(
    path: Path,
    key: str,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(key + "\n")


def choose_unseen_segment(
    segments: list[WalkingSegment],
    history: set[str],
    excluded_video_ids: set[str] | None = None,
) -> WalkingSegment | None:
    excluded_video_ids = excluded_video_ids or set()

    unseen = [
        segment
        for segment in segments
        if (
            segment.history_key not in history
            and segment.video_id not in excluded_video_ids
        )
    ]

    if not unseen:
        return None

    # Prefer a source video not seen before. After every source video has been
    # seen, continue through unseen segments from previously used videos.
    seen_video_ids = {
        key.split("|", 1)[0]
        for key in history
        if "|" in key
    }

    unseen_video_segments = [
        segment
        for segment in unseen
        if segment.video_id not in seen_video_ids
    ]

    pool = (
        unseen_video_segments
        if unseen_video_segments
        else unseen
    )

    return random.SystemRandom().choice(pool)


def format_seconds(value: float) -> str:
    whole_minutes, seconds = divmod(
        value,
        60.0,
    )
    hours, minutes = divmod(
        int(whole_minutes),
        60,
    )

    if hours:
        return (
            f"{hours:02d}:"
            f"{minutes:02d}:"
            f"{seconds:06.3f}"
        )

    return (
        f"{minutes:02d}:"
        f"{seconds:06.3f}"
    )


def youtube_url(video_id: str) -> str:
    return (
        "https://www.youtube.com/watch?v="
        f"{video_id}"
    )


def print_segment(
    segment: WalkingSegment,
) -> None:
    print()
    print("=" * 72)
    print("RANDOM WALKING SEGMENT")
    print("=" * 72)
    print(
        f"Mapping row:          "
        f"{segment.mapping_row_id}"
    )
    print(
        f"Locality:             "
        f"{segment.locality or 'N/A'}"
    )
    print(
        f"State/region:         "
        f"{segment.state or 'N/A'}"
    )
    print(
        f"Country:              "
        f"{segment.country or 'N/A'}"
    )
    print(
        f"Continent:            "
        f"{segment.continent or 'N/A'}"
    )
    print(
        f"Video ID:             "
        f"{segment.video_id}"
    )
    print(
        "Retained segment:     "
        f"{format_seconds(segment.start_time)} "
        "to "
        f"{format_seconds(segment.end_time)}"
    )
    print(
        "BROWSER START POINT:  "
        f"{format_seconds(segment.midpoint)} "
        f"({segment.midpoint:.3f} s)"
    )
    print(
        f"Walking environment:  "
        f"{segment.walking_environment or 'N/A'}"
    )
    print(
        f"Time of day:          "
        f"{segment.time_of_day or 'N/A'}"
    )
    print(
        f"YouTube source:       "
        f"{youtube_url(segment.video_id)}"
    )
    print()
    print(
        "If you continue, the browser will open around the midpoint of "
        "this retained walking segment. You will see the actual frame "
        "pixels before saving anything."
    )
    print("=" * 72)


def ask_segment_action() -> str:
    while True:
        answer = input(
            "\nOpen frame browser for this walking segment? "
            "[y = yes, r = another random segment, n = stop]: "
        ).strip().casefold()

        if answer in {"y", "yes"}:
            return "yes"

        if answer in {
            "r",
            "random",
            "another",
        }:
            return "reroll"

        if answer in {
            "n",
            "no",
            "",
        }:
            return "no"

        print("Please enter y, r, or n.")


def _remove_previous_download_attempts(
    temp_dir: Path,
) -> None:
    for path in temp_dir.glob("source*"):
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        except FileNotFoundError:
            pass


def _downloaded_source_file(
    temp_dir: Path,
) -> Path | None:
    candidates = [
        path
        for path in temp_dir.glob("source.*")
        if (
            path.is_file()
            and not path.name.endswith(".part")
            and path.suffix.lower()
            in {
                ".mp4",
                ".mkv",
                ".webm",
                ".mov",
                ".ts",
            }
        )
    ]

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda path: path.stat().st_size,
    )


def _auth_args(
    args: argparse.Namespace,
) -> list[str]:
    result: list[str] = []

    if args.cookies is not None:
        result.extend(
            [
                "--cookies",
                str(args.cookies),
            ]
        )

    if args.cookies_from_browser:
        result.extend(
            [
                "--cookies-from-browser",
                str(args.cookies_from_browser),
            ]
        )

    return result


def _brief_failure(output: str) -> str:
    useful: list[str] = []

    for raw_line in output.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        lowered = line.casefold()

        if (
            "403" in lowered
            or "forbidden" in lowered
            or "requested format" in lowered
            or "not available" in lowered
            or lowered.startswith("error:")
            or "[error]" in lowered
        ):
            # Do not echo long signed googlevideo URLs.
            if len(line) > 240:
                line = line[:237] + "..."

            useful.append(line)

    if not useful:
        useful = [
            line.strip()
            for line in output.splitlines()
            if line.strip()
        ][-4:]

    return "\n".join(useful[-6:])


def _run_download_attempt(
    *,
    segment: WalkingSegment,
    temp_dir: Path,
    clip_start: float,
    clip_end: float,
    args: argparse.Namespace,
    label: str,
    format_selector: str,
    player_client: str | None,
) -> tuple[Path | None, str]:
    _remove_previous_download_attempts(
        temp_dir
    )

    output_template = str(
        temp_dir / "source.%(ext)s"
    )

    command = [
        "yt-dlp",
        "--ignore-config",
        "--no-playlist",
        "--check-formats",
        "--retries",
        "3",
        "--fragment-retries",
        "3",
        "--retry-sleep",
        "1",
        "--download-sections",
        f"*{clip_start:.3f}-{clip_end:.3f}",
        "--force-keyframes-at-cuts",
        "-f",
        format_selector,
        "--merge-output-format",
        "mp4",
        "-o",
        output_template,
    ]

    if player_client:
        command.extend(
            [
                "--extractor-args",
                f"youtube:player_client={player_client}",
            ]
        )

    command.extend(
        _auth_args(args)
    )
    command.append(
        youtube_url(segment.video_id)
    )

    print(
        f"  Trying {label}..."
    )

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    source = _downloaded_source_file(
        temp_dir
    )

    if (
        result.returncode == 0
        and source is not None
        and source.stat().st_size > 0
    ):
        print(
            f"  Success with {label}."
        )
        return source, result.stdout

    print(
        f"  {label} failed."
    )

    brief = _brief_failure(
        result.stdout
    )

    if brief:
        for line in brief.splitlines():
            print(
                f"    {line}"
            )

    return None, result.stdout


def download_preview_clip(
    segment: WalkingSegment,
    temp_dir: Path,
    args: argparse.Namespace,
) -> tuple[Path, float]:
    """
    Download only a short section around the walking segment midpoint.

    YouTube currently applies different playback requirements to different
    clients and formats. A format URL that yt-dlp can discover may still give
    ffmpeg HTTP 403. We therefore do not trust one automatically selected
    format. Instead, we try several independent strategies.

    The first strategy prefers Safari HLS, which avoids the exact direct-format
    path that produced the reported itag 701 HTTP 403. Later strategies use
    other player clients and increasingly conservative formats.
    """
    centre = segment.midpoint

    clip_start = max(
        segment.start_time,
        centre - DOWNLOAD_CONTEXT_SECONDS,
        0.0,
    )

    clip_end = min(
        segment.end_time,
        centre + DOWNLOAD_CONTEXT_SECONDS,
    )

    if clip_end <= clip_start:
        raise RuntimeError(
            "The selected walking segment is too short "
            "to create a preview."
        )

    print()
    print(
        f"Downloading only {clip_start:.3f} to "
        f"{clip_end:.3f} seconds from the source video."
    )
    print(
        "If YouTube rejects one stream, the script will "
        "automatically try another."
    )

    attempts = [
        (
            "web_safari HLS up to 1080p",
            (
                "b[protocol^=m3u8][height<=1080]"
                "/bv*[protocol^=m3u8][height<=1080]+ba"
                "/b[height<=1080]"
            ),
            "web_safari",
        ),
        (
            "TV H.264/AAC up to 1080p",
            (
                "bv*[height<=1080][vcodec^=avc1]"
                "+ba[acodec^=mp4a]"
                "/b[height<=1080][ext=mp4]"
                "/b[height<=1080]"
            ),
            "tv",
        ),
        (
            "web embedded up to 720p",
            (
                "b[height<=720]"
                "/bv*[height<=720]+ba"
                "/b"
            ),
            "web_embedded",
        ),
        (
            "conservative progressive fallback",
            (
                "18"
                "/b[height<=480]"
                "/b"
            ),
            "web_safari",
        ),
    ]

    failures: list[str] = []

    for (
        label,
        format_selector,
        player_client,
    ) in attempts:
        source, output = _run_download_attempt(
            segment=segment,
            temp_dir=temp_dir,
            clip_start=clip_start,
            clip_end=clip_end,
            args=args,
            label=label,
            format_selector=format_selector,
            player_client=player_client,
        )

        if source is not None:
            return source, clip_start

        failures.append(
            f"{label}: {_brief_failure(output)}"
        )

    cookie_hint = ""

    if (
        args.cookies is None
        and not args.cookies_from_browser
    ):
        cookie_hint = (
            "\n\nIf the video plays in your browser but every strategy above "
            "still fails, first update yt-dlp and then retry with browser "
            "cookies, for example:\n"
            "  python3 extract_walking_frame_browser_v2.py "
            "--cookies-from-browser safari\n"
            "or replace 'safari' with the browser you actually use."
        )

    raise PreviewDownloadError(
        "YouTube rejected every preview stream for "
        f"{segment.video_id}. This source video will be skipped for the "
        "rest of the current run."
        + cookie_hint
    )


def probe_frame_times(
    clip_path: Path,
) -> list[float]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=best_effort_timestamp_time",
        "-of",
        "csv=p=0",
        str(clip_path),
    ]

    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )

    values: list[float] = []

    for line in result.stdout.splitlines():
        value = line.strip().split(",", 1)[0]

        try:
            values.append(float(value))
        except ValueError:
            continue

    if not values:
        raise RuntimeError(
            "ffprobe could not read frame timestamps."
        )

    first = values[0]

    return [
        value - first
        for value in values
    ]


def extract_preview_frames(
    clip_path: Path,
    clip_start: float,
    segment: WalkingSegment,
    temp_dir: Path,
) -> list[PreviewFrame]:
    preview_dir = (
        temp_dir / "preview_frames"
    )
    preview_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame_times = probe_frame_times(
        clip_path
    )

    scale_filter = (
        "scale="
        f"'min({PREVIEW_MAX_WIDTH},iw)':"
        f"'min({PREVIEW_MAX_HEIGHT},ih)':"
        "force_original_aspect_ratio=decrease"
    )

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(clip_path),
        "-vf",
        scale_filter,
        "-vsync",
        "0",
        "-y",
        str(
            preview_dir / "frame_%06d.png"
        ),
    ]

    print(
        "Decoding nearby frames for the browser..."
    )

    subprocess.run(
        command,
        check=True,
    )

    paths = sorted(
        preview_dir.glob("frame_*.png")
    )

    usable = min(
        len(paths),
        len(frame_times),
    )

    if usable == 0:
        raise RuntimeError(
            "No preview frames were decoded."
        )

    all_frames: list[PreviewFrame] = []

    for index in range(usable):
        all_frames.append(
            PreviewFrame(
                path=paths[index],
                source_time=(
                    clip_start
                    + frame_times[index]
                ),
                source_frame_index=index,
            )
        )

    lower = (
        segment.midpoint
        - BROWSE_RADIUS_SECONDS
    )
    upper = (
        segment.midpoint
        + BROWSE_RADIUS_SECONDS
    )

    nearby = [
        frame
        for frame in all_frames
        if lower
        <= frame.source_time
        <= upper
    ]

    if not nearby:
        closest = min(
            range(len(all_frames)),
            key=lambda i: abs(
                all_frames[i].source_time
                - segment.midpoint
            ),
        )

        start = max(
            0,
            closest - 30,
        )
        end = min(
            len(all_frames),
            closest + 31,
        )

        nearby = all_frames[start:end]

    return nearby


class FrameBrowser:
    def __init__(
        self,
        frames: list[PreviewFrame],
        segment: WalkingSegment,
        frame_step: int,
    ) -> None:
        try:
            import tkinter as tk
            from tkinter import ttk
        except ImportError as exc:
            raise RuntimeError(
                "Python tkinter is required for the "
                "interactive frame browser."
            ) from exc

        self.tk = tk
        self.ttk = ttk
        self.frames = frames
        self.segment = segment
        self.frame_step = max(
            1,
            frame_step,
        )

        self.index = min(
            range(len(frames)),
            key=lambda i: abs(
                frames[i].source_time
                - segment.midpoint
            ),
        )

        self.result: tuple[
            str,
            PreviewFrame | None,
        ] = ("cancel", None)

        self.root = tk.Tk()
        self.root.title(
            "Walking frame selector"
        )
        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.cancel,
        )

        self.image_label = ttk.Label(
            self.root
        )
        self.image_label.pack(
            padx=12,
            pady=(12, 6),
        )

        self.timestamp_label = ttk.Label(
            self.root,
            anchor="center",
            font=(
                "TkDefaultFont",
                13,
                "bold",
            ),
        )
        self.timestamp_label.pack(
            fill="x",
            padx=12,
        )

        self.details_label = ttk.Label(
            self.root,
            anchor="center",
            justify="center",
        )
        self.details_label.pack(
            fill="x",
            padx=12,
            pady=(3, 9),
        )

        controls = ttk.Frame(
            self.root
        )
        controls.pack(
            padx=12,
            pady=(0, 12),
            fill="x",
        )

        self.previous_button = ttk.Button(
            controls,
            text=f"◀ Previous {self.frame_step}",
            command=self.previous,
        )
        self.previous_button.pack(
            side="left",
            padx=4,
        )

        self.single_previous_button = ttk.Button(
            controls,
            text="◁ 1 frame",
            command=self.previous_one,
        )
        self.single_previous_button.pack(
            side="left",
            padx=4,
        )

        self.midpoint_button = ttk.Button(
            controls,
            text="Midpoint",
            command=self.jump_to_midpoint,
        )
        self.midpoint_button.pack(
            side="left",
            padx=4,
        )

        self.single_next_button = ttk.Button(
            controls,
            text="1 frame ▷",
            command=self.next_one,
        )
        self.single_next_button.pack(
            side="left",
            padx=4,
        )

        self.next_button = ttk.Button(
            controls,
            text=f"Next {self.frame_step} ▶",
            command=self.next,
        )
        self.next_button.pack(
            side="left",
            padx=4,
        )

        spacer = ttk.Frame(
            controls
        )
        spacer.pack(
            side="left",
            expand=True,
        )

        self.random_button = ttk.Button(
            controls,
            text="Different segment",
            command=self.reroll,
        )
        self.random_button.pack(
            side="left",
            padx=4,
        )

        self.cancel_button = ttk.Button(
            controls,
            text="Cancel",
            command=self.cancel,
        )
        self.cancel_button.pack(
            side="left",
            padx=4,
        )

        self.save_button = ttk.Button(
            controls,
            text="Save this frame",
            command=self.save,
        )
        self.save_button.pack(
            side="left",
            padx=4,
        )

        self.root.bind(
            "<Left>",
            lambda _event: self.previous(),
        )
        self.root.bind(
            "<Right>",
            lambda _event: self.next(),
        )
        self.root.bind(
            "<Shift-Left>",
            lambda _event: self.previous_one(),
        )
        self.root.bind(
            "<Shift-Right>",
            lambda _event: self.next_one(),
        )
        self.root.bind(
            "<Home>",
            lambda _event: self.jump_to_midpoint(),
        )
        self.root.bind(
            "<Return>",
            lambda _event: self.save(),
        )
        self.root.bind(
            "<Escape>",
            lambda _event: self.cancel(),
        )

        self.photo = None
        self.refresh()

    def refresh(self) -> None:
        frame = self.frames[self.index]

        self.photo = self.tk.PhotoImage(
            file=str(frame.path)
        )

        self.image_label.configure(
            image=self.photo
        )

        offset = (
            frame.source_time
            - self.segment.midpoint
        )

        self.timestamp_label.configure(
            text=(
                f"Frame {self.index + 1} / {len(self.frames)}"
                "    |    "
                f"Source time {frame.source_time:.3f} s"
                "    |    "
                f"{offset:+.3f} s from segment midpoint"
            )
        )

        location = ", ".join(
            value
            for value in (
                self.segment.locality,
                self.segment.state,
                self.segment.country,
            )
            if value
        )

        self.details_label.configure(
            text=(
                f"{location or 'Unknown location'}"
                "    |    "
                f"{self.segment.walking_environment or 'walking'}"
                "\n"
                f"Left/Right = {self.frame_step} frames, "
                "Shift+Left/Right = 1 frame. "
                "The displayed image is the frame that will be saved."
            )
        )

        self.previous_button.configure(
            state=(
                "normal"
                if self.index > 0
                else "disabled"
            )
        )

        self.single_previous_button.configure(
            state=(
                "normal"
                if self.index > 0
                else "disabled"
            )
        )

        self.next_button.configure(
            state=(
                "normal"
                if self.index
                < len(self.frames) - 1
                else "disabled"
            )
        )

        self.single_next_button.configure(
            state=(
                "normal"
                if self.index
                < len(self.frames) - 1
                else "disabled"
            )
        )

    def previous(self) -> None:
        self.index = max(
            0,
            self.index - self.frame_step,
        )
        self.refresh()

    def next(self) -> None:
        self.index = min(
            len(self.frames) - 1,
            self.index + self.frame_step,
        )
        self.refresh()

    def previous_one(self) -> None:
        self.index = max(
            0,
            self.index - 1,
        )
        self.refresh()

    def next_one(self) -> None:
        self.index = min(
            len(self.frames) - 1,
            self.index + 1,
        )
        self.refresh()

    def jump_to_midpoint(self) -> None:
        self.index = min(
            range(len(self.frames)),
            key=lambda i: abs(
                self.frames[i].source_time
                - self.segment.midpoint
            ),
        )
        self.refresh()

    def save(self) -> None:
        self.result = (
            "save",
            self.frames[self.index],
        )
        self.root.destroy()

    def reroll(self) -> None:
        self.result = (
            "reroll",
            None,
        )
        self.root.destroy()

    def cancel(self) -> None:
        self.result = (
            "cancel",
            None,
        )
        self.root.destroy()

    def run(
        self,
    ) -> tuple[str, PreviewFrame | None]:
        self.root.mainloop()
        return self.result


def output_path_for(
    segment: WalkingSegment,
    frame: PreviewFrame,
    output_dir: Path,
) -> Path:
    return output_dir / (
        f"{segment.video_id}_"
        f"{frame.source_time:.3f}.png"
    )


def extract_selected_full_resolution_frame(
    clip_path: Path,
    selected: PreviewFrame,
    destination: Path,
) -> None:
    """
    Extract by decoded frame index so the saved full resolution frame is the
    same frame that was displayed in the browser.
    """
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    select_filter = (
        "select="
        f"'eq(n\\,{selected.source_frame_index})'"
    )

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(clip_path),
        "-vf",
        select_filter,
        "-vsync",
        "0",
        "-frames:v",
        "1",
        "-y",
        str(destination),
    ]

    subprocess.run(
        command,
        check=True,
    )

    if (
        not destination.exists()
        or destination.stat().st_size == 0
    ):
        raise RuntimeError(
            "ffmpeg did not create the selected full resolution frame."
        )


def browse_segment(
    segment: WalkingSegment,
    args: argparse.Namespace,
) -> str:
    with tempfile.TemporaryDirectory(
        prefix="walking_frame_browser_"
    ) as temp_name:
        temp_dir = Path(temp_name)

        try:
            clip_path, clip_start = (
                download_preview_clip(
                    segment,
                    temp_dir,
                    args,
                )
            )
        except PreviewDownloadError as exc:
            print()
            print(f"DOWNLOAD FAILED: {exc}")
            return "download_failed"

        frames = extract_preview_frames(
            clip_path,
            clip_start,
            segment,
            temp_dir,
        )

        print(
            f"Opening {len(frames):,} nearby decoded frames "
            "in the selector..."
        )

        browser = FrameBrowser(
            frames,
            segment,
            args.frame_step,
        )

        action, selected = browser.run()

        if action == "reroll":
            return "reroll"

        if (
            action != "save"
            or selected is None
        ):
            print(
                "Nothing saved. Temporary preview files were removed."
            )
            return "cancelled"

        destination = output_path_for(
            segment,
            selected,
            args.output_dir,
        )

        extract_selected_full_resolution_frame(
            clip_path,
            selected,
            destination,
        )

        print()
        print("=" * 72)
        print("SAVED ONE WALKING FRAME")
        print("=" * 72)
        print(
            f"File:             "
            f"{destination.resolve()}"
        )
        print(
            f"Video ID:         "
            f"{segment.video_id}"
        )
        print(
            f"Source time:      "
            f"{selected.source_time:.3f} s"
        )
        print(
            f"Locality:         "
            f"{segment.locality or 'N/A'}"
        )
        print(
            f"Country:          "
            f"{segment.country or 'N/A'}"
        )
        print(
            f"Environment:      "
            f"{segment.walking_environment or 'N/A'}"
        )
        print("=" * 72)

        return "saved"


def main() -> int:
    args = parse_args()

    if args.frame_step < 1:
        raise ValueError(
            "--frame-step must be at least 1."
        )

    require_program("yt-dlp")
    require_program("ffmpeg")
    require_program("ffprobe")

    if args.reset_history:
        if args.history_file.exists():
            args.history_file.unlink()

        print(
            f"Walking frame history reset: "
            f"{args.history_file}"
        )

    print(
        f"Reading walking mapping file: "
        f"{args.mapping}"
    )

    mapping_rows = read_mapping_rows(
        args.mapping
    )

    segments = build_segment_pool(
        mapping_rows
    )

    if not segments:
        raise RuntimeError(
            "No valid walking segments were found in mapping.csv."
        )

    history = load_history(
        args.history_file
    )

    unique_videos = {
        segment.video_id
        for segment in segments
    }

    unseen = [
        segment
        for segment in segments
        if segment.history_key not in history
    ]

    print()
    print("=== Walking candidate pool ===")
    print(
        f"Mapping rows:              "
        f"{len(mapping_rows):,}"
    )
    print(
        f"Walking segments:          "
        f"{len(segments):,}"
    )
    print(
        f"Source videos:             "
        f"{len(unique_videos):,}"
    )
    print(
        f"Previously shown segments: "
        f"{len(history):,}"
    )
    print(
        f"Unseen walking segments:   "
        f"{len(unseen):,}"
    )
    print()

    failed_video_ids: set[str] = set()

    while True:
        segment = choose_unseen_segment(
            segments,
            history,
            failed_video_ids,
        )

        if segment is None:
            remaining_unseen = [
                item
                for item in segments
                if item.history_key not in history
            ]

            if (
                remaining_unseen
                and failed_video_ids
            ):
                print(
                    "No selectable source videos remain in this run because "
                    f"{len(failed_video_ids):,} source video(s) failed "
                    "YouTube preview download."
                )
                print(
                    "Your normal walking history was not expanded by those "
                    "failed downloads."
                )
            else:
                print(
                    "No unseen walking segments remain. "
                    "History was NOT reset."
                )
                print(
                    "Run with --reset-history only if you want to "
                    "allow previously shown segments again."
                )

            return 0

        print_segment(
            segment
        )

        action = ask_segment_action()

        if action == "reroll":
            # The segment was genuinely shown to the user, so remember it.
            history.add(
                segment.history_key
            )
            append_history(
                args.history_file,
                segment.history_key,
            )
            continue

        if action == "no":
            history.add(
                segment.history_key
            )
            append_history(
                args.history_file,
                segment.history_key,
            )
            print(
                "Nothing downloaded or saved."
            )
            return 0

        result = browse_segment(
            segment,
            args,
        )

        if result == "download_failed":
            # A 403 or similar source failure is usually video-wide. Skip the
            # entire source video for this invocation, but do NOT add the
            # segment to persistent history because no frame was shown.
            failed_video_ids.add(
                segment.video_id
            )
            print(
                "Selecting another walking segment from a different "
                "source video..."
            )
            continue

        # The browser successfully opened, so this segment has now genuinely
        # been shown and should not be proposed again on a later run.
        history.add(
            segment.history_key
        )
        append_history(
            args.history_file,
            segment.history_key,
        )

        if result == "reroll":
            continue

        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(
            "\nCancelled. Nothing else will be saved."
        )
        raise SystemExit(130)
    except Exception as exc:
        print(
            f"\nERROR: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
