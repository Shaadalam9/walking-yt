"""Top level orchestration for the walking video pipeline."""

from __future__ import annotations

import time
from typing import Any, Dict

from . import settings
from .location import run_location_stage
from .metadata_filter import run_text_stage
from .output_writer import write_output_csv
from .overlapped_pipeline import run_overlapped_stages
from .runtime import ExecutionPlan, validate_runtime
from .shared import empty_state, load_json, log, require_binary, save_state
from .video_filter import (
    reset_video_download_temp_directory,
    run_visual_stage,
)
from .youtube_discovery import YouTubeDiscovery, load_api_keys


FINAL_VIDEO_STATUSES = {
    "complete",
    "text_rejected",
    "visual_rejected",
}

PIPELINE_SCHEMA_VERSION = "walking_incremental_location_csv_v2"


def validate_environment() -> ExecutionPlan:
    """Validate command line tools, storage, and visible compute resources."""
    require_binary("ffmpeg")
    require_binary("ffprobe")
    require_binary("yt-dlp")
    return validate_runtime()


def load_state() -> Dict[str, Any]:
    state = load_json(settings.STATE_JSON, empty_state())
    if not isinstance(state, dict):
        raise RuntimeError("The pipeline state must be a JSON object.")
    state.setdefault("videos", {})
    state.setdefault("locality_ids", {})
    return state


def record_requires_processing(record: Any) -> bool:
    """Return whether a discovered video still needs pipeline work."""
    if not isinstance(record, dict):
        return True

    text_decision = record.get("text_decision")
    if not isinstance(text_decision, dict):
        return True
    if not text_decision.get("include"):
        return False
    if record.get("status") not in FINAL_VIDEO_STATUSES:
        return True
    return (
        record.get("visual_review_version")
        != settings.VISUAL_REVIEW_VERSION
    )


def count_unfinished_videos(state: Dict[str, Any]) -> int:
    """Count videos that have not reached a final pipeline decision."""
    videos = state.get("videos", {})
    if not isinstance(videos, dict):
        return 0
    return sum(
        1 for record in videos.values() if record_requires_processing(record)
    )


def discover_if_batch_complete(state: Dict[str, Any]) -> int:
    """Discover a new batch only after the current batch is finished."""
    unfinished = count_unfinished_videos(state)
    if unfinished:
        log(
            "Skipping YouTube discovery until the existing batch is "
            f"complete. Unfinished videos: {unfinished}"
        )
        return 0

    discovery = YouTubeDiscovery(load_api_keys())
    return discovery.discover(state)


def _run_model_stages(
    state: Dict[str, Any],
    execution_plan: ExecutionPlan,
) -> tuple[int, int]:
    if execution_plan.mode == "overlap":
        log(
            "Using application stage overlap with one persistent model "
            "worker per GPU"
        )
        return run_overlapped_stages(state)

    log(
        "Using sequential model execution on "
        f"{execution_plan.sequential_device}"
    )
    text_processed = run_text_stage(state)
    visual_processed = run_visual_stage(
        state, after_video=_write_incremental_output
    )
    return text_processed, visual_processed


def _write_incremental_output(state: Dict[str, Any]) -> None:
    """Resolve completed video locations before refreshing the CSV."""
    locations_processed = run_location_stage(
        state, complete_only=True
    )
    if locations_processed:
        log(
            "Locations resolved before incremental CSV update: "
            f"{locations_processed}"
        )
    write_output_csv(state)


def _run_cycle(
    state: Dict[str, Any],
    execution_plan: ExecutionPlan,
    cycle_number: int,
) -> tuple[int, int, int, int]:
    log(f"Starting pipeline cycle {cycle_number}")
    discovered = discover_if_batch_complete(state)
    log(f"New YouTube candidates discovered: {discovered}")

    text_processed, visual_processed = _run_model_stages(
        state, execution_plan
    )
    log(f"Candidates processed by the text LLM: {text_processed}")
    log(f"Videos processed visually this run: {visual_processed}")

    locations_processed = run_location_stage(
        state, complete_only=True
    )
    log(f"Locations processed: {locations_processed}")

    write_output_csv(state)
    save_state(state)
    unfinished = count_unfinished_videos(state)
    complete = sum(
        1
        for record in state.get("videos", {}).values()
        if record.get("status") == "complete"
    )
    log(f"Accepted videos in final output: {complete}")
    if unfinished:
        log(
            "Existing discovery batch still has unfinished videos: "
            f"{unfinished}. The next cycle will continue this batch without "
            "calling the YouTube API."
        )
    else:
        log(
            "The current discovery batch is complete. The next cycle may "
            "discover a new batch."
        )
    log(f"Detailed state: {settings.STATE_JSON}")
    log(f"Locality CSV: {settings.OUTPUT_CSV}")
    return discovered, text_processed, visual_processed, unfinished


def _pause_before_next_cycle(seconds: int, reason: str) -> None:
    if seconds <= 0:
        return
    log(f"{reason} Waiting {seconds} second(s) before the next cycle.")
    time.sleep(seconds)


def main() -> None:
    execution_plan = validate_environment()
    reset_video_download_temp_directory()
    state = load_state()

    if settings.CONTINUOUS_BATCH_MODE:
        limit = settings.MAX_NEW_CANDIDATES
        limit_text = "unlimited" if limit is None else str(limit)
        log(
            "Continuous batch mode enabled: up to "
            f"{limit_text} new candidate(s) per discovery batch"
        )

    cycle_number = 0
    try:
        while True:
            cycle_number += 1
            (
                discovered,
                text_processed,
                visual_processed,
                unfinished,
            ) = _run_cycle(state, execution_plan, cycle_number)

            if not settings.CONTINUOUS_BATCH_MODE:
                return

            made_progress = any(
                (discovered, text_processed, visual_processed)
            )
            if not made_progress:
                _pause_before_next_cycle(
                    settings.CONTINUOUS_IDLE_PAUSE_SECONDS,
                    "The cycle made no progress.",
                )
                continue

            if unfinished:
                reason = (
                    "Continuing the existing discovery batch with "
                    f"{unfinished} unfinished video(s)."
                )
            else:
                reason = "Starting the next discovery batch."
            _pause_before_next_cycle(
                settings.CONTINUOUS_BATCH_PAUSE_SECONDS,
                reason,
            )
    except KeyboardInterrupt:
        write_output_csv(state)
        save_state(state)
        log(
            "Continuous processing stopped by the user. Current state "
            "and CSV output were saved."
        )
