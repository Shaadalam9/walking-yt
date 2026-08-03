"""Top level orchestration for the walking video pipeline."""

from __future__ import annotations

from typing import Any, Dict

from . import settings
from .location import run_location_stage
from .metadata_filter import run_text_stage
from .output_writer import write_output_csv
from .overlapped_pipeline import run_overlapped_stages
from .runtime import ExecutionPlan, validate_runtime
from .shared import empty_state, load_json, log, require_binary, save_state
from .video_filter import run_visual_stage
from .youtube_discovery import YouTubeDiscovery, load_api_keys


FINAL_VIDEO_STATUSES = {
    "complete",
    "text_rejected",
    "visual_rejected",
}


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
    return record.get("status") not in FINAL_VIDEO_STATUSES


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
        state, after_video=write_output_csv
    )
    return text_processed, visual_processed


def main() -> None:
    execution_plan = validate_environment()
    state = load_state()

    discovered = discover_if_batch_complete(state)
    log(f"New YouTube candidates discovered: {discovered}")

    text_processed, visual_processed = _run_model_stages(
        state, execution_plan
    )
    log(f"Candidates processed by the text LLM: {text_processed}")
    log(f"Videos processed visually this run: {visual_processed}")

    locations_processed = run_location_stage(state)
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
            f"{unfinished}. The next run will continue this batch without "
            "calling the YouTube API."
        )
    else:
        log(
            "The current discovery batch is complete. The next run may "
            "discover a new batch."
        )
    log(f"Detailed state: {settings.STATE_JSON}")
    log(f"Locality CSV: {settings.OUTPUT_CSV}")
