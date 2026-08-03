"""Two GPU producer consumer execution for text and visual filtering."""

from __future__ import annotations

import copy
import multiprocessing as mp
import os
import queue
import time
import traceback
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from . import settings
from .runtime import cuda_index, resolve_execution_plan
from .shared import log, save_state


TaskQueue = Any
ResultQueue = Any


def overlap_is_available() -> bool:
    """Return whether the resolved runtime plan uses two GPU overlap."""
    return resolve_execution_plan().mode == "overlap"


def _configure_cuda_device(device: str) -> None:
    index = cuda_index(device)
    if index is None:
        return
    import torch

    torch.cuda.set_device(index)


def _text_worker(tasks: TaskQueue, results: ResultQueue) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    judge: Any = None
    try:
        _configure_cuda_device(settings.TEXT_DEVICE)
        from .metadata_filter import TextMetadataJudge

        judge = TextMetadataJudge(
            settings.TEXT_MODEL_NAME,
            device=settings.TEXT_DEVICE,
        )
        results.put({"kind": "ready", "worker": "text"})

        while True:
            task = tasks.get()
            if task is None:
                break
            video_id, metadata = task
            try:
                decision = judge.judge(metadata)
                results.put(
                    {
                        "kind": "text_result",
                        "video_id": video_id,
                        "decision": decision,
                        "error": None,
                    }
                )
            except BaseException as exc:
                results.put(
                    {
                        "kind": "text_result",
                        "video_id": video_id,
                        "decision": None,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
    except BaseException as exc:
        results.put(
            {
                "kind": "startup_error",
                "worker": "text",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        if judge is not None:
            from .shared import unload_model

            unload_model(judge)


def _visual_worker(tasks: TaskQueue, results: ResultQueue) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    judge: Any = None
    try:
        _configure_cuda_device(settings.VLM_DEVICE)
        from .video_filter import InternVLVisualJudge, process_visual_candidate

        judge = InternVLVisualJudge(
            settings.VLM_MODEL_NAME,
            device=settings.VLM_DEVICE,
        )
        results.put({"kind": "ready", "worker": "visual"})

        while True:
            task = tasks.get()
            if task is None:
                break
            video_id, record = task
            try:
                process_visual_candidate(video_id, record, judge)
                results.put(
                    {
                        "kind": "visual_result",
                        "video_id": video_id,
                        "record": record,
                        "error": None,
                    }
                )
            except BaseException as exc:
                record["status"] = "pipeline_error"
                record["error"] = str(exc)
                results.put(
                    {
                        "kind": "visual_result",
                        "video_id": video_id,
                        "record": record,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
    except BaseException as exc:
        results.put(
            {
                "kind": "startup_error",
                "worker": "visual",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        if judge is not None:
            from .shared import unload_model

            unload_model(judge)


def _pending_text_records(
    state: Dict[str, Any],
) -> List[Tuple[str, Dict[str, Any]]]:
    return [
        (video_id, record)
        for video_id, record in state.get("videos", {}).items()
        if not isinstance(record.get("text_decision"), dict)
    ]


def _record_needs_visual_processing(record: Dict[str, Any]) -> bool:
    decision = record.get("text_decision")
    if not isinstance(decision, dict) or not decision.get("include"):
        return False
    return record.get("status") not in {"complete", "visual_rejected"}


def _pending_visual_ids(state: Dict[str, Any]) -> Deque[str]:
    return deque(
        video_id
        for video_id, record in state.get("videos", {}).items()
        if isinstance(record, dict) and _record_needs_visual_processing(record)
    )


def _apply_text_result(
    state: Dict[str, Any],
    message: Dict[str, Any],
    ready_visual: Deque[str],
    known_visual_ids: set[str],
) -> None:
    video_id = str(message["video_id"])
    record = state["videos"][video_id]
    decision = message.get("decision")
    error = message.get("error")

    if not isinstance(decision, dict):
        record["status"] = "text_error"
        record["error"] = error or "The text worker returned no decision"
        log(f"Text stage failed for {video_id}: {record['error']}")
        return

    record["text_decision"] = decision
    record["status"] = (
        "text_accepted" if decision.get("include") else "text_rejected"
    )
    record["error"] = decision.get("error")
    log(
        f"Text result {video_id}: include={bool(decision.get('include'))}, "
        f"confidence={float(decision.get('confidence') or 0.0):.2f}"
    )

    if decision.get("include") and video_id not in known_visual_ids:
        ready_visual.append(video_id)
        known_visual_ids.add(video_id)


def _apply_visual_result(
    state: Dict[str, Any],
    message: Dict[str, Any],
    after_video: Optional[Callable[[Dict[str, Any]], None]],
) -> None:
    video_id = str(message["video_id"])
    returned_record = message.get("record")
    if isinstance(returned_record, dict):
        state["videos"][video_id] = returned_record
    else:
        state["videos"][video_id]["status"] = "pipeline_error"
        state["videos"][video_id]["error"] = (
            message.get("error") or "The visual worker returned no record"
        )

    record = state["videos"][video_id]
    log(f"Visual result {video_id}: status={record.get('status')}")
    if after_video:
        after_video(state)


def _raise_worker_startup_error(message: Dict[str, Any]) -> None:
    worker = message.get("worker") or "unknown"
    error = message.get("error") or "unknown error"
    details = message.get("traceback") or ""
    raise RuntimeError(
        f"The {worker} worker could not start: {error}\n{details}"
    )


def _terminate_processes(processes: List[mp.Process]) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=10)


def run_overlapped_stages(
    state: Dict[str, Any],
    after_video: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[int, int]:
    """Run metadata and visual filtering concurrently on two GPUs.

    The text worker continuously produces accepted candidates. The visual
    worker consumes them with a bounded outstanding count. With the default
    value of two visual jobs, one video may run while the next waits ready.
    """
    pending_text = _pending_text_records(state)
    ready_visual = _pending_visual_ids(state)
    known_visual_ids = set(ready_visual)

    visual_budget = settings.MAX_VIDEOS_PER_RUN
    if visual_budget is not None:
        visual_budget = max(0, visual_budget)

    if not pending_text and not ready_visual:
        return 0, 0

    context = mp.get_context("spawn")
    text_tasks = context.Queue(maxsize=settings.TEXT_MAX_OUTSTANDING)
    text_results = context.Queue()
    visual_tasks = context.Queue(maxsize=settings.VISUAL_MAX_OUTSTANDING)
    visual_results = context.Queue()

    text_process = context.Process(
        target=_text_worker,
        args=(text_tasks, text_results),
        name="walking-text-worker",
    )
    visual_process = context.Process(
        target=_visual_worker,
        args=(visual_tasks, visual_results),
        name="walking-visual-worker",
    )
    processes = [text_process, visual_process]

    text_process.start()
    visual_process.start()
    log(
        "Started overlapped pipeline: "
        f"text={settings.TEXT_DEVICE}, visual={settings.VLM_DEVICE}, "
        f"visual outstanding limit={settings.VISUAL_MAX_OUTSTANDING}"
    )

    text_sent = 0
    text_finished = 0
    text_processed = 0
    visual_sent = 0
    visual_finished = 0
    visual_processed = 0
    text_worker_ready = False
    visual_worker_ready = False
    started_at = time.monotonic()

    try:
        while True:
            made_progress = False

            while (
                text_sent < len(pending_text)
                and text_sent - text_finished < settings.TEXT_MAX_OUTSTANDING
            ):
                video_id, record = pending_text[text_sent]
                metadata = copy.deepcopy(record.get("metadata", {}))
                try:
                    text_tasks.put_nowait((video_id, metadata))
                except queue.Full:
                    break
                text_sent += 1
                made_progress = True

            visual_limit_reached = (
                visual_budget is not None and visual_sent >= visual_budget
            )
            while (
                ready_visual
                and not visual_limit_reached
                and visual_sent - visual_finished
                < settings.VISUAL_MAX_OUTSTANDING
            ):
                video_id = ready_visual.popleft()
                record = copy.deepcopy(state["videos"][video_id])
                try:
                    visual_tasks.put_nowait((video_id, record))
                except queue.Full:
                    ready_visual.appendleft(video_id)
                    break
                visual_sent += 1
                made_progress = True
                visual_limit_reached = (
                    visual_budget is not None and visual_sent >= visual_budget
                )

            while True:
                try:
                    message = text_results.get_nowait()
                except queue.Empty:
                    break
                made_progress = True
                kind = message.get("kind")
                if kind == "ready":
                    text_worker_ready = True
                    log("Text worker is ready")
                elif kind == "startup_error":
                    _raise_worker_startup_error(message)
                elif kind == "text_result":
                    text_finished += 1
                    text_processed += 1
                    _apply_text_result(
                        state,
                        message,
                        ready_visual,
                        known_visual_ids,
                    )
                    save_state(state)

            while True:
                try:
                    message = visual_results.get_nowait()
                except queue.Empty:
                    break
                made_progress = True
                kind = message.get("kind")
                if kind == "ready":
                    visual_worker_ready = True
                    log("Visual worker is ready")
                elif kind == "startup_error":
                    _raise_worker_startup_error(message)
                elif kind == "visual_result":
                    visual_finished += 1
                    visual_processed += 1
                    _apply_visual_result(state, message, after_video)
                    save_state(state)

            text_complete = text_finished == len(pending_text)
            visual_complete = visual_finished == visual_sent
            no_more_visual_dispatch = (
                visual_budget is not None and visual_sent >= visual_budget
            ) or (text_complete and not ready_visual)

            if text_complete and visual_complete and no_more_visual_dispatch:
                break

            elapsed = time.monotonic() - started_at
            if elapsed > settings.WORKER_START_TIMEOUT_SECONDS:
                if not text_worker_ready:
                    raise TimeoutError("The text worker did not become ready")
                if not visual_worker_ready:
                    raise TimeoutError("The visual worker did not become ready")

            if not text_process.is_alive() and text_finished < text_sent:
                raise RuntimeError(
                    "The text worker exited before returning all results"
                )
            if not visual_process.is_alive() and visual_finished < visual_sent:
                raise RuntimeError(
                    "The visual worker exited before returning all results"
                )

            if not made_progress:
                time.sleep(settings.WORKER_POLL_SECONDS)

        text_tasks.put(None)
        visual_tasks.put(None)
        text_process.join(timeout=60)
        visual_process.join(timeout=60)
        if text_process.is_alive() or visual_process.is_alive():
            _terminate_processes(processes)
    except KeyboardInterrupt:
        save_state(state)
        _terminate_processes(processes)
        raise
    except BaseException:
        save_state(state)
        _terminate_processes(processes)
        raise
    finally:
        for task_queue in (text_tasks, visual_tasks):
            task_queue.close()
        for result_queue in (text_results, visual_results):
            result_queue.close()

    return text_processed, visual_processed
