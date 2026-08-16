"""Shared state, JSON, process, and normalisation helpers."""

from __future__ import annotations

import gc
import json
import math
import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch  # type: ignore

from . import settings


def log(message: str) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}", flush=True)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Could not read valid JSON from {path}: {exc}") from exc


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def run_command(
    command: Sequence[str], timeout: Optional[int] = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )


def require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required command is not available: {name}")


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def optional_text(value: Any) -> Optional[str]:
    text = clean_text(value)
    if not text or text.lower() in {"none", "null", "unknown", "n/a"}:
        return None
    return text


def clamp_float(value: Any, lower: float, upper: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return lower
    if not math.isfinite(number):
        return lower
    return min(upper, max(lower, number))


def normalise_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return False


def normalise_string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, str):
        raw_items = re.split(r"[;,]", value)
    else:
        raw_items = []

    result: List[str] = []
    seen: set[str] = set()
    for item in raw_items:
        text = optional_text(item)
        if text and text.casefold() not in seen:
            result.append(text)
            seen.add(text.casefold())
    return result


def recover_json(text: str) -> Optional[Dict[str, Any]]:
    candidate = text.strip()
    try:
        data = json.loads(candidate)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, TypeError):
        pass

    first = candidate.find("{")
    last = candidate.rfind("}")
    if first >= 0 and last > first:
        try:
            data = json.loads(candidate[first: last + 1])
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, TypeError):
            pass
    return None


def empty_state() -> Dict[str, Any]:
    now = datetime.now().isoformat(timespec="seconds")
    return {
        "schema_version": 1,
        "created_at": now,
        "updated_at": now,
        "videos": {},
        "locality_ids": {},
    }


def save_state(state: Dict[str, Any]) -> None:
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    write_json_atomic(settings.STATE_JSON, state)


def unload_model(owner: Any) -> None:
    for attribute in ("model", "processor", "tokenizer"):
        if hasattr(owner, attribute):
            try:
                delattr(owner, attribute)
            except Exception:
                pass
    del owner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
