"""Qwen based title and description filtering."""

from __future__ import annotations

from typing import Any, Dict

import torch

from . import settings
from .model_loading import load_model_with_fallback
from .shared import (
    clamp_float,
    clean_text,
    log,
    normalise_bool,
    optional_text,
    recover_json,
    save_state,
    unload_model,
)


class TextMetadataJudge:
    def __init__(self, model_name: str, device: str | None = None) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        selected_device = device or settings.SEQUENTIAL_DEVICE
        log(f"Loading text LLM: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = load_model_with_fallback(
            AutoModelForCausalLM.from_pretrained,
            model_name,
            device=selected_device,
            load_in_4bit=settings.TEXT_LOAD_IN_4BIT,
            model_label="text LLM",
        ).eval()
        log("Text LLM loaded")

    def judge(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        messages = [
            {"role": "user", "content": self._build_prompt(metadata)}
        ]
        model_inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(
                **model_inputs,
                max_new_tokens=settings.TEXT_MAX_NEW_TOKENS,
                do_sample=False,
            )
        prompt_length = model_inputs["input_ids"].shape[1]
        answer = self.tokenizer.decode(
            generated[0, prompt_length:], skip_special_tokens=True
        )
        data = recover_json(answer)
        if data is None:
            return self._invalid_response(answer)

        include = normalise_bool(data.get("include"))
        confidence = clamp_float(data.get("confidence"), 0.0, 1.0)
        include = include and confidence >= settings.MIN_TEXT_CONFIDENCE
        return {
            "include": include,
            "confidence": confidence,
            "short_reason": clean_text(data.get("short_reason")),
            "locality": optional_text(data.get("locality")),
            "state": optional_text(data.get("state")),
            "country": optional_text(data.get("country")),
            "raw_response": answer,
            "error": None,
        }

    @staticmethod
    def _invalid_response(answer: str) -> Dict[str, Any]:
        return {
            "include": False,
            "confidence": 0.0,
            "short_reason": "The text LLM did not return valid JSON.",
            "locality": None,
            "state": None,
            "country": None,
            "raw_response": answer,
            "error": "json_recovery_failed",
        }

    @staticmethod
    def _build_prompt(metadata: Dict[str, Any]) -> str:
        title = clean_text(metadata.get("title"))
        description = clean_text(metadata.get("description"))[:4000]
        return f"""
You are the first stage of a research pipeline collecting pedestrian walking
videos through real physical locations.

Decide from the title and description whether the video is worth downloading.
Include likely pedestrian walking tours through cities, towns, villages,
streets, neighbourhoods, markets, parks, beaches, malls, campuses, stations,
trails, or other real places.

Reject workouts, treadmills, The Walking Dead, games, animation, music videos,
product reviews, driving tours, cycling tours, bus or train rides, drone
videos, flights, boat rides, talking head videos, slideshows, maps, static
ambience, and unrelated uses of the word walking.

Extract only the most specific locality, state, and country supported by the
text. Do not infer an ISO code or continent. Do not invent a location. A
candidate may still be included when it appears to be a valid walking video
but no location can be found. Return null for unknown location fields.

Title: {title}
Description: {description}

Return valid JSON only:
{{
  "include": true,
  "confidence": 0.0,
  "short_reason": "one short sentence",
  "locality": null,
  "state": null,
  "country": null
}}
""".strip()


def run_text_stage(state: Dict[str, Any]) -> int:
    pending = [
        (video_id, record)
        for video_id, record in state.get("videos", {}).items()
        if not isinstance(record.get("text_decision"), dict)
    ]
    if not pending:
        return 0

    judge = TextMetadataJudge(settings.TEXT_MODEL_NAME)
    processed = 0
    try:
        for video_id, record in pending:
            metadata = record.get("metadata", {})
            log(f"Text LLM judging {video_id}: {metadata.get('title') or ''}")
            try:
                decision = judge.judge(metadata)
                record["text_decision"] = decision
                record["status"] = (
                    "text_accepted"
                    if decision.get("include")
                    else "text_rejected"
                )
                record["error"] = decision.get("error")
            except KeyboardInterrupt:
                save_state(state)
                raise
            except Exception as exc:
                record["status"] = "text_error"
                record["error"] = str(exc)
                log(f"Text stage failed for {video_id}: {exc}")
            save_state(state)
            processed += 1
    finally:
        unload_model(judge)
    return processed
