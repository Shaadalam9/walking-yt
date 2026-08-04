"""YouTube Data API discovery with API key rotation."""

from __future__ import annotations

import os
import re
from typing import Any, Callable, Dict, Iterable, List, Optional

from . import settings
from .shared import clean_text, log, optional_text, save_state


def load_api_keys() -> List[str]:
    raw = (
        os.environ.get("YOUTUBE_API_KEYS")
        or os.environ.get("YOUTUBE_API_KEY")
        or ""
    )
    if not raw:
        try:
            import common  # type: ignore

            raw = (
                common.get_secrets("google-api-keys")
                or common.get_secrets("google-api-key")
                or ""
            )
        except Exception:
            raw = ""

    if isinstance(raw, str):
        candidates: Iterable[Any] = re.split(r"[;,\n]+", raw)
    elif isinstance(raw, (list, tuple, set)):
        candidates = raw
    else:
        candidates = [raw]

    result: List[str] = []
    for item in candidates:
        key = clean_text(item)
        if key and key not in result:
            result.append(key)
    return result


def split_queries(value: Optional[str]) -> List[str]:
    if not value:
        return list(settings.DEFAULT_WALKING_QUERIES)

    queries: List[str] = []
    for item in re.split(r"[\n;,]+", value):
        query = clean_text(item)
        if query and query not in queries:
            queries.append(query)
    return queries or list(settings.DEFAULT_WALKING_QUERIES)


def normalise_published_bound(value: Optional[str]) -> Optional[str]:
    text = optional_text(value)
    if not text or text.upper() == "YYYY-MM-DD":
        return None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return f"{text}T00:00:00Z"
    return text


def duration_to_seconds(iso_duration: str) -> int:
    match = re.fullmatch(
        r"P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?",
        iso_duration or "",
    )
    if not match:
        return 0
    days = int(match.group(1) or 0)
    hours = int(match.group(2) or 0)
    minutes = int(match.group(3) or 0)
    seconds = int(match.group(4) or 0)
    return (((days * 24) + hours) * 60 + minutes) * 60 + seconds


def video_duration_is_eligible(duration: int) -> bool:
    """Return whether a video meets the configured minimum duration."""
    return duration >= settings.MIN_VIDEO_DURATION_SECONDS


class YouTubeDiscovery:
    def __init__(self, api_keys: List[str]) -> None:
        if not api_keys:
            raise RuntimeError(
                "No YouTube API key found. Set YOUTUBE_API_KEYS or "
                "YOUTUBE_API_KEY."
            )
        self.api_keys = api_keys
        self.key_index = 0
        self.client = self._build_client()

    def _build_client(self) -> Any:
        from googleapiclient.discovery import build  # type: ignore

        return build(
            "youtube",
            "v3",
            developerKey=self.api_keys[self.key_index],
            cache_discovery=False,
        )

    def _rotate_key(self) -> bool:
        self.key_index += 1
        if self.key_index >= len(self.api_keys):
            return False
        self.client = self._build_client()
        log(
            "Rotated to YouTube API key "
            f"{self.key_index + 1}/{len(self.api_keys)}"
        )
        return True

    def execute(self, request_factory: Callable[[Any], Any]) -> Dict[str, Any]:
        while True:
            try:
                response = request_factory(self.client).execute()
                return response if isinstance(response, dict) else {}
            except Exception as exc:
                message = str(exc)
                quota_error = any(
                    token in message
                    for token in (
                        "quotaExceeded",
                        "dailyLimitExceeded",
                        "rateLimitExceeded",
                    )
                )
                if quota_error and self._rotate_key():
                    continue
                raise

    def discover(self, state: Dict[str, Any]) -> int:
        queries = split_queries(os.environ.get("WALKING_QUERIES"))
        published_after = normalise_published_bound(settings.PUBLISHED_AFTER)
        published_before = normalise_published_bound(settings.PUBLISHED_BEFORE)
        videos: Dict[str, Any] = state.setdefault("videos", {})
        known_ids = set(videos)
        discovered_count = 0

        for query in queries:
            page_token: Optional[str] = None
            log(f"YouTube API search: {query}")

            for _ in range(settings.MAX_PAGES_PER_QUERY):
                if self._limit_reached(discovered_count):
                    return discovered_count

                def search_request(client: Any) -> Any:
                    parameters: Dict[str, Any] = {
                        "q": query,
                        "part": "snippet",
                        "type": "video",
                        "maxResults": settings.RESULTS_PER_PAGE,
                    }
                    if page_token:
                        parameters["pageToken"] = page_token
                    if published_after:
                        parameters["publishedAfter"] = published_after
                    if published_before:
                        parameters["publishedBefore"] = published_before
                    return client.search().list(**parameters)

                try:
                    search_response = self.execute(search_request)
                except Exception as exc:
                    log(f"YouTube search failed for query '{query}': {exc}")
                    break

                page_ids, search_snippets = self._new_page_ids(
                    search_response, known_ids
                )
                if page_ids:
                    discovered_count += self._save_video_details(
                        query=query,
                        page_ids=page_ids,
                        search_snippets=search_snippets,
                        known_ids=known_ids,
                        videos=videos,
                        remaining_limit=self._remaining_limit(
                            discovered_count
                        ),
                    )
                    save_state(state)

                page_token = optional_text(
                    search_response.get("nextPageToken")
                )
                if not page_token:
                    break

        return discovered_count

    @staticmethod
    def _new_page_ids(
        response: Dict[str, Any], known_ids: set[str]
    ) -> tuple[List[str], Dict[str, Dict[str, Any]]]:
        page_ids: List[str] = []
        snippets: Dict[str, Dict[str, Any]] = {}
        for item in response.get("items", []):
            video_id = (
                item.get("id", {}).get("videoId")
                if isinstance(item, dict)
                else None
            )
            if not video_id or video_id in known_ids:
                continue
            page_ids.append(str(video_id))
            snippet = item.get("snippet", {})
            snippets[str(video_id)] = (
                snippet if isinstance(snippet, dict) else {}
            )
        return page_ids, snippets

    def _save_video_details(
        self,
        query: str,
        page_ids: List[str],
        search_snippets: Dict[str, Dict[str, Any]],
        known_ids: set[str],
        videos: Dict[str, Any],
        remaining_limit: Optional[int],
    ) -> int:
        joined_ids = ",".join(page_ids)

        def details_request(client: Any) -> Any:
            return client.videos().list(
                part="snippet,contentDetails",
                id=joined_ids,
                maxResults=50,
            )

        try:
            response = self.execute(details_request)
        except Exception as exc:
            log(f"Could not fetch video details: {exc}")
            return 0

        saved = 0
        for item in response.get("items", []):
            if not isinstance(item, dict):
                continue
            video_id = optional_text(item.get("id"))
            if not video_id or video_id in known_ids:
                continue

            snippet = item.get("snippet", {})
            if not isinstance(snippet, dict):
                snippet = search_snippets.get(video_id, {})
            content = item.get("contentDetails", {})
            if not isinstance(content, dict):
                content = {}

            duration = duration_to_seconds(str(content.get("duration", "")))
            if not video_duration_is_eligible(duration):
                continue

            videos[video_id] = self._new_video_record(
                video_id, query, snippet, duration
            )
            known_ids.add(video_id)
            saved += 1
            if remaining_limit is not None and saved >= remaining_limit:
                break
        return saved

    @staticmethod
    def _new_video_record(
        video_id: str,
        query: str,
        snippet: Dict[str, Any],
        duration: int,
    ) -> Dict[str, Any]:
        return {
            "metadata": {
                "video_id": video_id,
                "title": optional_text(snippet.get("title")),
                "description": optional_text(snippet.get("description")),
                "channel": optional_text(snippet.get("channelId")),
                "channel_title": optional_text(snippet.get("channelTitle")),
                "upload_date": optional_text(snippet.get("publishedAt")),
                "duration_seconds": duration,
                "source_query": query,
                "video_url": (
                    f"https://www.youtube.com/watch?v={video_id}"
                ),
            },
            "status": "discovered",
            "text_decision": None,
            "visual_decision": None,
            "cuts": None,
            "segments": None,
            "error": None,
        }

    @staticmethod
    def _limit_reached(count: int) -> bool:
        return (
            settings.MAX_NEW_CANDIDATES is not None
            and count >= settings.MAX_NEW_CANDIDATES
        )

    @staticmethod
    def _remaining_limit(discovered_count: int) -> Optional[int]:
        if settings.MAX_NEW_CANDIDATES is None:
            return None
        return max(0, settings.MAX_NEW_CANDIDATES - discovered_count)
