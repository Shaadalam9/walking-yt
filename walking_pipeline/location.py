"""Locality resolution, geocoding, and country normalisation."""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from . import settings
from .shared import (
    clean_text,
    load_json,
    log,
    normalise_string_list,
    optional_text,
    save_state,
    write_json_atomic,
)


AFRICA_CODES = set(
    "AO BF BI BJ BW CD CF CG CI CM CV DJ DZ EG EH ER ET GA GH GM GN GQ "
    "GW KE KM LR LS LY MA MG ML MR MU MW MZ NA NE NG RE RW SC SD SH SL "
    "SN SO SS ST SZ TD TG TN TZ UG YT ZA ZM ZW".split()
)
ASIA_CODES = set(
    "AE AF AM AZ BD BH BN BT CC CN CX CY GE HK ID IL IN IO IQ IR JO JP "
    "KG KH KP KR KW KZ LA LB LK MM MN MO MV MY NP OM PH PK PS QA SA SG "
    "SY TH TJ TL TM TR TW UZ VN YE".split()
)
EUROPE_CODES = set(
    "AD AL AT AX BA BE BG BY CH CZ DE DK EE ES FI FO FR GB GG GI GR HR "
    "HU IE IM IS IT JE LI LT LU LV MC MD ME MK MT NL NO PL PT RO RS RU "
    "SE SI SJ SK SM UA VA XK".split()
)
NORTH_AMERICA_CODES = set(
    "AG AI AW BB BL BM BQ BS BZ CA CR CU CW DM DO GD GL GP GT HN HT JM "
    "KN KY LC MF MQ MS MX NI PA PM PR SV SX TC TT US VC VG VI".split()
)
SOUTH_AMERICA_CODES = set(
    "AR BO BR BV CL CO EC FK GF GS GY PE PY SR UY VE".split()
)
OCEANIA_CODES = set(
    "AS AU CK FJ FM GU HM KI MH MP NC NF NR NU NZ PF PG PN PW SB TK TO "
    "TV UM VU WF WS".split()
)
ANTARCTICA_CODES = {"AQ", "TF"}


def continent_from_iso2(iso2: Optional[str]) -> Optional[str]:
    code = clean_text(iso2).upper()
    if code in AFRICA_CODES:
        return "Africa"
    if code in ASIA_CODES:
        return "Asia"
    if code in EUROPE_CODES:
        return "Europe"
    if code in NORTH_AMERICA_CODES:
        return "North America"
    if code in SOUTH_AMERICA_CODES:
        return "South America"
    if code in OCEANIA_CODES:
        return "Oceania"
    if code in ANTARCTICA_CODES:
        return "Antarctica"
    return None


def iso3_from_iso2(iso2: Optional[str]) -> Optional[str]:
    code = clean_text(iso2).upper()
    if not re.fullmatch(r"[A-Z]{2}", code):
        return None
    try:
        import pycountry  # type: ignore

        country = pycountry.countries.get(alpha_2=code)
        return str(country.alpha_3) if country else None
    except Exception:
        return None


def geocode_location(
    decision: Dict[str, Any], cache: Dict[str, Any]
) -> Dict[str, Any]:
    locality = optional_text(decision.get("locality"))
    state_name = optional_text(decision.get("state"))
    country = optional_text(decision.get("country"))
    query_parts = [item for item in (locality, state_name, country) if item]

    base = {
        "locality": locality,
        "locality_aka": normalise_string_list(
            decision.get("locality_aka")
        ),
        "state": state_name,
        "country": country,
        "iso3": None,
        "continent": None,
        "lat": None,
        "lon": None,
        "geocode_query": None,
        "geocode_status": "not_attempted",
    }
    if not query_parts or not settings.ENABLE_GEOCODING:
        return base

    query = ", ".join(query_parts)
    cache_key = query.casefold()
    if cache_key in cache:
        cached = cache[cache_key]
        return cached if isinstance(cached, dict) else base

    request = _geocode_request(query)
    resolved = dict(base)
    resolved["geocode_query"] = query
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        time.sleep(settings.GEOCODER_DELAY_SECONDS)
    except Exception as exc:
        resolved["geocode_status"] = f"failed: {exc}"
        return resolved

    if not isinstance(payload, list) or not payload:
        resolved["geocode_status"] = "not_found"
        cache[cache_key] = resolved
        return resolved

    first = payload[0] if isinstance(payload[0], dict) else {}
    address = first.get("address", {})
    if not isinstance(address, dict):
        address = {}

    _apply_locality(resolved, address, locality)
    _apply_country_fields(resolved, address, state_name, country)
    _apply_coordinates(resolved, first)
    _apply_alternative_names(resolved, first)

    resolved["geocode_status"] = "resolved"
    cache[cache_key] = resolved
    return resolved


def _geocode_request(query: str) -> urllib.request.Request:
    parameters = urllib.parse.urlencode(
        {
            "q": query,
            "format": "jsonv2",
            "limit": 1,
            "addressdetails": 1,
            "namedetails": 1,
            "accept-language": "en",
        }
    )
    return urllib.request.Request(
        "https://nominatim.openstreetmap.org/search?" + parameters,
        headers={"User-Agent": settings.GEOCODER_USER_AGENT},
    )


def _apply_locality(
    resolved: Dict[str, Any],
    address: Dict[str, Any],
    original_locality: Optional[str],
) -> None:
    resolved_locality = next(
        (
            optional_text(address.get(key))
            for key in (
                "city",
                "town",
                "village",
                "municipality",
                "borough",
            )
            if optional_text(address.get(key))
        ),
        original_locality,
    )
    if (
        original_locality
        and resolved_locality
        and original_locality.casefold() != resolved_locality.casefold()
    ):
        resolved["locality_aka"] = normalise_string_list(
            [*resolved["locality_aka"], original_locality]
        )
    resolved["locality"] = resolved_locality


def _apply_country_fields(
    resolved: Dict[str, Any],
    address: Dict[str, Any],
    original_state: Optional[str],
    original_country: Optional[str],
) -> None:
    iso2 = optional_text(address.get("country_code"))
    iso2 = iso2.upper() if iso2 else None
    resolved["country"] = (
        optional_text(address.get("country")) or original_country
    )
    resolved_state = optional_text(address.get("state")) or original_state
    subdivision_code = optional_text(
        address.get("ISO3166-2-lvl4")
        or address.get("ISO3166-2-lvl3")
    )
    if (
        iso2 in {"US", "CA"}
        and subdivision_code
        and "-" in subdivision_code
    ):
        resolved_state = subdivision_code.rsplit("-", 1)[-1]
    resolved["state"] = resolved_state
    resolved["iso3"] = iso3_from_iso2(iso2) or resolved.get("iso3")
    resolved["continent"] = (
        continent_from_iso2(iso2) or resolved.get("continent")
    )


def _apply_coordinates(
    resolved: Dict[str, Any], geocode_result: Dict[str, Any]
) -> None:
    try:
        resolved["lat"] = round(float(geocode_result.get("lat")), 7)
        resolved["lon"] = round(float(geocode_result.get("lon")), 7)
    except (TypeError, ValueError):
        resolved["lat"] = None
        resolved["lon"] = None


def _apply_alternative_names(
    resolved: Dict[str, Any], geocode_result: Dict[str, Any]
) -> None:
    namedetails = geocode_result.get("namedetails", {})
    if not isinstance(namedetails, dict):
        return

    alternatives: List[str] = list(resolved["locality_aka"])
    for key in ("name:en", "official_name", "alt_name"):
        value = optional_text(namedetails.get(key))
        if value:
            alternatives.extend(re.split(r"[;,]", value))
    resolved["locality_aka"] = [
        name
        for name in normalise_string_list(alternatives)
        if not resolved["locality"]
        or name.casefold() != str(resolved["locality"]).casefold()
    ]


def run_location_stage(state: Dict[str, Any]) -> int:
    cache = load_json(settings.GEOCODE_CACHE_JSON, {})
    if not isinstance(cache, dict):
        cache = {}

    processed = 0
    for video_id, record in state.get("videos", {}).items():
        decision = record.get("text_decision")
        if not isinstance(decision, dict) or not decision.get("include"):
            continue

        existing_location = record.get("location")
        if isinstance(existing_location, dict):
            status = clean_text(existing_location.get("geocode_status"))
            if not status.startswith("failed:"):
                continue

        log(f"Resolving location for {video_id}")
        record["location"] = geocode_location(decision, cache)
        write_json_atomic(settings.GEOCODE_CACHE_JSON, cache)
        save_state(state)
        processed += 1
    return processed
