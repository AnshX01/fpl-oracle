"""
Fuzzy player name matcher for manual / paste / upload squad input.
Supports plain text lists, JSON arrays/objects, and CSV formats with accented characters.
"""

import csv
import difflib
import io
import json
import re
import unicodedata
from typing import Any

from fpl_oracle.api.models import Element


def strip_accents(text: str) -> str:
    """Normalize and remove diacritics/accents (e.g. Ødegaard -> Odegaard, Magalhães -> Magalhaes)."""
    norm = unicodedata.normalize("NFKD", text)
    return "".join(c for c in norm if not unicodedata.combining(c)).replace("ø", "o").replace("Ø", "O")


def clean_query(text: str) -> str:
    """Normalize text: remove bullets, numbering, parenthetical club/price/position tags."""
    t = text.strip()
    # Strip leading bullets/numbers: "1. ", "- ", "* "
    t = re.sub(r"^[\d\.\-\*\•\)\(\]\:\s]+", "", t)
    # Strip parenthetical annotations: "(Arsenal)", "(£5.5m)", "[DEF]", "(MID)"
    t = re.sub(r"\(.*?\)", " ", t)
    t = re.sub(r"\[.*?\]", " ", t)
    # Strip common position suffixes/prefixes: "GKP: ", "- DEF", "FWD "
    t = re.sub(r"\b(GKP|DEF|MID|FWD|GK)\b", " ", t, flags=re.IGNORECASE)
    # Strip price suffixes: "5.5m", "£5.5"
    t = re.sub(r"£?\d+\.\d+m?", " ", t, flags=re.IGNORECASE)
    # Strip non-alphanumeric except spaces, hyphens, periods
    t = re.sub(r"[^\w\s\-\.]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


class FuzzyPlayerMatcher:
    def match_single_name(
        self, query: str, elements: list[Element], team_map: dict[int, str] | None = None
    ) -> dict[str, Any]:
        """
        Fuzzy match a single player name string against element candidates.
        """
        cleaned = clean_query(query)
        if not cleaned:
            return {"query": query, "matched": None, "confidence": 0.0, "alternatives": []}

        cleaned_norm = strip_accents(cleaned).lower()

        scored_candidates = []
        for elem in elements:
            web_name_norm = strip_accents(elem.web_name).lower()
            first_name_norm = strip_accents(str(elem.first_name or "")).lower()
            second_name_norm = strip_accents(str(elem.second_name or "")).lower()
            full_name_norm = f"{first_name_norm} {second_name_norm}"

            # Exact matching
            if cleaned_norm in (web_name_norm, second_name_norm, full_name_norm):
                score = 1.0
            else:
                r1 = difflib.SequenceMatcher(None, cleaned_norm, web_name_norm).ratio()
                r2 = difflib.SequenceMatcher(None, cleaned_norm, full_name_norm).ratio()
                r3 = difflib.SequenceMatcher(None, cleaned_norm, second_name_norm).ratio()

                # Substring bonus: e.g. "Salah" in "Mohamed Salah"
                if cleaned_norm == second_name_norm or cleaned_norm in web_name_norm:
                    score = max(r1, r2, r3, 0.95)
                elif cleaned_norm in full_name_norm:
                    score = max(r1, r2, r3, 0.90)
                else:
                    score = max(r1, r2, r3)

            if score >= 0.40:
                team_short = team_map.get(elem.team, str(elem.team)) if team_map else str(elem.team)
                pos_str = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}.get(elem.element_type, "MID")
                scored_candidates.append(
                    {
                        "element": elem.id,
                        "web_name": elem.web_name,
                        "full_name": f"{elem.first_name} {elem.second_name}",
                        "team": elem.team,
                        "team_short": team_short,
                        "position": pos_str,
                        "cost": round(elem.now_cost / 10.0, 1),
                        "score": round(score, 2),
                    }
                )

        scored_candidates.sort(key=lambda x: float(str(x["score"])), reverse=True)

        if scored_candidates and float(str(scored_candidates[0]["score"])) >= 0.55:
            best = scored_candidates[0]
            alts = scored_candidates[1:5]
            return {
                "query": query,
                "cleaned": cleaned,
                "matched": best,
                "confidence": best["score"],
                "alternatives": alts,
            }
        else:
            return {
                "query": query,
                "cleaned": cleaned,
                "matched": None,
                "confidence": 0.0,
                "alternatives": scored_candidates[:4],
            }

    def extract_names_from_raw(self, raw_input: str | list[Any]) -> list[str]:
        """
        Parse raw input string or object list into individual player name queries.
        Handles JSON arrays/objects, CSV strings, or newline-delimited text.
        """
        if isinstance(raw_input, list):
            names = []
            for item in raw_input:
                if isinstance(item, dict):
                    name = item.get("web_name") or item.get("name") or item.get("player") or str(item)
                    names.append(str(name))
                else:
                    names.append(str(item))
            return [n.strip() for n in names if n.strip()]

        raw_str = raw_input.strip()

        # Try JSON parsing
        if (raw_str.startswith("[") and raw_str.endswith("]")) or (raw_str.startswith("{") and raw_str.endswith("}")):
            try:
                parsed = json.loads(raw_str)
                if isinstance(parsed, list):
                    return self.extract_names_from_raw(parsed)
                elif isinstance(parsed, dict):
                    # Check for "players", "squad", "elements"
                    for key in ["players", "squad", "elements", "picks"]:
                        if key in parsed and isinstance(parsed[key], list):
                            return self.extract_names_from_raw(parsed[key])
            except Exception:
                pass

        # Try CSV parsing if comma or tab found
        if "\n" in raw_str and ("," in raw_str or "\t" in raw_str):
            try:
                delimiter = "\t" if "\t" in raw_str else ","
                reader = csv.reader(io.StringIO(raw_str), delimiter=delimiter)
                rows = list(reader)
                if rows:
                    # Check if first row is header
                    first_row_lower = [col.lower().strip() for col in rows[0]]
                    name_col_idx = 0
                    start_row = 0
                    for idx, col_name in enumerate(first_row_lower):
                        if col_name in ["name", "player", "player_name", "web_name", "element"]:
                            name_col_idx = idx
                            start_row = 1
                            break

                    csv_names = []
                    for row in rows[start_row:]:
                        if row and len(row) > name_col_idx:
                            val = row[name_col_idx].strip()
                            if val:
                                csv_names.append(val)
                    if len(csv_names) >= 1:
                        return csv_names
            except Exception:
                pass

        # Default: split by newlines, semicolons, or commas (if no newlines)
        if "\n" in raw_str:
            lines = [line.strip() for line in raw_str.split("\n") if line.strip()]
        else:
            lines = [chunk.strip() for chunk in re.split(r"[,;]+", raw_str) if chunk.strip()]

        return lines

    def parse_and_match_squad(
        self, raw_input: str | list[str], elements: list[Element], team_map: dict[int, str] | None = None
    ) -> dict[str, Any]:
        """
        Parse raw input (plain text, JSON, CSV) and fuzzy match all candidate players.
        """
        candidate_queries = self.extract_names_from_raw(raw_input)

        matches = []
        unmatched = []
        matched_elem_ids = set()

        for q in candidate_queries:
            res = self.match_single_name(q, elements, team_map)
            if res["matched"]:
                elem_id = res["matched"]["element"]
                if elem_id not in matched_elem_ids:
                    matches.append(res["matched"])
                    matched_elem_ids.add(elem_id)
                else:
                    # Duplicate entry in input, skip
                    continue
            else:
                unmatched.append(res)

        # Validate formation & constraints
        pos_counts = {"GKP": 0, "DEF": 0, "MID": 0, "FWD": 0}
        club_counts: dict[int, int] = {}
        total_cost = 0.0

        for m in matches:
            pos_counts[m["position"]] = pos_counts.get(m["position"], 0) + 1
            club_counts[m["team"]] = club_counts.get(m["team"], 0) + 1
            total_cost += m["cost"]

        is_valid_15 = (
            len(matches) == 15
            and pos_counts["GKP"] == 2
            and pos_counts["DEF"] == 5
            and pos_counts["MID"] == 5
            and pos_counts["FWD"] == 3
            and all(count <= 3 for count in club_counts.values())
        )

        return {
            "total_matched": len(matches),
            "is_valid_15": is_valid_15,
            "matches": matches,
            "unmatched": unmatched,
            "position_counts": pos_counts,
            "club_counts": club_counts,
            "total_cost": round(total_cost, 1),
        }


fuzzy_matcher = FuzzyPlayerMatcher()
