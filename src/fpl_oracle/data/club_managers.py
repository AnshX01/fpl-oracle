"""
Authoritative 2026/27 Premier League Club Managers & Reality Grounding Metadata.
Prevents LLMs and conversational agents from hallucinating pre-2026 historical realities.
"""

from __future__ import annotations

# Verified Premier League Managers for the 2026/27 Season
CLUB_MANAGERS_2026_27: dict[str, str] = {
    "ARS": "Mikel Arteta",
    "AVL": "Unai Emery",
    "BOU": "Andoni Iraola",
    "BRE": "Thomas Frank",
    "BHA": "Fabian Hürzeler",
    "CHE": "Liam Rosenior",
    "COV": "Mark Robins",
    "CRY": "Oliver Glasner",
    "EVE": "Sean Dyche",
    "FUL": "Marco Silva",
    "HUL": "Tim Walter",
    "IPS": "Kieran McKenna",
    "LEE": "Daniel Farke",
    "LIV": "Arne Slot",
    "MCI": "Enzo Maresca",  # Appointed June 2026 succeeding Pep Guardiola
    "MUN": "Rúben Amorim",
    "NEW": "Eddie Howe",
    "NFO": "Nuno Espírito Santo",
    "TOT": "Ange Postecoglou",
    "SUN": "Régis Le Bris",
}


def _load_season_transfers() -> tuple[dict[str, str], dict[str, str]]:
    import json
    from pathlib import Path

    data_file = Path(__file__).resolve().parents[3] / "data" / "season_2026_27_transfers.json"
    if data_file.exists():
        try:
            d = json.loads(data_file.read_text(encoding="utf-8"))
            return d.get("departed", {}), d.get("transfers", {})
        except Exception:
            pass
    return {}, {}


DEPARTED_PLAYERS_2026_27, KEY_2026_27_TRANSFERS = _load_season_transfers()
