"""
Tests for fuzzy player matching, manual squad input, JSON/CSV parsing, and persistence.
"""

import json

import httpx
import pytest

from fpl_oracle.data.fuzzy_match import clean_query, fuzzy_matcher, strip_accents
from fpl_oracle.server.main import app


def test_clean_query_and_accents():
    # Accented characters
    assert strip_accents("Ødegaard") == "Odegaard"
    assert strip_accents("Magalhães") == "Magalhaes"

    # Parenthetical club and position stripping
    assert "Arsenal" not in clean_query("David Raya (Arsenal)")
    assert "DEF" not in clean_query("Gabriel [DEF] - £6.0m")
    assert clean_query("1. Erling Haaland") == "Erling Haaland"
    assert clean_query("* Mohamed Salah (LIV)") == "Mohamed Salah"

def test_extract_names_json_and_csv():
    # JSON list of strings
    json_str = '["Raya", "Gabriel", "Haaland", "Salah"]'
    extracted = fuzzy_matcher.extract_names_from_raw(json_str)
    assert extracted == ["Raya", "Gabriel", "Haaland", "Salah"]

    # JSON dict with players key
    json_dict_str = json.dumps({"players": ["Palmer", "Saka", "Watkins"]})
    extracted_dict = fuzzy_matcher.extract_names_from_raw(json_dict_str)
    assert extracted_dict == ["Palmer", "Saka", "Watkins"]

    # CSV string with header
    csv_str = "Name,Position,Price\nRaya,GKP,5.5\nGabriel,DEF,6.0\nHaaland,FWD,15.2"
    extracted_csv = fuzzy_matcher.extract_names_from_raw(csv_str)
    assert "Raya" in extracted_csv
    assert "Gabriel" in extracted_csv
    assert "Haaland" in extracted_csv

    # Plain text newline
    text_str = "Raya\nGabriel\nHaaland\nSalah"
    extracted_text = fuzzy_matcher.extract_names_from_raw(text_str)
    assert extracted_text == ["Raya", "Gabriel", "Haaland", "Salah"]

@pytest.mark.anyio
async def test_manual_squad_endpoints_and_persistence(tmp_path):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Match 15 players
        raw_names = (
            "David Raya\nFabianski\n"
            "Gabriel Magalhães\nAlexander-Arnold\nGvardiol\nRobinson\nKonsa\n"
            "Mohamed Salah\nBukayo Saka\nCole Palmer\nBryan Mbeumo\nMorgan Rogers\n"
            "Erling Haaland\nOllie Watkins\nChris Wood"
        )
        match_res = await client.post("/api/squad/match", json={"raw_text": raw_names})
        assert match_res.status_code == 200
        match_data = match_res.json()
        assert match_data["total_matched"] >= 14 # Vast majority matched cleanly

        # Pick 15 valid IDs from the database/API to test manual squad submission
        squad_res = await client.get("/api/squad")
        assert squad_res.status_code == 200
        starters = squad_res.json()["starters"]
        bench = squad_res.json()["bench"]
        fifteen_ids = [p["element"] for p in starters] + [p["element"] for p in bench]
        assert len(fifteen_ids) == 15

        # 2. Save manual squad
        save_res = await client.post("/api/squad/manual", json={
            "player_ids": fifteen_ids,
            "bank": 1.5,
            "free_transfers": 2
        })
        assert save_res.status_code == 200
        assert save_res.json()["status"] == "success"

        # 3. Verify profile persistence
        prof_res = await client.get("/api/profile")
        assert prof_res.status_code == 200
        prof_data = prof_res.json()
        assert prof_data["bank"] == 1.5
        assert prof_data["free_transfers"] == 2

        # 4. Verify squad endpoint loads this manual squad when manager_id is unset
        # Temporarily clear manager_id
        await client.post("/api/profile", json={"manager_id": None})
        squad_manual = await client.get("/api/squad")
        assert squad_manual.status_code == 200
        squad_json = squad_manual.json()
        assert len(squad_json["starters"]) + len(squad_json["bench"]) == 15
        assert squad_json["bank_millions"] == 1.5
