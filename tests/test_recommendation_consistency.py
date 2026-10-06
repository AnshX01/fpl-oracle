"""
Cross-Surface Recommendation Consistency Tests (Q2).
Verifies that the same underlying team and market state produces 100% consistent tactical
recommendations across all system surfaces:
- Unified Decision Card (/api/decision-card)
- Squad & Lineup Workbench (/api/squad)
- Transfer Optimizer (/api/optimize)
- Chip Strategy Engine (/api/chips)
"""

import asyncio
import json

from fpl_oracle.server.routes.api import (
    OptimizeRequest,
    get_chip_strategy,
    get_decision_card_endpoint,
    get_squad,
    run_optimizer,
)


def _get_json_data(response):
    """Extract dict payload from SafeJSONResponse."""
    if hasattr(response, "body"):
        return json.loads(response.body.decode("utf-8"))
    return response


from unittest.mock import AsyncMock, patch

def test_cross_surface_recommendation_consistency():
    """Assert that decision card, squad, and optimizer agree on transfers, captaincy, and chips."""
    async def _run():
        with patch("fpl_oracle.news.ingest.news_ingestion.fetch_rss_articles", new=AsyncMock(return_value=[])):
            # 1. Fetch Decision Card
            dc_resp = await get_decision_card_endpoint()
            dc_data = _get_json_data(dc_resp)

            # 2. Fetch Squad & Lineup
            sq_resp = await get_squad()
            sq_data = _get_json_data(sq_resp)

            # 3. Fetch Transfer Optimization
            opt_resp = await run_optimizer(OptimizeRequest())
            opt_data = _get_json_data(opt_resp)

            # 4. Fetch Chip Strategy
            chip_resp = await get_chip_strategy()
            chip_data = _get_json_data(chip_resp)

        # --- Consistency Check 1: Captaincy Alignment ---
        dc_captain_id = dc_data["captain"]["element"]
        sq_captain_id = sq_data["captain"]["element"]
        assert dc_captain_id == sq_captain_id, (
            f"Captain mismatch: Decision Card recommends {dc_data['captain']['web_name']} (id {dc_captain_id}) "
            f"while Squad recommends {sq_data['captain']['web_name']} (id {sq_captain_id})"
        )

        # Vice-Captain alignment
        dc_vice_id = dc_data["vice_captain"]["element"]
        sq_vice_id = sq_data["vice_captain"]["element"]
        assert dc_vice_id == sq_vice_id, (
            f"Vice-captain mismatch: Decision Card recommends id {dc_vice_id} "
            f"while Squad recommends id {sq_vice_id}"
        )

        # --- Consistency Check 2: Transfer Move Alignment ---
        opt_rec = opt_data["recommended_plan"]
        dc_transfers = dc_data["transfers"]

        if opt_rec.get("plan_type") == "ROLL_TRANSFER":
            assert dc_transfers["is_roll"] is True, "Optimizer recommended ROLL but Decision Card did not"
            assert dc_transfers["action"] == "ROLL"
        else:
            assert dc_transfers["is_roll"] is False, "Optimizer recommended transfers but Decision Card rolled"
            opt_in_ids = {p.get("element") for p in opt_rec.get("transfers_in", []) if isinstance(p, dict)}
            dc_in_ids = {p["element"] for p in dc_transfers.get("in", [])}
            assert opt_in_ids == dc_in_ids, (
                f"Transfers In mismatch: Optimizer={opt_in_ids} vs Decision Card={dc_in_ids}"
            )

            opt_out_ids = {p.get("element") for p in opt_rec.get("transfers_out", []) if isinstance(p, dict)}
            dc_out_ids = {p["element"] for p in dc_transfers.get("out", [])}
            assert opt_out_ids == dc_out_ids, (
                f"Transfers Out mismatch: Optimizer={opt_out_ids} vs Decision Card={dc_out_ids}"
            )

        # Hit count agreement
        assert dc_transfers["hits_count"] == opt_rec.get("hits", 0)

        # --- Consistency Check 3: Chip Recommendation Alignment ---
        chip_rec_flag = chip_data.get("recommend_chip_this_gw", False)
        assert dc_data["chip"]["recommend"] == chip_rec_flag, (
            f"Chip recommendation mismatch: Chip Engine={chip_rec_flag} vs Decision Card={dc_data['chip']['recommend']}"
        )

        # --- Consistency Check 4: Formation Agreement ---
        assert dc_data["formation"] == sq_data["formation"], (
            f"Formation mismatch: Decision Card={dc_data['formation']} vs Squad={sq_data['formation']}"
        )

    asyncio.run(_run())
