"""
Unit and regression tests for Wave 3 Blind Spots Remediation.
Guarantees mathematical coherence, data pipeline integrity, and rule correctness.
Zero hardcoded player names in test code.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from fpl_oracle.advisor.funding import AffordabilityPathFinder
from fpl_oracle.advisor.timing import TransferTimingAdvisor
from fpl_oracle.api.cache import CacheManager
from fpl_oracle.api.models import Element
from fpl_oracle.data.fuzzy_match import FuzzyPlayerMatcher
from fpl_oracle.data.store import data_store
from fpl_oracle.ml.components.defcon import DefConModel
from fpl_oracle.ml.components.minutes import MinutesModel
from fpl_oracle.news.gemini_extractor import GeminiEvidenceExtractor
from fpl_oracle.news.models import EvidenceCategory
from fpl_oracle.news.reconcile import official_chance_of_playing
from fpl_oracle.sim.match_sim import SimulationResult


# 1. Gemini Extractor: Markdown fences and JSON sanitation
@pytest.mark.asyncio
async def test_gemini_extractor_sanitizes_markdown_fences():
    extractor = GeminiEvidenceExtractor()
    fence_json = '```json\n[{"player_name": "Test Player", "category": "available", "confidence": 0.9, "quote": "Some report"}]\n```'

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"candidates": [{"content": {"parts": [{"text": fence_json}]}}]}

    env_patch = {
        "GEMINI_API_KEY": "mock_key",
        "GEMINI_TIMEOUT_SECONDS": "30.0",
        "GEMINI_FREE_TIER_CONFIRMED": "true",
    }

    with (
        patch.object(extractor, "is_configured", return_value=(True, "Ready")),
        patch.dict("os.environ", env_patch),
        patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post,
    ):
        mock_post.return_value = mock_resp
        payload = await extractor.extract_evidence_from_text(
            article_text="Some report about Test Player",
            article_url="https://example.com/test",
            published_at=datetime.now(UTC),
            candidate_players=[{"id": 1, "web_name": "Test Player", "team_name": "Arsenal", "team": 1}],
            target_gw=10,
        )

        assert len(payload.evidences) == 1
        assert payload.evidences[0].category == EvidenceCategory.AVAILABLE


# 2. SQLite Cache Invalidation
def test_cache_invalidation_deletes_from_sqlite_data_store():
    cache = CacheManager()
    test_key = "test_key_wave3"
    test_val = {"key": "val123"}

    cache.set(test_key, test_val, ttl_seconds=300)
    assert cache.get(test_key) == test_val
    assert data_store.get_cache_entry(test_key) is not None

    cache.invalidate(test_key)
    # Both in-memory and SQLite must be purged
    assert cache._mem_cache.get(test_key) is None
    assert data_store.get_cache_entry(test_key) is None


# 3. News Reconciliation: Stale passed return dates ramp from first_gw, unknown return date stays 0.0
def test_stale_return_date_ramps_from_first_gw_and_unknown_stays_zero():
    class MockElementStale:
        status = "i"
        chance_of_playing_next_round = 0
        news = "Expected back 15 Sep"

    class MockElementUnknown:
        status = "i"
        chance_of_playing_next_round = 0
        news = "Unknown return date"

    class MockEvent:
        id = 10
        deadline_time = "2026-10-24T10:00:00Z"

    # Stale return date ramps from first_gw (50.0%)
    cop_stale = official_chance_of_playing(
        element=MockElementStale(),
        target_gw=10,
        first_gw=10,
        events=[MockEvent()],
    )
    assert cop_stale == 50.0

    # Unknown return date stays strictly 0.0%
    cop_unknown = official_chance_of_playing(
        element=MockElementUnknown(),
        target_gw=10,
        first_gw=10,
        events=[MockEvent()],
    )
    assert cop_unknown == 0.0


# 4. Transfer Timing: Independent probability union
def test_transfer_timing_uses_independent_union_probability():
    advisor = TransferTimingAdvisor()
    # P(target rise) = 0.60, P(outgoing fall) = 0.50
    # Union = 0.60 + 0.50 - 0.30 = 0.80 (80%), NOT max(0.60, 0.50) = 0.60
    rec = advisor.evaluate_timing(
        bank=0.0,
        target_rise_prob=0.60,
        outgoing_fall_prob=0.50,
        pressers_concluded=False,
        midweek_matches_pending=0,
    )
    assert np.isclose(rec.price_change_prob, 0.80, atol=1e-3)
    # 80% price change risk with zero bank triggers EXECUTE_TONIGHT_PRICED_OUT
    assert rec.urgency == "EXECUTE_TONIGHT_PRICED_OUT"


# 5. Critique: Half-profit rule in transfer sell price
def test_critique_evaluates_sell_price_with_half_profit_rule():
    from fpl_oracle.advisor.critique import PlanCritiqueService

    service = PlanCritiqueService()
    # Outgoing player: purchase_price = 50 (£5.0m), now_cost = 60 (£6.0m)
    # Official sell price = 50 + floor((60 - 50)/2) = 55 (£5.5m), NOT 60 (£6.0m)
    squad_df = pd.DataFrame(
        [
            {
                "element": 101,
                "position": "MID",
                "purchase_price": 50,
                "now_cost": 60,
                "value": 60,
                "expected_points": 4.0,
            }
        ]
    )
    # Target player: now_cost = 60 (£6.0m)
    pool_df = pd.DataFrame(
        [
            {
                "element": 101,
                "position": "MID",
                "now_cost": 60,
                "value": 60,
                "expected_points": 4.0,
            },
            {
                "element": 202,
                "position": "MID",
                "now_cost": 60,
                "value": 60,
                "expected_points": 4.5,
            },
        ]
    )

    # With bank = 0: sell price 55 - buy cost 60 = -5 net bank (unaffordable)
    # Without half-profit rule, bank was credited 60 - 60 = 0 (affordable)
    mock_opt_plan = MagicMock()
    mock_opt_plan.total_objective = 20.0
    mock_opt_plan.recommended_action.transfers_out = []
    mock_opt_plan.recommended_action.transfers_in = []

    with patch.object(service.optimizer, "optimize", return_value=mock_opt_plan):
        res = service.evaluate_plan(
            proposed_transfers=[{"out": 101, "in": 202}],
            proposed_chip=None,
            proposed_captain=None,
            current_squad_df=squad_df,
            player_pool_df=pool_df,
            bank=0.0,
            free_transfers=1,
            horizon_projections={10: pool_df},
            current_gw=10,
            target_gw=10,
            horizon_len=1,
        )
        assert res.verdict == "WORSE THAN ALTERNATIVE"
        assert "exceeds available budget" in res.numeric_reason.lower()


# 6. Funding: Bench harvest upgrades respect club cap
def test_bench_harvest_upgrades_enforces_three_player_club_cap():
    finder = AffordabilityPathFinder()
    # Squad already has 3 players from Team 1 (e.g. IDs 1, 2, 3)
    squad_df = pd.DataFrame(
        [
            {
                "element": 1,
                "position": "DEF",
                "team": 1,
                "purchase_price": 50,
                "now_cost": 50,
                "value": 50,
                "expected_points": 4.0,
            },
            {
                "element": 2,
                "position": "MID",
                "team": 1,
                "purchase_price": 70,
                "now_cost": 70,
                "value": 70,
                "expected_points": 5.0,
            },
            {
                "element": 3,
                "position": "MID",
                "team": 1,
                "purchase_price": 80,
                "now_cost": 80,
                "value": 80,
                "expected_points": 6.0,
            },
            {
                "element": 4,
                "position": "DEF",
                "team": 2,
                "purchase_price": 55,
                "now_cost": 55,
                "value": 55,
                "expected_points": 3.0,
            },  # Bench harvest candidate
            {
                "element": 5,
                "position": "FWD",
                "team": 3,
                "purchase_price": 75,
                "now_cost": 75,
                "value": 75,
                "expected_points": 5.5,
            },
        ]
    )

    # Player pool contains an upgrade candidate from Team 1 (would make 4 Team 1 players)
    # and from Team 4 (legal)
    pool_df = pd.DataFrame(
        [
            {
                "element": 10,
                "position": "DEF",
                "team": 5,
                "value": 40,
                "now_cost": 40,
                "expected_points": 1.5,
                "chance_of_playing": 100,
            },  # enabler
            {
                "element": 20,
                "position": "FWD",
                "team": 1,
                "value": 85,
                "now_cost": 85,
                "expected_points": 8.0,
                "chance_of_playing": 100,
            },  # Team 1 (illegal)
            {
                "element": 30,
                "position": "FWD",
                "team": 4,
                "value": 85,
                "now_cost": 85,
                "expected_points": 7.5,
                "chance_of_playing": 100,
            },  # Team 4 (legal)
        ]
    )

    with (
        patch("fpl_oracle.advisor.health.SquadHealthDiagnosis.diagnose_capital_efficiency") as mock_cap,
        patch("fpl_oracle.advisor.health.SquadHealthDiagnosis.diagnose_squad") as mock_diag,
    ):
        mock_cap_rep = MagicMock()
        mock_cap_rep.bench_bloat_detected = True
        mock_cap_rep.bench_harvest_candidates = [{"element": 4, "position": "DEF", "role": "Sub 1"}]
        mock_cap.return_value = mock_cap_rep

        mock_health_rep = MagicMock()
        mock_health_rep.starter_ids = [1, 2, 3, 5]
        mock_diag.return_value = mock_health_rep

        plans = finder.find_bench_harvest_upgrades(squad_df, pool_df, bank_tenths=15, min_gain_xp=1.0)
        # Upgrades to Team 1 must not be recommended because squad already has 3 Team 1 players
        for p in plans:
            assert p["upgrade_in_element"] != 20


# 7. ML Predict: DGW strictly monotonic quantiles
def test_dgw_quantiles_maintain_strict_monotonicity_p10_p50_p90():
    from fpl_oracle.ml.predict import ProjectionEngine

    engine = ProjectionEngine()
    engine.is_loaded = True
    # Mock features dataframe with two fixtures for element 101
    feat_df = pd.DataFrame(
        [
            {
                "element": 101,
                "web_name": "TestPlayer",
                "team": 1,
                "position": "MID",
                "value": 80.0,
                "target_gw": 10,
                "is_bgw": False,
                "is_dgw": True,
                "fixture_id": 1001,
                "chance_of_playing": 100.0,
                "news_start_probability": np.nan,
                "news_minutes_limit": np.nan,
                "pos_MID": 1.0,
                "pos_DEF": 0.0,
                "pos_GKP": 0.0,
                "pos_FWD": 0.0,
            },
            {
                "element": 101,
                "web_name": "TestPlayer",
                "team": 1,
                "position": "MID",
                "value": 80.0,
                "target_gw": 10,
                "is_bgw": False,
                "is_dgw": True,
                "fixture_id": 1002,
                "chance_of_playing": 100.0,
                "news_start_probability": np.nan,
                "news_minutes_limit": np.nan,
                "pos_MID": 1.0,
                "pos_DEF": 0.0,
                "pos_GKP": 0.0,
                "pos_FWD": 0.0,
            },
        ]
    )

    from fpl_oracle.data.features import FEATURE_COLUMNS

    extra_cols = ["days_rest", "was_home", "opponent_difficulty", "net_transfer_ratio", "opponent_team", "is_home"]
    for c in list(FEATURE_COLUMNS) + extra_cols:
        if c not in feat_df.columns:
            feat_df[c] = 1.0

    engine.minutes_model = MagicMock()
    engine.minutes_model.predict.return_value = {
        "p_starts": np.array([0.9, 0.9]),
        "p_min60": np.array([0.8, 0.8]),
        "p_play": np.array([0.95, 0.95]),
        "expected_minutes": np.array([85.0, 85.0]),
    }
    engine.attacking_model = MagicMock()
    engine.attacking_model.predict.return_value = {
        "expected_goals": np.array([0.4, 0.4]),
        "expected_assists": np.array([0.3, 0.3]),
    }
    engine.defending_model = MagicMock()
    engine.defending_model.predict.return_value = {
        "p_clean_sheet": np.array([0.4, 0.4]),
        "expected_goals_conceded": np.array([1.1, 1.1]),
    }
    engine.defcon_model = MagicMock()
    engine.defcon_model.predict.return_value = {
        "p_defcon": np.array([0.2, 0.2]),
        "expected_defcon_actions": np.array([3.0, 3.0]),
    }
    engine.bonus_model = MagicMock()
    engine.bonus_model.predict.return_value = {
        "expected_bonus": np.array([0.5, 0.5]),
    }
    engine.cards_saves_model = MagicMock()
    engine.cards_saves_model.predict.return_value = {
        "expected_saves": np.array([0.0, 0.0]),
        "expected_penalties_saved": np.array([0.0, 0.0]),
        "expected_card_deduction": np.array([0.1, 0.1]),
    }

    mock_bootstrap = MagicMock()
    mock_fixtures = [MagicMock()]

    with (
        patch("fpl_oracle.data.features.feature_engineering.extract_live_features_for_upcoming", return_value=feat_df),
        patch("fpl_oracle.ml.model_registry.model_registry.get_active_version", return_value={"version": "default"}),
    ):
        preds = engine.predict_gameweek(10, mock_bootstrap, mock_fixtures)
        row = preds.iloc[0]
        assert row["p10"] <= row["p50"] <= row["p90"]


# 8. Feature Alignment: Reindexing missing columns without crashing
def test_ml_components_reindex_missing_features_without_lightgbm_crash():
    min_model = MinutesModel()
    min_model.is_fitted = True
    min_model.clf_starts = MagicMock()
    min_model.clf_starts.feature_name_ = ["col_A", "col_B", "col_C"]
    min_model.clf_starts.predict_proba.return_value = np.array([[0.1, 0.9]])

    min_model.clf_min60 = MagicMock()
    min_model.clf_min60.predict_proba.return_value = np.array([[0.2, 0.8]])

    min_model.reg_minutes = MagicMock()
    min_model.reg_minutes.predict.return_value = np.array([85.0])

    min_model.calibrator_starts = MagicMock()
    min_model.calibrator_starts.calibrate.return_value = np.array([0.9])
    min_model.calibrator_min60 = MagicMock()
    min_model.calibrator_min60.calibrate.return_value = np.array([0.8])

    # X only has col_A and col_B (col_C missing)
    X = pd.DataFrame([{"col_A": 1.0, "col_B": 2.0}])
    preds = min_model.predict(X)

    assert "p_starts" in preds
    assert len(preds["p_starts"]) == 1
    # Verify predict_proba was called with 3 columns (col_C filled with 0.0)
    called_df = min_model.clf_starts.predict_proba.call_args[0][0]
    assert list(called_df.columns) == ["col_A", "col_B", "col_C"]
    assert called_df["col_C"].iloc[0] == 0.0


# 9. DefCon: Zero clamping for non-playing players
def test_defcon_clamps_zero_expected_actions_and_substitutes():
    model = DefConModel()
    model.is_fitted = True
    model.selected_model_per_pos = {"DEF": "count"}
    model.reg_count = MagicMock()
    # pred_mu = 0.005 (near zero expected defensive actions)
    model.reg_count.predict.return_value = np.array([0.005])
    model.calibrator = MagicMock()
    model.calibrator.calibrate.return_value = np.array([0.06])  # un-clamped Platt intercept

    X = pd.DataFrame([{"pos_DEF": 1.0, "pos_MID": 0.0, "pos_FWD": 0.0, "pos_GKP": 0.0}])
    res = model.predict(X)

    # Must be clamped to 0.0
    assert res["p_defcon"][0] == 0.0


# 10. Fuzzy Matcher: Position hint awareness in sequence matching
def test_fuzzy_matcher_penalizes_position_mismatch_in_fuzzy_search():
    matcher = FuzzyPlayerMatcher()
    e_def = Element(
        id=1,
        code=1,
        web_name="Player",
        first_name="Gabriel",
        second_name="Defender",
        team=1,
        element_type=2,
        now_cost=55,
    )
    e_fwd = Element(
        id=2,
        code=2,
        web_name="Player",
        first_name="Gabriel",
        second_name="Striker",
        team=1,
        element_type=4,
        now_cost=75,
    )

    # Query asking for Striker Gabriel
    res = matcher.match_single_name("Gabriel (FWD)", [e_def, e_fwd])
    matched = res.get("matched")
    assert matched is not None
    assert matched["element"] == 2
    assert matched["position"] == "FWD"


# 11. Match Simulator: Position-legal bench auto-substitution
def test_match_simulator_formation_legal_bench_autosub():
    # 1 GKP, 3 DEF, 2 MID, 1 FWD (minimal setup)
    scores = np.array(
        [
            [2.0],  # Starter 1: GKP (0 mins)
            [6.0],  # Starter 2: DEF (90 mins)
            [5.0],  # Starter 3: DEF (90 mins)
            [4.0],  # Starter 4: DEF (90 mins)
            [8.0],  # Starter 5: MID (90 mins)
            [7.0],  # Starter 6: MID (90 mins)
            [9.0],  # Starter 7: FWD (90 mins)
            [3.0],  # Bench 8: Backup GKP (90 mins)
            [5.0],  # Bench 9: Bench MID (90 mins)
        ]
    )
    minutes = np.array(
        [
            [0],  # Starter 1 unplayed
            [90],
            [90],
            [90],
            [90],
            [90],
            [90],
            [90],  # Bench GKP played
            [90],  # Bench MID played
        ]
    )

    sim_res = SimulationResult(
        scores=scores,
        player_ids=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        player_names=np.array(["G1", "D1", "D2", "D3", "M1", "M2", "F1", "G2", "M3"]),
        positions=np.array(["GKP", "DEF", "DEF", "DEF", "MID", "MID", "FWD", "GKP", "MID"]),
        teams=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
        n_sims=1,
        analytic_xp=np.zeros(9),
        minutes=minutes,
        goals=np.zeros((9, 1)),
        assists=np.zeros((9, 1)),
        clean_sheets=np.zeros((9, 1)),
        goals_conceded=np.zeros((9, 1)),
        saves=np.zeros((9, 1)),
        defcon=np.zeros((9, 1)),
        bonus=np.zeros((9, 1)),
        cards=np.zeros((9, 1)),
        own_goals=np.zeros((9, 1)),
        penalties_missed=np.zeros((9, 1)),
        penalties_saved=np.zeros((9, 1)),
    )

    # Starters 1..7; Bench: 9 (MID) before 8 (GKP) in bench_ids order
    # Even if Bench MID is listed first, Starter GKP must ONLY be replaced by Bench GKP!
    squad_pts = sim_res.get_squad_score_distribution(
        starter_ids=[1, 2, 3, 4, 5, 6, 7],
        bench_ids=[9, 8],
        captain_id=7,
        vice_id=6,
    )
    # Expected starters sum = 2 (unplayed) + 6 + 5 + 4 + 8 + 7 + 9 + 9 (cap bonus on 7) = 50
    # Starter GKP (2 pts) is replaced by Bench GKP (3 pts), so +3 pts = 53
    # Bench MID (5 pts) must NOT be substituted for a GKP
    assert squad_pts[0] == 53.0
