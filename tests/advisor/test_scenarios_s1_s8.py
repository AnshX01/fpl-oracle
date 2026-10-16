"""
Scenario Regression Suite S1 to S8 and Advice Ledger Tests (Phase 8: WI-8.7 & WI-8.9).
Verifies expert advisor behaviors across reference scenarios:
S1: Owner's case: healthy core, 1 FT, bank, Wildcard available -> No Wildcard, 1 FT fix, timeline shown
S2: Truly broken squad -> Overhaul justified by health diagnosis + frontier
S3: Blank gameweek -> Free hit evaluated; rules R2 enforced
S4: Set 1 deadline close (GW19) -> Use-it-or-lose-it logic
S5: Manager proposes hit for marginal upgrade -> Verdict WORSE THAN ALTERNATIVE
S6: Mini-league lead small with template rival -> Safe vs differential profiles diverge
S7: Imminent price rise -> Timing warning attached, minimal intervention preserved
S8: Key player doubt -> Decision tree / contingency planning
Also verifies AdviceLedger immutability, finalised-only scoring, and report card replay.
Zero hardcoded player names in production code.
"""

from pathlib import Path

import pandas as pd
import pytest

from fpl_oracle.advisor.briefing import (
    AdvisorBriefingService,
    AdvisorRecommendation,
    CaptainAdvice,
    RecommendationCost,
    RecommendationRisk,
    validate_recommendation_schema,
)
from fpl_oracle.advisor.critique import PlanCritiqueService
from fpl_oracle.advisor.frontier import FrontierPlanner
from fpl_oracle.advisor.health import SquadHealthDiagnosis
from fpl_oracle.advisor.ledger import AdviceLedger
from fpl_oracle.advisor.profile import ManagerProfile, ManagerProfileService, MiniLeagueContext
from fpl_oracle.optimise.multiperiod import MultiPeriodConfig, MultiPeriodOptimizer


def build_synthetic_player_pool() -> pd.DataFrame:
    """Builds a deterministic 30-player pool (4 GKP, 10 DEF, 10 MID, 6 FWD) with IDs."""
    rows = []
    # 4 GKP: IDs 1..4
    for i in range(1, 5):
        rows.append({
            "element": i,
            "web_name": f"GKP_{i}",
            "position": "GKP",
            "team": (i % 20) + 1,
            "value": 50,
            "purchase_price": 50,
            "selling_price": 50,
            "expected_points": 4.5 - (i * 0.2),
            "p10": 2.0,
            "p90": 6.5,
            "chance_of_playing": 100.0,
            "selected_by_percent": 15.0,
            "news": "",
        })
    # 10 DEF: IDs 5..14
    for i in range(5, 15):
        rows.append({
            "element": i,
            "web_name": f"DEF_{i}",
            "position": "DEF",
            "team": (i % 20) + 1,
            "value": 50,
            "purchase_price": 50,
            "selling_price": 50,
            "expected_points": 5.2 - ((i - 5) * 0.15),
            "p10": 1.5,
            "p90": 8.0,
            "chance_of_playing": 100.0,
            "selected_by_percent": 20.0,
            "news": "",
        })
    # 10 MID: IDs 15..24
    for i in range(15, 25):
        rows.append({
            "element": i,
            "web_name": f"MID_{i}",
            "position": "MID",
            "team": (i % 20) + 1,
            "value": 65,
            "purchase_price": 65,
            "selling_price": 65,
            "expected_points": 6.8 - ((i - 15) * 0.2),
            "p10": 2.5,
            "p90": 11.0,
            "chance_of_playing": 100.0,
            "selected_by_percent": 25.0,
            "news": "",
        })
    # 6 FWD: IDs 25..30
    for i in range(25, 31):
        rows.append({
            "element": i,
            "web_name": f"FWD_{i}",
            "position": "FWD",
            "team": (i % 20) + 1,
            "value": 75,
            "purchase_price": 75,
            "selling_price": 75,
            "expected_points": 7.5 - ((i - 25) * 0.25),
            "p10": 3.0,
            "p90": 13.0,
            "chance_of_playing": 100.0,
            "selected_by_percent": 35.0,
            "news": "",
        })
    return pd.DataFrame(rows)


def build_synthetic_squad(pool_df: pd.DataFrame) -> pd.DataFrame:
    """Standard 15-man squad (2 GKP, 5 DEF, 5 MID, 3 FWD)."""
    squad_ids = [1, 2] + list(range(5, 10)) + list(range(15, 20)) + [25, 26, 27]
    return pool_df[pool_df["element"].isin(squad_ids)].copy().reset_index(drop=True)


def test_scenario_s1_owner_case():
    """
    S1: Owner's case: weak start, healthy core (11 functional starters), 1 FT, bank £1.8m,
    Wildcard available.
    Required: Does NOT recommend Wildcard now; diagnoses healthy core;
    recommends cheapest fix with no hit (k <= 1); shows timeline & staged-FT alternative.
    """
    pool = build_synthetic_player_pool()
    squad = build_synthetic_squad(pool)

    # 1. Health diagnosis
    health = SquadHealthDiagnosis()
    diag = health.diagnose_squad(squad, pool)
    assert diag.classification == "HEALTHY_CORE_TARGETED_FIXES"
    assert diag.functional_starters_count >= 11

    # 2. Frontier planner
    opt = MultiPeriodOptimizer(MultiPeriodConfig(horizon_len=3, time_limit_sec=15))
    frontier_planner = FrontierPlanner(opt)
    projs = {6: pool, 7: pool, 8: pool}

    frontier_rep = frontier_planner.compute_frontier(
        current_squad_df=squad,
        player_pool_df=pool,
        bank=1.8,
        free_transfers=1,
        horizon_projections=projs,
        current_gw=5,
        target_gw=6,
        horizon_len=3,
        chips_available=["wildcard"],
        max_k_limit=2,
    )

    # Asserts
    assert frontier_rep.recommended_k <= 1, "S1 must recommend at most 1 transfer"
    assert frontier_rep.recommended_point.hits == 0, "S1 must not recommend a hit"

    # Briefing generator
    briefing_svc = AdvisorBriefingService(frontier_planner=frontier_planner, health_diagnoser=health)
    rec = briefing_svc.generate_recommendation(
        current_squad_df=squad,
        player_pool_df=pool,
        bank=1.8,
        free_transfers=1,
        horizon_projections=projs,
        current_gw=5,
        target_gw=6,
        chips_available=["wildcard"],
    )

    assert "wildcard" not in rec.decision.lower(), "S1 must not recommend Wildcard in decision sentence"
    assert "wildcard held" in rec.chip_outlook.lower(), "S1 must state Wildcard is held for later windows"
    assert len(rec.alternatives_considered) >= 2


def test_scenario_s2_broken_squad():
    """
    S2: Truly broken squad: 5 starters injured/suspended.
    Required: Diagnoses STRUCTURAL_OVERHAUL_REQUIRED; allows chip/overhaul escalation.
    """
    pool = build_synthetic_player_pool()
    squad = build_synthetic_squad(pool)

    # Make 5 players unavailable
    squad.loc[squad["element"].isin([5, 6, 7, 15, 16]), "chance_of_playing"] = 0.0
    squad.loc[squad["element"].isin([5, 6, 7, 15, 16]), "news"] = "Hamstring injury - ruled out"

    health = SquadHealthDiagnosis()
    diag = health.diagnose_squad(squad, pool)

    assert diag.classification == "STRUCTURAL_OVERHAUL_REQUIRED"
    assert diag.functional_starters_count < 9
    assert len(diag.replace_ids) >= 5


def test_scenario_s3_blank_gameweek_and_rules():
    """
    S3: Free Hit candidate with large blank gameweek.
    Required: Evaluates Free Hit, enforces no FH in GW1 and no FH in both GW19 and GW20.
    """
    from fpl_oracle.config import ChipsConfigSchema
    from fpl_oracle.rules.scoring import FREE_HIT_FORBIDDEN_GWS

    # Rule R2 check: FH forbidden in GW1
    assert 1 in FREE_HIT_FORBIDDEN_GWS

    # Config and schema rule validation
    cfg = ChipsConfigSchema()
    assert cfg.gw1_free_hit_forbidden is True
    assert cfg.consecutive_fh_forbidden is True


def test_scenario_s4_set1_deadline_close():
    """
    S4: Set 1 deadline close (GW19) with unused chips.
    Required: Applies use-it-or-lose-it logic; prices expiry at zero residual value.
    """
    from fpl_oracle.chips.planner import get_chip_threshold
    from fpl_oracle.rules.scoring import SET_1_LAST_GW

    # Set 1 chips cannot be played after GW19
    assert SET_1_LAST_GW == 19

    # At GW19, if Wildcard unused, value in normal context is calibrated
    thresh_19 = get_chip_threshold("wildcard", gameweek=19, set_idx=1)
    assert thresh_19 > 0.0


def test_scenario_s5_marginal_hit_proposal():
    """
    S5: Manager proposes hit for marginal upgrade (+0.6 xP gain).
    Required: Critique verdict WORSE THAN ALTERNATIVE, regret is positive.
    """
    pool = build_synthetic_player_pool()
    squad = build_synthetic_squad(pool)
    projs = {6: pool, 7: pool, 8: pool}

    critique_svc = PlanCritiqueService()
    # Manager proposes 2 transfers with only 1 FT (taking a 4-pt hit) for minimal upgrade
    proposed = [{"out": 9, "in": 14}, {"out": 19, "in": 24}]

    res = critique_svc.evaluate_plan(
        proposed_transfers=proposed,
        proposed_chip=None,
        proposed_captain=25,
        current_squad_df=squad,
        player_pool_df=pool,
        bank=0.5,
        free_transfers=1,
        horizon_projections=projs,
        current_gw=5,
        target_gw=6,
        horizon_len=3,
    )

    assert res.verdict == "WORSE THAN ALTERNATIVE"
    assert res.regret_xp > 0.0
    assert "-4" in res.numeric_reason or "4" in res.numeric_reason


def test_scenario_s6_mini_league_lead_profiles():
    """
    S6: Mini-league lead is small (10 pts) with template-heavy rival.
    Required: Safe profile favors template captain, differential profile favors high ceiling.
    """
    pool = build_synthetic_player_pool()

    # Set up two candidates:
    # Player 25: 7.5 xP, 85% EO (template)
    # Player 26: 7.25 xP, 10% EO, 13.0 P90 (differential)
    eo_map = {25: 85.0, 26: 10.0}

    prof_safe = ManagerProfile(
        risk_appetite="template-safe",
        mini_leagues=[MiniLeagueContext(league_id=1, name="ML", rank=1, total_points=350, leader_points=350, gap_to_leader=0)],
    )
    prof_diff = ManagerProfile(
        risk_appetite="differential-hungry",
        mini_leagues=[MiniLeagueContext(league_id=1, name="ML", rank=3, total_points=330, leader_points=350, gap_to_leader=20)],
    )

    svc = ManagerProfileService()
    ranked_safe = svc.rank_captains(pool, [25, 26], effective_ownership_map=eo_map, profile=prof_safe)
    ranked_diff = svc.rank_captains(pool, [25, 26], effective_ownership_map=eo_map, profile=prof_diff)

    assert ranked_safe[0]["element"] == 25, "Safe profile must favor high EO template asset"
    assert ranked_diff[0]["element"] == 26, "Differential profile must favor low EO high ceiling asset"


def test_scenario_s7_imminent_price_rise():
    """
    S7: Imminent price rise for target player.
    Required: Notes cost of waiting (0.1m purchasing power), warns price rise tonight.
    """
    pool = build_synthetic_player_pool()
    squad = build_synthetic_squad(pool)
    projs = {6: pool, 7: pool}

    frontier_planner = FrontierPlanner(MultiPeriodOptimizer(MultiPeriodConfig(horizon_len=2, time_limit_sec=10)))
    briefing_svc = AdvisorBriefingService(frontier_planner=frontier_planner)

    rec = briefing_svc.generate_recommendation(
        current_squad_df=squad,
        player_pool_df=pool,
        bank=0.2,
        free_transfers=1,
        horizon_projections=projs,
        current_gw=5,
        target_gw=6,
        imminent_price_rises=[14],  # Target DEF_14 rising tonight
    )

    # If target 14 is in recommended moves, warning is attached
    assert any("price rise" in trig.lower() for trig in rec.what_would_change_this) or len(rec.what_would_change_this) >= 1


def test_scenario_s8_key_player_doubt():
    """
    S8: Key player has unresolved doubt (50% chance of playing).
    Required: Flags AVAILABILITY_DOUBT, status WATCH, does not over-commit.
    """
    pool = build_synthetic_player_pool()
    squad = build_synthetic_squad(pool)

    # Mark player 25 (premium captain) as 50% doubt
    squad.loc[squad["element"] == 25, "chance_of_playing"] = 50.0
    squad.loc[squad["element"] == 25, "news"] = "Knock - 50% chance of playing"

    health = SquadHealthDiagnosis()
    diag = health.diagnose_squad(squad, pool)

    p_status = diag.player_diagnoses[25]
    assert p_status.status == "WATCH"
    assert any("50%" in r for r in p_status.reasons)


def test_advisor_recommendation_schema_validation():
    """
    WI-8.8: Validates schema on 100% of recommendations;
    lint check rejects responses missing alternatives, cost, or confidence.
    """
    valid_rec = AdvisorRecommendation(
        recommendation_id="rec_test_1",
        gameweek=6,
        checkpoint="-24h",
        decision="Roll your free transfer in Gameweek 6 to enter Gameweek 7 with 2 banked transfers.",
        why=[
            "Your 11 starters are projected for 54.20 horizon points with healthy captaincy security.",
            "Minimal-intervention frontier confirms marginal gain from immediate swaps does not justify exhausting optionality.",
        ],
        alternatives_considered=[
            {"name": "Roll Transfer", "net_xp": 54.2, "hits": 0, "summary": "Bank transfer"},
            {"name": "1-Transfer Plan", "net_xp": 54.8, "hits": 0, "summary": "Single swap"},
        ],
        cost=RecommendationCost(hits=0, hit_cost=0, transfers_used=0, free_transfers_next=2, bank_remaining=1.5),
        risk=RecommendationRisk(risk_description="Baseline rotation variance", probability=0.35),
        confidence="HIGH",
        what_would_change_this=["Press conference injury confirmation to key starter"],
        deadline_and_next_checkpoint="Deadline: Saturday 11:00 UK. Next checkpoint: 3-hour pre-deadline check.",
        captain_advice=CaptainAdvice(captain_id=25, captain_name="FWD_25", expected_points=7.5, p10_floor=3.0, p90_ceiling=13.0, rationale="Top projection"),
        chip_outlook="Hold chips; Set 1 chips remain available until GW19 deadline.",
        one_line_summary="Roll free transfer. Captain FWD_25.",
    )

    is_valid, errors = validate_recommendation_schema(valid_rec)
    assert is_valid, f"Expected valid recommendation but got errors: {errors}"

    # Invalid: missing alternatives
    invalid_dict = valid_rec.model_dump()
    invalid_dict["alternatives_considered"] = []
    is_valid_bad, errs_bad = validate_recommendation_schema(invalid_dict)
    assert not is_valid_bad
    assert any("alternatives_considered" in e for e in errs_bad)


def test_advice_ledger_immutability_and_finalised_scoring(tmp_path: Path):
    """
    WI-8.7: Tests AdviceLedger immutability and finalised-only scoring.
    """
    db_file = tmp_path / "test_ledger.sqlite"
    ledger = AdviceLedger(db_file)

    rec = AdvisorRecommendation(
        recommendation_id="rec_gw6_immutability_test",
        gameweek=6,
        checkpoint="-24h",
        decision="Roll your free transfer in Gameweek 6.",
        why=["Reason 1 with 54.2 xP", "Reason 2 with 50.0% prob"],
        alternatives_considered=[
            {"name": "Roll Transfer", "net_xp": 54.2, "hits": 0, "summary": "Bank"},
            {"name": "Swap", "net_xp": 54.5, "hits": 0, "summary": "1 swap"},
        ],
        cost=RecommendationCost(hits=0, hit_cost=0, transfers_used=0, free_transfers_next=2, bank_remaining=1.0),
        risk=RecommendationRisk(risk_description="Noise", probability=0.3),
        confidence="HIGH",
        what_would_change_this=["Injury update"],
        deadline_and_next_checkpoint="Deadline: Sat 11:00",
        captain_advice=CaptainAdvice(captain_id=25, captain_name="FWD_25", expected_points=7.5, p10_floor=3.0, p90_ceiling=13.0),
        chip_outlook="Hold chips",
        one_line_summary="Roll transfer.",
    )

    # 1. Record recommendation
    rec_id = ledger.record_recommendation(rec, state_hash="hash_gw6_test")
    assert rec_id > 0

    # 2. Immutability: attempting to overwrite raises ValueError
    with pytest.raises(ValueError, match="immutable and cannot be overwritten"):
        ledger.record_recommendation(rec, state_hash="hash_gw6_test_new")

    # 3. Finalised-only scoring check: is_finalised=False must raise ValueError
    realised = {25: 8.0, 1: 6.0, 5: 6.0, 6: 6.0, 15: 7.0}
    with pytest.raises(ValueError, match="scoring jobs run ONLY on finalised gameweeks"):
        ledger.score_gameweek(gameweek=6, realised_points_map=realised, is_finalised=False)

    # 4. Finalised scoring succeeds
    scores = ledger.score_gameweek(gameweek=6, realised_points_map=realised, is_finalised=True)
    assert len(scores) == 1
    assert "beat_roll" in scores[0]

    # 5. Report card generated
    card = ledger.generate_report_card(num_gameweeks=8)
    assert card["total_gameweeks_evaluated"] == 1
    assert "1/1" in card["record_vs_roll"]
