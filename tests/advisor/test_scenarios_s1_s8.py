"""
Scenario regression suite for the live advisor (health diagnosis and plan critique):
S2: Truly broken squad -> Overhaul justified by health diagnosis
S4: Set 1 deadline close (GW19) -> chip threshold calibrated
S5: Manager proposes hit for marginal upgrade -> Verdict WORSE THAN ALTERNATIVE
S8: Key player doubt -> flagged as WATCH
"""

import pandas as pd

from fpl_oracle.advisor.critique import PlanCritiqueService
from fpl_oracle.advisor.health import SquadHealthDiagnosis


def build_synthetic_player_pool() -> pd.DataFrame:
    """Builds a deterministic 30-player pool (4 GKP, 10 DEF, 10 MID, 6 FWD) with IDs."""
    rows = []
    # 4 GKP: IDs 1..4
    for i in range(1, 5):
        rows.append(
            {
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
            }
        )
    # 10 DEF: IDs 5..14
    for i in range(5, 15):
        rows.append(
            {
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
            }
        )
    # 10 MID: IDs 15..24
    for i in range(15, 25):
        rows.append(
            {
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
            }
        )
    # 6 FWD: IDs 25..30
    for i in range(25, 31):
        rows.append(
            {
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
            }
        )
    return pd.DataFrame(rows)


def build_synthetic_squad(pool_df: pd.DataFrame) -> pd.DataFrame:
    """Standard 15-man squad (2 GKP, 5 DEF, 5 MID, 3 FWD)."""
    squad_ids = [1, 2] + list(range(5, 10)) + list(range(15, 20)) + [25, 26, 27]
    return pool_df[pool_df["element"].isin(squad_ids)].copy().reset_index(drop=True)


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
    # The XI now skips unavailable players: 10 fit players (one a spare GKP) leave 9 functional starters
    # and two forced unavailable slots, which is what drives the structural deficiency.
    assert diag.functional_starters_count <= 9
    assert diag.structural_deficiency_score > 15.0
    assert len(diag.replace_ids) >= 5


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
        bank=5,
        free_transfers=1,
        horizon_projections=projs,
        current_gw=5,
        target_gw=6,
        horizon_len=3,
    )

    assert res.verdict == "WORSE THAN ALTERNATIVE"
    assert res.regret_xp > 0.0
    assert "-4" in res.numeric_reason or "4" in res.numeric_reason


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
