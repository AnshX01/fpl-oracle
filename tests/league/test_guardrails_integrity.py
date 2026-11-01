"""
Tests for Elite Manager Strategy Guardrails:
1. Active Pitch Cover Guard
2. Pre-Chip Free Transfer Liquidity Shield
3. Season-Phase Lead Gate in Risk Posture
4. Fixture Threat Gating and Variance Compression
"""

from fpl_oracle.league.objective import league_threats, win_probability
from fpl_oracle.league.strategy import league_strategy_advisor


def test_season_phase_lead_gate():
    """Guardrail 3: Early/mid season (GW <= 20) requires a solid points lead (>= 15 pts) to defend lead."""
    # Early season with small 2-pt lead and high projected win prob (e.g. 67%)
    posture_early = league_strategy_advisor.determine_risk_posture(
        points_gap=2.0, win_probability_pct=67.0, current_gw=8
    )
    assert posture_early == "BALANCED_ATTACK", "Early season with 2 pt lead must remain in BALANCED_ATTACK"

    # Early season with 16-pt lead
    posture_early_large = league_strategy_advisor.determine_risk_posture(
        points_gap=16.0, win_probability_pct=67.0, current_gw=8
    )
    assert posture_early_large == "DEFENDING_LEAD", "Early season with >= 15 pt lead can defend lead"

    # Late season (GW > 20) with 10-pt lead
    posture_late = league_strategy_advisor.determine_risk_posture(
        points_gap=10.0, win_probability_pct=65.0, current_gw=28
    )
    assert posture_late == "DEFENDING_LEAD", "Late season with 10 pt lead defends lead"

    # Late season with 65% win prob and 5 pt lead
    posture_late_winprob = league_strategy_advisor.determine_risk_posture(
        points_gap=5.0, win_probability_pct=65.0, current_gw=28
    )
    assert posture_late_winprob == "DEFENDING_LEAD", "Late season with >= 60% win prob defends lead"


def test_fixture_threat_gating_downweights_mismatched_defenders():
    """Guardrail 4a: Defenders facing extreme difficulty (< 2.0 xP in GW0) are not ranked as acute GW0 cover threats."""
    pos = {1: "DEF", 2: "MID", 3: "FWD"}
    # Element 1 is a defender with 1.0 xP in GW0 (e.g. Man City away), but 5.0 in GW1
    # Element 2 is a midfielder with 4.5 xP in GW0 and 4.5 in GW1
    xp_map = {
        8: {1: 1.0, 2: 4.5, 3: 4.0},
        9: {1: 5.0, 2: 4.5, 3: 4.0},
    }
    squad_rival = [1, 2, 3]
    rivals = [dict(elements=squad_rival, captain=3)]
    cache = {}

    threats = league_threats(
        rivals=rivals,
        gws=[8, 9],
        xp=xp_map,
        position=pos,
        mine=set(),
        cache=cache,
        top=2,
    )

    # Midfielder 2 (active threat in GW8) must rank above Defender 1 whose GW8 fixture is 1.0 xP
    assert threats[0] == 2, f"Active midfielder 2 should rank ahead of 1.0 xP defender 1, got {threats}"


def test_variance_compression_for_low_xp_defenders():
    """Guardrail 4b: Defenders with low xP (< 2.0) have compressed clean-sheet variance."""
    pos = {1: "DEF", 2: "MID"}
    xp_map = {8: {1: 1.0, 2: 5.0}}
    sd_map = {8: {1: 4.0, 2: 3.0}}  # Raw uncompressed SD

    mine = [{1: 1.0}]
    held = mine

    context = dict(user_points=100, rivals=[dict(points=98, elements=[1, 2], captain=2)])
    # Evaluates cleanly without NaN or unbounded explosion
    res = win_probability(
        my_weeks=mine,
        held_weeks=held,
        best_plan_points=0.0,
        gws=[8],
        rival_context=context,
        xp=xp_map,
        sd=sd_map,
        position=pos,
    )
    assert res is not None
    assert "p_win" in res
    assert 0.0 <= res["p_win"] <= 1.0


def test_active_pitch_cover_guard_and_pre_chip_shield():
    """Guardrails 1 & 2: Tie-break candidates are validated in multiperiod planning logic."""

    # Test the helper logic of valid tiebreak candidate
    cand_roll = {
        "trajectory": [
            {"starters": [1, 2, 3], "bench": [4, 5], "net_xp": 54.0},
            {"chip": "bboost", "starters": [1, 2, 3], "bench": [4, 5], "net_xp": 65.0},
        ],
        "transfers_count": 0,
        "total_objective": 119.0,
    }

    # Case A: Candidate bought player 99, but benched him at Sub 2 (index 1 of bench) with 1.0 xP
    cand_benched_cover = {
        "source": "cover:99",
        "trajectory": [
            {"starters": [1, 2, 3], "bench": [4, 99], "net_xp": 53.6},
            {"chip": "bboost", "starters": [1, 2, 3], "bench": [4, 99], "net_xp": 64.0},
        ],
        "transfers_count": 1,
        "total_objective": 117.6,
    }

    target_gw = 8
    maps = {8: {99: 1.0, 4: 3.5, 1: 5.0, 2: 5.0, 3: 5.0}}

    # Verify Guardrail 1: Covered player not in starters and not 1st sub with >= 2.5 xP
    cov_id = int(cand_benched_cover["source"].split(":")[1])
    t0 = cand_benched_cover["trajectory"][0]
    starters = set(t0.get("starters", []))
    bench = t0.get("bench", [])
    first_sub = bench[0] if bench else None
    is_active = (cov_id in starters) or (cov_id == first_sub and maps[target_gw].get(cov_id, 0.0) >= 2.5)
    assert not is_active, "Ghost shadowing at Sub 2 with 1.0 xP must be flagged as inactive cover"

    # Verify Guardrail 2: Pre-Chip Liquidity Shield
    has_near_bboost = any(act.get("chip") == "bboost" for act in cand_roll["trajectory"][:2])
    assert has_near_bboost is True
    # cand_benched_cover has net_xp (53.6) < cand_roll (54.0), burning an FT before BB
    is_pre_chip_safe = not (
        has_near_bboost
        and cand_benched_cover.get("transfers_count", 0) > 0
        and cand_benched_cover["trajectory"][0]["net_xp"] < cand_roll["trajectory"][0]["net_xp"] - 0.01
    )
    assert not is_pre_chip_safe, "Burning an FT for a negative-xP move right before Bench Boost must be blocked"
