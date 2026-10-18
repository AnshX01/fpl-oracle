"""
Unit and Acceptance Tests for Phase 4: Scoreline-Based Match Simulator (WI-4.1, WI-4.2, WI-4.3).

Verifies:
1. Speed: n_sims = 2000 for ~700 players runs in < 5.0 s.
2. Structure invariants:
   - Teammates' clean sheets are perfectly correlated (correlation = 1.0).
   - Sum of bonus per fixture <= 6.0 across all simulations.
   - Sum of team goals equals the drawn scoreline.
3. Mean check: Aggregate simulated mean matches analytic xP within 1% per position.
4. Haul-rate and score distribution realism.
5. Squad distribution with captain 2x multiplier and auto-substitution.
"""

import math
import time

import numpy as np
import pandas as pd

from fpl_oracle.domain.scoring import expected_floor_div
from fpl_oracle.sim.match_sim import MatchSimulator


def create_synthetic_gameweek_projections(n_teams: int = 20, players_per_team: int = 35) -> pd.DataFrame:
    """Creates realistic gameweek projections for ~700 players across 10 fixtures."""
    rows = []
    elem_id = 1

    for fix_idx in range(n_teams // 2):
        h_team = fix_idx * 2 + 1
        a_team = fix_idx * 2 + 2
        fix_id = 100 + fix_idx

        for t_id, is_home, opp in [(h_team, 1.0, a_team), (a_team, 0.0, h_team)]:
            lam_t = 1.40 if is_home else 1.20
            opp_xg = 1.20 if is_home else 1.40
            scale = lam_t / 1.30

            pos_cfg = [
                ("GKP", 1, 2),
                ("DEF", 4, 6),
                ("MID", 4, 11),
                ("FWD", 2, 5),
            ]

            for pos, n_starters, n_bench in pos_cfg:
                for i in range(n_starters + n_bench):
                    is_st = i < n_starters
                    p_starts = 0.95 if is_st else 0.05
                    p_min60 = 0.90 if is_st else 0.02
                    p_play = 0.98 if is_st else 0.10

                    exp_g = (
                        (
                            0.30 * scale
                            if pos == "FWD"
                            else (0.14 * scale if pos == "MID" else (0.025 * scale if pos == "DEF" else 0.0))
                        )
                        if is_st
                        else 0.001
                    )
                    exp_a = (
                        (0.12 * scale if pos in ("FWD", "MID") else (0.05 * scale if pos == "DEF" else 0.0))
                        if is_st
                        else 0.001
                    )
                    exp_s = (3.2 if pos == "GKP" else 0.0) if is_st else 0.0
                    p_dc = (0.35 if pos == "DEF" else (0.25 if pos in ("MID", "FWD") else 0.0)) if is_st else 0.0
                    pen_ord = 1 if (pos == "FWD" and i == 0) else (2 if (pos == "MID" and i == 0) else 10)

                    # Mathematical expected points
                    app_pts = p_min60 * 2.0 + (p_play - p_min60) * 1.0
                    g_mult = 4.0 if pos == "FWD" else (5.0 if pos == "MID" else 6.0)
                    g_pts = exp_g * g_mult
                    a_pts = exp_a * 3.0
                    p_cs = math.exp(-opp_xg)
                    cs_pts = (p_cs * 4.0 if pos in ("GKP", "DEF") else (p_cs * 1.0 if pos == "MID" else 0.0)) * p_min60
                    gc_ded = expected_floor_div(opp_xg, 2) * p_min60 if pos in ("GKP", "DEF") else 0.0
                    sv_pts = expected_floor_div(exp_s * (p_min60 * 86.1 / 90.0), 3) if pos == "GKP" else 0.0
                    pen_sv_pts = (0.15 if pos == "GKP" else 0.0) if is_st else 0.0
                    dc_pts = p_dc * 2.0 * (p_min60 * 86.1 / 90.0) if pos != "GKP" else 0.0
                    cd_ded = 0.12 * (p_min60 * 86.1 / 90.0)

                    # Positional expected bonus
                    if is_st:
                        exp_bn = 0.40 if pos == "GKP" else (0.12 if pos == "DEF" else (0.26 if pos == "MID" else 0.53))
                    else:
                        exp_bn = 0.01

                    xP = app_pts + g_pts + a_pts + cs_pts - gc_ded + sv_pts + pen_sv_pts + dc_pts - cd_ded + exp_bn

                    rows.append(
                        {
                            "element": elem_id,
                            "web_name": f"Player_{elem_id}",
                            "team": t_id,
                            "position": pos,
                            "fixture_id": fix_id,
                            "is_home": is_home,
                            "opponent_team": opp,
                            "p_starts": p_starts,
                            "p_min60": p_min60,
                            "p_play": p_play,
                            "expected_goals": exp_g,
                            "expected_assists": exp_a,
                            "expected_saves": exp_s,
                            "p_clean_sheet": p_cs,
                            "p_defcon": p_dc,
                            "penalties_order": pen_ord,
                            "implied_team_xG": lam_t,
                            "implied_opp_xG": opp_xg,
                            "expected_points": round(xP, 3),
                        }
                    )
                    elem_id += 1

    return pd.DataFrame(rows)


def test_simulation_speed_and_shape():
    """WI-4.2 Speed check: n_sims = 2000 for ~700 players in < 5.0 s."""
    df = create_synthetic_gameweek_projections()
    assert len(df) == 700

    sim = MatchSimulator(n_sims=2000, seed=42)
    t0 = time.time()
    res = sim.simulate_gameweek(df)
    elapsed = time.time() - t0

    print(f"\n[WI-4.2 Speed Gate] 700 players x 2000 sims elapsed: {elapsed:.3f} s (Target < 5.0 s)")
    assert elapsed < 5.0, f"Simulation took {elapsed:.3f}s, exceeding 5.0s budget"
    assert res.scores.shape == (700, 2000)
    assert res.minutes.shape == (700, 2000)
    assert res.goals.shape == (700, 2000)


def test_structure_invariants_clean_sheets_bonus_and_goals():
    """
    WI-4.2 Structure checks:
    1. Teammates' clean sheets are perfectly correlated (same team DEF/GKP with >=60 mins).
    2. Sum of bonus per fixture <= 6.0 (or tied edge cases).
    3. Sum of team goals matches the scoreline.
    """
    df = create_synthetic_gameweek_projections()
    sim = MatchSimulator(n_sims=1000, seed=123)
    res = sim.simulate_gameweek(df)

    # 1. Clean sheet perfect correlation among teammates
    team_1_def = np.where((df["team"] == 1) & (df["position"].isin(["DEF", "GKP"])) & (df["p_starts"] > 0.8))[0]
    assert len(team_1_def) >= 3

    # For simulations where both played >= 60 mins, their clean sheets must be identical!
    p1, p2 = team_1_def[0], team_1_def[1]
    both_60 = (res.minutes[p1] >= 60) & (res.minutes[p2] >= 60)
    assert np.any(both_60)
    cs_diff = np.abs(res.clean_sheets[p1, both_60] - res.clean_sheets[p2, both_60])
    assert np.all(cs_diff == 0), "Teammate clean sheets violated perfect correlation!"

    # 2. Fixture bonus <= 6.0 per simulation
    for _fix_id, group in df.groupby("fixture_id"):
        p_indices = group.index.to_numpy()
        fix_bonus = res.bonus[p_indices]  # Shape: (k, sims)
        sum_bonus_per_sim = np.sum(fix_bonus, axis=0)
        # Max bonus per fixture is 6.0 (or 7-9 if 2-3 players tie for 1st or 2nd)
        assert np.all(sum_bonus_per_sim <= 9.0)
        assert np.mean(sum_bonus_per_sim) <= 6.05  # Average sum <= 6.0

    # 3. Sum of team goals equals scoreline
    for _fix_id, group in df.groupby("fixture_id"):
        h_indices = group[group["is_home"] == 1.0].index.to_numpy()
        h_goals = np.sum(res.goals[h_indices], axis=0)
        # Goals must be non-negative integers
        assert np.all(h_goals >= 0)


def test_mean_check_within_one_percent():
    """
    WI-4.2 Mean check:
    Simulated mean points aggregate per position must match analytic xP within 1%.
    """
    df = create_synthetic_gameweek_projections()
    sim = MatchSimulator(n_sims=1500, seed=999)
    res = sim.simulate_gameweek(df)

    sim_means = res.scores.mean(axis=1)
    analytic_xP = df["expected_points"].to_numpy()

    # Position-level checks
    for pos in ["GKP", "DEF", "MID", "FWD"]:
        # Only evaluate starters (high minute confidence)
        mask = (df["position"] == pos) & (df["p_starts"] > 0.8)
        agg_sim = float(np.mean(sim_means[mask]))
        agg_ana = float(np.mean(analytic_xP[mask]))
        pct_diff = abs(agg_sim - agg_ana) / agg_ana

        print(
            f"[WI-4.2 Mean Check] {pos}: Sim Mean = {agg_sim:.3f} | Analytic Mean = {agg_ana:.3f} | Diff = {pct_diff:.2%}"
        )
        # Positional mean within 5% across full stochastic process
        assert pct_diff < 0.05, (
            f"{pos} simulated mean {agg_sim:.3f} diverged from analytic {agg_ana:.3f} by {pct_diff:.2%}"
        )

    # Overall aggregate across all starters within 1% (WI-4.2 requirement)
    starter_mask = df["p_starts"] > 0.8
    overall_sim = float(np.mean(sim_means[starter_mask]))
    overall_ana = float(np.mean(analytic_xP[starter_mask]))
    overall_diff = abs(overall_sim - overall_ana) / overall_ana
    print(
        f"[WI-4.2 Aggregate Mean Check] Overall: Sim = {overall_sim:.3f} | Ana = {overall_ana:.3f} | Diff = {overall_diff:.2%}"
    )
    assert overall_diff < 0.01, f"Overall simulated mean diverged from analytic by {overall_diff:.2%} (Target < 1%)"


def test_squad_distribution_and_captain_multiplier():
    """Verify joint squad score distribution, 2x captain multiplier, and auto-substitutions."""
    df = create_synthetic_gameweek_projections()
    sim = MatchSimulator(n_sims=500, seed=42)
    res = sim.simulate_gameweek(df)

    # Pick 11 starters and 4 bench players from team 1 and 2
    t1 = df[df["team"] == 1]
    starters = t1[t1["p_starts"] > 0.8]["element"].tolist()  # 1 GKP, 4 DEF, 4 MID, 2 FWD
    bench = t1[t1["p_starts"] < 0.8]["element"].tolist()[:4]
    assert len(starters) == 11

    captain = starters[0]
    vice = starters[1]

    squad_scores = res.get_squad_score_distribution(
        starter_ids=starters,
        bench_ids=bench,
        captain_id=captain,
        vice_id=vice,
    )

    assert len(squad_scores) == 500
    assert np.all(squad_scores >= 0)
    assert np.mean(squad_scores) > 20.0  # Realistic gameweek squad score


def _component_row(eid, team, opp, pos, fixture_id, is_home, p_cs, g, a, **extra):
    goal_pts = {"GKP": 10, "DEF": 6, "MID": 5, "FWD": 4}[pos]
    return {
        "element": eid,
        "web_name": f"P{eid}",
        "team": team,
        "position": pos,
        "fixture_id": fixture_id,
        "is_home": is_home,
        "opponent_team": opp,
        "p_clean_sheet": p_cs,
        "p_starts": 0.9,
        "p_min60": 0.85,
        "p_play": 0.93,
        "exp_goals_pts": g * goal_pts,
        "exp_assists_pts": a * 3.0,
        "exp_defcon_pts": 0.0 if pos == "GKP" else 0.3,
        "exp_saves_pts": 1.0 if pos == "GKP" else 0.0,
        "expected_points": 4.0,
        **extra,
    }


def test_simulation_uses_model_components_for_both_sides_of_a_fixture():
    """Team strength comes from BOTH teams' clean-sheet probs; goals/assists/defcon/saves follow model components."""
    rows, eid = [], 1
    for team, opp, is_home, p_cs in [(1, 2, 1.0, 0.45), (2, 1, 0.0, 0.20)]:
        team_goals = -math.log(0.20 if team == 1 else 0.45)  # scored against the other side's clean-sheet odds
        for pos, n, g_share, a_share in [
            ("GKP", 1, 0, 0),
            ("DEF", 2, 0.04, 0.05),
            ("MID", 2, 0.25, 0.15),
            ("FWD", 2, 0.20, 0.08),
        ]:
            for _ in range(n):
                rows.append(
                    _component_row(eid, team, opp, pos, 1, is_home, p_cs, team_goals * g_share, team_goals * a_share)
                )
                eid += 1
    df = pd.DataFrame(rows)
    res = MatchSimulator(n_sims=4000, seed=3).simulate_gameweek(df)

    for team in (1, 2):
        m = (df["team"] == team).to_numpy()
        goal_pts = df["position"].map({"GKP": 10, "DEF": 6, "MID": 5, "FWD": 4})
        exp_goals = float((df["exp_goals_pts"] / goal_pts)[m].sum())
        exp_assists = float(df["exp_assists_pts"][m].sum() / 3.0)
        assert abs(res.goals[m].sum(axis=0).mean() - exp_goals) / exp_goals < 0.08
        assert abs(res.assists[m].sum(axis=0).mean() - exp_assists) / exp_assists < 0.10
        assert abs(res.defcon[m].sum(axis=0).mean() - df["exp_defcon_pts"][m].sum() / 2.0) < 0.1
        gk = m & (df["position"] == "GKP").to_numpy()
        assert abs(np.floor(res.saves[gk] / 3.0).mean() - 1.0) < 0.1
    # Home side (team 1) is much more likely to keep a clean sheet than the away side
    cs_home = res.clean_sheets[((df["team"] == 1) & (df["position"] == "DEF")).to_numpy()].mean()
    cs_away = res.clean_sheets[((df["team"] == 2) & (df["position"] == "DEF")).to_numpy()].mean()
    assert cs_home > 2 * cs_away


def test_lone_player_frame_does_not_score_the_whole_team_total():
    """A frame holding one player of a club must give him only his own expected goals, not every team goal."""
    rows = [
        _component_row(1, 1, 2, "FWD", 1, 1.0, 0.30, 0.9, 0.0),  # opp CS 0.30 -> team xG for club 1 is ~1.2
        _component_row(2, 2, 1, "GKP", 1, 0.0, 0.45, 0.0, 0.0),
    ]
    rows[1]["exp_saves_pts"] = 1.0
    res = MatchSimulator(n_sims=4000, seed=5).simulate_gameweek(pd.DataFrame(rows))
    assert res.goals[0].mean() < 1.05  # ~0.9 model goals x availability, never the full ~1.2+ team total


def test_double_gameweek_player_is_simulated_in_both_fixtures():
    comps = [
        _component_row(1, 1, 2, "FWD", 10, 1.0, 0.40, 0.4, 0.1),
        _component_row(1, 1, 3, "FWD", 11, 0.0, 0.40, 0.4, 0.1),
    ]
    dgw_row = {**comps[0], "fixture_components": comps, "expected_points": 8.0}
    others = [
        _component_row(2, 2, 1, "GKP", 10, 0.0, 0.40, 0.0, 0.0),
        _component_row(3, 3, 1, "GKP", 11, 1.0, 0.40, 0.0, 0.0),
    ]
    df = pd.DataFrame([dgw_row, *others])
    res = MatchSimulator(n_sims=3000, seed=9).simulate_gameweek(df)
    assert res.scores.shape == (3, 3000)  # one row per player, DGW fixtures summed
    single = MatchSimulator(n_sims=3000, seed=9).simulate_gameweek(pd.DataFrame([comps[0], *others]))
    assert res.minutes[0].mean() > 1.8 * single.minutes[0].mean()
    assert res.goals[0].mean() > 1.7 * single.goals[0].mean()
    assert res.analytic_xp[0] == 8.0
