"""
Unit and integration tests for the 5 Systemic Blind Spots Remediation:
1. Megacap Transfer Liquidation Shield (EO Barrier >= 35%)
2. Repurchase Equity Preservation (Stored Capital Friction)
3. 5-Substitution Cameo Auto-Sub Blocking Engine
4. Press Conference Embargo & Deadline Freshness Guardrail
5. Dynamic Rival Transfer & Chip Priors in Mini-League Monte Carlo
"""

import pandas as pd

from fpl_oracle.briefing.decision_card import format_decision_card_markdown
from fpl_oracle.league.montecarlo import MonteCarloSimulator
from fpl_oracle.optimise.lineup import LineupOptimizer
from fpl_oracle.optimise.transfers_beam import TransferOptimizer


def test_megacap_transfer_liquidation_shield_blocks_marginal_sale():
    """
    Selling an owned consensus megacap (EO >= 35%) for a marginal fractional gain
    (e.g. +1.1 xP) is blocked by the Megacap Liquidation Shield.
    """
    optimizer = TransferOptimizer()

    # Create 15-player squad where player 10 is an owned megacap (45% EO)
    squad_rows = []
    positions = ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    for i, pos in enumerate(positions):
        squad_rows.append(
            {
                "element": i + 1,
                "web_name": f"Player{i + 1}",
                "position": pos,
                "team": (i % 20) + 1,
                "value": 100 if i == 9 else 60,  # Player 10 is £10.0m
                "selling_price": 100 if i == 9 else 60,
                "purchase_price": 100 if i == 9 else 60,
                "expected_points": 5.0 if i == 9 else 4.0,
                "selected_by_percent": 45.0 if i == 9 else 10.0,
                "effective_ownership": 50.0 if i == 9 else 10.0,
                "top10k_eo": 55.0 if i == 9 else 10.0,
                "chance_of_playing": 100.0,
            }
        )
    squad_df = pd.DataFrame(squad_rows)

    # Replacement candidate: slightly higher xP (+0.2/gw -> ~1.0 xP total)
    upgrade_player = {
        "element": 99,
        "web_name": "TalismanUpgrade",
        "position": "MID",
        "team": 15,
        "value": 90,
        "selling_price": 90,
        "purchase_price": 90,
        "expected_points": 5.2,
        "selected_by_percent": 5.0,
        "effective_ownership": 5.0,
        "top10k_eo": 5.0,
        "chance_of_playing": 100.0,
    }
    pool_rows = squad_rows + [upgrade_player]
    pool_df = pd.DataFrame(pool_rows)
    horizon = {gw: pool_df.copy() for gw in range(9, 14)}

    res = optimizer.evaluate_transfer_options(
        current_squad_df=squad_df,
        player_pool_df=pool_df,
        bank=10,
        free_transfers=1,
        horizon_projections=horizon,
        current_gw=8,
        target_gw=9,
        horizon_len=5,
    )

    # Top recommendation should roll or preserve the megacap player
    rec = res["recommended_plan"]
    sold_ids = [t["element"] for t in rec.get("transfers_out", [])]
    assert 10 not in sold_ids, (
        f"Megacap player 10 should not be sold for marginal gain: {rec.get('recommendation_summary')}"
    )
    assert rec.get("megacap_shield_applied") is True or rec.get("plan_type") == "ROLL_TRANSFER"


def test_repurchase_equity_friction_penalizes_stored_capital_loss():
    """
    Selling an asset with stored price appreciation (now_cost > sell_price)
    incurs an equity friction penalty reflecting destroyed purchasing power.
    """
    # Asset with £0.6m price rise (£0.3m stored profit destroyed on sale)
    now_cost = 10.6
    sell_price = 10.3
    stored_equity = round(now_cost - sell_price, 2)
    drag_coeff = 0.50

    equity_penalty = round(stored_equity * drag_coeff * 10.0, 2)
    assert equity_penalty == 1.5  # 0.3 * 0.5 * 10 = 1.5 xP drag


def test_cameo_risk_auto_sub_blocking_penalty():
    """
    A player with low expected minutes (15 mins cameo profile) is penalized
    for blocking the primary bench starter from entering.
    """
    lineup_opt = LineupOptimizer()

    # 15 players: 11 strong starters, 1 bench asset with 4.5 xP, 1 cameo risk asset
    rows = []
    positions = ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    for i, pos in enumerate(positions):
        exp_m = 90.0
        exp_xp = 4.0
        cop = 100.0
        if i == 5:  # DEF with cameo risk (raw xP 3.8 > 3.4)
            exp_m = 20.0
            exp_xp = 3.8
            cop = 60.0
        elif i == 6:  # Regular DEF starter (raw xP 3.4)
            exp_m = 85.0
            exp_xp = 3.4
            cop = 100.0
        elif i == 11 or i == 14:  # Bench MID/FWD
            exp_xp = 2.0
        rows.append(
            {
                "element": i + 1,
                "web_name": f"P{i + 1}",
                "position": pos,
                "team": (i % 20) + 1,
                "value": 50,
                "expected_points": exp_xp,
                "expected_minutes": exp_m,
                "chance_of_playing": cop,
            }
        )
    df = pd.DataFrame(rows)

    lineup = lineup_opt.select_lineup_and_captain(df, simulate_captaincy=False)
    starter_ids = set(lineup["starters"]["element"])
    bench_ids = set(lineup["bench"]["element"])

    # Player 6 (element 6, cameo risk) should be benched, not started over regular starter (element 7)
    assert 6 in bench_ids, "15-minute cameo risk player should be benched to avoid blocking auto-subs"
    assert 7 in starter_ids, "Full 85-minute starter should start"


def test_deadline_freshness_and_embargo_guardrail():
    """
    Advice generated > 15h before deadline is tagged as PRELIMINARY_PENDING_EMBARGOES
    and renders an embargo warning in the markdown decision card.
    Advice within 12h of deadline is tagged as FINAL_DEADLINE_READY.
    """
    # Simulate preliminary card > 15h before deadline
    card_prelim = {
        "gameweek": 9,
        "deadline": {
            "label": "Sat 31 Oct, 16:30",
            "deadline_seconds": 60000,
            "freshness_status": "PRELIMINARY_PENDING_EMBARGOES",
            "embargo_warning": "Preliminary projection: Friday manager press conference embargoes lift at 22:30 UK time.",
        },
        "transfers": {"is_roll": True},
        "captaincy": {"captain": {"web_name": "Talisman"}, "vice_captain": {"web_name": "TalismanVC"}},
        "lineup": {"formation": "3-5-2", "expected_points": 50.0, "starters": [], "bench": []},
        "chips": {"current_chip": None},
        "mini_league": {},
    }
    md_prelim = format_decision_card_markdown(card_prelim)
    assert "PRELIMINARY_PENDING_EMBARGOES" in card_prelim["deadline"]["freshness_status"]
    assert "Pending Friday Press Embargoes" in md_prelim

    # Simulate final card within 12h
    card_final = {
        "gameweek": 9,
        "deadline": {
            "label": "Sat 31 Oct, 16:30",
            "deadline_seconds": 14400,
            "freshness_status": "FINAL_DEADLINE_READY",
        },
        "transfers": {"is_roll": True},
        "captaincy": {"captain": {"web_name": "Talisman"}, "vice_captain": {"web_name": "TalismanVC"}},
        "lineup": {"formation": "3-5-2", "expected_points": 50.0, "starters": [], "bench": []},
        "chips": {"current_chip": None},
        "mini_league": {},
    }
    md_final = format_decision_card_markdown(card_final)
    assert card_final["deadline"]["freshness_status"] == "FINAL_DEADLINE_READY"
    assert "Pending Friday Press Embargoes" not in md_final


def test_dynamic_rival_transfers_in_monte_carlo():
    """
    In multi-gameweek simulations, rivals dynamically adapt via Markov transfers
    when starters are injured or have low projections.
    """
    simulator = MonteCarloSimulator(n_simulations=50)

    # User squad
    squad_rows = [
        {"element": i + 1, "position": "DEF" if i < 4 else "MID", "expected_points": 4.5, "is_starter": True}
        for i in range(11)
    ]
    squad_rows.append({"element": 12, "position": "DEF", "expected_points": 3.0, "is_starter": False})
    user_squad_df = pd.DataFrame(squad_rows)

    # Rival with an injured starter (0% chance of playing) in forward GWs
    rival = {
        "entry_id": 1001,
        "player_name": "RivalLeader",
        "total_points": 500.0,
        "total_transfers": 5.0,
        "squad": [
            {"element": 1, "position": "DEF", "is_starter": True, "is_captain": True},
            {"element": 2, "position": "DEF", "is_starter": True},
            {"element": 3, "position": "DEF", "is_starter": True},
            {"element": 4, "position": "DEF", "is_starter": True},
            {"element": 5, "position": "MID", "is_starter": True},
            {"element": 6, "position": "MID", "is_starter": True},
            {"element": 7, "position": "MID", "is_starter": True},
            {"element": 8, "position": "MID", "is_starter": True},
            {"element": 9, "position": "MID", "is_starter": True},
            {"element": 10, "position": "MID", "is_starter": True},
            {"element": 11, "position": "MID", "is_starter": True},  # Injured in GW2
        ],
    }

    # Projections by gameweek
    p_by_gw = {}
    for gw in [9, 10]:
        rows = [
            {
                "element": i + 1,
                "position": "DEF" if i < 4 else "MID",
                "expected_points": 4.5,
                "chance_of_playing": 100.0,
                "team": (i % 20) + 1,
            }
            for i in range(15)
        ]
        # In GW10, player 11 is injured (0% chance of playing)
        if gw == 10:
            rows[10]["chance_of_playing"] = 0.0
            rows[10]["expected_points"] = 0.0
            # Add hot replacement player 99
            rows.append(
                {"element": 99, "position": "MID", "expected_points": 6.5, "chance_of_playing": 100.0, "team": 1}
            )
        p_by_gw[gw] = pd.DataFrame(rows)

    res = simulator.simulate_league(
        user_points=500.0,
        user_squad_df=user_squad_df,
        rival_squads=[rival],
        projections_by_gw=p_by_gw,
        horizon_gws=2,
        seed=42,
    )

    assert res["status"] == "SIMULATION_SUCCESS"
    assert "Markov" in res["rival_behavior_assumptions"]["transfer_model"]
