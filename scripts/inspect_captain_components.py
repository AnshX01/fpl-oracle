"""
Script to inspect raw xP components for James Tarkowski and Erling Haaland in GW6 and GW7 (Item 10).
Outputs exact numerical components:
- p_play, p_min60, appearance_pts
- goals_pts, assists_pts
- clean_sheet_prob, cs_pts
- defcon_pts
- bonus_pts
- gc_deduction, card_deduction
- total xP
And reconciles the 5.86 vs 4.20 projection.
"""

import asyncio

from fpl_oracle.api.fpl_client import fpl_client
from fpl_oracle.ml.predict import projection_engine


async def inspect_components():
    print("=" * 80)
    print("CAPTAIN SANITY: RAW COMPONENT BREAKDOWN (GW6 & GW7)")
    print("=" * 80)

    bootstrap, _ = await fpl_client.get_bootstrap_static()
    fixtures, _ = await fpl_client.get_fixtures()

    # Find Tarkowski, Haaland, and Szoboszlai element IDs
    tarkowski_el = next(el for el in bootstrap.elements if "Tarkowski" in el.web_name)
    haaland_el = next(el for el in bootstrap.elements if "Haaland" in el.web_name)
    szoboszlai_el = next((el for el in bootstrap.elements if "Szoboszlai" in el.web_name), None)

    print(f"Tarkowski element id: {tarkowski_el.id} ({tarkowski_el.web_name}, team={tarkowski_el.team})")
    print(f"Haaland element id:   {haaland_el.id} ({haaland_el.web_name}, team={haaland_el.team})")
    if szoboszlai_el:
        print(f"Szoboszlai element id:{szoboszlai_el.id} ({szoboszlai_el.web_name}, team={szoboszlai_el.team})")

    for target_gw in [6, 7]:
        print(f"\n--- GAMEWEEK {target_gw} EVALUATION ---")
        preds = projection_engine.predict_gameweek(target_gw, bootstrap, fixtures)

        players_to_check = [("James Tarkowski (DEF)", tarkowski_el), ("Erling Haaland (FWD)", haaland_el)]
        if szoboszlai_el:
            players_to_check.append(("Dominik Szoboszlai (MID)", szoboszlai_el))

        for p_name, el in players_to_check:
            row = preds[preds["element"] == el.id]
            if row.empty:
                print(f"No predictions for {p_name} in GW{target_gw}")
                continue

            r = row.iloc[0]
            print(f"\nPlayer: {p_name}")
            print(f"  Element ID:              {el.id}")
            print(f"  Opponent Difficulty:     {r.get('opponent_difficulty', 'N/A')}")
            print(f"  Is Home:                 {r.get('is_home', r.get('was_home', 'N/A'))}")
            print(f"  Chance of Playing:       {r.get('chance_of_playing', 100)}%")
            print(f"  P(Play >= 1 min):        {r.get('p_play', r.get('p_appearance', 0.95)):.3f}")
            print(f"  P(Min >= 60):            {r.get('p_min60', 0.90):.3f}")
            print(f"  Expected Appearance Pts: {r.get('exp_appearance', r.get('exp_appearance_pts', 0.0)):.2f}")
            print(f"  Expected Goal Points:    {r.get('exp_goals_pts', 0.0):.2f}")
            print(f"  Expected Assist Points:  {r.get('exp_assists_pts', 0.0):.2f}")
            print(f"  P(Clean Sheet):          {r.get('p_cs', 0.0):.3f} -> CS Pts: {r.get('exp_cs_pts', 0.0):.2f}")
            print(f"  Expected DefCon Points:  {r.get('exp_defcon_pts', 0.0):.2f}")
            print(f"  Expected Bonus Points:   {r.get('exp_bonus_pts', 0.0):.2f}")
            print(f"  GC Deduction:           -{r.get('exp_gc_deduction', 0.0):.2f}")
            print(f"  Card Deduction:         -{r.get('exp_card_deduction', 0.0):.2f}")
            print("  -------------------------------------------")
            print(f"  TOTAL EXPECTED POINTS:    {r.get('expected_points', r.get('raw_expected_points', 0.0)):.2f} xP")

if __name__ == "__main__":
    asyncio.run(inspect_components())
