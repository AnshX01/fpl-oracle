import json

import httpx

client = httpx.Client(timeout=15.0, headers={"User-Agent": "Mozilla/5.0"})

# 1. Picks for Event 6
r_picks = client.get("https://fantasy.premierleague.com/api/entry/1085141/event/6/picks/")
print("Picks Status:", r_picks.status_code)
picks_data = r_picks.json()
print("Entry history GW6:", json.dumps(picks_data.get("entry_history"), indent=2))
print("Active chip GW6:", picks_data.get("active_chip"))
print("Picks count:", len(picks_data.get("picks", [])))
print("Picks elements:", [p["element"] for p in picks_data.get("picks", [])])

# Picks for Event 5 (to see pre-transfer squad entering GW6)
r_picks5 = client.get("https://fantasy.premierleague.com/api/entry/1085141/event/5/picks/")
print("\nGW5 Picks Status:", r_picks5.status_code)
picks5_data = r_picks5.json()
print("Entry history GW5:", json.dumps(picks5_data.get("entry_history"), indent=2))
print("Active chip GW5:", picks5_data.get("active_chip"))
print("GW5 Picks elements:", [p["element"] for p in picks5_data.get("picks", [])])

# 2. Transfers made in GW6 (via transfers history)
r_transfers = client.get("https://fantasy.premierleague.com/api/entry/1085141/transfers/")
print("\nTransfers Status:", r_transfers.status_code)
if r_transfers.status_code == 200:
    transfers = r_transfers.json()
    gw6_transfers = [t for t in transfers if t.get("event") == 6]
    print(f"GW6 Transfers made: {json.dumps(gw6_transfers, indent=2)}")

# 3. History
r_hist = client.get("https://fantasy.premierleague.com/api/entry/1085141/history/")
print("\nHist Status:", r_hist.status_code)
hist_data = r_hist.json()
for gw in hist_data.get("current", []):
    if gw["event"] in [4, 5, 6]:
        print(
            f"GW{gw['event']}: pts={gw['points']}, total_pts={gw['total_points']}, rank={gw['overall_rank']}, bank={gw['bank']}, val={gw['value']}, transfers={gw['event_transfers']}, cost={gw['event_transfers_cost']}"
        )
print("Chips used in history:", hist_data.get("chips"))

# 4. Standings
r_std = client.get("https://fantasy.premierleague.com/api/leagues-classic/435956/standings/")
print("\nStandings Status:", r_std.status_code)
std_data = r_std.json()
print("League name:", std_data.get("league", {}).get("name"))
for row in std_data.get("standings", {}).get("results", [])[:5]:
    print(f"Rank {row['rank']}: {row['player_name']} ({row['entry_name']}) - {row['total']} pts")
