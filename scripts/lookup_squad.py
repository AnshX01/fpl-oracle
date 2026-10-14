from fpl_oracle.data.historical import historical_manager

df = historical_manager.ensure_dataset_ready()
s26 = df[df["season"] == "2026-27"]
gw5_data = s26[s26["round"] == 5]
gw6_data = s26[s26["round"] == 6]

squad_eids = [496, 8, 418, 87, 427, 154, 368, 286, 124, 411, 346, 109, 165, 10, 259]
print("Squad Entering GW6 (from GW5):")
for eid in squad_eids:
    row = gw5_data[gw5_data["element"] == eid]
    if not row.empty:
        r = row.iloc[0]
        print(f"ID {eid:3d}: {r['name']:25s} | Pos: {r['position']} | Team: {r['team']} | GW5 Val: {r['value']}")
    else:
        row_all = s26[s26["element"] == eid]
        if not row_all.empty:
            r = row_all.iloc[-1]
            print(f"ID {eid:3d}: {r['name']:25s} | Pos: {r['position']} | Team: {r['team']} | Val: {r['value']}")
        else:
            print(f"ID {eid:3d}: Not found in 2026-27")

print("\nTransfers made in GW6 by user:")
r_out = s26[s26["element"] == 427].iloc[0]
r_in = s26[s26["element"] == 12].iloc[0]
print(f"OUT: ID 427: {r_out['name']} ({r_out['position']}) | Value: {r_out['value']}")
print(f"IN:  ID  12: {r_in['name']} ({r_in['position']}) | Value: {r_in['value']}")
