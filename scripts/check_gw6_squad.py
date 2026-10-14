from fpl_oracle.data.historical import historical_manager
from fpl_oracle.domain.manager_state import ManagerStateService

calculate_selling_price = ManagerStateService.calculate_selling_price

df = historical_manager.ensure_dataset_ready()
s26 = df[df['season'] == '2026-27']
gw1 = s26[s26['round'] == 1]
gw5 = s26[s26['round'] == 5]

squad_eids = [496, 8, 418, 87, 427, 154, 368, 286, 124, 411, 346, 109, 165, 10, 259]
transfers_in_cost = {87: 45, 154: 97, 165: 77, 286: 51}

total_sell_value = 0
squad_details = []
for eid in squad_eids:
    r5 = gw5[gw5['element'] == eid].iloc[0]
    if eid in transfers_in_cost:
        buy_cost = transfers_in_cost[eid]
    else:
        r1 = gw1[gw1['element'] == eid].iloc[0]
        buy_cost = r1['value']
    current_cost = r5['value']
    sell_price = calculate_selling_price(buy_cost, current_cost)
    total_sell_value += sell_price
    squad_details.append({
        "element": eid,
        "name": r5['name'],
        "position": r5['position'],
        "team": r5['team'],
        "buy_cost": buy_cost,
        "current_cost": current_cost,
        "sell_price": sell_price
    })
    print(f"{r5['name']:28s} | Pos: {r5['position']:3s} | Buy: {buy_cost/10:4.1f} | Curr: {current_cost/10:4.1f} | Sell: {sell_price/10:4.1f}")

print(f"Total squad sell value: {total_sell_value/10:.1f}")
print("Bank: 1.8")
print(f"Total liquid value: {(total_sell_value + 18)/10:.1f}")
