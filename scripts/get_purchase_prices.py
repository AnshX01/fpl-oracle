
import httpx

client = httpx.Client(timeout=15.0, headers={"User-Agent": "Mozilla/5.0"})
r = client.get("https://fantasy.premierleague.com/api/entry/1085141/transfers/")
transfers = r.json()
print("All transfers made:")
for t in transfers:
    print(f"GW{t['event']}: in={t['element_in']} (cost {t['element_in_cost']}), out={t['element_out']} (cost {t['element_out_cost']})")

r_gw1 = client.get("https://fantasy.premierleague.com/api/entry/1085141/event/1/picks/")
gw1_picks = [p["element"] for p in r_gw1.json()["picks"]]
print("\nGW1 picks:", gw1_picks)
