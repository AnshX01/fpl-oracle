"""
Comprehensive test of all FPL Oracle FastAPI server endpoints end-to-end
using httpx AsyncClient with the ASGI app.
"""

import asyncio
import sys
from pathlib import Path

# Configure stdout encoding for Windows UTF-8 support
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import httpx
from fpl_oracle.server.main import app

async def run_tests():
    print("Testing FPL Oracle FastAPI app endpoints end-to-end...\n")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=30.0) as client:
        # 1. Health
        res = await client.get("/api/health")
        print(f"1.  GET /api/health -> Status {res.status_code}")
        assert res.status_code == 200, f"Health check failed: {res.text}"
        data = res.json()
        print(f"    Season: {data.get('season')}, Current GW: {data.get('current_gameweek')}, Next GW: {data.get('next_gameweek')}")

        # 2. Profile GET
        res = await client.get("/api/profile")
        print(f"2.  GET /api/profile -> Status {res.status_code}")
        assert res.status_code == 200
        prof = res.json()
        print(f"    Manager ID: {prof.get('manager_id')}, Risk: {prof.get('risk_preference')}")

        # 3. Profile POST
        res = await client.post("/api/profile", json={"risk_preference": "balanced"})
        print(f"3.  POST /api/profile -> Status {res.status_code}")
        assert res.status_code == 200

        # 4. Squad
        res = await client.get("/api/squad")
        print(f"4.  GET /api/squad -> Status {res.status_code}")
        assert res.status_code == 200
        squad_data = res.json()
        starters = squad_data.get("starters", [])
        bench = squad_data.get("bench", [])
        total_players = len(starters) + len(bench)
        print(f"    Total squad players: {total_players} (Starters: {len(starters)}, Bench: {len(bench)})")
        print(f"    Formation: {squad_data.get('formation')}, Captain: {squad_data.get('captain', {}).get('web_name')}")
        assert total_players == 15, f"Expected 15 players, got {total_players}"

        # 5. Projections
        res = await client.get("/api/projections?horizon=3")
        print(f"5.  GET /api/projections -> Status {res.status_code}")
        assert res.status_code == 200
        proj_data = res.json()
        print(f"    Projected players returned: {len(proj_data.get('players', []))}")
        assert len(proj_data.get("players", [])) > 0

        # 6. Optimize
        res = await client.post("/api/optimize", json={})
        print(f"6.  POST /api/optimize -> Status {res.status_code}")
        assert res.status_code == 200
        opt_data = res.json()
        print(f"    Recommended Plan: {opt_data.get('recommended_plan', {}).get('plan_type')}")
        print(f"    Hit Verdict: {opt_data.get('hit_verdict')}")

        # 7. Chips
        res = await client.get("/api/chips")
        print(f"7.  GET /api/chips -> Status {res.status_code}")
        assert res.status_code == 200
        chips_data = res.json()
        print(f"    Chip plan entries: {len(chips_data.get('chip_plan_table', []))}")
        assert len(chips_data.get("chip_plan_table", [])) > 0

        # 8. League Intel (unconfigured and with ID)
        res = await client.get("/api/league")
        print(f"8a. GET /api/league (unconfigured) -> Status {res.status_code}")
        assert res.status_code == 200
        league_unconf = res.json()
        print(f"    Status: {league_unconf.get('status')}")

        res = await client.get("/api/league?league_id=314")
        print(f"8b. GET /api/league?league_id=314 -> Status {res.status_code}")
        assert res.status_code == 200
        league_data = res.json()
        print(f"    League Name: {league_data.get('league_name')}, Total Teams: {league_data.get('total_teams')}")

        # 9. Briefing
        res = await client.get("/api/briefing")
        print(f"9.  GET /api/briefing -> Status {res.status_code}")
        assert res.status_code == 200
        briefing_data = res.json()
        print(f"    Briefing target GW: {briefing_data.get('target_gameweek')}")
        print(f"    Markdown length: {len(briefing_data.get('markdown', ''))} chars")
        assert len(briefing_data.get("markdown", "")) > 100

        # 10. Price changes
        res = await client.get("/api/price-changes")
        print(f"10. GET /api/price-changes -> Status {res.status_code}")
        assert res.status_code == 200
        pc_data = res.json()
        print(f"    Rises: {len(pc_data.get('rises', []))}, Falls: {len(pc_data.get('falls', []))}")

        # 11. Chat endpoint
        res = await client.post("/api/chat", json={"message": "Who is the best captaincy pick for Gameweek 6?"})
        print(f"11. POST /api/chat -> Status {res.status_code}")
        assert res.status_code == 200
        chat_data = res.json()
        resp_text = chat_data.get("response", "")
        # Safe ASCII preview to prevent any terminal encoding traps
        print(f"    Chat response preview: {resp_text[:80].encode('ascii', 'backslashreplace').decode('ascii')}...")
        assert len(resp_text) > 20

        # 12. Web UI root
        res = await client.get("/")
        print(f"12. GET / (Web UI root) -> Status {res.status_code}")
        assert res.status_code == 200
        assert "<title>FPL Oracle" in res.text

    print("\n========================================================")
    print("ALL 12 SERVER ENDPOINTS VERIFIED AND RETURNING 200 OK!")
    print("========================================================")

if __name__ == "__main__":
    asyncio.run(run_tests())
