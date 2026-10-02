"""
Comprehensive test of all FPL Oracle FastAPI server endpoints end-to-end
using httpx AsyncClient with the ASGI app.
Covers:
1. Live network mode (13 endpoints: health, game-state, profile, squad, projections,
   optimize, chips, league unconfigured & live, briefing, price-changes, chat, web UI)
2. Fault injection mode (429 rate limits, 503 downtime, timeouts, schema drift, malformed inputs)
"""

import asyncio
import json
import sys
from pathlib import Path

# Configure stdout encoding for Windows UTF-8 support
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import httpx
from fpl_oracle.server.main import app
from fpl_oracle.api.fpl_client import fpl_client
from fpl_oracle.api.cache import cache_manager

class FaultInjectingTransport(httpx.AsyncBaseTransport):
    def __init__(self, mode: str = "429"):
        self.mode = mode

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if self.mode == "429":
            return httpx.Response(
                status_code=429,
                headers={"Retry-After": "1"},
                content=b'{"detail": "Request was throttled. Expected available in 1 second."}',
                request=request
            )
        elif self.mode == "503":
            return httpx.Response(
                status_code=503,
                content=b'{"detail": "FPL Service Temporarily Unavailable - Updating Game"}',
                request=request
            )
        elif self.mode == "timeout":
            raise httpx.ReadTimeout(f"Read timeout connecting to {url_str}")
        elif self.mode == "drift":
            cached = cache_manager.get_stale("bootstrap-static")
            if cached:
                drifted = dict(cached)
                drifted["new_unexpected_2026_rule"] = {"active": True, "token": "xyz"}
                if drifted.get("elements"):
                    drifted["elements"] = [dict(e, unknown_metric_xG90=9.99) for e in drifted["elements"]]
                return httpx.Response(status_code=200, content=json.dumps(drifted).encode("utf-8"), request=request)
            return httpx.Response(status_code=200, content=b'{"events":[],"elements":[]}', request=request)
        return httpx.Response(status_code=500, content=b'{"error": "Fault"}', request=request)


async def run_tests():
    print("========================================================")
    print("PART 1: Testing Live FPL Oracle Server Endpoints")
    print("========================================================")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=30.0) as client:
        # 1. Health
        res = await client.get("/api/health")
        print(f"1.  GET /api/health -> Status {res.status_code}")
        assert res.status_code == 200, f"Health check failed: {res.text}"
        data = res.json()
        print(f"    Season: {data.get('season')}, Current GW: {data.get('current_gameweek')}, Next GW: {data.get('next_gameweek')}")
        assert "game_state" in data
        assert "cache_age_seconds" in data
        assert "model" in data
        assert "news" in data
        assert "llm" in data

        # 2. Game State
        res = await client.get("/api/game-state")
        print(f"2.  GET /api/game-state -> Status {res.status_code}")
        assert res.status_code == 200
        gs_data = res.json()
        print(f"    Phase: {gs_data.get('phase')}, Seconds to deadline: {gs_data.get('seconds_to_deadline')}")

        # 3. Profile GET
        res = await client.get("/api/profile")
        print(f"3.  GET /api/profile -> Status {res.status_code}")
        assert res.status_code == 200
        prof = res.json()
        print(f"    Manager ID: {prof.get('manager_id')}, Risk: {prof.get('risk_preference')}")

        # 4. Profile POST
        res = await client.post("/api/profile", json={"risk_preference": "balanced"})
        print(f"4.  POST /api/profile -> Status {res.status_code}")
        assert res.status_code == 200

        # 5. Squad
        res = await client.get("/api/squad")
        print(f"5.  GET /api/squad -> Status {res.status_code}")
        assert res.status_code == 200
        squad_data = res.json()
        starters = squad_data.get("starters", [])
        bench = squad_data.get("bench", [])
        total_players = len(starters) + len(bench)
        print(f"    Total squad players: {total_players} (Starters: {len(starters)}, Bench: {len(bench)})")
        print(f"    Formation: {squad_data.get('formation')}, Captain: {squad_data.get('captain', {}).get('web_name')}")
        print(f"    Data as of: {squad_data.get('data_as_of')}, Stale: {squad_data.get('stale')}")
        assert total_players == 15, f"Expected 15 players, got {total_players}"

        # 6. Projections
        res = await client.get("/api/projections?horizon=3")
        print(f"6.  GET /api/projections -> Status {res.status_code}")
        assert res.status_code == 200
        proj_data = res.json()
        print(f"    Projected players returned: {len(proj_data.get('players', []))}")
        assert len(proj_data.get("players", [])) > 0

        # 7. Optimize
        res = await client.post("/api/optimize", json={})
        print(f"7.  POST /api/optimize -> Status {res.status_code}")
        assert res.status_code == 200
        opt_data = res.json()
        print(f"    Recommended Plan: {opt_data.get('recommended_plan', {}).get('plan_type')}")
        print(f"    Hit Verdict: {opt_data.get('hit_verdict')}")

        # 8. Chips
        res = await client.get("/api/chips")
        print(f"8.  GET /api/chips -> Status {res.status_code}")
        assert res.status_code == 200
        chips_data = res.json()
        print(f"    Chip plan entries: {len(chips_data.get('chip_plan_table', []))}")
        assert len(chips_data.get("chip_plan_table", [])) > 0

        # 9. League Intel (unconfigured and with ID)
        res = await client.get("/api/league")
        print(f"9a. GET /api/league (unconfigured) -> Status {res.status_code}")
        assert res.status_code == 200
        league_unconf = res.json()
        print(f"    Status: {league_unconf.get('status')}")

        res = await client.get("/api/league?league_id=314")
        print(f"9b. GET /api/league?league_id=314 -> Status {res.status_code}")
        assert res.status_code == 200
        league_data = res.json()
        print(f"    League Name: {league_data.get('league_name')}, Total Teams: {league_data.get('total_teams')}")

        # 10. Briefing
        res = await client.get("/api/briefing")
        print(f"10. GET /api/briefing -> Status {res.status_code}")
        assert res.status_code == 200
        briefing_data = res.json()
        print(f"    Briefing target GW: {briefing_data.get('target_gameweek')}")
        print(f"    Markdown length: {len(briefing_data.get('markdown', ''))} chars")
        assert len(briefing_data.get("markdown", "")) > 100

        # 11. Price changes
        res = await client.get("/api/price-changes")
        print(f"11. GET /api/price-changes -> Status {res.status_code}")
        assert res.status_code == 200
        pc_data = res.json()
        print(f"    Rises: {len(pc_data.get('rises', []))}, Falls: {len(pc_data.get('falls', []))}")

        # 12. Chat endpoint
        res = await client.post("/api/chat", json={"message": "Who is the best captaincy pick for Gameweek 6?"})
        print(f"12. POST /api/chat -> Status {res.status_code}")
        assert res.status_code == 200
        chat_data = res.json()
        resp_text = chat_data.get("response", "")
        print(f"    Chat response preview: {resp_text[:80].encode('ascii', 'backslashreplace').decode('ascii')}...")
        assert len(resp_text) > 20

        # 13. Web UI root
        res = await client.get("/")
        print(f"13. GET / (Web UI root) -> Status {res.status_code}")
        assert res.status_code == 200
        assert "<title>FPL Oracle" in res.text

    print("\n========================================================")
    print("PART 2: Testing Fault Injection & Degraded States")
    print("========================================================")
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=30.0) as client:
        # Fault 1: 429 Rate Limit Fallback
        fpl_client.set_transport(FaultInjectingTransport(mode="429"))
        try:
            boot, is_stale = await fpl_client._fetch_json("bootstrap-static/", "bootstrap-static", ttl_seconds=1, retries=1, force_refresh=True)
            print(f"F1. 429 Rate Limit Injection -> Stale fallback engaged: {is_stale}")
            assert is_stale is True
        finally:
            fpl_client.reset_transport()

        # Fault 2: 503 Service Unavailable on server endpoint
        cache_manager.force_expire("bootstrap-static")
        fpl_client.set_transport(FaultInjectingTransport(mode="503"))
        try:
            res = await client.get("/api/projections?horizon=2")
            print(f"F2. 503 Server Error Injection -> Status {res.status_code}, Stale: {res.json().get('stale')}")
            assert res.status_code == 200
            assert res.json().get("stale") is True
        finally:
            fpl_client.reset_transport()

        # Fault 3: Read Timeout on server endpoint
        cache_manager.force_expire("bootstrap-static")
        fpl_client.set_transport(FaultInjectingTransport(mode="timeout"))
        try:
            res = await client.get("/api/projections?horizon=2")
            print(f"F3. Read Timeout Injection -> Status {res.status_code}, Stale: {res.json().get('stale')}")
            assert res.status_code == 200
            assert res.json().get("stale") is True
        finally:
            fpl_client.reset_transport()

        # Fault 4: Schema Drift Resilience
        fpl_client.set_transport(FaultInjectingTransport(mode="drift"))
        try:
            boot, _ = await fpl_client.get_bootstrap_static(force_refresh=True)
            print(f"F4. Schema Drift Injection -> Elements parsed defensively: {len(boot.elements)}")
            assert len(boot.elements) > 0
        finally:
            fpl_client.reset_transport()

        # Fault 5: Malformed Squad Validation
        res = await client.post("/api/squad/manual", json={"player_ids": [1, 2, 3]})
        print(f"F5. Malformed Squad Validation -> Status {res.status_code}, Error: {res.json().get('error')}")
        assert res.status_code == 400
        assert res.json().get("code") == 400
        assert res.json().get("fallback_used") is False

    await fpl_client.aclose()
    print("\n========================================================")
    print("ALL LIVE ENDPOINTS AND FAULT INJECTION TESTS PASSED 100%!")
    print("========================================================")
    sys.stdout.flush()

if __name__ == "__main__":
    asyncio.run(run_tests())
    sys.exit(0)
