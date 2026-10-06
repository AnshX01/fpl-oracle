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

from fpl_oracle.api.cache import cache_manager
from fpl_oracle.api.fpl_client import fpl_client
from fpl_oracle.server.main import app


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
                request=request,
            )
        elif self.mode == "503":
            return httpx.Response(
                status_code=503,
                content=b'{"detail": "FPL Service Temporarily Unavailable - Updating Game"}',
                request=request,
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
        print(
            f"    Season: {data.get('season')}, Current GW: {data.get('current_gameweek')}, Next GW: {data.get('next_gameweek')}"
        )
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

        # 3b. Sync Status GET
        res = await client.get("/api/sync/status")
        print(f"3b. GET /api/sync/status -> Status {res.status_code}")
        assert res.status_code == 200
        sync_stat = res.json()
        print(f"    Pipeline running: {sync_stat.get('is_running')}, Step: {sync_stat.get('current_step')}")

        # 3c. Sync Stream SSE GET
        async with client.stream("GET", "/api/sync/stream") as sse_stream:
            print(
                f"3c. GET /api/sync/stream -> Status {sse_stream.status_code}, Media: {sse_stream.headers.get('content-type')}"
            )
            assert sse_stream.status_code == 200
            assert "text/event-stream" in sse_stream.headers.get("content-type", "")
            async for chunk in sse_stream.aiter_lines():
                if chunk.startswith("data:"):
                    stream_payload = json.loads(chunk[5:].strip())
                    print(
                        f"    SSE Event received: step={stream_payload.get('step')}, progress={stream_payload.get('progress_pct')}%"
                    )
                    break

        # 4. Profile POST with overrides (Bank & Free Transfers)
        res = await client.post("/api/profile", json={"risk_preference": "balanced", "bank": 1.5, "free_transfers": 3})
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
        print(f"    Bank: £{squad_data.get('bank_millions')}m, Free Transfers: {squad_data.get('free_transfers')}")
        print(
            f"    Squad Val: £{squad_data.get('total_squad_value')}m, Selling Val: £{squad_data.get('total_selling_value')}m, Team Val: £{squad_data.get('total_team_value')}m"
        )
        print(f"    Data as of: {squad_data.get('data_as_of')}, Stale: {squad_data.get('stale')}")
        assert total_players == 15, f"Expected 15 players, got {total_players}"
        assert squad_data.get("bank_millions") == 1.5
        assert squad_data.get("free_transfers") == 3
        assert squad_data.get("total_squad_value", 0) > 80.0
        assert squad_data.get("total_selling_value", 0) > 80.0

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

        # 13. Contingency Plans
        res = await client.get("/api/contingency/plans")
        print(f"13. GET /api/contingency/plans -> Status {res.status_code}")
        assert res.status_code == 200
        contingency_plans = res.json()
        print(
            f"    Plans returned: plan_a={bool(contingency_plans.get('plan_a'))}, plan_b={bool(contingency_plans.get('plan_b'))}, plan_c={bool(contingency_plans.get('plan_c'))}"
        )
        assert "plan_a" in contingency_plans
        assert "plan_b" in contingency_plans
        assert "plan_c" in contingency_plans

        # 14. Injury Contingency Matrix
        res = await client.get("/api/contingency/matrix")
        print(f"14. GET /api/contingency/matrix -> Status {res.status_code}")
        assert res.status_code == 200
        matrix_data = res.json()
        print(f"    Matrix entries: {len(matrix_data.get('contingency_matrix', []))}")
        assert "contingency_matrix" in matrix_data
        assert len(matrix_data["contingency_matrix"]) > 0

        # 15. Panic Button Re-optimizer
        res = await client.post("/api/contingency/panic", json={"query": "haaland out injured"})
        print(f"15. POST /api/contingency/panic -> Status {res.status_code}")
        assert res.status_code == 200
        panic_data = res.json()
        print(
            f"    Panic response status: {panic_data.get('status')}, Recommendation: {panic_data.get('recommendation', '')[:60]}..."
        )
        assert panic_data.get("status") == "crisis_resolved"
        assert "lineup_action" in panic_data

        # 16. Pre-Deadline Checklist
        res = await client.get("/api/contingency/checklist")
        print(f"16. GET /api/contingency/checklist -> Status {res.status_code}")
        assert res.status_code == 200
        chk_data = res.json()
        print(f"    Checklist gameweek: {chk_data.get('gameweek')}, Items: {len(chk_data.get('checklist', []))}")
        assert len(chk_data.get("checklist", [])) >= 5

        # 17. Post-GW Review
        res = await client.get("/api/review")
        print(f"17. GET /api/review -> Status {res.status_code}")
        assert res.status_code == 200
        rev_data = res.json()
        print(
            f"    Review target GW: {rev_data.get('target_gameweek')}, Markdown length: {len(rev_data.get('review_markdown', ''))}"
        )
        assert len(rev_data.get("review_markdown", "")) > 50

        # 18. System Jobs
        res = await client.get("/api/system/jobs")
        print(f"18. GET /api/system/jobs -> Status {res.status_code}")
        assert res.status_code == 200
        jobs_data = res.json()
        print(
            f"    Jobs registered: {jobs_data.get('total_jobs')}, Scheduler running: {jobs_data.get('scheduler_running')}"
        )
        assert jobs_data.get("total_jobs") == 5

        # 19. On-Demand Job Execution
        res = await client.post("/api/system/jobs/price_snapshot/run")
        print(f"19. POST /api/system/jobs/price_snapshot/run -> Status {res.status_code}")
        assert res.status_code == 200
        assert res.json().get("success") is True

        # 20. System Models & Versioning
        res = await client.get("/api/system/models")
        print(f"20. GET /api/system/models -> Status {res.status_code}")
        assert res.status_code == 200
        models_data = res.json()
        print(
            f"    Active Model: {models_data.get('active_version', {}).get('version')}, History count: {len(models_data.get('history', []))}"
        )
        assert "active_version" in models_data

        # 21. System Diagnostic Status
        res = await client.get("/api/system/status")
        print(f"21. GET /api/system/status -> Status {res.status_code}")
        assert res.status_code == 200
        sys_status = res.json()
        print(
            f"    System Phase: {sys_status.get('game_state', {}).get('phase')}, Active Model MAE: {sys_status.get('model', {}).get('mae')}"
        )
        assert "scheduler" in sys_status

        # 22. Web UI root
        res = await client.get("/")
        print(f"22. GET / (Web UI root) -> Status {res.status_code}")
        assert res.status_code == 200
        assert "<title>FPL Oracle" in res.text

    print("\n========================================================")
    print("PART 2: Testing Fault Injection & Degraded States")
    print("========================================================")
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=30.0) as client:
        # Fault 1: 429 Rate Limit Fallback
        fpl_client.set_transport(FaultInjectingTransport(mode="429"))
        try:
            boot, is_stale = await fpl_client._fetch_json(
                "bootstrap-static/", "bootstrap-static", ttl_seconds=1, retries=1, force_refresh=True
            )
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
