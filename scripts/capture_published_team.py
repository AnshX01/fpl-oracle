import json
import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from playwright.sync_api import sync_playwright

root = Path(__file__).resolve().parent.parent
out = root / "reports" / "published-team-ui"
out.mkdir(parents=True, exist_ok=True)
from scripts.capture_screenshots import fixtures  # noqa: E402

source = fixtures()
rows = source["/api/squad"]["starters"] + source["/api/squad"]["bench"]
names = [
    "Kinsky",
    "Calafiori",
    "Maguire",
    "Ajer",
    "Mbeumo",
    "Palmer",
    "Szoboszlai",
    "Belloumi",
    "Groß",
    "Haaland",
    "Calvert-Lewin",
    "Verbruggen",
    "João Pedro",
    "White",
    "Diop",
]
positions = ["GKP"] + ["DEF"] * 3 + ["MID"] * 5 + ["FWD"] * 2 + ["GKP", "FWD", "DEF", "DEF"]
for i, p in enumerate(rows):
    p.update(
        web_name=names[i],
        position=positions[i],
        now_cost=5.5,
        expected_points=4.5,
        bench_order=max(0, i - 10),
        is_captain=i == 9,
        is_vice_captain=i == 5,
    )
squad = source["/api/squad"]
squad.update(
    captain=rows[9],
    vice_captain=rows[5],
    total_squad_value=100.0,
    total_selling_value=98.2,
    bank_millions=1.8,
    free_transfers=1,
    total_expected_points=57.5,
    published_gameweek=5,
)
source["/api/decision-card"].update(captain=rows[9], xi=rows[:11])
source["/api/team/confirmation"] = {
    "locked": False,
    "state": {
        "team_confirmed": False,
        "target_gw": 6,
        "bank_tenths": 18,
        "free_transfers": 1,
        "squad": [{**p, "purchase_price": 55, "selling_price": 55} for p in rows],
        "chips_remaining_set_1": ["wildcard"],
        "deadline_utc": "2099-01-01T00:00:00Z",
    },
    "players": [{"element": p["element"], "name": p["web_name"], "price": 55} for p in rows],
}
source["/api/league"] = {
    "overall_rank": 1380941,
    "current_league_rank": 2,
    "league_name": "Fixture league",
    "standings": [],
}
source["/api/briefing"] = {"markdown": "Gameweek 6\n\nSave your transfer. Captain Haaland."}
source["/api/checklist"] = {"checklist": [{"item": "Captain", "status": "PASS", "detail": "Haaland"}]}
server = ThreadingHTTPServer(("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(root / "web")))
threading.Thread(target=server.serve_forever, daemon=True).start()
with sync_playwright() as pw:
    b = pw.chromium.launch()
    for width in [1440, 390, 360, 320]:
        for theme in ["dark", "light"]:
            page = b.new_page(viewport={"width": width, "height": 900})
            errors = []
            page.on("pageerror", lambda e, errors=errors: errors.append(str(e)))
            page.add_init_script(f"localStorage.setItem('fpl_oracle_theme','{theme}')")
            status = {"mode": "calculating", "locked": False}
            requests = []

            def route(r, *, status=status, requests=requests):
                from urllib.parse import urlsplit

                path = urlsplit(r.request.url).path
                requests.append(path)
                if path == "/api/advice/start" or path.startswith("/api/advice/status/"):
                    body = (
                        {
                            "id": "fixture",
                            "status": status["mode"],
                            "stage": "Choosing your transfers",
                            "result": {
                                "squadData": squad,
                                "decisionCard": source["/api/decision-card"],
                                "contingencyPlans": source["/api/contingency/plans"],
                            },
                        }
                        if status["mode"] == "ready"
                        else {"id": "fixture", "status": "calculating", "stage": "Choosing your transfers"}
                    )
                elif path == "/api/team/confirmation":
                    body = {**source[path], "locked": status["locked"]}
                elif path == "/api/team/followed":
                    status["locked"] = True
                    body = {"status": "followed"}
                elif path == "/api/team/undo":
                    status["locked"] = False
                    body = {"status": "undone"}
                else:
                    body = source.get(path, {})
                r.fulfill(content_type="application/json", body=json.dumps(body))

            page.route("**/api/**", route)
            page.goto(f"http://127.0.0.1:{server.server_port}/index.html")
            page.wait_for_function("window.__fpl_vm__?.hasBasicSquad && window.__fpl_vm__.decisionCardLoading")
            assert "Confirm my team" not in page.locator("body").inner_text()
            for tab in ["overview", "team"]:
                page.evaluate(f"window.__fpl_vm__.activeTab='{tab}'")
                if tab == "team":
                    assert page.locator(".pitch .player").count() == 11 and page.locator(".bench .player").count() == 4
                assert page.evaluate("document.documentElement.scrollWidth<=innerWidth")
                page.wait_for_function(
                    "document.getAnimations().filter(a => a.effect.getTiming().iterations !== Infinity).every(a => a.playState !== 'running')"
                )
                page.screenshot(path=str(out / f"{width}-{theme}-loading-{tab}.png"), full_page=True)
            status["mode"] = "ready"
            page.wait_for_function(
                "window.__fpl_vm__.hasLoadedSquad && !window.__fpl_vm__.decisionCardLoading", timeout=10000
            )
            for tab in ["overview", "team", "transfers", "league"]:
                page.evaluate(f"window.__fpl_vm__.activeTab='{tab}'")
                assert page.evaluate("document.documentElement.scrollWidth<=innerWidth")
                page.wait_for_function(
                    "document.getAnimations().filter(a => a.effect.getTiming().iterations !== Infinity).every(a => a.playState !== 'running')"
                )
                page.screenshot(path=str(out / f"{width}-{theme}-ready-{tab}.png"), full_page=True)
            for prop in ["showBriefingModal", "showChecklistModal"]:
                page.evaluate(f"window.__fpl_vm__.{prop}=true")
                page.wait_for_function(
                    "document.getAnimations().filter(a => a.effect.getTiming().iterations !== Infinity).every(a => a.playState !== 'running')"
                )
                page.screenshot(path=str(out / f"{width}-{theme}-{prop}.png"), full_page=True)
                page.evaluate(f"window.__fpl_vm__.{prop}=false")
            page.evaluate("window.__fpl_vm__.confirmFollowedAdvice()")
            page.wait_for_function("window.__fpl_vm__.followedLocked")
            n = requests.count("/api/advice/start")
            page.reload()
            page.wait_for_function("window.__fpl_vm__.followedLocked")
            assert requests.count("/api/advice/start") == n
            page.evaluate("window.__fpl_vm__.activeTab='overview'")
            page.wait_for_function(
                "document.getAnimations().filter(a => a.effect.getTiming().iterations !== Infinity).every(a => a.playState !== 'running')"
            )
            page.screenshot(path=str(out / f"{width}-{theme}-done.png"), full_page=True)
            page.evaluate("window.__fpl_vm__.undoFollowedAdvice()")
            page.wait_for_function("window.__fpl_vm__.hasLoadedSquad && !window.__fpl_vm__.followedLocked")
            page.evaluate("window.__fpl_vm__.openConfirmTeam().then(()=>window.__fpl_vm__.confirmMode='different')")
            page.wait_for_function("window.__fpl_vm__.confirmRows.length===15")
            page.wait_for_function(
                "document.getAnimations().filter(a => a.effect.getTiming().iterations !== Infinity).every(a => a.playState !== 'running')"
            )
            page.screenshot(path=str(out / f"{width}-{theme}-update-team.png"), full_page=True)
            assert not errors, errors
            page.close()
    b.close()
server.shutdown()
print(
    "PASS: default published advice, loading team, pages, modals, Done/reload/Undo, no overflow or page errors at 1440/390/360/320 in both themes."
)
