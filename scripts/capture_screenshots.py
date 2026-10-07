"""Deterministic UI integration captures; fixtures are labeled, never private/live evidence."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "screenshots"


def fixtures():
    rows = [
        {
            "element": i,
            "web_name": f"Fixture Player {i}",
            "position": "MID",
            "team": i,
            "status": "a",
            "chance_of_playing": 100,
            "expected_points": 0.0,
        }
        for i in range(1, 16)
    ]
    squad = {
        "starters": rows[:11],
        "bench": rows[11:],
        "captain": rows[0],
        "vice_captain": rows[1],
        "bank_millions": 0.0,
        "free_transfers": 0,
        "bank_source": "fixture",
        "ft_source": "fixture",
        "is_stale": False,
        "formation": "3-5-2",
        "starters_expected_points": 0.0,
        "captain_bonus_expected_points": 0.0,
        "total_expected_points": 0.0,
    }
    plan = {"plan_type": "ROLL_TRANSFER", "transfers_in": [], "transfers_out": [], "next_banked_ft": 1}
    card = {
        "transfers": {
            "is_roll": True,
            "in": [],
            "out": [],
            "ft_remaining": 0,
            "ft_next_gw": 1,
            "bank_after": 0.0,
            "hit_cost": 0,
            "net_gain_vs_roll": 0.0,
        },
        "captain": rows[0],
        "formation": "3-5-2",
        "two_line_reasoning": "Deterministic fixture only.",
        "caveats": [],
    }
    return {
        "/api/squad": squad,
        "/api/squad/basic": squad,
        "/api/contingency/plans": {"plan_a": plan},
        "/api/decision-card": card,
        "/api/game-state": {"next_gw": 6, "deadline_time": "2099-01-01T00:00:00Z"},
        "/api/league": {"status": "empty", "standings": [], "simulation": None},
    }


def capture(base_url):
    OUT.mkdir(parents=True, exist_ok=True)
    source = fixtures()
    manifest = {
        "source_kind": "deterministic_http_fixtures",
        "code_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "captures": [],
    }
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for theme in ("dark", "light"):
            for size in ((1920, 1080), (390, 844)):
                errors = []
                unexpected = []
                mode = {"value": "success"}
                page = browser.new_page(viewport={"width": size[0], "height": size[1]})
                page.add_init_script(f"localStorage.setItem('fpl_oracle_theme', '{theme}')")
                page.on("pageerror", lambda error, errors=errors: errors.append(str(error)))
                page.on(
                    "response",
                    lambda res, unexpected=unexpected, mode=mode: (
                        unexpected.append(res.url) if res.status >= 400 and mode["value"] == "success" else None
                    ),
                )

                def route(req, request=None, *, mode=mode):
                    from urllib.parse import urlsplit

                    path = urlsplit(req.request.url).path
                    if mode["value"] == "failure" and path in (
                        "/api/squad",
                        "/api/decision-card",
                        "/api/contingency/plans",
                    ):
                        req.fulfill(
                            status=500, content_type="application/json", body='{"error":"intentional fixture failure"}'
                        )
                    elif mode["value"] == "missing" and path in ("/api/squad", "/api/squad/basic"):
                        req.fulfill(content_type="application/json", body='{"starters":[],"bench":[]}')
                    else:
                        req.fulfill(content_type="application/json", body=json.dumps(source.get(path, {})))

                page.route("**/api/**", route)
                page.goto(base_url)
                page.wait_for_function(
                    "window.__fpl_vm__ && window.__fpl_vm__.hasLoadedSquad && !window.__fpl_vm__.decisionCardLoading && !window.__fpl_vm__.plansLoading",
                    timeout=30000,
                )
                assert page.evaluate("window.__fpl_vm__.squadData.starters.length") == 11
                assert page.evaluate("window.__fpl_vm__.squadData.bench.length") == 4
                assert page.evaluate(
                    "window.__fpl_vm__.squadData.starters.some(p=>p.element===window.__fpl_vm__.squadData.captain.element)"
                )
                assert "Roll Free Transfer" in page.locator("#decision-card").inner_text()
                assert page.evaluate("window.__fpl_vm__.ftDisplay") == "0"
                assert not errors and not unexpected, (errors, unexpected)
                for state in ("populated", "failed_after_success", "missing"):
                    if state != "populated":
                        mode["value"] = "failure" if state == "failed_after_success" else "missing"
                        page.evaluate("window.__fpl_vm__.refreshAll()")
                        page.wait_for_function(
                            "!window.__fpl_vm__.squadLoading && !window.__fpl_vm__.plansLoading && !window.__fpl_vm__.decisionCardLoading"
                        )
                        assert "Transfer recommendations unavailable" in page.locator("#decision-card").inner_text()
                        assert not page.evaluate("window.__fpl_vm__.hasLoadedSquad")
                        assert "Save your free transfer" not in page.locator("body").inner_text()
                    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), "Horizontal overflow"
                    title = page.locator("#decision-card h2")
                    assert title.is_visible() and title.inner_text().strip()
                    name = f"{theme}_{size[0]}_{state}.png"
                    file = OUT / name
                    page.screenshot(path=str(file), full_page=True)
                    manifest["captures"].append(
                        {
                            "file": name,
                            "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                            "state": state,
                            "theme": theme,
                            "viewport": list(size),
                            "assertions": "passed",
                            "source_kind": "deterministic_http_fixtures",
                        }
                    )
                page.close()
        browser.close()
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--base-url", default="http://127.0.0.1:8000")
    capture(args.parse_args().base_url)
