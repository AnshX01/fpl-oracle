"""Automated UI Screenshot Capture & Verification Script (Repair B & G3).

Spawns the local FastAPI server, asserts HTTP 200 on key API endpoints,
validates truthful loaded/empty/error states, enforces 0 emojis across DOM,
tracks console errors, and captures:
- Desktop populated dashboard & decision card (1920x1080)
- Mobile populated dashboard & decision card (390x844 Android-sized viewport)
- Deliberate empty squad state (desktop & mobile)
- Deliberate failed API state (desktop & mobile)
- Deliberate empty league state (desktop)
- Deliberate loading state (desktop)

Saved directly into reports/screenshots/ with structured manifest.json metadata.
"""

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
SCREENSHOTS_DIR = ROOT / "reports" / "screenshots"
PORT = 8000
SERVER_URL = f"http://127.0.0.1:{PORT}"

ENDPOINTS_TO_CHECK = [
    "/",
    "/api/health",
    "/api/squad",
    "/api/decision-card",
    "/api/contingency/plans",
    "/api/profile",
]


def is_emoji(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x1F300 <= cp <= 0x1FAFF
        or 0x2600 <= cp <= 0x27BF
        or 0x2B50 <= cp <= 0x2B55
        or 0x2300 <= cp <= 0x23FF
        or 0xFE00 <= cp <= 0xFE0F
    )


def find_emojis(text: str) -> list[str]:
    return [ch for ch in text if is_emoji(ch)]


def wait_for_server(timeout_sec: float = 45.0) -> bool:
    """Poll until server responds with HTTP 200 on root."""
    start = time.time()
    while time.time() - start < timeout_sec:
        try:
            with urllib.request.urlopen(f"{SERVER_URL}/", timeout=1.5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def verify_endpoints() -> tuple[dict, str]:
    """Verify HTTP 200 on all required endpoints and identify squad data provenance."""
    print("Verifying backend API endpoints:")
    squad_data = {}
    for path in ENDPOINTS_TO_CHECK:
        url = f"{SERVER_URL}{path}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "FPL-Oracle-Screenshot-Test"})
            with urllib.request.urlopen(req, timeout=120.0) as resp:
                status = resp.status
                assert status == 200, f"Endpoint {path} returned status {status}"
                body_bytes = resp.read()
                data_len = len(body_bytes)
                print(f"  [OK 200] {path} ({data_len:,} bytes)")
                if path == "/api/squad":
                    try:
                        squad_data = json.loads(body_bytes.decode("utf-8"))
                    except Exception:
                        pass
        except Exception as e:
            print(f"  [FAILED] {path}: {e}")
            raise

    starters = squad_data.get("starters", [])
    bench = squad_data.get("bench", [])
    mgr_id = squad_data.get("manager_id")

    if len(starters) == 11 and len(bench) == 4 and mgr_id:
        provenance = "local_private_live"
        print(f"\nSquad Provenance: LIVE PRIVATE CONFIG (Manager ID: {mgr_id}, 15 players loaded)")
    elif len(starters) == 11 and len(bench) == 4:
        provenance = "fixture"
        print("\nSquad Provenance: POPULATED FIXTURE (15 players loaded, unconfigured manager ID)")
    else:
        provenance = "unconfigured_empty"
        print(f"\nSquad Provenance: UNCONFIGURED (Starters: {len(starters)}, Bench: {len(bench)})")

    return squad_data, provenance


def capture_screenshots():
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    python_exe = sys.executable
    venv_py = ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_py.exists():
        python_exe = str(venv_py)

    print("=" * 65)
    print("FPL ORACLE -- RESPONSIVE UI VERIFICATION & SCREENSHOTS (REPAIR B)")
    print("=" * 65)

    # 1. Start backend server
    print(f"Starting uvicorn server on {SERVER_URL} using {python_exe}...")
    sub_env = os.environ.copy()
    sub_env["PYTHONPATH"] = str(ROOT / "src")

    server_proc = subprocess.Popen(
        [
            python_exe,
            "-m",
            "uvicorn",
            "fpl_oracle.server.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(PORT),
            "--log-level",
            "warning",
        ],
        cwd=ROOT,
        env=sub_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    captured_manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": "unknown",
        "screenshots": {},
        "verifications": {},
    }

    try:
        print("Waiting for server startup...")
        if not wait_for_server():
            print("ERROR: Server failed to start within timeout.")
            sys.exit(1)
        print("Server is up and responsive!\n")

        # Verify backend endpoints
        squad_data, provenance = verify_endpoints()
        captured_manifest["provenance"] = provenance

        # 2. Launch headless browser
        with sync_playwright() as p:
            print("\nLaunching headless Chromium...")
            browser = p.chromium.launch(headless=True)

            console_errors = []

            def handle_console(msg):
                if msg.type == "error":
                    console_errors.append(msg.text)

            def handle_page_error(err):
                console_errors.append(str(err))

            # ============================================================
            # STATE 1: POPULATED DESKTOP VIEWPORT (1920x1080)
            # ============================================================
            print("\n[State 1: Populated Desktop Viewport (1920x1080)]")
            page = browser.new_page(viewport={"width": 1920, "height": 1080})
            page.on("console", handle_console)
            page.on("pageerror", handle_page_error)

            page.goto(SERVER_URL, wait_until="commit", timeout=45000)
            page.wait_for_selector("#decision-card", timeout=45000)
            page.wait_for_timeout(3000)  # Wait for Vue reactive bindings to settle

            # Assert decision card
            card_el = page.query_selector("#decision-card")
            assert card_el is not None, "Decision card element '#decision-card' not found in DOM"
            card_text = card_el.inner_text().strip()
            assert len(card_text) > 20, f"Decision card text too short: '{card_text}'"
            assert "undefined" not in card_text, "Found unrendered 'undefined' in decision card"
            assert "NaN" not in card_text, "Found unrendered 'NaN' in decision card"

            # Assert squad elements
            has_squad = page.evaluate("() => window.__fpl_vm__.hasLoadedSquad")
            if has_squad:
                starters_count = page.evaluate("() => window.__fpl_vm__.squadData.starters.length")
                bench_count = page.evaluate("() => window.__fpl_vm__.squadData.bench.length")
                assert starters_count == 11, f"Expected 11 starters, got {starters_count}"
                assert bench_count == 4, f"Expected 4 bench players, got {bench_count}"
                captain_name = page.evaluate(
                    "() => window.__fpl_vm__.squadData.captain ? window.__fpl_vm__.squadData.captain.web_name : null"
                )
                assert captain_name is not None, "Captain missing from populated squad"
                print(f"  Squad verified: 11 starters, 4 bench, Captain: {captain_name}")

            # Assert emoji purge
            body_text = page.query_selector("body").inner_text()
            emojis_found = find_emojis(body_text)
            assert not emojis_found, f"Found emojis in desktop DOM: {emojis_found}"

            # Assert console error tracking
            assert not console_errors, f"Encountered unexpected console errors: {console_errors}"

            # Capture desktop dashboard
            dash_desktop_path = SCREENSHOTS_DIR / "dashboard_desktop.png"
            page.screenshot(path=str(dash_desktop_path), full_page=True)
            (SCREENSHOTS_DIR / "dashboard.png").write_bytes(dash_desktop_path.read_bytes())
            print(f"  Captured: {dash_desktop_path.name} ({dash_desktop_path.stat().st_size:,} bytes)")
            captured_manifest["screenshots"]["dashboard_desktop"] = {
                "file": dash_desktop_path.name,
                "size_bytes": dash_desktop_path.stat().st_size,
                "dimensions": "1920x1080",
                "state": "populated",
            }

            # Capture decision card
            dc_desktop_path = SCREENSHOTS_DIR / "decision_card_desktop.png"
            card_el.screenshot(path=str(dc_desktop_path))
            (SCREENSHOTS_DIR / "decision_card.png").write_bytes(dc_desktop_path.read_bytes())
            print(f"  Captured: {dc_desktop_path.name} ({dc_desktop_path.stat().st_size:,} bytes)")
            captured_manifest["screenshots"]["decision_card_desktop"] = {
                "file": dc_desktop_path.name,
                "size_bytes": dc_desktop_path.stat().st_size,
                "state": "populated",
            }

            # ============================================================
            # STATE 2: POPULATED MOBILE VIEWPORT (390x844 - Android-sized)
            # ============================================================
            print("\n[State 2: Populated Android-Sized Mobile Viewport (390x844)]")
            print("  Label: Android-sized mobile browser viewport (390x844 CSS px) - browser emulation, not native app")
            page.set_viewport_size({"width": 390, "height": 844})
            page.wait_for_timeout(1500)

            card_el_m = page.query_selector("#decision-card")
            assert card_el_m is not None, "Decision card missing in mobile viewport"
            card_text_m = card_el_m.inner_text().strip()
            assert len(card_text_m) > 20, "Decision card empty in mobile viewport"

            # Check no horizontal blowout on card
            box = card_el_m.bounding_box()
            assert box and box["width"] <= 390, f"Decision card overflowing mobile viewport: {box['width']}px"

            body_text_m = page.query_selector("body").inner_text()
            emojis_m = find_emojis(body_text_m)
            assert not emojis_m, f"Found emojis in mobile DOM: {emojis_m}"

            dash_mobile_path = SCREENSHOTS_DIR / "dashboard_mobile.png"
            page.screenshot(path=str(dash_mobile_path), full_page=True)
            print(f"  Captured: {dash_mobile_path.name} ({dash_mobile_path.stat().st_size:,} bytes)")
            captured_manifest["screenshots"]["dashboard_mobile"] = {
                "file": dash_mobile_path.name,
                "size_bytes": dash_mobile_path.stat().st_size,
                "dimensions": "390x844",
                "state": "populated",
            }

            dc_mobile_path = SCREENSHOTS_DIR / "decision_card_mobile.png"
            card_el_m.screenshot(path=str(dc_mobile_path))
            print(f"  Captured: {dc_mobile_path.name} ({dc_mobile_path.stat().st_size:,} bytes)")
            captured_manifest["screenshots"]["decision_card_mobile"] = {
                "file": dc_mobile_path.name,
                "size_bytes": dc_mobile_path.stat().st_size,
                "dimensions": "390x844 element",
                "state": "populated",
            }

            # ============================================================
            # STATE 3: INTENTIONAL MISSING SQUAD / UNCONFIGURED STATE
            # ============================================================
            print("\n[State 3: Intentional Missing Squad / Unconfigured State]")
            page.set_viewport_size({"width": 1920, "height": 1080})

            # Transition to missing squad via VM state
            page.evaluate("""() => {
                window.__fpl_vm__.squadData = { starters: [], bench: [] };
                window.__fpl_vm__.decisionCard = null;
                window.__fpl_vm__.contingencyPlans = {};
                window.__fpl_vm__.squadLoading = false;
                window.__fpl_vm__.plansLoading = false;
                window.__fpl_vm__.decisionCardLoading = false;
            }""")
            page.wait_for_timeout(1000)

            missing_body = page.query_selector("body").inner_text()

            # Verifications for honest missing squad state
            assert "Squad data unavailable" in missing_body, (
                "Missing squad state did not display 'Squad data unavailable'"
            )
            assert "Transfer recommendations unavailable" in missing_body, (
                "Decision card did not state 'Transfer recommendations unavailable'"
            )
            assert "All squad players are fully fit" not in missing_body, (
                "Falsely claimed all players fit on missing squad data!"
            )
            assert "Save your free transfer" not in missing_body, "Falsely recommended rolling on absent plan!"
            assert not find_emojis(missing_body), "Found emojis in missing squad view"

            dash_empty_path = SCREENSHOTS_DIR / "dashboard_empty_squad_desktop.png"
            page.screenshot(path=str(dash_empty_path), full_page=True)
            print(f"  Captured: {dash_empty_path.name} ({dash_empty_path.stat().st_size:,} bytes)")
            captured_manifest["screenshots"]["dashboard_empty_squad_desktop"] = {
                "file": dash_empty_path.name,
                "size_bytes": dash_empty_path.stat().st_size,
                "dimensions": "1920x1080",
                "state": "empty_squad_intentional",
            }

            # Mobile view of missing squad
            page.set_viewport_size({"width": 390, "height": 844})
            page.wait_for_timeout(500)
            dash_empty_m_path = SCREENSHOTS_DIR / "dashboard_empty_squad_mobile.png"
            page.screenshot(path=str(dash_empty_m_path), full_page=True)
            print(f"  Captured: {dash_empty_m_path.name} ({dash_empty_m_path.stat().st_size:,} bytes)")
            captured_manifest["screenshots"]["dashboard_empty_squad_mobile"] = {
                "file": dash_empty_m_path.name,
                "size_bytes": dash_empty_m_path.stat().st_size,
                "dimensions": "390x844",
                "state": "empty_squad_intentional",
            }

            # ============================================================
            # STATE 4: INTENTIONAL FAILED API / GRACEFUL ERROR STATE
            # ============================================================
            print("\n[State 4: Intentional Failed API / Graceful Error State]")
            page.set_viewport_size({"width": 1920, "height": 1080})

            page.evaluate("""() => {
                window.__fpl_vm__.squadData = {};
                window.__fpl_vm__.decisionCard = null;
                window.__fpl_vm__.contingencyPlans = {};
                window.__fpl_vm__.squadError = "Network error: 500 Internal Server Error";
                window.__fpl_vm__.decisionCardError = "Network error loading recommendations";
                window.__fpl_vm__.plansError = "Network error loading transfer plans";
                window.__fpl_vm__.squadLoading = false;
                window.__fpl_vm__.plansLoading = false;
                window.__fpl_vm__.decisionCardLoading = false;
            }""")
            page.wait_for_timeout(1000)

            error_body = page.query_selector("body").inner_text()

            # Asserts graceful fallback without leaking raw traceback
            assert "Traceback (most recent call last)" not in error_body, "Raw Python traceback leaked in UI!"
            assert "Transfer recommendations unavailable" in error_body
            assert not find_emojis(error_body), "Found emojis in failed API view"

            dash_error_path = SCREENSHOTS_DIR / "dashboard_failed_api_desktop.png"
            page.screenshot(path=str(dash_error_path), full_page=True)
            print(f"  Captured: {dash_error_path.name} ({dash_error_path.stat().st_size:,} bytes)")
            captured_manifest["screenshots"]["dashboard_failed_api_desktop"] = {
                "file": dash_error_path.name,
                "size_bytes": dash_error_path.stat().st_size,
                "dimensions": "1920x1080",
                "state": "failed_api_intentional",
            }

            page.set_viewport_size({"width": 390, "height": 844})
            page.wait_for_timeout(500)
            dash_error_m_path = SCREENSHOTS_DIR / "dashboard_failed_api_mobile.png"
            page.screenshot(path=str(dash_error_m_path), full_page=True)
            print(f"  Captured: {dash_error_m_path.name} ({dash_error_m_path.stat().st_size:,} bytes)")
            captured_manifest["screenshots"]["dashboard_failed_api_mobile"] = {
                "file": dash_error_m_path.name,
                "size_bytes": dash_error_m_path.stat().st_size,
                "dimensions": "390x844",
                "state": "failed_api_intentional",
            }

            # ============================================================
            # STATE 5: INTENTIONAL EMPTY LEAGUE STANDINGS STATE
            # ============================================================
            print("\n[State 5: Intentional Empty League Standings State]")
            page.set_viewport_size({"width": 1920, "height": 1080})

            page.evaluate("""() => {
                window.__fpl_vm__.activeTab = 'league';
                window.__fpl_vm__.leagueData = {
                    league_name: 'Unconfigured League',
                    standings: [],
                    simulation: null,
                    strategy: { mode_title: 'Unconfigured' }
                };
            }""")
            page.wait_for_timeout(1000)

            dash_league_path = SCREENSHOTS_DIR / "dashboard_empty_league_desktop.png"
            page.screenshot(path=str(dash_league_path), full_page=True)
            print(f"  Captured: {dash_league_path.name} ({dash_league_path.stat().st_size:,} bytes)")
            captured_manifest["screenshots"]["dashboard_empty_league_desktop"] = {
                "file": dash_league_path.name,
                "size_bytes": dash_league_path.stat().st_size,
                "dimensions": "1920x1080",
                "state": "empty_league_intentional",
            }

            # ============================================================
            # STATE 6: INTENTIONAL LOADING STATE
            # ============================================================
            print("\n[State 6: Intentional Loading State]")
            page.evaluate("""() => {
                window.__fpl_vm__.activeTab = 'overview';
                window.__fpl_vm__.squadLoading = true;
                window.__fpl_vm__.plansLoading = true;
                window.__fpl_vm__.decisionCardLoading = true;
            }""")
            page.wait_for_timeout(500)

            loading_body = page.query_selector("body").inner_text()
            assert "Loading squad availability data..." in loading_body
            assert "Loading recommendations..." in loading_body
            assert not find_emojis(loading_body), "Found emojis in loading view"

            dash_loading_path = SCREENSHOTS_DIR / "dashboard_loading_desktop.png"
            page.screenshot(path=str(dash_loading_path), full_page=True)
            print(f"  Captured: {dash_loading_path.name} ({dash_loading_path.stat().st_size:,} bytes)")
            captured_manifest["screenshots"]["dashboard_loading_desktop"] = {
                "file": dash_loading_path.name,
                "size_bytes": dash_loading_path.stat().st_size,
                "dimensions": "1920x1080",
                "state": "loading_intentional",
            }

            # Close page and browser
            page.close()
            browser.close()
            print("\nBrowser execution closed successfully.")

    finally:
        print("Terminating server process...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        print("Server shutdown complete.")

    # Save manifest.json
    manifest_path = SCREENSHOTS_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(captured_manifest, f, indent=2)
    print(f"Wrote screenshot metadata manifest to: {manifest_path.name}")

    # Remove temporary scratch files if any
    for scratch_file in [
        ROOT / "scratch" / "test_route_pw.py",
        ROOT / "scratch" / "test_vm_pw.py",
        ROOT / "scratch" / "test_vm_detail.py",
    ]:
        if scratch_file.exists():
            try:
                scratch_file.unlink()
            except Exception:
                pass

    print("=" * 65)
    print("SUCCESS: All visual screenshots and truthfulness assertions passed!")
    print("=" * 65)


if __name__ == "__main__":
    capture_screenshots()
