"""Automated UI Screenshot Capture Script using Playwright Chromium (F14 & G3).

Spawns the local FastAPI server, asserts HTTP 200 on key API endpoints,
asserts non-empty recommendation elements in the UI, and captures:
- 1920x1080 desktop dashboard
- 1920x1080 desktop decision card
- 390x844 mobile dashboard
- 390x844 mobile decision card
Saved directly into reports/screenshots/.
"""

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
    "/api/v1/health",
    "/api/v1/squad/current",
    "/api/v1/briefing/decision-card",
    "/api/v1/transfers/plans",
]


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


def verify_endpoints():
    """Verify HTTP 200 on all required endpoints."""
    print("Verifying backend API endpoints:")
    for path in ENDPOINTS_TO_CHECK:
        url = f"{SERVER_URL}{path}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "FPL-Oracle-Screenshot-Test"})
            with urllib.request.urlopen(req, timeout=120.0) as resp:
                status = resp.status
                assert status == 200, f"Endpoint {path} returned status {status}"
                data_len = len(resp.read())
                print(f"  [OK 200] {path} ({data_len:,} bytes)")
        except Exception as e:
            print(f"  [FAILED] {path}: {e}")
            raise


def capture_screenshots():
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    python_exe = sys.executable

    print("=" * 60)
    print("FPL ORACLE -- RESPONSIVE UI VERIFICATION & SCREENSHOTS (G3)")
    print("=" * 60)

    # 1. Start backend server
    print(f"Starting uvicorn server on {SERVER_URL}...")
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
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        print("Waiting for server startup...")
        if not wait_for_server():
            print("ERROR: Server failed to start within timeout.")
            sys.exit(1)
        print("Server is up and responsive!\n")

        # Verify backend endpoints
        verify_endpoints()
        print("")

        # 2. Launch headless browser
        with sync_playwright() as p:
            print("Launching headless Chromium...")
            browser = p.chromium.launch(headless=True)

            # --- DESKTOP VIEWPORT (1920x1080) ---
            print("\n[Desktop Viewport: 1920x1080]")
            context_desktop = browser.new_context(viewport={"width": 1920, "height": 1080})
            page_desktop = context_desktop.new_page()

            page_desktop.goto(SERVER_URL, wait_until="domcontentloaded", timeout=45000)
            page_desktop.wait_for_selector("#decision-card", timeout=45000)
            page_desktop.wait_for_timeout(3000)  # Wait for Vue to finish reactive render

            # Assert recommendation element exists and is non-empty
            card_el = page_desktop.query_selector("#decision-card")
            assert card_el is not None, "Decision card element '#decision-card' not found in DOM"
            card_text = card_el.inner_text().strip()
            assert len(card_text) > 20, f"Decision card content is unexpectedly empty or short: {card_text}"
            print(f"  Decision card rendered with {len(card_text)} characters of text content.")

            # Capture desktop dashboard
            dash_desktop_path = SCREENSHOTS_DIR / "dashboard_desktop.png"
            page_desktop.screenshot(path=str(dash_desktop_path), full_page=True)
            print(f"  Captured: {dash_desktop_path.name} ({dash_desktop_path.stat().st_size:,} bytes)")

            # Also maintain dashboard.png symlink/copy for compatibility
            (SCREENSHOTS_DIR / "dashboard.png").write_bytes(dash_desktop_path.read_bytes())

            # Capture desktop decision card (scoped to element)
            dc_desktop_path = SCREENSHOTS_DIR / "decision_card_desktop.png"
            card_el.screenshot(path=str(dc_desktop_path))
            print(f"  Captured: {dc_desktop_path.name} ({dc_desktop_path.stat().st_size:,} bytes)")
            (SCREENSHOTS_DIR / "decision_card.png").write_bytes(dc_desktop_path.read_bytes())

            # --- MOBILE VIEWPORT (390x844) ---
            print("\n[Mobile Viewport: 390x844]")
            page_desktop.set_viewport_size({"width": 390, "height": 844})
            page_desktop.wait_for_timeout(1500)

            # Assert recommendation element in mobile
            card_el_m = page_desktop.query_selector("#decision-card")
            assert card_el_m is not None, "Decision card element '#decision-card' not found in mobile DOM"
            card_text_m = card_el_m.inner_text().strip()
            assert len(card_text_m) > 20, f"Decision card mobile content is empty: {card_text_m}"

            # Capture mobile dashboard
            dash_mobile_path = SCREENSHOTS_DIR / "dashboard_mobile.png"
            page_desktop.screenshot(path=str(dash_mobile_path), full_page=True)
            print(f"  Captured: {dash_mobile_path.name} ({dash_mobile_path.stat().st_size:,} bytes)")

            # Capture mobile decision card (scoped to element)
            dc_mobile_path = SCREENSHOTS_DIR / "decision_card_mobile.png"
            card_el_m.screenshot(path=str(dc_mobile_path))
            print(f"  Captured: {dc_mobile_path.name} ({dc_mobile_path.stat().st_size:,} bytes)")

            context_desktop.close()
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

    print("=" * 60)
    print("SUCCESS: All visual screenshots and endpoint assertions passed!")
    print("=" * 60)


if __name__ == "__main__":
    capture_screenshots()
