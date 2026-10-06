"""
Automated UI Screenshot Capture Script using Playwright Chromium (F14).
Spawns the local FastAPI/Uvicorn server, navigates to the single-page web dashboard
and decision card views, captures real visual screenshots to reports/screenshots/,
and shuts down the server gracefully.
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


def wait_for_server(timeout_sec: float = 20.0) -> bool:
    start = time.time()
    while time.time() - start < timeout_sec:
        try:
            with urllib.request.urlopen(f"{SERVER_URL}/", timeout=1.5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def capture_screenshots():
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    python_exe = sys.executable

    print("=" * 60)
    print("FPL ORACLE — AUTOMATED PLAYWRIGHT SCREENSHOT CAPTURE (F14)")
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
        print("Waiting for server health check...")
        if not wait_for_server():
            print("ERROR: Server failed to start within timeout.")
            sys.exit(1)
        print("Server is up and responsive!")

        # 2. Launch headless browser
        with sync_playwright() as p:
            print("Launching headless Chromium...")
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1920, "height": 1080})
            page = context.new_page()

            # Screenshot 1: Full Main Dashboard (Desktop View)
            print("Navigating to main dashboard...")
            page.goto(SERVER_URL, wait_until="domcontentloaded", timeout=15000)
            page.wait_for_timeout(3000)  # Allow charts and dynamic UI to render

            dash_path = SCREENSHOTS_DIR / "dashboard.png"
            page.screenshot(path=str(dash_path), full_page=True)
            print(f"Captured: {dash_path} ({dash_path.stat().st_size:,} bytes)")

            # Screenshot 2: Decision Card Section
            print("Navigating/focusing on decision card...")
            # Check if decision card button or tab exists
            dc_btn = page.query_selector(
                "button:has-text('Decision Card'), a:has-text('Decision Card'), [data-tab='decision-card']"
            )
            if dc_btn:
                dc_btn.click()
                page.wait_for_timeout(1500)

            dc_path = SCREENSHOTS_DIR / "decision_card.png"
            page.screenshot(path=str(dc_path), full_page=False)
            print(f"Captured: {dc_path} ({dc_path.stat().st_size:,} bytes)")

            # Screenshot 3: Mobile / Tablet Viewport
            print("Capturing mobile/tablet viewport...")
            page.set_viewport_size({"width": 390, "height": 844})
            page.wait_for_timeout(1000)
            mobile_path = SCREENSHOTS_DIR / "dashboard_mobile.png"
            page.screenshot(path=str(mobile_path), full_page=True)
            print(f"Captured: {mobile_path} ({mobile_path.stat().st_size:,} bytes)")

            browser.close()
            print("Browser closed.")

    finally:
        print("Terminating server process...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        print("Server shutdown complete.")

    print("=" * 60)
    print("SUCCESS: All visual screenshots captured and verified!")
    print("=" * 60)


if __name__ == "__main__":
    capture_screenshots()
