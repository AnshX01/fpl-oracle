"""Record evidence for G3 (UI honesty, deadline logic, Playwright screenshots)."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE_FILE = ROOT / "reports" / "evidence" / "G3-ui.txt"
EVIDENCE_FILE.parent.mkdir(parents=True, exist_ok=True)
PYTHON_EXE = sys.executable

lines = []
lines.append("=== G3 UI HONESTY, DEADLINE & ATLAS/COUNCIL LOOK VERIFICATION ===\n\n")

# 1. Run pytest tests/test_ui_honesty.py
cmd1 = [PYTHON_EXE, "-m", "pytest", "tests/test_ui_honesty.py", "-v"]
cmd1_str = " ".join(cmd1)
print(f"Running: {cmd1_str}")
lines.append(f"COMMAND: {cmd1_str}\n")
p1 = subprocess.run(cmd1, capture_output=True, text=True, errors="replace", cwd=ROOT)
lines.append(f"EXIT CODE: {p1.returncode}\n")
lines.append("OUTPUT:\n" + (p1.stdout or "") + (p1.stderr or "") + "\n")
assert p1.returncode == 0, f"pytest failed with code {p1.returncode}"

# 2. Run capture_screenshots.py
cmd2 = [PYTHON_EXE, "scripts/capture_screenshots.py"]
cmd2_str = " ".join(cmd2)
print(f"Running: {cmd2_str}")
lines.append(f"COMMAND: {cmd2_str}\n")
p2 = subprocess.run(cmd2, capture_output=True, text=True, errors="replace", cwd=ROOT)
lines.append(f"EXIT CODE: {p2.returncode}\n")
lines.append("OUTPUT:\n" + (p2.stdout or "") + (p2.stderr or "") + "\n")
assert p2.returncode == 0, f"capture_screenshots.py failed with code {p2.returncode}"

# 3. List screenshot files
screenshots_dir = ROOT / "reports" / "screenshots"
lines.append("SCREENSHOT ARTIFACTS IN reports/screenshots/:\n")
for p in sorted(screenshots_dir.glob("*.png")):
    lines.append(f"  {p.name}: {p.stat().st_size:,} bytes\n")

output_text = "".join(lines)
EVIDENCE_FILE.write_text(output_text, encoding="utf-8")
print(f"Successfully generated {EVIDENCE_FILE}")
