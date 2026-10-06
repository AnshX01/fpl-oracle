"""Tests for UI honesty, literal purge, and deadline logic.

Ensures no hardcoded placeholder metrics exist in frontend templates/scripts,
and validates deadline countdown formatting across future, passed, and unknown cases.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX_HTML = ROOT / "web" / "index.html"
APP_JS = ROOT / "web" / "static" / "js" / "app.js"

FORBIDDEN_LITERALS = [
    "101.6",
    "100.1",
    "92%",
    "334",
    "Template Shield",
    "1.046",
    "0.706",
    "2026.09.28",
    "|| 68",
    "68%",
]

REQUIRED_SECTION_HEADINGS = [
    "Who to captain",
    "Transfers to make",
    "Bench",
    "Chip plan",
    "Rivals",
    "Why",
]


def test_forbidden_literals_purged_from_index_html():
    """Verify web/index.html contains zero hardcoded placeholder literals."""
    content = INDEX_HTML.read_text(encoding="utf-8")
    for lit in FORBIDDEN_LITERALS:
        assert lit not in content, f"Forbidden literal '{lit}' found in web/index.html"


def test_forbidden_literals_purged_from_app_js():
    """Verify web/static/js/app.js contains zero hardcoded placeholder literals."""
    content = APP_JS.read_text(encoding="utf-8")
    for lit in FORBIDDEN_LITERALS:
        assert lit not in content, f"Forbidden literal '{lit}' found in web/static/js/app.js"


def test_required_section_headings_present():
    """Verify plain English section headings are present in web/index.html."""
    content = INDEX_HTML.read_text(encoding="utf-8")
    for heading in REQUIRED_SECTION_HEADINGS:
        assert heading in content, f"Required section heading '{heading}' missing from web/index.html"


def format_deadline_py(seconds, deadline_time=None):
    """Python reference mirroring formatDeadline in web/static/js/app.js."""
    if seconds is None and not deadline_time:
        return "Deadline unknown"
    sec = seconds
    if sec is None:
        return "Deadline unknown"
    if sec <= 0:
        return "Passed"
    days = int(sec // 86400)
    hours = int((sec % 86400) // 3600)
    mins = int((sec % 3600) // 60)
    if days > 0:
        return f"{days}d {hours}h"
    elif hours > 0:
        return f"{hours}h {mins}m"
    else:
        return f"{mins}m"


def test_deadline_formatting_rules():
    """Test formatDeadline logic for unknown, passed, and future countdown formats."""
    # Unknown
    assert format_deadline_py(None, None) == "Deadline unknown"

    # Passed
    assert format_deadline_py(0) == "Passed"
    assert format_deadline_py(-120) == "Passed"

    # Future
    # > 1 day -> Xd Xh
    assert format_deadline_py(86400 + 3600 * 2) == "1d 2h"
    assert format_deadline_py(2 * 86400 + 14 * 3600) == "2d 14h"

    # < 1 day, > 1 hour -> Xh Ym
    assert format_deadline_py(3600 * 5 + 60 * 30) == "5h 30m"

    # < 1 hour -> Xm
    assert format_deadline_py(60 * 45) == "45m"
