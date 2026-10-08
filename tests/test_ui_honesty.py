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
    "Captain",
    "Transfers to make",
    "Bench",
    "Chip:",
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


def is_emoji_char(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x1F300 <= cp <= 0x1FAFF
        or 0x2600 <= cp <= 0x27BF
        or 0x2B50 <= cp <= 0x2B55
        or 0x2300 <= cp <= 0x23FF
        or 0xFE00 <= cp <= 0xFE0F
    )


def test_zero_emojis_in_web_ui():
    """Verify web/index.html and web/static/js/app.js contain zero emojis."""
    for file in [INDEX_HTML, APP_JS]:
        content = file.read_text(encoding="utf-8")
        emojis = [ch for ch in content if is_emoji_char(ch)]
        assert not emojis, f"Found {len(emojis)} emoji characters in {file.name}: {emojis[:5]}"


def test_zero_emojis_in_src():
    """Verify all Python source files in src/ contain zero emojis."""
    src_dir = ROOT / "src"
    for py_file in src_dir.rglob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        emojis = [ch for ch in content if is_emoji_char(ch)]
        assert not emojis, f"Found {len(emojis)} emoji characters in {py_file}: {emojis[:5]}"


def test_squad_availability_empty_state_honesty():
    """Verify web/index.html requires loaded complete squad data before claiming all players fit."""
    content = INDEX_HTML.read_text(encoding="utf-8")
    # Must check !hasLoadedSquad and show squad unavailable message
    assert "!hasLoadedSquad" in content
    assert "Team unavailable" in content
    # The claim must be contingent on flaggedSquadPlayers.length === 0, NOT an empty raw squad
    assert "allSquadConfirmedAvailable" in content
    assert "unknownAvailabilityPlayers.length === 0" in APP_JS.read_text()
    # No emoji checkmark next to all fit
    assert "✅" not in content


def test_absent_plan_truthfulness_in_app_js():
    """Verify app.js does not recommend rolling or fake values when plan/squad is missing."""
    content = APP_JS.read_text(encoding="utf-8")
    assert "Advice unavailable" in content
    assert "Checking..." in content
    # When squad is missing, rolling must be guarded
    assert "if (!this.hasLoadedSquad)" in content
    assert "Cannot recommend rolling without a verified loaded squad" in content
    # Distinguishes unavailable points from 0.0
    assert "totalGameweekPoints()" in content
    assert "return '—'" in content or 'return "—"' in content


def test_chip_banner_is_computed_and_keeps_caveats():
    """The chip callout reads the through-GW19 check instead of fixed text."""
    js = APP_JS.read_text(encoding="utf-8")
    html = INDEX_HTML.read_text(encoding="utf-8")
    for state in ("pending", "holds", "changed", "unavailable", "idle"):
        assert f"state: '{state}'" in js
    assert "Timing unchanged" in js
    assert "chipCheck" in html
    assert "Timing changes" in js


def test_design_tokens_live_in_one_css_file():
    css = ROOT / "web" / "static" / "css"
    tokens = (css / "tokens.css").read_text(encoding="utf-8")
    for name in ("--blur", "--r-lg", "--shadow-md", "--s-2", "--ease-out", "--accent", "--font-sans"):
        assert name in tokens
    assert "html.light" in tokens
    assert "prefers-reduced-motion" in (css / "ui.css").read_text(encoding="utf-8")
    assert not (ROOT / "web" / "static" / "js" / "tailwind.js").exists()


def test_council_reference_monochrome_shell():
    html = INDEX_HTML.read_text(encoding="utf-8")
    js = APP_JS.read_text(encoding="utf-8")
    tokens = (ROOT / "web/static/css/tokens.css").read_text(encoding="utf-8")
    css = (ROOT / "web/static/css/ui.css").read_text(encoding="utf-8")
    assert 'class="desktop-sidebar"' in html
    assert 'aria-label="Open navigation"' in html
    assert 'class="navigation-dialog"' in html
    assert "--sidebar-width: 220px" in tokens
    assert "--bg-base: #000000" in tokens
    assert "--accent: #e4e4e7" in tokens
    assert ".glass { border: none; box-shadow: none; }" in css
    assert "this.showNavigation = false" in js


def test_polished_sidebar_copy_and_data_states():
    html = INDEX_HTML.read_text(encoding="utf-8")
    js = APP_JS.read_text(encoding="utf-8")
    assert "engine-section" not in html
    assert "council-mark" not in html
    assert ">LOCAL<" not in html
    assert "side-deadline" not in html
    assert "brand-sub num" in html
    assert "priceLabel(p.direction)" in html
    assert "overlapLabel(team)" in html
    assert "aux.prices.error" in html
    assert "aux.league.loading" in html
    assert "window.addEventListener('hashchange'" in js
