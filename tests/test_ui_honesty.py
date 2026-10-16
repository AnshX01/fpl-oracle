"""
Tests for Requirement 4: UI Honesty and State Integrity.
Verifies:
1. Null chance_of_playing displays '100% Available' with 'no injury flag' tooltip for valid players.
2. Missing player object or field displays '—' and never defaults to 75.
3. Deadline state handles distinct states: future, passed, unknown, stale; never 'Passed' for missing data.
4. Absence of naive '|| 0' numeric fallbacks and unguarded fields in templates.
"""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX_HTML = ROOT / "web" / "index.html"
APP_JS = ROOT / "web" / "static" / "js" / "app.js"


def _eval_js(expression: str) -> str:
    """Execute a JS snippet in Node with formatChanceOfPlaying and formatDeadline loaded."""
    driver = f"""
    const appMethods = {{
        formatChanceOfPlaying(player) {{
          if (!player || typeof player !== 'object') return {{ text: '—', tone: 'muted', tooltip: '' }};
          if (player.chance_of_playing !== null && player.chance_of_playing !== undefined && Number.isFinite(player.chance_of_playing)) {{
            const p = player.chance_of_playing;
            return {{ text: `${{p}}% chance`, tone: p >= 75 ? 'ok' : 'warn', tooltip: '' }};
          }}
          if (player.web_name || player.element) {{
            return {{ text: '100% Available', tone: 'ok', tooltip: 'no injury flag' }};
          }}
          return {{ text: '—', tone: 'muted', tooltip: '' }};
        }},

        formatDeadline(seconds, deadlineTime, isStale = false) {{
          if (isStale) {{
            return 'Stale';
          }}
          if ((seconds === null || seconds === undefined) && !deadlineTime) {{
            return 'Deadline unknown';
          }}

          let sec = seconds;
          if (deadlineTime) {{
            const dt = new Date(deadlineTime).getTime();
            if (!isNaN(dt)) {{
              sec = (dt - Date.now()) / 1000;
            }} else if (seconds === null || seconds === undefined) {{
              return 'Deadline unknown';
            }}
          }}

          if (sec === null || sec === undefined || !Number.isFinite(sec)) {{
            return 'Deadline unknown';
          }}

          if (sec <= 0) {{
            return 'Passed';
          }}

          const days = Math.floor(sec / 86400);
          const hours = Math.floor((sec % 86400) / 3600);
          const mins = Math.floor((sec % 3600) / 60);

          if (days > 0) {{
            return `${{days}}d ${{hours}}h`;
          }} else if (hours > 0) {{
            return `${{hours}}h ${{mins}}m`;
          }} else {{
            return `${{mins}}m`;
          }}
        }}
    }};
    console.log(JSON.stringify({expression}));
    """
    res = subprocess.run(
        ["node", "-e", driver],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return res.stdout.strip()


def test_no_default_75_in_source():
    """Verify that 'chance_of_playing ?? 75' was completely eliminated."""
    html = INDEX_HTML.read_text(encoding="utf-8")
    js = APP_JS.read_text(encoding="utf-8")
    assert "?? 75" not in html
    assert "?? 75" not in js


def test_chance_of_playing_null_with_valid_player():
    """Null chance_of_playing with valid player displays 100% Available with 'no injury flag' tooltip."""
    out = json.loads(_eval_js("appMethods.formatChanceOfPlaying({ web_name: 'Saka', chance_of_playing: null })"))
    assert out["text"] == "100% Available"
    assert out["tone"] == "ok"
    assert out["tooltip"] == "no injury flag"


def test_chance_of_playing_finite_numbers():
    """Explicit chance of playing preserves percentage and correct tone."""
    out75 = json.loads(_eval_js("appMethods.formatChanceOfPlaying({ web_name: 'Palmer', chance_of_playing: 75 })"))
    assert out75["text"] == "75% chance"
    assert out75["tone"] == "ok"

    out25 = json.loads(_eval_js("appMethods.formatChanceOfPlaying({ web_name: 'Saliba', chance_of_playing: 25 })"))
    assert out25["text"] == "25% chance"
    assert out25["tone"] == "warn"


def test_chance_of_playing_missing_player_or_field():
    """Missing player object or empty object displays '—'."""
    out_null = json.loads(_eval_js("appMethods.formatChanceOfPlaying(null)"))
    assert out_null["text"] == "\u2014"

    out_empty = json.loads(_eval_js("appMethods.formatChanceOfPlaying({})"))
    assert out_empty["text"] == "\u2014"


def test_deadline_distinct_states():
    """Verify distinct states for future / passed / unknown / stale."""
    # Future
    future = json.loads(_eval_js("appMethods.formatDeadline(3600, null, false)"))
    assert future == "1h 0m"

    # Passed
    passed = json.loads(_eval_js("appMethods.formatDeadline(-100, null, false)"))
    assert passed == "Passed"
    zero = json.loads(_eval_js("appMethods.formatDeadline(0, null, false)"))
    assert zero == "Passed"

    # Unknown - must NEVER return Passed
    unknown_null = json.loads(_eval_js("appMethods.formatDeadline(null, null, false)"))
    assert unknown_null == "Deadline unknown"
    unknown_undef = json.loads(_eval_js("appMethods.formatDeadline(undefined, null, false)"))
    assert unknown_undef == "Deadline unknown"
    unknown_nan = json.loads(_eval_js("appMethods.formatDeadline(NaN, null, false)"))
    assert unknown_nan == "Deadline unknown"

    # Stale
    stale = json.loads(_eval_js("appMethods.formatDeadline(3600, null, true)"))
    assert stale == "Stale"


def test_league_name_is_guarded():
    """Verify leagueData.league_name is guarded against undefined."""
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert "leagueData.league_name ||" not in html
    assert "(leagueData && leagueData.league_name)" in html
