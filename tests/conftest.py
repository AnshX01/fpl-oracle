import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Tests exercise quarantine/rollback/promotion; never let them touch the served bundle.
# Same for the profile DB, caches and history: tests write manual squads and bank/FT overrides.
_DATA_COPY = Path(tempfile.mkdtemp(prefix="fpl_data_")) / "data"
shutil.copytree(ROOT / "data", _DATA_COPY, ignore=shutil.ignore_patterns("*.db", "profile.json", "cache", "snapshots"))
os.environ["FPL_ORACLE_DATA_DIR"] = str(_DATA_COPY)

pytest_plugins = ["tests.repair_fixtures"]
