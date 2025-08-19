#!/usr/bin/env python3

import os
import re
import sys
import glob
import time
import logging
import subprocess
from datetime import date
from typing import Optional, Tuple


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)


def project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def ensure_weekly_dir(root: str) -> str:
    weekly = os.path.join(root, "Weekly_Data")
    os.makedirs(weekly, exist_ok=True)
    return weekly


def latest_week_csv(weekly_dir: str) -> Optional[str]:
    """Return path to the newest Week<week>_<year>.csv by (year, week) ordering."""
    best: Tuple[int, int, str] = (-1, -1, "")
    for fname in os.listdir(weekly_dir):
        m = re.match(r"^Week(\d+)_(\d+)\s*\.csv$", fname)
        if not m:
            continue
        w = int(m.group(1))
        y = int(m.group(2))
        if (y, w) > (best[0], best[1]):
            best = (y, w, fname)
    return os.path.join(weekly_dir, best[2]) if best[2] else None


def run_jira_to_csv(py_exe: str, root: str) -> None:
    """Invoke fetch_jira_resolved.py to produce the weekly CSV in Weekly_Data."""
    script = os.path.join(root, "fetch_jira_resolved.py")
    # Intentionally no args: script resolves credentials/settings and defaults
    # to previous week window but writes to Weekly_Data/Week<week>_<year>.csv
    cmd = [py_exe, script]
    logging.info("Running Jira fetch: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=root, check=True)


def run_gerrit_json(py_exe: str, root: str, csv_path: str) -> None:
    """Invoke fetch_gerrit_comments.py to consolidate Gerrit comments to JSON."""
    script = os.path.join(root, "fetch_gerrit_comments.py")
    # Output argument is required by the script interface but the script writes
    # Week<week>_<year>.json under Weekly_Data. Pass a placeholder.
    placeholder_out = os.path.join(root, "Weekly_Data", "consolidated.json")
    cmd = [py_exe, script, "--csv", csv_path, "--output", placeholder_out]
    logging.info("Running Gerrit consolidation: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=root, check=True)


def main() -> int:
    root = project_root()
    weekly = ensure_weekly_dir(root)

    # Use the same Python interpreter by default
    py_exe = sys.executable or sys.argv[0] or "python3"

    # Step 1: Generate weekly CSV from Jira
    run_jira_to_csv(py_exe, root)

    # Small wait for filesystem visibility (cron/network FS safety)
    time.sleep(1)

    # Discover the newest CSV produced
    csv_file = latest_week_csv(weekly)
    if not csv_file or not os.path.exists(csv_file):
        logging.error("Weekly CSV not found in %s after Jira fetch.", weekly)
        return 2
    logging.info("Using weekly CSV: %s", csv_file)

    # Step 2: Generate consolidated Gerrit JSON using the CSV
    run_gerrit_json(py_exe, root, csv_file)

    # Optionally log where the Week JSON is expected
    week_num = date.today().isocalendar()[1]
    year_num = date.today().year
    expected_json = os.path.join(weekly, f"Week{week_num}_{year_num}.json")
    if os.path.exists(expected_json):
        logging.info("Generated weekly JSON: %s", expected_json)
    else:
        # Best-effort detect any Week*.json created
        candidates = sorted(
            glob.glob(os.path.join(weekly, "Week*_*.json")), key=os.path.getmtime, reverse=True
        )
        if candidates:
            logging.info("Generated weekly JSON: %s", candidates[0])
        else:
            logging.warning("No Week*.json found in %s. Check Gerrit credentials/settings.", weekly)

    logging.info("Weekly pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


