#!/usr/bin/env python3

import argparse
import csv
import sys
import time
import os
import json
import requests
import tempfile
from typing import List, Dict, Any, Tuple
from datetime import date, timedelta, datetime

DEFAULT_FIELDS = [
    "key", "summary", "status", "assignee", "reporter",
    "issuetype", "created", "updated", "comment", "project",
    "priority", "resolution", "resolutiondate"
]

# Team members dictionary from the Flask app
TEAM_MEMBERS_DICT = {
    'SQL': ['mythili.thangavel'],
    'FIT': ['riswana.shagul', 'uma.subramani'],
    'Qual': ['gopal.sampathi', 'araut.thinkpalm', 'yuvaraj.selvam', 'revathi.balachandran'],
    'Yoda': ['rajasekar.appusamy'],
    'Imanis': ['jishnu.jayasree', 'satheesh.ayyappan', 'mathew.joseph', 'prabakaransarangu.s'],
    'Nexus': ['archana.balaguru', 'ulagammal.essaki', 'muthukaruppan.a', 'jayaprakash.b', 'andry.precika2'],
    'SAP': ['ajeeth.kumar', 'dinesh.selvaraj'],
    'Cloud Infra': ['saiprakash.reddy', 'lakshmanan.kannan'],
    'IRIS': ['jayapriya.subramanian', 'vishnupriya.ravichandran'],
    'Cloud': ['sandeep.sriram', 'rajendran.ravichandran'],
    'Eightfold': ['Sabarinathan Balu', 'Kalaivani K', 'dsampath', 'Umamageswari Balaganesan']
}


def get_previous_week_range() -> Tuple[str, str]:
    """Return start and end dates (YYYY-MM-DD) for the previous calendar week (Mon-Sun)."""
    today = date.today()
    # Monday=0 ... Sunday=6
    start_of_this_week = today - timedelta(days=today.weekday())
    start_prev = start_of_this_week - timedelta(days=7)
    end_prev = start_of_this_week - timedelta(days=1)
    return start_prev.strftime('%Y-%m-%d'), end_prev.strftime('%Y-%m-%d')


def build_jql(weeks: int, assignees: List[str], projects: List[str], jql_override: str = "", start_date: str = "", end_date: str = "", team: str = "") -> str:
    if jql_override:
        return jql_override
    
    clauses = []
    
    # Always include all team members from all teams
    all_team_members = []
    for team_members in TEAM_MEMBERS_DICT.values():
        all_team_members.extend(team_members)
    
    if all_team_members:
        # For bugs: reporter in all team members, for non-bugs: assignee in all team members
        member_list = ', '.join([f'"{member}"' for member in all_team_members])
        bug_clause = f'(issuetype = Bug AND reporter in ({member_list}))'
        nonbug_clause = f'(issuetype != Bug AND assignee in ({member_list}))'
        clauses.append(f'({bug_clause} OR {nonbug_clause})')
    
    # Add resolved status filter
    clauses.append('status = Resolved')
    
    # Add date filters for when tickets were resolved
    if start_date:
        clauses.append(f'"resolutiondate" >= "{start_date}"')
    if end_date:
        clauses.append(f'"resolutiondate" <= "{end_date}"')
    if not start_date and not end_date:
        clauses.append(f"resolutiondate >= -{weeks}w")
    
    # Add project filters if specified
    if projects:
        proj = ", ".join(projects)
        clauses.append(f"project in ({proj})")
    
    # If no clauses were added, add a basic one
    if not clauses:
        clauses.append("resolutiondate >= -1w")
    
    return " AND ".join(clauses)


def load_settings() -> Dict[str, Any]:
    """Load Jira settings from known locations."""
    candidates = [
        os.path.join(os.path.dirname(__file__), 'flask_app', 'settings.json'),
        os.path.join(os.path.dirname(__file__), 'settings.json'),
    ]
    for path in candidates:
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass
    return {}


def resolve_config(args: argparse.Namespace) -> Tuple[str, str, str, str]:
    """Resolve server, username, token, output filename using precedence: CLI > env > settings > defaults."""
    settings = load_settings()

    server = (
        args.server
        or os.environ.get('JIRA_SERVER')
        or settings.get('server_url')
        or 'https://jira.cohesity.com/'
    )
    username = (
        args.username
        or os.environ.get('JIRA_USERNAME')
        or settings.get('username')
        or ''
    )
    token = (
        args.token
        or os.environ.get('JIRA_TOKEN')
        or settings.get('password')
        or ''
    )
    # Derive a safe basename for output
    output_basename_raw = (args.output or '').strip()
    output_basename = os.path.basename(output_basename_raw)
    if not output_basename or output_basename in {'.', '/', 'Weekly_Data'}:
        # Include team name in filename if specified, otherwise show it's all engineers
        team_suffix = f"_{args.team}" if args.team else "_all_engineers"
        output_basename = f"jira_tickets{team_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    # Ensure .csv extension
    if not output_basename.lower().endswith('.csv'):
        output_basename += '.csv'

    return server, username, token, output_basename


def ensure_project_weekly_dir(project_root: str) -> str:
    """Ensure and return the 'Weekly_Data' directory under the project root. Raise on failure."""
    weekly_dir = os.path.join(project_root, 'Weekly_Data')
    # If path exists but not a directory, error
    if os.path.exists(weekly_dir) and not os.path.isdir(weekly_dir):
        raise PermissionError(f"Path exists and is not a directory: {weekly_dir}")
    # Create if missing
    try:
        os.makedirs(weekly_dir, exist_ok=True)
    except Exception as e:
        raise PermissionError(f"Cannot create Weekly_Data directory at {weekly_dir}: {e}")
    # Check writable
    if not os.access(weekly_dir, os.W_OK):
        raise PermissionError(f"Weekly_Data directory is not writable: {weekly_dir}")
    return weekly_dir


def fetch_issues(
    server: str,
    username: str,
    token: str,
    jql: str,
    fields: List[str] = DEFAULT_FIELDS,
    expand: List[str] = ["comments"],
    page_size: int = 100,
    max_results: int = 10000,
    request_timeout: int = 60,
) -> List[Dict[str, Any]]:
    url = f"{server.rstrip('/')}/rest/api/2/search"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }

    all_issues: List[Dict[str, Any]] = []
    start_at = 0

    while start_at < max_results:
        remaining = max_results - start_at
        batch_size = page_size if remaining > page_size else remaining
        payload = {
            "jql": jql,
            "startAt": start_at,
            "maxResults": batch_size,
            "fields": fields,
            "expand": expand,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=request_timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"Jira API error: {resp.status_code} - {resp.text}")
        data = resp.json()
        issues = data.get("issues", [])
        all_issues.extend(issues)
        if len(issues) < batch_size:
            break
        start_at += batch_size
        # avoid rate-limit bursts
        time.sleep(0.1)

    return all_issues


def normalize_user(display_obj: Dict[str, Any]) -> str:
    if not display_obj:
        return ""
    return display_obj.get("displayName", "")


def flatten_issue(issue: Dict[str, Any]) -> Dict[str, Any]:
    fields = issue.get("fields", {})

    comments = fields.get("comment", {}).get("comments", [])
    comment_text = " ".join([c.get("body", "") for c in comments])

    row = {
        "Issue key": issue.get("key", ""),
        "Summary": fields.get("summary", ""),
        "Status": (fields.get("status") or {}).get("name", ""),
        "Assignee": normalize_user(fields.get("assignee")),
        "Reporter": normalize_user(fields.get("reporter")),
        "Issue Type": (fields.get("issuetype") or {}).get("name", ""),
        "Created": fields.get("created", ""),
        "Updated": fields.get("updated", ""),
        "Resolved": fields.get("resolutiondate", ""),
        "Resolution": (fields.get("resolution") or {}).get("name", ""),
        "Project key": (fields.get("project") or {}).get("key", ""),
        "Priority": (fields.get("priority") or {}).get("name", ""),
        "Comment": comment_text,
    }
    return row


def write_csv(rows: List[Dict[str, Any]], output_path: str) -> int:
    if os.path.isdir(output_path):
        # Safety: if somehow a directory path is passed, switch to a new filename
        output_path = os.path.join(output_path, f"jira_tickets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    if not rows:
        # write empty with headers
        headers = [
            "Issue key", "Summary", "Status", "Assignee", "Reporter",
            "Issue Type", "Created", "Updated", "Resolved", "Resolution",
            "Project key", "Priority", "Comment",
        ]
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
        return 0

    headers = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Jira tickets for all team engineers and export to CSV")
    parser.add_argument("--server", help="Jira server URL, e.g. https://jira.example.com/")
    parser.add_argument("--username", help="Jira username/email")
    parser.add_argument("--token", help="Jira API token (PAT)")
    parser.add_argument("--team", choices=list(TEAM_MEMBERS_DICT.keys()), help="Optional: Specific team name to filter data (default: all teams)")
    parser.add_argument("--weeks", type=int, default=4, help="Number of weeks back for 'resolution' filter (default: 4). Ignored if start/end date provided.")
    parser.add_argument("--start-date", dest="start_date", default="", help="Resolution start date (YYYY-MM-DD). Overrides weeks when used.")
    parser.add_argument("--end-date", dest="end_date", default="", help="Resolution end date (YYYY-MM-DD). Overrides weeks when used.")
    parser.add_argument("--previous-week", action="store_true", help="Fetch only the previous calendar week (Mon-Sun). Overrides weeks and start/end.")
    parser.add_argument("--assignee", action="append", default=[], help="Assignee name (can be specified multiple times)")
    parser.add_argument("--project", action="append", default=[], help="Project key (can be specified multiple times)")
    parser.add_argument("--jql", default="", help="JQL override. If supplied, it will be used as-is.")
    parser.add_argument("--output", help="Output CSV filename (basename). The file will be saved under the 'Weekly_Data' folder.")
    parser.add_argument("--max", type=int, default=10000, help="Max issues to fetch (default: 10000)")
    parser.add_argument("--page-size", type=int, default=100, help="Page size per API call (default: 100)")
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    # Resolve config with fallbacks
    server, username, token, output_basename = resolve_config(args)

    if not username or not token:
        print("Error: Jira credentials are not set. Provide via CLI, env (JIRA_USERNAME/JIRA_TOKEN), or settings.json.", file=sys.stderr)
        return 1

    # Determine date filters. Default to previous week if nothing provided and no JQL override
    if args.previous_week or (not args.start_date and not args.end_date and not args.jql):
        start_date_str, end_date_str = get_previous_week_range()
    else:
        start_date_str, end_date_str = args.start_date, args.end_date

    jql = build_jql(args.weeks, args.assignee, args.project, args.jql, start_date_str, end_date_str, args.team)
    
    # Print the JQL query for debugging
    print(f"JQL Query: {jql}")

    try:
        issues = fetch_issues(
            server=server,
            username=username,
            token=token,
            jql=jql,
            fields=DEFAULT_FIELDS,
            expand=["comments"],
            page_size=args.page_size,
            max_results=args.max,
        )
        rows = [flatten_issue(iss) for iss in issues]

        # Strictly use project root Weekly_Data folder
        project_root = os.path.dirname(os.path.abspath(__file__))
        weekly_dir = ensure_project_weekly_dir(project_root)

        # Build Week<weeknumber>_<year>.csv filename
        week_num = date.today().isocalendar()[1]
        year_num = date.today().year
        auto_name = f"Week{week_num}_{year_num}.csv"

        # Always write inside 'Weekly_Data' with the auto name
        output_path = os.path.join(weekly_dir, auto_name)
        if os.path.isdir(output_path):
            output_path = os.path.join(weekly_dir, f"jira_tickets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

        count = write_csv(rows, output_path)
        print(f"Fetched {len(issues)} issues; wrote {count} rows to {output_path}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:])) 