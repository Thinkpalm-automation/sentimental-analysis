#!/usr/bin/env python3

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any

import requests
import pandas as pd
import subprocess
import shutil


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)


TEAM_MEMBERS_DICT: Dict[str, List[str]] = {
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


def load_settings() -> Dict[str, Any]:
    settings_path = os.path.expanduser('~/Documents/sentiment-analysis/settings.json')
    if os.path.exists(settings_path):
        try:
            with open(settings_path, 'r') as f:
                settings = json.load(f)
            logging.info("Loaded settings from %s", settings_path)
            return settings
        except Exception as e:
            logging.warning("Could not load settings: %s", e)
    return {}


def resolve_config(settings: Dict[str, Any]) -> Dict[str, str]:
    config = {
        'server': settings.get('server_url') or settings.get('jira_server') or os.environ.get('JIRA_SERVER', ''),
        'username': settings.get('username') or settings.get('jira_username') or os.environ.get('JIRA_USERNAME', ''),
        # Prefer settings.json for token; fallback to env var
        'token': settings.get('password') or settings.get('jira_token') or os.environ.get('JIRA_TOKEN', ''),
    }
    missing = [k for k, v in config.items() if not v]
    if missing:
        logging.error("Missing Jira configuration keys: %s", ", ".join(missing))
        logging.error("Expected settings.json keys: server_url, username, password")
        sys.exit(1)
    return config


def get_previous_day_range() -> (datetime, datetime, str):
    yesterday = datetime.now() - timedelta(days=1)
    start_dt = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
    end_dt = yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)
    date_str = yesterday.strftime('%Y%m%d')
    return start_dt, end_dt, date_str


def all_engineers() -> List[str]:
    seen = set()
    ordered: List[str] = []
    for members in TEAM_MEMBERS_DICT.values():
        for m in members:
            if m not in seen:
                seen.add(m)
                ordered.append(m)
    return ordered


def build_jql(members: List[str], start_dt: datetime, end_dt: datetime, projects: List[str] | None = None) -> str:
    start_str = start_dt.strftime('%Y-%m-%d %H:%M')
    end_str = end_dt.strftime('%Y-%m-%d %H:%M')
    created_jql = f'created >= "{start_str}" AND created <= "{end_str}"'
    updated_jql = f'updated >= "{start_str}" AND updated <= "{end_str}"'
    member_filter = ' OR '.join([f'reporter = "{m}" OR assignee = "{m}"' for m in members])
    proj_filter = ''
    if projects:
        proj_filter = ' AND (' + ' OR '.join([f'project = "{p}"' for p in projects]) + ')'
    base = f'({created_jql} OR {updated_jql}) AND ({member_filter}){proj_filter}'
    return base


def jira_search(config: Dict[str, str], jql: str) -> List[Dict[str, Any]]:
    url = config['server'].rstrip('/') + '/rest/api/2/search'
    headers_common = {'Content-Type': 'application/json'}
    token = config['token']

    start_at = 0
    max_results = 100
    issues: List[Dict[str, Any]] = []

    while True:
        payload = {
            'jql': jql,
            'fields': [
                'key', 'summary', 'status', 'assignee', 'reporter', 'created', 'updated', 'resolutiondate',
                'project', 'issuetype', 'priority'
            ],
            'maxResults': max_results,
            'startAt': start_at,
        }

        # Try Bearer token first (PAT)
        resp = requests.post(
            url,
            headers={**headers_common, 'Authorization': f'Bearer {token}'},
            json=payload,
            timeout=30
        )

        # If unauthorized/forbidden, try Basic using username/token
        if resp.status_code in (401, 403):
            resp = requests.post(
                url,
                headers=headers_common,
                auth=(config['username'], token),
                json=payload,
                timeout=30
            )

        if resp.status_code == 401:
            raise SystemExit("Authentication failed (401). Check username/token in settings.json")
        if resp.status_code == 403:
            raise SystemExit("Access forbidden (403). Your user may lack API permissions.")
        if resp.status_code != 200:
            raise SystemExit(f"Jira search failed: {resp.status_code} - {resp.text[:200]}")

        data = resp.json()
        batch = data.get('issues', [])
        issues.extend(batch)
        if len(batch) < max_results:
            break
        start_at += max_results
    return issues


def flatten_issue(issue: Dict[str, Any]) -> Dict[str, Any]:
    fields = issue.get('fields', {})
    assignee = fields.get('assignee') or {}
    reporter = fields.get('reporter') or {}
    status = fields.get('status') or {}
    project = fields.get('project') or {}
    issuetype = fields.get('issuetype') or {}
    priority = fields.get('priority') or {}
    return {
        'key': issue.get('key'),
        'summary': fields.get('summary'),
        'status': (status.get('name') if isinstance(status, dict) else status),
        'assignee': assignee.get('name') or assignee.get('displayName'),
        'reporter': reporter.get('name') or reporter.get('displayName'),
        'created': fields.get('created'),
        'updated': fields.get('updated'),
        'resolutiondate': fields.get('resolutiondate'),
        'project': project.get('key') or project.get('name'),
        'issuetype': issuetype.get('name'),
        'priority': priority.get('name'),
    }


def save_csv(rows: List[Dict[str, Any]], output_dir: str, date_str: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f'jira_{date_str}.csv')
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    return csv_path


def main() -> int:
    settings = load_settings()
    config = resolve_config(settings)

    start_dt, end_dt, date_str = get_previous_day_range()
    logging.info("Date range (previous day): %s to %s", start_dt, end_dt)

    members = all_engineers()
    logging.info("Fetching for %d engineers", len(members))

    jql = build_jql(members, start_dt, end_dt)
    logging.info("JQL: %s", jql)

    issues = jira_search(config, jql)
    logging.info("Fetched %d issues", len(issues))

    rows = [flatten_issue(it) for it in issues]

    root = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(root, 'daily_update')
    csv_path = save_csv(rows, out_dir, date_str)
    logging.info("Saved CSV: %s", csv_path)

    # Invoke Gerrit consolidation with credentials from settings.json if available
    gerrit_script = os.path.join(root, 'fetch_gerrit_comments.py')
    if os.path.exists(gerrit_script):
        daily_json_dir = out_dir
        os.makedirs(daily_json_dir, exist_ok=True)
        out_json = os.path.join(daily_json_dir, f'gerrit_comments_{date_str}.json')
        cmd = [sys.executable or 'python3', gerrit_script, '--csv', csv_path, '--output', out_json]
        g_user = settings.get('gerrit_username')
        g_pass = settings.get('gerrit_password')
        if g_user and g_pass:
            cmd += ['--user', g_user, '--password', g_pass]
        logging.info('Running: %s', ' '.join(cmd))
        res = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
        if res.stdout:
            logging.info(res.stdout.strip())
        if res.stderr:
            logging.warning(res.stderr.strip())
        if os.path.exists(out_json):
            logging.info('Saved Gerrit JSON: %s', out_json)
        else:
            # Fallback: copy the latest consolidated JSON from Weekly_Data to daily_update
            weekly_dir = os.path.join(root, 'Weekly_Data')
            copied = False
            if os.path.isdir(weekly_dir):
                candidates = [f for f in os.listdir(weekly_dir) if f.lower().endswith('.json')]
                candidates.sort(reverse=True)
                for fname in candidates:
                    src = os.path.join(weekly_dir, fname)
                    try:
                        shutil.copy2(src, out_json)
                        logging.info('Copied consolidated JSON from %s to %s', src, out_json)
                        copied = True
                        break
                    except Exception as e:
                        logging.warning('Failed copying %s: %s', src, e)
            if not copied:
                logging.warning('Expected Gerrit JSON not found at %s and no fallback available', out_json)
    else:
        logging.warning('fetch_gerrit_comments.py not found; skipping Gerrit step')
    return 0


if __name__ == '__main__':
    raise SystemExit(main()) 