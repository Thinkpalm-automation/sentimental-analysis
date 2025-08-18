import csv
import re
import requests
import json
import os
import argparse
import time
import logging
import glob
from datetime import date

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# Gerrit credentials will be set from command-line arguments
AUTH = None

def ensure_weekly_data_dir():
    """Ensure the Weekly_Data directory exists in the project root."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    weekly_data_dir = os.path.join(project_root, 'Weekly_Data')
    os.makedirs(weekly_data_dir, exist_ok=True)
    return weekly_data_dir

def load_settings():
    """Load settings from settings.json (project root or flask_app/)."""
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

def extract_issue_keys_and_links(csv_file):
    """
    Extract issue keys and Gerrit links from the given JIRA CSV file.
    Returns a tuple: (list of issue keys, list of Gerrit links)
    """
    issue_keys = []
    gerrit_links = []

    try:
        with open(csv_file, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                issue_key = row.get('Issue key')
                if issue_key:
                    issue_keys.append(issue_key.strip())

                for key, value in row.items():
                    if 'Comment' in key and value:
                        links = re.findall(r'https://gerrit\.eng\.cohesity\.com/c/[^ \n;,\)\]]+', value)
                        gerrit_links.extend(links)
    except Exception as e:
        logging.error(f"Error reading CSV file {csv_file}: {e}")
    return issue_keys, gerrit_links

def convert_to_commit_url(change_number):
    """Return the Gerrit API URL for the commit message of a given change number."""
    return f'https://gerrit.eng.cohesity.com/a/changes/{change_number}/revisions/current/commit'

def convert_to_comments_url(change_number):
    """Return the Gerrit API URL for the comments of a given change number."""
    return f'https://gerrit.eng.cohesity.com/a/changes/{change_number}/comments'

def fetch_commit_message(commit_url, max_retries=3):
    """
    Fetch the commit message from Gerrit for the given commit URL.
    Retries up to max_retries times on failure.
    Returns the commit message string, or empty string on failure.
    """
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(commit_url, auth=AUTH)
            if response.status_code == 200:
                content = response.text.split('\n', 1)[1]
                commit_data = json.loads(content)
                return commit_data.get('message', '')
            else:
                logging.error(f"Attempt {attempt}: Failed to fetch commit message from {commit_url}: {response.status_code} {response.reason}\nResponse: {response.text}")
        except Exception as e:
            logging.error(f"Attempt {attempt}: Error fetching commit message from {commit_url}: {e}")
        if attempt < max_retries:
            time.sleep(1)
    return ''

def download_comments(change_number, project, issue_key, weekly_data_dir, max_retries=3):
    """
    Download comments for a Gerrit change and save to a JSON file in Weekly_Data folder.
    Retries up to max_retries times on failure.
    Returns the comments as a Python object (dict) if successful, else None.
    """
    comments_url = convert_to_comments_url(change_number)
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(comments_url, auth=AUTH)
            if response.status_code == 200:
                try:
                    # Remove Gerrit JSON prefix if present
                    text = response.text.lstrip()
                    if text.startswith(")]}'"):
                        text = text.split('\n', 1)[1]
                    if text and (text.startswith('{') or text.startswith('[')):
                        comments = json.loads(text)
                        filename = os.path.join(weekly_data_dir, f'{issue_key}_{project}_{change_number}.json')
                        with open(filename, 'w', encoding='utf-8') as f:
                            json.dump(comments, f, indent=2)
                        logging.info(f'Comments saved to {filename}')
                        return comments
                    else:
                        logging.warning(f'No valid JSON content for {issue_key}_{project}_{change_number}.json. Response: {response.text[:200]}')
                        return None
                except Exception as fe:
                    logging.error(f'Error processing/writing comments for {issue_key}_{project}_{change_number}.json: {fe}. Response: {response.text[:200]}')
                    return None
            else:
                logging.error(f"Attempt {attempt}: Failed to fetch comments from {comments_url}: {response.status_code} {response.reason}\nResponse: {response.text}")
        except Exception as e:
            logging.error(f"Attempt {attempt}: Error fetching comments from {comments_url}: {e}")
        if attempt < max_retries:
            time.sleep(1)
    return None

def main():
    """
    Main entry point: parses arguments, extracts issue keys and links, fetches commit messages and downloads comments as needed.
    After all downloads, consolidates all data into a single JSON file in Weekly_Data folder.
    """
    parser = argparse.ArgumentParser(description='Fetch and consolidate Gerrit comments for JIRA issues.')
    parser.add_argument('--csv', required=True, help='Path to the JIRA CSV file')
    parser.add_argument('--output', required=True, help='Path to the output JSON file')
    parser.add_argument('--user', help='Gerrit username (optional for auto mode)')
    parser.add_argument('--password', help='Gerrit password (optional for auto mode)')
    args = parser.parse_args()

    # Ensure Weekly_Data directory exists
    weekly_data_dir = ensure_weekly_data_dir()

    # Set up authentication if provided
    if args.user and args.password:
        global AUTH
        AUTH = (args.user, args.password)
    else:
        # Try to load Gerrit credentials from settings.json
        settings = load_settings()
        g_user = settings.get('gerrit_username')
        g_pass = settings.get('gerrit_password')
        if g_user and g_pass:
            AUTH = (g_user, g_pass)
            logging.info("Loaded Gerrit credentials from settings.json")
        else:
            logging.warning("No Gerrit credentials provided. Skipping Gerrit comment processing.")
            AUTH = None

    issue_keys, gerrit_links = extract_issue_keys_and_links(args.csv)

    consolidated = []
    
    # Only process Gerrit links if we have authentication
    if AUTH and gerrit_links:
        for link in gerrit_links:
            match = re.search(r'https://gerrit\.eng\.cohesity\.com/c/([^/]+)/\+/(\d+)', link)
            if match:
                project, change_number = match.groups()
                commit_url = convert_to_commit_url(change_number)
                commit_message = fetch_commit_message(commit_url)

                addressed_keys = re.findall(r'[A-Z]+-\d+', commit_message)
                for key in issue_keys:
                    if key in addressed_keys:
                        comments = download_comments(change_number, project, key, weekly_data_dir)
                        if comments is not None:
                            consolidated.append({
                                'issue_key': key,
                                'project': project,
                                'change_number': change_number,
                                'comments': comments
                            })
    else:
        # Create empty consolidated structure for auto mode without Gerrit
        logging.info("Creating empty Gerrit structure for auto mode")
        for key in issue_keys:
            consolidated.append({
                'issue_key': key,
                'project': '',
                'change_number': '',
                'comments': {}
            })

    # Write consolidated data to the specified output file in Weekly_Data folder
    week_num = date.today().isocalendar()[1]
    year_num = date.today().year
    auto_name = f'Week{week_num}_{year_num}.json'
    output_path = os.path.join(weekly_data_dir, auto_name)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(consolidated, f, indent=2)
    logging.info(f'Consolidated data written to {output_path}')

    # Delete all intermediate JSON files in Weekly_Data directory except the consolidated file
    consolidated_path = os.path.abspath(output_path)
    for fname in glob.glob(os.path.join(weekly_data_dir, '*.json')):
        if os.path.abspath(fname) == consolidated_path:
            continue
        try:
            os.remove(fname)
            logging.info(f"Deleted intermediate file: {fname}")
        except Exception as e:
            logging.warning(f"Could not delete {fname}: {e}")

if __name__ == '__main__':
    main()
