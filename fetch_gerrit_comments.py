import csv
import re
import requests
import json
import os
import argparse
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# Gerrit credentials will be set from command-line arguments
AUTH = None

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

def download_comments(change_number, project, issue_key, max_retries=3):
    """
    Download comments for a Gerrit change and save to a JSON file named <issue_key>_<project>_<change_number>.json.
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
                        filename = f'{issue_key}_{project}_{change_number}.json'
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
    After all downloads, consolidates all data into a single JSON file.
    """
    parser = argparse.ArgumentParser(description='Fetch and consolidate Gerrit comments for JIRA issues.')
    parser.add_argument('--csv', required=True, help='Path to the JIRA CSV file')
    parser.add_argument('--user', required=True, help='Gerrit username')
    parser.add_argument('--password', required=True, help='Gerrit password')
    args = parser.parse_args()

    global AUTH
    AUTH = (args.user, args.password)

    issue_keys, gerrit_links = extract_issue_keys_and_links(args.csv)

    consolidated = []
    for link in gerrit_links:
        match = re.search(r'https://gerrit\.eng\.cohesity\.com/c/([^/]+)/\+/(\d+)', link)
        if match:
            project, change_number = match.groups()
            commit_url = convert_to_commit_url(change_number)
            commit_message = fetch_commit_message(commit_url)

            addressed_keys = re.findall(r'[A-Z]+-\d+', commit_message)
            for key in issue_keys:
                if key in addressed_keys:
                    comments = download_comments(change_number, project, key)
                    if comments is not None:
                        consolidated.append({
                            'issue_key': key,
                            'project': project,
                            'change_number': change_number,
                            'comments': comments
                        })

    # Write consolidated data to a single JSON file
    output_file = 'consolidated_gerrit_comments.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(consolidated, f, indent=2)
    logging.info(f'Consolidated data written to {output_file}')

if __name__ == '__main__':
    main()
