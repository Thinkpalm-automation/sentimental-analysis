from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_from_directory
import os
import pandas as pd
from werkzeug.utils import secure_filename
import numpy as np
import json
import requests
from datetime import datetime, timedelta
import subprocess
import tempfile
import logging

# --- Sentiment Analysis Keyword Lists and Functions (from app.py) ---
import json
import os

# Settings file path
SETTINGS_FILE = 'settings.json'

def load_settings():
    """Load settings from JSON file"""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading settings: {e}")
    return {}

def save_settings(settings):
    """Save settings to JSON file"""
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving settings: {e}")
        return False

BUG_NEGATIVE_KEYWORDS = [
    "not reproducible", "cannot reproduce", "unable to reproduce", "not a bug", "works as designed", "as per design", "expected behavior", "duplicate issue", "duplicate of", "already fixed", "fix available", "configuration issue", "environment issue", "invalid bug", "invalid issue", "not valid", "not required", "user error", "tester error",
    "no steps provided", "need more info", "insufficient information", "steps missing", "logs not attached", "data missing", "test case issue", "not enough context",
    "out of scope", "triage required", "needs triage", "no action required", "won't fix", "working as designed", "working as expected", "invalid scenario", "not applicable", "edge case", "use case not supported", "intended behavior", "reported incorrectly"
]
BUG_POSITIVE_KEYWORDS = [
    "good catch", "nice catch", "valid finding", "well documented", "clearly explained", "triaged correctly", "good triage", "logs helped", "good investigation", "thanks for reporting", "appreciate the report", "thanks for the catch"
]
OTHER_NEGATIVE_KEYWORDS = [
    "moved to next sprint", "spillover", "slipped to next sprint", "carried forward", "not completed in sprint", "sprint goal missed", "pushed to backlog",
    "code review pending", "pending review", "pending approval", "review not done", "review comments not addressed", "gerrit review pending", "awaiting review",
    "commit abandoned", "commit reverted", "change reverted", "revert pushed", "patch not merged", "merge conflict", "build failed", "build break", "compilation error", "integration failed", "jenkins failure",
    "blocked", "waiting on dependency", "dependency delay", "on hold", "no update", "stuck", "waiting for input", "pending clarification",
    "use case not supported", "task reopened", "story reopened", "duplicate work", "abandoned", "discarded", "redo required"
]
OTHER_POSITIVE_KEYWORDS = [
    "good work", "great job", "well done", "nice work", "appreciated", "kudos", "thanks for the effort", "excellent", "great effort", "well executed", "nicely handled", "awesome job", "no review comments"
]
GERRIT_NEGATIVE_KEYWORDS = [
    "not needed", "unused variable", "remove this", "should be deleted", "unnecessary", "redundant", "typo", "incorrect", "does not work", "breaks", "missing", "needs improvement", "not clear", "confusing", "bad practice", "hardcoded", "magic number", "not efficient", "performance issue", "security issue", "potential bug", "should be refactored", "not following convention", "incomplete", "wrong", "fails", "deprecated", "not tested", "test missing", "should be documented", "no comments", "unclear logic", "duplicate code", "not readable", "too complex", "overcomplicated", "not optimal", "incorrect indentation", "formatting issue", "conflicts with", "not reviewed", "needs changes", "needs update", "needs rebase", "not aligned", "not matching", "not consistent", "not handled", "not covered", "not robust", "not thread safe", "race condition", "memory leak", "null pointer", "exception not handled", "error prone", "not scalable", "not maintainable"
]
import re
import pandas as pd

def validate_jira_closure_comment(comment: str) -> dict:
    comment = str(comment).strip().lower()
    reasons = []
    gerrit_link = re.search(r'https?://gerrit\.', comment)
    commit_hash = re.search(r'\b[0-9a-f]{7,40}\b', comment)
    descriptive = len(comment.split()) > 4 and not comment.isspace()
    weak_phrases = [
        "done", "fixed", "completed", "closing story", "story closed", "as discussed",
        "work completed", "task done", "issue resolved", "resolved", "closed", "finished", "implemented", "ok", "fine", "good", "completed task", "work finished", "all done", "all set", "work is done", "task completed", "work closed", "solved", "addressed", "taken care", "work fixed", "fixed issue", "issue closed", "pr merged", "merged"
    ]
    is_weak = any(comment == phrase or phrase in comment for phrase in weak_phrases)
    if gerrit_link:
        return {"valid": True, "reason": "See Gerrit review table"}
    if commit_hash or descriptive:
        return {"valid": True, "reason": ""}
    if is_weak:
        reasons.append("Generic closure phrase used.")
    if len(comment.split()) <= 4:
        reasons.append("Too short or vague.")
    return {
        "valid": False,
        "reason": "; ".join(reasons) or "Lacks meaningful content."
    }

def get_sentiment(comment, issue_type=None, status=None):
    if pd.isna(comment) or str(comment).strip() == "":
        issue_type = (issue_type or "").strip().lower()
        status = (status or "").strip().lower()
        if issue_type == "bug" and status == "open":
            return "Neutral", ""
        return "Negative", "Missing Comment"
    comment_lower = str(comment).lower()
    issue_type = (issue_type or "").strip().lower()
    if issue_type == "bug":
        neg_keywords = BUG_NEGATIVE_KEYWORDS
        pos_keywords = BUG_POSITIVE_KEYWORDS
    else:
        neg_keywords = OTHER_NEGATIVE_KEYWORDS
        pos_keywords = OTHER_POSITIVE_KEYWORDS
    for kw in neg_keywords:
        if kw in comment_lower:
            return "Negative", f'Comment-based: "{kw}" in "{comment}"'
    for kw in pos_keywords:
        if kw in comment_lower:
            return "Positive", ""
    if issue_type != "bug" and issue_type != "service request":
        validation = validate_jira_closure_comment(comment)
        if validation["valid"]:
            return "Neutral", validation["reason"]
        else:
            return "Negative", validation["reason"]
    # Fallback to model (for bug and service request only)
    # NOTE: In Flask, you will need to pass a pipeline or use a global one
    return "Neutral", ""  # Placeholder for Flask version

def story_points_flag(row):
    try:
        sp = float(row.get("Custom field (Story Points)", 0))
    except (TypeError, ValueError):
        sp = 0
    return "🚩 High Story Points" if str(row.get("Issue Type", "")).strip().lower() != "bug" and sp > 8 else ""

def get_gerrit_sentiment(comment):
    if pd.isna(comment) or str(comment).strip() == "":
        return "Neutral", "", ""
    comment_lower = str(comment).lower()
    for kw in GERRIT_NEGATIVE_KEYWORDS:
        if kw in comment_lower:
            return "Negative", str(comment), kw
    return "Neutral", "", ""

def process_data(df):
    required_columns = [
        "Summary", "Issue key", "Issue Type", "Status", "Project key", "Priority", "Resolution",
        "Assignee", "Reporter", "Created", "Updated", "Resolved", "Sprint",
        "Custom field (Story Points)", "Custom field ([CHART] Date of First Response)", "Comment"
    ]
    available_columns = [col for col in required_columns if col in df.columns]
    df_filtered = df[available_columns].copy()
    def custom_story_points_flag(row):
        issue_type = str(row.get("Issue Type", "")).strip().lower()
        if issue_type != "bug" and issue_type != "service request":
            return story_points_flag(row)
        return ""
    def custom_comment_flag(row):
        issue_type = str(row.get("Issue Type", "")).strip().lower()
        comment = row.get("Comment", "")
        status = str(row.get("Status", "")).strip().lower()
        if issue_type == "bug":
            if (pd.isna(comment) or str(comment).strip() == ""):
                if status == "open":
                    return ""  # No flag for open bugs with empty comment
                else:
                    return "⚠️ Follow-up required"
            return ""
        elif issue_type != "service request":
            if pd.isna(comment) or str(comment).strip() == "":
                return "❌ Missing Comment"
        return ""
    df_filtered["Story Points Flag"] = df_filtered.apply(custom_story_points_flag, axis=1)
    df_filtered["Comment Flag"] = df_filtered.apply(custom_comment_flag, axis=1)
    def sentiment_and_reason(row):
        issue_type = str(row.get("Issue Type", "")).strip().lower()
        comment = row.get("Comment", "")
        sp_flag = row.get("Story Points Flag", "")
        comment_flag = row.get("Comment Flag", "")
        status = str(row.get("Status", "")).strip().lower()
        if issue_type != "bug" and issue_type != "service request":
            if "High Story Points" in sp_flag and "Missing Comment" in comment_flag:
                return pd.Series({"Customer Sentiment": "Negative", "Negative Reason": "High Story Points and Missing Comment"})
            elif "High Story Points" in sp_flag:
                return pd.Series({"Customer Sentiment": "Negative", "Negative Reason": "High Story Points"})
            elif "Missing Comment" in comment_flag:
                return pd.Series({"Customer Sentiment": "Negative", "Negative Reason": "Missing Comment"})
            else:
                sentiment, neg_reason = get_sentiment(comment, issue_type, status)
                if sentiment == "Negative":
                    if neg_reason:
                        return pd.Series({"Customer Sentiment": "Negative", "Negative Reason": neg_reason})
                    else:
                        return pd.Series({"Customer Sentiment": "TBD", "Negative Reason": neg_reason or f'Comment-based: "{comment}"'})
                else:
                    return pd.Series({"Customer Sentiment": sentiment, "Negative Reason": neg_reason})
        elif issue_type == "bug":
            # For bugs, handle follow-up required
            if "Follow-up required" in comment_flag:
                return pd.Series({"Customer Sentiment": "Negative", "Negative Reason": "Follow-up required"})
            comment_sentiment, neg_reason = get_sentiment(comment, issue_type, status)
            resolution = row.get("Resolution", "")
            negative_reason_parts = []
            if comment_sentiment == "Negative" and neg_reason:
                negative_reason_parts.append(neg_reason)
            resolution_triggers_negative = resolution and pd.notna(resolution) and "fixed" not in str(resolution).lower()
            if resolution_triggers_negative:
                if comment_sentiment != "Negative":
                    return pd.Series({"Customer Sentiment": "Negative", "Negative Reason": f'Resolution-based: {resolution}'})
                else:
                    negative_reason_parts.append(f'Resolution-based: {resolution}')
            if negative_reason_parts:
                return pd.Series({"Customer Sentiment": "Negative", "Negative Reason": ", ".join(negative_reason_parts)})
            else:
                return pd.Series({"Customer Sentiment": comment_sentiment, "Negative Reason": neg_reason})
        else:
            sentiment, neg_reason = get_sentiment(comment, issue_type, status)
            return pd.Series({"Customer Sentiment": sentiment, "Negative Reason": neg_reason})
    df_filtered[["Customer Sentiment", "Negative Reason"]] = df_filtered.apply(sentiment_and_reason, axis=1)
    if "Resolution" in df_filtered.columns and "Issue Type" in df_filtered.columns:
        df_filtered["Customer Sentiment"] = df_filtered.apply(
            lambda row: "Negative"
            if str(row["Issue Type"]).strip().lower() == "bug"
            and pd.notna(row["Resolution"])
            and "fixed" not in str(row["Resolution"]).lower()
            else row["Customer Sentiment"],
            axis=1
        )
    df_filtered["Sentiment Flag"] = df_filtered["Customer Sentiment"].apply(
        lambda x: "⚠️ Negative Sentiment" if x == "Negative" else ""
    )
    df_filtered["Validation Flags"] = df_filtered[
        ["Sentiment Flag", "Story Points Flag", "Comment Flag"]
    ].apply(lambda flags: ", ".join(flag for flag in flags if flag), axis=1)
    df_filtered.drop(columns=["Sentiment Flag", "Story Points Flag", "Comment Flag"], inplace=True)
    if 'Custom field (Story Points)' in df_filtered.columns:
        df_filtered['Custom field (Story Points)'] = df_filtered['Custom field (Story Points)'].apply(lambda x: int(x) if pd.notna(x) and x != '' else '')
    return df_filtered


app = Flask(__name__)
app.secret_key = os.urandom(24)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_summary_stats(df):
    total_issues = len(df)
    bug_issues = df[df['Issue Type'].str.lower().str.contains('bug', na=False)]
    nonbug_issues = df[~df['Issue Type'].str.lower().str.contains('bug', na=False)]
    unique_engineers = df['Assignee'].nunique() if 'Assignee' in df.columns else 0
    return {
        'total_issues': total_issues,
        'bug_issues': len(bug_issues),
        'nonbug_issues': len(nonbug_issues),
        'unique_engineers': unique_engineers
    }

def to_native_int(obj):
    if isinstance(obj, dict):
        return {k: to_native_int(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_native_int(v) for v in obj]
    elif isinstance(obj, (np.integer, pd.Int64Dtype, pd.UInt64Dtype)):
        return int(obj)
    elif hasattr(obj, 'item') and callable(obj.item):
        try:
            return int(obj.item())
        except Exception:
            return obj
    else:
        return int(obj) if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)) else obj

# Team members by team
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
# Add 'All' team with all unique members from all teams
TEAM_MEMBERS_DICT['All'] = sorted({member for members in TEAM_MEMBERS_DICT.values() for member in members})

@app.before_request
def set_default_team():
    if 'selected_team' not in session or not session['selected_team']:
        session['selected_team'] = 'All'

@app.context_processor
def inject_team_context():
    return {
        'team_list': list(TEAM_MEMBERS_DICT.keys()),
        'selected_team': session.get('selected_team', '')
    }

@app.route('/', methods=['GET', 'POST'])
def home():
    import json
    # Load saved settings
    saved_settings = load_settings()
    
    filters = {
        "sprints": [],
        "assignees": [],
        "issue_types": [],
        "reporters": [],
        "date_start": "",
        "date_end": ""
    }
    kpis = {
        "total_comments": 0,
        "positive_pct": 0,
        "negative_pct": 0,
        "users_flagged": 0
    }
    flags = []
    table_data = []
    chart_data = {
        "pie": [0, 0, 0],
        "bar_labels": [],
        "bar_values": []
    }
    last_updated = ""
    processed_df = None
    gerrit_data = None
    team_sentiment = {}
    overall_sentiment = {"Positive": 0, "Neutral": 0, "Negative": 0}
    selected_team = session.get('selected_team', '')
    total_bugs = 0
    total_nonbugs = 0
    total_gerrit = 0
    if request.method == 'POST':
        mode = request.form.get('mode', 'manual')
        
        if mode == 'manual':
            # Manual mode - handle file uploads
            # JIRA file (required)
            if 'jira_file' not in request.files or request.files['jira_file'].filename == '':
                flash('Please upload a JIRA CSV or Excel file.', 'danger')
                return redirect(request.url)
            file = request.files['jira_file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                session['jira_file'] = filename
                try:
                    if filename.lower().endswith('.csv'):
                        df = pd.read_csv(filepath)
                    else:
                        df = pd.read_excel(filepath)
                    processed_df = process_data(df)
                except Exception as e:
                    flash(f'Error processing JIRA file: {e}', 'danger')
                    return redirect(request.url)
            else:
                flash('Invalid JIRA file type. Please upload a CSV or Excel file.', 'danger')
                return redirect(request.url)
            # Gerrit file (optional)
            gerrit_file = request.files.get('gerrit_file')
            if gerrit_file and gerrit_file.filename != '' and gerrit_file.filename.lower().endswith('.json'):
                gerrit_filename = secure_filename(gerrit_file.filename)
                gerrit_filepath = os.path.join(app.config['UPLOAD_FOLDER'], gerrit_filename)
                gerrit_file.save(gerrit_filepath)
                session['gerrit_file'] = gerrit_filename
            elif gerrit_file and gerrit_file.filename != '':
                flash('Invalid Gerrit file type. Please upload a JSON file.', 'danger')
                return redirect(request.url)
            selected_team = request.form.get('selected_team', '') or 'All'
            session['selected_team'] = selected_team
        elif mode == 'auto':
            # Auto mode - data should already be fetched via AJAX
            # This will be handled by the auto_fetch_data route
            pass
    elif 'jira_file' in session:
        filename = session['jira_file']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            if filename.lower().endswith('.csv'):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)
            processed_df = process_data(df)
        except Exception:
            processed_df = None
    # --- Load Gerrit JSON if present ---
    if 'gerrit_file' in session:
        gerrit_filename = session['gerrit_file']
        gerrit_filepath = os.path.join(app.config['UPLOAD_FOLDER'], gerrit_filename)
        try:
            with open(gerrit_filepath, 'r', encoding='utf-8') as f:
                gerrit_data = json.load(f)
        except Exception:
            gerrit_data = None
    # --- Sentiment Aggregation ---
    team_sentiment_dict = {}
    if processed_df is not None:
        filters = {
            "sprints": sorted(processed_df['Sprint'].dropna().unique().tolist()) if 'Sprint' in processed_df else [],
            "assignees": sorted(processed_df['Assignee'].dropna().unique().tolist()) if 'Assignee' in processed_df else [],
            "issue_types": sorted(processed_df['Issue Type'].dropna().unique().tolist()) if 'Issue Type' in processed_df else [],
            "reporters": sorted(processed_df['Reporter'].dropna().unique().tolist()) if 'Reporter' in processed_df else [],
            "date_start": str(processed_df['Created'].min())[:10] if 'Created' in processed_df and not processed_df['Created'].isnull().all() else "",
            "date_end": str(processed_df['Created'].max())[:10] if 'Created' in processed_df and not processed_df['Created'].isnull().all() else ""
        }
        # Calculate total_comments for overall CSV (not filtered)
        total_comments = len(processed_df)
        # Filter for selected team for all other KPIs and table
        if selected_team in TEAM_MEMBERS_DICT:
            team_members = TEAM_MEMBERS_DICT[selected_team]
            bug_mask = (processed_df['Issue Type'].str.lower().str.strip() == 'bug')
            is_team_reporter = processed_df['Reporter'].apply(lambda x: is_team_member(x, team_members))
            nonbug_mask = (processed_df['Issue Type'].str.lower().str.strip() != 'bug')
            is_team_assignee = processed_df['Assignee'].apply(lambda x: is_team_member(x, team_members))
            filtered_df = pd.concat([
                processed_df[bug_mask & is_team_reporter],
                processed_df[nonbug_mask & is_team_assignee]
            ])
        else:
            filtered_df = processed_df.iloc[0:0]
        positive = (filtered_df['Customer Sentiment'] == 'Positive').sum()
        neutral = (filtered_df['Customer Sentiment'] == 'Neutral').sum()
        negative = (filtered_df['Customer Sentiment'] == 'Negative').sum()
        users_flagged = filtered_df[filtered_df['Customer Sentiment'] == 'Negative']['Assignee'].nunique() if 'Assignee' in filtered_df else 0
        kpis = {
            "total_comments": total_comments,
            "positive_pct": round(positive / len(filtered_df) * 100) if len(filtered_df) else 0,
            "neutral_pct": round(neutral / len(filtered_df) * 100) if len(filtered_df) else 0,
            "negative_pct": round(negative / len(filtered_df) * 100) if len(filtered_df) else 0,
            "users_flagged": users_flagged
        }
        flags = [
            {"issue_key": row["Issue key"], "flag_reason": row["Validation Flags"]}
            for _, row in filtered_df.iterrows()
            if row.get("Validation Flags")
        ]
        table_data = filtered_df[["Issue key", "Status", "Assignee", "Customer Sentiment", "Validation Flags", "Comment"]].to_dict(orient="records")
        # --- Team Sentiment Aggregation ---
        # Only include core team members for the selected team
        if selected_team in TEAM_MEMBERS_DICT:
            team_members = TEAM_MEMBERS_DICT[selected_team]
            # Get assignees that match team members
            available_assignees = processed_df['Assignee'].dropna().unique().tolist()
            assignees = [a for a in available_assignees if is_team_member(a, team_members)]
        else:
            assignees = []
        issue_key_to_assignee = dict(zip(processed_df['Issue key'], processed_df['Assignee']))
        for assignee in assignees:
            # Use normalized name as key for consistency
            normalized_name = normalize_name(assignee)
            team_sentiment[normalized_name] = {"Positive": 0, "Neutral": 0, "Negative": 0}
            # Bugs (by Reporter, to match engineer drilldown)
            bug_rows = processed_df[(processed_df['Reporter'].apply(lambda x: is_team_member(x, [normalized_name]))) & (processed_df['Issue Type'].str.lower() == 'bug')]
            for sentiment in ["Positive", "Neutral", "Negative"]:
                team_sentiment[normalized_name][sentiment] += (bug_rows['Customer Sentiment'] == sentiment).sum()
            # Non-bugs (by Assignee)
            nonbug_rows = processed_df[(processed_df['Assignee'].apply(lambda x: is_team_member(x, [normalized_name]))) & (processed_df['Issue Type'].str.lower() != 'bug')]
            for sentiment in ["Positive", "Neutral", "Negative"]:
                team_sentiment[normalized_name][sentiment] += (nonbug_rows['Customer Sentiment'] == sentiment).sum()
        # 2. Gerrit comments (if available)
        if gerrit_data:
            for entry in gerrit_data:
                issue_key = entry.get('issue_key')
                assignee = issue_key_to_assignee.get(issue_key)
                if not assignee or not is_team_member(assignee, TEAM_MEMBERS_DICT.get(selected_team, [])):
                    continue
                # Find the matching normalized name for this assignee
                matching_normalized = None
                for normalized_name in team_sentiment.keys():
                    if is_team_member(assignee, [normalized_name]):
                        matching_normalized = normalized_name
                        break
                if not matching_normalized:
                    continue
                comments = entry.get('comments', {})
                if isinstance(comments, dict):
                    for comment_list in comments.values():
                        if isinstance(comment_list, list):
                            for comment in comment_list:
                                if isinstance(comment, dict):
                                    msg = comment.get('message')
                                    # Count for assignee regardless of author
                                    sentiment, _, _ = get_gerrit_sentiment(msg)
                                    if sentiment in team_sentiment[matching_normalized]:
                                        team_sentiment[matching_normalized][sentiment] += 1
        # --- Overall sentiment sum for chart ---
        for assignee, counts in team_sentiment.items():
            for sentiment in ["Positive", "Neutral", "Negative"]:
                overall_sentiment[sentiment] += counts[sentiment]
        # Sort bar chart data by negative count descending
        sorted_bar = sorted(team_sentiment.items(), key=lambda x: x[1]["Negative"], reverse=True)
        chart_data = {
            "pie": [
                int(overall_sentiment.get("Positive", 0)),
                int(overall_sentiment.get("Neutral", 0)),
                int(overall_sentiment.get("Negative", 0))
            ],
            "bar_labels": [x[0] for x in sorted_bar],
            "bar_values": [x[1]["Negative"] for x in sorted_bar]
        }
        last_updated = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        # --- Convert all values to native int for JSON serialization ---
        team_sentiment = to_native_int(team_sentiment)
        chart_data = to_native_int(chart_data)
        # Ensure selected_team is always set in team_sentiment_dict for the charts
        if selected_team in TEAM_MEMBERS_DICT:
            team_sentiment_dict[selected_team] = team_sentiment
        # Only build team_sentiment_dict for all teams (for drilldown), do not overwrite selected_team's data
        for team, members in TEAM_MEMBERS_DICT.items():
            if team == selected_team:
                # Do not overwrite selected_team's Gerrit-inclusive data
                continue
            assignees = [a for a in members if a in processed_df['Assignee'].dropna().unique().tolist()]
            team_sentiment_other = {}
            issue_key_to_assignee = dict(zip(processed_df['Issue key'], processed_df['Assignee']))
            for assignee in assignees:
                team_sentiment_other[assignee] = {"Positive": 0, "Neutral": 0, "Negative": 0}
                # Bugs
                bug_rows = processed_df[(processed_df['Assignee'] == assignee) & (processed_df['Issue Type'].str.lower() == 'bug')]
                for sentiment in ["Positive", "Neutral", "Negative"]:
                    team_sentiment_other[assignee][sentiment] += (bug_rows['Customer Sentiment'] == sentiment).sum()
                # Non-bugs
                nonbug_rows = processed_df[(processed_df['Assignee'] == assignee) & (processed_df['Issue Type'].str.lower() != 'bug')]
                for sentiment in ["Positive", "Neutral", "Negative"]:
                    team_sentiment_other[assignee][sentiment] += (nonbug_rows['Customer Sentiment'] == sentiment).sum()
            team_sentiment_dict[team] = to_native_int(team_sentiment_other)
        if selected_team in TEAM_MEMBERS_DICT:
            team_members = TEAM_MEMBERS_DICT[selected_team]
            # Total bugs: Reporter is a team member
            bug_mask = processed_df['Issue Type'].str.lower().str.strip() == 'bug'
            is_team_reporter = processed_df['Reporter'].apply(lambda x: is_team_member(x, team_members))
            total_bugs = processed_df[bug_mask & is_team_reporter].shape[0]
            # Total non-bugs: Assignee is a team member
            nonbug_mask = processed_df['Issue Type'].str.lower().str.strip() != 'bug'
            is_team_assignee = processed_df['Assignee'].apply(lambda x: is_team_member(x, team_members))
            total_nonbugs = processed_df[nonbug_mask & is_team_assignee].shape[0]
            # Gerrit matching (as before)
            valid_rows = processed_df[is_team_assignee]
            valid_issue_keys = set(valid_rows['Issue key'].dropna().unique())
            total_gerrit = 0
            if gerrit_data:
                gerrit_issue_keys = set()
                for entry in gerrit_data:
                    issue_key = entry.get('issue_key')
                    if issue_key in valid_issue_keys:
                        gerrit_issue_keys.add(issue_key)
                total_gerrit = len(gerrit_issue_keys)
        else:
            total_bugs = 0
            total_nonbugs = 0
            total_gerrit = 0
    return render_template(
        "home.html",
        filters=filters,
        kpis=kpis,
        flags=flags,
        table_data=table_data,
        chart_data=chart_data,
        last_updated=last_updated,
        team_sentiment=team_sentiment,
        team_sentiment_dict=team_sentiment_dict,
        selected_team=selected_team,
        team_list=list(TEAM_MEMBERS_DICT.keys()),
        total_bugs=total_bugs,
        total_nonbugs=total_nonbugs,
        total_gerrit=total_gerrit,
        settings=saved_settings
    )

@app.route('/set_team', methods=['POST'])
def set_team():
    team = request.form.get('selected_team', '')
    session['selected_team'] = team  # Always set, even if empty
    print('Setting selected_team:', team)
    print('Session after set:', dict(session))
    referrer = request.headers.get('Referer')
    return redirect(referrer or url_for('home'))

@app.route('/clear_data', methods=['POST'])
def clear_data():
    """
    Clear existing data from session and optionally remove uploaded files
    """
    try:
        # Clear session data
        if 'jira_file' in session:
            # Optionally remove the uploaded file
            try:
                jira_filepath = os.path.join(app.config['UPLOAD_FOLDER'], session['jira_file'])
                if os.path.exists(jira_filepath):
                    os.remove(jira_filepath)
            except Exception as e:
                print(f"Warning: Could not remove Jira file: {e}")
            del session['jira_file']
        
        if 'gerrit_file' in session:
            # Optionally remove the uploaded file
            try:
                gerrit_filepath = os.path.join(app.config['UPLOAD_FOLDER'], session['gerrit_file'])
                if os.path.exists(gerrit_filepath):
                    os.remove(gerrit_filepath)
            except Exception as e:
                print(f"Warning: Could not remove Gerrit file: {e}")
            del session['gerrit_file']
        
        # Clear any other session data related to data processing
        session_keys_to_clear = ['selected_team']
        for key in session_keys_to_clear:
            if key in session:
                del session[key]
        
        return jsonify({'success': True, 'message': 'Data cleared successfully'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Placeholder routes for multi-page dashboard
@app.route('/engineer')
def engineer():
    selected_team = session.get('selected_team', '')
    print('[ENGINEER] Current selected_team in session:', selected_team)
    # Prepare summary for each team member
    summary = []
    filters = {"members": []}
    if 'jira_file' in session:
        filename = session['jira_file']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            if filename.lower().endswith('.csv'):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)
            processed_df = process_data(df)
            # Use only TEAM_MEMBERS for selected team
            if selected_team in TEAM_MEMBERS_DICT:
                team_members = TEAM_MEMBERS_DICT[selected_team]
                # Filter to only include members that appear in the data
                available_assignees = processed_df['Assignee'].dropna().unique().tolist()
                team_members = [m for m in team_members if any(is_team_member(assignee, [m]) for assignee in available_assignees)]
            else:
                team_members = []
            filters["members"] = team_members
            selected_member = request.args.get('member', 'All')
            # Bugs (filter by Reporter for team membership)
            bug_df = processed_df[(processed_df['Issue Type'].str.lower() == 'bug')] if 'Issue Type' in processed_df else processed_df
            bug_df = bug_df[bug_df['Reporter'].apply(lambda x: is_team_member(x, team_members))] if 'Reporter' in bug_df else bug_df
            # Non-bugs (filter by Assignee)
            nonbug_df = processed_df[(processed_df['Issue Type'].str.lower() != 'bug')] if 'Issue Type' in processed_df else processed_df
            nonbug_df = nonbug_df[nonbug_df['Assignee'].apply(lambda x: is_team_member(x, team_members))] if 'Assignee' in nonbug_df else nonbug_df
            # Gerrit
            gerrit_counts = {m: {"Positive": 0, "Neutral": 0, "Negative": 0} for m in team_members}
            if 'gerrit_file' in session:
                gerrit_filename = session['gerrit_file']
                gerrit_filepath = os.path.join(app.config['UPLOAD_FOLDER'], gerrit_filename)
                import json
                with open(gerrit_filepath, 'r', encoding='utf-8') as f:
                    gerrit_data = json.load(f)
                issue_key_to_assignee = dict(zip(processed_df['Issue key'], processed_df['Assignee']))
                for entry in gerrit_data:
                    issue_key = entry.get('issue_key')
                    assignee = issue_key_to_assignee.get(issue_key)
                    if not assignee or not is_team_member(assignee, team_members):
                        continue
                    # Find the matching team member for this assignee
                    matching_member = None
                    for member in team_members:
                        if is_team_member(assignee, [member]):
                            matching_member = member
                            break
                    if not matching_member:
                        continue
                    comments = entry.get('comments', {})
                    if isinstance(comments, dict):
                        for comment_list in comments.values():
                            if isinstance(comment_list, list):
                                for comment in comment_list:
                                    if isinstance(comment, dict):
                                        msg = comment.get('message')
                                        sentiment, _, _ = get_gerrit_sentiment(msg)
                                        if sentiment in gerrit_counts[matching_member]:
                                            gerrit_counts[matching_member][sentiment] += 1
            for member in team_members:
                if selected_member != 'All' and member != selected_member:
                    continue
                # Use the normalized name for filtering
                bugs = bug_df[bug_df['Reporter'].apply(lambda x: is_team_member(x, [member]))] if 'Reporter' in bug_df else pd.DataFrame()
                bug_counts = bugs['Customer Sentiment'].value_counts().to_dict() if not bugs.empty else {}
                nonbugs = nonbug_df[nonbug_df['Assignee'].apply(lambda x: is_team_member(x, [member]))] if 'Assignee' in nonbug_df else pd.DataFrame()
                nonbug_counts = nonbugs['Customer Sentiment'].value_counts().to_dict() if not nonbugs.empty else {}
                gerrit = gerrit_counts.get(member, {})
                summary.append({
                    'member': member,
                    'bugs': {
                        'Positive': bug_counts.get('Positive', 0),
                        'Negative': bug_counts.get('Negative', 0)
                    },
                    'nonbugs': {
                        'Positive': nonbug_counts.get('Positive', 0),
                        'Negative': nonbug_counts.get('Negative', 0)
                    },
                    'gerrit': {
                        'Positive': gerrit.get('Positive', 0),
                        'Negative': gerrit.get('Negative', 0)
                    }
                })
        except Exception as e:
            summary = []
    return render_template('engineer.html', summary=summary, filters=filters, selected_member=request.args.get('member', 'All'))

@app.route('/bugs')
def bugs():
    selected_team = session.get('selected_team', '')
    print('[BUGS] Current selected_team in session:', selected_team)
    table_data = []
    filters = {"reporters": []}
    reporter_filter = request.args.get('reporter')
    sentiment_filter = request.args.get('sentiment')
    if 'jira_file' in session:
        filename = session['jira_file']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            if filename.lower().endswith('.csv'):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)
            processed_df = process_data(df)
            # Filter for bug issues where Reporter is a team member of selected team
            bug_df = processed_df[(processed_df['Issue Type'].str.lower() == 'bug')] if 'Issue Type' in processed_df else processed_df
            if selected_team in TEAM_MEMBERS_DICT:
                team_members = TEAM_MEMBERS_DICT[selected_team]
                bug_df = bug_df[bug_df['Reporter'].apply(lambda x: is_team_member(x, team_members))] if 'Reporter' in bug_df else bug_df
                filters["reporters"] = [r for r in team_members if any(is_team_member(reporter, [r]) for reporter in bug_df['Reporter'].dropna().unique().tolist())] if 'Reporter' in bug_df else []
            else:
                bug_df = bug_df.iloc[0:0]  # Empty
                filters["reporters"] = []
            if reporter_filter:
                bug_df = bug_df[bug_df['Reporter'] == reporter_filter]
            if sentiment_filter:
                bug_df = bug_df[bug_df['Customer Sentiment'] == sentiment_filter]
            table_data = bug_df[["Issue key", "Status", "Reporter", "Customer Sentiment", "Comment", "Negative Reason", "Validation Flags"]].rename(columns={
                "Issue key": "issue_key",
                "Status": "status",
                "Reporter": "reporter",
                "Customer Sentiment": "sentiment",
                "Comment": "comment",
                "Negative Reason": "negative_reason",
                "Validation Flags": "validation_flags"
            }).to_dict(orient="records")
        except Exception:
            table_data = []
    return render_template('bugs.html', table_data=table_data, filters=filters)

@app.route('/nonbugs', methods=['GET', 'POST'])
def nonbugs():
    selected_team = session.get('selected_team', '')
    print('[NONBUGS] Current selected_team in session:', selected_team)
    table_data = []
    filters = {"assignees": []}
    assignee_filter = request.args.get('assignee')
    sentiment_filter = request.args.get('sentiment')
    if request.method == 'POST':
        if 'gerrit_file' in request.files:
            file = request.files['gerrit_file']
            if file.filename != '' and file.filename.lower().endswith('.json'):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                session['gerrit_file'] = filename
            else:
                flash('Invalid file type. Please upload a JSON file.', 'danger')
    if 'jira_file' in session:
        filename = session['jira_file']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            if filename.lower().endswith('.csv'):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)
            processed_df = process_data(df)
            # Filter for non-bug issues and by selected team
            nonbug_df = processed_df[processed_df['Issue Type'].str.lower() != 'bug'] if 'Issue Type' in processed_df else processed_df
            nonbug_df = nonbug_df.fillna('')
            if selected_team in TEAM_MEMBERS_DICT:
                team_members = TEAM_MEMBERS_DICT[selected_team]
                nonbug_df = nonbug_df[nonbug_df['Assignee'].apply(lambda x: is_team_member(x, team_members))] if 'Assignee' in nonbug_df else nonbug_df
                filters["assignees"] = sorted([a for a in team_members if any(is_team_member(assignee, [a]) for assignee in nonbug_df['Assignee'].dropna().unique().tolist())]) if 'Assignee' in nonbug_df else []
            else:
                nonbug_df = nonbug_df.iloc[0:0]
                filters["assignees"] = []
            if assignee_filter:
                nonbug_df = nonbug_df[nonbug_df['Assignee'] == assignee_filter]
            if sentiment_filter:
                nonbug_df = nonbug_df[nonbug_df['Customer Sentiment'] == sentiment_filter]
            table_data = nonbug_df[["Issue key", "Assignee", "Customer Sentiment", "Comment", "Negative Reason", "Custom field (Story Points)", "Validation Flags"]].rename(columns={
                "Issue key": "issue_key",
                "Assignee": "assignee",
                "Customer Sentiment": "sentiment",
                "Comment": "comment",
                "Negative Reason": "negative_reason",
                "Custom field (Story Points)": "story_points",
                "Validation Flags": "validation_flags"
            }).to_dict(orient="records")
        except Exception:
            table_data = []
    return render_template('nonbugs.html', table_data=table_data, filters=filters)

@app.route('/gerrit')
def gerrit():
    selected_team = session.get('selected_team', '')
    print('[GERRIT] Current selected_team in session:', selected_team)
    gerrit_table = []
    total_bugs = 0
    total_nonbugs = 0
    total_gerrit = 0
    import json
    # Check both files are present
    missing_files = False
    assignee_filter = request.args.get('assignee')
    sentiment_filter = request.args.get('sentiment')
    if 'jira_file' not in session or 'gerrit_file' not in session:
        missing_files = True
        return render_template('gerrit.html', gerrit_table=[], missing_files=missing_files, total_bugs=0, total_nonbugs=0, total_gerrit=0)
    # Load JIRA issue keys and filter by team
    filename = session['jira_file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        if filename.lower().endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
        processed_df = process_data(df)
        if selected_team in TEAM_MEMBERS_DICT:
            team_members = TEAM_MEMBERS_DICT[selected_team]
            valid_rows = processed_df[processed_df['Assignee'].apply(lambda x: is_team_member(x, team_members))]
            valid_issue_keys = set(valid_rows['Issue key'].dropna().unique())
            issue_key_to_assignee = dict(zip(valid_rows['Issue key'], valid_rows['Assignee']))
        else:
            valid_rows = processed_df.iloc[0:0]
            valid_issue_keys = set()
            issue_key_to_assignee = {}
        # Calculate total bugs and non-bugs for selected team
        total_bugs = valid_rows[valid_rows['Issue Type'].str.lower() == 'bug'].shape[0]
        total_nonbugs = valid_rows[valid_rows['Issue Type'].str.lower() != 'bug'].shape[0]
        print('[GERRIT] issue_key_to_assignee:', issue_key_to_assignee)
    except Exception as e:
        flash(f'Error processing JIRA file: {e}', 'danger')
        return render_template('gerrit.html', gerrit_table=[], missing_files=missing_files, total_bugs=0, total_nonbugs=0, total_gerrit=0)
    # Load Gerrit JSON
    gerrit_filename = session['gerrit_file']
    gerrit_filepath = os.path.join(app.config['UPLOAD_FOLDER'], gerrit_filename)
    try:
        with open(gerrit_filepath, 'r', encoding='utf-8') as f:
            gerrit_data = json.load(f)
        gerrit_issue_keys = set()
        for entry in gerrit_data:
            issue_key = entry.get('issue_key')
            assignee = issue_key_to_assignee.get(issue_key)
            if not assignee or not is_team_member(assignee, TEAM_MEMBERS_DICT.get(selected_team, [])):
                continue
            if assignee_filter and not is_team_member(assignee, [assignee_filter]):
                continue
            comments = entry.get('comments', {})
            print(f"[GERRIT DEBUG] Issue: {issue_key}, Assignee: {assignee}, Comments empty: {not comments}")
            if isinstance(comments, dict) and comments:  # Only process if comments exist and are not empty
                for comment_list in comments.values():
                    if isinstance(comment_list, list):
                        for comment in comment_list:
                            if isinstance(comment, dict):
                                msg = comment.get('message')
                                sentiment, _, neg_keyword = get_gerrit_sentiment(msg)
                                if sentiment_filter and sentiment != sentiment_filter:
                                    continue
                                # Robustly extract author name
                                author = ''
                                author_obj = comment.get('author')
                                if isinstance(author_obj, dict):
                                    author = author_obj.get('name', '')
                                elif isinstance(author_obj, str):
                                    author = author_obj
                                commit_id = comment.get('commit_id')
                                change_number = entry.get('change_number')
                                gerrit_commit_link = ''
                                if change_number:
                                    gerrit_commit_link = f'https://gerrit.eng.cohesity.com/c/restricted/+/{change_number}'
                                gerrit_table.append({
                                    'Patch Set': comment.get('patch_set'),
                                    'Author': author,
                                    'Line Number': comment.get('line'),
                                    'Message': msg,
                                    'Gerrit Sentiment': sentiment,
                                    'Negative keyword observed': neg_keyword,
                                    'Issue Key': issue_key,
                                    'Assignee': assignee,
                                    'Gerrit Commit Link': gerrit_commit_link
                                })
                                gerrit_issue_keys.add(issue_key)
        total_gerrit = len(gerrit_issue_keys)
        print(f"[GERRIT DEBUG] Total entries processed: {len(gerrit_data)}, Total with comments: {len(gerrit_issue_keys)}, Table rows: {len(gerrit_table)}")
        # Sort gerrit_table: by Assignee, then Author, then Patch Set (as int if possible)
        def patch_set_key(x):
            try:
                return int(x.get('Patch Set') or 0)
            except Exception:
                return 0
        gerrit_table.sort(key=lambda x: (x.get('Issue Key',''), patch_set_key(x)))
    except Exception as e:
        flash(f'Error processing Gerrit JSON: {e}', 'danger')
        gerrit_table = []
    return render_template('gerrit.html', gerrit_table=gerrit_table, missing_files=missing_files, total_bugs=total_bugs, total_nonbugs=total_nonbugs, total_gerrit=total_gerrit)

# Add Jira integration imports and functions
def get_jira_issues_with_all_fields(jira_url, username, api_token, jql_query):
    """
    Fetch issues from Jira using REST API with all available fields
    """
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_token}'
    }
    
    # Jira REST API endpoint for search
    search_url = f"{jira_url.rstrip('/')}/rest/api/2/search"
    
    payload = {
        "jql": jql_query,
        "maxResults": 1000,
        "fields": [
            "key", "summary", "status", "assignee", "reporter", 
            "issuetype", "created", "updated", "comment", "project",
            "priority", "resolution", "resolutiondate", "description",
            "components", "labels", "fixVersions", "versions",
            "customfield_10016", "customfield_10000", "customfield_10001",
            "customfield_10002", "customfield_10003", "customfield_10004",
            "customfield_10005", "customfield_10006", "customfield_10007",
            "customfield_10008", "customfield_10009", "customfield_10010",
            "customfield_10011", "customfield_10012", "customfield_10013",
            "customfield_10014", "customfield_10015", "customfield_10017",
            "customfield_10018", "customfield_10019", "customfield_10020"
        ],
        "expand": ["changelog", "comments"]  # Include comments and changelog
    }
    
    try:
        response = requests.post(
            search_url,
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Jira API error: {response.status_code} - {response.text}")
            
    except Exception as e:
        raise Exception(f"Failed to fetch from Jira: {str(e)}")

def generate_jql_for_team(team_name, team_members, weeks_duration=4):
    """
    Generate JQL query for a specific team
    Format: ((issue_type=bug AND reporter IN (team_members)) OR (issue_type != bug AND assignee IN (team_members)))
    Only fetch issues from the past 6 weeks.
    """
    if team_name == "All":
        # For all teams, include all team members
        all_members = []
        for members in TEAM_MEMBERS_DICT.values():
            if isinstance(members, list):
                all_members.extend(members)
        team_members = list(set(all_members))
    
    if not team_members:
        return ""
    
    # Format team members for JQL
    member_list = ', '.join([f'"{member}"' for member in team_members])
    # Updated JQL: Bugs by created date, non-bugs by resolved date
    jql = (
        f'((issuetype = Bug AND created >= -{weeks_duration}w AND reporter IN ({member_list})) OR '
        f'(issuetype != Bug AND resolved >= -{weeks_duration}w AND assignee IN ({member_list})))'
    )
    return jql

def normalize_name(name):
    """
    Normalize Jira display names to match team member dictionary format
    Converts 'Firstname Lastname [C]' to 'firstname.lastname'
    """
    if not name or pd.isna(name) or isinstance(name, float):
        return name
    
    # Remove [C] suffix and clean up
    name = name.replace(' [C]', '').strip()
    
    # Split into parts and convert to lowercase
    parts = name.split()
    if len(parts) >= 2:
        # Convert 'Firstname Lastname' to 'firstname.lastname'
        return f"{parts[0].lower()}.{parts[1].lower()}"
    elif len(parts) == 1:
        return parts[0].lower()
    else:
        return name.lower()

def is_team_member(name, team_members):
    """
    Check if a name (from CSV) matches any team member in the list
    """
    if not name:
        return False
    
    normalized_name = normalize_name(name)
    return normalized_name in team_members

def jira_data_to_csv_with_all_fields(jira_data, output_file):
    """
    Convert Jira API response to CSV format with all available fields
    """
    if not jira_data or 'issues' not in jira_data:
        raise Exception("No issues found in Jira response")
    
    csv_data = []
    
    for issue in jira_data['issues']:
        fields = issue.get('fields', {})
        
        # Extract comments
        comments = fields.get('comment', {}).get('comments', [])
        comment_text = ' '.join([comment.get('body', '') for comment in comments])
        
        # Get and normalize assignee and reporter names
        assignee_display = fields.get('assignee', {}).get('displayName', '') if fields.get('assignee') else ''
        reporter_display = fields.get('reporter', {}).get('displayName', '') if fields.get('reporter') else ''
        
        # Create row with all available fields
        row = {
            'Issue key': issue.get('key', ''),
            'Summary': fields.get('summary', ''),
            'Status': fields.get('status', {}).get('name', '') if fields.get('status') else '',
            'Assignee': assignee_display,  # Keep original for display
            'Reporter': reporter_display,  # Keep original for display
            'Issue Type': fields.get('issuetype', {}).get('name', '') if fields.get('issuetype') else '',
            'Created': fields.get('created', ''),
            'Updated': fields.get('updated', ''),
            'Comment': comment_text,
            'Project key': fields.get('project', {}).get('key', '') if fields.get('project') else '',
            'Priority': fields.get('priority', {}).get('name', '') if fields.get('priority') else '',
            'Resolution': fields.get('resolution', {}).get('name', '') if fields.get('resolution') else '',
            'Resolved': fields.get('resolutiondate', ''),
            'Sprint': '',  # Sprint info might be in custom fields - will be populated from custom fields
            'Custom field (Story Points)': fields.get('customfield_10016', ''),  # Common story points field
            'Custom field ([CHART] Date of First Response)': ''  # This field is required by process_data
        }
        
        # Add any additional custom fields that might be present
        for field_name, field_value in fields.items():
            if field_name.startswith('customfield_') and field_name not in ['customfield_10016']:
                # Convert custom field name to a more readable format
                display_name = f'Custom field ({field_name})'
                if isinstance(field_value, dict):
                    field_str = field_value.get('value', '') or field_value.get('name', '') or str(field_value)
                    row[display_name] = field_str
                    # Check if this might be Sprint information
                    if 'sprint' in str(field_value).lower() or 'sprint' in str(field_value).lower():
                        row['Sprint'] = field_str
                else:
                    field_str = str(field_value) if field_value is not None else ''
                    row[display_name] = field_str
                    # Check if this might be Sprint information
                    if 'sprint' in str(field_value).lower():
                        row['Sprint'] = field_str
        
        csv_data.append(row)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(csv_data)
    df.to_csv(output_file, index=False)
    
    return len(csv_data)

def run_gerrit_script(csv_file, output_json, gerrit_username=None, gerrit_password=None):
    """
    Run the fetch_gerrit_comments.py script with the generated CSV
    """
    try:
        # Get the path to the gerrit script
        script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'fetch_gerrit_comments.py')
        
        # Build command with optional Gerrit credentials
        cmd = ['python3', script_path, '--csv', csv_file, '--output', output_json]
        
        if gerrit_username and gerrit_password:
            cmd.extend(['--user', gerrit_username, '--password', gerrit_password])
            logging.info(f"Adding Gerrit credentials to command")
        else:
            logging.info(f"No Gerrit credentials provided - will create empty structure")
        
        # Run the script
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            raise Exception(f"Gerrit script failed: {result.stderr}")
        
        return True
        
    except subprocess.TimeoutExpired:
        raise Exception("Gerrit script timed out")
    except Exception as e:
        raise Exception(f"Failed to run Gerrit script: {str(e)}")

@app.route('/auto_fetch_data', methods=['POST'])
def auto_fetch_data():
    """
    Handle automatic data fetching from Jira
    """
    try:
        # Load saved settings
        saved_settings = load_settings()
        
        # Get form data with saved settings as defaults
        jira_url = request.form.get('jira_url', saved_settings.get('server_url', 'https://jira.cohesity.com/'))
        username = request.form.get('jira_username', saved_settings.get('username', ''))
        api_token = request.form.get('jira_token', saved_settings.get('password', ''))
        weeks_duration = int(request.form.get('weeks_duration', 4))
        # Always fetch all data (all teams)
        selected_team = 'All'
        team_members = TEAM_MEMBERS_DICT['All']
        # Remove any use of selected_team from the form data in this route
        fetch_gerrit = request.form.get('fetch_gerrit')
        gerrit_username = request.form.get('gerrit_username', saved_settings.get('gerrit_username', ''))
        gerrit_password = request.form.get('gerrit_password', saved_settings.get('gerrit_password', ''))
        
        # Only require Gerrit credentials if fetch_gerrit is checked
        if fetch_gerrit:
            if not gerrit_username or not gerrit_password:
                return jsonify({'success': False, 'error': 'Gerrit username and password are required to fetch Gerrit data.'})
        
        # Validate inputs
        if not jira_url or not username or not api_token:
            return jsonify({'success': False, 'error': 'Missing required Jira credentials (username and API token)'})
        
        # Get team members for the selected team
        if selected_team in TEAM_MEMBERS_DICT:
            team_members = TEAM_MEMBERS_DICT[selected_team]
        else:
            return jsonify({'success': False, 'error': f'Invalid team: {selected_team}'})
        
        # Generate JQL query for the selected team and weeks duration
        jql_query = generate_jql_for_team(selected_team, team_members, weeks_duration)
        if not jql_query:
            return jsonify({'success': False, 'error': 'No team members found'})
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as csv_file:
            csv_path = csv_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as json_file:
            json_path = json_file.name
        
        try:
            # Step 1: Fetch data from Jira with all fields
            logging.info(f"Fetching data from Jira with JQL: {jql_query}")
            jira_data = get_jira_issues_with_all_fields(jira_url, username, api_token, jql_query)
            
            # Step 2: Convert to CSV with all fields
            logging.info("Converting Jira data to CSV with all fields")
            issue_count = jira_data_to_csv_with_all_fields(jira_data, csv_path)
            
            if issue_count == 0:
                return jsonify({'success': False, 'error': 'No issues found for the selected team'})
            
            # When running the Gerrit script, only do so if fetch_gerrit is checked
            if fetch_gerrit:
                # Step 3: Run Gerrit script
                username_display = '*' * len(gerrit_username) if gerrit_username else 'None'
                password_display = '*' * len(gerrit_password) if gerrit_password else 'None'
                logging.info(f"Running Gerrit script with credentials: username={username_display}, password={password_display}")
                gerrit_success = run_gerrit_script(csv_path, json_path, gerrit_username, gerrit_password)
                if not gerrit_success:
                    return jsonify({'success': False, 'error': 'Failed to process Gerrit comments'})
            else:
                # If not fetching Gerrit, create an empty JSON file for Gerrit data
                with open(json_path, 'w') as f:
                    f.write('[]')
            
            # Step 4: Save files to uploads directory
            upload_dir = app.config['UPLOAD_FOLDER']
            
            # Save CSV
            csv_filename = f"jira_auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            csv_upload_path = os.path.join(upload_dir, csv_filename)
            with open(csv_path, 'r') as src, open(csv_upload_path, 'w') as dst:
                dst.write(src.read())
            
            # Save JSON
            json_filename = f"gerrit_auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            json_upload_path = os.path.join(upload_dir, json_filename)
            with open(json_path, 'r') as src, open(json_upload_path, 'w') as dst:
                dst.write(src.read())
            
            # Store in session
            session['jira_file'] = csv_filename
            session['gerrit_file'] = json_filename
            session['selected_team'] = selected_team
            
            return jsonify({
                'success': True, 
                'message': f'Successfully fetched {issue_count} issues from Jira with all fields',
                'csv_file': csv_filename,
                'json_file': json_filename,
                'download_csv_url': url_for('download_auto_csv'),
                'view_csv_url': url_for('view_auto_csv')
            })
            
        finally:
            # Clean up temporary files
            try:
                os.unlink(csv_path)
                os.unlink(json_path)
            except:
                pass
                
    except Exception as e:
        logging.error(f"Auto fetch error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    """Settings page for Gerrit configuration"""
    show_redirect = False
    if request.method == 'POST':
        try:
            settings_data = {
                'server_url': request.form.get('server_url', ''),
                'username': request.form.get('username', ''),
                'password': request.form.get('password', ''),
                'gerrit_username': request.form.get('gerrit_username', ''),
                'gerrit_password': request.form.get('gerrit_password', '')
            }
            
            if save_settings(settings_data):
                flash('Settings saved successfully!', 'success')
                show_redirect = True
            else:
                flash('Error saving settings!', 'danger')
                
        except Exception as e:
            flash(f'Error: {str(e)}', 'danger')
    
    # Load current settings
    current_settings = load_settings()
    
    return render_template('settings.html', settings=current_settings, show_redirect=show_redirect)

@app.route('/api/settings', methods=['GET'])
def get_settings():
    """API endpoint to get settings"""
    settings = load_settings()
    return jsonify(settings)

@app.route('/download_auto_csv', methods=['GET'])
def download_auto_csv():
    """Download the CSV generated in auto mode"""
    filename = session.get('jira_file')
    if not filename or not filename.startswith('jira_auto_'):
        return jsonify({'success': False, 'error': 'No auto-mode CSV available'}), 404
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'success': False, 'error': 'CSV not found'}), 404
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

@app.route('/view_auto_csv', methods=['GET'])
def view_auto_csv():
    """Render a simple HTML table view of the auto-mode CSV"""
    filename = session.get('jira_file')
    if not filename or not filename.startswith('jira_auto_'):
        return "No auto-mode CSV available", 404
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return "CSV not found", 404
    try:
        df = pd.read_csv(filepath)
    except Exception:
        return "Failed to read CSV", 500
    if len(df) > 1000:
        df = df.head(1000)
    table_html = df.to_html(index=False, border=0)
    return f"""
    <html>
      <head>
        <title>{filename}</title>
        <style>
          table {{ border-collapse: collapse }}
          td, th {{ padding: 6px 10px; border: 1px solid #ddd }}
          body {{ font-family: Arial, sans-serif; margin: 20px; }}
          a {{ text-decoration: none; color: #0a58ca; }}
        </style>
      </head>
      <body>
        <h3>{filename}</h3>
        <div><a href="{url_for('download_auto_csv')}">Download CSV</a></div>
        <div style="margin-top: 12px;">{table_html}</div>
      </body>
    </html>
    """

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8000, debug=True) 