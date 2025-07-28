from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
import pandas as pd
from werkzeug.utils import secure_filename
import numpy as np

# --- Sentiment Analysis Keyword Lists and Functions (from app.py) ---
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
            is_team_reporter = processed_df['Reporter'].isin(team_members)
            nonbug_mask = (processed_df['Issue Type'].str.lower().str.strip() != 'bug')
            is_team_assignee = processed_df['Assignee'].isin(team_members)
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
            assignees = [a for a in TEAM_MEMBERS_DICT[selected_team] if a in processed_df['Assignee'].dropna().unique().tolist()]
        else:
            assignees = []
        issue_key_to_assignee = dict(zip(processed_df['Issue key'], processed_df['Assignee']))
        for assignee in assignees:
            team_sentiment[assignee] = {"Positive": 0, "Neutral": 0, "Negative": 0}
            # Bugs (by Reporter, to match engineer drilldown)
            bug_rows = processed_df[(processed_df['Reporter'] == assignee) & (processed_df['Issue Type'].str.lower() == 'bug')]
            for sentiment in ["Positive", "Neutral", "Negative"]:
                team_sentiment[assignee][sentiment] += (bug_rows['Customer Sentiment'] == sentiment).sum()
            # Non-bugs (by Assignee)
            nonbug_rows = processed_df[(processed_df['Assignee'] == assignee) & (processed_df['Issue Type'].str.lower() != 'bug')]
            for sentiment in ["Positive", "Neutral", "Negative"]:
                team_sentiment[assignee][sentiment] += (nonbug_rows['Customer Sentiment'] == sentiment).sum()
        # 2. Gerrit comments (if available)
        if gerrit_data:
            for entry in gerrit_data:
                issue_key = entry.get('issue_key')
                assignee = issue_key_to_assignee.get(issue_key)
                if not assignee or assignee not in TEAM_MEMBERS_DICT.get(selected_team, []):
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
                                    if sentiment in team_sentiment[assignee]:
                                        team_sentiment[assignee][sentiment] += 1
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
            is_team_reporter = processed_df['Reporter'].isin(team_members)
            total_bugs = processed_df[bug_mask & is_team_reporter].shape[0]
            # Total non-bugs: Assignee is a team member
            nonbug_mask = processed_df['Issue Type'].str.lower().str.strip() != 'bug'
            is_team_assignee = processed_df['Assignee'].isin(team_members)
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
        total_gerrit=total_gerrit
    )

@app.route('/set_team', methods=['POST'])
def set_team():
    team = request.form.get('selected_team', '')
    session['selected_team'] = team  # Always set, even if empty
    print('Setting selected_team:', team)
    print('Session after set:', dict(session))
    referrer = request.headers.get('Referer')
    return redirect(referrer or url_for('home'))

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
                team_members = [m for m in TEAM_MEMBERS_DICT[selected_team] if m in processed_df['Assignee'].dropna().unique().tolist()]
            else:
                team_members = []
            filters["members"] = team_members
            selected_member = request.args.get('member', 'All')
            # Bugs (filter by Reporter for team membership)
            bug_df = processed_df[(processed_df['Issue Type'].str.lower() == 'bug')] if 'Issue Type' in processed_df else processed_df
            bug_df = bug_df[bug_df['Reporter'].isin(team_members)] if 'Reporter' in bug_df else bug_df
            # Non-bugs (filter by Assignee)
            nonbug_df = processed_df[(processed_df['Issue Type'].str.lower() != 'bug')] if 'Issue Type' in processed_df else processed_df
            nonbug_df = nonbug_df[nonbug_df['Assignee'].isin(team_members)] if 'Assignee' in nonbug_df else nonbug_df
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
                    if not assignee or assignee not in team_members:
                        continue
                    comments = entry.get('comments', {})
                    if isinstance(comments, dict):
                        for comment_list in comments.values():
                            if isinstance(comment_list, list):
                                for comment in comment_list:
                                    if isinstance(comment, dict):
                                        msg = comment.get('message')
                                        sentiment, _, _ = get_gerrit_sentiment(msg)
                                        if sentiment in gerrit_counts[assignee]:
                                            gerrit_counts[assignee][sentiment] += 1
            for member in team_members:
                if selected_member != 'All' and member != selected_member:
                    continue
                bugs = bug_df[bug_df['Reporter'] == member] if 'Reporter' in bug_df else pd.DataFrame()
                bug_counts = bugs['Customer Sentiment'].value_counts().to_dict() if not bugs.empty else {}
                nonbugs = nonbug_df[nonbug_df['Assignee'] == member] if 'Assignee' in nonbug_df else pd.DataFrame()
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
                bug_df = bug_df[bug_df['Reporter'].isin(TEAM_MEMBERS_DICT[selected_team])] if 'Reporter' in bug_df else bug_df
                filters["reporters"] = [r for r in TEAM_MEMBERS_DICT[selected_team] if r in bug_df['Reporter'].dropna().unique().tolist()] if 'Reporter' in bug_df else []
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
                nonbug_df = nonbug_df[nonbug_df['Assignee'].isin(TEAM_MEMBERS_DICT[selected_team])] if 'Assignee' in nonbug_df else nonbug_df
                filters["assignees"] = sorted([a for a in TEAM_MEMBERS_DICT[selected_team] if a in nonbug_df['Assignee'].dropna().unique().tolist()]) if 'Assignee' in nonbug_df else []
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
            valid_rows = processed_df[processed_df['Assignee'].isin(team_members)]
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
            if not assignee or assignee not in TEAM_MEMBERS_DICT.get(selected_team, []):
                continue
            if assignee_filter and assignee != assignee_filter:
                continue
            comments = entry.get('comments', {})
            if isinstance(comments, dict):
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

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8000, debug=True) 