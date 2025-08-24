#!/usr/bin/env python3

import os
import re
import json
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional, Tuple, List

import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
WEEKLY_DIR = os.path.join(PROJECT_ROOT, 'Weekly_Data')

# Reuse processing logic from the Flask app
from flask_app.app_flask import process_data, get_gerrit_sentiment


def find_latest_week_files(weekly_dir: str) -> Tuple[Optional[str], Optional[str], Optional[Tuple[int, int]]]:
    """Return latest (csv_path, json_path, (year, week))."""
    latest: Tuple[int, int, str] = (-1, -1, '')
    for fname in os.listdir(weekly_dir):
        m = re.match(r'^Week(\d+)_(\d+)\s*\.csv$', fname)
        if m:
            w = int(m.group(1))
            y = int(m.group(2))
            if (y, w) > (latest[0], latest[1]):
                latest = (y, w, fname)
    if not latest[2]:
        return None, None, None
    csv_path = os.path.join(weekly_dir, latest[2])
    base = latest[2].rsplit('.', 1)[0]
    json_candidate = os.path.join(weekly_dir, base + '.json')
    json_path = json_candidate if os.path.exists(json_candidate) else None
    return csv_path, json_path, (latest[0], latest[1])


def build_gerrit_table(json_path: Optional[str]) -> pd.DataFrame:
    if not json_path or not os.path.exists(json_path):
        return pd.DataFrame(columns=["Issue", "Project", "Change", "Author", "Message", "Sentiment", "Reason", "Updated", "Patch Set"])  # empty
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return pd.DataFrame(columns=["Issue", "Project", "Change", "Author", "Message", "Sentiment", "Reason", "Updated", "Patch Set"])  # empty

    rows: List[dict] = []
    # Process each item in the JSON array
    for item in data if isinstance(data, list) else []:
        issue_key = item.get('issue_key', '')
        project = item.get('project', '')
        change_number = item.get('change_number', '')
        comments = item.get('comments', {})
        
        # Process all comment types (PATCHSET_LEVEL, file comments, etc.)
        for comment_type, comment_list in comments.items():
            if not isinstance(comment_list, list):
                continue
                
            for comment in comment_list:
                if not isinstance(comment, dict):
                    continue
                    
                message = comment.get('message', '')
                author_obj = comment.get('author', {})
                author = author_obj.get('display_name') or author_obj.get('name', '')
                updated = comment.get('updated', '')
                patch_set = comment.get('patch_set', '')
                
                # Skip empty messages or system messages
                if not message or message.startswith('$'):
                    continue
                    
                # Analyze sentiment
                sentiment, echo, reason = get_gerrit_sentiment(message)
                
                # Only include comments with meaningful content
                if sentiment != "Neutral" or len(message.strip()) > 10:
                    rows.append({
                        "Issue": issue_key,
                        "Project": project,
                        "Change": change_number,
                        "Author": author,
                        "Message": message[:200] + "..." if len(message) > 200 else message,
                        "Sentiment": sentiment,
                        "Reason": reason,
                        "Updated": updated,
                        "Patch Set": patch_set
                    })
    
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=["Issue", "Change", "Updated"], ascending=[True, True, False])
    return df


def read_settings() -> dict:
    # Prefer Documents/sentiment-analysis/settings.json to be consistent with app
    settings_dir = os.path.join(os.path.expanduser('~'), 'Documents', 'sentiment-analysis')
    settings_file = os.path.join(settings_dir, 'settings.json')
    try:
        if os.path.exists(settings_file):
            with open(settings_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    # Fallbacks inside repo
    for p in [os.path.join(PROJECT_ROOT, 'flask_app', 'settings.json'), os.path.join(PROJECT_ROOT, 'settings.json')]:
        try:
            if os.path.exists(p):
                with open(p, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass
    return {}


def render_html(bugs: pd.DataFrame, nonbugs: pd.DataFrame, gerrit: pd.DataFrame, title_suffix: str) -> str:
    def table_html(df: pd.DataFrame) -> str:
        if df.empty:
            return '<div style="color: #6c757d; font-style: italic; padding: 20px; text-align: center;">No data available</div>'
        
        # Limit very large tables
        safe_df = df.head(2000)
        
        # Create email-friendly table with inline styles
        html_table = '<table style="width: 100%; border-collapse: collapse; margin: 10px 0; font-family: Arial, sans-serif; font-size: 12px;">'
        
        # Add header row
        html_table += '<thead><tr style="background-color: #f8f9fa; border-bottom: 2px solid #dee2e6;">'
        for col in safe_df.columns:
            html_table += f'<th style="padding: 12px 8px; text-align: left; font-weight: bold; color: #495057; border: 1px solid #dee2e6;">{col}</th>'
        html_table += '</tr></thead>'
        
        # Add data rows
        html_table += '<tbody>'
        for idx, row in safe_df.iterrows():
            # Alternate row colors for better readability
            bg_color = '#ffffff' if idx % 2 == 0 else '#f8f9fa'
            html_table += f'<tr style="background-color: {bg_color}; border-bottom: 1px solid #dee2e6;">'
            for col in safe_df.columns:
                cell_value = str(row[col]) if pd.notna(row[col]) else ''
                # Truncate long text to prevent table overflow
                if len(cell_value) > 100:
                    cell_value = cell_value[:97] + '...'
                html_table += f'<td style="padding: 8px; text-align: left; border: 1px solid #dee2e6; vertical-align: top;">{cell_value}</td>'
            html_table += '</tr>'
        html_table += '</tbody></table>'
        
        return html_table

    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Weekly Sentiment Report {title_suffix}</title>
</head>
<body style="font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5;">
  <div style="max-width: 1200px; margin: 0 auto; background-color: #ffffff; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
    
    <h2 style="color: #2c3e50; margin-bottom: 30px; padding-bottom: 10px; border-bottom: 3px solid #3498db;">
      Weekly Sentiment Report {title_suffix}
    </h2>

    <div style="margin-bottom: 40px;">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
        <h3 style="color: #e74c3c; margin: 0;">Bugs (Negative Only)</h3>
        <span style="background-color: #e74c3c; color: white; padding: 5px 12px; border-radius: 15px; font-size: 12px; font-weight: bold;">{len(bugs)}</span>
      </div>
      {table_html(bugs)}
    </div>

    <div style="margin-bottom: 40px;">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
        <h3 style="color: #f39c12; margin: 0;">Non-Bugs (Negative Only)</h3>
        <span style="background-color: #f39c12; color: white; padding: 5px 12px; border-radius: 15px; font-size: 12px; font-weight: bold;">{len(nonbugs)}</span>
      </div>
      {table_html(nonbugs)}
    </div>

    <div style="margin-bottom: 40px;">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
        <h3 style="color: #9b59b6; margin: 0;">Gerrit Analysis (Negative Only)</h3>
        <span style="background-color: #9b59b6; color: white; padding: 5px 12px; border-radius: 15px; font-size: 12px; font-weight: bold;">{len(gerrit)}</span>
      </div>
      {table_html(gerrit)}
    </div>

    <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ecf0f1; text-align: center; color: #7f8c8d; font-size: 12px;">
      <p>This report was generated automatically. Please contact the development team for any questions.</p>
    </div>
    
  </div>
</body>
</html>
"""


def send_email_report(html_content: str, subject: str,
                      smtp_host: str,
                      smtp_port: int,
                      smtp_user: str,
                      sender_email: str,
                      sender_password: str,
                      recipient_email: str,
                      debug: bool = False,
                      allow_plain_25: bool = True) -> None:
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg.attach(MIMEText(html_content, 'html', 'utf-8'))

    context = ssl.create_default_context()
    # First try STARTTLS (e.g., 587)
    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
            if debug:
                server.set_debuglevel(1)
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()
            if smtp_user and sender_password:
                server.login(smtp_user, sender_password)
            server.sendmail(sender_email, [recipient_email], msg.as_string())
            return
    except smtplib.SMTPAuthenticationError as e:
        # Fall through to SSL attempt
        auth_error = e
    except Exception:
        auth_error = None

    # Fallback to SMTPS (SSL) typically on 465
    ssl_port = 465
    try:
        with smtplib.SMTP_SSL(smtp_host, ssl_port, context=context, timeout=30) as server:
            if debug:
                server.set_debuglevel(1)
            if smtp_user and sender_password:
                server.login(smtp_user, sender_password)
            server.sendmail(sender_email, [recipient_email], msg.as_string())
            return
    except Exception as e:
        ssl_error = e

    # Optional: last resort try plain SMTP on port 25 without TLS (some internal relays)
    if allow_plain_25:
        try:
            with smtplib.SMTP(smtp_host, 25, timeout=30) as server:
                if debug:
                    server.set_debuglevel(1)
                # Some relays accept mail without auth; attempt auth if provided
                try:
                    if smtp_user and sender_password:
                        server.login(smtp_user, sender_password)
                except Exception:
                    pass
                server.sendmail(sender_email, [recipient_email], msg.as_string())
                return
        except Exception as e:
            raise RuntimeError(f"SMTP send failed. STARTTLS auth error: {auth_error}; SSL error: {ssl_error}; PLAIN: {e}")
    else:
        raise RuntimeError(f"SMTP send failed. STARTTLS auth error: {auth_error}; SSL error: {ssl_error}")


def main() -> int:
    os.makedirs(WEEKLY_DIR, exist_ok=True)
    csv_path, json_path, yw = find_latest_week_files(WEEKLY_DIR)
    if not csv_path:
        print("No weekly CSV found in Weekly_Data")
        return 2

    # Read CSV and process sentiments
    df_raw = pd.read_csv(csv_path)
    df_proc = process_data(df_raw)

    bugs = df_proc[df_proc['Issue Type'].str.lower() == 'bug'][[
        'Issue key', 'Summary', 'Status', 'Reporter', 'Resolved', 'Customer Sentiment', 'Negative Reason'
    ]].copy() if 'Customer Sentiment' in df_proc.columns else pd.DataFrame()

    # Filter bugs to negative-only
    if not bugs.empty and 'Customer Sentiment' in bugs.columns:
        bugs = bugs[bugs['Customer Sentiment'] == 'Negative']

    nonbugs = df_proc[df_proc['Issue Type'].str.lower() != 'bug'][[
        'Issue key', 'Summary', 'Status', 'Assignee', 'Resolved', 'Customer Sentiment', 'Negative Reason'
    ]].copy() if 'Customer Sentiment' in df_proc.columns else pd.DataFrame()

    # Filter non-bugs to negative-only
    if not nonbugs.empty and 'Customer Sentiment' in nonbugs.columns:
        nonbugs = nonbugs[nonbugs['Customer Sentiment'] == 'Negative']

    gerrit = build_gerrit_table(json_path)
    
    # Filter to negative-only for Gerrit analysis
    if not gerrit.empty and 'Sentiment' in gerrit.columns:
        gerrit = gerrit[gerrit['Sentiment'] == 'Negative']
    
    # Select relevant columns for display
    gerrit_display = gerrit[["Issue", "Project", "Change", "Author", "Message", "Sentiment", "Reason"]].copy() if not gerrit.empty else gerrit

    suffix = '' if not yw else f"(Week {yw[1]}, {yw[0]})"
    html_text = render_html(bugs, nonbugs, gerrit_display, suffix)

    # Write HTML to Weekly_Data
    out_name = 'Weekly_Sentiment_Report.html' if not yw else f"Weekly_Sentiment_Report_Week{yw[1]}_{yw[0]}.html"
    out_path = os.path.join(WEEKLY_DIR, out_name)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html_text)
    print(f"Report written to {out_path}")

    # Email the report using provided credentials (can adjust SMTP host/port if needed)
    subject = f"Weekly Sentiment Report {suffix}".strip()
    smtp_host = os.environ.get('SMTP_HOST', 'smtp.gmail.com')
    smtp_port = int(os.environ.get('SMTP_PORT', '587'))
    sender_email = os.environ.get('SENDER_EMAIL', 'tharunsudharsan11@gmail.com')
    smtp_user = os.environ.get('SMTP_USER', sender_email)
    sender_password = os.environ.get('SENDER_PASSWORD', 'wwqp aoog wwrh wwii')
    recipient_email = os.environ.get('RECIPIENT_EMAIL', 'tharun11sudharsan@gmail.com')
    smtp_debug = os.environ.get('SMTP_DEBUG', '0') == '1'
    allow_plain_25 = os.environ.get('SMTP_ALLOW_PLAIN_25', '1') == '1'

    try:
        send_email_report(html_text, subject, smtp_host, smtp_port, smtp_user, sender_email, sender_password, recipient_email, debug=smtp_debug, allow_plain_25=allow_plain_25)
        print(f"Email sent to {recipient_email}")
    except Exception as e:
        print(f"Failed to send email: {e}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())


