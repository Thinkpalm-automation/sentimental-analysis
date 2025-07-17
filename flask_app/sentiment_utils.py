import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import re

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
    "not needed", "unused variable", "remove this", "should be deleted", "unnecessary", "redundant", "typo", "incorrect", "does not work", "breaks", "missing", "needs improvement", "not clear", "confusing", "bad practice", "hardcoded", "magic number", "not efficient", "performance issue", "security issue", "potential bug", "should be refactored", "not following convention", "incomplete", "wrong", "fails", "deprecated", "not tested", "test missing", "should be documented", "no comments", "unclear logic", "duplicate code", "not readable", "too complex", "overcomplicated", "not optimal", "incorrect indentation", "formatting issue", "merge conflict", "conflicts with", "not reviewed", "needs changes", "needs update", "needs rebase", "not aligned", "not matching", "not consistent", "not handled", "not covered", "not robust", "not thread safe", "race condition", "memory leak", "null pointer", "exception not handled", "error prone", "not scalable", "not maintainable"
]

def load_sentiment_pipeline():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

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

def get_sentiment(comment, issue_type, sentiment_pipeline):
    if pd.isna(comment) or str(comment).strip() == "":
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
    try:
        result = sentiment_pipeline(str(comment))[0]
        label = result['label'].lower()
        if label == 'positive':
            return "Positive", ""
        elif label == 'neutral':
            return "Neutral", ""
        elif label == 'negative':
            return "Negative", f'Comment-based: "{comment}"'
        else:
            return "Neutral", ""
    except Exception:
        return "Neutral", ""

def story_points_flag(row):
    try:
        sp = float(row.get("Custom field (Story Points)", 0))
    except (TypeError, ValueError):
        sp = 0
    return "🚩 High Story Points" if str(row.get("Issue Type", "")).strip().lower() != "bug" and sp >= 5 else ""

def get_gerrit_sentiment(comment):
    if pd.isna(comment) or str(comment).strip() == "":
        return "Neutral", "", ""
    comment_lower = str(comment).lower()
    for kw in GERRIT_NEGATIVE_KEYWORDS:
        if kw in comment_lower:
            return "Negative", str(comment), kw
    return "Neutral", "", ""

def process_data(df, sentiment_pipeline):
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
        if issue_type != "bug" and issue_type != "service request":
            x = row.get("Comment", "")
            return "❌ Missing Comment" if pd.isna(x) or str(x).strip() == "" else ""
        return ""
    df_filtered["Story Points Flag"] = df_filtered.apply(custom_story_points_flag, axis=1)
    df_filtered["Comment Flag"] = df_filtered.apply(custom_comment_flag, axis=1)
    def sentiment_and_reason(row):
        issue_type = str(row.get("Issue Type", "")).strip().lower()
        comment = row.get("Comment", "")
        sp_flag = row.get("Story Points Flag", "")
        comment_flag = row.get("Comment Flag", "")
        if issue_type != "bug" and issue_type != "service request":
            if "High Story Points" in sp_flag and "Missing Comment" in comment_flag:
                return pd.Series({"Customer Sentiment": "Negative", "Negative Reason": "High Story Points and Missing Comment"})
            elif "High Story Points" in sp_flag:
                return pd.Series({"Customer Sentiment": "Negative", "Negative Reason": "High Story Points"})
            elif "Missing Comment" in comment_flag:
                return pd.Series({"Customer Sentiment": "Negative", "Negative Reason": "Missing Comment"})
            else:
                sentiment, neg_reason = get_sentiment(comment, issue_type, sentiment_pipeline)
                if sentiment == "Negative":
                    return pd.Series({"Customer Sentiment": "TBD", "Negative Reason": neg_reason or f'Comment-based: "{comment}"'})
                else:
                    return pd.Series({"Customer Sentiment": sentiment, "Negative Reason": neg_reason})
        elif issue_type == "bug":
            comment_sentiment, neg_reason = get_sentiment(comment, issue_type, sentiment_pipeline)
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
            sentiment, neg_reason = get_sentiment(comment, issue_type, sentiment_pipeline)
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