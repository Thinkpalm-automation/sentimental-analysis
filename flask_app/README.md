# Customer Sentiment Analysis Dashboard 

This project provides an interactive dashboard for analyzing customer sentiment from JIRA CSV exports, with advanced filtering, sentiment analysis, and visualizations. The dashboard is built using **Flask** and leverages pandas and custom keyword-based logic for sentiment classification.

## Features
- Upload and analyze JIRA CSV or Excel files
- Sentiment analysis using a 3-class model (positive, neutral, negative) and custom keyword-based overrides
- Filterable tables with row highlighting for negative sentiment
- Negative sentiment breakdown by Assignee or Reporter
- Sentiment distribution charts (Chart.js)
- Optional upload and analysis of consolidated Gerrit comments (JSON)
- Multi-page dashboard: Home, Engineer Drilldown, Bugs, Non-Bugs, Gerrit Analysis
- Professional UI using Bootstrap, DataTables, and Chart.js

## Requirements
- Python 3.10 or higher
- The following Python packages (installed via `requirements.txt`):
  - Flask
  - pandas
  - numpy
  - openpyxl

## Folder Structure
```
flask_app/
  app_flask.py           # Main Flask application
  requirements.txt       # Python dependencies
  sentiment_utils.py     # Helper functions for sentiment analysis
  templates/             # HTML templates (Jinja2)
  static/                # Static assets (images, CSS, JS)
  uploads/               # Uploaded JIRA and Gerrit files
```

## Installation
1. **Navigate to the Flask app directory:**
   ```sh
   cd flask_app
   ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

## Usage
1. **Start the Flask app:**
   ```sh
   python app_flask.py
   ```
2. **Open your browser** and go to [http://localhost:5000](http://localhost:5000)
3. **Upload your JIRA CSV or Excel file** using the uploader on the Home page.
4. **(Optional):** To fetch Gerrit data, check the "Fetch Gerrit Data" box and provide your Gerrit username and password. If left unchecked, only Jira data will be processed.
5. **Explore:**
   - Use the navigation bar to access Home, Engineer Drilldown, Bugs, Non-Bugs, and Gerrit Analysis pages.
   - Use the team selection in the navigation bar to filter data by team after loading all data (auto-fetch always fetches all teams).
   - Filter and drill down by team member, sentiment, reporter, or assignee.
   - View KPIs, sentiment distribution, and detailed tables with color-coded highlights.

## Notes
- For best results, use JIRA exports with columns: Summary, Issue key, Issue Type, Status, Project key, Priority, Resolution, Assignee, Reporter, Created, Updated, Resolved, Sprint, Custom field (Story Points), Custom field ([CHART] Date of First Response), Comment.
- The app uses custom keyword lists for bug/non-bug sentiment overrides and supports Gerrit JSON analysis.
- Table display and filtering is handled by DataTables (JS) for a responsive UI.
- **Auto-fetch always fetches all teams. Team filtering is done after data is loaded using the navigation bar.**
- **Gerrit data is optional and controlled by a checkbox. If you want Gerrit analysis, check the box and provide credentials.**

## License
This project is provided under the MIT License. 