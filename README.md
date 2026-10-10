# CreditWise — Credit Card Default Risk Intelligence

A runnable hackathon prototype that trains a model on labelled credit-card data, scores customers by predicted default risk, creates a ranked review queue, suggests review actions, and provides a Streamlit dashboard.

> **Important:** This is a decision-support demo, not a production credit-scoring system. Do not use predictions as the sole basis for credit limits, lending, collections, or other adverse decisions. The built-in synthetic dataset is artificial and is only for testing the app.

## Features

- Upload CSV, XLSX, or XLS data.
- Built-in synthetic demo data, so the app can run before the real dataset is downloaded.
- Detects common target column names, including `default.payment.next.month`.
- Excludes obvious ID columns from model training.
- Train/test split and held-out metrics: ROC-AUC, PR-AUC, precision, recall, accuracy, confusion matrix.
- Ranked Top-500 review queue (configurable).
- Top-20 action plan and CSV exports.
- Risk distribution and feature-importance charts.
- Human-review recommendations and visible limitations.

## 1. Requirements

- Windows 10/11, macOS, or Linux
- Python 3.10–3.12 recommended
- Internet access for first-time package installation

Check Python:
```bash
python --version
```
On some Windows computers, use `py --version`.

## 2. Run on Windows

1. Extract `CreditWise_Project.zip`.
2. Open the extracted `CreditWise_Project` folder in VS Code.
3. Open Terminal → New Terminal.
4. Run:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If `python` is not recognized, try replacing `python` with `py` in the commands above.

The app opens in a browser, usually at `http://localhost:8501`.

### Easy Windows option
Double-click `run_windows.bat`. It creates a virtual environment if needed, installs dependencies, and launches the app.

## 3. Run on macOS / Linux

From the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Or run `bash run_mac_linux.sh`.

## 4. Test immediately (no real dataset needed)

On the left sidebar, keep **Use synthetic demo data** selected. The dashboard will generate artificial records and run the complete workflow.

**Do not present synthetic demo metrics as real-world performance.**

## 5. Use the real UCI dataset

The project brief refers to **Default of Credit Card Clients** (UCI / Kaggle). Download the dataset from the UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

Download the data, then upload the CSV/XLS/XLSX file using the app's sidebar.

If the original Excel file has a two-row header or an extra explanatory row, save a clean copy with one header row and one row per customer before uploading. The target should be named `default.payment.next.month` (or a recognizable default/target column). Expected target values are `0 = no default`, `1 = default`.

## 6. Output files

From the **Risk review queue** tab:
- `creditwise_ranked_queue.csv` — all scored rows, ranked by predicted risk.
- `creditwise_top20_actions.csv` — highest-ranked 20 rows with suggested actions.

The app offers download buttons; files are saved wherever you choose in the browser's download settings.

## 7. Suggested hackathon demo

1. Start with the business problem: default losses vs. false alarms.
2. Show dataset size and columns, and explain why IDs are not model features.
3. Show held-out precision, recall, PR-AUC, and the confusion matrix.
4. Open the ranked Top-500 queue and explain the first few risk indicators.
5. Show the Top-20 review actions and download the CSV.
6. Close with limitations: data quality, fairness review, threshold trade-off, and human decision-making.

## Troubleshooting

- **`python` not found:** Install Python 3.10–3.12 from https://www.python.org/downloads/ and enable “Add Python to PATH”, then reopen VS Code.
- **`streamlit` not found:** Activate `.venv` and run `pip install -r requirements.txt`.
- **Port already in use:** Run `streamlit run app.py --server.port 8502`.
- **Upload target not found:** Check the column names; use `default.payment.next.month` and confirm values are 0/1.
- **Model training fails:** Check that both target classes (0 and 1) exist and that the uploaded file has at least 50 valid labelled rows.
- **Dashboard opens but data is not real:** The synthetic demo checkbox is enabled. Upload the real dataset to evaluate on real data.


## Fix for uploaded UCI `.xls` target-value error

Use the updated `app.py` in this ZIP. It preserves missing target values until rows with invalid/missing labels are filtered, avoiding the `Cannot convert non-finite values (NA or inf) to integer` error. Replace your extracted project's `app.py` with this version, then restart Streamlit.
