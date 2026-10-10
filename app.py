from __future__ import annotations
import os
from groq import Groq

import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

warnings.filterwarnings("ignore")

st.set_page_config(page_title="CreditWise | Default Risk Intelligence", page_icon="💳", layout="wide")

TARGET_CANDIDATES = [
    "default.payment.next.month", "default payment next month", "default", "target", "y", "label"
]
ID_CANDIDATES = ["id", "customer_id", "customer id", "unnamed: 0"]

def norm_name(value):
    return str(value).strip().lower().replace("_", " ").replace("-", " ")

def find_target(df):
    lookup = {norm_name(c): c for c in df.columns}
    for candidate in TARGET_CANDIDATES:
        if norm_name(candidate) in lookup:
            return lookup[norm_name(candidate)]
    # UCI Excel commonly labels the target as "default.payment.next.month"
    for c in df.columns:
        n = norm_name(c)
        if "default" in n and ("month" in n or "payment" in n):
            return c
    return None

def clean_dataframe(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    df = df.dropna(axis=0, how="all")
    # Remove Excel header row accidentally included as data.
    first = df.columns[0]
    df = df[df[first].astype(str).str.lower().ne(str(first).lower())]
    return df.reset_index(drop=True)

def normalize_target(y):
    y = y.copy()
    if y.dtype == object:
        y = y.astype(str).str.strip().str.lower().map({
            "1": 1, "yes": 1, "true": 1, "default": 1, "defaulted": 1,
            "0": 0, "no": 0, "false": 0, "paid": 0, "non-default": 0,
            "non default": 0
        })
    y = pd.to_numeric(y, errors="coerce")
    # Keep missing target values as NaN here. prepare_features() filters them
    # before converting labels to integers; casting first raises on NA/inf.
    y = y.replace([np.inf, -np.inf], np.nan)
    vals = set(y.dropna().unique().tolist())
    if vals.issubset({0, 1}) and len(vals) == 2:
        return y
    return y

def make_demo_data(n=3000, seed=42):
    """Clearly labelled synthetic data for testing the app when the real dataset is unavailable."""
    rng = np.random.default_rng(seed)
    limit = rng.choice([10000, 20000, 50000, 100000, 200000, 300000], n,
                       p=[.08, .14, .22, .25, .20, .11])
    age = np.clip(rng.normal(36, 10, n), 21, 75).astype(int)
    pay0 = rng.choice([-2, -1, 0, 1, 2, 3], n, p=[.08, .12, .48, .19, .09, .04])
    pay2 = np.clip(pay0 + rng.choice([-1, 0, 1, 2], n, p=[.12, .52, .27, .09]), -2, 8)
    bill1 = np.maximum(0, limit * rng.uniform(.05, 1.15, n))
    pay_amt1 = np.maximum(0, bill1 * rng.uniform(.02, .55, n))
    logits = -2.0 + 0.62 * np.maximum(pay0, 0) + 0.24 * np.maximum(pay2, 0) + 1.1 * (pay_amt1 < bill1 * .1) + .35 * (bill1 > limit * .85)
    p = 1 / (1 + np.exp(-logits))
    y = rng.binomial(1, p)
    return pd.DataFrame({
        "ID": np.arange(1, n + 1), "LIMIT_BAL": limit, "SEX": rng.integers(1, 3, n),
        "EDUCATION": rng.integers(1, 5, n), "MARRIAGE": rng.integers(1, 4, n), "AGE": age,
        "PAY_0": pay0, "PAY_2": pay2, "PAY_3": np.clip(pay2 + rng.integers(-1, 2, n), -2, 8),
        "BILL_AMT1": bill1.round(0), "PAY_AMT1": pay_amt1.round(0),
        "PAY_AMT2": (pay_amt1 * rng.uniform(.4, 1.3, n)).round(0),
        "default.payment.next.month": y
    })

def prepare_features(df, target_col):
    work = clean_dataframe(df)
    y = normalize_target(work[target_col])
    valid = y.isin([0, 1])
    work, y = work.loc[valid].copy(), y.loc[valid].astype(int)
    if len(work) < 50:
        raise ValueError("At least 50 labelled rows are required to train a meaningful prototype.")
    # Remove target and obvious identifier columns. ID must never be a model feature.
    feature_cols = []
    for c in work.columns:
        if c == target_col:
            continue
        if norm_name(c) in ID_CANDIDATES or norm_name(c).startswith("unnamed"):
            continue
        feature_cols.append(c)
    if not feature_cols:
        raise ValueError("No usable feature columns remain after removing target and ID columns.")
    X = work[feature_cols].copy()
    # Convert numeric-looking strings to numbers where possible.
    for c in X.columns:
        if X[c].dtype == object:
            converted = pd.to_numeric(X[c], errors="coerce")
            if converted.notna().mean() >= 0.85:
                X[c] = converted
    # Drop constant or all-empty columns.
    X = X.loc[:, X.nunique(dropna=True) > 1]
    X = X.dropna(axis=1, how="all")
    if X.shape[1] == 0:
        raise ValueError("No informative columns found.")
    return X.reset_index(drop=True), y.reset_index(drop=True), feature_cols

def build_model(X):
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical))
    prep = ColumnTransformer(transformers=transformers, remainder="drop")
    model = RandomForestClassifier(
        n_estimators=220, min_samples_leaf=3, class_weight="balanced",
        random_state=42, n_jobs=-1, max_features="sqrt"
    )
    return Pipeline([("prep", prep), ("model", model)])

def reason_for_row(row, importances):
    # Transparent, heuristic explanations; these are not causal explanations.
    names = list(importances.index)
    vals = []
    def get_col(fragment):
        for name in names:
            if fragment.lower() in norm_name(name):
                return name
        return None
    c_pay0 = get_col("pay_0") or get_col("pay 0") or get_col("repayment status")
    c_pay2 = get_col("pay_2") or get_col("pay 2")
    c_limit = get_col("limit_bal") or get_col("limit bal") or get_col("credit limit")
    c_bill = get_col("bill_amt1") or get_col("bill amt1")
    c_payamt = get_col("pay_amt1") or get_col("pay amt1")
    if c_pay0:
        try:
            v = float(row[c_pay0])
            if v > 0: vals.append((f"Recent repayment delay (value {v:g})", float(importances.get(c_pay0, 0)) * (1 + min(v, 6) / 3)))
        except: pass
    if c_pay2:
        try:
            v = float(row[c_pay2])
            if v > 0: vals.append((f"Earlier repayment delay (value {v:g})", float(importances.get(c_pay2, 0)) * (1 + min(v, 6) / 3)))
        except: pass
    if c_bill and c_limit:
        try:
            ratio = float(row[c_bill]) / max(float(row[c_limit]), 1)
            if ratio > .75: vals.append((f"High bill-to-limit ratio ({ratio:.0%})", float(importances.get(c_bill, 0)) + float(importances.get(c_limit, 0)) + ratio * .01))
        except: pass
    if c_payamt and c_bill:
        try:
            ratio = float(row[c_payamt]) / max(abs(float(row[c_bill])), 1)
            if ratio < .10: vals.append((f"Low payment relative to bill ({ratio:.0%})", float(importances.get(c_payamt, 0)) + float(importances.get(c_bill, 0)) + .01))
        except: pass
    # Fill remaining reasons with influential features, explicitly described as model signals.
    for name in importances.sort_values(ascending=False).index:
        if len(vals) >= 3: break
        if any(name.lower() in item[0].lower() for item in vals): continue
        vals.append((f"Model signal: {name}", float(importances.get(name, 0))))
    vals = sorted(vals, key=lambda x: x[1], reverse=True)
    return [v[0] for v in vals[:3]]

def get_importances(model, feature_names):
    try:
        prep = model.named_steps["prep"]
        rf = model.named_steps["model"]
        transformed = prep.get_feature_names_out()
        raw_imp = pd.Series(rf.feature_importances_, index=transformed)
        result = {}
        for feat in feature_names:
            matches = [idx for idx in raw_imp.index if idx.split("__")[-1] == feat or idx.split("__")[-1].startswith(feat + "_")]
            result[feat] = float(raw_imp[matches].sum()) if matches else 0.0
        s = pd.Series(result).sort_values(ascending=False)
        return s / s.sum() if s.sum() else s
    except Exception:
        return pd.Series({c: 1 / max(len(feature_names), 1) for c in feature_names})

def load_uploaded_file(upload):
    name = upload.name.lower()
    if name.endswith(".csv"):
        try: return pd.read_csv(upload)
        except UnicodeDecodeError:
            upload.seek(0); return pd.read_csv(upload, encoding="latin-1")
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(upload)
    raise ValueError("Please upload a CSV, XLSX or XLS file.")

def main():
    st.title("💳 CreditWise")
    st.caption("Credit-card default risk prioritization | Hackathon prototype")
    st.info("Decision-support prototype only. Predictions are not proof of future default and must not be used as the sole basis for adverse credit decisions.")

    with st.sidebar:
        st.header("Data")
        uploaded = st.file_uploader("Upload labelled customer data", type=["csv", "xlsx", "xls"])
        demo = st.checkbox("Use synthetic demo data (not real customers)", value=not bool(uploaded))
        st.caption("Expected target: default.payment.next.month (0 = no default, 1 = default). The UCI dataset is also supported.")
        st.divider()
        st.header("Risk queue")
        queue_size = st.slider("Customers in review queue", 50, 1000, 500, 50)
        threshold = st.slider("High-risk threshold", 0.10, 0.90, 0.50, 0.05)

    if uploaded is not None:
        try:
            raw = load_uploaded_file(uploaded)
            data_source = f"Uploaded file: {uploaded.name}"
        except Exception as e:
            st.error(f"Could not read the file: {e}")
            st.stop()
    elif demo:
        raw = make_demo_data()
        data_source = "Synthetic demo data — do not present as real-world model performance"
    else:
        st.warning("Upload a labelled dataset or select the synthetic demo option in the sidebar.")
        st.stop()

    raw = clean_dataframe(raw)
    target_col = find_target(raw)
    if target_col is None:
        st.error("Target column not found. Expected 'default.payment.next.month' (or a column named default/target/Y).")
        st.write("Columns detected:", list(raw.columns))
        st.stop()

    try:
        X, y, original_features = prepare_features(raw, target_col)
    except Exception as e:
        st.error(str(e)); st.stop()

    if y.nunique() < 2:
        st.error("Target must contain both classes: 0 (no default) and 1 (default).")
        st.stop()

    st.caption(f"Data source: {data_source} | {len(X):,} usable rows | {X.shape[1]} model features | target: {target_col}")
    class_counts = y.value_counts()
    stratify = y if class_counts.min() >= 2 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=stratify
        )
        model = build_model(X)
        with st.spinner("Training model and evaluating held-out data..."):
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= threshold).astype(int)
    except Exception as e:
        st.error(f"Model training failed: {e}")
        st.stop()

    roc = roc_auc_score(y_test, probs) if y_test.nunique() == 2 else float("nan")
    ap = average_precision_score(y_test, probs) if y_test.nunique() == 2 else float("nan")
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    accuracy = accuracy_score(y_test, preds)

    # Score all records with a model trained on training split. For hackathon ranking,
    # held-out metrics remain separately reported to avoid claiming training scores as test scores.
    all_probs = model.predict_proba(X)[:, 1]
    ranked = raw.loc[X.index].copy()
    ranked["Predicted_Default_Risk"] = all_probs
    ranked["Risk_Band"] = pd.cut(all_probs, bins=[-0.001, .25, .50, .75, 1.001],
                                  labels=["Low", "Moderate", "High", "Very High"])
    importances = get_importances(model, X.columns.tolist())
    reasons = [reason_for_row(X.iloc[i], importances) for i in range(len(X))]
    ranked["Top_Risk_Reasons"] = ["; ".join(r) for r in reasons]
    ranked["Recommended_Action"] = np.where(
        all_probs >= .75, "Priority human review: contact customer and discuss payment options",
        np.where(all_probs >= .50, "Review recent payment history; offer reminder or payment-plan discussion",
                 np.where(all_probs >= .25, "Send standard reminder and monitor next payment", "Routine monitoring"))
    )
    # Preserve source identifiers for review display, but never feed them to the model.
    id_col = next((c for c in raw.columns if norm_name(c) in ID_CANDIDATES), None)
    display_id = id_col if id_col else None
    ranked = ranked.sort_values("Predicted_Default_Risk", ascending=False).reset_index(drop=True)
    ranked.insert(0, "Review_Rank", np.arange(1, len(ranked) + 1))
    queue = ranked.head(min(queue_size, len(ranked))).copy()

    st.subheader("Portfolio overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers scored", f"{len(ranked):,}")
    c2.metric("High-risk at threshold", f"{int((all_probs >= threshold).sum()):,}")
    c3.metric("Held-out ROC-AUC", f"{roc:.3f}" if np.isfinite(roc) else "N/A")
    c4.metric("Held-out PR-AUC", f"{ap:.3f}" if np.isfinite(ap) else "N/A")

    tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Risk review queue", "Model evaluation", "Data & methodology"])
    with tab1:
        left, right = st.columns(2)
        with left:
            band_counts = ranked["Risk_Band"].value_counts().reindex(["Low", "Moderate", "High", "Very High"], fill_value=0).reset_index()
            band_counts.columns = ["Risk band", "Customers"]
            st.plotly_chart(px.bar(band_counts, x="Risk band", y="Customers", title="Customers by predicted risk band"), use_container_width=True)
        with right:
            st.plotly_chart(px.histogram(ranked, x="Predicted_Default_Risk", nbins=25, title="Predicted risk distribution"), use_container_width=True)
        st.markdown("**Most influential model features**")
        imp_df = importances.head(10).sort_values().reset_index()
        imp_df.columns = ["Feature", "Relative importance"]
        st.plotly_chart(px.bar(imp_df, x="Relative importance", y="Feature", orientation="h"), use_container_width=True)
        st.caption("Feature importance describes model reliance, not causation. Validate explanations and fairness before production use.")

    with tab2:
        st.write(f"Showing the top {len(queue):,} customers, sorted by predicted default risk.")
        search = st.text_input("Filter by customer ID (if available) or risk reason")
        view = queue.copy()
        if search:
            mask = view.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)
            view = view[mask]
        columns_to_show = (["Review_Rank", display_id] if display_id else ["Review_Rank"]) + [
            "Predicted_Default_Risk", "Risk_Band", "Top_Risk_Reasons", "Recommended_Action"
        ]
        columns_to_show = [c for c in columns_to_show if c in view.columns]
        show = view[columns_to_show].copy()
        show["Predicted_Default_Risk"] = (show["Predicted_Default_Risk"] * 100).round(1).astype(str) + "%"
        st.dataframe(show, use_container_width=True, hide_index=True)
        top20 = ranked.head(min(20, len(ranked)))
        st.markdown("**Top-20 action plan** — click 🤖 Ask AI on any row")
        ai_top20_table(top20, X.columns, display_id)
        st.markdown("**Top-20 action plan**")
        st.dataframe(top20[columns_to_show].assign(
            Predicted_Default_Risk=(top20["Predicted_Default_Risk"] * 100).round(1).astype(str) + "%"
        ), use_container_width=True, hide_index=True)

        
        st.download_button("⬇️ Download full ranked queue (CSV)", ranked.to_csv(index=False).encode("utf-8"), "creditwise_ranked_queue.csv", "text/csv")
        st.download_button("⬇️ Download Top-20 action plan (CSV)", top20.to_csv(index=False).encode("utf-8"), "creditwise_top20_actions.csv", "text/csv")

    with tab3:
        st.write("Metrics are calculated on a held-out 25% test split.")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{accuracy:.3f}")
        m2.metric("Precision", f"{precision:.3f}")
        m3.metric("Recall", f"{recall:.3f}")
        m4.metric("PR-AUC", f"{ap:.3f}" if np.isfinite(ap) else "N/A")
        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        st.plotly_chart(px.imshow(cm, text_auto=True, x=["Predicted no default", "Predicted default"],
                                  y=["Actual no default", "Actual default"], title="Confusion matrix"), use_container_width=True)
        st.code(classification_report(y_test, preds, zero_division=0), language="text")
        st.caption("Threshold changes the precision/recall trade-off. Choose it based on review capacity and validated business costs, not convenience.")
    with tab4:
        st.markdown("""
        **Workflow**
        1. Remove identifier columns from model features.
        2. Split labelled records into train/test sets before fitting preprocessing.
        3. Impute missing values and one-hot encode categorical values inside the training pipeline.
        4. Train a class-weighted Random Forest.
        5. Evaluate on held-out records and rank customers by predicted probability.
        6. Generate review suggestions for a human analyst.

        **Important limitations**
        - The synthetic demo dataset is artificial and must never be described as actual bank data or real performance.
        - Risk reasons are heuristic indicators based on available columns and feature importance; they are not causal explanations.
        - Demographic variables can create fairness and legal concerns. Review whether each feature is appropriate before operational use.
        - This is a hackathon prototype, not a validated lending or collections system.
        """)
        st.write("Columns in source dataset:", list(raw.columns))
        st.write("Model features used:", list(X.columns))
        st.write("Identifier columns excluded from training when detected:", [c for c in raw.columns if norm_name(c) in ID_CANDIDATES or norm_name(c).startswith("unnamed")])
SENSITIVE = {"sex", "marriage", "education", "age"}

SYSTEM_PROMPT = (
    "You are a credit-risk analyst assistant. Explain in 3-4 plain sentences why a customer "
    "was ranked high in a default-risk review queue. Use ONLY the facts provided. "
    "Do not invent numbers. Describe model signals, not proof of future default. "
    "End with one suggested follow-up action."
)

def get_api_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.getenv("GROQ_API_KEY")

def build_facts(row, columns):
    facts = {}
    for c in columns:
        if norm_name(c) in SENSITIVE:
            continue
        try:
            facts[c] = round(float(row[c]), 2)
        except Exception:
            pass
    return facts

@st.fragment
def ai_top20_table(top20, feature_cols, display_id):
    if not get_api_key():
        st.caption("Add GROQ_API_KEY to .streamlit/secrets.toml to enable AI explanations.")
        return
    widths = [0.7, 1, 1, 1.2, 4, 1.5]
    for col, h in zip(st.columns(widths), ["Rank", "ID", "Risk", "Band", "Top risk reasons", "AI"]):
        col.markdown(f"**{h}**")
    for _, row in top20.iterrows():
        rank = int(row["Review_Rank"])
        key = f"ai_{rank}_{row['Predicted_Default_Risk']:.4f}"
        c = st.columns(widths)
        c[0].write(rank)
        c[1].write(row[display_id] if display_id else "-")
        c[2].write(f"{row['Predicted_Default_Risk']:.1%}")
        c[3].write(str(row["Risk_Band"]))
        c[4].write(row["Top_Risk_Reasons"])
        if c[5].button("🤖 Ask AI", key="btn_" + key):
            with st.spinner("Asking Groq..."):
                try:
                    st.session_state[key] = ai_reason(
                        f"{row['Predicted_Default_Risk']:.1%}",
                        row["Top_Risk_Reasons"],
                        build_facts(row, feature_cols),
                    )
                except Exception as e:
                    st.session_state[key] = f"⚠️ Groq request failed: {e}"
        if key in st.session_state:
            st.info(st.session_state[key])
@st.cache_data(show_spinner=False, ttl=3600)
def ai_reason(risk_pct, rule_reasons, facts):
    client = Groq(api_key=get_api_key())
    user_msg = (
        f"Predicted default risk: {risk_pct}\n"
        f"Rule-based reasons: {rule_reasons}\n"
        f"Customer data: {facts}\n"
        "Pay_0/Pay_2.. values: -1 = paid on time, 1+ = months of delay."
    )
    resp = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        temperature=0.2,
        max_tokens=1500,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise ValueError("Model returned empty text; increase max_tokens")
    return text
if __name__ == "__main__":
    main()
