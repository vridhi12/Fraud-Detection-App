import json
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import xgboost as xgb

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")

HERE = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_model():
    return joblib.load(os.path.join(HERE, "fraud_model.pkl"))


@st.cache_data
def load_samples():
    return pd.read_csv(os.path.join(HERE, "sample_transactions.csv"))


@st.cache_data
def load_metrics():
    with open(os.path.join(HERE, "metrics.json")) as f:
        return json.load(f)


model = load_model()
samples = load_samples()
metrics = load_metrics()
THRESHOLD = metrics["threshold"]
FEATURES = ["Scaled_Amount", "Scaled_Time"] + [f"V{i}" for i in range(1, 29)]
FRIENDLY = {"Scaled_Amount": "Transaction amount", "Scaled_Time": "Time of transaction"}


def friendly(name):
    return FRIENDLY.get(name, f"Anonymised feature {name}")


st.title("💳 Real-Time Credit Card Fraud Detection")
st.caption("Vridhi Gautam | Pratinik Infotech Machine Learning Internship | XGBoost + Streamlit")

tab_live, tab_perf = st.tabs(["🔍 Check a transaction", "📊 Model performance"])

# ---------------------------------------------------------------- live check
with tab_live:
    st.write(
        "Pick a real transaction from the **test set** (data the model never saw during training). "
        "The model scores it, and the chart shows which features pushed the decision."
    )

    labels = [f"Transaction #{i + 1}  (amount: {row.Amount:,.2f})" for i, row in samples.iterrows()]
    choice = st.selectbox("Choose a transaction", labels)
    idx = labels.index(choice)
    row = samples.iloc[[idx]]

    if st.button("Run fraud analysis", type="primary"):
        x = row[FEATURES]
        p_fraud = float(model.predict_proba(x)[0, 1])
        flagged = p_fraud >= THRESHOLD

        st.subheader("1. Model verdict")
        if flagged:
            st.error(f"🚨 **Flagged as FRAUD**: fraud probability {p_fraud * 100:.1f}% "
                     f"(alert threshold {THRESHOLD * 100:.0f}%).")
        else:
            st.success(f"✅ **Looks LEGITIMATE**: fraud probability {p_fraud * 100:.1f}% "
                       f"(alert threshold {THRESHOLD * 100:.0f}%).")

        actual = int(row["Class"].iloc[0])
        correct = flagged == bool(actual)
        st.write(f"Actual label in the dataset: **{'Fraud' if actual else 'Legitimate'}** → "
                 f"model was **{'correct ✔' if correct else 'wrong ✘'}** on this transaction.")

        st.subheader("2. Why did the model decide this?")
        booster = model.get_booster()
        contrib = booster.predict(xgb.DMatrix(x), pred_contribs=True)[0][:-1]  # last value is the bias
        contrib = pd.Series(contrib, index=FEATURES)
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(6)[::-1]

        fig, ax = plt.subplots(figsize=(8, 3.2))
        colors = ["#dc2626" if v > 0 else "#16a34a" for v in top.values]
        ax.barh([friendly(n) for n in top.index], top.values, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Push toward fraud (→)  /  toward legitimate (←)")
        ax.set_title("Top 6 factors for THIS transaction")
        fig.tight_layout()
        st.pyplot(fig)
        st.caption(
            "Red bars push the score toward fraud, green bars toward legitimate. V1 to V28 are anonymised "
            "(PCA-transformed) columns in the public dataset, so their real-world meaning is hidden by design."
        )

# ------------------------------------------------------------ model performance
with tab_perf:
    d, t = metrics["dataset"], metrics["tuned"]
    st.write(
        f"Evaluated on **{d['test_rows']:,} unseen test transactions** "
        f"({d['test_frauds']} of them fraud). The dataset has {d['rows']:,} transactions in total, "
        f"of which only {d['fraud_pct']:.3f}% are fraud."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision (fraud)", f"{t['precision'] * 100:.1f}%")
    c2.metric("Recall (fraud)", f"{t['recall'] * 100:.1f}%")
    c3.metric("F1-score", f"{t['f1'] * 100:.1f}%")
    c4.metric("PR-AUC", f"{metrics['pr_auc']:.3f}")
    st.write(
        f"At the chosen alert threshold the model caught **{t['tp']} of {t['tp'] + t['fn']}** frauds "
        f"and wrongly flagged **{t['fp']}** legitimate transactions out of {t['tn'] + t['fp']:,}."
    )

    col_a, col_b = st.columns(2)
    for col, fname in [(col_a, "confusion_matrix.png"), (col_b, "pr_curve.png")]:
        path = os.path.join(HERE, fname)
        if os.path.exists(path):
            col.image(path)

    st.info(
        "**How it was built:** the data was split 80/20 before scaling (no leakage); class imbalance is "
        "handled with `scale_pos_weight` in XGBoost; the alert threshold was tuned with cross-validation on "
        "the training data only, then the test set was used once for the numbers above."
    )
