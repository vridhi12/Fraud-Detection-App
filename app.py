import json
import os
import time

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")
HERE = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_bundle():
    return joblib.load(os.path.join(HERE, "fraud_model.pkl"))


@st.cache_data
def load_samples():
    return pd.read_csv(os.path.join(HERE, "sample_transactions.csv"))


@st.cache_data
def load_metrics():
    with open(os.path.join(HERE, "metrics.json")) as f:
        return json.load(f)


bundle = load_bundle()
model, THRESHOLD, FEATURES, BASELINE = bundle["model"], bundle["threshold"], bundle["features"], bundle["baseline"]
samples = load_samples()
M = load_metrics()
T = M["test"]

# plain-English factors: each group of columns is "made typical" one at a time to see how much it mattered
GROUPS = {
    "Transaction amount": ["amt", "log_amt"],
    "Amount compared with this card's usual spending": ["amt_ratio", "amt_z", "past_mean_amt"],
    "Time of day": ["hour", "night"],
    "Day of week": ["dayofweek"],
    "Merchant category": [c for c in FEATURES if c.startswith("cat_")],
    "Distance from cardholder's home to merchant": ["distance_km"],
    "Card activity earlier the same day": ["n_today_before", "amt_today_before"],
    "Time since the card's previous transaction": ["secs_since_prev"],
    "Distance / speed from the previous transaction": ["dist_from_prev_km", "speed_kmh"],
    "First time at this merchant or category": ["new_merchant", "new_category"],
    "Cardholder profile": ["age", "gender_m", "city_pop", "past_count"],
}


def explain(row):
    """Local explanation: how much would the fraud probability change if this factor looked typical?"""
    base_p = float(model.predict_proba(row)[0, 1])
    variants = []
    for cols in GROUPS.values():
        v = row.copy()
        for c in cols:
            v[c] = BASELINE[c]
        variants.append(v)
    p_typ = model.predict_proba(pd.concat(variants, ignore_index=True))[:, 1]
    return base_p, pd.Series(base_p - p_typ, index=list(GROUPS.keys()))


st.title("💳 Real-Time Credit Card Fraud Detection")
st.caption("Vridhi Gautam | Pratinik Infotech Machine Learning Internship | Gradient Boosted Trees + Streamlit")

tab_live, tab_data, tab_perf, tab_mon = st.tabs(
    ["🔍 Check a transaction", "📁 Dataset", "📊 Model performance", "🛡️ Monitoring & impact"])

# ---------------------------------------------------------------- live check
with tab_live:
    st.write(
        "Pick a real transaction from the **test set** (July to December 2020, after the training period). "
        "The model scores it in milliseconds and explains what drove the decision."
    )
    labels = [f"Transaction #{i + 1}: {r.category.replace('_', ' ')}, amount {r.amt:,.2f}, hour {int(r.hour):02d}:00"
              for i, r in samples.iterrows()]
    choice = st.selectbox("Choose a transaction", labels)
    row = samples.iloc[[labels.index(choice)]]

    if st.button("Run fraud analysis", type="primary"):
        x = row[FEATURES]
        start = time.perf_counter()
        p_fraud = float(model.predict_proba(x)[0, 1])
        latency_ms = (time.perf_counter() - start) * 1000
        flagged = p_fraud >= THRESHOLD

        st.subheader("1. Model verdict")
        if flagged:
            st.error(f"🚨 **Flagged as FRAUD**: fraud score {p_fraud * 100:.1f}% (alert threshold {THRESHOLD * 100:.1f}%).")
        else:
            st.success(f"✅ **Looks LEGITIMATE**: fraud score {p_fraud * 100:.1f}% (alert threshold {THRESHOLD * 100:.1f}%).")
        st.caption(f"⚡ Scored in {latency_ms:.1f} ms")

        actual = int(row["y"].iloc[0])
        ok = flagged == bool(actual)
        st.write(f"Actual label in the dataset: **{'Fraud' if actual else 'Legitimate'}**, so the model was "
                 f"**{'correct ✔' if ok else 'wrong ✘'}** on this transaction.")

        st.subheader("2. Why did the model decide this?")
        _, delta = explain(x)
        top = delta.reindex(delta.abs().sort_values(ascending=False).index).head(6)[::-1]
        fig, ax = plt.subplots(figsize=(8, 3.4))
        ax.barh(top.index, top.values * 100, color=["#dc2626" if v > 0 else "#16a34a" for v in top.values])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Change in fraud score if this factor looked typical (percentage points)")
        ax.set_title("Top 6 factors for THIS transaction")
        fig.tight_layout()
        st.pyplot(fig)
        st.caption("Red: this factor pushed the score toward fraud (a typical value would have lowered it). "
                   "Green: it pushed toward legitimate.")

# -------------------------------------------------------------------- dataset
with tab_data:
    d = M["dataset"]
    st.subheader("About the dataset")
    st.write(
        "**Credit Card Transactions Fraud Detection Dataset** from Kaggle: a **simulated** set of card "
        "transactions (generated with the Sparkov simulator) with cardholder, merchant, location and time details. "
        "Because it is simulated, results here show how the pipeline works and do not guarantee the same "
        "accuracy on real bank data."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training transactions", f"{d['train_rows']:,}")
    c2.metric("Training frauds", f"{d['train_frauds']:,}")
    c3.metric("Test transactions", f"{d['test_rows']:,}")
    c4.metric("Test frauds", f"{d['test_frauds']:,}")
    st.markdown(
        f"- **Training period:** {d['train_period'][0][:10]} to {d['train_period'][1][:10]}\n"
        f"- **Test period (never used for training or tuning):** {d['test_period'][0][:10]} to {d['test_period'][1][:10]}\n"
        f"- **Validation:** the last part of the training period ({d['val_rows']:,} transactions, "
        f"{d['val_frauds']:,} frauds) was used only to choose the alert threshold\n"
        f"- **Class imbalance:** only about {100 * d['train_frauds'] / d['train_rows']:.2f}% of transactions are fraud"
    )
    st.markdown(
        "**Features used by the model** ({} in total)\n"
        "- Transaction: amount, hour, day of week, merchant category\n"
        "- Cardholder: age, gender, city population\n"
        "- Location: distance from home to merchant\n"
        "- **Engineered card-behaviour features** (calculated from the card's *past* transactions only): "
        "amount compared with the card's usual spending, time since the previous transaction, "
        "activity earlier the same day, first time at a merchant or category, and distance / speed "
        "from the previous transaction".format(d["n_features"])
    )
    st.markdown("**Preview: 10 real test transactions bundled with this app**")
    st.dataframe(samples[["category", "amt", "hour", "distance_km", "amt_ratio", "secs_since_prev", "y"]].head(10)
                 .rename(columns={"y": "is_fraud"}), hide_index=True)

# ------------------------------------------------------------ model performance
with tab_perf:
    d = M["dataset"]
    st.write(f"Evaluated once on **{d['test_rows']:,} unseen test transactions** ({d['test_frauds']:,} frauds), "
             f"all later in time than the training data.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision (fraud)", f"{T['precision'] * 100:.1f}%")
    c2.metric("Recall (fraud)", f"{T['recall'] * 100:.1f}%")
    c3.metric("F1-score", f"{T['f1'] * 100:.1f}%")
    c4.metric("PR-AUC", f"{M['pr_auc']:.3f}")
    st.write(f"The model caught **{T['tp']:,} of {T['tp'] + T['fn']:,}** frauds and wrongly flagged "
             f"**{T['fp']:,}** of {T['tn'] + T['fp']:,} legitimate transactions.")
    ca, cb = st.columns(2)
    ca.image(os.path.join(HERE, "confusion_matrix.png"))
    cb.image(os.path.join(HERE, "pr_curve.png"))
    st.image(os.path.join(HERE, "feature_importance.png"))
    st.info(
        "**How it was built:** features use only past information; the data is split by time (train on the past, "
        "test on the future); the alert threshold was chosen on a validation period inside the training data, "
        "so the test numbers above are honest."
    )

# ------------------------------------------------------------ monitoring & impact
with tab_mon:
    L, mo = M["latency"], M["money"]
    st.subheader("Speed")
    st.write(f"Scoring one transaction takes about **{L['median_ms']:.1f} ms** (95th percentile "
             f"{L['p95_ms']:.1f} ms), measured on the training machine. That is fast enough for real-time use.")

    st.subheader("Financial impact (test period)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Fraud value in test period", f"{mo['fraud_value_total']:,.0f}")
    c2.metric("Fraud value caught", f"{mo['fraud_value_caught']:,.0f}", f"{mo['pct_fraud_value_caught']:.1f}% of total")
    c3.metric("Fraud value missed", f"{mo['fraud_value_missed']:,.0f}")
    st.caption(f"Cost of false alarms: {T['fp']:,} legitimate transactions "
               f"(total value {mo['legit_value_wrongly_flagged']:,.0f}) were flagged for review.")

    st.subheader("Continuous monitoring")
    st.write("The frozen model and threshold were checked month by month on later data, "
             "the same check that would run in production:")
    st.image(os.path.join(HERE, "monthly_performance.png"))
    st.dataframe(pd.DataFrame(M["monthly"]).assign(
        precision=lambda x: (x.precision * 100).round(1), recall=lambda x: (x.recall * 100).round(1)
    ).rename(columns={"n": "transactions"}), hide_index=True)
    st.markdown(
        "**Improvement plan**\n"
        "- Track precision and recall every month once confirmed fraud labels arrive.\n"
        "- Retrain when either metric falls below the 90% target, or on a fixed schedule.\n"
        "- Re-tune the alert threshold on the newest validation data each time."
    )
