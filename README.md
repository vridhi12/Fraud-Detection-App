# Real-Time Credit Card Fraud Detection

**Vridhi Gautam | Pratinik Infotech Machine Learning Internship (Project 2: Fraud Detection in Credit Card Transactions)**

A gradient-boosted-tree model that scores a card transaction in a few milliseconds, explains the decision in plain English, and is served through a Streamlit app.

## Results (held-out test set: 555,719 transactions, 2,145 frauds, all later in time than the training data)

| Metric | Value |
|---|---|
| Precision | 91.8% |
| Recall | 90.9% |
| F1-score | 91.3% |
| PR-AUC | 0.962 |
| Scoring time per transaction | about 3 ms |
| Fraud value caught | about 95% of the total fraud amount |

The alert threshold was chosen on a validation period inside the training data. The test file was used once.

## Dataset
[Credit Card Transactions Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection) (Kaggle). It is **simulated** data (Sparkov generator), so results show that the pipeline works but do not guarantee the same accuracy on real bank data. The CSVs are too large for this repository, so download `fraudTrain.csv` and `fraudTest.csv` yourself.

## Files
- `train.py`: feature engineering, training, threshold selection, evaluation, plots
- `app.py`: Streamlit app (check a transaction, dataset, model performance, monitoring and impact)
- `fraud_model.pkl`, `metrics.json`, `sample_transactions.csv`, `*.png`: outputs used by the app

## Run
```
pip install -r requirements.txt
streamlit run app.py
```
To retrain: `python train.py --data_dir path/to/csv_folder --out_dir .` (about 4 minutes).
If the app reports a scikit-learn version error, retrain with `train.py` using the same scikit-learn version you deploy with.

## Method
- Split by time: the model never sees the future.
- Engineered card-behaviour features that use only a card's *past* transactions: amount compared with the card's usual spending, time since previous transaction, activity earlier the same day, first time at a merchant or category, distance and speed from the previous transaction.
- Class imbalance (about 0.6% fraud) handled by choosing the alert threshold on validation data instead of 0.5.
- Local explanation: each group of factors is replaced by a typical value, and the change in fraud score is shown.

## Limitations
- Simulated data. Real transactions have more noise.
- The last test month (December 2020) shows precision dropping to about 85%, so the model would need monitoring and retraining.
