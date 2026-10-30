"""
Real-Time Credit Card Fraud Detection - training pipeline
Pratinik Infotech Machine Learning Internship | Vridhi Gautam

Dataset : "Credit Card Transactions Fraud Detection Dataset" (Kaggle, simulated with Sparkov)
          needs fraudTrain.csv and fraudTest.csv
Usage   : python train.py --data_dir path/to/folder_with_csvs --out_dir .

Outputs : fraud_model.pkl, metrics.json, sample_transactions.csv,
          confusion_matrix.png, pr_curve.png, monthly_performance.png, feature_importance.png
"""
import argparse, json, os, time
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, ConfusionMatrixDisplay)

CATS = ['entertainment', 'food_dining', 'gas_transport', 'grocery_net', 'grocery_pos', 'health_fitness',
        'home', 'kids_pets', 'misc_net', 'misc_pos', 'personal_care', 'shopping_net', 'shopping_pos', 'travel']

PARAMS = dict(max_iter=400, learning_rate=0.1, max_depth=8, max_leaf_nodes=63,
              l2_regularization=1.0, random_state=42)


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def load(path, split):
    cols = ['trans_date_trans_time', 'cc_num', 'merchant', 'category', 'amt', 'gender', 'lat', 'long',
            'city_pop', 'dob', 'merch_lat', 'merch_long', 'is_fraud']
    df = pd.read_csv(path, usecols=cols)
    df['split'] = split
    return df


def build_features(df):
    """Every behaviour feature uses only PAST transactions of the same card (no label, no future data)."""
    df = df.copy()
    df['time'] = pd.to_datetime(df['trans_date_trans_time'])
    df = df.sort_values(['cc_num', 'time']).reset_index(drop=True)
    hour = df['time'].dt.hour

    out = pd.DataFrame(index=df.index)
    out['amt'] = df['amt']
    out['log_amt'] = np.log1p(df['amt'])
    out['hour'] = hour
    out['dayofweek'] = df['time'].dt.dayofweek
    out['night'] = ((hour >= 22) | (hour <= 3)).astype(int)
    out['age'] = (df['time'] - pd.to_datetime(df['dob'])).dt.days / 365.25
    out['gender_m'] = (df['gender'] == 'M').astype(int)
    out['city_pop'] = df['city_pop']
    out['distance_km'] = haversine(df['lat'], df['long'], df['merch_lat'], df['merch_long'])
    for c in CATS:
        out['cat_' + c] = (df['category'] == c).astype(int)

    g = df.groupby('cc_num')
    out['secs_since_prev'] = g['time'].diff().dt.total_seconds().fillna(1e7)
    past_mean = g['amt'].transform(lambda s: s.shift(1).expanding().mean())
    past_std = g['amt'].transform(lambda s: s.shift(1).expanding().std())
    out['past_mean_amt'] = past_mean.fillna(df['amt'].median())
    out['amt_ratio'] = df['amt'] / (out['past_mean_amt'] + 1)
    out['amt_z'] = ((df['amt'] - out['past_mean_amt']) / (past_std.fillna(1) + 1)).clip(-20, 20)
    out['past_count'] = g.cumcount()
    df['date'] = df['time'].dt.date
    gd = df.groupby(['cc_num', 'date'])
    out['n_today_before'] = gd.cumcount()
    out['amt_today_before'] = gd['amt'].cumsum() - df['amt']
    out['new_merchant'] = (df.groupby(['cc_num', 'merchant']).cumcount() == 0).astype(int)
    out['new_category'] = (df.groupby(['cc_num', 'category']).cumcount() == 0).astype(int)
    prev_lat, prev_lon = g['merch_lat'].shift(1), g['merch_long'].shift(1)
    d_prev = haversine(prev_lat, prev_lon, df['merch_lat'], df['merch_long']).fillna(0)
    out['dist_from_prev_km'] = d_prev
    out['speed_kmh'] = (d_prev / (out['secs_since_prev'] / 3600 + 1e-3)).clip(0, 5000)

    out['category'] = df['category']
    out['time'] = df['time']
    out['split'] = df['split']
    out['y'] = df['is_fraud']
    return out.sort_values('time').reset_index(drop=True)


def main(data_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    raw = pd.concat([load(os.path.join(data_dir, 'fraudTrain.csv'), 0),
                     load(os.path.join(data_dir, 'fraudTest.csv'), 1)], ignore_index=True)
    d = build_features(raw)
    print('features built: %s  (%.0fs)' % (d.shape, time.time() - t0), flush=True)

    FEATS = [c for c in d.columns if c not in ('category', 'time', 'split', 'y')]
    train, test = d[d.split == 0], d[d.split == 1].reset_index(drop=True)

    # time-based split INSIDE the training file: fit on the earlier part, pick the threshold on the later part
    fit_mask = train.time < '2020-03-01'
    fit, val = train[fit_mask], train[~fit_mask]
    model = HistGradientBoostingClassifier(**PARAMS).fit(fit[FEATS], fit['y'])

    pv = model.predict_proba(val[FEATS])[:, 1]
    p, r, th = precision_recall_curve(val['y'], pv)
    thr = float(th[np.minimum(p[:-1], r[:-1]).argmax()])      # best balance of precision and recall on validation

    # the test file is touched once
    pt = model.predict_proba(test[FEATS])[:, 1]
    pred = (pt >= thr).astype(int)
    y = test['y'].values
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    prec, rec, f1 = precision_score(y, pred), recall_score(y, pred), f1_score(y, pred)
    print('TEST precision %.4f recall %.4f f1 %.4f' % (prec, rec, f1), flush=True)

    # ---- money view
    amt = test['amt'].values
    loss = dict(fraud_value_total=float(amt[y == 1].sum()), fraud_value_caught=float(amt[(y == 1) & (pred == 1)].sum()),
                fraud_value_missed=float(amt[(y == 1) & (pred == 0)].sum()),
                legit_value_wrongly_flagged=float(amt[(y == 0) & (pred == 1)].sum()))
    loss['pct_fraud_value_caught'] = 100 * loss['fraud_value_caught'] / loss['fraud_value_total']

    # ---- monitoring: same frozen model + threshold, month by month
    test['pred'] = pred
    monthly = []
    for per, g in test.groupby(test['time'].dt.to_period('M')):
        if g['y'].sum() == 0:
            continue
        monthly.append(dict(month=str(per), n=int(len(g)), frauds=int(g['y'].sum()),
                            precision=float(precision_score(g['y'], g['pred'], zero_division=0)),
                            recall=float(recall_score(g['y'], g['pred']))))

    # ---- latency of scoring ONE transaction
    one = test[FEATS].iloc[[0]]
    for _ in range(20):
        model.predict_proba(one)
    ts = []
    for i in range(300):
        s = time.perf_counter(); model.predict_proba(test[FEATS].iloc[[i]]); ts.append((time.perf_counter() - s) * 1000)
    latency = dict(median_ms=float(np.median(ts)), p95_ms=float(np.percentile(ts, 95)))
    print('latency', latency, flush=True)

    # ---- feature importance (permutation, on a random 80k test sample)
    sub = test.sample(80000, random_state=1)
    pi = permutation_importance(model, sub[FEATS], sub['y'], scoring='average_precision',
                                n_repeats=2, random_state=1, n_jobs=1)
    imp = pd.Series(pi.importances_mean, index=FEATS).sort_values(ascending=False)

    # ---- plots
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix(y, pred), display_labels=['Legitimate', 'Fraud']).plot(
        ax=ax, cmap='Blues', values_format=',d')
    ax.set_title('Confusion matrix (test set)'); fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'confusion_matrix.png'), dpi=150); plt.close(fig)

    pp, rr, _ = precision_recall_curve(y, pt)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rr, pp, lw=2); ax.scatter([rec], [prec], color='red', zorder=3, label='chosen threshold')
    ax.axhline(0.9, ls='--', c='grey', lw=.8); ax.axvline(0.9, ls='--', c='grey', lw=.8)
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); ax.set_ylim(0, 1.02)
    ax.set_title('Precision-Recall curve (PR-AUC = %.3f)' % average_precision_score(y, pt)); ax.legend(loc='lower left')
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'pr_curve.png'), dpi=150); plt.close(fig)

    mdf = pd.DataFrame(monthly)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(mdf['month'], mdf['precision'] * 100, marker='o', label='Precision')
    ax.plot(mdf['month'], mdf['recall'] * 100, marker='s', label='Recall')
    ax.axhline(90, ls='--', c='grey', lw=.8); ax.set_ylim(60, 100); ax.set_ylabel('%')
    ax.set_title('Monitoring: frozen model, month by month'); ax.legend(); plt.xticks(rotation=30)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'monthly_performance.png'), dpi=150); plt.close(fig)

    top = imp.head(10)[::-1]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.barh(top.index, top.values, color='#2563eb'); ax.set_xlabel('Drop in PR-AUC when the feature is shuffled')
    ax.set_title('Top 10 features (permutation importance)'); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'feature_importance.png'), dpi=150); plt.close(fig)

    # ---- sample rows for the demo app (real test transactions)
    rng = np.random.RandomState(7)
    f_idx = rng.choice(test.index[test.y == 1], 15, replace=False)
    l_idx = rng.choice(test.index[test.y == 0], 25, replace=False)
    samp = test.loc[np.concatenate([f_idx, l_idx])].sample(frac=1, random_state=3)
    samp[FEATS + ['category', 'y']].to_csv(os.path.join(out_dir, 'sample_transactions.csv'), index=False)

    baseline = train.loc[train.y == 0, FEATS].median().to_dict()
    joblib.dump(dict(model=model, threshold=thr, features=FEATS, baseline=baseline),
                os.path.join(out_dir, 'fraud_model.pkl'), compress=3)

    metrics = dict(
        dataset=dict(name='Credit Card Transactions Fraud Detection Dataset (Kaggle, simulated)',
                     train_rows=int((d.split == 0).sum()), train_frauds=int(train.y.sum()),
                     train_period=[str(train.time.min()), str(train.time.max())],
                     test_rows=int(len(test)), test_frauds=int(y.sum()),
                     test_period=[str(test.time.min()), str(test.time.max())],
                     fit_rows=int(len(fit)), val_rows=int(len(val)), val_frauds=int(val.y.sum()),
                     n_features=len(FEATS)),
        model=dict(name='Gradient Boosted Trees (sklearn HistGradientBoostingClassifier)', **PARAMS),
        threshold=thr,
        validation=dict(precision=float(p[np.minimum(p[:-1], r[:-1]).argmax()]), recall=float(r[np.minimum(p[:-1], r[:-1]).argmax()])),
        test=dict(precision=float(prec), recall=float(rec), f1=float(f1), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn)),
        pr_auc=float(average_precision_score(y, pt)), roc_auc=float(roc_auc_score(y, pt)),
        money=loss, monthly=monthly, latency=latency,
        top_features=[[k, float(v)] for k, v in imp.head(10).items()],
    )
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print('done in %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='data')
    ap.add_argument('--out_dir', default='.')
    a = ap.parse_args()
    main(a.data_dir, a.out_dir)
