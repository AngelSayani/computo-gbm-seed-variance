"""
Seed-variance study of gradient-boosted tree pipelines on the AI4I 2020
predictive-maintenance (industrial sensor fault detection) dataset.

Decomposes seed variance into:
  (a) data-split seed only (model internal seed fixed)
  (b) model-internal seed only (data split fixed)
  (c) both varied jointly (the "naive single run" scenario)

Across two GBM implementations (sklearn HistGradientBoostingClassifier,
sklearn GradientBoostingClassifier) and two imbalance-handling settings
(none vs class-weighting), on the real, imbalanced (3.39% positive) binary
"Machine failure" target.

All randomness is explicit and seeded; no results are hand-edited.
"""
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.metrics import average_precision_score, f1_score, brier_score_loss, roc_curve
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

DATA_PATH = "/home/claude/computo_gbm/data/clean/ai4i2020.csv"
OUT_PATH = "/home/claude/computo_gbm/results/raw_runs.jsonl"

N_SEEDS = 60          # seeds per scenario per model per imbalance setting
FIXED_SEED = 0        # the "other" seed held fixed in single-source scenarios
TEST_SIZE = 0.3
TARGET_FPR = 0.01      # recall-at-1%-FPR, a realistic alarm-budget operating point

FEATURES_NUM = ["Air temperature [K]", "Process temperature [K]",
                "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
FEATURES_CAT = ["Type"]
TARGET = "Machine failure"


def load_data():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    X = df[FEATURES_NUM + FEATURES_CAT].copy()
    y = df[TARGET].astype(int).values
    return X, y


def build_pipeline(model_name, split_balance, model_seed):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
    ], remainder="passthrough")

    if model_name == "hist_gbm":
        clf = HistGradientBoostingClassifier(
            random_state=model_seed,
            class_weight=("balanced" if split_balance == "balanced" else None),
            max_iter=200,
        )
    elif model_name == "grad_boost":
        # GradientBoostingClassifier has no class_weight; emulate via sample_weight at fit time
        clf = GradientBoostingClassifier(
            random_state=model_seed,
            subsample=0.8,
            n_estimators=200,
        )
    else:
        raise ValueError(model_name)

    return Pipeline([("pre", pre), ("clf", clf)])


def recall_at_fpr(y_true, scores, target_fpr):
    fpr, tpr, thr = roc_curve(y_true, scores)
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    idx = max(idx, 0)
    return float(tpr[idx])


def fit_eval(model_name, split_balance, split_seed, model_seed, X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=split_seed, stratify=y
    )
    pipe = build_pipeline(model_name, split_balance, model_seed)

    if model_name == "grad_boost" and split_balance == "balanced":
        # sample_weight-based class balancing for the implementation that lacks class_weight
        pos_rate = y_tr.mean()
        w_pos = 0.5 / pos_rate
        w_neg = 0.5 / (1 - pos_rate)
        sw = np.where(y_tr == 1, w_pos, w_neg)
        pipe.fit(X_tr, y_tr, clf__sample_weight=sw)
    else:
        pipe.fit(X_tr, y_tr)

    scores = pipe.predict_proba(X_te)[:, 1]
    preds = (scores >= 0.5).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_te, scores)),
        "f1_at_0.5": float(f1_score(y_te, preds, zero_division=0)),
        "recall_at_1pct_fpr": recall_at_fpr(y_te, scores, TARGET_FPR),
        "brier": float(brier_score_loss(y_te, scores)),
        "n_test_pos": int(y_te.sum()),
    }


def main():
    X, y = load_data()
    print(f"Loaded {len(X)} rows, {y.sum()} positives ({y.mean()*100:.2f}%)")

    rng = np.random.RandomState(12345)
    seeds = rng.randint(0, 1_000_000, size=N_SEEDS)

    scenarios = ["split_only", "model_only", "both"]
    models = ["hist_gbm", "grad_boost"]
    balances = ["none", "balanced"]

    records = []
    t0 = time.time()
    total = len(scenarios) * len(models) * len(balances) * N_SEEDS
    done = 0
    for model_name in models:
        for bal in balances:
            for scenario in scenarios:
                for i, s in enumerate(seeds):
                    if scenario == "split_only":
                        split_seed, model_seed = int(s), FIXED_SEED
                    elif scenario == "model_only":
                        split_seed, model_seed = FIXED_SEED, int(s)
                    else:  # both
                        # use two independent seed streams so "both" isn't
                        # accidentally correlated with the single-source runs
                        split_seed, model_seed = int(s), int(seeds[(i + 1) % N_SEEDS])

                    metrics = fit_eval(model_name, bal, split_seed, model_seed, X, y)
                    rec = {
                        "model": model_name, "balance": bal, "scenario": scenario,
                        "run_idx": i, "split_seed": split_seed, "model_seed": model_seed,
                        **metrics,
                    }
                    records.append(rec)
                    done += 1
                    if done % 100 == 0:
                        elapsed = time.time() - t0
                        print(f"{done}/{total} runs, {elapsed:.0f}s elapsed")

    with open(OUT_PATH, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(records)} runs to {OUT_PATH} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
