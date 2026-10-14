"""Supplementary condition: HistGradientBoostingClassifier with early_stopping
forced True, isolating random_state's effect on the internal validation split.
Same scenarios/balances/seeds as the main sweep, written to a separate file."""
import json, time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import run_experiment as R

OUT_PATH = "/home/claude/computo_gbm/results/raw_runs_es.jsonl"

def build_pipeline_es(balance, model_seed):
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), R.FEATURES_CAT)],
                             remainder="passthrough")
    clf = HistGradientBoostingClassifier(
        random_state=model_seed,
        class_weight=("balanced" if balance == "balanced" else None),
        max_iter=200,
        early_stopping=True,
    )
    return Pipeline([("pre", pre), ("clf", clf)])

def fit_eval_es(balance, split_seed, model_seed, X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=R.TEST_SIZE,
                                               random_state=split_seed, stratify=y)
    pipe = build_pipeline_es(balance, model_seed)
    pipe.fit(X_tr, y_tr)
    scores = pipe.predict_proba(X_te)[:, 1]
    preds = (scores >= 0.5).astype(int)
    from sklearn.metrics import average_precision_score, f1_score, brier_score_loss
    return {
        "pr_auc": float(average_precision_score(y_te, scores)),
        "f1_at_0.5": float(f1_score(y_te, preds, zero_division=0)),
        "recall_at_1pct_fpr": R.recall_at_fpr(y_te, scores, R.TARGET_FPR),
        "brier": float(brier_score_loss(y_te, scores)),
        "n_test_pos": int(y_te.sum()),
    }

def main():
    X, y = R.load_data()
    rng = np.random.RandomState(12345)
    seeds = rng.randint(0, 1_000_000, size=R.N_SEEDS)
    scenarios = ["split_only", "model_only", "both"]
    balances = ["none", "balanced"]
    records = []
    t0 = time.time()
    total = len(scenarios) * len(balances) * R.N_SEEDS
    done = 0
    for bal in balances:
        for scenario in scenarios:
            for i, s in enumerate(seeds):
                if scenario == "split_only":
                    split_seed, model_seed = int(s), R.FIXED_SEED
                elif scenario == "model_only":
                    split_seed, model_seed = R.FIXED_SEED, int(s)
                else:
                    split_seed, model_seed = int(s), int(seeds[(i + 1) % R.N_SEEDS])
                metrics = fit_eval_es(bal, split_seed, model_seed, X, y)
                rec = {"model": "hist_gbm_earlystop", "balance": bal, "scenario": scenario,
                       "run_idx": i, "split_seed": split_seed, "model_seed": model_seed, **metrics}
                records.append(rec)
                done += 1
                if done % 100 == 0:
                    print(f"{done}/{total} runs, {time.time()-t0:.0f}s elapsed")
    with open(OUT_PATH, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(records)} runs in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
