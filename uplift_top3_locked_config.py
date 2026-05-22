import gc
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor


RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

DATA_DIR = Path("кей")
TRAIN_PATH = DATA_DIR / "train.parquet"
TEST_PATH = DATA_DIR / "test.parquet"

ID_COL = "user_id"
TARGET_COL = "rec_spend"
TREATMENT_COL = "treatment_flg"

CATBOOST_THREAD_COUNT = int(os.environ.get("CATBOOST_THREAD_COUNT", max(1, (os.cpu_count() or 2) // 2)))
FINAL_SEEDS = [
    int(seed.strip())
    for seed in os.environ.get("FINAL_SEEDS", "42").split(",")
    if seed.strip()
]

# Config from top20_selected_configs.csv that produced the strong leaderboard run.
LOCKED_CONFIG = {
    "iterations": 450,
    "learning_rate": 0.025,
    "depth": 6,
    "l2_leaf_reg": 55.0,
    "random_strength": 6.0,
    "rsm": 0.75,
    "bagging_temperature": 7.0,
    "min_data_in_leaf": 50,
    "border_count": 254,
}


def add_engineered_features(df, base_missing_cols=None):
    out = df.copy()
    ratio_specs = [
        ("mark_view_per_offer", "cus_mark_n_view", "cus_mark_n_offers"),
        ("mark_view_per_rule", "cus_mark_n_view", "cus_mark_n_rule"),
        ("offers_per_rule", "cus_mark_n_offers", "cus_mark_n_rule"),
        ("rto_per_trn", "rto", "n_trn"),
        ("days_per_trn", "n_days_last_visit", "n_trn"),
        ("sku_per_trn", "n_sku", "n_trn"),
        ("cat7_per_sku", "n_cat_7", "n_sku"),
    ]
    for new_col, num, den in ratio_specs:
        if num in out.columns and den in out.columns:
            out[new_col] = out[num].astype(float) / (out[den].astype(float).abs() + 1.0)

    if base_missing_cols is not None:
        for col in base_missing_cols:
            if col in out.columns:
                out[f"{col}__isna"] = out[col].isna().astype("int8")
    return out


def make_pruned_features(train_df, test_df, features, threshold=0.985, sample_n=180_000):
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(train_df[c])]
    if not numeric:
        return features, []

    both = pd.concat([train_df[numeric], test_df[numeric]], axis=0)
    if len(both) > sample_n:
        both = both.sample(sample_n, random_state=RANDOM_STATE)

    corr = both.corr(method="spearman").abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    null_rate = both.isna().mean()
    variance = both.var(numeric_only=True).fillna(0)

    drop = set()
    for col in upper.columns:
        partners = upper.index[upper[col] > threshold].tolist()
        for partner in partners:
            if col in drop or partner in drop:
                continue
            score_col = (1.0 - null_rate.get(col, 1.0), variance.get(col, 0.0))
            score_partner = (1.0 - null_rate.get(partner, 1.0), variance.get(partner, 0.0))
            if score_col >= score_partner:
                drop.add(partner)
            else:
                drop.add(col)

    return [c for c in features if c not in drop], sorted(drop)


def categorical_features(df, features):
    cats = []
    for col in features:
        if col not in df.columns:
            continue
        if (
            pd.api.types.is_object_dtype(df[col])
            or pd.api.types.is_string_dtype(df[col])
            or pd.api.types.is_categorical_dtype(df[col])
            or pd.api.types.is_bool_dtype(df[col])
        ):
            cats.append(col)
    return cats


def prepare_x(df, features):
    x = df[features].copy()
    for col in categorical_features(x, features):
        x[col] = x[col].astype("string").fillna("__MISSING__")
    return x


def common_params(seed):
    return {
        "iterations": int(LOCKED_CONFIG["iterations"]),
        "learning_rate": float(LOCKED_CONFIG["learning_rate"]),
        "depth": int(LOCKED_CONFIG["depth"]),
        "l2_leaf_reg": float(LOCKED_CONFIG["l2_leaf_reg"]),
        "random_strength": float(LOCKED_CONFIG["random_strength"]),
        "rsm": float(LOCKED_CONFIG["rsm"]),
        "bagging_temperature": float(LOCKED_CONFIG["bagging_temperature"]),
        "bootstrap_type": "Bayesian",
        "min_data_in_leaf": int(LOCKED_CONFIG["min_data_in_leaf"]),
        "border_count": int(LOCKED_CONFIG["border_count"]),
        "allow_writing_files": False,
        "random_seed": int(seed),
        "thread_count": CATBOOST_THREAD_COUNT,
        "verbose": False,
    }


def fit_hurdle_t_learner(train_df, features, seed):
    x_all = prepare_x(train_df, features)
    cat_features = categorical_features(x_all, features)
    models = {}

    for arm in [0, 1]:
        mask = train_df[TREATMENT_COL].to_numpy() == arm
        arm_df = train_df.loc[mask]
        x_arm = x_all.loc[mask]
        y_nonzero = arm_df[TARGET_COL].gt(0).astype(int).to_numpy()

        clf_params = common_params(seed + 11 + arm)
        clf_params.update({"loss_function": "Logloss", "eval_metric": "AUC"})
        clf = CatBoostClassifier(**clf_params)
        clf.fit(x_arm, y_nonzero, cat_features=cat_features)

        pos_mask = mask & train_df[TARGET_COL].gt(0).to_numpy()
        x_pos = x_all.loc[pos_mask]
        y_pos = np.log1p(train_df.loc[pos_mask, TARGET_COL].to_numpy(dtype=float))

        reg_params = common_params(seed + 101 + arm)
        reg_params.update({"loss_function": "RMSE", "eval_metric": "RMSE"})
        reg = CatBoostRegressor(**reg_params)
        reg.fit(x_pos, y_pos, cat_features=cat_features)

        models[arm] = {"clf": clf, "reg": reg}
        gc.collect()

    return models


def predict_hurdle_t_learner(models, df, features, amount_cap):
    x = prepare_x(df, features)
    expected = {}
    for arm in [0, 1]:
        p = models[arm]["clf"].predict_proba(x)[:, 1]
        amount = np.expm1(models[arm]["reg"].predict(x))
        amount = np.clip(amount, 0, amount_cap)
        expected[arm] = p * amount
    return expected[1] - expected[0]


def rank_average(scores_list):
    out = np.zeros_like(np.asarray(scores_list[0], dtype=float))
    weight = 1.0 / len(scores_list)
    for scores in scores_list:
        out += weight * pd.Series(scores).rank(method="average").to_numpy()
    return out


def save_submission(test_df, scores, path):
    sub = pd.DataFrame({ID_COL: test_df[ID_COL].to_numpy(), "UPLIFT_SCORE": np.asarray(scores, dtype=float)})
    assert len(sub) == len(test_df)
    assert sub[ID_COL].equals(test_df[ID_COL].reset_index(drop=True))
    assert sub["UPLIFT_SCORE"].notna().all()
    sub.to_csv(path, index=False)
    print(f"saved {path}: {sub.shape}")
    return sub


def main():
    print("locked config:", LOCKED_CONFIG)
    print("FINAL_SEEDS:", FINAL_SEEDS)
    print("CATBOOST_THREAD_COUNT:", CATBOOST_THREAD_COUNT)

    train_raw = pd.read_parquet(TRAIN_PATH)
    test_raw = pd.read_parquet(TEST_PATH)

    base_feature_cols = [c for c in test_raw.columns if c != ID_COL]
    miss_rate = pd.concat([train_raw[base_feature_cols], test_raw[base_feature_cols]], axis=0).isna().mean()
    missing_flag_cols = miss_rate[(miss_rate >= 0.03) & (miss_rate < 0.98)].sort_values(ascending=False).head(35).index.tolist()

    train = add_engineered_features(train_raw, missing_flag_cols)
    test = add_engineered_features(test_raw, missing_flag_cols)

    full_features = [c for c in test.columns if c != ID_COL]
    features, dropped_cols = make_pruned_features(train, test, full_features)
    amount_cap = np.nanpercentile(train[TARGET_COL].to_numpy(dtype=float), 99.9)

    print("train:", train.shape, "test:", test.shape)
    print("full_features:", len(full_features), "pruned_features:", len(features), "dropped:", len(dropped_cols))
    print("dropped cols head:", dropped_cols[:30])

    seed_scores = []
    for seed in FINAL_SEEDS:
        print("fit seed", seed)
        models = fit_hurdle_t_learner(train, features, seed=seed)
        scores = predict_hurdle_t_learner(models, test, features, amount_cap)
        seed_scores.append(scores)
        save_submission(test, scores, f"predictions_top3_locked_seed{seed}.csv")
        del models
        gc.collect()

    if len(seed_scores) == 1:
        main_scores = seed_scores[0]
    else:
        main_scores = rank_average(seed_scores)
        save_submission(test, main_scores, "predictions_top3_locked_rank_ensemble.csv")

    save_submission(test, main_scores, "predictions.csv")
    print("main submission -> predictions.csv")


if __name__ == "__main__":
    main()
