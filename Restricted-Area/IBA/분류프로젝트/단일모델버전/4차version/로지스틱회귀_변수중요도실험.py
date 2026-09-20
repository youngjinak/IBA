"""로지스틱 회귀의 변수 중요도 순위와 선택 효과를 계산한다."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import 랜덤포레스트_변수중요도실험 as core


def make_model(features: list[str]) -> Pipeline:
    numeric = [col for col in features if col in core.NUMERIC]
    categorical = [col for col in features if col not in core.NUMERIC]
    transformers = []
    if numeric:
        transformers.append(("numeric", StandardScaler(), numeric))
    if categorical:
        transformers.append((
            "categorical",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical,
        ))
    return Pipeline([
        ("preprocess", ColumnTransformer(transformers)),
        ("classifier", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=3000,
            random_state=core.SEED,
        )),
    ])


def run_experiment(name: str, model_factory, progress: bool = True) -> dict:
    data = pd.read_csv(core.TRAIN_PATH, encoding="utf-8-sig")
    X = core.prepare_features(data)
    y = data["Risk"].eq("bad").astype(int)
    train_idx, valid_idx = train_test_split(
        np.arange(len(data)),
        test_size=0.20,
        random_state=core.SEED,
        stratify=y,
    )
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=core.SEED)
    splits = list(inner.split(X_train, y_train))

    if progress:
        print(f"\n[{name}] 변수 중요도 계산", flush=True)
    importance = core.rank_by_permutation(
        X_train, y_train, splits, repeats=5, model_factory=model_factory
    )
    if progress:
        print(f"[{name}] 변수 개수 비교", flush=True)
    candidates = core.evaluate_feature_counts(
        X_train, y_train, importance["feature"].tolist(), splits,
        model_factory=model_factory,
    )
    best = candidates.sort_values(
        ["mean_bad_f1", "n_features"], ascending=[False, True]
    ).iloc[0]
    selected_features = list(best["features"])

    baseline_model = model_factory(core.MODEL_FEATURES).fit(X_train, y_train)
    selected_model = model_factory(selected_features).fit(X_train, y_train)
    baseline_prediction = core.predict_bad(baseline_model, X_valid)
    selected_prediction = core.predict_bad(selected_model, X_valid)
    baseline = core.bad_metrics(y_valid, baseline_prediction)
    selected = core.bad_metrics(y_valid, selected_prediction)
    bootstrap = core.stratified_paired_bootstrap(
        y_valid.to_numpy(), baseline_prediction, selected_prediction
    )
    result = {
        "name": name,
        "baseline": baseline,
        "selected": selected,
        "difference": selected["f1"] - baseline["f1"],
        "selected_features": selected_features,
        "importance": importance,
        "candidates": candidates,
        "bootstrap_ci": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "improvement_probability": float(np.mean(bootstrap > 0)),
        "model_factory": model_factory,
    }
    if progress:
        print(
            f"[{name}] 기존 {baseline['f1']:.4f} / 선택 {selected['f1']:.4f} / "
            f"차이 {result['difference']:+.4f}",
            flush=True,
        )
    return result


def main() -> None:
    result = run_experiment("Logistic Regression", make_model)
    printable = {
        "model": result["name"],
        "baseline_bad_f1": result["baseline"]["f1"],
        "selected_bad_f1": result["selected"]["f1"],
        "difference": result["difference"],
        "selected_features": result["selected_features"],
        "importance": result["importance"].to_dict(orient="records"),
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
