"""Extra Trees의 변수 중요도 순위와 선택 효과를 계산한다."""

from __future__ import annotations

import json

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import 랜덤포레스트_변수중요도실험 as core
from 로지스틱회귀_변수중요도실험 import run_experiment


def make_model(features: list[str]) -> Pipeline:
    numeric = [col for col in features if col in core.NUMERIC]
    categorical = [col for col in features if col not in core.NUMERIC]
    transformers = []
    if numeric:
        transformers.append(("numeric", "passthrough", numeric))
    if categorical:
        transformers.append((
            "categorical",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical,
        ))
    return Pipeline([
        ("preprocess", ColumnTransformer(transformers)),
        ("classifier", ExtraTreesClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=core.SEED,
            n_jobs=-1,
        )),
    ])


def main() -> None:
    result = run_experiment("Extra Trees", make_model)
    print(json.dumps({
        "model": result["name"],
        "baseline_bad_f1": result["baseline"]["f1"],
        "selected_bad_f1": result["selected"]["f1"],
        "difference": result["difference"],
        "selected_features": result["selected_features"],
        "importance": result["importance"].to_dict(orient="records"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
