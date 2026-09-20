"""RBF SVM의 변수 중요도 순위와 선택 효과를 계산한다."""

from __future__ import annotations

import json
import warnings

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

import 랜덤포레스트_변수중요도실험 as core
from 로지스틱회귀_변수중요도실험 import run_experiment


warnings.filterwarnings(
    "ignore", message="The .probability. parameter was deprecated.*", category=FutureWarning
)


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
        ("classifier", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=core.SEED,
        )),
    ])


def main() -> None:
    result = run_experiment("SVM", make_model)
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
