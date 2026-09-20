"""CatBoost의 변수 중요도 순위와 선택 효과를 계산한다."""

from __future__ import annotations

import json

from catboost import CatBoostClassifier

import 랜덤포레스트_변수중요도실험 as core
from 로지스틱회귀_변수중요도실험 import run_experiment


class CatBoostSubsetModel:
    """CatBoost에 현재 후보 칼럼만 전달하는 작은 래퍼다."""

    def __init__(self, features: list[str]):
        self.features = list(features)
        categorical = [col for col in self.features if col not in core.NUMERIC]
        self.model = CatBoostClassifier(
            iterations=500,
            depth=5,
            learning_rate=0.05,
            l2_leaf_reg=5,
            loss_function="Logloss",
            auto_class_weights="Balanced",
            cat_features=categorical,
            random_seed=core.SEED,
            thread_count=2,
            verbose=False,
            allow_writing_files=False,
        )

    def fit(self, X, y):
        self.model.fit(X[self.features], y)
        self.classes_ = self.model.classes_
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X[self.features])


def make_model(features: list[str]) -> CatBoostSubsetModel:
    return CatBoostSubsetModel(features)


def main() -> None:
    result = run_experiment("CatBoost", make_model)
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
