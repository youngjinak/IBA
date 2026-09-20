"""최적화된 3차 스태킹 모델을 학습하고 정답 입력 데이터의 Risk를 채운다.

최종 구성
- Random Forest + CatBoost의 bad 확률을 입력으로 사용
- 균형 로지스틱 회귀 메타 모델로 스태킹
- CatBoost l2_leaf_reg=13
- bad 판정 임계값=0.50

실행 결과는 같은 폴더의 ``정답_입력데이터_3차예측.csv``에 저장한다.
원본 정답 입력 데이터의 Risk 이외 칼럼은 변경하지 않는다.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[1]
SECOND_VERSION = PROJECT / "단일모델버전" / "2차version"
sys.path.insert(0, str(SECOND_VERSION))

from logistic_regression import (  # noqa: E402
    CATEGORICAL,
    SEED,
    fresh_model,
    learn_rule,
    load_training,
    make_reference_forest,
    prepare_features,
)


ANSWER_PATH = PROJECT / "정답 입력 데이터.csv"
SUBMISSION_PATH = OUT / "정답_입력데이터_3차예측.csv"
THRESHOLD = 0.50
CATBOOST_L2 = 13


def make_catboost() -> CatBoostClassifier:
    """세부 탐색과 반복 교차검증에서 선택한 CatBoost를 만든다."""
    return CatBoostClassifier(
        iterations=500,
        depth=5,
        learning_rate=0.05,
        l2_leaf_reg=CATBOOST_L2,
        random_strength=1,
        loss_function="Logloss",
        auto_class_weights="Balanced",
        cat_features=CATEGORICAL,
        random_seed=SEED,
        thread_count=2,
        verbose=False,
        allow_writing_files=False,
    )


def make_meta_model():
    """Random Forest와 CatBoost 확률을 결합하는 균형 스태킹 모델."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1,
            max_iter=2000,
            class_weight="balanced",
            random_state=SEED,
        ),
    )


def bad_probability(model, frame: pd.DataFrame) -> np.ndarray:
    bad_index = list(model.classes_).index(1)
    return model.predict_proba(frame)[:, bad_index]


def train_final_model():
    """800명의 OOF 확률로 메타 모델을 학습하고 기본 모델을 전체 재학습한다."""
    X, y, source_sha256 = load_training()
    oof = np.zeros((len(y), 2), dtype=float)
    visited = np.zeros(len(y), dtype=int)

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for train_position, valid_position in splitter.split(X, y):
        X_train, y_train = X.iloc[train_position], y.iloc[train_position]
        X_valid = X.iloc[valid_position]

        forest = fresh_model(make_reference_forest()).fit(X_train, y_train)
        catboost = make_catboost().fit(X_train, y_train)
        oof[valid_position, 0] = bad_probability(forest, X_valid)
        oof[valid_position, 1] = bad_probability(catboost, X_valid)
        visited[valid_position] += 1

    if not np.all(visited == 1) or not np.isfinite(oof).all():
        raise RuntimeError("OOF 확률 생성에 실패했습니다.")

    meta = make_meta_model().fit(oof, y)
    forest = fresh_model(make_reference_forest()).fit(X, y)
    catboost = make_catboost().fit(X, y)
    return {
        "forest": forest,
        "catboost": catboost,
        "meta": meta,
        "rule": learn_rule(X, y),
        "source_sha256": source_sha256,
    }


def predict_labels(fitted: dict, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    base_probability = np.column_stack(
        [
            bad_probability(fitted["forest"], frame),
            bad_probability(fitted["catboost"], frame),
        ]
    )
    bad_scores = fitted["meta"].predict_proba(base_probability)[:, 1]
    predicted_bad = bad_scores >= THRESHOLD

    rule = fitted["rule"]
    forced_good = (
        (frame["Duration"] < rule["duration"])
        | (frame["Credit amount"] < rule["amount"])
    ).to_numpy()
    predicted_bad[forced_good] = False
    return np.where(predicted_bad, "bad", "good"), bad_scores


def write_submission(labels: np.ndarray) -> None:
    """원본 문자열을 유지하면서 Risk 값만 바꾼다."""
    with ANSWER_PATH.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        original_rows = list(reader)

    if "Risk" not in fields:
        raise ValueError("정답 입력 데이터에 Risk 칼럼이 없습니다.")
    if len(original_rows) != len(labels):
        raise ValueError("정답 입력 데이터 행 수와 예측 수가 다릅니다.")

    changed_rows = []
    for original, label in zip(original_rows, labels):
        changed = dict(original)
        changed["Risk"] = str(label)
        if any(changed[name] != original[name] for name in fields if name != "Risk"):
            raise RuntimeError("Risk 이외의 값이 변경되었습니다.")
        changed_rows.append(changed)

    with SUBMISSION_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(changed_rows)


def main() -> None:
    answer = pd.read_csv(ANSWER_PATH, encoding="utf-8-sig", keep_default_na=False)
    X_answer = prepare_features(answer)
    fitted = train_final_model()
    labels, scores = predict_labels(fitted, X_answer)
    write_submission(labels)

    bad_count = int(np.sum(labels == "bad"))
    good_count = int(np.sum(labels == "good"))
    print("3차 최종 모델 학습 및 예측 완료")
    print(f"CatBoost L2: {CATBOOST_L2} / 임계값: {THRESHOLD:.2f}")
    print(f"예측: good {good_count}명 / bad {bad_count}명")
    print(f"Bad 점수 범위: {scores.min():.4f}~{scores.max():.4f}")
    print(f"CSV: {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
