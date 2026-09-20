"""랜덤포레스트 변수 선택이 bad F1을 개선하는지 검증한다.

변수 선택은 640명 안에서만 수행하고, 1차와 동일한 검증 160명은 마지막
비교에만 사용한다. 결과물은 같은 폴더의 결과보고서.md 하나뿐이다.
"""

from __future__ import annotations

from pathlib import Path
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*sklearn.utils.parallel.delayed.*",
    category=UserWarning,
)

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


SEED = 42
THRESHOLD = 0.50
PROJECT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
TRAIN_PATH = PROJECT / "분류_학습_데이터셋.csv"

NUMERIC = ["Age", "Credit amount", "Duration"]
RAW_FEATURES = [
    "Age", "Sex", "Job", "Housing", "Saving accounts", "Checking account",
    "Credit amount", "Duration", "Purpose",
]
MODEL_FEATURES = RAW_FEATURES + ["Account information"]


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """1차 모델처럼 계좌 결측을 Unknown으로 두고 계좌 보유 상태를 추가한다."""
    result = frame[RAW_FEATURES].copy()
    for col in NUMERIC:
        result[col] = pd.to_numeric(result[col], errors="raise")
    saving_present = result["Saving accounts"].notna()
    checking_present = result["Checking account"].notna()
    result["Account information"] = np.select(
        [
            saving_present & checking_present,
            saving_present & ~checking_present,
            ~saving_present & checking_present,
        ],
        ["both_present", "saving_only", "checking_only"],
        default="both_missing",
    )
    result[["Saving accounts", "Checking account"]] = result[
        ["Saving accounts", "Checking account"]
    ].fillna("Unknown")
    return result


def make_model(features: list[str]) -> Pipeline:
    numeric = [col for col in features if col in NUMERIC]
    categorical = [col for col in features if col not in NUMERIC]
    transformers = []
    if numeric:
        transformers.append(("numeric", "passthrough", numeric))
    if categorical:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            )
        )
    return Pipeline(
        [
            ("preprocess", ColumnTransformer(transformers)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=400,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    max_features="sqrt",
                    random_state=SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def predict_bad(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    bad_index = list(model.classes_).index(1)
    probability = model.predict_proba(frame)[:, bad_index]
    return (probability >= THRESHOLD).astype(int)


def bad_metrics(y_true, prediction: np.ndarray) -> dict[str, float]:
    return {
        "precision": precision_score(y_true, prediction, zero_division=0),
        "recall": recall_score(y_true, prediction, zero_division=0),
        "f1": f1_score(y_true, prediction, zero_division=0),
    }


def rank_by_permutation(
    X: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    repeats: int = 5,
    model_factory=make_model,
) -> pd.DataFrame:
    """칼럼 하나를 섞었을 때 bad F1 하락 폭으로 원본 칼럼 중요도를 잰다."""
    rng = np.random.default_rng(SEED)
    drops: dict[str, list[float]] = {feature: [] for feature in MODEL_FEATURES}
    for fold, (train_position, valid_position) in enumerate(splits, start=1):
        X_train = X.iloc[train_position]
        X_valid = X.iloc[valid_position]
        y_train = y.iloc[train_position]
        y_valid = y.iloc[valid_position]
        model = model_factory(MODEL_FEATURES).fit(X_train, y_train)
        baseline = f1_score(y_valid, predict_bad(model, X_valid), zero_division=0)
        for feature in MODEL_FEATURES:
            original = X_valid[feature].to_numpy(copy=True)
            for _ in range(repeats):
                shuffled = X_valid.copy()
                shuffled[feature] = rng.permutation(original)
                permuted_f1 = f1_score(
                    y_valid, predict_bad(model, shuffled), zero_division=0
                )
                drops[feature].append(baseline - permuted_f1)
        print(f"중요도 계산 {fold}/{len(splits)} 완료", flush=True)
    rows = []
    for feature, values in drops.items():
        values_array = np.asarray(values)
        rows.append(
            {
                "feature": feature,
                "importance_mean": float(values_array.mean()),
                "importance_std": float(values_array.std(ddof=1)),
                "positive_rate": float(np.mean(values_array > 0)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["importance_mean", "positive_rate"], ascending=False, ignore_index=True
    )


def evaluate_feature_counts(
    X: pd.DataFrame,
    y: pd.Series,
    ranking: list[str],
    splits: list[tuple[np.ndarray, np.ndarray]],
    model_factory=make_model,
) -> pd.DataFrame:
    rows = []
    for count in range(1, len(MODEL_FEATURES) + 1):
        features = ranking[:count]
        scores = []
        for train_position, valid_position in splits:
            model = model_factory(features).fit(
                X.iloc[train_position], y.iloc[train_position]
            )
            prediction = predict_bad(model, X.iloc[valid_position])
            scores.append(
                f1_score(y.iloc[valid_position], prediction, zero_division=0)
            )
        rows.append(
            {
                "n_features": count,
                "features": features,
                "mean_bad_f1": float(np.mean(scores)),
                "std_bad_f1": float(np.std(scores, ddof=1)),
            }
        )
        print(f"변수 개수 비교 {count}/{len(MODEL_FEATURES)} 완료", flush=True)
    return pd.DataFrame(rows)


def stratified_paired_bootstrap(
    truth: np.ndarray,
    baseline_prediction: np.ndarray,
    selected_prediction: np.ndarray,
    repeats: int = 5000,
) -> np.ndarray:
    """같은 160명에서 두 모델의 bad F1 차이 불확실성을 계산한다."""
    rng = np.random.default_rng(20260913)
    good_position = np.flatnonzero(truth == 0)
    bad_position = np.flatnonzero(truth == 1)
    differences = np.empty(repeats)
    for repeat in range(repeats):
        sample = np.concatenate(
            [
                rng.choice(good_position, size=len(good_position), replace=True),
                rng.choice(bad_position, size=len(bad_position), replace=True),
            ]
        )
        base_f1 = f1_score(
            truth[sample], baseline_prediction[sample], zero_division=0
        )
        selected_f1 = f1_score(
            truth[sample], selected_prediction[sample], zero_division=0
        )
        differences[repeat] = selected_f1 - base_f1
    return differences


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    separator = "|" + "|".join(["---"] + ["---:" for _ in headers[1:]]) + "|"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([head, separator, *body])


def main() -> None:
    data = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
    X = prepare_features(data)
    y = data["Risk"].eq("bad").astype(int)
    train_idx, valid_idx = train_test_split(
        np.arange(len(data)), test_size=0.20, random_state=SEED, stratify=y
    )
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
    inner_splits = list(inner.split(X_train, y_train))
    importance = rank_by_permutation(X_train, y_train, inner_splits)
    ranking = importance["feature"].tolist()
    candidates = evaluate_feature_counts(X_train, y_train, ranking, inner_splits)
    best = candidates.sort_values(
        ["mean_bad_f1", "n_features"], ascending=[False, True]
    ).iloc[0]
    selected_features = list(best["features"])

    baseline_model = make_model(MODEL_FEATURES).fit(X_train, y_train)
    selected_model = make_model(selected_features).fit(X_train, y_train)
    baseline_prediction = predict_bad(baseline_model, X_valid)
    selected_prediction = predict_bad(selected_model, X_valid)
    baseline = bad_metrics(y_valid, baseline_prediction)
    selected = bad_metrics(y_valid, selected_prediction)
    difference = selected["f1"] - baseline["f1"]

    bootstrap = stratified_paired_bootstrap(
        y_valid.to_numpy(), baseline_prediction, selected_prediction
    )
    ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975])
    improvement_probability = float(np.mean(bootstrap > 0))

    importance_rows = [
        [
            str(index + 1),
            row.feature,
            f"{row.importance_mean:.4f}",
            f"{row.importance_std:.4f}",
            f"{row.positive_rate * 100:.1f}%",
        ]
        for index, row in enumerate(importance.itertuples(index=False))
    ]
    importance_table = markdown_table(
        ["순위", "변수", "bad F1 평균 하락", "표준편차", "양의 중요도 비율"],
        importance_rows,
    )
    candidate_rows = [
        [
            str(int(row.n_features)),
            f"{row.mean_bad_f1:.4f}",
            f"{row.std_bad_f1:.4f}",
            ", ".join(row.features),
        ]
        for row in candidates.itertuples(index=False)
    ]
    candidate_table = markdown_table(
        ["변수 수", "내부 평균 bad F1", "표준편차", "포함 변수"],
        candidate_rows,
    )

    old_reported_f1 = 0.5531914893617021
    reproduced = np.isclose(baseline["f1"], old_reported_f1)
    if difference > 0 and ci_low > 0:
        conclusion = "변수 선택의 개선 효과가 이번 검증셋에서 뚜렷했다."
    elif difference > 0:
        conclusion = (
            "점수는 개선됐지만 95% 부트스트랩 구간이 0을 포함해 "
            "개선이 확실하다고 단정하기는 어렵다."
        )
    elif difference == 0:
        conclusion = "변수 선택으로 bad F1이 달라지지 않았다."
    else:
        conclusion = "변수 선택 모델의 bad F1이 낮아져 기존 전체 변수 모델이 더 좋았다."

    report = f"""# 랜덤포레스트 변수 중요도·선택 실험

## 결론

1차와 동일한 검증 160명에서 기존 전체 변수 모델의 bad F1은 **{baseline['f1']:.4f}**, 변수 선택 모델은 **{selected['f1']:.4f}**였다. 차이는 **{difference:+.4f}**이다. {conclusion}

bad F1 차이의 층화 부트스트랩 95% 구간은 **[{ci_low:+.4f}, {ci_high:+.4f}]**, 선택 모델의 점수가 더 높게 재표집된 비율은 **{improvement_probability * 100:.1f}%**였다.

## 비교 조건

- 학습 640명에서만 변수 중요도 순위와 사용할 변수 개수를 결정했다.
- 1차와 같은 검증 160명은 마지막 성능 비교에만 사용했다.
- 모델은 1차와 같은 랜덤포레스트 400개 트리, min_samples_leaf=2, class_weight=balanced_subsample, max_features=sqrt를 사용했다.
- bad 판정 임계값도 1차와 같은 0.50으로 고정했다.
- 결측 계좌값은 Unknown이라는 독립 범주로 유지했다.
- 기존 1차 보고 점수 0.5532의 재현 여부: **{'일치' if reproduced else '불일치'}**.

## 검증 160명 최종 비교

| 방법 | bad 정밀도 | bad 재현율 | bad F1 | 사용 변수 |
|---|---:|---:|---:|---|
| 기존 전체 변수 | {baseline['precision']:.4f} | {baseline['recall']:.4f} | {baseline['f1']:.4f} | {len(MODEL_FEATURES)}개 |
| 변수 선택 | {selected['precision']:.4f} | {selected['recall']:.4f} | {selected['f1']:.4f} | {len(selected_features)}개: {', '.join(selected_features)} |

## 변수 중요도 순위

각 변수를 검증 데이터에서 여러 번 섞었을 때 bad F1이 얼마나 떨어졌는지 계산했다. 평균 하락이 클수록 해당 변수의 정보가 예측에 중요했다는 뜻이다. 양의 중요도 비율은 반복 실험 중 그 변수를 섞었을 때 실제로 점수가 하락한 비율이다.

{importance_table}

0 이하의 중요도는 현재 랜덤포레스트에서 안정적으로 도움이 됐다고 보기 어렵다. 순위는 절대적인 신용평가 기준이 아니라 이 데이터와 모델 안에서의 예측 기여도다.

## 변수 개수 비교

중요도 상위 1개부터 10개까지 640명 내부의 4겹 교차검증으로 비교했다. 이 표에서 가장 높은 평균 bad F1을 낸 구성을 고른 뒤 검증 160명에 한 번만 적용했다.

{candidate_table}

## 선택 결과 해석

선택된 변수는 **{', '.join(selected_features)}**이다. 이 변수들이 다른 변수보다 우선해서 볼 후보라는 뜻이며, 각 변수의 구체적인 어떤 값이 bad 또는 good 방향으로 작용했는지는 다음 값 수준 분석에서 별도로 확인해야 한다.
"""
    (OUT / "결과보고서.md").write_text(report, encoding="utf-8")

    print("\n최종 결과")
    print(f"기존 전체 변수 bad F1: {baseline['f1']:.4f}")
    print(f"변수 선택 bad F1:     {selected['f1']:.4f}")
    print(f"차이:                  {difference:+.4f}")
    print(f"95% 부트스트랩 구간:   [{ci_low:+.4f}, {ci_high:+.4f}]")
    print(f"선택 변수: {', '.join(selected_features)}")
    print(f"보고서: {OUT / '결과보고서.md'}")


if __name__ == "__main__":
    main()
