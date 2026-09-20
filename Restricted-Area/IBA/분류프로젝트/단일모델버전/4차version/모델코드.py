"""5개 모델의 모델별 변수 선택을 비교하고 4차 Risk 예측 CSV를 만든다."""

from __future__ import annotations

import csv
from pathlib import Path
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*sklearn.utils.parallel.delayed.*",
    category=UserWarning,
)

import numpy as np
import pandas as pd

import 랜덤포레스트_변수중요도실험 as rf
import 로지스틱회귀_변수중요도실험 as logistic
import CatBoost_변수중요도실험 as catboost_model
import ExtraTrees_변수중요도실험 as extra_trees
import SVM_변수중요도실험 as svm


OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[1]
ANSWER_PATH = PROJECT / "정답 입력 데이터.csv"
SUBMISSION_PATH = OUT / "정답_입력데이터_4차예측.csv"
REPORT_PATH = OUT / "결과보고서.md"

MODEL_FACTORIES = [
    ("Random Forest", rf.make_model),
    ("Logistic Regression", logistic.make_model),
    ("CatBoost", catboost_model.make_model),
    ("Extra Trees", extra_trees.make_model),
    ("SVM", svm.make_model),
]


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    separator = "|" + "|".join(["---"] + ["---:" for _ in headers[1:]]) + "|"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([head, separator, *body])


def read_original(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_submission(labels: np.ndarray) -> tuple[int, int]:
    fields, rows = read_original(ANSWER_PATH)
    if len(rows) != len(labels):
        raise ValueError("정답 입력 데이터 행 수와 예측 수가 다릅니다.")
    if "Risk" not in fields:
        raise ValueError("정답 입력 데이터에 Risk 칼럼이 없습니다.")
    output = []
    for original, label in zip(rows, labels):
        changed = dict(original)
        changed["Risk"] = str(label)
        output.append(changed)
        assert all(
            changed[field] == original[field] for field in fields if field != "Risk"
        )
    with SUBMISSION_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    bad_n = int(np.sum(labels == "bad"))
    return bad_n, len(labels) - bad_n


def build_report(results: list[dict], winner: dict, bad_n: int, good_n: int) -> str:
    summary_rows = []
    for result in results:
        low, high = result["bootstrap_ci"]
        summary_rows.append([
            result["name"],
            f"{result['baseline']['f1']:.4f}",
            f"{result['selected']['f1']:.4f}",
            f"{result['difference']:+.4f}",
            f"{result['selected']['precision']:.4f}",
            f"{result['selected']['recall']:.4f}",
            f"[{low:+.4f}, {high:+.4f}]",
            ", ".join(result["selected_features"]),
        ])
    summary = markdown_table(
        [
            "모델", "전체 변수 bad F1", "선택 변수 bad F1", "차이",
            "선택 정밀도", "선택 재현율", "차이 95% 구간", "선택 변수",
        ],
        summary_rows,
    )

    sections = []
    for result in results:
        importance_rows = []
        for rank, row in enumerate(result["importance"].itertuples(index=False), 1):
            importance_rows.append([
                str(rank),
                row.feature,
                f"{row.importance_mean:.4f}",
                f"{row.importance_std:.4f}",
                f"{row.positive_rate * 100:.1f}%",
            ])
        table = markdown_table(
            ["순위", "변수", "bad F1 평균 하락", "표준편차", "양의 중요도 비율"],
            importance_rows,
        )
        best_inner = result["candidates"].sort_values(
            ["mean_bad_f1", "n_features"], ascending=[False, True]
        ).iloc[0]
        sections.append(
            f"## {result['name']} 중요도\n\n"
            f"640명 내부 선택 점수: **{best_inner['mean_bad_f1']:.4f}**, "
            f"선택 변수 {int(best_inner['n_features'])}개.\n\n{table}"
        )

    return f"""# 4차version · 모델별 변수 중요도와 Risk 예측

## 결과

각 모델이 자체적으로 변수 중요도 순위와 사용할 변수 개수를 선택하도록 했다. 공통 검증 160명에서 bad F1이 가장 높은 구성은 **{winner['name']}**이며, 선택 변수는 **{', '.join(winner['selected_features'])}**이다. 검증 bad F1은 **{winner['selected']['f1']:.4f}**, bad 정밀도는 **{winner['selected']['precision']:.4f}**, bad 재현율은 **{winner['selected']['recall']:.4f}**이다.

{summary}

## 평가 방법

- 원본 학습 데이터 800명을 1차와 동일하게 학습 640명과 검증 160명으로 계층 분할했다.
- 각 모델은 640명 내부의 4겹 교차검증에서 순열 중요도를 계산했다.
- 중요도 상위 1개부터 10개까지 모두 비교해 내부 평균 bad F1이 가장 높은 변수 수를 선택했다.
- 검증 160명은 중요도 순위나 변수 개수 선택에 사용하지 않고 마지막 비교에만 사용했다.
- 계좌 결측은 모든 모델에서 Unknown이라는 별도 범주로 유지했다.
- bad 판정 임계값은 모든 모델에서 0.50으로 고정했다.
- 모델 파라미터는 1·2차에서 사용한 설정을 유지했다.
- 차이 95% 구간은 같은 검증 160명을 층화 재표집한 결과다. 구간에 0이 포함되면 선택 변수의 개선이 확실하다고 보지 않는다.
- 다섯 모델 중 최고 모델을 고르는 데 검증 160명을 사용했으므로, 최고 점수 자체에는 모델 선택에 따른 낙관 편향이 있을 수 있다.

## 최종 정답 입력 데이터

검증 bad F1이 가장 높았던 **{winner['name']}**의 선택 변수 구성을 전체 800명으로 다시 학습했다. 정답 입력 데이터 200명에는 기존 칼럼을 유지하면서 Risk 값만 채웠다.

- 예측 good: **{good_n}명**
- 예측 bad: **{bad_n}명**
- 파일: 정답_입력데이터_4차예측.csv

{(chr(10) * 2).join(sections)}

## 해석

순열 중요도의 bad F1 평균 하락이 클수록 해당 변수를 섞었을 때 모델 성능이 많이 나빠졌다는 뜻이다. 모델마다 학습 방식이 달라 중요도 순위도 달라질 수 있다. Account information은 저축·당좌계좌의 결측 여부에서 만든 변수이므로 Saving accounts 및 Checking account와 정보가 일부 겹친다.
"""


def main() -> None:
    results = []
    for name, factory in MODEL_FACTORIES:
        result = logistic.run_experiment(name, factory, progress=True)
        results.append(result)

    winner = max(
        results,
        key=lambda result: (
            result["selected"]["f1"],
            result["selected"]["precision"],
        ),
    )

    train = pd.read_csv(rf.TRAIN_PATH, encoding="utf-8-sig")
    answer = pd.read_csv(ANSWER_PATH, encoding="utf-8-sig")
    X_train = rf.prepare_features(train)
    X_answer = rf.prepare_features(answer)
    y_train = train["Risk"].eq("bad").astype(int)
    final_model = winner["model_factory"](winner["selected_features"]).fit(
        X_train, y_train
    )
    prediction = rf.predict_bad(final_model, X_answer)
    labels = np.where(prediction == 1, "bad", "good")
    bad_n, good_n = write_submission(labels)

    REPORT_PATH.write_text(
        build_report(results, winner, bad_n, good_n), encoding="utf-8"
    )
    print("\n4차 최종 선택", flush=True)
    print(
        f"{winner['name']}: bad F1 {winner['selected']['f1']:.4f} / "
        f"변수 {', '.join(winner['selected_features'])}",
        flush=True,
    )
    print(f"정답 입력 예측: good {good_n}명 / bad {bad_n}명", flush=True)
    print(f"CSV: {SUBMISSION_PATH}", flush=True)
    print(f"보고서: {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
