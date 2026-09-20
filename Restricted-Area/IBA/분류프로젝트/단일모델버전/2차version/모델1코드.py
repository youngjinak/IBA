"""2차 단일 모델 실험의 공통 실행 진입점.

사용법:
    python 모델코드.py                  # 네 모델 모두 실행
    python 모델코드.py --model catboost # 한 모델만 실행

모델 구현과 공통 전처리는 같은 폴더의 개별 Python 파일에 있다.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


MODELS = {
    "logistic": "logistic_regression.py",
    "catboost": "catboost_model.py",
    "extra_trees": "extra_trees.py",
    "svm": "svm_model.py",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["all", *MODELS], default="all")
    args = parser.parse_args()

    folder = Path(__file__).resolve().parent
    selected = MODELS if args.model == "all" else {args.model: MODELS[args.model]}
    for label, filename in selected.items():
        print(f"\n=== {label} ===", flush=True)
        subprocess.run([sys.executable, "-B", str(folder / filename)], check=True)


if __name__ == "__main__":
    main()
