"""2차 CatBoost. 같은 폴더의 logistic_regression.py에서 공통 평가 코드를 가져온다.

python -B catboost_model.py
필요 패키지: numpy, pandas, scikit-learn, catboost
catboost_info, 모델 파일, CSV, 이미지 등 추가 파일은 생성하지 않는다.
"""
import sys
sys.dont_write_bytecode = True
import json
import catboost
from catboost import CatBoostClassifier
from logistic_regression import CATEGORICAL, SEED, THRESHOLD, load_training, evaluate, train_full, predict

def make_catboost():
    return CatBoostClassifier(iterations=500, depth=5, learning_rate=0.05,
        l2_leaf_reg=5, loss_function='Logloss', auto_class_weights='Balanced',
        cat_features=CATEGORICAL, random_seed=SEED, thread_count=2,
        verbose=False, allow_writing_files=False)

def run():
    X, y, digest = load_training()
    result = {'source_sha256': digest, 'n': len(y), 'bad_n': int(y.sum()),
        'seed': SEED, 'threshold': THRESHOLD, 'catboost_version': catboost.__version__,
        'catboost': evaluate(make_catboost(), X, y)}
    fitted = train_full(make_catboost(), X, y)
    predictions = predict(fitted['model'], X.iloc[:3], fitted['rule'])[0]
    assert set(predictions).issubset({0, 1})
    result['full_model_trained'] = True
    return result, fitted

if __name__ == '__main__':
    results, trained_model = run()
    print(json.dumps(results, ensure_ascii=False, indent=2))
