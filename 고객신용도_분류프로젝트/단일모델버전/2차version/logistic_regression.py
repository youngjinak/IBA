"""2차 로지스틱 회귀. 결과는 콘솔에만 출력하며 파일을 생성하지 않는다.

python -B logistic_regression.py
python -B logistic_regression.py --with-rf  # 같은 분할의 1차 모델 비교 포함
필요 패키지: numpy, pandas, scikit-learn
"""
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

PROJECT = Path(__file__).resolve().parents[2]
SEED = 42
THRESHOLD = 0.5
NUMERIC = ['Age', 'Credit amount', 'Duration']
FEATURES = ['Age', 'Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account',
            'Credit amount', 'Duration', 'Purpose']
CATEGORICAL = [c for c in FEATURES if c not in NUMERIC] + ['Account information']

def prepare_features(data):
    result = data[FEATURES].copy()
    for name in NUMERIC:
        result[name] = pd.to_numeric(result[name], errors='raise')
    saving = result['Saving accounts'].ne('')
    checking = result['Checking account'].ne('')
    result['Account information'] = np.select(
        [saving & checking, saving, checking],
        ['both_present', 'saving_only', 'checking_only'], default='both_missing')
    for name in ['Saving accounts', 'Checking account']:
        result[name] = result[name].replace('', 'Unknown')
    for name in CATEGORICAL:
        result[name] = result[name].astype(str)
    return result

def load_training():
    path = PROJECT / '분류_학습_데이터셋.csv'
    data = pd.read_csv(path, keep_default_na=False)
    assert len(data) == 800 and data['Risk'].isin(['good', 'bad']).all()
    return prepare_features(data), data['Risk'].eq('bad').astype(int), hashlib.sha256(path.read_bytes()).hexdigest()

def make_logistic():
    preprocessor = ColumnTransformer([
        ('numeric', StandardScaler(), NUMERIC),
        ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL)])
    return Pipeline([('preprocess', preprocessor),
        ('model', LogisticRegression(C=1.0, class_weight='balanced', max_iter=3000, random_state=SEED))])

def make_reference_forest():
    """1차와 같은 설정을 동일한 새 교차검증 분할에서 재평가한다."""
    preprocessor = ColumnTransformer([
        ('numeric', 'passthrough', NUMERIC),
        ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL)])
    return Pipeline([('preprocess', preprocessor), ('model', RandomForestClassifier(
        n_estimators=400, min_samples_leaf=2, class_weight='balanced_subsample',
        max_features='sqrt', random_state=SEED, n_jobs=2))])

def learn_rule(X, y):
    bad = X.loc[y.eq(1)]
    return {'duration': int(bad['Duration'].min()), 'amount': int(bad['Credit amount'].min())}

def predict(model, X, rule):
    scores = model.predict_proba(X)[:, list(model.classes_).index(1)]
    raw = (scores >= THRESHOLD).astype(int)
    forced = ((X['Duration'] < rule['duration']) | (X['Credit amount'] < rule['amount'])).to_numpy()
    return np.where(forced, 0, raw), raw, scores, forced

def metrics(y, predicted):
    return {'macro_f1': float(f1_score(y, predicted, average='macro', zero_division=0)),
        'bad_f1': float(f1_score(y, predicted, zero_division=0)),
        'good_f1': float(f1_score(y, predicted, pos_label=0, zero_division=0)),
        'bad_precision': float(precision_score(y, predicted, zero_division=0)),
        'bad_recall': float(recall_score(y, predicted, zero_division=0)),
        'accuracy': float(accuracy_score(y, predicted)),
        'confusion_good_bad': confusion_matrix(y, predicted, labels=[0, 1]).tolist()}

def fresh_model(estimator):
    # CatBoost의 범주 목록 복사 방식과 sklearn clone의 버전별 호환성 처리.
    if type(estimator).__module__.startswith('catboost'):
        return type(estimator)(**estimator.get_params())
    return clone(estimator)

def evaluate(estimator, X, y):
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.full(len(y), -1, dtype=int)
    raw_oof = np.full(len(y), -1, dtype=int)
    visited = np.zeros(len(y), dtype=int)
    fold_results = []
    for fold, (tr, va) in enumerate(splitter.split(X, y), 1):
        model = fresh_model(estimator).fit(X.iloc[tr], y.iloc[tr])
        rule = learn_rule(X.iloc[tr], y.iloc[tr])
        predicted, raw, scores, forced = predict(model, X.iloc[va], rule)
        oof[va], raw_oof[va] = predicted, raw
        visited[va] += 1
        fold_results.append({'fold': fold, **metrics(y.iloc[va], predicted), 'rule': rule,
            'rule_hits': int(forced.sum()), 'rule_actual_bad': int(y.iloc[va].to_numpy()[forced].sum()),
            'rule_changed': int((predicted != raw).sum())})
    assert (visited == 1).all() and np.isin(oof, [0, 1]).all()
    values = [r['macro_f1'] for r in fold_results]
    return {'oof': metrics(y, oof), 'raw_without_rule': metrics(y, raw_oof),
        'fold_macro_mean': float(np.mean(values)), 'fold_macro_sd': float(np.std(values, ddof=1)),
        'folds': fold_results}

def train_full(estimator, X, y):
    """최종 모델을 메모리에 반환한다. 파일 저장이나 200명 예측은 하지 않는다."""
    return {'model': fresh_model(estimator).fit(X, y), 'rule': learn_rule(X, y), 'threshold': THRESHOLD}

def run(with_rf=False):
    X, y, digest = load_training()
    result = {'source_sha256': digest, 'n': len(y), 'bad_n': int(y.sum()), 'seed': SEED,
              'threshold': THRESHOLD, 'logistic': evaluate(make_logistic(), X, y)}
    if with_rf:
        result['random_forest_reference'] = evaluate(make_reference_forest(), X, y)
    fitted = train_full(make_logistic(), X, y)
    assert np.isin(predict(fitted['model'], X.iloc[:3], fitted['rule'])[0], [0, 1]).all()
    result['full_model_trained'] = True
    return result, fitted

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--with-rf', action='store_true')
    arguments = parser.parse_args()
    results, trained_model = run(with_rf=arguments.with_rf)
    print(json.dumps(results, ensure_ascii=False, indent=2))
