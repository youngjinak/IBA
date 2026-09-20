"""Extra Trees: python -B extra_trees.py. 추가 파일을 생성하지 않는다."""
import sys
sys.dont_write_bytecode = True
import json
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesClassifier
from logistic_regression import NUMERIC, CATEGORICAL, SEED, load_training, evaluate, train_full

def make_extra_trees():
    preprocessing = ColumnTransformer([
        ('numeric', 'passthrough', NUMERIC),
        ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL)])
    return Pipeline([('preprocess', preprocessing), ('model', ExtraTreesClassifier(
        n_estimators=400, min_samples_leaf=2, max_features='sqrt',
        class_weight='balanced', random_state=SEED, n_jobs=2))])

def run():
    # 기존과 같은 5분할·50% 판정·학습 부분의 bad 최솟값 good 우선 규칙.
    X, y, digest = load_training()
    result = evaluate(make_extra_trees(), X, y)
    result['source_sha256'] = digest
    fitted = train_full(make_extra_trees(), X, y)
    return result, fitted

if __name__ == '__main__':
    result, trained_model = run()
    print(json.dumps({'model': 'Extra Trees', **result['oof'],
        'fold_macro_mean': result['fold_macro_mean'], 'fold_macro_sd': result['fold_macro_sd']}, indent=2))
