"""RBF SVM: python -B svm_model.py. 추가 파일을 생성하지 않는다."""
import sys
sys.dont_write_bytecode = True
import json
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from logistic_regression import NUMERIC, CATEGORICAL, SEED, load_training, evaluate, train_full

def make_svm():
    preprocessing = ColumnTransformer([
        ('numeric', StandardScaler(), NUMERIC),
        ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL)])
    # probability=True: 학습 자료 내부에서 추정한 확률의 50% 기준으로 비교한다.
    # SVC.predict의 결정함수 0 기준과 결과가 다를 수 있다.
    return Pipeline([('preprocess', preprocessing), ('model', SVC(
        kernel='rbf', C=1.0, gamma='scale', class_weight='balanced',
        probability=True, random_state=SEED))])

def run():
    X, y, digest = load_training()
    result = evaluate(make_svm(), X, y)
    result['source_sha256'] = digest
    fitted = train_full(make_svm(), X, y)
    return result, fitted

if __name__ == '__main__':
    result, trained_model = run()
    print(json.dumps({'model': 'SVM', **result['oof'],
        'fold_macro_mean': result['fold_macro_mean'], 'fold_macro_sd': result['fold_macro_sd']}, indent=2))
