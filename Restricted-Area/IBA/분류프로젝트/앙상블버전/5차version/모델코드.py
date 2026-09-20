"""3차 스태킹 + 희찬식 SVC: 내부 선택/외부 평가/최종 제출 생성."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import csv
import hashlib
import importlib.util
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, recall_score

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[1]
spec = importlib.util.spec_from_file_location('third_final', PROJECT/'앙상블버전'/'3차version'/'모델코드.py')
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)
WEIGHTS = np.round(np.arange(0, 1.01, .1), 2)
THRESHOLDS = np.round(np.arange(.35, .751, .01), 2)
NOMINAL = ['Sex', 'Housing', 'Purpose', 'Job']
NUMERIC = ['Age', 'Credit amount', 'Duration']

def svc_features(raw):
    x = raw[NOMINAL + NUMERIC].copy()
    x['Job'] = x['Job'].astype(str)
    for name, prefix, mapping, fill in [
        ('Checking account', 'chk', {'little':1,'moderate':2,'rich':3}, 2.),
        ('Saving accounts', 'sav', {'little':1,'moderate':2,'quite rich':3,'rich':4}, 2.5)]:
        x[prefix+'_lv'] = raw[name].map(mapping).fillna(fill)
        x[prefix+'_none'] = raw[name].eq('').astype(int)
    return x

def make_svc():
    return Pipeline([('encoder', ColumnTransformer([
        ('nominal', OneHotEncoder(handle_unknown='ignore'), NOMINAL),
        ('numeric', StandardScaler(), NUMERIC+['chk_lv','chk_none','sav_lv','sav_none'])])),
        ('model', SVC(C=.5, gamma=.1, kernel='rbf', random_state=42))])

def base_pair(train, y, test):
    rf = v3.fresh_model(v3.make_reference_forest()).fit(train, y)
    cb = v3.make_catboost().fit(train, y)
    return np.column_stack([v3.bad_probability(rf,test),v3.bad_probability(cb,test)])

def component_scores(train, y, test, seed):
    """평가 정답을 받지 않으며 스태킹 메타 학습에서도 평가 행을 제외한다."""
    x, z = v3.prepare_features(train), v3.prepare_features(test)
    p = np.empty((len(y),2))
    for tr, va in StratifiedKFold(3, shuffle=True, random_state=seed).split(x,y):
        p[va] = base_pair(x.iloc[tr], y.iloc[tr], x.iloc[va])
    meta = v3.make_meta_model().fit(p,y)
    third = meta.predict_proba(base_pair(x,y,z))[:,1]
    rule = v3.learn_rule(x,y)
    forced = ((z['Duration']<rule['duration']) | (z['Credit amount']<rule['amount'])).to_numpy()
    third[forced] = 0.
    svc = make_svc().fit(svc_features(train),y)
    svc_score = svc.decision_function(svc_features(test))
    # 배치별 순위: 희찬 방식. 대상 행들의 정답은 사용하지 않는다.
    ranks = np.column_stack([rankdata(third)/len(test),rankdata(svc_score)/len(test)])
    return ranks, third

def inner_oof(raw,y,seed,progress=False):
    ranks = np.empty((len(y),2))
    for i,(tr,va) in enumerate(StratifiedKFold(3,shuffle=True,random_state=seed).split(raw,y),1):
        ranks[va],_ = component_scores(raw.iloc[tr],y.iloc[tr],raw.iloc[va],seed+100+i)
        if progress: print(f'  selection fold {i}/3 done',flush=True)
    return ranks

def select(ranks,y,weights=WEIGHTS):
    best = None
    for weight in weights:
        scores = weight*ranks[:,0] + (1-weight)*ranks[:,1]
        for threshold in THRESHOLDS:
            f1 = f1_score(y,scores>=threshold,zero_division=0)
            # 동일 성능이면 단일 모델, 그 다음 50:50에 가까운 비중 우선
            key = (f1, int(weight in (0,1)), -abs(weight-.5), -abs(threshold-.5))
            if best is None or key>best[0]: best = (key,float(weight),float(threshold))
    return best[1:]

def predict(ranks,config):
    weight,threshold = config
    return weight*ranks[:,0] + (1-weight)*ranks[:,1] >= threshold

def metrics(y,p):
    return [precision_score(y,p,zero_division=0),recall_score(y,p,zero_division=0),f1_score(y,p,zero_division=0)]

def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        return reader.fieldnames,list(reader)

def main():
    train_path=PROJECT/'분류_학습_데이터셋.csv'
    answer_path=PROJECT/'정답 입력 데이터.csv'
    hashes={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in [train_path,answer_path]}
    raw=pd.read_csv(train_path,keep_default_na=False)
    answer=pd.read_csv(answer_path,keep_default_na=False)
    y=raw['Risk'].eq('bad').astype(int)
    assert len(raw)==800 and len(answer)==200 and y.sum()==240
    predictions={name:np.zeros(len(y),dtype=bool) for name in ['5차 선택 절차','3차 고정 0.50','희찬식 SVC 내부 임계값 선택']}
    details=[]
    for fold,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(raw,y),1):
        print(f'outer fold {fold}/5: selection',flush=True)
        train,yt=raw.iloc[tr],y.iloc[tr]
        inside=inner_oof(train,yt,142+fold,progress=True)
        config=select(inside,yt)
        svc_config=select(inside,yt,weights=[0.])
        outside,third=component_scores(train,yt,raw.iloc[va],242+fold)
        predictions['5차 선택 절차'][va]=predict(outside,config)
        predictions['3차 고정 0.50'][va]=third>=.5
        predictions['희찬식 SVC 내부 임계값 선택'][va]=predict(outside,svc_config)
        result=metrics(y.iloc[va],predictions['5차 선택 절차'][va])
        details.append((fold,*config,result[2]))
        print(f'outer {fold}/5 done: weight={config[0]:.1f}, threshold={config[1]:.2f}, F1={result[2]:.4f}',flush=True)
    print('EXTERNAL RESULTS',flush=True)
    for name,p in predictions.items(): print(name,metrics(y,p),flush=True)
    print('final selection on 800 training rows',flush=True)
    inside=inner_oof(raw,y,342,progress=True)
    final_config=select(inside,y)
    ranks,_=component_scores(raw,y,answer,442)
    labels=np.where(predict(ranks,final_config),'bad','good')
    fields,rows=read_csv(answer_path)
    for row,label in zip(rows,labels): row['Risk']=label
    target=OUT/'정답_입력데이터_5차예측.csv'
    with target.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    of,original=read_csv(answer_path);tf,generated=read_csv(target)
    assert of==tf and len(generated)==200
    assert all(a[k]==b[k] for a,b in zip(original,generated) for k in of if k!='Risk')
    assert all(r['Risk'] in ['good','bad'] for r in generated)
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==h for p,h in hashes.items())
    w,t=final_config
    report=['# 5차 모델: 3차 스태킹과 희찬식 SVC의 순위 결합','',
        '## 외부 검증 결과','',
        '800명을 외부 5분할(seed=42)로 나누었다. 각 외부 학습 640명 내부의 3분할 OOF에서 가중치와 임계값을 선택하고, 선택에서 제외된 160명에 적용했다. 스태킹 메타 학습에도 별도 3분할 OOF를 사용했다.',
        '', '| 모델 | Bad 정밀도 | Bad 재현율 | Bad F1 |','|---|---:|---:|---:|']
    for name,p in predictions.items():
        a,b,c=metrics(y,p);report.append(f'| {name} | {a:.4f} | {b:.4f} | {c:.4f} |')
    score5=metrics(y,predictions['5차 선택 절차'])[2]
    best_baseline=max(metrics(y,predictions[name])[2] for name in predictions if name!='5차 선택 절차')
    report+=['', ('이번 외부 검증에서 5차가 두 기준 모델을 앞섰다. 다른 분할에서도 확인해야 한다.' if score5>best_baseline else '이번 외부 검증에서 5차 결합은 기준 모델의 최고 성능을 넘지 못했다. 제출 후보 CSV는 생성했지만 성능 개선을 근거로 기존 모델을 교체하는 것은 권하지 않는다.')]
    report+=['','각 점수는 외부 예측 800개를 합쳐 계산한 값이다. 과거 3차 반복 평균 0.5849와 직접 차이를 계산하지 않고 위 동일 분할 기준으로 비교한다. 희찬식 SVC 역시 전체 데이터로 정한 0.5921을 재사용하지 않고 학습 부분에서 임계값을 선택했다.',
        '', '| 외부 분할 | 3차 비중 | 임계값 | Bad F1 |','|---|---:|---:|---:|']
    for fold,weight,threshold,f in details: report.append(f'| {fold} | {weight:.1f} | {threshold:.2f} | {f:.4f} |')
    report+=['','## 구현과 최종 제출 설정','',
        '- 3차: RF 400 trees, min_samples_leaf=2, balanced_subsample; CatBoost 500 iterations, depth=5, learning_rate=0.05, L2=13, Balanced. 균형 로지스틱 회귀(C=1)로 스태킹한다.',
        '- 3차의 계좌 Unknown과 Account information, 학습 부분 Bad 최소 기간·금액보다 작으면 Good으로 보는 규칙을 유지했다. 규칙은 3차 구성 점수에만 적용하며 SVC 판정을 강제하지 않는다.',
        '- SVC: C=0.5, gamma=0.1, 가중치 없음. 계좌 수준 순위 및 결측 플래그, 나머지 명목형 원핫·수치형 표준화를 사용한다. 빈칸의 실제 의미를 임의로 모두 미보유라고 확정하지 않는다.',
        '- 두 모델의 점수를 예측 배치 안에서 순위(0~1)로 변환한 뒤 가중 평균한다. 확률과 순위를 직접 섞지 않는다. 이 결합 순위점수에 임계값을 적용하며 다시 순위화하지 않는다.',
        '- 가중치: 3차 비중 0~1, 0.1 간격. 임계값: 0.35~0.75, 0.01 간격. 가중치 0 또는 1도 허용해 결합이 불리하면 단일 구성으로 돌아갈 수 있다.',
        f'- 최종 800명 내부 선택: 3차 {w:.0%}, SVC {1-w:.0%}, 결합 점수 임계값 {t:.2f}.',
        f'- 제출: good {int((labels=="good").sum())}명, bad {int((labels=="bad").sum())}명. 원본 11개 열/200행을 보존하고 Risk만 변경했다.',
        '- 최종 기본 모델은 800명 전체로 재학습하고 메타 모델은 3분할 OOF로 학습한다. 이번 5차에서는 검증과 최종 스태킹의 내부 분할 수를 3으로 통일했다. 기존 3차 제출 코드의 최종 메타 학습은 5분할이므로 동일 학습 객체의 재사용은 아니다.',
        '', '## 해석 범위','',
        '보고 F1은 분할마다 구성을 고르는 전체 선택 절차의 평가다. 최종 800명에서 고른 특정 가중치 하나의 독립 평가값으로 해석하지 않는다. 과거 실험에도 같은 800명이 사용되었으며 새 seed는 새로운 데이터가 아니다. 이번에는 반복 외부 검증까지 추가하지 않았다.',
        '순위는 대상 배치의 구성에 의존하므로 같은 사람도 함께 예측하는 사람이 달라지면 판정이 바뀔 수 있다. 정답 입력 200명의 실제 Risk를 모르므로 제출 F1은 여기서 계산할 수 없다.',
        '', '## 파일','',
        '- 모델코드.py: 모델 구현과 평가·예측 코드. 2·3차 공통 전처리/모델 정의를 재사용한다.',
        '- 결과보고서.md: 당시 검증 결과와 해석. 실행하면 예측 CSV를 다시 생성할 수 있다.',
        '', f'원본 학습 SHA256: {hashes[train_path]}']
    (OUT/'결과보고서.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(f'FINAL: third={w}, svc={1-w}, threshold={t}, bad={(labels=="bad").sum()}',flush=True)

if __name__=='__main__': main()
