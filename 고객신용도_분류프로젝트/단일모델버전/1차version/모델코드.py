# %% 1. 원본 데이터와 1차 모델 설정
"""1차 Risk 모델: RandomForest + 학습 bad 최솟값에 근거한 good 우선 규칙.
python 모델코드.py 로 결과를 재생성한다. 원본 CSV는 수정하지 않는다.
"""
from pathlib import Path
import csv
import hashlib
import json
import os
import tempfile
import platform
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'iba-matplotlib-cache'))
import numpy as np
import pandas as pd
import sklearn
import joblib
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

candidates=[]
for parent in [Path.cwd(),*Path.cwd().parents]:
    candidates.extend([parent,parent/'vscode'/'1급극비파일'/'IBA'/'분류프로젝트'])
PROJECT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
OUT.mkdir(exist_ok=True)
TRAIN_PATH=PROJECT/'분류_학습_데이터셋.csv'
ANSWER_PATH=PROJECT/'정답 입력 데이터.csv'
source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [TRAIN_PATH,ANSWER_PATH]}
def read_original(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        return reader.fieldnames,list(reader)
train_fields,train_records=read_original(TRAIN_PATH)
answer_fields,answer_records=read_original(ANSWER_PATH)
train=pd.DataFrame(train_records)
answers=pd.DataFrame(answer_records)
assert len(train)==800 and len(answers)==200
assert train['Risk'].isin(['good','bad']).all() and answers['Risk'].eq('').all()
assert train[''].is_unique and answers[''].is_unique
assert set(train['']).isdisjoint(set(answers['']))
SEED=42
PROBABILITY_THRESHOLD=.5
NUMERIC=['Age','Credit amount','Duration']
ORIGINAL_FEATURES=['Age','Sex','Job','Housing','Saving accounts','Checking account','Credit amount','Duration','Purpose']
CATEGORICAL=[c for c in ORIGINAL_FEATURES if c not in NUMERIC]+['Account information']

def prepare_features(records):
    frame=records[ORIGINAL_FEATURES].copy()
    for col in NUMERIC:frame[col]=pd.to_numeric(frame[col],errors='raise')
    saving_present=frame['Saving accounts'].ne('')
    checking_present=frame['Checking account'].ne('')
    frame['Account information']=np.select(
        [saving_present&checking_present,saving_present,checking_present],
        ['both_present','saving_only','checking_only'],default='both_missing')
    for col in ['Saving accounts','Checking account']:
        frame[col]=frame[col].replace('','Unknown')
    return frame

X=prepare_features(train)
y=train['Risk'].eq('bad').astype(int)
X_answer=prepare_features(answers)
training_idx,validation_idx=train_test_split(train.index,test_size=.2,random_state=SEED,stratify=y)
assert set(training_idx).isdisjoint(set(validation_idx))
preprocessor=ColumnTransformer([
    ('numeric','passthrough',NUMERIC),
    ('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=False),CATEGORICAL)])
template=Pipeline([
    ('preprocess',preprocessor),
    ('classifier',RandomForestClassifier(n_estimators=400,min_samples_leaf=2,
        class_weight='balanced_subsample',max_features='sqrt',random_state=SEED,n_jobs=2))])
print(f'원본 학습 {len(train)}명 / 정답 입력 {len(answers)}명')
print(f'학습 {len(training_idx)}명(bad {int(y.loc[training_idx].sum())}) / 검증 {len(validation_idx)}명(bad {int(y.loc[validation_idx].sum())})')

# %% 2. 학습 부분만으로 모델과 good 우선 규칙 만들기
def learn_minimum_rule(features,target):
    bad=features.loc[target.eq(1)]
    if bad.empty:raise ValueError('bad 표본 없이 최솟값 규칙을 만들 수 없습니다.')
    return {'minimum_bad_duration':int(bad['Duration'].min()),
            'minimum_bad_credit_amount':int(bad['Credit amount'].min()),
            'operator':'OR','comparison':'strictly_less_than'}

def matches_good_rule(features,rule):
    # 조건 하나라도 만족하면 good. 최솟값과 같은 경우는 해당하지 않는다.
    return ((features['Duration']<rule['minimum_bad_duration']) |
            (features['Credit amount']<rule['minimum_bad_credit_amount'])).to_numpy()

def predict_with_rule(model,features,rule):
    bad_column=list(model.classes_).index(1)
    scores=model.predict_proba(features)[:,bad_column]
    model_pred=(scores>=PROBABILITY_THRESHOLD).astype(int)
    rule_hit=matches_good_rule(features,rule)
    final_pred=np.where(rule_hit,0,model_pred)
    return scores,model_pred,rule_hit,final_pred

validation_model=clone(template).fit(X.loc[training_idx],y.loc[training_idx])
validation_rule=learn_minimum_rule(X.loc[training_idx],y.loc[training_idx])
val_scores,val_model_pred,val_rule_hit,val_final_pred=predict_with_rule(validation_model,X.loc[validation_idx],validation_rule)
print('검증용 규칙 (학습 640명에서만 계산):',validation_rule)

# %% 3. 검증 F1과 규칙 적용 전후 비교
def measure(name,truth,pred):
    tn,fp,fn,tp=confusion_matrix(truth,pred,labels=[0,1]).ravel()
    return {'방법':name,'bad_precision':precision_score(truth,pred,zero_division=0),
        'bad_recall':recall_score(truth,pred,zero_division=0),
        'bad_F1':f1_score(truth,pred,zero_division=0),
        'good_F1':f1_score(truth,pred,pos_label=0,zero_division=0),
        'macro_F1':f1_score(truth,pred,average='macro',zero_division=0),
        'accuracy':accuracy_score(truth,pred),'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp)}
metrics=pd.DataFrame([
    measure('모두 good 기준',y.loc[validation_idx],np.zeros(len(validation_idx),dtype=int)),
    measure('랜덤포레스트',y.loc[validation_idx],val_model_pred),
    measure('랜덤포레스트 + good 우선 규칙',y.loc[validation_idx],val_final_pred)])
metrics.to_csv(OUT/'검증성능.csv',index=False,encoding='utf-8-sig')
validation_output=train.loc[validation_idx].copy()
validation_output['모델_bad_점수']=val_scores
validation_output['모델_예측']=np.where(val_model_pred==1,'bad','good')
validation_output['good우선규칙_적용']=val_rule_hit
validation_output['최종_예측']=np.where(val_final_pred==1,'bad','good')
validation_output['정답여부']=val_final_pred==y.loc[validation_idx].to_numpy()
validation_output.to_csv(OUT/'검증160명_예측비교.csv',index=False,encoding='utf-8-sig')
pd.DataFrame({'원본_ID':train[''],'구분':np.where(train.index.isin(training_idx),'학습','검증'),'Risk':train['Risk']}).to_csv(OUT/'학습검증_분할.csv',index=False,encoding='utf-8-sig')
print(metrics[['방법','bad_precision','bad_recall','bad_F1','macro_F1','accuracy']].to_string(index=False))
print(f'검증 규칙 해당 {int(val_rule_hit.sum())}명 / 해당자 실제 bad {int(y.loc[validation_idx].to_numpy()[val_rule_hit].sum())}명 / 모델 예측에서 변경 {int((val_model_pred!=val_final_pred).sum())}명')

# %% 4. 전체 800명으로 재학습 후 정답 입력 200명 예측
final_model=clone(template).fit(X,y)
final_rule=learn_minimum_rule(X,y)
answer_scores,answer_model_pred,answer_rule_hit,answer_final_pred=predict_with_rule(final_model,X_answer,final_rule)
answer_labels=np.where(answer_final_pred==1,'bad','good')
submission_rows=[{**row,'Risk':str(label)} for row,label in zip(answer_records,answer_labels)]
submission_path=OUT/'정답_입력데이터_1차예측.csv'
with submission_path.open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=answer_fields)
    writer.writeheader();writer.writerows(submission_rows)
answer_detail=pd.DataFrame(submission_rows)
answer_detail['모델_bad_점수']=answer_scores
answer_detail['모델_예측']=np.where(answer_model_pred==1,'bad','good')
answer_detail['good우선규칙_적용']=answer_rule_hit
answer_detail['최종판별근거']=np.where(answer_rule_hit,'학습 bad 최솟값 미만 → good','랜덤포레스트 점수 기준')
answer_detail.to_csv(OUT/'정답200명_예측상세.csv',index=False,encoding='utf-8-sig')
joblib.dump({'pipeline':final_model,'minimum_rule':final_rule,'probability_threshold':PROBABILITY_THRESHOLD,
    'positive_class':'bad','original_features':ORIGINAL_FEATURES,
    'feature_preparation':'모델코드.py의 prepare_features 함수를 동일하게 적용',
    'sklearn_version':sklearn.__version__},OUT/'1차모델.joblib',compress=3)
summary={'version':'1차version','seed':SEED,'train_n':len(training_idx),'validation_n':len(validation_idx),
    'validation_bad_n':int(y.loc[validation_idx].sum()),'probability_threshold':PROBABILITY_THRESHOLD,
    'validation_rule':validation_rule,'final_rule':final_rule,
    'validation_rule_hits':int(val_rule_hit.sum()),'validation_rule_actual_bad':int(y.loc[validation_idx].to_numpy()[val_rule_hit].sum()),
    'validation_rule_changed_predictions':int((val_model_pred!=val_final_pred).sum()),
    'answer_rule_hits':int(answer_rule_hit.sum()),'answer_rule_changed_predictions':int((answer_model_pred!=answer_final_pred).sum()),
    'answer_bad':int(answer_final_pred.sum()),'answer_good':int((answer_final_pred==0).sum()),
    'metrics':metrics.to_dict(orient='records'),'source_sha256':source_hashes,
    'versions':{'python':platform.python_version(),'scikit-learn':sklearn.__version__,'pandas':pd.__version__,'numpy':np.__version__}}
(OUT/'실행요약.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
assert len(submission_rows)==200
for before,after in zip(answer_records,submission_rows):
    assert list(before)==list(after) and after['Risk'] in ['good','bad']
    assert all(before[k]==after[k] for k in answer_fields if k!='Risk')
for source in [TRAIN_PATH,ANSWER_PATH]:assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hashes[source.name]
print(f'정답 입력 200명 예측: good {summary["answer_good"]}명 / bad {summary["answer_bad"]}명')
print('최종 규칙:',final_rule)
print(f'정답 입력에서 우선 규칙 적용 {summary["answer_rule_hits"]}명 / 모델 예측에서 변경 {summary["answer_rule_changed_predictions"]}명')

# %% 5. 검증 그래프와 1차 결과 보고서
plt.rcParams['font.family']='Malgun Gothic'
plt.rcParams['axes.unicode_minus']=False
cm=confusion_matrix(y.loc[validation_idx],val_final_pred,labels=[0,1])
fig,ax=plt.subplots(figsize=(6,5));ax.imshow(cm,cmap='Blues')
ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['good','bad'],yticklabels=['good','bad'],
    xlabel='예측',ylabel='실제 Risk',title='1차 모델 · 검증 160명 혼동행렬')
for (i,j),v in np.ndenumerate(cm):ax.text(j,i,str(v),ha='center',va='center',fontsize=18,color='white' if v>cm.max()/2 else 'black')
fig.tight_layout();fig.savefig(OUT/'검증_혼동행렬.png',dpi=150);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4))
positions=np.arange(3)
for j,key in enumerate(['bad_precision','bad_recall','bad_F1']):
    ax.bar(positions+(j-1)*.24,metrics[key],width=.24,label=key)
ax.set(xticks=positions,xticklabels=['모두 good','랜덤포레스트','모델 + 우선 규칙'],ylim=(0,1),ylabel='점수',title='동일 검증 자료에서 비교')
ax.legend();fig.tight_layout();fig.savefig(OUT/'검증_성능비교.png',dpi=150);plt.close(fig)
m=metrics.iloc[-1]
metric_table='\n'.join(['| 방법 | bad 정밀도 | bad 재현율 | bad F1 | macro F1 | 정확도 |','|---|---:|---:|---:|---:|---:|']+
    [f'| {r["방법"]} | {r["bad_precision"]:.3f} | {r["bad_recall"]:.3f} | {r["bad_F1"]:.3f} | {r["macro_F1"]:.3f} | {r["accuracy"]:.3f} |' for r in metrics.to_dict(orient='records')])
report=f'''# 1차version · 신용 위험 예측

**검증 160명에서 최종 bad F1은 {m['bad_F1']:.4f}, 정밀도는 {m['bad_precision']:.4f}, 재현율은 {m['bad_recall']:.4f}이다.** 같은 모델 구성을 전체 800명에 다시 학습하여 정답 입력 200명의 Risk를 작성했다.

## 사용 데이터와 모델

- 원본 학습 데이터 800명 사용. 이전에 Risk를 사용해 저축을 복원한 검토본은 사용하지 않았다.
- 80:20 계층 분할: 학습 640명(good 448, bad 192), 검증 160명(good 112, bad 48). seed=42.
- 입력은 원본 9개 특성과 계좌 정보 상태(둘 다 있음/저축만/당좌만/둘 다 결측)다. ID는 제외했고 Risk는 정답으로만 사용했다.
- 계좌 빈칸은 Unknown으로 표시했다. 잔액 수준을 추정 대체하지 않았다. 직업은 숫자 코드여도 범주형으로 처리했다.
- 범주형은 원핫 인코딩, 수치형은 원값을 사용했다. 인코더는 학습 부분에서만 학습했다.
- 랜덤포레스트: 나무 400개, 말단 최소 2명, 각 나무의 부트스트랩 표본에 맞춘 클래스 균형 가중치, max_features=sqrt. 깊이는 별도 제한하지 않았다.
- 모델의 bad 점수가 0.5 이상이면 bad, 미만이면 good. 이번 버전은 하이퍼파라미터·점수 기준 최적화를 하지 않은 기본 모델이다.

## 사용자 지정 good 우선 규칙

**대출기간 < 학습 bad의 최소 기간 OR 대출금액 < 학습 bad의 최소 금액이면 good으로 우선 판별**한다. 둘 중 하나만 만족해도 적용하고, 최솟값과 같은 경우에는 적용하지 않는다.

| 적용 단계 | 최소 기간 | 최소 금액 | 계산에 사용한 자료 |
|---|---:|---:|---|
| 검증 | {validation_rule['minimum_bad_duration']}개월 | {validation_rule['minimum_bad_credit_amount']} DM | 학습 640명만 |
| 정답 입력 예측 | {final_rule['minimum_bad_duration']}개월 | {final_rule['minimum_bad_credit_amount']} DM | 전체 학습 800명 |

전체 학습에서 규칙에 해당하는 14명은 모두 good이다. 이는 현재 표본에서 관찰한 결과이며 더 짧거나 작은 대출이 항상 good이라는 보장은 아니다. 최솟값은 임의의 새 데이터에서 바뀔 수 있다.

검증에서 해당자는 {summary['validation_rule_hits']}명이고 그중 실제 bad는 {summary['validation_rule_actual_bad']}명이다. 모델 단독 예측에서 {summary['validation_rule_changed_predictions']}명의 결과를 바꿨다. 검증용 최솟값 계산에 검증 정답을 사용하지 않았다.

## 검증 성능

{metric_table}

bad를 양성 클래스로 정의했다. F1은 정밀도와 재현율의 조화평균이다. 과제 슬라이드에서 F1의 양성 클래스·평균 방식이 명시되지 않아 macro F1도 함께 기록했다.

최종 혼동행렬: 실제 good을 good으로 {int(m['TN'])}명, 실제 good을 bad로 {int(m['FP'])}명, 실제 bad를 good으로 {int(m['FN'])}명, 실제 bad를 bad로 {int(m['TP'])}명 예측했다.

규칙 적용 전후의 차이는 이 검증 분할에서의 결과이며, 표본이 적으므로 일반적인 성능 향상을 확정하지 않는다. 한 번의 내부 분할 평가이고 앞선 EDA에서 전체 자료를 살펴본 이력이 있으므로 완전히 독립적인 외부 검증 결과로 해석하지 않는다.

## 정답 입력 200명 결과

- 예측 good: **{summary['answer_good']}명**, bad: **{summary['answer_bad']}명**.
- 우선 규칙에 해당: {summary['answer_rule_hits']}명. 모델 단독 판단에서 변경: {summary['answer_rule_changed_predictions']}명.
- `정답_입력데이터_1차예측.csv`는 원본 200행의 순서와 모든 입력값·헤더를 유지하고 Risk만 채운 결과다.
- `정답200명_예측상세.csv`에는 모델의 bad 점수, 규칙 적용 여부, 최종 판별 근거를 추가했다. 규칙으로 good이 되어도 원래 모델 점수를 0으로 바꾸지 않았다. 점수는 보정된 실제 위험 확률로 보장되지 않는다.
- 정답 입력 데이터의 실제 Risk가 없어 이 200명에 대한 F1은 계산할 수 없다. 위 F1은 따로 남긴 학습 자료 160명에서 계산한 것이다.

## 저장 파일과 재현

- `모델코드.py`: 데이터 전처리, 모델·규칙 학습, 검증, 최종 예측 코드.
- `결과보고서.md`: 당시 검증 결과와 해석.

재현은 필요한 패키지가 있는 Python에서 `python 모델코드.py`로 가능하다. 실행하면 예측 CSV·모델 파일·검증 그림 등이 다시 생성된다. 당시 실행 버전은 Python {platform.python_version()}, scikit-learn {sklearn.__version__}, pandas {pd.__version__}, numpy {np.__version__}이다.
'''
(OUT/'결과보고서.md').write_text(report,encoding='utf-8')
print('검증·예측·원본 보존 확인 완료. 1차 결과 파일 저장 완료.')
