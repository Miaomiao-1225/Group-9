from pathlib import Path
import ast, json, math, re
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from streamlit.testing.v1 import AppTest
ROOT=Path(__file__).resolve().parent
report=[]
def check(value,message):
    assert value,message
    report.append('PASS: '+message)
original=ast.parse((ROOT/'original_code.py').read_text())
updated=ast.parse((ROOT/'train_original.py').read_text())
functions=['preprocess_data','feature_engineering','apply_enn_standalone','calculate_gmean','calculate_ks']
namespace=dict(pd=pd,np=np,re=re,math=math,confusion_matrix=confusion_matrix,target_col='FraudFound')
for name in functions:
    a=next(v for v in original.body if isinstance(v,ast.FunctionDef) and v.name==name)
    b=next(v for v in updated.body if isinstance(v,ast.FunctionDef) and v.name==name)
    check(ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False),'Original function unchanged: '+name)
    exec(compile(ast.Module(body=[a],type_ignores=[]),'<original_functions>','exec'),namespace)
meta=json.loads((ROOT/'results/metadata.json').read_text())
data=pd.read_csv(ROOT/'data/vehicle_fraud.csv')
x=namespace['feature_engineering'](namespace['preprocess_data'](data)).drop(columns=['FraudFound'])
check(len(x.columns)==meta['engineered_fields'],'Full original engineered field count retained')
check(set(['MonthDiff','AgeGroup','IsWeekend','IsWeekendClaim','IsLuxuryBrand','PolicyTypeSimple']).issubset(x.columns),'Original derived features retained')
state=joblib.load(ROOT/'results/preprocessing.joblib')
for col in state['categorical_cols']:
    x[col]=x[col].astype(object).fillna('Unknown').map(lambda v:str(v) if not isinstance(v,str) else v)
encoded=pd.DataFrame(state['preprocessor'].transform(x),columns=state['preprocessor'].get_feature_names_out()).fillna(0)
scaled=state['scaler'].transform(encoded)
pred=pd.read_csv(ROOT/'results/test_predictions.csv')
metrics=pd.read_csv(ROOT/'results/original_metrics.csv')
expected=['SMOTE','ENN','SMOTE-Tomek','ADASYN','SMOTE-ENN']
check(list(meta['model_files'])==expected,'Exactly the original five methods; no added Baseline')
split=pd.read_csv(ROOT/'results/split_indices.csv')
check(split.Row_index.is_unique and len(split)==len(data),'70:30 train/test split is disjoint and covers dataset')
check(meta['train_rows']==10794 and meta['test_rows']==4626,'Original split sizes retained')
check(meta['encoded_fields']==scaled.shape[1] and scaled.shape[1]>10,'Models retain complete encoded input, not ten-field subset')
for method,filename in meta['model_files'].items():
    booster=xgb.Booster(); booster.load_model(ROOT/'results'/filename)
    check(booster.num_features()==scaled.shape[1],method+': native JSON retains all original model inputs')
    score=booster.predict(xgb.DMatrix(scaled[pred.Row_index]))
    check(np.allclose(score,pred[method],atol=1e-7),method+': JSON model predictions match exported original run')
    labels=np.where(score>.5,'Yes','No')
    y=pred.Actual.to_numpy()
    gmean=namespace['calculate_gmean'](y,labels)
    ks=namespace['calculate_ks'](y,score)
    cm=confusion_matrix(y,labels,labels=['No','Yes']); tn,fp,fn,tp=cm.ravel()
    values=[round(roc_auc_score(y=='Yes',score),4),round(tp/(tp+fn),4),round(gmean,4),round(ks,4)]
    actual=metrics.set_index('Method').loc[method,['AUC','Recall','G_mean','KS']].to_numpy(dtype=float)
    check(np.allclose(values,actual),method+': all four displayed metrics equal original formulas')
rank=pd.read_csv(ROOT/'results/feature_importance.csv')
check(len(rank.head(10))==10 and set(rank.Feature).issubset(x.columns),'Top 10 consists of original engineered fields')
check(np.isclose(rank.Gain_share.sum(),1),'Grouped gain shares sum to one across all fields')
for page in ['Dataset overview','Top 10 features','Original model comparison','Test case explorer','Validation & sharing']:
    at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
    at.sidebar.radio[0].set_value(page).run()
    check(not at.exception,'Streamlit page renders: '+page)
    if page=='Original model comparison':
        at.selectbox[0].set_value('SMOTE-ENN').run()
        check(not at.exception,'Confusion matrix model selector works')
    if page=='Test case explorer':
        at.number_input[0].set_value(10).run()
        check(not at.exception,'Test case selector works')
report.insert(0,'Original-workflow verification executed locally. Windows batch execution and online deployment are not tested here. Historical outputs were not supplied; checks compare against the original-code run in this environment.')
(ROOT/'validation_report.txt').write_text('\n'.join(report)+'\n')
print('\n'.join(report))
