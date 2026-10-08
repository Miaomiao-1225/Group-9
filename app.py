from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_curve

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='Vehicle Fraud — Original Project', page_icon='🚗', layout='wide')
@st.cache_data
def read_csv(file):
    return pd.read_csv(ROOT/file)
@st.cache_data
def metadata():
    return json.loads((ROOT/'results/metadata.json').read_text())

st.title('🚗 Vehicle Insurance Fraud Detection')
st.caption('Original project · Five sampling methods · All original engineered features retained | 保留原版流程')
if not (ROOT/'results/metadata.json').exists():
    st.error('Results missing. Run: python train_original.py')
    st.stop()
meta=metadata()
data=read_csv('data/vehicle_fraud.csv')
page=st.sidebar.radio('功能 / Explore', ['Dataset overview','Top 10 features','Original model comparison','Test case explorer','Validation & sharing'])
st.sidebar.info('Top 10 is a display ranking only. Models still use ALL original engineered features.\n\n前十特征仅用于展示，模型输入未删减。')

def download(label, frame, name):
    st.download_button(label,frame.to_csv(index=False).encode('utf-8-sig'),name,'text/csv')

if page=='Dataset overview':
    columns=st.columns(4)
    for col,label,value in zip(columns,['Claims / 记录','Fraud / 欺诈','Fraud rate / 欺诈率','Engineered fields / 工程后字段'],[len(data),int(data.FraudFound.eq('Yes').sum()),f'{data.FraudFound.eq("Yes").mean():.2%}',meta['engineered_fields']]):
        col.metric(label,value)
    st.bar_chart(data.FraudFound.value_counts().rename('Claims'))
    st.write(f"Train: {meta['train_rows']:,} · Test: {meta['test_rows']:,} · Encoded inputs: {meta['encoded_fields']}")
    st.dataframe(data.head(100),hide_index=True,width='stretch')
    st.markdown('''**Original feature engineering retained / 保留原来的特征工程**

- Range midpoint conversion: vehicle age, policy-holder age, vehicle price, policy/claim intervals, prior claims, supplements, address-change interval and car count.
- Accident month, claim month and absolute month difference.
- Accident/claim weekend flags, age groups and luxury-brand flag.
- Simplified policy type, one-hot encoding and standardization.
''')

elif page=='Top 10 features':
    st.subheader('Top 10 engineered fields / 前十重要特征')
    rank=read_csv('results/feature_importance.csv')
    top=read_csv('results/top10_features.csv').copy()
    st.dataframe(top,hide_index=True,width='stretch')
    st.bar_chart(top.set_index('Feature').Gain_share,horizontal=True)
    st.info(f"Ranking model: {meta['best_method_by_original_test_auc']}. Importance is the total split gain grouped by the original engineered field; one-hot categories belonging to one field are added together. 前十名仅展示，不筛掉其他模型输入。")
    st.caption('Gain_share is a share of model split gain, not the fraction of fraud explained. Correlated fields can share importance; ranking does not establish causation. This differs from the discarded ten-raw-field version’s validation permutation ranking.')
    download('Download Top 10 / 下载前十特征',top,'original_top10_features.csv')
    with st.expander('All features and encoding mapping'):
        st.dataframe(rank,hide_index=True,width='stretch')
        st.dataframe(read_csv('results/feature_mapping.csv'),hide_index=True,width='stretch')

elif page=='Original model comparison':
    st.subheader('Original five-method comparison / 原版五组模型比较')
    table=read_csv('results/original_metrics.csv')
    st.dataframe(table,hide_index=True,width='stretch')
    st.bar_chart(table.set_index('Method')[['AUC','Recall','G_mean','KS']])
    pred=read_csv('results/test_predictions.csv')
    curve=pd.DataFrame({'False positive rate':np.linspace(0,1,201)})
    for name in table.Method:
        fpr,tpr,_=roc_curve(pred.Actual.eq('Yes').astype(int),pred[name])
        points=pd.DataFrame({'fpr':fpr,'tpr':tpr}).groupby('fpr').tpr.max()
        curve[name]=np.interp(curve['False positive rate'],points.index,points.values)
    st.subheader('ROC curves')
    st.line_chart(curve,x='False positive rate')
    method=st.selectbox('Show confusion matrix / 查看混淆矩阵',table.Method.tolist())
    labels=np.where(pred[method]>.5,'Yes','No')
    matrix=confusion_matrix(pred.Actual,labels,labels=['No','Yes'])
    st.table(pd.DataFrame(matrix,index=['Actual No','Actual Yes'],columns=['Predicted No','Predicted Yes']))
    st.caption('Original decision rule: score > 0.5. Metrics follow the original formulas, including its KS calculation.')
    st.dataframe(read_csv('results/sampling_summary.csv'),hide_index=True,width='stretch')
    download('Download original metrics / 下载原版评价指标',table,'original_metrics.csv')
    st.warning('Faithful reproduction: original preprocessing occurs before splitting, and the test set is used for early stopping and best-model selection. These results are not an independent held-out estimate. 为保留原版，本次未改变这些评估步骤。')

elif page=='Test case explorer':
    st.subheader('Explore an existing test case / 查看已有测试样本')
    pred=read_csv('results/test_predictions.csv')
    method=st.selectbox('Sampling method / 采样方法',list(meta['model_files']))
    index=st.number_input('Test case number / 测试样本序号',min_value=1,max_value=len(pred),value=1,step=1)
    row=pred.iloc[int(index)-1]
    raw=data.iloc[[int(row.Row_index)]]
    st.dataframe(raw,hide_index=True,width='stretch')
    cols=st.columns(3)
    cols[0].metric('Actual label / 真实标签',str(row.Actual))
    cols[1].metric('Model score / 模型分数',f'{row[method]:.4f}')
    cols[2].metric('Original prediction / 原版预测','Yes' if row[method]>.5 else 'No')
    st.caption('This displays predictions saved from the original test run. It does not train a new model or accept new claims. Scores are not calibrated real-world fraud probabilities.')
    download('Download this test case',raw.assign(Model=method,Risk_score=row[method]),'test_case.csv')

else:
    st.subheader('What changed / 修改范围')
    st.markdown('''**Retained:** original cleaning, every engineered field, one-hot encoding, standardization, 70:30 split, five sampling methods, original XGBoost parameters/early stopping, original metric formulas and prediction rule.

**Added:** CSV reading for the supplied file; result export; native JSON model storage; Windows startup; Streamlit pages and a gain-based Top 10 display.

**Not carried over from the discarded rewrite:** ten-raw-field-only models, separate validation split, extra Baseline model and changed ENN parameters.

**Example for Week 10:** “We asked the coding agent to add a Top 10 feature display while retaining our original analysis. We checked that all original engineered inputs remained, that only the original five methods were shown, and that exported predictions and metrics matched the model run.”

**Your group's checks:** open the Top 10 page, switch the confusion-matrix model, explore a test case, download a result, and record what happened. Do not claim you personally tested it before doing so.

**Original limitations remain visible:** preprocessing uses all data; test labels are used for early stopping and model selection; original KS calculation can separate tied scores. Results depend on package versions. These were retained for reproduction and should be addressed separately if you later want independent evaluation.
''')
    st.json(meta)
    st.markdown('[Streamlit deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)')
