import pandas as pd
import numpy as np
import math
import random
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import xgboost as xgb
import matplotlib.pyplot as plt
import logging

def load_data(data_path: str = "Vehicle Insurance Fraud Detection.csv") -> pd.DataFrame:
    """加载数据集"""
    logging.info(f"尝试读取文件: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"找不到文件 {os.path.abspath(data_path)}")
    df = pd.read_csv(data_path)
    logging.info(f"原始数据行数:{df.shape[0]},列数:{df.shape[1]}")
    return df

def preprocess_data(df:pd.DataFrame) -> pd.DataFrame:
    """数据清洗"""
    logging.info("开始数据清洗")
    df = df.copy()
    df = df.dropna()
    df = df.drop_duplicates()
    logging.info(f"清洗后数据行数:{df.shape[0]}")
    return df

def feature_engineering(df:pd.DataFrame, target_col="fraud"):
    """特征工程，划分X,y，区分类别/数值特征"""
    y = df[target_col]
    X = df.drop(columns=[target_col])
    cat_cols = X.select_dtypes(include=["object","category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64","float64"]).columns.tolist()
    logging.info(f"类别特征：{cat_cols}")
    logging.info(f"数值特征：{num_cols}")
    return X,y,cat_cols,num_cols

def get_samplers():
    """定义多种采样策略"""
    sampler_dict = {
        "NoSampler": None,
        "RandomUnderSample": RandomUnderSampler(random_state=42),
        "SMOTE": SMOTE(random_state=42),
        "SMOTEENN": SMOTEENN(random_state=42)
    }
    return sampler_dict

def train_and_evaluate(X,y,cat_cols,num_cols,sampler=None):
    """训练XGBoost并评估"""
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    if sampler is not None:
        X_train, y_train = sampler.fit_resample(X_train,y_train)
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])
    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("xgb", xgb.XGBClassifier(random_state=42,use_label_encoder=False,eval_metric="logloss"))
    ])
    model.fit(X_train,y_train)
    y_pred_proba = model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test,y_pred_proba)
    return auc, model, y_test, y_pred_proba

def plot_results(result_dict):
    """绘图，保存图片"""
    sampler_names = list(result_dict.keys())
    auc_scores = [result_dict[n]["auc"] for n in sampler_names]
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.bar(sampler_names, auc_scores)
    plt.title("AUC of different sampling strategies")
    plt.ylabel("AUC Score")
    plt.xticks(rotation=15)
    plt.subplot(1,2,2)
    for name,data in result_dict.items():
        fpr,tpr,_ = roc_curve(data["y_true"],data["y_pred"])
        plt.plot(fpr,tpr,label=f"{name}, AUC={data['auc']:.3f}")
    plt.plot([0,1],[0,1],"k--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_auc_plot.png",dpi=300)
    plt.close()
    logging.info("图片已保存为 roc_auc_plot.png")

def run_full_pipeline(data_path:str="Vehicle Insurance Fraud Detection.csv"):
    logging.info("==== 完整流水线启动 ====")
    df = load_data(data_path)
    df_clean = preprocess_data(df)
    X,y,cat_cols,num_cols = feature_engineering(df_clean, target_col="fraud")
    samplers = get_samplers()
    result_store = {}
    for name, sampler in samplers.items():
        logging.info(f"正在训练：{name}")
        auc, model, y_true, y_pred = train_and_evaluate(X,y,cat_cols,num_cols,sampler)
        result_store[name] = {"auc":auc,"model":model,"y_true":y_true,"y_pred":y_pred}
        logging.info(f"{name} 完成，AUC={auc:.4f}")
    plot_results(result_store)
    result_df = pd.DataFrame([{"Sampling":k,"AUC":v["auc"]} for k,v in result_store.items()])
    return result_df, result_store
