# %%
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
import matplotlib.pyplot as plt

# 设置日志
logging.basicConfig(level=logging.INFO, format="%Y-%m-%d %H:%M:%S - %(levelname)s - %(message)s")

# %%
def load_data(file_path:str):
    logging.info(f"尝试读取文件: {file_path}")
    df = pd.read_csv(file_path)
    logging.info(f"原始数据行数:{df.shape[0]}, 列数:{df.shape[1]}")
    return df

# %%
def clean_data(df:pd.DataFrame):
    logging.info("开始数据清洗")
    df_clean = df.copy()
    df_clean = df_clean.dropna()
    logging.info(f"清洗后数据行数:{df_clean.shape[0]}")
    return df_clean

# %%
def feature_engineering(df:pd.DataFrame, target_col="FraudFound"):
    # 标签：FraudFound Yes/No 转为1/0
    y = df[target_col].map({"Yes":1, "No":0})
    X = df.drop(columns=[target_col])

    # 区分分类变量、数值变量
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64","float64"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])
    return X,y,preprocessor

# %%
def train_model(X,y,preprocessor,sample_method="none"):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

    if sample_method == "smote":
        sampler = SMOTE(random_state=42)
        X_train, y_train = sampler.fit_resample(preprocessor.fit_transform(X_train), y_train)
        X_test = preprocessor.transform(X_test)
    elif sample_method == "under":
        sampler = RandomUnderSampler(random_state=42)
        X_train, y_train = sampler.fit_resample(preprocessor.fit_transform(X_train), y_train)
        X_test = preprocessor.transform(X_test)
    else:
        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)

    model = XGBClassifier(random_state=42,use_label_encoder=False,eval_metric="logloss")
    model.fit(X_train,y_train)
    y_pred_prob = model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, y_pred_prob)
    return model, X_test, y_test, y_pred_prob, auc

# %%
def plot_result(results):
    plt.figure(figsize=(10,4))
    # ROC图
    plt.subplot(1,2,1)
    for name,data in results.items():
        fpr,tpr,_ = roc_curve(data["y_true"], data["y_pred"])
        auc = data["auc"]
        plt.plot(fpr,tpr,label=f"{name}, AUC={auc:.3f}")
    plt.plot([0,1],[0,1],"k--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()

    # AUC柱状对比
    plt.subplot(1,2,2)
    names = list(results.keys())
    aucs = [results[n]["auc"] for n in names]
    plt.bar(names,aucs)
    plt.ylim(0,1)
    plt.ylabel("AUC")
    plt.title("AUC对比")
    plt.tight_layout()
    plt.savefig("/content/roc_auc_plot.png",dpi=300)
    plt.show()

# %%
def run_pipeline(file_path):
    logging.info("===== 启动完整保险欺诈检测流水线 =====")
    df = load_data(file_path)
    df_clean = clean_data(df)
    X,y,preprocessor = feature_engineering(df_clean, target_col="FraudFound")

    results = {}
    for method_name in ["none","smote","under"]:
        logging.info(f"正在训练: {method_name}")
        model, X_test, y_test, y_pred_prob, auc = train_model(X,y,preprocessor,method_name)
        results[method_name] = {"y_true":y_test, "y_pred":y_pred_prob, "auc":auc}
        print(f"{method_name} AUC = {auc:.4f}")

    plot_result(results)
    logging.info("流水线全部结束，图片已保存")
    return results
