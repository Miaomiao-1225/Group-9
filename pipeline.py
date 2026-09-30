import os
import warnings
import pandas as pd
import numpy as np
import re
import math
import random
import logging

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

# 依赖兼容
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    warnings.warn("未安装 xgboost，模型训练部分将被跳过。")

try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.combine import SMOTETomek, SMOTEENN
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    warnings.warn("未安装 imbalanced‑learn，采样部分将被跳过。")

import matplotlib
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
target_col = "FraudFound"


def load_data(data_path: str = "Vehicle Insurance Fraud Detection.xlsx") -> pd.DataFrame:
    """加载数据集"""
    logging.info(f"尝试读取文件: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"找不到文件 {os.path.abspath(data_path)}")
    df = pd.read_excel(data_path)
    logging.info(f"原始数据行数:{df.shape[0]},列数:{df.shape[1]}")
    fraud_rate = (df[target_col] == "Yes").mean()
    logging.info(f"原始欺诈率: {fraud_rate*100:.2f}%")
    return df


def preprocess_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """数据清洗"""
    data = df_raw.copy()
    if "PolicyNumber" in data.columns:
        data = data.drop("PolicyNumber", axis=1)
        logging.info("已删除 PolicyNumber 列")

    if "Age" in data.columns:
        data.loc[data["Age"] == 0, "Age"] = np.nan

    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if data[col].isna().any():
            med = data[col].median()
            data[col] = data[col].fillna(med)

    char_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
    for col in char_cols:
        if data[col].isna().any():
            m = data[col].mode(dropna=True)
            if len(m) > 0:
                data[col] = data[col].fillna(m.iloc[0])

    if "Month" in data.columns and len(data) > 0:
        first_val = data["Month"].iloc[0]
        if pd.notna(first_val) and str(first_val).strip().lower() == "month":
            data = data.iloc[1:, :].reset_index(drop=True)
            logging.info("检测并移除重复表头行")
    return data.reset_index(drop=True)


def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    """特征工程"""
    df = data.copy()
    df[target_col] = df[target_col].astype("category")

    def convert_range_to_numeric(x):
        res = np.full(len(x), np.nan)
        for i, val in enumerate(x):
            if pd.isna(val):
                continue
            s = str(val).strip().lower()
            if s in ["none", "na", ""]:
                res[i] = 0
                continue
            if "more than" in s:
                nums = re.findall(r"\d+", s)
                if len(nums) > 0:
                    res[i] = int(nums[0]) + 1
                continue
            if "to" in s and re.search(r"\d", s):
                nums = [int(v) for v in re.findall(r"-?\d+", s)]
                if len(nums) >= 2:
                    res[i] = np.mean(nums[:2])
                continue
            if "year" in s:
                nums = re.findall(r"\d+", s)
                if len(nums) > 0:
                    res[i] = int(nums[0])
                continue
            try:
                res[i] = float(val)
            except (ValueError, TypeError):
                pass
        return res

    range_cols = [
        "AgeOfVehicle", "AgeOfPolicyHolder", "VehiclePrice",
        "Days:Policy‑Accident", "Days:Policy‑Claim",
        "PastNumberOfClaims", "NumberOfSuppliments",
        "AddressChange‑Claim", "NumberOfCars"
    ]
    for col in range_cols:
        if col in df.columns:
            new_col = col + "_num"
            df[new_col] = convert_range_to_numeric(df[col])
            if df[new_col].isna().any():
                med = df[new_col].median()
                df[new_col] = df[new_col].fillna(med)

    month_map = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
                 "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}
    if "Month" in df.columns:
        df["AccidentMonth_num"] = df["Month"].map(month_map)
    if "MonthClaimed" in df.columns:
        df["ClaimMonth_num"] = df["MonthClaimed"].map(month_map)
    if "AccidentMonth_num" in df.columns and "ClaimMonth_num" in df.columns:
        df["MonthDiff"] = (df["ClaimMonth_num"] - df["AccidentMonth_num"]).abs()
        df["MonthDiff"] = df["MonthDiff"].fillna(df["MonthDiff"].median())

    if "DayOfWeek" in df.columns:
        df["IsWeekend"] = df["DayOfWeek"].isin(["Saturday","Sunday"]).astype(int)
    if "DayOfWeekClaimed" in df.columns:
        df["IsWeekendClaim"] = df["DayOfWeekClaimed"].isin(["Saturday","Sunday"]).astype(int)

    if "Age" in df.columns:
        bins = [0,25,35,50,65,100]
        labels = ["Young","Adult","Middle","Senior","Elderly"]
        df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels, right=True)

    if "Make" in df.columns:
        luxury_brands = ["Accura","BMW","Jaguar","Porche","Mercedes","Lexus"]
        df["IsLuxuryBrand"] = df["Make"].isin(luxury_brands).astype(int)

    if "PolicyType" in df.columns:
        df["PolicyTypeSimple"] = df["PolicyType"].astype(str).str.split(" - ").str[0]

    drop_cols = [c for c in range_cols if c in df.columns]
    df = df.drop(drop_cols, axis=1)
    logging.info("特征工程完成")
    return df


def encode_scale(df_featured: pd.DataFrame):
    """onehot编码+标准化"""
    X = df_featured.drop(target_col, axis=1).copy()
    y = df_featured[target_col].copy()

    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    for col in categorical_cols:
        X[col] = X[col].astype(object).fillna("Unknown").map(
            lambda v: str(v) if not isinstance(v, str) else v
        )
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoder.fit(pd.DataFrame({"a":["x","y"]}))
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[("cat", encoder, categorical_cols)],
        remainder="passthrough"
    )
    X_encoded_raw = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()
    X_encoded = pd.DataFrame(X_encoded_raw, columns=feature_names, index=X.index).fillna(0)
    y_encoded = y.reset_index(drop=True)

    scaler = StandardScaler()
    X_scaled_arr = scaler.fit_transform(X_encoded)
    X_scaled = pd.DataFrame(X_scaled_arr, columns=X_encoded.columns)
    logging.info("特征编码与标准化完成")
    return X_scaled, y_encoded


def split_dataset(X_scaled, y_encoded):
    X_train_all, X_test, y_train_all, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.3, random_state=SEED, stratify=y_encoded
    )
    train_df = X_train_all.copy()
    train_df[target_col] = y_train_all.values
    logging.info(f"训练集大小:{X_train_all.shape[0]},测试集大小:{X_test.shape[0]}")
    return X_train_all, X_test, y_train_all, y_test, train_df


def apply_enn_standalone(data_df, k=3):
    feature_cols = [c for c in data_df.columns if c != target_col]
    X_mat = data_df[feature_cols].values
    y_ser = data_df[target_col]
    y_num = np.where(y_ser == "Yes", 1, 0)
    n = X_mat.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        xi = X_mat[i, :]
        dist = np.sqrt(np.sum((X_mat - xi) ** 2, axis=1))
        dist[i] = np.inf
        nn_idx = np.argsort(dist)[:k]
        neighbor_labels = y_num[nn_idx]
        majority = 1 if np.mean(neighbor_labels) > 0.5 else 0
        if majority != y_num[i]:
            keep[i] = False
    return data_df[keep].reset_index(drop=True)


def safe_resample(name, sampler, X_np, y_bin, columns):
    try:
        X_r, y_r = sampler.fit_resample(X_np, y_bin)
        X_r = pd.DataFrame(X_r, columns=columns)
        y_r = np.where(np.asarray(y_r) == 1, "Yes", "No")
        logging.info(f"{name}后样本数:{X_r.shape[0]},欺诈率:{(y_r=='Yes').mean()*100:.2f}%")
        return {"name": name, "X": X_r, "y": y_r}
    except Exception as e:
        warnings.warn(f"【{name}】采样失败，已跳过: {e}")
        return None


def get_sampling_methods(X_train_all, y_train_all, train_df):
    sampling_methods = []
    if not HAS_IMBLEARN:
        return sampling_methods
    logging.info("开始生成5种重采样数据集")
    X_tr_np = X_train_all.values
    y_tr_bin = np.where(y_train_all == "Yes", 1, 0)

    r = safe_resample("SMOTE", SMOTE(random_state=SEED), X_tr_np, y_tr_bin, X_train_all.columns)
    if r: sampling_methods.append(r)

    try:
        train_enn = apply_enn_standalone(train_df, k=3)
        if len(train_enn) > 0 and (train_enn[target_col] == "Yes").sum() > 0:
            sampling_methods.append({
                "name": "ENN",
                "X": train_enn.drop(target_col, axis=1),
                "y": train_enn[target_col].values
            })
    except Exception as e:
        warnings.warn(f"【ENN】失败，已跳过: {e}")

    r = safe_resample("SMOTE‑Tomek", SMOTETomek(random_state=SEED), X_tr_np, y_tr_bin, X_train_all.columns)
    if r: sampling_methods.append(r)
    r = safe_resample("ADASYN", ADASYN(random_state=SEED), X_tr_np, y_tr_bin, X_train_all.columns)
    if r: sampling_methods.append(r)
    r = safe_resample("SMOTE‑ENN", SMOTEENN(random_state=SEED), X_tr_np, y_tr_bin, X_train_all.columns)
    if r: sampling_methods.append(r)
    logging.info(f"成功加载 {len(sampling_methods)} 种采样方法")
    return sampling_methods


def calculate_gmean(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=["No","Yes"])
    tn, fp, fn, tp = cm.ravel()
    tpr = tp/(tp+fn) if (tp+fn) > 0 else 0
    tnr = tn/(tn+fp) if (tn+fp) > 0 else 0
    return math.sqrt(tpr * tnr)


def calculate_ks(y_true, y_prob):
    y_num = np.where(np.asarray(y_true) == "Yes", 1, 0)
    df_ks = pd.DataFrame({"prob": y_prob, "label": y_num}).sort_values("prob").reset_index(drop=True)
    n_pos = np.sum(y_num)
    n_neg = len(y_num) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    cum_pos = np.cumsum(df_ks["label"]) / n_pos
    cum_neg = np.cumsum(1 - df_ks["label"]) / n_neg
    return float(np.max(np.abs(cum_pos - cum_neg)))


def train_xgb_wrapper(X_tr, y_tr, X_test_arr, y_test_bin):
    y_tr_bin = np.where(np.asarray(y_tr) == "Yes", 1, 0)
    n_pos = np.sum(y_tr_bin == 1)
    n_neg = np.sum(y_tr_bin == 0)
    pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    dtrain = xgb.DMatrix(np.asarray(X_tr), label=y_tr_bin)
    dtest = xgb.DMatrix(X_test_arr, label=y_test_bin)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 6,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "scale_pos_weight": pos_weight,
        "seed": 123,
        "nthread": 1
    }
    watchlist = [(dtrain, "train"), (dtest, "test")]
    model = xgb.train(params, dtrain, num_boost_round=200,
                      evals=watchlist, early_stopping_rounds=20, verbose_eval=False)
    return model, dtest


def train_evaluate_all(sampling_methods, X_test, y_test):
    results_list = []
    roc_data = {}
    if not HAS_XGB or len(sampling_methods) == 0:
        return pd.DataFrame(results_list), roc_data
    y_test_bin = np.where(y_test == "Yes", 1, 0)
    X_test_arr = X_test.values
    logging.info("开始XGBoost多模型训练")
    for method in sampling_methods:
        m_name = method["name"]
        try:
            model, dtest = train_xgb_wrapper(method["X"], method["y"], X_test_arr, y_test_bin)
            y_prob = model.predict(dtest)
            y_pred = np.where(y_prob > 0.5, "Yes", "No")
            auc_val = roc_auc_score(y_test_bin, y_prob)
            cm = confusion_matrix(y_test, y_pred, labels=["No","Yes"])
            tn, fp, fn, tp = cm.ravel()
            recall = tp/(tp+fn) if (tp+fn) > 0 else 0
            gmean = calculate_gmean(y_test, y_pred)
            ks_val = calculate_ks(y_test, y_prob)
            results_list.append({
                "Method": m_name,
                "AUC": round(auc_val,4),
                "Recall": round(recall,4),
                "G_mean": round(gmean,4),
                "KS": round(ks_val,4)
            })
            fpr, tpr, _ = roc_curve(y_test_bin, y_prob)
            roc_data[m_name] = {"fpr": fpr, "tpr": tpr, "auc": auc_val}
            logging.info(f"{m_name} -> AUC:{auc_val:.4f} Recall:{recall:.4f}")
        except Exception as e:
            warnings.warn(f"{m_name}训练失败:{e}")
    results_df = pd.DataFrame(results_list)
    return results_df, roc_data


def set_chinese_font():
    from matplotlib import font_manager
    candidates = ["Arial Unicode MS", "PingFang SC", "Microsoft YaHei",
                  "SimHei", "Noto Sans CJK SC", "WenQuanYi Micro Hei"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for f in candidates:
        if f in available:
            plt.rcParams["font.sans‑serif"] = [f]
            break
    plt.rcParams["axes.unicode_minus"] = False


def draw_plots(results_df, roc_data):
    set_chinese_font()
    if HAS_SNS and len(results_df) > 0:
        results_long = pd.melt(results_df, id_vars=["Method"], var_name="Metric", value_name="Value")
        g1 = sns.catplot(data=results_long, x="Method", y="Value", hue="Method",
                         col="Metric", kind="bar", sharey=False, palette="Set2", legend=False)
        g1.fig.suptitle("五种采样方法 XGBoost 性能对比", y=1.03)
        for ax in g1.axes.flat:
            ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        plt.savefig("metric_bar.png", dpi=150)
        logging.info("已保存指标柱状图 metric_bar.png")

    if len(roc_data) > 0:
        colors = {"SMOTE":"#E41A1C","ENN":"#377EB8","SMOTE‑Tomek":"#4DAF4A",
                  "ADASYN":"#FF7F00","SMOTE‑ENN":"#984EA3"}
        linestyles = {"SMOTE":"solid","ENN":"dashed","SMOTE‑Tomek":"dotted",
                      "ADASYN":"dashdot","SMOTE‑ENN":"solid"}
        plt.figure(figsize=(10,8))
        for name, info in roc_data.items():
            plt.plot(info["fpr"], info["tpr"], color=colors.get(name,"black"),
                     linestyle=linestyles.get(name,"solid"), lw=1.2,
                     label=f"{name}, AUC={info['auc']:.4f}")
        plt.plot([0,1],[0,1], color="gray", linestyle="--", alpha=0.7)
        plt.xlabel("假正例率 (1 - Specificity)")
        plt.ylabel("真正例率 (Sensitivity)")
        plt.title("五种采样方法 XGBoost ROC曲线对比")
        plt.legend(loc="lower right")
        plt.savefig("roc_curve.png", dpi=150)
        logging.info("已保存ROC曲线图 roc_curve.png")


def run_full_pipeline(data_path="Vehicle Insurance Fraud Detection.xlsx"):
    """完整流水线入口函数"""
    df = load_data(data_path)
    df_clean = preprocess_data(df)
    df_featured = feature_engineering(df_clean)
    X_scaled, y_encoded = encode_scale(df_featured)
    X_train_all, X_test, y_train_all, y_test, train_df = split_dataset(X_scaled, y_encoded)
    sampling_methods = get_sampling_methods(X_train_all, y_train_all, train_df)
    results_df, roc_data = train_evaluate_all(sampling_methods, X_test, y_test)
    if len(results_df) > 0:
        best = results_df.loc[results_df["AUC"].idxmax()]
        logging.info(f"最优AUC:{best['AUC']},采样方法:{best['Method']}")
    draw_plots(results_df, roc_data)
    return results_df, roc_data
