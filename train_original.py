import os
import warnings
import pandas as pd
import numpy as np
import re
import math
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

# 依赖版本兼容导入
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    warnings.warn("未安装 xgboost，模型训练部分将被跳过。请执行: pip install xgboost")

try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.combine import SMOTETomek, SMOTEENN
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    warnings.warn("未安装 imbalanced-learn，采样部分将被跳过。请执行: pip install imbalanced-learn")

import matplotlib
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False

# 设置随机种子
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ---------------------------
# 1. 读取数据
# ---------------------------
from pathlib import Path
ROOT = Path(__file__).resolve().parent
data_path = ROOT / "data" / "vehicle_fraud.csv"   # <-- 如需修改路径，改这里
if not os.path.exists(data_path):
    raise FileNotFoundError(
        f"找不到数据文件: {os.path.abspath(data_path)}\n"
        "请将 Excel 文件放到脚本同级目录，或修改 data_path 变量。"
    )
df = pd.read_csv(data_path)
print(f"数据维度: {df.shape[0]} 行, {df.shape[1]} 列")
fraud_rate = (df["FraudFound"] == "Yes").mean()
print(f"欺诈率: {fraud_rate*100:.2f}%")

# ---------------------------
# 2. 数据清洗
# ---------------------------
def preprocess_data(df_raw):
    data = df_raw.copy()
    if "PolicyNumber" in data.columns:
        data = data.drop("PolicyNumber", axis=1)
        print("已删除 PolicyNumber 列")

    if "Age" in data.columns:
        data.loc[data["Age"] == 0, "Age"] = np.nan

    # 数值列：中位数填充
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if data[col].isna().any():
            med = data[col].median()
            data[col] = data[col].fillna(med)

    # 字符列：众数填充（用 pandas.mode，兼容所有 scipy 版本）
    # 用 exclude=[np.number] 选择非数值列，兼容 pandas 2/3 的 str dtype 迁移，避免 Pandas4Warning
    char_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
    for col in char_cols:
        if data[col].isna().any():
            m = data[col].mode(dropna=True)
            if len(m) > 0:
                data[col] = data[col].fillna(m.iloc[0])

    # 如果第一行是重复表头，则删掉
    if "Month" in data.columns and len(data) > 0:
        first_val = data["Month"].iloc[0]
        if pd.notna(first_val) and str(first_val).strip().lower() == "month":
            data = data.iloc[1:, :].reset_index(drop=True)
    return data.reset_index(drop=True)

# ---------------------------
# 3. 特征工程
# ---------------------------
def feature_engineering(data):
    df = data.copy()
    df["FraudFound"] = df["FraudFound"].astype("category")

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
        "Days:Policy-Accident", "Days:Policy-Claim",
        "PastNumberOfClaims", "NumberOfSuppliments",
        "AddressChange-Claim", "NumberOfCars"
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
    return df

df_clean = preprocess_data(df)
df_featured = feature_engineering(df_clean)

# ---------------------------
# 4. 特征编码（onehot，版本兼容参数）
# ---------------------------
target_col = "FraudFound"
X = df_featured.drop(target_col, axis=1).copy()
y = df_featured[target_col].copy()

categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

for col in categorical_cols:
    # 先填缺失，再统一转成字符串，避免某列混有 int 和 str 导致 OneHotEncoder 报错
    X[col] = X[col].astype(object).fillna("Unknown").map(
        lambda v: str(v) if not isinstance(v, str) else v
    )

# sklearn < 1.2 用 sparse=，>= 1.2 用 sparse_output=
try:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(pd.DataFrame({"a":["x","y"]}))  # 快速探测参数是否被支持
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

# ---------------------------
# 5. 划分训练测试集 7:3
# ---------------------------
X_train_all, X_test, y_train_all, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.3, random_state=SEED, stratify=y_encoded
)
train_df = X_train_all.copy()
train_df[target_col] = y_train_all.values

print("\n训练集大小:", X_train_all.shape[0], "，欺诈率: {:.2f}%".format((y_train_all=="Yes").mean()*100))
print("测试集大小:", X_test.shape[0], "，欺诈率: {:.2f}%".format((y_test=="Yes").mean()*100))

# ---------------------------
# 6. ENN 自定义实现
# ---------------------------
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

# ---------------------------
# 7. 5种采样方法（带容错：任一方法失败只警告，不中断）
# ---------------------------
sampling_methods = []

def safe_resample(name, sampler, X_np, y_bin, columns):
    try:
        X_r, y_r = sampler.fit_resample(X_np, y_bin)
        X_r = pd.DataFrame(X_r, columns=columns)
        y_r = np.where(np.asarray(y_r) == 1, "Yes", "No")
        print(f"{name}后: {X_r.shape[0]} 条，欺诈率: {(y_r=='Yes').mean()*100:.2f}%")
        return {"name": name, "X": X_r, "y": y_r}
    except Exception as e:
        warnings.warn(f"【{name}】采样失败，已跳过: {e}")
        return None

if HAS_IMBLEARN:
    print("\n=== 生成五种采样方法的数据集 ===")
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
            print(f"ENN后: {train_enn.shape[0]} 条，欺诈率: {(train_enn[target_col]=='Yes').mean()*100:.2f}%")
        else:
            warnings.warn("【ENN】清洗后无剩余样本或缺少正类，已跳过")
    except Exception as e:
        warnings.warn(f"【ENN】失败，已跳过: {e}")

    r = safe_resample("SMOTE-Tomek", SMOTETomek(random_state=SEED), X_tr_np, y_tr_bin, X_train_all.columns)
    if r: sampling_methods.append(r)
    r = safe_resample("ADASYN", ADASYN(random_state=SEED), X_tr_np, y_tr_bin, X_train_all.columns)
    if r: sampling_methods.append(r)
    r = safe_resample("SMOTE-ENN", SMOTEENN(random_state=SEED), X_tr_np, y_tr_bin, X_train_all.columns)
    if r: sampling_methods.append(r)
    print("\n=== 采样方法准备完成 ===")

# ---------------------------
# 8. 评估指标：Gmean，KS
# ---------------------------
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

# ---------------------------
# 9. 训练 XGBoost
# ---------------------------
trained_models = {}
saved_predictions = {}
results_list = []
auc_results = []
roc_data = {}

if HAS_XGB and len(sampling_methods) > 0:
    y_test_bin = np.where(y_test == "Yes", 1, 0)
    X_test_arr = X_test.values

    def train_xgb(X_tr, y_tr):
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

    print("\n=== 开始训练 XGBoost 模型 ===")
    for method in sampling_methods:
        m_name = method["name"]
        print(f"\n--- 训练方法: {m_name} ---")
        try:
            model, dtest = train_xgb(method["X"], method["y"])
            trained_models[m_name] = model
            y_prob = model.predict(dtest)
            saved_predictions[m_name] = y_prob
            y_pred = np.where(y_prob > 0.5, "Yes", "No")

            auc_val = roc_auc_score(y_test_bin, y_prob)
            cm = confusion_matrix(y_test, y_pred, labels=["No","Yes"])
            tn, fp, fn, tp = cm.ravel()
            recall = tp/(tp+fn) if (tp+fn) > 0 else 0
            gmean = calculate_gmean(y_test, y_pred)
            ks_val = calculate_ks(y_test, y_prob)

            results_list.append({"Method": m_name, "AUC": round(auc_val,4),
                                 "Recall": round(recall,4), "G_mean": round(gmean,4), "KS": round(ks_val,4)})
            fpr, tpr, _ = roc_curve(y_test_bin, y_prob)
            roc_data[m_name] = {"fpr": fpr, "tpr": tpr, "auc": auc_val}
            print(f"【{m_name}】AUC: {auc_val:.4f} | Recall: {recall:.4f} | G-mean: {gmean:.4f} | KS: {ks_val:.4f}")
        except Exception as e:
            warnings.warn(f"【{m_name}】训练失败，已跳过: {e}")

results_df = pd.DataFrame(results_list)
print("\n=== 结果对比 ===")
print(results_df if len(results_df) > 0 else "（没有成功训练的模型）")

if len(results_df) > 0:
    best = results_df.loc[results_df["AUC"].idxmax()]
    print(f"\n★ 最优AUC值：{best['AUC']:.4f}（采样方法：{best['Method']}）")


# Export only; original cleaning, feature engineering, sampling and fitting above are unchanged.
import json
import joblib
out = ROOT / 'results'
out.mkdir(exist_ok=True)
if len(trained_models) != 5:
    raise RuntimeError(f'Expected five original methods, got {list(trained_models)}')
results_df.to_csv(out/'original_metrics.csv', index=False)
predictions = pd.DataFrame({'Row_index': X_test.index, 'Actual': y_test.values})
for name, scores in saved_predictions.items():
    predictions[name] = scores
predictions.to_csv(out/'test_predictions.csv', index=False)
pd.DataFrame({'Row_index': list(X_train_all.index)+list(X_test.index),
              'Split': ['train']*len(X_train_all)+['test']*len(X_test)}).to_csv(out/'split_indices.csv', index=False)
pd.DataFrame([{'Method': v['name'], 'Rows': len(v['y']), 'Fraud_rows': int(np.sum(v['y']=='Yes'))} for v in sampling_methods]).to_csv(out/'sampling_summary.csv', index=False)
model_files = {}
for i, (name, model) in enumerate(trained_models.items()):
    file = f'xgb_{i}.json'
    model.save_model(out/file)
    model_files[name] = file
# Category indicators are grouped under their complete engineered field.
source_fields = []
encoder = preprocessor.named_transformers_['cat']
for field, categories in zip(categorical_cols, encoder.categories_):
    source_fields.extend([field]*len(categories))
source_fields.extend(numeric_cols)
assert len(source_fields) == len(feature_names)
field_map = dict(zip(feature_names, source_fields))
best_method = str(results_df.loc[results_df.AUC.idxmax(), 'Method'])
gains = trained_models[best_method].get_score(importance_type='total_gain')
# DMatrix was originally created from numpy arrays, so Booster names are f0, f1, ... .
rows = []
for i, field in enumerate(source_fields):
    rows.append({'Feature': field, 'Total_gain': float(gains.get(f'f{i}', 0))})
rank = pd.DataFrame(rows).groupby('Feature', as_index=False).Total_gain.sum()
rank = rank.sort_values(['Total_gain', 'Feature'], ascending=[False, True]).reset_index(drop=True)
rank.insert(0, 'Rank', np.arange(1, len(rank)+1))
rank['Gain_share'] = rank.Total_gain/rank.Total_gain.sum()
rank.to_csv(out/'feature_importance.csv', index=False)
rank.head(10).to_csv(out/'top10_features.csv', index=False)
pd.DataFrame({'Encoded_feature': feature_names, 'Engineered_field': source_fields}).to_csv(out/'feature_mapping.csv', index=False)
joblib.dump({'preprocessor':preprocessor, 'scaler':scaler, 'categorical_cols':categorical_cols,
             'numeric_cols':numeric_cols}, out/'preprocessing.joblib', compress=3)
metadata = {'rows':len(df), 'fraud_rows':int((df.FraudFound=='Yes').sum()),
            'train_rows':len(X_train_all), 'test_rows':len(X_test),
            'engineered_fields':len(X.columns), 'encoded_fields':len(feature_names),
            'best_method_by_original_test_auc':best_method, 'model_files':model_files,
            'ranking_method':'Grouped total split gain from the best original model',
            'top10_display_only':True,
            'original_limitations':['Preprocessing fitted before train/test split',
                                    'Test set used for early stopping and best-method selection',
                                    'Prediction uses all trees in the original saved Booster, as original code',
                                    'Original KS calculation may split tied scores'],
            'changes':['CSV input path', 'Native JSON model export', 'Result export and UI only']}
(out/'metadata.json').write_text(json.dumps(metadata, indent=2))
print(rank.head(10).to_string(index=False))
