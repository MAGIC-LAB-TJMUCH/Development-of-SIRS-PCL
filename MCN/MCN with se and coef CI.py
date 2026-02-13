# -*- coding: utf-8 -*-
"""
基于已有MCN模型生成训练集评价指标、混淆矩阵、ROC曲线和系数SE

直接读取已保存的模型和变量筛选结果，使用Rubin规则计算系数标准误
适用于5折交叉验证的MCN风险分级模型
"""

import pandas as pd
import numpy as np
import os
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                             classification_report, roc_auc_score)
from sklearn.utils import resample
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ===================== 全局设置 =====================
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 路径配置 =====================
# 【请根据实际路径修改】
# 已有结果路径
existing_results_path = r"G:\胰腺囊性病变分类与风险预测_1.2\MCN风险结果\MCN_Grade_Final_Results_ExtStd"

# 数据路径
base_path = r"E:\Pancreatic cancer\多中心胰腺囊性病变诊断与风险预测\完整数据集-五分类建模\MICE_model"

# 标准化参数文件路径
std_params_path = os.path.join(base_path, "Standardization_Params_MCN", "Standardization_Parameters.xlsx")

# 输出路径（保存到已有结果目录）
output_path = existing_results_path

# 核心参数
target_col = "Grade"
filter_col = "Dignosis"
filter_value = "MCN"
id_col = "key"
m = 10
n_splits = 5  # 5折交叉验证
seed = 3420
BOOTSTRAP_N = 1000
CI_LEVEL = 0.95

# 类别标签
labels = ['Benign/Low Risk', 'Medium Risk', 'High Risk']

# 变量清单
raw_categorical_vars = [
    "Gender", "Cyst wall thickness", "Uniform Cyst wall", "Cyst wall enhancement",
    "Mural nodule status", "Mural nodule enhancement", "Solid component enhancement",
    "Intracystic septations", "Uniform Septations", "Intracystic septa enhancement", "Capsule",
    "Main PD communication", "Pancreatic parenchymal atrophy", "MPD dilation",
    "Mural nodule in MPD", "Common bile duct dilation", "Vascular abutment", "Enlarged lymph nodes",
    "Distant metastasis", "Tumor lesion", "Lesion_Head_neck", "Lesion_body_tail", "Diabetes",
    "Jaundice"
]

num_vars = [
    "Short diameter of lesion (mm)",
    "Short diameter of solid component (mm)",
    "Short diameter of largest mural nodule (mm)",
    "CA-199", "CEA", "Age"
]

whitelist = [target_col, filter_col, id_col] + raw_categorical_vars + num_vars


# ===================== 标准化工具函数 =====================
def load_standardization_params(filepath):
    """加载标准化参数文件"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"未找到标准化参数文件: {filepath}")
    print(f"正在加载标准化参数: {filepath}")
    params_df = pd.read_excel(filepath)
    params_df.columns = [c.capitalize() for c in params_df.columns]
    params_dict = {}
    for _, row in params_df.iterrows():
        params_dict[row['Feature']] = {'mean': row['Mean'], 'std': row['Std']}
    return params_dict


def apply_external_standardization(df, numeric_vars, std_dict):
    """仅对数值型变量进行标准化，OHE变量保持不变"""
    df_out = df.copy()
    for col in df_out.columns:
        if col in numeric_vars and col in std_dict:
            mu = std_dict[col]['mean']
            sigma = std_dict[col]['std']
            if sigma == 0:
                sigma = 1e-8
            df_out[col] = (df_out[col] - mu) / sigma
    return df_out


# ===================== 数据预处理函数 =====================
def dynamic_preprocess_mcn(df):
    """MCN数据预处理"""
    df_sub = df[df[filter_col] == filter_value].copy()
    available_cols = [c for c in whitelist if c in df_sub.columns]
    df_clean = df_sub[available_cols].copy()

    # Grade编码
    df_clean[target_col] = df_clean[target_col].astype(str).str.strip().str.lower()
    grade_map = {
        'low risk': 0, 'benign': 0, 'lowrisk': 0,
        'medium risk': 1, 'medium': 1, 'mediumrisk': 1,
        'high risk': 2, 'high': 2, 'highrisk': 2,
    }
    df_clean[target_col] = df_clean[target_col].map(grade_map)
    df_clean = df_clean.dropna(subset=[target_col])
    df_clean[target_col] = df_clean[target_col].astype(int)

    # Enhancement变量处理
    enhancement_vars = ["Cyst wall enhancement", "Mural nodule enhancement",
                        "Solid component enhancement", "Intracystic septa enhancement"]
    phase_suffixes = ["Arterial Phase Enhancement", "Delayed enhancement"]

    for target_var in enhancement_vars:
        if target_var not in df_clean.columns:
            continue
        phase_cols = [target_var.replace(" enhancement", s) for s in phase_suffixes]
        existing_phases = [col for col in phase_cols if col in df.columns]

        orig = df_clean[target_var].astype(str).str.strip().str.lower()
        standard_map = {
            'absent cyst wall': 'Absent cyst wall',
            'arterial phase enhancement': 'Enhancement',
            'delayed enhancement': 'Enhancement',
            'no enhancement': 'No enhancement',
            'absence of solid tissue': 'Absence of solid tissue',
            'absent septations': 'Absent septations',
            'no mural nodule': 'No mural nodule'
        }
        cleaned_orig = orig.map(standard_map).fillna(orig)

        if existing_phases:
            phase_data = df.loc[df_clean.index, existing_phases].copy()
            arterial_present = pd.Series(False, index=df_clean.index)
            delayed_present = pd.Series(False, index=df_clean.index)

            if any("Arterial" in c for c in existing_phases):
                arterial_col = [c for c in existing_phases if "Arterial" in c][0]
                arterial_present = phase_data[arterial_col].astype(str).str.lower().str.strip().isin(
                    ['yes', '1', 'present'])

            if any("Delayed" in c for c in existing_phases):
                delayed_col = [c for c in existing_phases if "Delayed" in c][0]
                delayed_present = phase_data[delayed_col].astype(str).str.lower().str.strip().isin(
                    ['yes', '1', 'present'])

            no_enhancement_keywords = ['no enhancement', 'absent', 'no mural nodule', 'absence of solid tissue',
                                       'absent septations', np.nan]
            need_fill = cleaned_orig.isin(no_enhancement_keywords) | cleaned_orig.isna()

            final_value = cleaned_orig.copy()
            final_value[need_fill & arterial_present] = 'Arterial Phase Enhancement'
            final_value[need_fill & delayed_present & ~arterial_present] = 'Delayed enhancement'
            df_clean[target_var] = final_value
        else:
            df_clean[target_var] = cleaned_orig

        df_clean[target_var] = df_clean[target_var].astype(str)

    # 分类变量处理
    for var in raw_categorical_vars:
        if var not in df_clean.columns:
            continue
        unique_vals = df_clean[var].dropna().unique()
        if len(unique_vals) <= 2:
            mapping = {'male': 0, 'female': 1, 'no': 0, 'yes': 1}
            df_clean[var] = df_clean[var].astype(str).str.lower().str.strip().map(lambda x: mapping.get(x, 0))
            df_clean[var] = df_clean[var].fillna(0).astype(float)
        else:
            df_clean = pd.get_dummies(df_clean, columns=[var], drop_first=False, dtype=float)

    return df_clean


# ===================== 计算逻辑回归系数SE的函数 =====================
def compute_logistic_se_bootstrap(X, y, model, n_bootstrap=200):
    """使用Bootstrap方法计算多项逻辑回归系数的标准误"""
    n_samples = X.shape[0]
    n_classes = len(model.classes_)

    bootstrap_coefs = []
    bootstrap_intercepts = []

    np.random.seed(seed)
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X[indices]
        y_boot = y[indices]

        if len(np.unique(y_boot)) < n_classes:
            continue

        model_boot = LogisticRegression(
            penalty='l2', solver='lbfgs', multi_class='multinomial',
            max_iter=5000, random_state=seed
        )
        try:
            model_boot.fit(X_boot, y_boot)
            bootstrap_coefs.append(model_boot.coef_.flatten())
            bootstrap_intercepts.append(model_boot.intercept_)
        except:
            continue

    if len(bootstrap_coefs) > 10:
        coef_se = np.std(bootstrap_coefs, axis=0, ddof=1)
        intercept_se = np.std(bootstrap_intercepts, axis=0, ddof=1)
    else:
        coef_se = np.full(model.coef_.flatten().shape, np.nan)
        intercept_se = np.full(model.intercept_.shape, np.nan)

    return coef_se, intercept_se


def rubin_pooling_with_se(all_coefs, all_intercepts, all_coef_vars, all_intercept_vars, m):
    """使用Rubin规则池化系数和方差"""
    pooled_coef = np.mean(all_coefs, axis=0)
    pooled_intercept = np.mean(all_intercepts, axis=0)

    W_coef = np.mean(all_coef_vars, axis=0)
    W_intercept = np.mean(all_intercept_vars, axis=0)

    B_coef = np.var(all_coefs, axis=0, ddof=1)
    B_intercept = np.var(all_intercepts, axis=0, ddof=1)

    T_coef = W_coef + (1 + 1 / m) * B_coef
    T_intercept = W_intercept + (1 + 1 / m) * B_intercept

    pooled_coef_se = np.sqrt(T_coef)
    pooled_intercept_se = np.sqrt(T_intercept)

    return pooled_coef, pooled_intercept, pooled_coef_se, pooled_intercept_se


# ===================== Bootstrap置信区间计算函数 =====================
def bootstrap_metric_ci(y_true, y_pred, y_prob, class_labels, n_bootstrap=1000, ci_level=0.95):
    """使用Bootstrap方法计算评价指标的置信区间"""
    n_classes = len(class_labels)
    n_samples = len(y_true)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    bootstrap_results = {
        'precision': {label: [] for label in class_labels},
        'sensitivity': {label: [] for label in class_labels},
        'specificity': {label: [] for label in class_labels},
        'f1': {label: [] for label in class_labels},
        'auc': {label: [] for label in class_labels},
        'accuracy': [],
        'macro_auc': [],
        'weighted_auc': []
    }

    np.random.seed(seed)

    for _ in range(n_bootstrap):
        indices = resample(np.arange(n_samples), replace=True, n_samples=n_samples)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        y_prob_boot = y_prob[indices]
        y_true_bin_boot = y_true_bin[indices]

        if len(np.unique(y_true_boot)) < n_classes:
            continue

        cm = confusion_matrix(y_true_boot, y_pred_boot, labels=range(n_classes))

        for i, label in enumerate(class_labels):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

            bootstrap_results['precision'][label].append(precision)
            bootstrap_results['sensitivity'][label].append(sensitivity)
            bootstrap_results['specificity'][label].append(specificity)
            bootstrap_results['f1'][label].append(f1)

            try:
                auc_val = roc_auc_score(y_true_bin_boot[:, i], y_prob_boot[:, i])
                bootstrap_results['auc'][label].append(auc_val)
            except:
                pass

        bootstrap_results['accuracy'].append(np.mean(y_true_boot == y_pred_boot))

        try:
            macro_auc = roc_auc_score(y_true_bin_boot, y_prob_boot, average='macro')
            weighted_auc = roc_auc_score(y_true_bin_boot, y_prob_boot, average='weighted')
            bootstrap_results['macro_auc'].append(macro_auc)
            bootstrap_results['weighted_auc'].append(weighted_auc)
        except:
            pass

    alpha = 1 - ci_level
    lower_p = alpha / 2 * 100
    upper_p = (1 - alpha / 2) * 100

    ci_results = []

    for label in class_labels:
        row = {'Class': label}
        for metric_name in ['precision', 'sensitivity', 'specificity', 'f1', 'auc']:
            values = bootstrap_results[metric_name][label]
            if len(values) > 0:
                point_est = np.mean(values)
                ci_lower = np.percentile(values, lower_p)
                ci_upper = np.percentile(values, upper_p)
                row[f'{metric_name}_estimate'] = point_est
                row[f'{metric_name}_ci_lower'] = ci_lower
                row[f'{metric_name}_ci_upper'] = ci_upper
                row[f'{metric_name}_ci'] = f"{point_est:.3f} ({ci_lower:.3f}-{ci_upper:.3f})"
            else:
                row[f'{metric_name}_estimate'] = np.nan
                row[f'{metric_name}_ci_lower'] = np.nan
                row[f'{metric_name}_ci_upper'] = np.nan
                row[f'{metric_name}_ci'] = "N/A"
        ci_results.append(row)

    overall_row = {'Class': 'Overall'}
    if len(bootstrap_results['accuracy']) > 0:
        acc_vals = bootstrap_results['accuracy']
        overall_row['accuracy_estimate'] = np.mean(acc_vals)
        overall_row['accuracy_ci_lower'] = np.percentile(acc_vals, lower_p)
        overall_row['accuracy_ci_upper'] = np.percentile(acc_vals, upper_p)
        overall_row[
            'accuracy_ci'] = f"{np.mean(acc_vals):.3f} ({np.percentile(acc_vals, lower_p):.3f}-{np.percentile(acc_vals, upper_p):.3f})"

    if len(bootstrap_results['macro_auc']) > 0:
        macro_vals = bootstrap_results['macro_auc']
        overall_row['macro_auc_estimate'] = np.mean(macro_vals)
        overall_row['macro_auc_ci_lower'] = np.percentile(macro_vals, lower_p)
        overall_row['macro_auc_ci_upper'] = np.percentile(macro_vals, upper_p)
        overall_row[
            'macro_auc_ci'] = f"{np.mean(macro_vals):.3f} ({np.percentile(macro_vals, lower_p):.3f}-{np.percentile(macro_vals, upper_p):.3f})"

    if len(bootstrap_results['weighted_auc']) > 0:
        weighted_vals = bootstrap_results['weighted_auc']
        overall_row['weighted_auc_estimate'] = np.mean(weighted_vals)
        overall_row['weighted_auc_ci_lower'] = np.percentile(weighted_vals, lower_p)
        overall_row['weighted_auc_ci_upper'] = np.percentile(weighted_vals, upper_p)
        overall_row[
            'weighted_auc_ci'] = f"{np.mean(weighted_vals):.3f} ({np.percentile(weighted_vals, lower_p):.3f}-{np.percentile(weighted_vals, upper_p):.3f})"

    ci_results.append(overall_row)
    return pd.DataFrame(ci_results)


# ===================== 完整评价指标计算函数 =====================
def calculate_full_metrics(y_true, y_pred, y_prob, class_labels):
    """计算完整的评价指标"""
    n_classes = len(class_labels)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
    df_metrics = pd.DataFrame(report).T
    df_metrics = df_metrics.drop('accuracy', errors='ignore')

    specificity_list = []
    auc_list = []
    y_bin = label_binarize(y_true, classes=range(n_classes))

    for i in range(n_classes):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificity_list.append(specificity)
        try:
            auc_score = roc_auc_score(y_bin[:, i], y_prob[:, i])
        except:
            auc_score = 0.5
        auc_list.append(auc_score)

    df_metrics['Specificity'] = pd.Series(specificity_list + [np.nan, np.nan],
                                          index=class_labels + ['macro avg', 'weighted avg'])
    df_metrics['AUC'] = pd.Series(auc_list + [np.nan, np.nan],
                                  index=class_labels + ['macro avg', 'weighted avg'])
    df_metrics.rename(columns={'recall': 'Sensitivity'}, inplace=True)

    return df_metrics.round(4)


def calculate_full_metrics_with_ci(y_true, y_pred, y_prob, class_labels, n_bootstrap=1000, ci_level=0.95):
    """计算完整评价指标及其Bootstrap置信区间"""
    basic_metrics = calculate_full_metrics(y_true, y_pred, y_prob, class_labels)
    ci_df = bootstrap_metric_ci(y_true, y_pred, y_prob, class_labels, n_bootstrap, ci_level)
    return basic_metrics, ci_df


# ===================== 主流程 =====================
def main():
    print("=" * 60)
    print("基于已有MCN模型生成训练集评价指标和系数SE")
    print("=" * 60)

    # ===================== 1. 加载已有模型 =====================
    print("\n" + "=" * 50)
    print("Step 1: 加载已有模型和变量筛选结果")
    print("=" * 50)

    # 加载模型
    model_path = os.path.join(existing_results_path, "final_mcn_grade_model.pkl")
    if not os.path.exists(model_path):
        print(f"错误: 未找到模型文件: {model_path}")
        return
    final_model = joblib.load(model_path)
    print(f"成功加载模型: {model_path}")

    # 加载变量筛选结果
    selection_file = os.path.join(existing_results_path, "MCN_Variable_Selection_Results.xlsx")
    if not os.path.exists(selection_file):
        print(f"错误: 未找到变量筛选结果: {selection_file}")
        return

    # 读取最终建模特征
    try:
        best_features_df = pd.read_excel(selection_file, sheet_name='Final_Features')
        best_features = best_features_df.iloc[:, 0].tolist()
    except:
        best_features_df = pd.read_excel(selection_file, sheet_name='最终特征')
        best_features = best_features_df.iloc[:, 0].tolist()

    print(f"成功加载变量筛选结果，最终特征数: {len(best_features)}")
    print(f"特征列表: {best_features}")

    n_classes = len(labels)

    # ===================== 2. 加载标准化参数 =====================
    print("\n" + "=" * 50)
    print("Step 2: 加载标准化参数")
    print("=" * 50)

    try:
        std_params = load_standardization_params(std_params_path)
        print("标准化参数加载成功")
    except Exception as e:
        print(f"警告: {e}")
        print("将尝试不使用标准化参数继续...")
        std_params = {}

    # ===================== 3. 读取并处理数据 =====================
    print("\n" + "=" * 50)
    print("Step 3: 读取并处理数据集")
    print("=" * 50)

    processed_dfs = []
    full_column_set = set()

    for i in range(1, m + 1):
        file_path = os.path.join(base_path, f"df_model_imputed_{i}.xlsx")
        if not os.path.exists(file_path):
            print(f"警告: 未找到数据文件: {file_path}")
            continue
        raw_df = pd.read_excel(file_path)
        proc = dynamic_preprocess_mcn(raw_df)

        if proc.empty:
            print(f"警告: 第{i}组数据中无MCN样本")
            continue

        processed_dfs.append(proc)
        full_column_set.update(proc.columns)
        print(f"  处理第{i}/{m}组数据，MCN样本数: {len(proc)}")

    if len(processed_dfs) == 0:
        print("错误: 没有有效的数据集！")
        return

    # 特征对齐
    final_aligned_data = []
    for df in processed_dfs:
        for col in full_column_set:
            if col not in df.columns:
                df[col] = 0
        final_aligned_data.append(df)

    # ===================== 4. 5折交叉验证训练并计算系数SE =====================
    print("\n" + "=" * 50)
    print("Step 4: 5折交叉验证训练并计算系数SE (Rubin规则)")
    print("=" * 50)

    all_coefs = []
    all_intercepts = []
    all_coef_vars = []
    all_intercept_vars = []

    all_train_y_true = []
    all_train_y_pred = []
    all_train_y_prob = []
    all_train_X = []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_count = 0
    for imp_idx, df_ready in enumerate(final_aligned_data):
        for feat in best_features:
            if feat not in df_ready.columns:
                df_ready[feat] = 0

        X = df_ready[best_features].astype(float)
        y = df_ready[target_col].astype(int)
        ids = df_ready[id_col]

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            fold_count += 1
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # 应用标准化
            if std_params:
                X_train_s = apply_external_standardization(X_train, num_vars, std_params)
                X_test_s = apply_external_standardization(X_test, num_vars, std_params)
            else:
                X_train_s = X_train
                X_test_s = X_test

            # 训练模型
            lr_model = LogisticRegression(
                penalty='l2', solver='lbfgs', multi_class='multinomial',
                max_iter=5000, random_state=seed
            )
            lr_model.fit(X_train_s.values, y_train.values)

            all_coefs.append(lr_model.coef_.copy())
            all_intercepts.append(lr_model.intercept_.copy())

            # 计算该折的系数SE
            if fold_count <= 10:  # 只对前10折计算SE以节省时间
                coef_se, intercept_se = compute_logistic_se_bootstrap(
                    X_train_s.values, y_train.values, lr_model, n_bootstrap=100
                )
                all_coef_vars.append(coef_se ** 2)
                all_intercept_vars.append(intercept_se ** 2)

            # 预测训练集
            y_pred_train = lr_model.predict(X_train_s.values)
            y_prob_train = lr_model.predict_proba(X_train_s.values)

            all_train_y_true.extend(y_train.values)
            all_train_y_pred.extend(y_pred_train)
            all_train_y_prob.extend(y_prob_train)
            all_train_X.append(X_train_s.values)

        print(f"  完成第{imp_idx + 1}/{len(final_aligned_data)}组数据的5折交叉验证")

    print(f"  总共完成 {fold_count} 折训练")

    # ===================== 5. Rubin规则池化系数和SE =====================
    print("\n" + "=" * 50)
    print("Step 5: Rubin规则池化系数和SE")
    print("=" * 50)

    all_coefs_flat = [coef.flatten() for coef in all_coefs]

    # 使用所有折的系数计算组间方差
    pooled_coef_flat = np.mean(all_coefs_flat, axis=0)
    pooled_intercept = np.mean(all_intercepts, axis=0)

    # 组间方差
    B_coef = np.var(all_coefs_flat, axis=0, ddof=1)
    B_intercept = np.var(all_intercepts, axis=0, ddof=1)

    # 组内方差（使用Bootstrap SE的平均）
    if len(all_coef_vars) > 0:
        all_coef_vars_flat = [var.flatten() for var in all_coef_vars]
        W_coef = np.mean(all_coef_vars_flat, axis=0)
        W_intercept = np.mean(all_intercept_vars, axis=0)

        # 总方差
        n_folds = len(all_coefs)
        T_coef = W_coef + (1 + 1 / n_folds) * B_coef
        T_intercept = W_intercept + (1 + 1 / n_folds) * B_intercept

        pooled_coef_se_flat = np.sqrt(T_coef)
        pooled_intercept_se = np.sqrt(T_intercept)
    else:
        # 如果没有计算组内方差，只使用组间方差
        pooled_coef_se_flat = np.sqrt(B_coef)
        pooled_intercept_se = np.sqrt(B_intercept)

    pooled_coef = pooled_coef_flat.reshape(n_classes, len(best_features))
    pooled_coef_se = pooled_coef_se_flat.reshape(n_classes, len(best_features))

    # 计算Z值和P值
    z_values = pooled_coef / pooled_coef_se
    p_values = 2 * (1 - norm.cdf(np.abs(z_values)))

    # 计算95%置信区间
    ci_lower = pooled_coef - 1.96 * pooled_coef_se
    ci_upper = pooled_coef + 1.96 * pooled_coef_se

    # ===================== 6. 保存系数和SE结果 =====================
    print("\n" + "=" * 50)
    print("Step 6: 保存系数和SE结果")
    print("=" * 50)

    # 创建详细的系数表
    coef_results = []
    for class_idx, class_name in enumerate(labels):
        for feat_idx, feat_name in enumerate(best_features):
            coef_results.append({
                'Class': class_name,
                'Feature': feat_name,
                'Coefficient': pooled_coef[class_idx, feat_idx],
                'SE': pooled_coef_se[class_idx, feat_idx],
                'Z_value': z_values[class_idx, feat_idx],
                'P_value': p_values[class_idx, feat_idx],
                'CI_95_Lower': ci_lower[class_idx, feat_idx],
                'CI_95_Upper': ci_upper[class_idx, feat_idx],
                'CI_95': f"({ci_lower[class_idx, feat_idx]:.4f}, {ci_upper[class_idx, feat_idx]:.4f})"
            })
        # 添加截距
        coef_results.append({
            'Class': class_name,
            'Feature': 'Intercept',
            'Coefficient': pooled_intercept[class_idx],
            'SE': pooled_intercept_se[class_idx],
            'Z_value': pooled_intercept[class_idx] / pooled_intercept_se[class_idx] if pooled_intercept_se[
                                                                                           class_idx] != 0 else np.nan,
            'P_value': 2 * (1 - norm.cdf(np.abs(pooled_intercept[class_idx] / pooled_intercept_se[class_idx]))) if
            pooled_intercept_se[class_idx] != 0 else np.nan,
            'CI_95_Lower': pooled_intercept[class_idx] - 1.96 * pooled_intercept_se[class_idx],
            'CI_95_Upper': pooled_intercept[class_idx] + 1.96 * pooled_intercept_se[class_idx],
            'CI_95': f"({pooled_intercept[class_idx] - 1.96 * pooled_intercept_se[class_idx]:.4f}, {pooled_intercept[class_idx] + 1.96 * pooled_intercept_se[class_idx]:.4f})"
        })

    coef_df_detailed = pd.DataFrame(coef_results)
    coef_detailed_path = os.path.join(output_path, "Pooled_Coefficients_with_SE.xlsx")
    coef_df_detailed.to_excel(coef_detailed_path, index=False)
    print(f"  详细系数表（含SE）已保存: {coef_detailed_path}")

    # 创建宽格式的系数表
    coef_wide = pd.DataFrame(pooled_coef, columns=best_features, index=labels)
    coef_wide['Intercept'] = pooled_intercept

    se_wide = pd.DataFrame(pooled_coef_se, columns=[f"{f}_SE" for f in best_features], index=labels)
    se_wide['Intercept_SE'] = pooled_intercept_se

    with pd.ExcelWriter(os.path.join(output_path, "Pooled_Coefficients_SE_Wide.xlsx")) as writer:
        coef_wide.to_excel(writer, sheet_name='Coefficients')
        se_wide.to_excel(writer, sheet_name='Standard_Errors')
        p_wide = pd.DataFrame(p_values, columns=best_features, index=labels)
        p_wide.to_excel(writer, sheet_name='P_values')

    print(f"  宽格式系数表已保存: {os.path.join(output_path, 'Pooled_Coefficients_SE_Wide.xlsx')}")

    # ===================== 7. 生成训练集混淆矩阵 =====================
    print("\n" + "=" * 50)
    print("Step 7: 生成训练集混淆矩阵")
    print("=" * 50)

    y_train_all = np.array(all_train_y_true)
    y_pred_train_all = np.array(all_train_y_pred)
    y_prob_train_all = np.array(all_train_y_prob)

    cm_train = confusion_matrix(y_train_all, y_pred_train_all)

    # 绘制混淆矩阵
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 计数混淆矩阵
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Greens',
                xticklabels=labels, yticklabels=labels,
                annot_kws={'size': 12}, ax=axes[0])
    axes[0].set_title('MCN Training Set: Confusion Matrix - Counts', fontsize=14)
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_xlabel('Predicted Label', fontsize=12)

    # 归一化混淆矩阵
    cm_normalized = cm_train.astype('float') / cm_train.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens',
                xticklabels=labels, yticklabels=labels,
                annot_kws={'size': 12}, ax=axes[1])
    axes[1].set_title('MCN Training Set: Confusion Matrix - Normalized', fontsize=14)
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].set_xlabel('Predicted Label', fontsize=12)

    plt.tight_layout()
    cm_path = os.path.join(output_path, "train_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  训练集混淆矩阵已保存: {cm_path}")

    # ===================== 8. 生成训练集ROC曲线 =====================
    print("\n" + "=" * 50)
    print("Step 8: 生成训练集ROC曲线")
    print("=" * 50)

    y_train_bin = label_binarize(y_train_all, classes=[0, 1, 2])
    plt.figure(figsize=(10, 8))

    colors = ['#2ca02c', '#ff7f0e', '#d62728']
    for i in range(len(labels)):
        fpr, tpr, _ = roc_curve(y_train_bin[:, i], y_prob_train_all[:, i])
        roc_auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2,
                 label=f'{labels[i]} (AUC = {roc_auc_val:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Reference')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('MCN Training Set: Multi-class ROC Curve (5-Fold CV)', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(output_path, "train_roc_curve.png")
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  训练集ROC曲线已保存: {roc_path}")

    # ===================== 9. 计算并保存训练集评价指标 =====================
    print("\n" + "=" * 50)
    print("Step 9: 计算训练集评价指标")
    print("=" * 50)

    train_metrics_basic = calculate_full_metrics(y_train_all, y_pred_train_all, y_prob_train_all, labels)
    basic_metrics_path = os.path.join(output_path, "train_metrics.xlsx")
    train_metrics_basic.to_excel(basic_metrics_path)
    print(f"  训练集基础评价指标已保存: {basic_metrics_path}")

    print("  计算Bootstrap置信区间（这可能需要几分钟）...")
    _, train_ci = calculate_full_metrics_with_ci(
        y_train_all, y_pred_train_all, y_prob_train_all, labels,
        n_bootstrap=BOOTSTRAP_N, ci_level=CI_LEVEL
    )
    ci_metrics_path = os.path.join(output_path, "train_metrics_with_CI.xlsx")
    train_ci.to_excel(ci_metrics_path, index=False)
    print(f"  训练集评价指标(含95%CI)已保存: {ci_metrics_path}")

    # ===================== 10. 打印结果摘要 =====================
    print("\n" + "=" * 60)
    print("结果摘要")
    print("=" * 60)

    print("\n--- MCN训练集评价指标 ---")
    print(train_metrics_basic.to_string())

    print("\n\n--- 系数SE摘要（显示前15行） ---")
    print(coef_df_detailed.head(15).to_string(index=False))

    # ===================== 11. 输出文件清单 =====================
    print("\n" + "=" * 60)
    print("输出文件清单")
    print("=" * 60)
    print(f"""
输出目录: {output_path}

生成的文件:
1. train_confusion_matrix.png - 训练集混淆矩阵（含计数和归一化两种）
2. train_roc_curve.png - 训练集ROC曲线
3. train_metrics.xlsx - 训练集基础评价指标
4. train_metrics_with_CI.xlsx - 训练集评价指标 (含95% Bootstrap置信区间)
5. Pooled_Coefficients_with_SE.xlsx - 详细系数表 (含Coefficient, SE, Z值, P值, 95%CI)
6. Pooled_Coefficients_SE_Wide.xlsx - 宽格式系数表 (分别包含系数、SE、P值三个sheet)
""")

    print("\n=== MCN训练集指标和系数SE生成完毕 ===")


if __name__ == "__main__":
    main()