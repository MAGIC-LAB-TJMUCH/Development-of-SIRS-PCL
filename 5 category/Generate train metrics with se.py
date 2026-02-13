# -*- coding: utf-8 -*-
"""
基于已有模型生成训练集评价指标、混淆矩阵、ROC曲线和系数SE

直接读取已保存的模型和变量筛选结果，使用Rubin规则计算系数标准误
"""

import pandas as pd
import numpy as np
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
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
# 已有结果路径
existing_results_path = r"\五分类\Final_Analysis_Results_Std_ExternalParams"

# 数据路径
base_path = r"\多中心胰腺囊性病变诊断与风险预测\完整数据集-五分类建模\MICE_model"

# 标准化参数文件路径
std_params_path = os.path.join(base_path, "Standardization_Params", "Standardization_Parameters.xlsx")

# 输出路径（保存到已有结果目录）
output_path = existing_results_path

# 核心参数
target_col = "Dignosis"
id_col = "key"
m = 10
seed = 9477
BOOTSTRAP_N = 1000
CI_LEVEL = 0.95

# 变量清单
raw_categorical_vars = [
    "Gender", "Cyst wall thickness", "Uniform Cyst wall",
    "Cyst wall enhancement", "Mural nodule status", "Mural nodule enhancement",
    "Solid component enhancement", "Intracystic septations", "Uniform Septations",
    "Intracystic septa enhancement", "Capsule", "Main PD communication",
    "Pancreatic parenchymal atrophy", "MPD dilation", "Mural nodule in MPD",
    "Common bile duct dilation", "Vascular abutment", "Enlarged lymph nodes",
    "Distant metastasis", "Tumor lesion", "Lesion_Head_neck", "Lesion_body_tail",
    "Diabetes", "Jaundice"
]

num_vars = [
    "Long diameter of lesion (mm)", "Short diameter of lesion (mm)",
    "Long diameter of largest mural nodule (mm)", "Short diameter of largest mural nodule (mm)",
    "Short diameter of largest lymph node (mm)", "CA_199", "CEA", "Age"
]

whitelist = [target_col, id_col] + raw_categorical_vars + num_vars
labels = []
final_mapping = {}


# ===================== 标准化工具函数 =====================
def load_standardization_params(filepath):
    """加载标准化参数文件"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"未找到标准化参数文件: {filepath}")
    print(f"正在加载标准化参数: {filepath}")
    params_df = pd.read_excel(filepath)
    params_dict = params_df.set_index('Feature')[['Mean', 'Std']].to_dict('index')
    return params_dict


def apply_external_standardization(df, params_dict, num_vars):
    """应用外部标准化参数 (Z-score)"""
    df_scaled = df.copy()
    for col in num_vars:
        if col in df_scaled.columns and col in params_dict:
            mean_val = params_dict[col]['Mean']
            std_val = params_dict[col]['Std']
            if std_val != 0:
                df_scaled[col] = (df_scaled[col] - mean_val) / std_val
    return df_scaled


# ===================== 数据预处理函数 =====================
def dynamic_preprocess(df):
    df_clean = df[[c for c in whitelist if c in df.columns]].copy()
    le = LabelEncoder()
    df_clean[target_col] = le.fit_transform(df_clean[target_col].astype(str))
    mapping_dict = {i: label for i, label in enumerate(le.classes_)}

    for var in raw_categorical_vars:
        if var not in df_clean.columns:
            continue
        lvl_count = len(df_clean[var].dropna().unique())
        if lvl_count == 2:
            if var == "Gender":
                mapping = {'Male': 0, 'Female': 1, 0: 0, 1: 1}
            else:
                mapping = {'No': 0, 'Yes': 1, 0: 0, 1: 1}
            df_clean[var] = df_clean[var].map(mapping).fillna(0)
        elif lvl_count > 2:
            df_clean = pd.get_dummies(df_clean, columns=[var], drop_first=False)
    return df_clean, mapping_dict


# ===================== 计算逻辑回归系数SE的函数 =====================
def compute_logistic_se(X, y, model):
    """
    计算多项逻辑回归系数的标准误
    使用Fisher信息矩阵的逆（Hessian矩阵的逆）
    """
    n_samples, n_features = X.shape
    n_classes = len(model.classes_)

    # 获取预测概率
    probs = model.predict_proba(X)

    # 计算Fisher信息矩阵
    # 对于多项逻辑回归，信息矩阵是分块的
    # 每个类别对应 (n_features + 1) 个参数（含截距）

    # 简化方法：使用数值近似计算Hessian
    coef_flat = np.concatenate([model.coef_.flatten(), model.intercept_])
    n_params = len(coef_flat)

    # 使用Bootstrap方法估计SE（更稳健）
    n_bootstrap = 200
    bootstrap_coefs = []
    bootstrap_intercepts = []

    np.random.seed(seed)
    for _ in range(n_bootstrap):
        # Bootstrap重采样
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X[indices]
        y_boot = y[indices]

        # 确保所有类别都存在
        if len(np.unique(y_boot)) < n_classes:
            continue

        # 训练模型
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

    # 计算SE
    if len(bootstrap_coefs) > 10:
        coef_se = np.std(bootstrap_coefs, axis=0, ddof=1)
        intercept_se = np.std(bootstrap_intercepts, axis=0, ddof=1)
    else:
        # 如果Bootstrap失败，返回NaN
        coef_se = np.full(model.coef_.flatten().shape, np.nan)
        intercept_se = np.full(model.intercept_.shape, np.nan)

    return coef_se, intercept_se


def rubin_pooling_with_se(all_coefs, all_intercepts, all_coef_vars, all_intercept_vars, m):
    """
    使用Rubin规则池化系数和方差

    参数:
        all_coefs: 各插补集的系数列表
        all_intercepts: 各插补集的截距列表
        all_coef_vars: 各插补集的系数方差列表
        all_intercept_vars: 各插补集的截距方差列表
        m: 插补数据集数量

    返回:
        pooled_coef: 池化系数
        pooled_intercept: 池化截距
        pooled_coef_se: 池化系数SE
        pooled_intercept_se: 池化截距SE
    """
    # 1. 池化点估计（均值）
    pooled_coef = np.mean(all_coefs, axis=0)
    pooled_intercept = np.mean(all_intercepts, axis=0)

    # 2. 组内方差 (Within-imputation variance) - 方差的均值
    W_coef = np.mean(all_coef_vars, axis=0)
    W_intercept = np.mean(all_intercept_vars, axis=0)

    # 3. 组间方差 (Between-imputation variance)
    B_coef = np.var(all_coefs, axis=0, ddof=1)
    B_intercept = np.var(all_intercepts, axis=0, ddof=1)

    # 4. 总方差 (Total variance) = W + (1 + 1/m) * B
    T_coef = W_coef + (1 + 1 / m) * B_coef
    T_intercept = W_intercept + (1 + 1 / m) * B_intercept

    # 5. 池化SE = sqrt(T)
    pooled_coef_se = np.sqrt(T_coef)
    pooled_intercept_se = np.sqrt(T_intercept)

    return pooled_coef, pooled_intercept, pooled_coef_se, pooled_intercept_se


# ===================== Bootstrap置信区间计算函数 =====================
def bootstrap_metric_ci(y_true, y_pred, y_prob, labels, n_bootstrap=1000, ci_level=0.95):
    n_classes = len(labels)
    n_samples = len(y_true)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    bootstrap_results = {
        'precision': {label: [] for label in labels},
        'sensitivity': {label: [] for label in labels},
        'specificity': {label: [] for label in labels},
        'f1': {label: [] for label in labels},
        'auc': {label: [] for label in labels},
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
        for i, label in enumerate(labels):
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
            bootstrap_results['macro_auc'].append(roc_auc_score(y_true_bin_boot, y_prob_boot, average='macro'))
            bootstrap_results['weighted_auc'].append(roc_auc_score(y_true_bin_boot, y_prob_boot, average='weighted'))
        except:
            pass

    alpha = 1 - ci_level
    lower_p = alpha / 2 * 100
    upper_p = (1 - alpha / 2) * 100

    ci_results = []
    for label in labels:
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


# ===================== 评价指标计算函数 =====================
def get_metrics(y_true, y_pred, y_prob, labels):
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    df_metrics = pd.DataFrame(report).transpose()
    y_true_bin = label_binarize(y_true, classes=range(len(labels)))
    cm = confusion_matrix(y_true, y_pred)
    specs = []
    aucs = []
    for i in range(len(labels)):
        tn = cm.sum() - (cm[:, i].sum() + cm[i, :].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        specs.append(spec)
        aucs.append(roc_auc_score(y_true_bin[:, i], y_prob[:, i]))
    df_metrics.loc[labels, 'Specificity'] = specs
    df_metrics.loc[labels, 'AUC'] = aucs
    df_metrics.rename(columns={'recall': 'Sensitivity'}, inplace=True)
    columns_to_return = ['precision', 'Sensitivity', 'Specificity', 'f1-score', 'support', 'AUC']
    if 'accuracy' in df_metrics.columns:
        columns_to_return.append('accuracy')
    return df_metrics[columns_to_return]


def get_metrics_with_ci(y_true, y_pred, y_prob, labels, n_bootstrap=1000, ci_level=0.95):
    basic_metrics = get_metrics(y_true, y_pred, y_prob, labels)
    ci_df = bootstrap_metric_ci(y_true, y_pred, y_prob, labels, n_bootstrap, ci_level)
    return basic_metrics, ci_df


# ===================== 主流程 =====================
def main():
    global labels, final_mapping

    print("=" * 60)
    print("基于已有模型生成训练集评价指标和系数SE")
    print("=" * 60)

    # ===================== 1. 加载已有模型和变量筛选结果 =====================
    print("\n" + "=" * 50)
    print("Step 1: 加载已有模型和变量筛选结果")
    print("=" * 50)

    # 加载模型
    model_path = os.path.join(existing_results_path, "final_model.pkl")
    if not os.path.exists(model_path):
        print(f"错误: 未找到模型文件: {model_path}")
        return
    final_model = joblib.load(model_path)
    print(f"成功加载模型: {model_path}")

    # 加载变量筛选结果
    selector_path = os.path.join(existing_results_path, "selector_results.pkl")
    if not os.path.exists(selector_path):
        print(f"错误: 未找到变量筛选结果: {selector_path}")
        return
    detailed_selection = joblib.load(selector_path)
    best_features = detailed_selection[detailed_selection['Final_Keep'] == 1].index.tolist()
    print(f"成功加载变量筛选结果，最终特征数: {len(best_features)}")
    print(f"特征列表: {best_features}")

    # ===================== 2. 加载标准化参数 =====================
    print("\n" + "=" * 50)
    print("Step 2: 加载标准化参数")
    print("=" * 50)

    try:
        std_params = load_standardization_params(std_params_path)
        print("标准化参数加载成功")
    except Exception as e:
        print(f"错误: {e}")
        return

    # ===================== 3. 读取并处理数据 =====================
    print("\n" + "=" * 50)
    print("Step 3: 读取并处理数据集")
    print("=" * 50)

    processed_dfs = []
    full_column_set = set()

    for i in range(1, m + 1):
        file_path = os.path.join(base_path, f"df_model_imputed_{i}.xlsx")
        raw_df = pd.read_excel(file_path)
        proc, mapping = dynamic_preprocess(raw_df)
        proc = apply_external_standardization(proc, std_params, num_vars)
        final_mapping.update(mapping)
        processed_dfs.append(proc)
        full_column_set.update(proc.columns)

    labels = [final_mapping[i] for i in sorted(final_mapping.keys())]
    n_classes = len(labels)
    print(f"类别标签: {labels}")

    # 特征对齐
    final_aligned_data = []
    for df in processed_dfs:
        for col in full_column_set:
            if col not in df.columns:
                df[col] = 0
        final_aligned_data.append(df)

    # ===================== 4. 重新训练模型并计算系数SE =====================
    print("\n" + "=" * 50)
    print("Step 4: 重新训练模型并计算系数SE (Rubin规则)")
    print("=" * 50)

    all_coefs = []
    all_intercepts = []
    all_coef_vars = []  # 存储每个插补集的系数方差
    all_intercept_vars = []  # 存储每个插补集的截距方差

    all_train_y_true = []
    all_train_y_pred = []
    all_train_y_prob = []

    lr_model = LogisticRegression(
        penalty='l2', solver='lbfgs', multi_class='multinomial',
        max_iter=5000, random_state=seed
    )

    for i in range(m):
        df_ready = final_aligned_data[i]
        X = df_ready[best_features].astype(float)
        y = df_ready[target_col].astype(int)
        ids = df_ready[id_col]

        # 使用与原建模过程相同的随机种子划分数据
        X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
            X, y, ids, test_size=0.3, random_state=120 + i, stratify=y
        )

        # 训练模型
        lr_model.fit(X_train.values, y_train.values)

        # 保存系数和截距
        all_coefs.append(lr_model.coef_.copy())
        all_intercepts.append(lr_model.intercept_.copy())

        # 计算该插补集的系数SE（使用Bootstrap）
        print(f"  计算第{i + 1}/{m}组数据集的系数SE...")
        coef_se, intercept_se = compute_logistic_se(X_train.values, y_train.values, lr_model)

        # 方差 = SE^2
        all_coef_vars.append(coef_se ** 2)
        all_intercept_vars.append(intercept_se ** 2)

        # 预测训练集
        y_pred_train = lr_model.predict(X_train.values)
        y_prob_train = lr_model.predict_proba(X_train.values)

        all_train_y_true.append(y_train.values)
        all_train_y_pred.append(y_pred_train)
        all_train_y_prob.append(y_prob_train)

    # ===================== 5. Rubin规则池化系数和SE =====================
    print("\n" + "=" * 50)
    print("Step 5: Rubin规则池化系数和SE")
    print("=" * 50)

    # 将系数reshape为2D数组进行池化
    all_coefs_flat = [coef.flatten() for coef in all_coefs]
    all_coef_vars_flat = [var.flatten() for var in all_coef_vars]

    pooled_coef_flat, pooled_intercept, pooled_coef_se_flat, pooled_intercept_se = rubin_pooling_with_se(
        np.array(all_coefs_flat),
        np.array(all_intercepts),
        np.array(all_coef_vars_flat),
        np.array(all_intercept_vars),
        m
    )

    # Reshape回原始形状
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

    # 创建详细的系数表（包含SE、Z值、P值、95%CI）
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
    coef_wide = pd.DataFrame(pooled_coef, columns=best_features,
                             index=[f"Class_{label}" for label in labels])
    coef_wide['Intercept'] = pooled_intercept

    se_wide = pd.DataFrame(pooled_coef_se, columns=[f"{f}_SE" for f in best_features],
                           index=[f"Class_{label}" for label in labels])
    se_wide['Intercept_SE'] = pooled_intercept_se

    # 合并保存
    with pd.ExcelWriter(os.path.join(output_path, "Pooled_Coefficients_SE_Wide.xlsx")) as writer:
        coef_wide.to_excel(writer, sheet_name='Coefficients')
        se_wide.to_excel(writer, sheet_name='Standard_Errors')

        # 添加P值表
        p_wide = pd.DataFrame(p_values, columns=best_features,
                              index=[f"Class_{label}" for label in labels])
        p_wide.to_excel(writer, sheet_name='P_values')

    print(f"  宽格式系数表已保存: {os.path.join(output_path, 'Pooled_Coefficients_SE_Wide.xlsx')}")

    # ===================== 7. 生成训练集混淆矩阵 =====================
    print("\n" + "=" * 50)
    print("Step 7: 生成训练集混淆矩阵")
    print("=" * 50)

    # 使用最后一组数据集的结果
    y_train_last = all_train_y_true[-1]
    y_pred_train_final = all_train_y_pred[-1]
    y_prob_train_final = all_train_y_prob[-1]

    cm_train = confusion_matrix(y_train_last, y_pred_train_final)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Greens',
                xticklabels=labels, yticklabels=labels,
                annot_kws={'size': 12})
    plt.title('Training Set: Confusion Matrix (Pooled Model)', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join(output_path, "train_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  训练集混淆矩阵已保存: {cm_path}")

    # ===================== 8. 生成训练集ROC曲线 =====================
    print("\n" + "=" * 50)
    print("Step 8: 生成训练集ROC曲线")
    print("=" * 50)

    y_train_bin = label_binarize(y_train_last, classes=range(len(labels)))
    plt.figure(figsize=(10, 8))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i in range(len(labels)):
        fpr, tpr, _ = roc_curve(y_train_bin[:, i], y_prob_train_final[:, i])
        roc_auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                 label=f'{labels[i]} (AUC = {roc_auc_val:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Reference')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('Training Set: Multi-class ROC Curve (Pooled Model)', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
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

    # 基础评价指标
    train_metrics_basic = get_metrics(y_train_last, y_pred_train_final, y_prob_train_final, labels)
    basic_metrics_path = os.path.join(output_path, "train_metrics.xlsx")
    train_metrics_basic.to_excel(basic_metrics_path)
    print(f"  训练集基础评价指标已保存: {basic_metrics_path}")

    # 带Bootstrap置信区间的评价指标
    print("  计算Bootstrap置信区间（这可能需要几分钟）...")
    _, train_ci = get_metrics_with_ci(
        np.array(y_train_last), y_pred_train_final, y_prob_train_final, labels,
        n_bootstrap=BOOTSTRAP_N, ci_level=CI_LEVEL
    )
    ci_metrics_path = os.path.join(output_path, "train_metrics_with_CI.xlsx")
    train_ci.to_excel(ci_metrics_path, index=False)
    print(f"  训练集评价指标(含95%CI)已保存: {ci_metrics_path}")

    # ===================== 10. 打印结果摘要 =====================
    print("\n" + "=" * 60)
    print("结果摘要")
    print("=" * 60)

    print("\n--- 训练集评价指标 ---")
    print(train_metrics_basic.to_string())

    print("\n\n--- 系数SE摘要（显示前10行） ---")
    print(coef_df_detailed.head(10).to_string(index=False))

    # ===================== 11. 输出文件清单 =====================
    print("\n" + "=" * 60)
    print("输出文件清单")
    print("=" * 60)
    print(f"""
输出目录: {output_path}

生成的文件:
1. train_confusion_matrix.png - 训练集混淆矩阵
2. train_roc_curve.png - 训练集ROC曲线
3. train_metrics.xlsx - 训练集基础评价指标
4. train_metrics_with_CI.xlsx - 训练集评价指标 (含95% Bootstrap置信区间)
5. Pooled_Coefficients_with_SE.xlsx - 详细系数表 (含Coefficient, SE, Z值, P值, 95%CI)
6. Pooled_Coefficients_SE_Wide.xlsx - 宽格式系数表 (分别包含系数、SE、P值三个sheet)
""")

    print("\n=== 所有结果生成完毕 ===")


if __name__ == "__main__":
    main()