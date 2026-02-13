# -*- coding: utf-8 -*-
"""
MCN 风险分级预测模型 - 使用已训练模型对完整外部数据集进行验证
外部标准化参数版本

功能说明：
1. 加载MCN_Grade_Final_Results文件夹中已训练的final model
2. 【新增】使用外部标准化参数文件，仅对数值型变量进行标准化
3. 使用完整外部数据集（df_MCN_valid_comp）进行外部验证
4. 生成混淆矩阵、AUC曲线、Bootstrap置信区间
5. 保存编码后的完整数据集和预测结果
6. 计算外部验证的SHAP值
7. 生成综合结果汇总报告
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report, roc_auc_score
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import warnings
import json
from itertools import cycle

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ==========================================
# 1. 路径配置
# ==========================================
# 已训练模型所在路径（MCN_5_fold_with_enhanced_rubin生成的模型）
model_base_path = r"G:\胰腺囊性病变分类与风险预测_1.2\MCN风险结果\基于预测数据建模"

# 完整外部验证数据路径
external_data_path = r"E:\Pancreatic cancer\多中心胰腺囊性病变诊断与风险预测\完整数据集-验证"
external_data_file = os.path.join(external_data_path, "Data_MCN_Pre_TRUE.xlsx")

# 【新增】外部标准化参数路径
std_params_path = r"E:\Pancreatic cancer\多中心胰腺囊性病变诊断与风险预测\完整数据集-五分类建模\MICE_model\Standardization_Params_MCN"
std_params_file = os.path.join(std_params_path, "Standardization_Parameters.xlsx")

# 结果输出路径
results_path = os.path.join(model_base_path, "MCN_Complete_External_Validation_ExtStd_v2")
os.makedirs(results_path, exist_ok=True)

shap_output_path = os.path.join(results_path, "SHAP_Analysis")
os.makedirs(shap_output_path, exist_ok=True)

encoded_data_path = os.path.join(results_path, "Encoded_Datasets")
os.makedirs(encoded_data_path, exist_ok=True)

# 【新增】图表输出路径
figures_path = os.path.join(results_path, "Figures")
os.makedirs(figures_path, exist_ok=True)

target_col = "Grade"
id_col = "key"
filter_col = "Dignosis"
filter_value = "MCN"
seed = 1352

# Bootstrap置信区间参数
BOOTSTRAP_N = 1000
CI_LEVEL = 0.95

raw_categorical_vars = [
    "Gender", "Cyst wall thickness", "Uniform Cyst wall", "Cyst wall enhancement",
    "Mural nodule status", "Mural nodule enhancement", "Solid component enhancement",
    "Intracystic septations", "Uniform Septations", "Intracystic septa enhancement", "Capsule",
    "Main PD communication", "Pancreatic parenchymal atrophy", "MPD dilation",
    "Mural nodule in MPD", "Common bile duct dilation", "Vascular abutment", "Enlarged lymph nodes",
    "Distant metastasis", "Tumor lesion", "Lesion_Head_neck", "Lesion_body_tail", "Diabetes",
    "Jaundice"
]

# 【重要】数值型变量列表 - 仅对这些变量应用标准化
num_vars = ["Short diameter of lesion (mm)",
            "Short diameter of solid component (mm)",
            "Short diameter of largest mural nodule (mm)",
            "CA_199", "CEA", "Age"]

whitelist = [target_col, id_col] + raw_categorical_vars + num_vars
labels = ['Benign/Low Risk', 'Medium Risk', 'High Risk']


# ==========================================
# 2. 预处理函数
# ==========================================
def dynamic_preprocess_mcn(df):
    """MCN数据预处理函数（完整数据集版本，不需要按Dignosis筛选）"""
    df_sub = df[df[filter_col] == filter_value].copy()
    available_cols = [c for c in whitelist if c in df_sub.columns]
    df_clean = df_sub[available_cols].copy()

    df_clean[target_col] = df_clean[target_col].astype(str).str.strip().str.lower()
    grade_map = {
        'low risk': 0, 'benign': 0, 'lowrisk': 0,
        'medium risk': 1, 'medium': 1, 'mediumrisk': 1,
        'high risk': 2, 'high': 2, 'highrisk': 2,
    }
    df_clean[target_col] = df_clean[target_col].map(grade_map)
    before_drop = len(df_clean)
    df_clean = df_clean.dropna(subset=[target_col])
    after_drop = len(df_clean)
    if before_drop != after_drop:
        print(f"    警告：丢弃了 {before_drop - after_drop} 条无法识别的 Grade 标签样本")
    df_clean[target_col] = df_clean[target_col].astype(int)

    enhancement_vars = ["Cyst wall enhancement", "Mural nodule enhancement",
                        "Solid component enhancement", "Intracystic septa enhancement"]
    phase_suffixes = ["Arterial Phase Enhancement", "Delayed enhancement"]

    for target_var in enhancement_vars:
        if target_var not in df_clean.columns:
            continue
        phase_cols = [target_var.replace(" enhancement", s) for s in phase_suffixes]
        existing_phases = [col for col in phase_cols if col in df_sub.columns]

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
            phase_data = df_sub.loc[df_clean.index, existing_phases].copy()
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

    for var in raw_categorical_vars:
        if var not in df_clean.columns:
            continue
        unique_vals = df_clean[var].dropna().unique()
        if len(unique_vals) <= 2:
            mapping = {'male': 0, 'female': 1, 'no': 0, 'yes': 1, '0': 0, '1': 1}
            df_clean[var] = df_clean[var].astype(str).str.lower().str.strip().map(lambda x: mapping.get(x, 0))
            df_clean[var] = df_clean[var].fillna(0).astype(float)
        else:
            df_clean = pd.get_dummies(df_clean, columns=[var], drop_first=False, dtype=float)

    return df_clean


# ==========================================
# 3. 【新增】外部标准化参数加载与应用函数
# ==========================================
def load_external_std_params(file_path):
    """
    加载外部标准化参数文件 (Feature, Mean, Std)

    Parameters:
    -----------
    file_path : str
        标准化参数Excel文件路径

    Returns:
    --------
    dict : {'Feature_Name': {'mean': value, 'std': value}, ...}
    """
    try:
        df = pd.read_excel(file_path)
        # 兼容可能的列名大小写
        df.columns = [c.capitalize() for c in df.columns]

        if 'Feature' not in df.columns or 'Mean' not in df.columns or 'Std' not in df.columns:
            raise ValueError("标准化文件必须包含 Feature, Mean, Std 列")

        # 转换为字典: {'Age': {'mean': 60.5, 'std': 12.3}, ...}
        std_dict = {}
        for _, row in df.iterrows():
            std_dict[row['Feature']] = {'mean': row['Mean'], 'std': row['Std']}

        print(f"  成功加载外部标准化参数，共 {len(std_dict)} 个变量")
        print(f"  包含变量: {list(std_dict.keys())}")
        return std_dict
    except Exception as e:
        print(f"  加载标准化参数失败: {e}")
        raise


def apply_external_standardization(df, numeric_vars, std_dict):
    """
    仅对数值型变量进行标准化，OHE变量保持不变

    Parameters:
    -----------
    df : pd.DataFrame
        输入数据框
    numeric_vars : list
        数值型变量列表
    std_dict : dict
        标准化参数字典

    Returns:
    --------
    pd.DataFrame : 标准化后的数据框
    """
    df_out = df.copy()
    processed_count = 0
    missing_vars = []

    for col in df_out.columns:
        # 仅处理在 num_vars 列表中的列
        if col in numeric_vars:
            if col in std_dict:
                mu = std_dict[col]['mean']
                sigma = std_dict[col]['std']
                if sigma == 0 or pd.isna(sigma):
                    sigma = 1e-8  # 防止除零

                df_out[col] = (df_out[col] - mu) / sigma
                processed_count += 1
            else:
                missing_vars.append(col)

    if missing_vars:
        print(f"    警告: 以下数值变量未在标准化参数表中找到，保持原样: {missing_vars}")

    return df_out


def apply_external_standardization_array(X, feature_names, numeric_vars, std_dict):
    """
    对numpy数组应用外部标准化（仅数值型变量）

    Parameters:
    -----------
    X : np.ndarray
        输入数组
    feature_names : list
        特征名称列表
    numeric_vars : list
        数值型变量列表
    std_dict : dict
        标准化参数字典

    Returns:
    --------
    np.ndarray : 标准化后的数组
    """
    X_out = X.copy().astype(float)
    standardized_features = []

    for i, col in enumerate(feature_names):
        if col in numeric_vars and col in std_dict:
            mu = std_dict[col]['mean']
            sigma = std_dict[col]['std']
            if sigma == 0 or pd.isna(sigma):
                sigma = 1e-8
            X_out[:, i] = (X_out[:, i] - mu) / sigma
            standardized_features.append(col)

    return X_out, standardized_features


# ==========================================
# 4. 【新增】混淆矩阵绘制函数
# ==========================================
def plot_confusion_matrix(y_true, y_pred, class_labels, title, save_path, figsize=(8, 6)):
    """
    绘制混淆矩阵热力图

    Parameters:
    -----------
    y_true : array-like
        真实标签
    y_pred : array-like
        预测标签
    class_labels : list
        类别名称列表
    title : str
        图表标题
    save_path : str
        保存路径
    figsize : tuple
        图形大小
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))

    # 绘制原始计数混淆矩阵
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels,
                ax=axes[0], cbar_kws={'shrink': 0.8}, annot_kws={'size': 14})
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_title(f'{title} - Counts', fontsize=14)
    axes[0].tick_params(axis='both', labelsize=10)

    # 绘制归一化混淆矩阵
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels,
                ax=axes[1], cbar_kws={'shrink': 0.8}, annot_kws={'size': 14})
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].set_title(f'{title} - Normalized', fontsize=14)
    axes[1].tick_params(axis='both', labelsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  混淆矩阵已保存: {save_path}")
    return cm


# ==========================================
# 5. 【新增】ROC曲线绘制函数
# ==========================================
def plot_roc_curves(y_true, y_prob, class_labels, title, save_path, figsize=(10, 8)):
    """
    绘制多分类ROC曲线（One-vs-Rest方式）

    Parameters:
    -----------
    y_true : array-like
        真实标签
    y_prob : array-like
        预测概率矩阵 (n_samples, n_classes)
    class_labels : list
        类别名称列表
    title : str
        图表标题
    save_path : str
        保存路径
    figsize : tuple
        图形大小

    Returns:
    --------
    dict : 各类别及平均AUC值
    """
    n_classes = len(class_labels)
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    # 二值化真实标签
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    # 计算每个类别的ROC曲线和AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 计算微平均ROC曲线和AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # 计算宏平均ROC曲线和AUC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # 绘制ROC曲线
    fig, ax = plt.subplots(figsize=figsize)

    # 颜色循环
    colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])

    # 绘制每个类别的ROC曲线
    for i, (color, label) in enumerate(zip(colors, class_labels)):
        ax.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{label} (AUC = {roc_auc[i]:.3f})')

    # 绘制微平均ROC曲线
    ax.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', lw=3,
            label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})')

    # 绘制宏平均ROC曲线
    ax.plot(fpr["macro"], tpr["macro"], color='navy', linestyle='--', lw=3,
            label=f'Macro-average (AUC = {roc_auc["macro"]:.3f})')

    # 绘制对角线
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Random Classifier')

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # 添加AUC信息框
    textstr = f'Macro AUC: {roc_auc["macro"]:.3f}\nMicro AUC: {roc_auc["micro"]:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.60, 0.15, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ROC曲线已保存: {save_path}")

    # 返回AUC值字典
    return roc_auc


# ==========================================
# 6. Bootstrap置信区间计算函数
# ==========================================
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


# ==========================================
# 7. 完整评价指标计算函数
# ==========================================
def calculate_full_metrics(y_true, y_pred, y_prob, class_labels):
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

    try:
        macro_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
        weighted_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
    except:
        macro_auc = weighted_auc = 0.5

    return df_metrics.round(4), macro_auc, weighted_auc


# ==========================================
# 8. 增强版SHAP分析函数
# ==========================================
def compute_shap_analysis(model, X_data, feature_names, class_labels, sample_ids, data_type, output_path):
    """计算SHAP值并保存详细结果"""
    X_df = pd.DataFrame(X_data, columns=feature_names)

    print(f"  计算 {data_type} 数据的SHAP值...")

    N_SAMPLES = min(100, X_df.shape[0])
    background = shap.kmeans(X_df, N_SAMPLES) if X_df.shape[0] > N_SAMPLES else X_df

    explainer = shap.KernelExplainer(model.predict_proba, background)
    shap_explanation = explainer(X_df)

    # 1. 平均SHAP值
    mean_abs_shap_df = pd.DataFrame(index=feature_names)
    mean_raw_shap_df = pd.DataFrame(index=feature_names)

    for class_idx, class_name in enumerate(class_labels):
        class_shap_matrix = shap_explanation.values[:, :, class_idx]
        mean_abs_shap_df[f'Mean_Abs_SHAP_{class_name}'] = np.abs(class_shap_matrix).mean(axis=0)
        mean_raw_shap_df[f'Mean_Raw_SHAP_{class_name}'] = class_shap_matrix.mean(axis=0)

    mean_abs_shap_df.to_excel(os.path.join(output_path, f'{data_type}_mean_absolute_shap.xlsx'))
    mean_raw_shap_df.to_excel(os.path.join(output_path, f'{data_type}_mean_raw_shap.xlsx'))

    # 2. 样本级别SHAP值
    sample_shap_records = []
    sample_avg_shap_records = []

    for sample_idx in range(len(X_df)):
        sample_id = sample_ids[sample_idx] if sample_idx < len(sample_ids) else sample_idx

        full_record = {'ID': sample_id}
        for class_idx, class_name in enumerate(class_labels):
            for feat_idx, feature in enumerate(feature_names):
                shap_val = shap_explanation.values[sample_idx, feat_idx, class_idx]
                full_record[f'SHAP_{feature}_{class_name}'] = shap_val
        sample_shap_records.append(full_record)

        avg_record = {'ID': sample_id}
        for feat_idx, feature in enumerate(feature_names):
            avg_shap = np.mean([shap_explanation.values[sample_idx, feat_idx, class_idx]
                                for class_idx in range(len(class_labels))])
            abs_avg_shap = np.mean([np.abs(shap_explanation.values[sample_idx, feat_idx, class_idx])
                                    for class_idx in range(len(class_labels))])
            avg_record[f'Avg_SHAP_{feature}'] = avg_shap
            avg_record[f'Avg_Abs_SHAP_{feature}'] = abs_avg_shap
        sample_avg_shap_records.append(avg_record)

    sample_shap_df = pd.DataFrame(sample_shap_records)
    sample_avg_shap_df = pd.DataFrame(sample_avg_shap_records)

    sample_shap_df.to_excel(os.path.join(output_path, f'{data_type}_sample_level_shap_values.xlsx'), index=False)
    sample_avg_shap_df.to_excel(os.path.join(output_path, f'{data_type}_sample_level_avg_shap_values.xlsx'),
                                index=False)

    # 3. 每个类别的样本SHAP值表
    for class_idx, class_name in enumerate(class_labels):
        class_sample_shap = []
        for sample_idx in range(len(X_df)):
            record = {'ID': sample_ids[sample_idx] if sample_idx < len(sample_ids) else sample_idx}
            for feat_idx, feature in enumerate(feature_names):
                record[feature] = shap_explanation.values[sample_idx, feat_idx, class_idx]
            class_sample_shap.append(record)
        class_shap_df = pd.DataFrame(class_sample_shap)
        safe_class_name = class_name.replace("/", "_").replace(" ", "_")
        class_shap_df.to_excel(os.path.join(output_path, f'{data_type}_sample_shap_{safe_class_name}.xlsx'),
                               index=False)

    # 4. SHAP Summary Plot
    try:
        shap.summary_plot(shap_explanation, X_df, class_names=class_labels, show=False)
        plt.savefig(os.path.join(output_path, f'{data_type}_shap_summary_beeswarm.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"    SHAP Summary Plot生成失败: {e}")

    # 5. 每个类别单独的SHAP图
    for class_idx, class_name in enumerate(class_labels):
        try:
            class_shap_values = shap_explanation.values[:, :, class_idx]
            shap.summary_plot(class_shap_values, X_df, show=False)
            safe_class_name = class_name.replace("/", "_").replace(" ", "_")
            plt.title(f'SHAP Summary - {class_name}')
            plt.savefig(os.path.join(output_path, f'{data_type}_shap_summary_{safe_class_name}.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"    {class_name} SHAP图生成失败: {e}")

    print(f"  {data_type} SHAP分析完成")
    return shap_explanation, mean_abs_shap_df


# ==========================================
# 主程序开始
# ==========================================
print("=" * 70)
print("MCN 风险分级预测模型 - 完整外部数据集验证")
print("（外部标准化参数版本 - 仅标准化数值型变量）")
print("=" * 70)

# ==========================================
# Step 0: 加载外部标准化参数
# ==========================================
print("\nStep 0: 加载外部标准化参数...")
print(f"  标准化参数文件: {std_params_file}")

if not os.path.exists(std_params_file):
    raise FileNotFoundError(f"找不到标准化参数文件: {std_params_file}")

std_params_dict = load_external_std_params(std_params_file)

# 检查数值型变量是否都在参数表中
missing_num_vars = [v for v in num_vars if v not in std_params_dict]
if missing_num_vars:
    print(f"  警告: 以下数值型变量未在标准化参数表中找到: {missing_num_vars}")

# ==========================================
# Step 1: 加载已训练的模型和特征列表
# ==========================================
print("\nStep 1: 加载已训练的模型和特征列表...")

# 加载模型
model_path = os.path.join(model_base_path, "final_mcn_grade_model.pkl")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"找不到模型文件: {model_path}")

final_model = joblib.load(model_path)
print(f"  模型加载成功: {model_path}")

# 加载特征列表（优先从JSON文件读取）
std_params_json = os.path.join(model_base_path, "Standardization_Parameters.json")
std_params_xlsx = os.path.join(model_base_path, "Standardization_Parameters.xlsx")
used_features_xlsx = os.path.join(model_base_path, "Used_Features.xlsx")

best_features = None

# 尝试从不同来源加载特征列表
if os.path.exists(std_params_json):
    with open(std_params_json, 'r', encoding='utf-8') as f:
        model_params = json.load(f)
    best_features = model_params.get('features', None)
    print(f"  特征列表来源: {std_params_json}")
elif os.path.exists(used_features_xlsx):
    features_df = pd.read_excel(used_features_xlsx)
    col_name = features_df.columns[0]
    best_features = features_df[col_name].dropna().tolist()
    print(f"  特征列表来源: {used_features_xlsx}")
elif os.path.exists(std_params_xlsx):
    std_df = pd.read_excel(std_params_xlsx)
    best_features = std_df['Feature'].tolist()
    print(f"  特征列表来源: {std_params_xlsx}")

if best_features is None:
    raise FileNotFoundError("无法找到模型使用的特征列表文件")

print(f"  模型特征数: {len(best_features)}")

# 区分数值型和分类型特征
numeric_features_in_model = [f for f in best_features if f in num_vars]
categorical_features_in_model = [f for f in best_features if f not in num_vars]
print(f"\n特征类型统计:")
print(f"  数值型特征 (将被标准化): {len(numeric_features_in_model)} 个")
print(f"  分类型特征 (OHE, 不标准化): {len(categorical_features_in_model)} 个")

# 加载模型系数（用于报告）
coef_path = os.path.join(model_base_path, "Pooled_Coefficients.xlsx")
if os.path.exists(coef_path):
    coef_df = pd.read_excel(coef_path, index_col=0)
    print(f"  模型系数加载成功")
else:
    coef_df = None
    print(f"  警告: 未找到模型系数文件")

# ==========================================
# Step 2: 加载并预处理外部验证数据
# ==========================================
print("\nStep 2: 加载并预处理外部验证数据...")
print(f"  加载外部数据: {external_data_file}")

raw_external_df = pd.read_excel(external_data_file)
print(f"  原始数据样本数: {len(raw_external_df)}")

df_external = dynamic_preprocess_mcn(raw_external_df)
print(f"  外部数据预处理完成，样本数: {len(df_external)}")

# 验证特征是否存在
all_external_features = [c for c in df_external.columns if c not in [target_col, id_col]]
missing_features = [f for f in best_features if f not in all_external_features]

if missing_features:
    print(f"  警告：以下 {len(missing_features)} 个特征在外部数据中不存在:")
    for f in missing_features:
        print(f"    - {f}")
    print("  这些特征将填充为0")

# 对齐数据（填充缺失特征为0）
for feat in best_features:
    if feat not in df_external.columns:
        df_external[feat] = 0

# ==========================================
# Step 3: 保存编码后的完整数据集
# ==========================================
print("\nStep 3: 保存编码后的完整数据集...")

df_external.to_excel(os.path.join(encoded_data_path, "External_Complete_Encoded.xlsx"), index=False)
print(f"  编码后数据集已保存")

# 保存使用的特征（区分类型）
features_type_df = pd.DataFrame({
    'Feature': best_features,
    'Type': ['Numeric (Standardized)' if f in num_vars else 'Categorical (OHE)' for f in best_features]
})
features_type_df.to_excel(os.path.join(results_path, "Used_Features.xlsx"), index=False)

# ==========================================
# Step 4: 外部验证（使用外部标准化参数）
# ==========================================
print("\nStep 4: 执行外部验证（仅标准化数值型变量）...")

X_external = df_external[best_features].astype(float)
y_external = df_external[target_col].values
ids_external = df_external[id_col].values if id_col in df_external.columns else np.arange(len(df_external))

# 【修改】使用外部参数进行标准化（仅数值型变量）
X_external_s = apply_external_standardization(X_external, num_vars, std_params_dict)
if X_external_s.isna().any().any():
    X_external_s = X_external_s.fillna(0)

print(f"  已标准化的数值型变量: {[f for f in best_features if f in num_vars and f in std_params_dict]}")

# 预测
probs_external = final_model.predict_proba(X_external_s)
preds_external = final_model.predict(X_external_s)

# 保存预测结果
external_pred_df = pd.DataFrame({
    'ID': ids_external,
    'True_Grade': y_external,
    'Pred_Grade': preds_external,
    'True_Grade_Label': [labels[y] for y in y_external],
    'Pred_Grade_Label': [labels[p] for p in preds_external]
})
for j, lab in enumerate(labels):
    external_pred_df[f'Prob_{lab}'] = probs_external[:, j]
external_pred_df.to_excel(os.path.join(results_path, "external_predictions.xlsx"), index=False)

# ==========================================
# Step 5: 生成混淆矩阵（增强版）
# ==========================================
print("\nStep 5: 生成混淆矩阵...")

cm_external = plot_confusion_matrix(
    y_external, preds_external, labels,
    title="External Validation (Complete Dataset)",
    save_path=os.path.join(figures_path, "External_Confusion_Matrix.png")
)

# 保存混淆矩阵数据
cm_df = pd.DataFrame(cm_external, index=labels, columns=labels)
cm_df.to_excel(os.path.join(results_path, "external_confusion_matrix.xlsx"))

# ==========================================
# Step 6: 生成ROC曲线（增强版）
# ==========================================
print("\nStep 6: 生成ROC曲线...")

auc_dict = plot_roc_curves(
    y_external, probs_external, labels,
    title="External Validation ROC Curves (Complete Dataset)",
    save_path=os.path.join(figures_path, "External_ROC_Curves.png")
)

auc_values = [auc_dict[i] for i in range(len(labels))]

# 保存AUC值
auc_df = pd.DataFrame({
    'Class': labels + ['Micro-average', 'Macro-average'],
    'AUC': auc_values + [auc_dict['micro'], auc_dict['macro']]
})
auc_df.to_excel(os.path.join(results_path, "external_auc_values.xlsx"), index=False)

print(f"  各类别AUC: {dict(zip(labels, [f'{a:.3f}' for a in auc_values]))}")
print(f"  Macro AUC: {auc_dict['macro']:.3f}")
print(f"  Micro AUC: {auc_dict['micro']:.3f}")

# ==========================================
# Step 7: 计算完整评价指标
# ==========================================
print("\nStep 7: 计算完整评价指标...")

external_metrics, macro_auc, weighted_auc = calculate_full_metrics(y_external, preds_external, probs_external, labels)
external_metrics.to_excel(os.path.join(results_path, "external_full_metrics.xlsx"))

print(f"  Macro AUC: {macro_auc:.3f}")
print(f"  Weighted AUC: {weighted_auc:.3f}")

# ==========================================
# Step 8: 计算Bootstrap置信区间
# ==========================================
print("\nStep 8: 计算Bootstrap置信区间...")
external_ci = bootstrap_metric_ci(y_external, preds_external, probs_external, labels, BOOTSTRAP_N, CI_LEVEL)
external_ci.to_excel(os.path.join(results_path, "external_metrics_with_CI.xlsx"), index=False)

# 打印外部验证结果
print("\n" + "=" * 60)
print("外部验证评价指标及95%置信区间")
print("=" * 60)
for _, row in external_ci.iterrows():
    class_name = row['Class']
    if class_name == 'Overall':
        print(f"\n【整体指标】")
        print(f"  Accuracy: {row.get('accuracy_ci', 'N/A')}")
        print(f"  Macro AUC: {row.get('macro_auc_ci', 'N/A')}")
        print(f"  Weighted AUC: {row.get('weighted_auc_ci', 'N/A')}")
    else:
        print(f"\n【{class_name}】")
        print(f"  Precision: {row.get('precision_ci', 'N/A')}")
        print(f"  Sensitivity: {row.get('sensitivity_ci', 'N/A')}")
        print(f"  Specificity: {row.get('specificity_ci', 'N/A')}")
        print(f"  F1-Score: {row.get('f1_ci', 'N/A')}")
        print(f"  AUC: {row.get('auc_ci', 'N/A')}")

print(f"\n外部验证完成，样本数: {len(y_external)}")

# ==========================================
# Step 9: 外部验证SHAP分析
# ==========================================
print("\nStep 9: 外部验证SHAP分析...")
mean_shap_ext = None

try:
    shap_external, mean_shap_ext = compute_shap_analysis(
        final_model, X_external_s.values, best_features, labels,
        list(ids_external), "external", shap_output_path
    )
except Exception as e:
    print(f"  外部验证集SHAP分析失败: {e}")

# ==========================================
# Step 10: 生成综合报告
# ==========================================
print("\nStep 10: 生成综合报告...")

# 创建类别映射DataFrame
mapping_df = pd.DataFrame({
    'Grade_Code': [0, 1, 2],
    'Grade_Label': labels,
    'Description': ['良性/低风险', '中等风险', '高风险']
})

# 保存使用的标准化参数
used_std_params = {k: v for k, v in std_params_dict.items() if k in numeric_features_in_model}
std_params_df = pd.DataFrame({
    'Feature': list(used_std_params.keys()),
    'Mean': [v['mean'] for v in used_std_params.values()],
    'Std': [v['std'] for v in used_std_params.values()]
})

report_path = os.path.join(results_path, "Comprehensive_Results_Summary.xlsx")
with pd.ExcelWriter(report_path) as writer:
    # 类别映射
    mapping_df.to_excel(writer, sheet_name='类别映射', index=False)

    # 数据集信息
    info_df = pd.DataFrame({
        'Item': ['模型来源', '标准化参数来源', '外部数据文件', '外部数据样本数',
                 '使用特征数', '数值型特征数（标准化）', '分类型特征数（不标准化）',
                 'Macro AUC', 'Weighted AUC'],
        'Value': [model_base_path, std_params_file, external_data_file, str(len(df_external)),
                  str(len(best_features)), str(len(numeric_features_in_model)),
                  str(len(categorical_features_in_model)),
                  f'{macro_auc:.4f}', f'{weighted_auc:.4f}']
    })
    info_df.to_excel(writer, sheet_name='数据集信息', index=False)

    # 最终特征（区分类型）
    features_type_df.to_excel(writer, sheet_name='最终特征', index=False)

    # 外部验证指标
    external_ci.to_excel(writer, sheet_name='外部验证指标_CI', index=False)
    external_metrics.to_excel(writer, sheet_name='外部验证完整指标')

    # AUC汇总
    auc_summary = pd.DataFrame({
        'Class': labels + ['Micro-average', 'Macro-average', 'Weighted'],
        'AUC': auc_values + [auc_dict['micro'], auc_dict['macro'], weighted_auc]
    })
    auc_summary.to_excel(writer, sheet_name='各类别AUC', index=False)

    # 模型系数
    if coef_df is not None:
        coef_df.to_excel(writer, sheet_name='模型系数')

    # 混淆矩阵
    cm_df.to_excel(writer, sheet_name='混淆矩阵')

    # 标准化参数
    std_params_df.to_excel(writer, sheet_name='标准化参数', index=False)

    # SHAP特征重要性
    if mean_shap_ext is not None:
        mean_shap_ext.to_excel(writer, sheet_name='SHAP特征重要性')

print(f"综合报告保存至: {report_path}")

# 打印输出文件清单
print("\n" + "=" * 70)
print("输出文件清单")
print("=" * 70)
print(f"""
主要输出目录: {results_path}

1. 数据特点:
   - 模型来源: {model_base_path}
   - 标准化参数来源: {std_params_file}
   - 外部数据文件: {external_data_file}
   - 外部数据样本数: {len(df_external)}
   - 数值型特征（标准化）: {len(numeric_features_in_model)} 个
   - 分类型特征（不标准化）: {len(categorical_features_in_model)} 个

2. 编码后数据集 ({encoded_data_path}):
   - External_Complete_Encoded.xlsx

3. 外部验证结果:
   - external_predictions.xlsx (包含预测结果和概率)
   - external_confusion_matrix.xlsx
   - external_auc_values.xlsx
   - external_full_metrics.xlsx
   - external_metrics_with_CI.xlsx (带Bootstrap置信区间)

4. 图表 ({figures_path}):
   - External_Confusion_Matrix.png
   - External_ROC_Curves.png

5. SHAP分析 ({shap_output_path}):
   - external_mean_absolute_shap.xlsx
   - external_mean_raw_shap.xlsx
   - external_sample_level_shap_values.xlsx
   - external_sample_level_avg_shap_values.xlsx
   - external_shap_summary_beeswarm.png
   - external_shap_summary_*.png (各类别)

6. 综合报告:
   - Comprehensive_Results_Summary.xlsx
   - Used_Features.xlsx
""")

print("\n" + "=" * 70)
print("=== MCN 完整外部数据集验证（外部标准化参数版本）全部完成 ===")
print("=" * 70)