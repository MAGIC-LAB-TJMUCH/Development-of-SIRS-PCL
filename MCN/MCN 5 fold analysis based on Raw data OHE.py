# -*- coding: utf-8 -*-
"""
MCN 风险分级预测模型 - 5折交叉验证版本（基于MICE插补数据）
修改内容：
1. 引入外部标准化参数文件，仅对数值型变量进行标准化，不处理OHE变量。
2. 保持原有的特征筛选、AUC计算、Bootstrap置信区间和SHAP分析功能。
3. 新增：混淆矩阵绘制、ROC曲线绘制、综合结果汇总报告
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel
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
# 1. 变量清单与路径配置
# ==========================================
base_path = r"E:\Pancreatic cancer\多中心胰腺囊性病变诊断与风险预测\完整数据集-五分类建模\MICE_model"
valid_base_path = r"E:\Pancreatic cancer\多中心胰腺囊性病变诊断与风险预测\完整数据集-验证\MICE Valid"

# 标准化参数路径
std_params_path = r"E:\Pancreatic cancer\多中心胰腺囊性病变诊断与风险预测\完整数据集-五分类建模\MICE_model\Standardization_Params_MCN"
std_params_file = os.path.join(std_params_path, "Standardization_Parameters.xlsx")

results_path = os.path.join(base_path, "MCN_Grade_Final_Results_ExtStd")
os.makedirs(results_path, exist_ok=True)

shap_output_path = os.path.join(results_path, "SHAP_Analysis")
os.makedirs(shap_output_path, exist_ok=True)

encoded_data_path = os.path.join(results_path, "Encoded_Datasets")
os.makedirs(encoded_data_path, exist_ok=True)

# 新增：图表输出路径
figures_path = os.path.join(results_path, "Figures")
os.makedirs(figures_path, exist_ok=True)

output_selection_file = os.path.join(results_path, "MCN_Variable_Selection_Results.xlsx")
target_col = "Grade"
filter_col = "Dignosis"
filter_value = "MCN"
id_col = "key"
m = 10
seed = 3420

# Bootstrap置信区间参数
BOOTSTRAP_N = 1000
CI_LEVEL = 0.95

# 稳定性筛选阈值
STABILITY_THRESHOLD = 40  # 至少在40折中被选中（总共50折）

raw_categorical_vars = [
    "Gender", "Cyst wall thickness", "Uniform Cyst wall", "Cyst wall enhancement",
    "Mural nodule status", "Mural nodule enhancement", "Solid component enhancement",
    "Intracystic septations", "Uniform Septations", "Intracystic septa enhancement", "Capsule",
    "Main PD communication", "Pancreatic parenchymal atrophy", "MPD dilation",
    "Mural nodule in MPD", "Common bile duct dilation", "Vascular abutment", "Enlarged lymph nodes",
    "Distant metastasis", "Tumor lesion", "Lesion_Head_neck", "Lesion_body_tail", "Diabetes",
    "Jaundice"
]

# 数值型变量（仅对这些变量应用标准化）
num_vars = ["Short diameter of lesion (mm)",
            "Short diameter of solid component (mm)",
            "Short diameter of largest mural nodule (mm)",
            "CA-199", "CEA", "Age"]

whitelist = [target_col, filter_col, id_col] + raw_categorical_vars + num_vars
labels = ['Benign/Low Risk', 'Medium Risk', 'High Risk']

REDUNDANT_SUFFIXES = [
    'Absent cyst wall',
    'Absent septations',
    'Absence of solid tissue',
    'No mural nodule',
    'No enhancement'
]


# ==========================================
# 2. 预处理函数
# ==========================================
def dynamic_preprocess_mcn(df):
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
    df_clean = df_clean.dropna(subset=[target_col])
    df_clean[target_col] = df_clean[target_col].astype(int)

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


# ==========================================
# 辅助函数：标准化与特征处理
# ==========================================
def load_external_std_params(file_path):
    """加载外部标准化参数文件 (Feature, Mean, Std)"""
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
        return std_dict
    except Exception as e:
        print(f"  加载标准化参数失败: {e}")
        raise


def apply_external_standardization(df, numeric_vars, std_dict):
    """
    仅对数值型变量进行标准化，OHE变量保持不变
    """
    df_out = df.copy()
    processed_count = 0

    for col in df_out.columns:
        # 仅处理在 num_vars 列表中的列
        if col in numeric_vars:
            if col in std_dict:
                mu = std_dict[col]['mean']
                sigma = std_dict[col]['std']
                if sigma == 0: sigma = 1e-8  # 防止除零

                df_out[col] = (df_out[col] - mu) / sigma
                processed_count += 1
            else:
                # 如果数值变量不在参数表中，可以选择报错或跳过，这里选择警告并跳过
                pass
                # print(f"    警告: 数值变量 '{col}' 未在标准化参数表中找到，保持原样。")

    return df_out


def map_ohe_to_raw(feature_name, raw_vars):
    if feature_name in raw_vars or feature_name in num_vars: return feature_name
    for raw_var in raw_vars:
        if feature_name.startswith(raw_var + "_"): return raw_var
    return None


def get_all_ohe_for_raw_var(raw_var, all_features):
    ohe_features = []
    for feat in all_features:
        if feat == raw_var:
            ohe_features.append(feat)
        elif feat.startswith(raw_var + "_"):
            ohe_features.append(feat)
    return ohe_features


def is_redundant_absent_feature(feature_name):
    for suffix in REDUNDANT_SUFFIXES:
        if feature_name.endswith("_" + suffix): return True, suffix
    return False, None


def expand_and_deduplicate_features(selected_features, all_features, raw_vars):
    selected_raw_vars = set()
    for feat in selected_features:
        raw_var = map_ohe_to_raw(feat, raw_vars)
        if raw_var: selected_raw_vars.add(raw_var)
    print(f"\n  筛选后特征对应的原始变量: {sorted(selected_raw_vars)}")

    expanded_features = []
    for raw_var in selected_raw_vars:
        expanded_features.extend(get_all_ohe_for_raw_var(raw_var, all_features))
    expanded_features = list(dict.fromkeys(expanded_features))

    seen_redundant_suffixes = set()
    final_features = []
    removed_features = []

    for feat in expanded_features:
        is_redundant, suffix = is_redundant_absent_feature(feat)
        if is_redundant:
            if suffix not in seen_redundant_suffixes:
                seen_redundant_suffixes.add(suffix)
                final_features.append(feat)
            else:
                removed_features.append(feat)
        else:
            final_features.append(feat)
    return final_features, list(selected_raw_vars), removed_features


# ==========================================
# 评价指标函数
# ==========================================
def bootstrap_metric_ci(y_true, y_pred, y_prob, class_labels, n_bootstrap=1000, ci_level=0.95):
    n_classes = len(class_labels);
    n_samples = len(y_true)
    y_true = np.array(y_true);
    y_pred = np.array(y_pred);
    y_prob = np.array(y_prob)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    bootstrap_results = {k: {label: [] for label in class_labels} for k in
                         ['precision', 'sensitivity', 'specificity', 'f1', 'auc']}
    bootstrap_results.update({'accuracy': [], 'macro_auc': [], 'weighted_auc': []})

    np.random.seed(seed)
    for _ in range(n_bootstrap):
        indices = resample(np.arange(n_samples), replace=True, n_samples=n_samples)
        y_tb = y_true[indices];
        y_pb = y_pred[indices];
        y_prob_b = y_prob[indices];
        y_tbin_b = y_true_bin[indices]

        if len(np.unique(y_tb)) < n_classes: continue
        cm = confusion_matrix(y_tb, y_pb, labels=range(n_classes))

        for i, label in enumerate(class_labels):
            tp = cm[i, i];
            fp = cm[:, i].sum() - tp;
            fn = cm[i, :].sum() - tp;
            tn = cm.sum() - tp - fp - fn
            bootstrap_results['precision'][label].append(tp / (tp + fp) if (tp + fp) > 0 else 0)
            bootstrap_results['sensitivity'][label].append(tp / (tp + fn) if (tp + fn) > 0 else 0)
            bootstrap_results['specificity'][label].append(tn / (tn + fp) if (tn + fp) > 0 else 0)
            bootstrap_results['f1'][label].append(
                2 * bootstrap_results['precision'][label][-1] * bootstrap_results['sensitivity'][label][-1] / (
                            bootstrap_results['precision'][label][-1] + bootstrap_results['sensitivity'][label][
                        -1]) if (bootstrap_results['precision'][label][-1] + bootstrap_results['sensitivity'][label][
                    -1]) > 0 else 0)
            try:
                bootstrap_results['auc'][label].append(roc_auc_score(y_tbin_b[:, i], y_prob_b[:, i]))
            except:
                pass

        bootstrap_results['accuracy'].append(np.mean(y_tb == y_pb))
        try:
            bootstrap_results['macro_auc'].append(roc_auc_score(y_tbin_b, y_prob_b, average='macro'))
            bootstrap_results['weighted_auc'].append(roc_auc_score(y_tbin_b, y_prob_b, average='weighted'))
        except:
            pass

    alpha = 1 - ci_level;
    lower_p = alpha / 2 * 100;
    upper_p = (1 - alpha / 2) * 100
    ci_results = []
    for label in class_labels:
        row = {'Class': label}
        for metric in ['precision', 'sensitivity', 'specificity', 'f1', 'auc']:
            vals = bootstrap_results[metric][label]
            if vals:
                row[f'{metric}_estimate'] = np.mean(vals)
                row[
                    f'{metric}_ci'] = f"{np.mean(vals):.3f} ({np.percentile(vals, lower_p):.3f}-{np.percentile(vals, upper_p):.3f})"
            else:
                row[f'{metric}_estimate'] = np.nan
        ci_results.append(row)

    overall = {'Class': 'Overall'}
    for m_key in ['accuracy', 'macro_auc', 'weighted_auc']:
        if bootstrap_results[m_key]:
            overall[f'{m_key}_estimate'] = np.mean(bootstrap_results[m_key])
            overall[
                f'{m_key}_ci'] = f"{np.mean(bootstrap_results[m_key]):.3f} ({np.percentile(bootstrap_results[m_key], lower_p):.3f}-{np.percentile(bootstrap_results[m_key], upper_p):.3f})"
    ci_results.append(overall)
    return pd.DataFrame(ci_results)


def calculate_full_metrics(y_true, y_pred, y_prob, class_labels):
    n_classes = len(class_labels)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
    df_metrics = pd.DataFrame(report).T.drop('accuracy', errors='ignore')

    specificity_list = []
    auc_list = []
    y_bin = label_binarize(y_true, classes=range(n_classes))

    for i in range(n_classes):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        specificity_list.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        try:
            auc_list.append(roc_auc_score(y_bin[:, i], y_prob[:, i]))
        except:
            auc_list.append(0.5)

    df_metrics['Specificity'] = specificity_list + [np.nan, np.nan]
    df_metrics['AUC'] = auc_list + [np.nan, np.nan]
    df_metrics.rename(columns={'recall': 'Sensitivity'}, inplace=True)

    try:
        macro_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
        weighted_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
    except:
        macro_auc = weighted_auc = 0.5

    return df_metrics.round(4), macro_auc, weighted_auc


# ==========================================
# 新增：混淆矩阵绘制函数
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
                ax=axes[0], cbar_kws={'shrink': 0.8})
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_title(f'{title} - Counts', fontsize=14)
    axes[0].tick_params(axis='both', labelsize=10)

    # 绘制归一化混淆矩阵
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels,
                ax=axes[1], cbar_kws={'shrink': 0.8})
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
# 新增：ROC曲线绘制函数
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


def compute_shap_analysis(model, X_data, feature_names, class_labels, sample_ids, data_type, output_path):
    X_df = pd.DataFrame(X_data, columns=feature_names)
    print(f"  计算 {data_type} 数据的SHAP值...")

    # 使用 KMeans 汇总背景数据以加速
    N_SAMPLES = min(100, X_df.shape[0])
    background = shap.kmeans(X_df, N_SAMPLES) if X_df.shape[0] > N_SAMPLES else X_df

    explainer = shap.KernelExplainer(model.predict_proba, background)
    shap_explanation = explainer(X_df)

    mean_abs_shap_df = pd.DataFrame(index=feature_names)
    for class_idx, class_name in enumerate(class_labels):
        mean_abs_shap_df[f'Mean_Abs_SHAP_{class_name}'] = np.abs(shap_explanation.values[:, :, class_idx]).mean(axis=0)

    mean_abs_shap_df.to_excel(os.path.join(output_path, f'{data_type}_mean_absolute_shap.xlsx'))

    # 保存Summary Plot
    try:
        plt.figure()
        shap.summary_plot(shap_explanation, X_df, class_names=class_labels, show=False)
        plt.savefig(os.path.join(output_path, f'{data_type}_shap_summary_beeswarm.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except:
        pass

    return shap_explanation, mean_abs_shap_df


# ==========================================
# 主程序
# ==========================================
print("=" * 70)
print("MCN 风险分级 - 外部参数标准化版本")
print("=" * 70)

# 0. 加载标准化参数
print("\nStep 0: 加载外部标准化参数...")
if not os.path.exists(std_params_file):
    raise FileNotFoundError(f"找不到标准化参数文件: {std_params_file}")
std_params_dict = load_external_std_params(std_params_file)

# 1. 数据加载
print("\nStep 1: 训练集提取 MCN 子集...")
processed_dfs = []
full_column_set = set()

for i in range(1, m + 1):
    file_path = os.path.join(base_path, f"df_model_imputed_{i}.xlsx")
    if not os.path.exists(file_path): continue
    df_proc = dynamic_preprocess_mcn(pd.read_excel(file_path))
    processed_dfs.append(df_proc)
    full_column_set.update(df_proc.columns)

all_features_index = sorted(list(full_column_set - {target_col, id_col, filter_col}))
final_aligned_data = []
for df in processed_dfs:
    df_aligned = df.copy()
    for col in full_column_set:
        if col not in df_aligned.columns: df_aligned[col] = 0
    final_aligned_data.append(df_aligned[[target_col, id_col] + all_features_index])

# 保存编码后数据
print("\nStep 1.5: 保存编码后数据...")
all_encoded_internal = []
for i, df_encoded in enumerate(final_aligned_data):
    df_temp = df_encoded.copy()
    df_temp['Imputation_Set'] = i + 1
    all_encoded_internal.append(df_temp)
pd.concat(all_encoded_internal, ignore_index=True).to_excel(
    os.path.join(encoded_data_path, "All_Internal_Encoded_Combined.xlsx"), index=False)

# 2. 特征筛选 (使用外部标准化参数)
print(f"\nStep 2: 开始 5 折交叉验证特征筛选（共 {m * 5} 折）...")
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
detailed_selection = pd.DataFrame(0, index=all_features_index, columns=[f'Fold_{k + 1}' for k in range(n_folds * m)])
fold_counter = 0

for imp_idx, df_ready in enumerate(final_aligned_data):
    X = df_ready.drop(columns=[target_col, id_col], errors='ignore').astype(float)
    y = df_ready[target_col].astype(int)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        fold_counter += 1
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]

        # 【修改】使用外部参数进行标准化 (仅数值型)
        X_train_s = apply_external_standardization(X_train, num_vars, std_params_dict)
        if X_train_s.isna().any().any(): X_train_s = X_train_s.fillna(0)

        selected_count = pd.Series(0, index=all_features_index)

        # 1. RF
        try:
            rf = RandomForestClassifier(n_estimators=200, random_state=seed + fold_counter)
            rf.fit(X_train_s, y_train)
            selected_count += pd.Series(SelectFromModel(rf, threshold="median", prefit=True).get_support(),
                                        index=X_train_s.columns).reindex(all_features_index, fill_value=False).astype(
                int)
        except:
            pass

        # 2. RFE
        try:
            rfe = RFE(LogisticRegression(max_iter=2000, solver='saga', multi_class='multinomial', random_state=seed),
                      n_features_to_select=min(12, X_train_s.shape[1]))
            rfe.fit(X_train_s, y_train)
            selected_count += pd.Series(rfe.support_, index=X_train_s.columns).reindex(all_features_index,
                                                                                       fill_value=False).astype(int)
        except:
            pass

        # 3. Lasso
        try:
            l1 = LogisticRegression(penalty='l1', solver='saga', C=0.6, max_iter=2000, multi_class='multinomial',
                                    random_state=seed)
            l1.fit(X_train_s, y_train)
            selected_count += pd.Series(SelectFromModel(l1, prefit=True).get_support(),
                                        index=X_train_s.columns).reindex(all_features_index, fill_value=False).astype(
                int)
        except:
            pass

        detailed_selection[f'Fold_{fold_counter}'] = (selected_count >= 2).astype(int)

# 汇总筛选结果
detailed_selection['Total_Frequency'] = detailed_selection.sum(axis=1)
preliminary_features = detailed_selection[detailed_selection['Total_Frequency'] >= STABILITY_THRESHOLD].index.tolist()
print(f"初步筛选特征数: {len(preliminary_features)}")

# 扩展并去重特征
print("\nStep 2.5: 扩展特征并去重...")
best_features, selected_raw_vars, removed_features = expand_and_deduplicate_features(preliminary_features,
                                                                                     all_features_index,
                                                                                     raw_categorical_vars)
print(f"最终建模特征数: {len(best_features)}")

# 保存筛选报告
with pd.ExcelWriter(output_selection_file) as writer:
    detailed_selection.to_excel(writer, sheet_name='Stability_Result')
    pd.DataFrame({'Final_Model_Features': best_features}).to_excel(writer, sheet_name='Final_Features', index=False)

# 3. 模型训练 (使用外部标准化参数)
print("\nStep 3: 5折交叉验证训练模型并Rubin池化...")
all_coefs = [];
all_intercepts = [];
all_train_data = []
all_test_preds = [];
all_test_probs = [];
all_test_true = [];
all_test_ids = []

for imp_idx, df_ready in enumerate(final_aligned_data):
    X = df_ready[best_features].astype(float)
    y = df_ready[target_col].astype(int)
    ids = df_ready[id_col]

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        ids_train, ids_test = ids.iloc[train_idx], ids.iloc[test_idx]

        # 【修改】使用外部参数进行标准化
        X_train_s = apply_external_standardization(X_train, num_vars, std_params_dict)
        X_test_s = apply_external_standardization(X_test, num_vars, std_params_dict)

        lr = LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial', max_iter=5000,
                                random_state=seed)
        lr.fit(X_train_s, y_train)

        all_coefs.append(lr.coef_)
        all_intercepts.append(lr.intercept_)

        all_test_preds.extend(lr.predict(X_test_s))
        all_test_probs.extend(lr.predict_proba(X_test_s))
        all_test_true.extend(y_test)
        all_test_ids.extend(ids_test)

        all_train_data.append((X_train_s, y_train, ids_train))

# Rubin池化
final_model = LogisticRegression(multi_class='multinomial')
final_model.classes_ = np.array([0, 1, 2])
final_model.coef_ = np.mean(all_coefs, axis=0)
final_model.intercept_ = np.mean(all_intercepts, axis=0)
joblib.dump(final_model, os.path.join(results_path, "final_mcn_grade_model.pkl"))

# 保存使用的标准化参数
pd.DataFrame(std_params_dict).T.reset_index().rename(columns={'index': 'Feature'}).to_excel(
    os.path.join(results_path, "Used_Standardization_Parameters.xlsx"), index=False
)

# 创建模型系数DataFrame
coef_df = pd.DataFrame(
    final_model.coef_.T,
    index=best_features,
    columns=[f'Coef_{label}' for label in labels]
)
coef_df['Intercept'] = ''  # 占位列
for i, label in enumerate(labels):
    coef_df.loc[coef_df.index[0], f'Intercept_{label}'] = final_model.intercept_[i]
coef_df.to_excel(os.path.join(results_path, "Model_Coefficients.xlsx"))

# 4. 内部验证指标
y_true_cv = np.array(all_test_true);
y_pred_cv = np.array(all_test_preds);
y_prob_cv = np.array(all_test_probs)
internal_metrics, _, _ = calculate_full_metrics(y_true_cv, y_pred_cv, y_prob_cv, labels)
internal_metrics.to_excel(os.path.join(results_path, "internal_5fold_metrics.xlsx"))

print("\n计算内部验证置信区间...")
internal_ci = bootstrap_metric_ci(y_true_cv, y_pred_cv, y_prob_cv, labels, BOOTSTRAP_N, CI_LEVEL)
internal_ci.to_excel(os.path.join(results_path, "internal_5fold_metrics_with_CI.xlsx"), index=False)

# 新增：绘制内部验证混淆矩阵
print("\nStep 3.5: 绘制内部验证混淆矩阵和ROC曲线...")
internal_cm = plot_confusion_matrix(
    y_true_cv, y_pred_cv, labels,
    title="Internal Validation (5-Fold CV)",
    save_path=os.path.join(figures_path, "Internal_Confusion_Matrix.png")
)

# 新增：绘制内部验证ROC曲线
internal_auc_dict = plot_roc_curves(
    y_true_cv, y_prob_cv, labels,
    title="Internal Validation ROC Curves (5-Fold CV)",
    save_path=os.path.join(figures_path, "Internal_ROC_Curves.png")
)

# 5. SHAP分析 (使用标准化后的数据)
print("\nStep 4: 内部数据SHAP分析...")
mean_shap_all_classes = None
if len(all_train_data) > 0:
    # 取最后一折的训练数据（已经是标准化过的）
    X_train_last, _, ids_train_last = all_train_data[-1]
    try:
        _, mean_shap_all_classes = compute_shap_analysis(final_model, X_train_last, best_features, labels,
                                                         list(ids_train_last), "internal_train", shap_output_path)
    except Exception as e:
        print(f"SHAP Error: {e}")

# 6. 外部验证 (使用外部标准化参数)
print("\nStep 5: 外部验证...")
ext_probs_list = [];
ext_y_true_list = [];
all_encoded_external = []
external_ci = None  # 初始化

for i in range(1, m + 1):
    file_path = os.path.join(valid_base_path, f"df_valid_imputed_{i}.xlsx")
    if not os.path.exists(file_path): continue

    df_v = dynamic_preprocess_mcn(pd.read_excel(file_path))
    if df_v.empty: continue

    # 保存编码数据
    df_v_save = df_v.copy();
    df_v_save['Imputation_Set'] = i
    all_encoded_external.append(df_v_save)
    df_v.to_excel(os.path.join(encoded_data_path, f"External_Encoded_Dataset_{i}.xlsx"), index=False)

    # 对齐并标准化
    df_v_aligned = df_v.reindex(columns=best_features + [target_col, id_col], fill_value=0)
    X_v = df_v_aligned[best_features]
    y_v = df_v_aligned[target_col].values
    ids_v = df_v_aligned[id_col].values

    # 【修改】应用外部标准化
    X_v_s = apply_external_standardization(X_v, num_vars, std_params_dict)

    probs = final_model.predict_proba(X_v_s)
    preds = np.argmax(probs, axis=1)

    res_df = pd.DataFrame({'ID': ids_v, 'True_Grade': y_v, 'Pred_Grade': preds})
    for j, lab in enumerate(labels): res_df[f'Prob_{lab}'] = probs[:, j]
    res_df.to_excel(os.path.join(results_path, f"ext_pred_imp_{i}.xlsx"), index=False)

    ext_probs_list.append(probs)
    ext_y_true_list.append(y_v)

if all_encoded_external:
    pd.concat(all_encoded_external, ignore_index=True).to_excel(
        os.path.join(encoded_data_path, "All_External_Encoded_Combined.xlsx"), index=False)

if ext_probs_list:
    pooled_prob = np.mean(ext_probs_list, axis=0)
    pooled_pred = np.argmax(pooled_prob, axis=1)
    y_true_pooled = ext_y_true_list[0]

    external_metrics, _, _ = calculate_full_metrics(y_true_pooled, pooled_pred, pooled_prob, labels)
    external_metrics.to_excel(os.path.join(results_path, "external_pooled_metrics.xlsx"))

    print("\n计算外部验证置信区间...")
    external_ci = bootstrap_metric_ci(y_true_pooled, pooled_pred, pooled_prob, labels, BOOTSTRAP_N, CI_LEVEL)
    external_ci.to_excel(os.path.join(results_path, "external_pooled_metrics_with_CI.xlsx"), index=False)

    # 新增：绘制外部验证混淆矩阵
    print("\nStep 5.5: 绘制外部验证混淆矩阵和ROC曲线...")
    external_cm = plot_confusion_matrix(
        y_true_pooled, pooled_pred, labels,
        title="External Validation (Pooled)",
        save_path=os.path.join(figures_path, "External_Confusion_Matrix.png")
    )

    # 新增：绘制外部验证ROC曲线
    external_auc_dict = plot_roc_curves(
        y_true_pooled, pooled_prob, labels,
        title="External Validation ROC Curves (Pooled)",
        save_path=os.path.join(figures_path, "External_ROC_Curves.png")
    )

    # 外部SHAP (取最后一组数据演示)
    try:
        compute_shap_analysis(final_model, X_v_s, best_features, labels, list(ids_v), "external", shap_output_path)
    except:
        pass

# ==========================================
# 7. 新增：综合结果汇总报告
# ==========================================
print("\nStep 6: 生成综合结果汇总报告...")

# 创建类别映射DataFrame
mapping_df = pd.DataFrame({
    'Grade_Code': [0, 1, 2],
    'Grade_Label': labels,
    'Description': ['良性/低风险', '中等风险', '高风险']
})

# 确保SHAP结果存在
if mean_shap_all_classes is None:
    mean_shap_all_classes = pd.DataFrame({'Note': ['SHAP分析未成功完成']})

# 生成综合报告
summary_report_path = os.path.join(results_path, "Comprehensive_Results_Summary.xlsx")
with pd.ExcelWriter(summary_report_path) as writer:
    mapping_df.to_excel(writer, sheet_name='类别映射', index=False)
    pd.DataFrame({'Selected_Features': best_features}).to_excel(writer, sheet_name='最终特征', index=False)
    internal_ci.to_excel(writer, sheet_name='内部验证指标_CI', index=False)
    if external_ci is not None:
        external_ci.to_excel(writer, sheet_name='外部验证指标_CI', index=False)
    else:
        pd.DataFrame({'Note': ['无外部验证数据']}).to_excel(writer, sheet_name='外部验证指标_CI', index=False)
    coef_df.to_excel(writer, sheet_name='模型系数')
    mean_shap_all_classes.to_excel(writer, sheet_name='SHAP特征重要性')

    # 额外添加：内部和外部的混淆矩阵
    pd.DataFrame(internal_cm, index=labels, columns=labels).to_excel(writer, sheet_name='内部验证混淆矩阵')
    if ext_probs_list:
        pd.DataFrame(external_cm, index=labels, columns=labels).to_excel(writer, sheet_name='外部验证混淆矩阵')

print(f"\n综合报告已保存: {summary_report_path}")
print("\n" + "=" * 70)
print("程序执行完毕。")
print("=" * 70)
print(f"\n输出文件路径汇总:")
print(f"  - 主结果目录: {results_path}")
print(f"  - 图表目录: {figures_path}")
print(f"  - SHAP分析目录: {shap_output_path}")
print(f"  - 编码数据目录: {encoded_data_path}")
print(f"  - 综合报告: {summary_report_path}")