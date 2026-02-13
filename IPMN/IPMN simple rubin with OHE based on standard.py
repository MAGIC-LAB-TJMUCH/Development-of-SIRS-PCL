# -*- coding: utf-8 -*-
"""
IPMN 风险分级模型 - 外部参数标准化版本 (External Standardization)

功能增强：
1. 读取外部 Standardization_Parameters.xlsx 文件
2. 强制使用外部参数对所有数据集进行数值标准化 (X - Mean) / Std
3. 保持原有 Rubin 规则池化、SHAP 分析等流程不变
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
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

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 变量清单与路径配置
# ==========================================
base_path = r"E:\Pancreatic cancer\多中心胰腺囊性病变诊断与风险预测\完整数据集-五分类建模\MICE_model"
valid_base_path = r"E:\Pancreatic cancer\多中心胰腺囊性病变诊断与风险预测\完整数据集-验证\MICE Valid"

# 标准化参数文件路径
std_params_path = os.path.join(base_path, "Standardization_Params_IPMN", "Standardization_Parameters.xlsx")

# 输出路径 (修改以区分版本)
results_path = os.path.join(base_path, "IPMN_Grade_Std_ExternalParams")
os.makedirs(results_path, exist_ok=True)

shap_output_path = os.path.join(results_path, "SHAP_Analysis")
os.makedirs(shap_output_path, exist_ok=True)

encoded_data_path = os.path.join(results_path, "Encoded_Datasets")
os.makedirs(encoded_data_path, exist_ok=True)

output_selection_file = os.path.join(results_path, "IPMN_Variable_Selection_Results.xlsx")
target_col = "Grade"
filter_col = "Dignosis"
filter_value = "IPMN"
id_col = "key"
m = 10
corr_threshold = 0.90
seed = 3420

# Bootstrap置信区间参数
BOOTSTRAP_N = 1000
CI_LEVEL = 0.95

# 稳定性筛选阈值（至少在多少轮中被选中）
STABILITY_THRESHOLD = 8

raw_categorical_vars = [
    "Cyst wall thickness", "Uniform Cyst wall", "Cyst wall enhancement",
    "Mural nodule enhancement", "Solid component enhancement",
    "Intracystic septations", "Uniform Septations", "Intracystic septa enhancement",
    "Capsule", "Main PD communication", "Pancreatic parenchymal atrophy",
    "Common bile duct dilation", "Tumor lesion"
    "Vascular abutment", "Enlarged lymph nodes",
    "Lesion_Head_neck", "Lesion_body_tail", "Diabetes", "Jaundice"
]

num_vars = ["Short diameter of lesion (mm)",
            "Short diameter of solid component (mm)",
            "Short diameter of largest mural nodule (mm)", "Diameter of MPD (mm)", "Diameter of CBD"
            "CA_199"]

whitelist = [target_col, filter_col, id_col] + raw_categorical_vars + num_vars
labels = ['Low Risk', 'Medium Risk', 'High Risk']

# 需要去重的冗余OHE类别（只保留一个）
REDUNDANT_SUFFIXES = [
    'Absent cyst wall',
    'Absent septations',
    'Absence of solid tissue',
    'No mural nodule',
    'No enhancement'
]


# ==========================================
# 1.5 标准化工具函数
# ==========================================
def load_standardization_params(filepath):
    """加载标准化参数文件"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"未找到标准化参数文件: {filepath}")

    print(f"正在加载标准化参数: {filepath}")
    params_df = pd.read_excel(filepath)
    # 转换为字典格式: {'FeatureName': {'Mean': x, 'Std': y}}
    params_dict = params_df.set_index('Feature')[['Mean', 'Std']].to_dict('index')
    return params_dict


def apply_external_standardization(df, params_dict, num_vars):
    """应用外部标准化参数 (Z-score)"""
    df_scaled = df.copy()

    for col in num_vars:
        if col in df_scaled.columns and col in params_dict:
            mean_val = params_dict[col]['Mean']
            std_val = params_dict[col]['Std']

            if std_val == 0:
                print(f"警告: 变量 {col} 标准差为0，跳过标准化。")
            else:
                # 执行标准化 (X - Mean) / Std
                df_scaled[col] = df_scaled[col].astype(float)
                df_scaled[col] = (df_scaled[col] - mean_val) / std_val
        elif col in df_scaled.columns and col not in params_dict:
            print(f"注意: 变量 {col} 未在参数文件中找到，保持原始值。")

    return df_scaled


# ==========================================
# 2. 预处理函数
# ==========================================
def dynamic_preprocess_ipmn(df):
    df_sub = df[df[filter_col] == filter_value].copy()
    available_cols = [c for c in whitelist if c in df_sub.columns]
    df_clean = df_sub[available_cols].copy()

    df_clean[target_col] = df_clean[target_col].astype(str).str.strip().str.lower()
    grade_map = {
        'low risk': 0, 'low': 0, 'lowrisk': 0,
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

    enhancement_vars = [
        "Cyst wall enhancement", "Mural nodule enhancement",
        "Solid component enhancement", "Intracystic septa enhancement"
    ]
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
            mapping = {'male': 0, 'female': 1, 'no': 0, 'yes': 1, '0': 0, '1': 1}
            df_clean[var] = df_clean[var].astype(str).str.lower().str.strip().map(lambda x: mapping.get(x, 0))
            df_clean[var] = df_clean[var].fillna(0).astype(float)
        else:
            df_clean = pd.get_dummies(df_clean, columns=[var], drop_first=False, dtype=float)

    return df_clean


# ==========================================
# OHE特征到原始变量的映射函数
# ==========================================
def map_ohe_to_raw(feature_name, raw_vars):
    """将OHE特征名映射回原始变量名"""
    if feature_name in raw_vars or feature_name in num_vars:
        return feature_name
    for raw_var in raw_vars:
        if feature_name.startswith(raw_var + "_"):
            return raw_var
    return None


def get_all_ohe_for_raw_var(raw_var, all_features):
    """获取原始变量对应的所有OHE特征"""
    ohe_features = []
    for feat in all_features:
        if feat == raw_var:
            ohe_features.append(feat)
        elif feat.startswith(raw_var + "_"):
            ohe_features.append(feat)
    return ohe_features


def is_redundant_absent_feature(feature_name):
    """检查是否是冗余的absent类别特征"""
    for suffix in REDUNDANT_SUFFIXES:
        if feature_name.endswith("_" + suffix):
            return True, suffix
    return False, None


def expand_and_deduplicate_features(selected_features, all_features, raw_vars):
    """将筛选后的OHE特征扩展为原始变量的所有OHE，并去除冗余的absent类别"""
    selected_raw_vars = set()
    for feat in selected_features:
        raw_var = map_ohe_to_raw(feat, raw_vars)
        if raw_var:
            selected_raw_vars.add(raw_var)

    print(f"\n  筛选后特征对应的原始变量: {sorted(selected_raw_vars)}")

    expanded_features = []
    for raw_var in selected_raw_vars:
        ohe_features = get_all_ohe_for_raw_var(raw_var, all_features)
        expanded_features.extend(ohe_features)

    expanded_features = list(dict.fromkeys(expanded_features))
    print(f"  扩展后的OHE特征数: {len(expanded_features)}")

    seen_redundant_suffixes = set()
    final_features = []
    removed_features = []

    for feat in expanded_features:
        is_redundant, suffix = is_redundant_absent_feature(feat)

        if is_redundant:
            if suffix not in seen_redundant_suffixes:
                seen_redundant_suffixes.add(suffix)
                final_features.append(feat)
                print(f"    保留冗余类别特征: {feat}")
            else:
                removed_features.append(feat)
                print(f"    移除冗余类别特征: {feat} (已存在 {suffix} 类型)")
        else:
            final_features.append(feat)

    print(f"\n  去除冗余后的最终特征数: {len(final_features)}")
    if removed_features:
        print(f"  被移除的冗余特征: {removed_features}")

    return final_features, list(selected_raw_vars), removed_features


# ==========================================
# Bootstrap置信区间计算函数
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
# 完整评价指标计算函数
# ==========================================
def calculate_full_metrics(y_true, y_pred, y_prob, class_labels):
    n_classes = len(class_labels)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
    df_metrics = pd.DataFrame(report).T
    overall_accuracy = df_metrics.loc['accuracy', 'support'] if 'accuracy' in df_metrics.index else np.nan
    df_metrics = df_metrics.drop('accuracy', errors='ignore')

    specificity_list = []
    auc_list = []
    y_bin = label_binarize(y_true, classes=range(n_classes))

    for i in range(n_classes):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificity_list.append(specificity)
        auc_score = roc_auc_score(y_bin[:, i], y_prob[:, i])
        auc_list.append(auc_score)

    df_metrics['Specificity'] = pd.Series(specificity_list + [np.nan, np.nan],
                                          index=class_labels + ['macro avg', 'weighted avg'])
    df_metrics['AUC'] = pd.Series(auc_list + [np.nan, np.nan],
                                  index=class_labels + ['macro avg', 'weighted avg'])
    df_metrics.rename(columns={'recall': 'Sensitivity'}, inplace=True)

    macro_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    weighted_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')

    return df_metrics.round(4), macro_auc, weighted_auc


def calculate_full_metrics_with_ci(y_true, y_pred, y_prob, class_labels, n_bootstrap=1000, ci_level=0.95):
    """计算完整评价指标及其Bootstrap置信区间"""
    basic_metrics, macro_auc, weighted_auc = calculate_full_metrics(y_true, y_pred, y_prob, class_labels)
    ci_df = bootstrap_metric_ci(y_true, y_pred, y_prob, class_labels, n_bootstrap, ci_level)
    return basic_metrics, ci_df


# ==========================================
# 增强版SHAP分析函数
# ==========================================
def compute_shap_analysis(model, X_data, feature_names, class_labels, sample_ids, data_type="internal",
                          output_path=None):
    """计算SHAP值并保存详细结果"""
    if output_path is None:
        output_path = shap_output_path

    X_df = pd.DataFrame(X_data, columns=feature_names)

    print(f"  计算 {data_type} 数据的SHAP值...")

    N_SAMPLES = min(100, X_df.shape[0])
    background = shap.kmeans(X_df, N_SAMPLES) if X_df.shape[0] > N_SAMPLES else X_df

    explainer = shap.KernelExplainer(model.predict_proba, background)
    shap_explanation = explainer(X_df)

    # 1. 计算每个类别的平均SHAP值
    mean_abs_shap_df = pd.DataFrame(index=feature_names)
    mean_raw_shap_df = pd.DataFrame(index=feature_names)

    for class_idx, class_name in enumerate(class_labels):
        class_shap_matrix = shap_explanation.values[:, :, class_idx]
        mean_abs_shap_df[f'Mean_Abs_SHAP_{class_name}'] = np.abs(class_shap_matrix).mean(axis=0)
        mean_raw_shap_df[f'Mean_Raw_SHAP_{class_name}'] = class_shap_matrix.mean(axis=0)

    mean_abs_shap_df.to_excel(os.path.join(output_path, f'{data_type}_mean_absolute_shap.xlsx'))
    mean_raw_shap_df.to_excel(os.path.join(output_path, f'{data_type}_mean_raw_shap.xlsx'))

    # 2. 计算每个样本的SHAP值
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

    # 3. 为每个类别生成单独的样本SHAP值表
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

    # 4. 生成SHAP Summary Plot
    try:
        shap.summary_plot(shap_explanation, X_df, class_names=class_labels, show=False)
        plt.savefig(os.path.join(output_path, f'{data_type}_shap_summary_beeswarm.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"    SHAP Summary Plot生成失败: {e}")

    print(f"  {data_type} SHAP分析完成")

    return shap_explanation, mean_abs_shap_df, sample_shap_df


# ==========================================
# 3. 数据加载与对齐
# ==========================================
print("Step 0: 加载标准化参数...")
try:
    std_params = load_standardization_params(std_params_path)
    print("标准化参数加载成功。")
except Exception as e:
    print(f"错误: 无法加载标准化参数文件。请确认路径: {std_params_path}")
    print(f"详细错误: {e}")
    exit()

print("\nStep 1: 训练集提取 IPMN 子集 (含标准化)...")
processed_dfs = []
full_column_set = set()

for i in range(1, m + 1):
    file_path = os.path.join(base_path, f"df_model_imputed_{i}.xlsx")
    raw_df = pd.read_excel(file_path)

    # 1. 基础预处理
    df_proc = dynamic_preprocess_ipmn(raw_df)

    # 2. 【关键步骤】应用外部标准化参数
    df_proc = apply_external_standardization(df_proc, std_params, num_vars)

    processed_dfs.append(df_proc)
    full_column_set.update(df_proc.columns)

all_features_index = sorted(list(full_column_set - {target_col, id_col, filter_col}))
final_aligned_data = []

for df in processed_dfs:
    df_aligned = df.copy()
    for col in full_column_set:
        if col not in df_aligned.columns:
            df_aligned[col] = 0
    final_aligned_data.append(df_aligned[[target_col, id_col] + all_features_index])

# ==========================================
# 保存所有编码后的内部建模数据集
# ==========================================
print("\nStep 1.5: 保存编码后的内部建模数据集 (已标准化)...")
all_encoded_internal = []

for i, df_encoded in enumerate(final_aligned_data):
    encoded_file = os.path.join(encoded_data_path, f"Internal_Encoded_Dataset_{i + 1}_Std.xlsx")
    df_encoded.to_excel(encoded_file, index=False)
    print(f"  保存编码后内部数据集 {i + 1}/{m}")

    df_temp = df_encoded.copy()
    df_temp['Imputation_Set'] = i + 1
    all_encoded_internal.append(df_temp)

combined_internal_encoded = pd.concat(all_encoded_internal, ignore_index=True)
combined_internal_encoded.to_excel(os.path.join(encoded_data_path, "All_Internal_Encoded_Combined_Std.xlsx"),
                                   index=False)
print(f"  合并编码后内部数据集保存完成，共 {len(combined_internal_encoded)} 条记录")

# ==========================================
# 4. 特征稳定性筛选
# ==========================================
detailed_selection = pd.DataFrame(index=all_features_index)
print("\nStep 2: 执行稳定性筛选（基于标准化数据）...")

for i, df_ready in enumerate(final_aligned_data):
    current_seed = seed + i
    print(f"  处理第 {i + 1}/{m} 组插补数据...")

    X = df_ready.drop(columns=[target_col, id_col], errors='ignore').astype(float)
    y = df_ready[target_col].astype(int)

    unique_classes = y.unique()
    if len(unique_classes) < 2:
        print(f"    警告：第 {i + 1} 组插补数据整体只有 {len(unique_classes)} 个类别，跳过本轮筛选")
        detailed_selection[f'Imputation_{i + 1}'] = 0
        continue

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y)

    train_classes = y_train.unique()
    n_train_classes = len(train_classes)

    # 此时X_train已经是标准化过的，不需要再fit_transform
    X_train_processed = X_train.copy()

    if n_train_classes < 2:
        use_rfe = False
        use_l1 = False
    else:
        use_rfe = True
        use_l1 = True

        corr_matrix = X_train_processed.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop_corr = [column for column in upper_tri.columns if any(upper_tri[column] > corr_threshold)]

        X_train_s = X_train_processed.drop(columns=to_drop_corr, errors='ignore')

    selected_count = pd.Series(0, index=all_features_index)

    # Random Forest
    try:
        rf = RandomForestClassifier(n_estimators=100, random_state=seed)
        rf.fit(X_train_s, y_train)
        rf_support = SelectFromModel(rf, prefit=True, threshold="median").get_support()
        rf_sel = pd.Series(rf_support, index=X_train_s.columns).reindex(all_features_index, fill_value=False)
        selected_count += rf_sel.astype(int)
    except Exception as e:
        print(f"    Random Forest 失败: {e}")

    # RFE
    if use_rfe:
        try:
            rfe_estimator = LogisticRegression(max_iter=2000, solver='saga', multi_class='multinomial',
                                               random_state=seed)
            n_features = 15
            rfe = RFE(estimator=rfe_estimator, n_features_to_select=n_features)
            rfe.fit(X_train_s, y_train)
            rfe_support = rfe.support_
            rfe_sel = pd.Series(rfe_support, index=X_train_s.columns).reindex(all_features_index, fill_value=False)
            selected_count += rfe_sel.astype(int)
        except Exception as e:
            print(f"    RFE 失败: {e}")

    # L1
    if use_l1:
        try:
            l1_model = LogisticRegression(penalty='l1', solver='saga', C=1.0, max_iter=2000,
                                          multi_class='multinomial', random_state=seed)
            l1_model.fit(X_train_s, y_train)
            l1_support = SelectFromModel(l1_model, prefit=True).get_support()
            l1_sel = pd.Series(l1_support, index=X_train_s.columns).reindex(all_features_index, fill_value=False)
            selected_count += l1_sel.astype(int)
        except Exception as e:
            print(f"    L1 筛选失败: {e}")

    round_selected = (selected_count >= 2).astype(int)
    detailed_selection[f'Imputation_{i + 1}'] = round_selected

print("稳定性筛选完成。")

# ==========================================
# 5. 特征扩展与冗余去除
# ==========================================
detailed_selection['Total_Frequency'] = detailed_selection.sum(axis=1)
detailed_selection['Stability_Score'] = (detailed_selection['Total_Frequency'] / m).round(3)

# 初步筛选
preliminary_features = detailed_selection[detailed_selection['Total_Frequency'] >= STABILITY_THRESHOLD].index.tolist()
print(f"\n初步筛选特征数（频次>={STABILITY_THRESHOLD}/{m}）：{len(preliminary_features)}")
print(f"初步筛选特征: {preliminary_features}")

if len(preliminary_features) == 0:
    raise ValueError(f"没有特征被选中！请降低稳定性阈值（当前为{STABILITY_THRESHOLD}）")

# 扩展为原始变量的所有OHE，并去除冗余
print("\nStep 2.5: 将筛选特征转换为原始变量并扩展OHE...")
best_features, selected_raw_vars, removed_features = expand_and_deduplicate_features(
    preliminary_features, all_features_index, raw_categorical_vars
)

print(f"\n最终建模特征数: {len(best_features)}")
print(f"最终建模特征: {best_features}")

# 更新选择结果
detailed_selection['Selected'] = detailed_selection.index.isin(best_features).astype(int)
detailed_selection = detailed_selection.sort_values(by='Total_Frequency', ascending=False)

# 保存筛选结果
with pd.ExcelWriter(output_selection_file) as writer:
    detailed_selection.to_excel(writer, sheet_name='IPMN_Grade_Variable_Stability')
    pd.DataFrame({'Preliminary_Features': preliminary_features}).to_excel(
        writer, sheet_name='Preliminary_Features', index=False)
    pd.DataFrame({'Selected_Raw_Variables': sorted(selected_raw_vars)}).to_excel(
        writer, sheet_name='Selected_Raw_Variables', index=False)
    pd.DataFrame({'Final_Model_Features': best_features}).to_excel(
        writer, sheet_name='Final_Model_Features', index=False)
    if removed_features:
        pd.DataFrame({'Removed_Redundant_Features': removed_features}).to_excel(
            writer, sheet_name='Removed_Redundant_Features', index=False)

print(f"特征筛选结果已保存至: {output_selection_file}")

# ==========================================
# 6. 模型训练与简单平均Rubin池化
# ==========================================
print("\nStep 3: 基于筛选后的特征进行模型训练与 Rubin 池化...")

all_intercepts = []
all_coefs = []
internal_test_sets = []
internal_train_sets = []

for i in range(m):
    df_ready = final_aligned_data[i]
    X = df_ready[best_features].astype(float)
    y = df_ready[target_col].astype(int)
    ids = df_ready[id_col]

    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, ids, test_size=0.3, random_state=888 + i, stratify=y)

    # X_train 已经是标准化的
    X_train_s = X_train.values
    X_test_s = X_test.values

    lr = LogisticRegression(
        penalty='l2', solver='lbfgs', multi_class='multinomial', max_iter=5000,
        random_state=seed
    )
    lr.fit(X_train_s, y_train)
    lr.classes_ = np.array([0, 1, 2])

    all_coefs.append(lr.coef_)
    all_intercepts.append(lr.intercept_)

    internal_test_sets.append((X_test_s, y_test, ids_test))
    internal_train_sets.append((X_train_s, y_train, ids_train))

    # 保存完整转换后数据
    df_full_processed = df_ready[[id_col, target_col] + best_features].copy()
    df_full_processed.columns = ['ID', 'True_Grade'] + best_features
    df_full_processed['Data_Split'] = 'Unknown'
    df_full_processed.loc[df_full_processed['ID'].isin(ids_train), 'Data_Split'] = 'Train'
    df_full_processed.loc[df_full_processed['ID'].isin(ids_test), 'Data_Split'] = 'Test'
    full_processed_path = os.path.join(encoded_data_path, f"internal_processed_data_imputation_{i + 1}_Std.xlsx")
    df_full_processed.to_excel(full_processed_path, index=False)

    # 在测试集上预测并保存完整结果
    y_pred_test = lr.predict(X_test_s)
    y_prob_test = lr.predict_proba(X_test_s)

    pred_df = pd.DataFrame({
        'ID': ids_test.values,
        'True_Grade': y_test.values,
        'Predicted_Grade': y_pred_test
    })
    for j, lab in enumerate(labels):
        pred_df[f'Prob_{lab}'] = y_prob_test[:, j]

    pred_result_path = os.path.join(results_path, f"internal_prediction_imputation_{i + 1}.xlsx")
    pred_df.to_excel(pred_result_path, index=False)

    print(f"  第{i + 1}/{m}组模型训练完成")

# Rubin 简单平均池化
pooled_coef = np.mean(all_coefs, axis=0)
pooled_intercept = np.mean(all_intercepts, axis=0)

final_model = LogisticRegression(multi_class='multinomial', solver='saga')
final_model.classes_ = np.array([0, 1, 2])
final_model.coef_ = pooled_coef
final_model.intercept_ = pooled_intercept

# 保存最终模型
joblib.dump(final_model, os.path.join(results_path, "final_ipmn_grade_model.pkl"))
print("最终池化模型已保存: final_ipmn_grade_model.pkl")

# 保存模型系数
coef_df = pd.DataFrame(pooled_coef, columns=best_features, index=labels)
coef_df.insert(0, 'Intercept', pooled_intercept)
coef_df.to_excel(os.path.join(results_path, "Pooled_Coefficients.xlsx"))

# ==========================================
# 6.5 内部验证（含置信区间）
# ==========================================
print("\nStep 3.5: 执行内部验证（含置信区间）...")

X_test_s_last, y_test_last, ids_test_last = internal_test_sets[-1]

y_pred_internal = final_model.predict(X_test_s_last)
y_prob_internal = final_model.predict_proba(X_test_s_last)

# 混淆矩阵
cm_internal = confusion_matrix(y_test_last, y_pred_internal)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_internal, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Internal Validation: Confusion Matrix (Pooled Model, Std)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(results_path, "internal_confusion_matrix.png"), dpi=200, bbox_inches='tight')
plt.close()

# ROC曲线
y_test_bin_internal = label_binarize(y_test_last, classes=[0, 1, 2])
plt.figure(figsize=(10, 8))
for i in range(len(labels)):
    fpr, tpr, _ = roc_curve(y_test_bin_internal[:, i], y_prob_internal[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{labels[i]} (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Internal Validation: Multi-class ROC Curve')
plt.legend(loc='lower right')
plt.savefig(os.path.join(results_path, "internal_roc_curve.png"), dpi=200, bbox_inches='tight')
plt.close()

# 基础评价指标
internal_metrics_df, _, _ = calculate_full_metrics(y_test_last, y_pred_internal, y_prob_internal, labels)
internal_metrics_df.to_excel(os.path.join(results_path, "internal_validation_full_metrics.xlsx"))

# 计算置信区间
print("  计算内部验证指标置信区间...")
internal_metrics_basic, internal_ci = calculate_full_metrics_with_ci(
    np.array(y_test_last), y_pred_internal, y_prob_internal, labels,
    n_bootstrap=BOOTSTRAP_N, ci_level=CI_LEVEL
)
internal_ci.to_excel(os.path.join(results_path, "internal_validation_metrics_with_CI.xlsx"), index=False)

print("内部验证完成，指标及置信区间已保存")

# ==========================================
# 内部数据SHAP分析
# ==========================================
print("\nStep 4: 内部数据SHAP分析...")

# 内部测试集SHAP分析
try:
    print("  计算内部测试集SHAP值...")
    shap_internal_test, mean_shap_test, sample_shap_test = compute_shap_analysis(
        final_model, X_test_s_last, best_features, labels,
        ids_test_last.tolist() if hasattr(ids_test_last, 'tolist') else list(ids_test_last),
        data_type="internal_test", output_path=shap_output_path
    )
    print("  内部测试集SHAP分析完成")
except Exception as e:
    print(f"  内部测试集SHAP分析失败: {e}")

# 内部训练集SHAP分析
try:
    X_train_s_last, y_train_last, ids_train_last = internal_train_sets[-1]
    print("  计算内部训练集SHAP值...")
    shap_internal_train, mean_shap_train, sample_shap_train = compute_shap_analysis(
        final_model, X_train_s_last, best_features, labels,
        ids_train_last.tolist() if hasattr(ids_train_last, 'tolist') else list(ids_train_last),
        data_type="internal_train", output_path=shap_output_path
    )
    print("  内部训练集SHAP分析完成")
except Exception as e:
    print(f"  内部训练集SHAP分析失败: {e}")

# ==========================================
# 7. 外部验证（含置信区间和SHAP，含标准化）
# ==========================================
print("\nStep 5: 执行外部验证 (含标准化)...")

all_encoded_external = []
ext_probs_list = []
ext_y_true_list = []

for i in range(1, m + 1):
    raw_v_df = pd.read_excel(os.path.join(valid_base_path, f"df_valid_imputed_{i}.xlsx"))

    # 1. 基础预处理
    df_v = dynamic_preprocess_ipmn(raw_v_df)

    if df_v.empty:
        print(f"警告：第 {i} 组外部验证数据中无 IPMN 样本，跳过。")
        continue

    # 2. 【关键步骤】应用外部标准化参数
    df_v = apply_external_standardization(df_v, std_params, num_vars)

    # 保存编码后的外部验证数据集
    df_v_save = df_v.copy()
    df_v_save['Imputation_Set'] = i
    all_encoded_external.append(df_v_save)
    df_v.to_excel(os.path.join(encoded_data_path, f"External_Encoded_Dataset_{i}_Std.xlsx"), index=False)

    df_v_aligned = df_v.reindex(columns=best_features + [target_col, id_col], fill_value=0)

    X_v = df_v_aligned[best_features].values
    y_v = df_v_aligned[target_col].values
    ids_v = df_v_aligned[id_col].values

    # 直接使用标准化后的数据进行预测
    X_v_s = X_v

    probs = final_model.predict_proba(X_v_s)
    preds = np.argmax(probs, axis=1)

    res_df = pd.DataFrame({
        'ID': ids_v,
        'True_Grade': y_v,
        'Pred_Grade': preds
    })
    for j, lab in enumerate(labels):
        res_df[f'Prob_{lab}'] = probs[:, j]
    res_df.to_excel(os.path.join(results_path, f"ext_pred_imp_{i}.xlsx"), index=False)

    # 绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    cm = confusion_matrix(y_v, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax1)
    ax1.set_title(f'External Confusion Matrix - Imp {i}')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')

    y_bin = label_binarize(y_v, classes=[0, 1, 2])
    for j in range(3):
        fpr, tpr, _ = roc_curve(y_bin[:, j], probs[:, j])
        ax2.plot(fpr, tpr, label=f'{labels[j]} (AUC = {auc(fpr, tpr):.3f})')
    ax2.plot([0, 1], [0, 1], 'k--')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title(f'External ROC - Imp {i}')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f"ext_val_plots_imp_{i}.png"), dpi=200)
    plt.close()

    ext_probs_list.append(probs)
    ext_y_true_list.append(y_v)

# 保存合并的外部验证编码数据集
if len(all_encoded_external) > 0:
    combined_external_encoded = pd.concat(all_encoded_external, ignore_index=True)
    combined_external_encoded.to_excel(os.path.join(encoded_data_path, "All_External_Encoded_Combined_Std.xlsx"),
                                       index=False)
    print(f"  外部验证编码数据集已保存，共 {len(combined_external_encoded)} 条记录")

# 池化外部验证结果
if len(ext_probs_list) > 0:
    pooled_prob = np.mean(ext_probs_list, axis=0)
    pooled_pred = np.argmax(pooled_prob, axis=1)
    y_true_pooled = ext_y_true_list[0]

    # 混淆矩阵
    cm_pooled = confusion_matrix(y_true_pooled, pooled_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_pooled, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('External Validation: Pooled Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(results_path, "external_pooled_confusion_matrix.png"), dpi=200, bbox_inches='tight')
    plt.close()

    # ROC曲线
    y_bin_pooled = label_binarize(y_true_pooled, classes=[0, 1, 2])
    plt.figure(figsize=(10, 8))
    for i in range(len(labels)):
        fpr, tpr, _ = roc_curve(y_bin_pooled[:, i], pooled_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{labels[i]} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('External Validation: Pooled Multi-class ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(results_path, "external_pooled_roc_curve.png"), dpi=200, bbox_inches='tight')
    plt.close()

    # 基础评价指标
    external_metrics_df, _, _ = calculate_full_metrics(y_true_pooled, pooled_pred, pooled_prob, labels)
    external_metrics_df.to_excel(os.path.join(results_path, "external_pooled_full_metrics.xlsx"))

    # 计算置信区间
    print("  计算外部验证指标置信区间...")
    external_metrics_basic, external_ci = calculate_full_metrics_with_ci(
        y_true_pooled, pooled_pred, pooled_prob, labels,
        n_bootstrap=BOOTSTRAP_N, ci_level=CI_LEVEL
    )
    external_ci.to_excel(os.path.join(results_path, "external_pooled_metrics_with_CI.xlsx"), index=False)

    print("外部验证完成，指标及置信区间已保存")

    # 外部验证集SHAP分析
    try:
        print("  计算外部验证集SHAP值...")
        # [修改] 使用标准化后的数值
        X_v_s_last = df_v_aligned[best_features].values
        ids_v_last = df_v_aligned[id_col].tolist()

        shap_external, mean_shap_ext, sample_shap_ext = compute_shap_analysis(
            final_model, X_v_s_last, best_features, labels,
            ids_v_last, data_type="external", output_path=shap_output_path
        )
        print("  外部验证集SHAP分析完成")
    except Exception as e:
        print(f"  外部验证集SHAP分析失败: {e}")

else:
    print("错误：所有外部验证集中均无 IPMN 样本！")

# ==========================================
# 生成综合报告
# ==========================================
print("\nStep 6: 生成综合报告...")

report_path = os.path.join(results_path, "Comprehensive_Results_Summary.xlsx")
with pd.ExcelWriter(report_path) as writer:
    pd.DataFrame({'Selected_Raw_Variables': sorted(selected_raw_vars)}).to_excel(
        writer, sheet_name='选中的原始变量', index=False)
    pd.DataFrame({'Final_Model_Features': best_features}).to_excel(
        writer, sheet_name='最终建模特征', index=False)
    if removed_features:
        pd.DataFrame({'Removed_Redundant_Features': removed_features}).to_excel(
            writer, sheet_name='移除的冗余特征', index=False)
    internal_ci.to_excel(writer, sheet_name='内部验证指标_CI', index=False)
    if 'external_ci' in dir():
        external_ci.to_excel(writer, sheet_name='外部验证指标_CI', index=False)
    coef_df.to_excel(writer, sheet_name='模型系数')

print(f"综合报告保存至: {report_path}")
print("\n=== IPMN 风险分级分析（External Params Standardization）全部完成 ===")