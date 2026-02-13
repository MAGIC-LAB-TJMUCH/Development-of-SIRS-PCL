
# -*- coding: utf-8 -*-
"""
胰腺囊性病变五分类预测模型 - 完整数据集敏感性分析 (外部参数标准化版本)
包含7:3内部建模、外部验证、SHAP分析全流程

【定制修改版】
1. 数值型变量标准化：严格基于外部提供的 Parameters 文件进行 (Z-score转换)
2. AUC等评价指标的Bootstrap置信区间计算
3. 保存编码后的完整内部建模和验证数据集 (已标准化)
"""

import pandas as pd
import numpy as np
import os
import json
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                             classification_report, roc_auc_score)
from sklearn.utils import resample
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib

# 忽略相关警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示

# ===================== 路径与参数配置 =====================
base_path = r"\五分类"
base_path2 = r"多中心胰腺囊性病变诊断与风险预测\完整数据集-五分类建模"
base_path3 = r"\多中心胰腺囊性病变诊断与风险预测\完整数据集-验证"
base_path4 = r"\多中心胰腺囊性病变诊断与风险预测\完整数据集-五分类建模\MICE_model"
# 标准化参数文件路径
std_params_path = os.path.join(base_path4, "Standardization_Params", "Standardization_Parameters.xlsx")

# 输出路径 (修改以区分版本)
results_path = os.path.join(base_path, "Final_Analysis_Results_Std_ExternalParams_CompleteCases")
sensitivity_results_path = os.path.join(results_path, "Sensitivity_Analysis")
encoded_data_path = os.path.join(sensitivity_results_path, "Encoded_Datasets")
shap_output_path = os.path.join(sensitivity_results_path, "SHAP_Analysis_comp")

# 创建目录
os.makedirs(results_path, exist_ok=True)
os.makedirs(sensitivity_results_path, exist_ok=True)
os.makedirs(encoded_data_path, exist_ok=True)
os.makedirs(shap_output_path, exist_ok=True)

# 加载selector结果以获取best_features (假设之前已运行过特征筛选)
# 注意：如果路径变更，请确保此文件存在，或手动指定 best_features 列表
selector_path = os.path.join(base_path, "Final_Analysis_Results_Std_ExternalParams", "selector_results.pkl")
# 如果找不到以前的结果，尝试从当前路径查找，或使用默认逻辑
if not os.path.exists(selector_path):
    # 尝试在新的输出路径找（如果是第一次运行可能需要调整）
    # 这里为了稳健性，建议先运行之前的特征筛选脚本
    print(f"警告：未找到特征筛选文件 {selector_path}，将尝试使用备用路径或需要手动指定特征。")
else:
    detailed_selection = joblib.load(selector_path)
    best_features = detailed_selection[detailed_selection['Final_Keep'] == 1].index.tolist()

# 加载类别映射（从之前保存的Excel）
mapping_path = os.path.join(base_path, "Final_Analysis_Results_Std_ExternalParams", "Variable_Selection_Results_Enhance.xlsx")
if os.path.exists(mapping_path):
    mapping_df = pd.read_excel(mapping_path, sheet_name='Diagnosis_Mapping')
    final_mapping = dict(zip(mapping_df['Code'], mapping_df['Diagnosis_Label']))
    labels = [final_mapping[i] for i in sorted(final_mapping.keys())]
else:
    # 默认映射（防错）
    print("警告：未找到类别映射文件，使用默认映射。")
    # 这里的逻辑需要确保与之前的流程一致
    pass

# 加载完全数据集
complete_data_path = os.path.join(base_path2, "df_model_complete_cases.xlsx")
raw_df_complete = pd.read_excel(complete_data_path)

# 加载外部验证数据集
complete_valid_path = os.path.join(base_path3, "df_valid_complete_cases.xlsx")
raw_df_valid = pd.read_excel(complete_valid_path)

# 核心参数
target_col = "Dignosis"
id_col = "key"
seed = 7468
TOP_N = 8
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
    "Long diameter of solid component (mm)", "Short diameter of solid component (mm)",
    "Long diameter of largest mural nodule (mm)", "Short diameter of largest mural nodule (mm)",
    "Short diameter of largest lymph node (mm)", "CA_199", "CEA", "Age"
]

whitelist = [target_col, id_col] + raw_categorical_vars + num_vars


# ===================== 标准化工具函数 =====================
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
    scaled_count = 0

    for col in num_vars:
        if col in df_scaled.columns and col in params_dict:
            mean_val = params_dict[col]['Mean']
            std_val = params_dict[col]['Std']

            if std_val == 0:
                print(f"警告: 变量 {col} 标准差为0，跳过标准化。")
            else:
                # 执行标准化 (X - Mean) / Std
                # 确保是float类型
                df_scaled[col] = df_scaled[col].astype(float)
                df_scaled[col] = (df_scaled[col] - mean_val) / std_val
                scaled_count += 1
        elif col in df_scaled.columns and col not in params_dict:
            print(f"注意: 变量 {col} 未在参数文件中找到，保持原始值。")

    return df_scaled


# ===================== 数据预处理 =====================
def dynamic_preprocess(df):
    df_clean = df[[c for c in whitelist if c in df.columns]].copy()

    # 结局变量编码
    df_clean[target_col] = df_clean[target_col].map({v: k for k, v in final_mapping.items()})
    df_clean = df_clean.dropna(subset=[target_col])
    df_clean[target_col] = df_clean[target_col].astype(int)

    # 分类变量处理
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

    return df_clean


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
    print("=" * 60)
    print("完整数据集敏感性分析 - 增强版 (基于外部参数标准化)")
    print("=" * 60)

    # ===================== Step 0: 加载标准化参数 =====================
    print("\n" + "=" * 50)
    print("Step 0: 加载标准化参数")
    print("=" * 50)

    try:
        std_params = load_standardization_params(std_params_path)
        print("标准化参数加载成功。")
    except Exception as e:
        print(f"错误: 无法加载标准化参数文件。请确认路径: {std_params_path}")
        print(f"详细错误: {e}")
        return

    # ===================== Step 1: 数据预处理与保存编码后数据集 =====================
    print("\n" + "=" * 50)
    print("Step 1: 数据预处理与保存编码后数据集 (含标准化)")
    print("=" * 50)

    # 1.1 内部完整数据集处理
    df_complete_processed = dynamic_preprocess(raw_df_complete)

    # 【关键步骤】应用外部标准化参数
    df_complete_processed = apply_external_standardization(df_complete_processed, std_params, num_vars)

    # 特征对齐（补全缺失列）
    for col in best_features:
        if col not in df_complete_processed.columns:
            df_complete_processed[col] = 0

    # 保存编码后的内部建模完整数据集
    df_complete_processed.to_excel(
        os.path.join(encoded_data_path, "Internal_Complete_Cases_Encoded_Std.xlsx"),
        index=False
    )
    print(f"  内部建模编码数据集(已标准化)已保存，样本数: {len(df_complete_processed)}")

    # 1.2 外部验证数据集处理
    df_valid_processed = dynamic_preprocess(raw_df_valid)

    # 【关键步骤】应用外部标准化参数 (与内部使用同一套参数)
    df_valid_processed = apply_external_standardization(df_valid_processed, std_params, num_vars)

    # 特征对齐
    for col in best_features:
        if col not in df_valid_processed.columns:
            df_valid_processed[col] = 0

    # 保存编码后的外部验证完整数据集
    df_valid_processed.to_excel(
        os.path.join(encoded_data_path, "External_Complete_Cases_Encoded_Std.xlsx"),
        index=False
    )
    print(f"  外部验证编码数据集(已标准化)已保存，样本数: {len(df_valid_processed)}")

    # ===================== Step 2: 7:3 分层建模 =====================
    print("\n" + "=" * 50)
    print("Step 2: 7:3 分层建模")
    print("=" * 50)

    X_complete = df_complete_processed[best_features].astype(float)
    y_complete = df_complete_processed[target_col].astype(int)
    ids_complete = df_complete_processed[id_col]

    # 7:3 分层划分
    X_train_comp, X_test_comp, y_train_comp, y_test_comp, ids_train_comp, ids_test_comp = train_test_split(
        X_complete, y_complete, ids_complete, test_size=0.3, random_state=seed, stratify=y_complete
    )

    print(f"  训练集样本数: {len(X_train_comp)}")
    print(f"  测试集样本数: {len(X_test_comp)}")
    print("  提示：数据已使用外部参数标准化。")

    # 转换为numpy array
    X_train_comp_s = X_train_comp.values
    X_test_comp_s = X_test_comp.values

    # 训练logistic回归模型
    lr_comp = LogisticRegression(
        penalty='l2', solver='lbfgs', multi_class='multinomial', max_iter=5000, random_state=seed
    )
    lr_comp.fit(X_train_comp_s, y_train_comp)

    # 保存系数表格
    coef_df = pd.DataFrame(lr_comp.coef_, columns=best_features, index=labels)
    coef_df['Intercept'] = lr_comp.intercept_
    coef_df.to_excel(os.path.join(sensitivity_results_path, "complete_cases_coefficients.xlsx"))

    # 预测
    y_pred_train_comp = lr_comp.predict(X_train_comp_s)
    y_prob_train_comp = lr_comp.predict_proba(X_train_comp_s)
    y_pred_test_comp = lr_comp.predict(X_test_comp_s)
    y_prob_test_comp = lr_comp.predict_proba(X_test_comp_s)

    # 保存模型
    complete_model_path = os.path.join(sensitivity_results_path, "complete_cases_model.pkl")
    joblib.dump(lr_comp, complete_model_path)

    print(f"  模型训练完成，已保存至: {complete_model_path}")

    # ===================== Step 3: 保存预测结果 =====================
    print("\n" + "=" * 50)
    print("Step 3: 保存预测结果")
    print("=" * 50)

    prob_cols = [f"Prob_{labels[j]}" for j in range(len(labels))]

    # 训练集结果
    train_result_comp = pd.DataFrame({
        'ID': ids_train_comp,
        'Set_Type': 'Train',
        'True_Label_Encoded': y_train_comp,
        'True_Label': [labels[code] for code in y_train_comp],
        'Predicted_Label_Encoded': y_pred_train_comp,
        'Predicted_Label': [labels[code] for code in y_pred_train_comp]
    })
    train_result_comp[prob_cols] = y_prob_train_comp

    # 测试集结果
    test_result_comp = pd.DataFrame({
        'ID': ids_test_comp,
        'Set_Type': 'Test',
        'True_Label_Encoded': y_test_comp,
        'True_Label': [labels[code] for code in y_test_comp],
        'Predicted_Label_Encoded': y_pred_test_comp,
        'Predicted_Label': [labels[code] for code in y_pred_test_comp]
    })
    test_result_comp[prob_cols] = y_prob_test_comp

    # 合并保存
    result_comp_df = pd.concat([train_result_comp, test_result_comp], ignore_index=True)
    result_comp_path = os.path.join(sensitivity_results_path, "complete_cases_predictions.xlsx")
    result_comp_df.to_excel(result_comp_path, index=False)

    print(f"  预测结果已保存")

    # ===================== Step 4: 评价指标与置信区间（测试集） =====================
    print("\n" + "=" * 50)
    print("Step 4: 计算测试集评价指标及置信区间")
    print("=" * 50)

    # 混淆矩阵
    cm_comp = confusion_matrix(y_test_comp, y_pred_test_comp)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_comp, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Complete Cases: Test Set Confusion Matrix')
    plt.savefig(os.path.join(sensitivity_results_path, "complete_cases_confusion_matrix.png"), bbox_inches='tight')
    plt.close()

    # ROC曲线
    y_test_bin_comp = label_binarize(y_test_comp, classes=range(len(labels)))
    plt.figure(figsize=(10, 8))
    for i in range(len(labels)):
        fpr, tpr, _ = roc_curve(y_test_bin_comp[:, i], y_prob_test_comp[:, i])
        roc_auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC {labels[i]} (AUC = {roc_auc_val:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(sensitivity_results_path, "complete_cases_roc_curve.png"), bbox_inches='tight')
    plt.close()

    # 评价指标（基础）
    metrics_comp = get_metrics(y_test_comp, y_pred_test_comp, y_prob_test_comp, labels)
    metrics_comp.to_excel(os.path.join(sensitivity_results_path, "complete_cases_test_metrics.xlsx"))

    # 计算置信区间
    print("  计算Bootstrap置信区间（这可能需要几分钟）...")
    metrics_comp_basic, metrics_comp_ci = get_metrics_with_ci(
        y_test_comp, y_pred_test_comp, y_prob_test_comp, labels,
        n_bootstrap=BOOTSTRAP_N, ci_level=CI_LEVEL
    )
    metrics_comp_ci.to_excel(os.path.join(sensitivity_results_path, "complete_cases_test_metrics_with_CI.xlsx"),
                             index=False)
    print(f"  测试集评价指标及置信区间已保存")

    # ===================== Step 5: SHAP分析（内部建模数据集） =====================
    print("\n" + "=" * 50)
    print("Step 5: SHAP分析")
    print("=" * 50)

    X_test_df = pd.DataFrame(X_test_comp_s, columns=best_features)

    print("  计算SHAP值（这可能需要几分钟）...")
    background_size = min(100, len(X_test_df))
    background_data = shap.kmeans(X_test_df, background_size)

    explainer = shap.KernelExplainer(lr_comp.predict_proba, background_data)
    shap_explanation = explainer(X_test_df)

    joblib.dump(shap_explanation, os.path.join(shap_output_path, "shap_explanation_test.pkl"))
    np.save(os.path.join(shap_output_path, "shap_values_test.npy"), shap_explanation.values)

    shap.summary_plot(shap_explanation, X_test_df, class_names=labels, show=False)
    plt.savefig(os.path.join(shap_output_path, "shap_summary_beeswarm.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 计算平均SHAP
    mean_shap_all_classes = pd.DataFrame(index=best_features)
    all_top_features = set()

    for class_idx, class_name in enumerate(labels):
        class_shap_values = shap_explanation.values[:, :, class_idx]
        mean_shap = np.mean(class_shap_values, axis=0)
        mean_shap_all_classes[f'Mean_SHAP_{class_name}'] = mean_shap

        abs_mean_shap = np.mean(np.abs(class_shap_values), axis=0)
        mean_shap_all_classes[f'Mean_Abs_SHAP_{class_name}'] = abs_mean_shap

        abs_shap_series = pd.Series(np.abs(mean_shap), index=best_features)
        top_features = abs_shap_series.nlargest(TOP_N).index.tolist()
        all_top_features.update(top_features)

    mean_shap_all_classes.to_excel(os.path.join(shap_output_path, "mean_shap_table_all_classes.xlsx"))

    final_top_feature_list = sorted(list(all_top_features))
    pd.Series(final_top_feature_list).to_csv(
        os.path.join(shap_output_path, "top_features_per_class_merged.csv"),
        index=False, header=['Feature']
    )

    # ===================== Step 6: 外部验证 =====================
    print("\n" + "=" * 50)
    print("Step 6: 外部验证 (使用相同外部参数标准化)")
    print("=" * 50)

    X_valid = df_valid_processed[best_features].astype(float)
    y_valid = df_valid_processed[target_col].astype(int)
    ids_valid = df_valid_processed[id_col]

    # 直接使用已标准化的数据
    X_valid_s = X_valid.values

    y_pred_valid = lr_comp.predict(X_valid_s)
    y_prob_valid = lr_comp.predict_proba(X_valid_s)

    valid_result = pd.DataFrame({
        'ID': ids_valid,
        'Set_Type': 'External Validation',
        'True_Label_Encoded': y_valid,
        'True_Label': [labels[code] for code in y_valid],
        'Predicted_Label_Encoded': y_pred_valid,
        'Predicted_Label': [labels[code] for code in y_pred_valid]
    })
    valid_result[prob_cols] = y_prob_valid
    valid_result.to_excel(os.path.join(sensitivity_results_path, "complete_cases_external_predictions.xlsx"),
                          index=False)

    cm_valid = confusion_matrix(y_valid, y_pred_valid)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_valid, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Complete Cases: External Validation Confusion Matrix')
    plt.savefig(os.path.join(sensitivity_results_path, "complete_cases_external_confusion_matrix.png"),
                bbox_inches='tight')
    plt.close()

    y_valid_bin = label_binarize(y_valid, classes=range(len(labels)))
    plt.figure(figsize=(10, 8))
    for i in range(len(labels)):
        fpr, tpr, _ = roc_curve(y_valid_bin[:, i], y_prob_valid[:, i])
        plt.plot(fpr, tpr, label=f'ROC {labels[i]} (AUC = {auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(sensitivity_results_path, "complete_cases_external_roc_curve.png"), bbox_inches='tight')
    plt.close()

    metrics_valid = get_metrics(y_valid, y_pred_valid, y_prob_valid, labels)
    metrics_valid.to_excel(os.path.join(sensitivity_results_path, "complete_cases_external_metrics.xlsx"))

    print("  计算外部验证集Bootstrap置信区间...")
    metrics_valid_basic, metrics_valid_ci = get_metrics_with_ci(
        y_valid, y_pred_valid, y_prob_valid, labels,
        n_bootstrap=BOOTSTRAP_N, ci_level=CI_LEVEL
    )
    metrics_valid_ci.to_excel(os.path.join(sensitivity_results_path, "complete_cases_external_metrics_with_CI.xlsx"),
                              index=False)
    print(f"  外部验证评价指标及置信区间已保存")

    # ===================== Step 7: 生成综合报告 =====================
    print("\n" + "=" * 50)
    print("Step 7: 生成综合结果报告")
    print("=" * 50)

    summary_report_path = os.path.join(sensitivity_results_path, "Comprehensive_Results_Summary.xlsx")

    with pd.ExcelWriter(summary_report_path) as writer:
        if 'mapping_df' in locals():
            mapping_df.to_excel(writer, sheet_name='类别映射', index=False)
        pd.DataFrame({'Selected_Features': best_features}).to_excel(writer, sheet_name='最终特征', index=False)
        metrics_comp_ci.to_excel(writer, sheet_name='内部测试集指标_CI', index=False)
        metrics_valid_ci.to_excel(writer, sheet_name='外部验证集指标_CI', index=False)
        coef_df.to_excel(writer, sheet_name='模型系数')
        mean_shap_all_classes.to_excel(writer, sheet_name='SHAP特征重要性')

    print(f"综合报告保存至: {summary_report_path}")
    print("\n=== 完整数据集敏感性分析完成 (Standardization Applied) ===")


if __name__ == "__main__":
    main()