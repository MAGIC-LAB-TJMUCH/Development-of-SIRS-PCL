# -*- coding: utf-8 -*-
"""
MICE池化模型在外部完全病例数据集上的独立验证 (外部参数标准化版本)
- 模型：原有Rubin规则池化的多项Logistic回归模型
- 验证集：外部完全病例数据集（无缺失值，无插补）

【定制修改版】
1. 数值型变量标准化：严格基于外部提供的 Parameters 文件进行 (Z-score转换)
2. AUC等评价指标的Bootstrap置信区间计算
3. 完整保存外部样本的预测结果与对应概率
"""

import pandas as pd
import numpy as np
import os
import json
import warnings
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                             classification_report, roc_auc_score)
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 路径配置 =====================
base_path = r"\五分类最终"
valid_base_path = r"\多中心胰腺囊性病变诊断与风险预测\完整数据集-验证"
base_path4 = r"\多中心胰腺囊性病变诊断与风险预测\完整数据集-五分类建模\MICE_model"
results_path = os.path.join(base_path, "Final_Analysis_Results_Std_ExternalParams")  # 确保路径与之前一致

# 标准化参数文件路径
std_params_path = os.path.join(base_path4, "Standardization_Params", "Standardization_Parameters.xlsx")

# 输出路径
output_subfolder = os.path.join(results_path, "MICE_on_Complete_External_Validation_Std_1")
os.makedirs(output_subfolder, exist_ok=True)

# 关键文件路径
# 注意：这里需要加载的是基于标准化数据训练的模型
final_model_path = os.path.join(results_path, "final_model.pkl")
selector_path = os.path.join(results_path, "selector_results.pkl")
mapping_path = os.path.join(base_path, "Final_Analysis_Results_Std_ExternalParams", "Variable_Selection_Results_Enhance.xlsx")

# 外部完全病例数据集文件名
complete_valid_file = os.path.join(valid_base_path, "df_valid_complete_cases.xlsx")

# ===================== Bootstrap置信区间参数 =====================
BOOTSTRAP_N = 1000
CI_LEVEL = 0.95
SEED = 4969


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


# ===================== 加载MICE模型必要组件 =====================
print("正在加载MICE池化模型组件...")

if not os.path.exists(final_model_path):
    raise FileNotFoundError(f"未找到模型文件: {final_model_path}，请检查路径配置。")

final_model = joblib.load(final_model_path)

if not os.path.exists(selector_path):
    # 尝试备用路径
    selector_path_alt = os.path.join(base_path, "Final_Analysis_Results_NoStd_Model", "selector_results.pkl")
    if os.path.exists(selector_path_alt):
        selector_path = selector_path_alt
    else:
        print("警告: 未找到特征筛选结果文件。")

detailed_selection = joblib.load(selector_path)
best_features = detailed_selection[detailed_selection['Final_Keep'] == 1].index.tolist()
print(f"加载特征数: {len(best_features)}")

# 加载标准化参数
try:
    std_params = load_standardization_params(std_params_path)
    print("标准化参数加载成功。")
except Exception as e:
    print(f"错误: 无法加载标准化参数文件。请确认路径: {std_params_path}")
    print(f"详细错误: {e}")
    exit()

# 加载诊断类别映射
if os.path.exists(mapping_path):
    mapping_df = pd.read_excel(mapping_path, sheet_name='Diagnosis_Mapping')
    final_mapping = dict(zip(mapping_df['Code'], mapping_df['Diagnosis_Label']))
    labels = [final_mapping[i] for i in sorted(final_mapping.keys())]
else:
    # 默认映射（防错）
    print("警告：未找到类别映射文件，使用默认映射逻辑。")
    # 这里需要根据你的实际情况补充默认逻辑，或者抛出错误
    pass

print(f"诊断类别: {labels}")

# ===================== 数据预处理函数 =====================
target_col = "Dignosis"
id_col = "key"

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


def dynamic_preprocess(df):
    df_clean = df[[c for c in whitelist if c in df.columns]].copy()

    # 结局编码
    reverse_mapping = {v: k for k, v in final_mapping.items()}
    # 处理可能的字符串/数字不匹配问题
    df_clean[target_col] = df_clean[target_col].map(reverse_mapping)
    df_clean = df_clean.dropna(subset=[target_col])
    df_clean[target_col] = df_clean[target_col].astype(int)

    # 分类变量处理
    for var in raw_categorical_vars:
        if var not in df_clean.columns:
            continue
        lvl_count = len(df_clean[var].dropna().unique())
        if lvl_count == 2:
            mapping = {'Male': 0, 'Female': 1, 0: 0, 1: 1} if var == "Gender" else {'No': 0, 'Yes': 1, 0: 0, 1: 1}
            df_clean[var] = df_clean[var].map(mapping).fillna(0)
        elif lvl_count > 2:
            df_clean = pd.get_dummies(df_clean, columns=[var], drop_first=False)

    return df_clean


# ===================== Bootstrap置信区间计算函数 =====================
def bootstrap_metric_ci(y_true, y_pred, y_prob, labels, n_bootstrap=1000, ci_level=0.95, seed=42):
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
    return df_metrics[['precision', 'Sensitivity', 'Specificity', 'f1-score', 'support', 'AUC']]


def get_metrics_with_ci(y_true, y_pred, y_prob, labels, n_bootstrap=1000, ci_level=0.95, seed=42):
    basic_metrics = get_metrics(y_true, y_pred, y_prob, labels)
    ci_df = bootstrap_metric_ci(y_true, y_pred, y_prob, labels, n_bootstrap, ci_level, seed)
    return basic_metrics, ci_df


# ===================== 加载并预处理外部完全病例数据集 =====================
print("加载外部完全病例验证数据集...")
raw_df_valid = pd.read_excel(complete_valid_file)

# 1. 基础预处理
df_valid_processed = dynamic_preprocess(raw_df_valid)

# 2. 【关键步骤】应用外部标准化参数
# 注意：即使是完全病例，也需要进行同样的标准化处理才能输入模型
df_valid_processed = apply_external_standardization(df_valid_processed, std_params, num_vars)

# 特征对齐
for col in best_features:
    if col not in df_valid_processed.columns:
        df_valid_processed[col] = 0

# 准备模型输入
X_valid = df_valid_processed[best_features].astype(float).values
y_valid = df_valid_processed[target_col].astype(int).values
ids_valid = df_valid_processed[id_col].values

print(f"外部验证样本数: {len(y_valid)}")
print(f"类别分布: {np.bincount(y_valid)}")
print("提示：已使用外部参数进行标准化处理。")

# ===================== 保存编码和数据集 =====================
print("保存编码后的验证数据集...")

# 1. 保存编码后的完整数据
encoded_data_path = os.path.join(output_subfolder, "External_Validation_Encoded_Data_Std.xlsx")
df_valid_processed.to_excel(encoded_data_path, index=False)
print(f"  编码后数据(已标准化)已保存: {encoded_data_path}")

# 2. 保存模型输入特征矩阵
model_input_df = pd.DataFrame(
    X_valid,
    columns=best_features
)
model_input_df.insert(0, id_col, ids_valid)
model_input_df.insert(1, target_col, y_valid)
model_input_df.insert(2, 'Diagnosis_Label', [labels[code] for code in y_valid])

model_input_path = os.path.join(output_subfolder, "External_Validation_Model_Input_Features_Std.xlsx")
model_input_df.to_excel(model_input_path, index=False)
print(f"  模型输入特征矩阵已保存: {model_input_path}")

# 3. 保存原始数据（用于对照）
raw_data_path = os.path.join(output_subfolder, "External_Validation_Raw_Data.xlsx")
raw_df_valid.to_excel(raw_data_path, index=False)
print(f"  原始数据已保存: {raw_data_path}")

print("数据集保存完成！\n")

# ===================== 模型预测 =====================
print("正在进行预测...")
y_prob_valid = final_model.predict_proba(X_valid)
y_pred_valid = final_model.predict(X_valid)

# ===================== 保存完整预测结果 =====================
print("保存预测结果...")

prob_cols = [f"Prob_{label}" for label in labels]

result_df = pd.DataFrame({
    'ID': ids_valid,
    'True_Label_Encoded': y_valid,
    'True_Label': [labels[code] for code in y_valid],
    'Predicted_Label_Encoded': y_pred_valid,
    'Predicted_Label': [labels[code] for code in y_pred_valid],
    'Correct_Prediction': (y_valid == y_pred_valid).astype(int)
})

result_df[prob_cols] = y_prob_valid
result_df['Prediction_Confidence'] = np.max(y_prob_valid, axis=1)

epsilon = 1e-10
entropy = -np.sum(y_prob_valid * np.log(y_prob_valid + epsilon), axis=1)
result_df['Prediction_Entropy'] = entropy

result_df = result_df.sort_values('ID').reset_index(drop=True)
result_path = os.path.join(output_subfolder, "MICE_on_complete_external_predictions.xlsx")
result_df.to_excel(result_path, index=False)

result_df.to_csv(os.path.join(output_subfolder, "MICE_on_complete_external_predictions.csv"),
                 index=False, encoding='utf-8-sig')

print(f"  预测结果已保存: {result_path}")

# ===================== 评价指标与置信区间 =====================
print(f"计算Bootstrap置信区间（{BOOTSTRAP_N}次重采样，{CI_LEVEL * 100:.0f}%置信水平）...")
print("  这可能需要几分钟...")

metrics_basic, metrics_ci = get_metrics_with_ci(
    y_valid, y_pred_valid, y_prob_valid, labels,
    n_bootstrap=BOOTSTRAP_N, ci_level=CI_LEVEL, seed=SEED
)

metrics_ci_path = os.path.join(output_subfolder, "MICE_on_complete_external_metrics_with_CI.xlsx")
metrics_ci.to_excel(metrics_ci_path, index=False)
print(f"  评价指标置信区间已保存: {metrics_ci_path}")

# ===================== 生成格式化指标摘要 =====================
print("生成格式化指标摘要...")

summary_rows = []
for _, row in metrics_ci.iterrows():
    if row['Class'] == 'Overall':
        summary_rows.append({
            '类别': 'Overall',
            'Accuracy': row.get('accuracy_ci', 'N/A'),
            'Macro_AUC': row.get('macro_auc_ci', 'N/A'),
            'Weighted_AUC': row.get('weighted_auc_ci', 'N/A')
        })
    else:
        summary_rows.append({
            '类别': row['Class'],
            'Precision': row.get('precision_ci', 'N/A'),
            'Sensitivity': row.get('sensitivity_ci', 'N/A'),
            'Specificity': row.get('specificity_ci', 'N/A'),
            'F1-Score': row.get('f1_ci', 'N/A'),
            'AUC': row.get('auc_ci', 'N/A')
        })

summary_df = pd.DataFrame(summary_rows)
summary_path = os.path.join(output_subfolder, "MICE_on_complete_external_metrics_summary.xlsx")
summary_df.to_excel(summary_path, index=False)

# ===================== 可视化 =====================
print("生成可视化图表...")

# 1. 混淆矩阵
cm = confusion_matrix(y_valid, y_pred_valid)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('MICE Pooled Model - External Complete Cases\nConfusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(output_subfolder, "MICE_on_complete_external_confusion_matrix.png"), dpi=300,
            bbox_inches='tight')
plt.close()

# 2. 归一化混淆矩阵
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(10, 8))
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('MICE Pooled Model - External Complete Cases\nNormalized Confusion Matrix (Recall)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(output_subfolder, "MICE_on_complete_external_confusion_matrix_normalized.png"), dpi=300,
            bbox_inches='tight')
plt.close()

# 3. ROC曲线（带置信区间）
y_bin = label_binarize(y_valid, classes=range(len(labels)))
plt.figure(figsize=(12, 10))

for i in range(len(labels)):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob_valid[:, i])
    roc_auc = auc(fpr, tpr)

    ci_row = metrics_ci[metrics_ci['Class'] == labels[i]]
    if len(ci_row) > 0 and 'auc_ci_lower' in ci_row.columns:
        ci_lower = ci_row['auc_ci_lower'].values[0]
        ci_upper = ci_row['auc_ci_upper'].values[0]
        label_text = f'{labels[i]} (AUC = {roc_auc:.3f}, 95%CI: {ci_lower:.3f}-{ci_upper:.3f})'
    else:
        label_text = f'{labels[i]} (AUC = {roc_auc:.3f})'

    plt.plot(fpr, tpr, linewidth=2, label=label_text)

plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('MICE Pooled Model - External Complete Cases\nMulti-class ROC Curve with 95% CI')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_subfolder, "MICE_on_complete_external_roc_curve.png"), dpi=300, bbox_inches='tight')
plt.close()

# 4. 预测概率分布
fig, axes = plt.subplots(1, len(labels), figsize=(4 * len(labels), 4))
if len(labels) == 1: axes = [axes]  # Handle case if only 1 class (unlikely)

for i, label in enumerate(labels):
    ax = axes[i]
    correct_mask = (y_valid == i) & (y_pred_valid == i)
    incorrect_mask = (y_valid == i) & (y_pred_valid != i)

    if correct_mask.sum() > 0:
        ax.hist(y_prob_valid[correct_mask, i], bins=20, alpha=0.7, label='Correct', color='green')
    if incorrect_mask.sum() > 0:
        ax.hist(y_prob_valid[incorrect_mask, i], bins=20, alpha=0.7, label='Incorrect', color='red')

    ax.set_xlabel(f'Probability of {label}')
    ax.set_ylabel('Count')
    ax.set_title(f'{label}')
    ax.legend()

plt.suptitle('Prediction Probability Distribution by Class')
plt.tight_layout()
plt.savefig(os.path.join(output_subfolder, "MICE_on_complete_external_prob_distribution.png"), dpi=300,
            bbox_inches='tight')
plt.close()

# ===================== 生成综合报告 =====================
print("生成综合报告...")

report_path = os.path.join(output_subfolder, "Comprehensive_Validation_Report.xlsx")
with pd.ExcelWriter(report_path) as writer:
    # 1. 验证概况
    overview_df = pd.DataFrame({
        '项目': ['验证数据集', '样本数', '特征数', '类别数', '标准化策略'],
        '内容': ['外部完全病例数据集', len(y_valid), len(best_features), len(labels), '基于外部参数文件(Std)']
    })
    overview_df.to_excel(writer, sheet_name='验证概况', index=False)

    # 2. 类别分布
    class_dist = pd.DataFrame({
        '类别编码': range(len(labels)),
        '类别名称': labels,
        '样本数': [np.sum(y_valid == i) for i in range(len(labels))],
        '占比(%)': [np.sum(y_valid == i) / len(y_valid) * 100 for i in range(len(labels))]
    })
    class_dist.to_excel(writer, sheet_name='类别分布', index=False)

    # 3. 评价指标
    metrics_ci.to_excel(writer, sheet_name='评价指标_CI', index=False)
    summary_df.to_excel(writer, sheet_name='指标摘要', index=False)

    # 4. 混淆矩阵
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_excel(writer, sheet_name='混淆矩阵')

    # 5. 归一化混淆矩阵
    cm_norm_df = pd.DataFrame(cm_normalized, index=labels, columns=labels)
    cm_norm_df.to_excel(writer, sheet_name='归一化混淆矩阵')

    # 6. 预测结果摘要
    pred_summary = result_df.groupby('True_Label').agg({
        'Correct_Prediction': ['sum', 'count', 'mean'],
        'Prediction_Confidence': 'mean',
        'Prediction_Entropy': 'mean'
    }).round(4)
    pred_summary.columns = ['正确数', '总数', '准确率', '平均置信度', '平均熵']
    pred_summary.to_excel(writer, sheet_name='预测摘要')

print(f"综合报告已保存: {report_path}")
print("\n=== 验证完成 (Standardization Applied) ===")