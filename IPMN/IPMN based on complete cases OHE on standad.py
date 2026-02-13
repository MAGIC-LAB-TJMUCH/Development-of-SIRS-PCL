# -*- coding: utf-8 -*-
"""
IPMN 完整数据集（Complete Cases）建模与验证 - 外部参数标准化版本

变更点：
1. 移除内部计算的 StandardScaler
2. 读取外部 Standardization_Parameters.xlsx
3. 对内部训练集、测试集和外部验证集统一使用外部参数进行标准化
"""

import pandas as pd
import numpy as np
import os
import json
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                             classification_report, roc_auc_score)
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 路径与参数配置
# ==========================================
base_path = r"E:\Pancreatic cancer\多中心胰腺囊性病变诊断与风险预测\完整数据集-五分类建模\MICE_model"
valid_base_path = r"E:\Pancreatic cancer\多中心胰腺囊性病变诊断与风险预测\完整数据集-验证"
base_path2 = r"E:\Pancreatic cancer\多中心胰腺囊性病变诊断与风险预测\完整数据集-五分类建模"

# 简单平均Rubin版本的结果路径（用于加载筛选的特征）
rawdata_results_path = os.path.join(base_path, "IPMN_Grade_Std_ExternalParams")

# 【新增】标准化参数文件路径
std_params_path = r"E:\Pancreatic cancer\多中心胰腺囊性病变诊断与风险预测\完整数据集-五分类建模\MICE_model\Standardization_Params_IPMN\Standardization_Parameters.xlsx"

# 本次完整数据集分析的输出路径
results_path = os.path.join(base_path, "IPMN_Complete_Cases_Features_Std_External")
os.makedirs(results_path, exist_ok=True)

# SHAP输出路径
shap_output_path = os.path.join(results_path, "SHAP_Analysis")
os.makedirs(shap_output_path, exist_ok=True)

# 编码数据集输出路径
encoded_data_path = os.path.join(results_path, "Encoded_Datasets")
os.makedirs(encoded_data_path, exist_ok=True)

# 数据文件路径
internal_data_file = os.path.join(base_path2, "df_IPMN_complete.xlsx")
external_data_file = os.path.join(valid_base_path, "df_IPMN_valid_comp.xlsx")

# 核心参数
target_col = "Grade"
id_col = "key"
seed = 3774
BOOTSTRAP_N = 1000
CI_LEVEL = 0.95

labels = ['Low Risk', 'Medium Risk', 'High Risk']

# 分类变量（用于预处理）
raw_categorical_vars = [
    "Gender", "Cyst wall thickness", "Uniform Cyst wall", "Cyst wall enhancement",
    "Mural nodule status", "Mural nodule enhancement", "Solid component enhancement",
    "Intracystic septations", "Uniform Septations", "Intracystic septa enhancement",
    "Capsule", "Main PD communication", "Pancreatic parenchymal atrophy",
    "Mural nodule in MPD", "Common bile duct dilation",
    "Vascular abutment", "Enlarged lymph nodes", "Distant metastasis",
    "Tumor lesion", "Lesion_Head_neck", "Lesion_body_tail", "Diabetes", "Jaundice"
]

# 数值变量 (请确保名称与Excel中的Feature列一致，或能对应上)
num_vars = ["Long diameter of lesion (mm)", "Short diameter of lesion (mm)",
            "Long diameter of solid component (mm)", "Short diameter of solid component (mm)",
            "Long diameter of largest mural nodule (mm)",
            "Short diameter of largest mural nodule (mm)", "Diameter of MPD (mm)",
            "CA-199", "CEA", "Age"]


# 注意：CA-199 在某些文件中可能是 CA_199，下文函数会尝试自动匹配

# ==========================================
# 1.5 标准化工具函数
# ==========================================
def load_standardization_params(filepath):
    """加载标准化参数文件"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"未找到标准化参数文件: {filepath}")

    print(f"正在加载标准化参数: {filepath}")
    try:
        params_df = pd.read_excel(filepath)
        # 规范化列名 (首字母大写)
        params_df.columns = [c.capitalize() for c in params_df.columns]

        # 确保包含必要列
        if 'Feature' not in params_df.columns:
            # 如果没有Feature列，尝试使用第一列
            params_df.rename(columns={params_df.columns[0]: 'Feature'}, inplace=True)

        # 转换为字典格式: {'FeatureName': {'Mean': x, 'Std': y}}
        params_dict = params_df.set_index('Feature')[['Mean', 'Std']].to_dict('index')
        return params_dict
    except Exception as e:
        print(f"读取标准化参数文件失败: {e}")
        raise


def apply_external_standardization(df, params_dict, num_vars):
    """应用外部标准化参数 (Z-score)"""
    df_scaled = df.copy()

    for col in num_vars:
        # 处理列名可能的不一致 (e.g. CA-199 vs CA_199)
        target_col = col
        if col not in df_scaled.columns:
            if col.replace('-', '_') in df_scaled.columns:
                target_col = col.replace('-', '_')
            elif col.replace('_', '-') in df_scaled.columns:
                target_col = col.replace('_', '-')

        if target_col in df_scaled.columns:
            # 查找参数字典中的key
            param_key = None
            if col in params_dict:
                param_key = col
            elif target_col in params_dict:
                param_key = target_col
            elif col.replace('-', '_') in params_dict:
                param_key = col.replace('-', '_')

            if param_key:
                mean_val = params_dict[param_key]['Mean']
                std_val = params_dict[param_key]['Std']

                if std_val == 0:
                    print(f"  警告: 变量 {col} 标准差为0，跳过标准化。")
                else:
                    df_scaled[target_col] = df_scaled[target_col].astype(float)
                    df_scaled[target_col] = (df_scaled[target_col] - mean_val) / std_val
            else:
                print(f"  注意: 变量 {col} 未在参数文件中找到，保持原始值。")

    return df_scaled


# ==========================================
# 2. 加载筛选特征
# ==========================================
print("=" * 60)
print("Step 1: 加载简单平均Rubin版本筛选的特征")
print("=" * 60)

# 从简单平均Rubin版本的结果中加载特征
feature_file = os.path.join(rawdata_results_path, "IPMN_Variable_Selection_Results.xlsx")
if os.path.exists(feature_file):
    feature_df = pd.read_excel(feature_file, sheet_name='Final_Model_Features')
    best_features = feature_df['Final_Model_Features'].tolist()
    print(f"成功加载特征数: {len(best_features)}")
    print(f"特征列表: {best_features}")
else:
    raise FileNotFoundError(f"未找到特征文件: {feature_file}\n请先运行简单平均Rubin版本的IPMN代码生成特征筛选结果。")


# ==========================================
# 3. 预处理函数
# ==========================================
def dynamic_preprocess_ipmn(df):
    """IPMN数据预处理函数"""
    df_clean = df.copy()

    # 标签统一处理
    df_clean[target_col] = df_clean[target_col].astype(str).str.strip().str.lower()
    grade_map = {
        'low risk': 0, 'low': 0, 'lowrisk': 0,
        'medium risk': 1, 'medium': 1, 'mediumrisk': 1,
        'high risk': 2, 'high': 2, 'highrisk': 2,
    }
    df_clean[target_col] = df_clean[target_col].map(grade_map)
    df_clean = df_clean.dropna(subset=[target_col])
    df_clean[target_col] = df_clean[target_col].astype(int)

    # 变量编码
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
# 4. Bootstrap置信区间计算函数
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
# 5. 评价指标计算函数
# ==========================================
def calculate_full_metrics(y_true, y_pred, y_prob, class_labels):
    """计算完整评价指标"""
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
# 6. SHAP分析函数
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
        class_shap_df.to_excel(
            os.path.join(output_path, f'{data_type}_sample_shap_{class_name.replace(" ", "_")}.xlsx'), index=False)

    # 4. SHAP Summary Plot
    try:
        shap.summary_plot(shap_explanation, X_df, class_names=class_labels, show=False)
        plt.savefig(os.path.join(output_path, f'{data_type}_shap_summary_beeswarm.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"    SHAP Summary Plot生成失败: {e}")

    print(f"  {data_type} SHAP分析完成")
    return shap_explanation, mean_abs_shap_df


# ==========================================
# 7. 加载并预处理内部数据
# ==========================================
print("\n" + "=" * 60)
print("Step 2: 加载并预处理内部完整数据集")
print("=" * 60)

# 【新增】加载外部标准化参数
try:
    std_params = load_standardization_params(std_params_path)
    print("标准化参数加载成功。")
except Exception as e:
    print(f"错误: 无法加载标准化参数文件。请确认路径: {std_params_path}")
    exit()

df_internal_raw = pd.read_excel(internal_data_file)
print(f"原始内部数据集样本数: {len(df_internal_raw)}")

# 预处理
df_internal = dynamic_preprocess_ipmn(df_internal_raw)

# 【新增】应用外部标准化到内部数据集
df_internal = apply_external_standardization(df_internal, std_params, num_vars)
print(f"预处理及标准化后样本数: {len(df_internal)}")

# 特征对齐
for col in best_features:
    if col not in df_internal.columns:
        df_internal[col] = 0
        print(f"  补充缺失特征: {col}")

# 保存编码后的内部数据集
df_internal_encoded = df_internal[[id_col, target_col] + best_features].copy()
df_internal_encoded.to_excel(os.path.join(encoded_data_path, "Internal_Complete_Encoded_Std.xlsx"), index=False)
df_internal_encoded.to_csv(os.path.join(encoded_data_path, "Internal_Complete_Encoded_Std.csv"), index=False,
                           encoding='utf-8-sig')
print(f"编码后内部数据集已保存")

# ==========================================
# 8. 7:3分层划分训练/测试集
# ==========================================
print("\n" + "=" * 60)
print("Step 3: 7:3分层划分训练/测试集")
print("=" * 60)

X = df_internal[best_features].astype(float)
y = df_internal[target_col].astype(int)
ids = df_internal[id_col]

X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
    X, y, ids, test_size=0.3, random_state=seed, stratify=y
)

print(f"训练集样本数: {len(X_train)}")
print(f"测试集样本数: {len(X_test)}")
print(f"训练集类别分布: {np.bincount(y_train)}")
print(f"测试集类别分布: {np.bincount(y_test)}")

# 【修改】不再使用 StandardScaler fit_transform，直接使用已标准化的数据
X_train_s = X_train.values
X_test_s = X_test.values

# ==========================================
# 9. 模型训练
# ==========================================
print("\n" + "=" * 60)
print("Step 4: 模型训练")
print("=" * 60)

model = LogisticRegression(
    penalty='l2',
    solver='lbfgs',
    multi_class='multinomial',
    max_iter=5000,
    random_state=seed
)
model.fit(X_train_s, y_train)

# 保存模型
joblib.dump(model, os.path.join(results_path, "complete_cases_model.pkl"))
# joblib.dump(scaler, ...) # Scaler不再需要保存，因为使用外部参数

# 保存模型系数
coef_df = pd.DataFrame(model.coef_, columns=best_features, index=labels)
coef_df.insert(0, 'Intercept', model.intercept_)
coef_df.to_excel(os.path.join(results_path, "Model_Coefficients.xlsx"))

print("模型训练完成，已保存")

# ==========================================
# 10. 内部验证（测试集）
# ==========================================
print("\n" + "=" * 60)
print("Step 5: 内部验证（测试集）")
print("=" * 60)

y_pred_test = model.predict(X_test_s)
y_prob_test = model.predict_proba(X_test_s)

# 保存预测结果
pred_df_test = pd.DataFrame({
    'ID': ids_test.values,
    'True_Grade_Encoded': y_test.values,
    'True_Grade': [labels[code] for code in y_test],
    'Predicted_Grade_Encoded': y_pred_test,
    'Predicted_Grade': [labels[code] for code in y_pred_test],
    'Correct': (y_test.values == y_pred_test).astype(int)
})
for j, lab in enumerate(labels):
    pred_df_test[f'Prob_{lab}'] = y_prob_test[:, j]
pred_df_test.to_excel(os.path.join(results_path, "internal_test_predictions.xlsx"), index=False)

# 混淆矩阵
cm_test = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Internal Validation (Test Set): Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(results_path, "internal_confusion_matrix.png"), dpi=300, bbox_inches='tight')
plt.close()

# ROC曲线
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
plt.figure(figsize=(10, 8))
for i in range(len(labels)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob_test[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{labels[i]} (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Internal Validation (Test Set): Multi-class ROC Curve')
plt.legend(loc='lower right')
plt.savefig(os.path.join(results_path, "internal_roc_curve.png"), dpi=300, bbox_inches='tight')
plt.close()

# 评价指标
internal_metrics, internal_macro_auc, internal_weighted_auc = calculate_full_metrics(
    y_test, y_pred_test, y_prob_test, labels
)
internal_metrics.to_excel(os.path.join(results_path, "internal_validation_metrics.xlsx"))

# 置信区间
print("  计算内部验证指标置信区间...")
internal_ci = bootstrap_metric_ci(y_test.values, y_pred_test, y_prob_test, labels, BOOTSTRAP_N, CI_LEVEL)
internal_ci.to_excel(os.path.join(results_path, "internal_validation_metrics_with_CI.xlsx"), index=False)

print(f"内部验证完成")
print(f"  Accuracy: {np.mean(y_test == y_pred_test):.4f}")
print(f"  Macro AUC: {internal_macro_auc:.4f}")
print(f"  Weighted AUC: {internal_weighted_auc:.4f}")

# ==========================================
# 11. 内部SHAP分析
# ==========================================
print("\n" + "=" * 60)
print("Step 6: 内部SHAP分析")
print("=" * 60)

# 训练集SHAP
try:
    shap_train, mean_shap_train = compute_shap_analysis(
        model, X_train_s, best_features, labels,
        ids_train.tolist(), "internal_train", shap_output_path
    )
except Exception as e:
    print(f"  训练集SHAP分析失败: {e}")

# 测试集SHAP
try:
    shap_test, mean_shap_test = compute_shap_analysis(
        model, X_test_s, best_features, labels,
        ids_test.tolist(), "internal_test", shap_output_path
    )
except Exception as e:
    print(f"  测试集SHAP分析失败: {e}")

# ==========================================
# 12. 外部验证
# ==========================================
print("\n" + "=" * 60)
print("Step 7: 外部验证")
print("=" * 60)

if os.path.exists(external_data_file):
    df_external_raw = pd.read_excel(external_data_file)
    print(f"原始外部数据集样本数: {len(df_external_raw)}")

    df_external = dynamic_preprocess_ipmn(df_external_raw)

    # 【新增】应用外部标准化到外部数据集
    df_external = apply_external_standardization(df_external, std_params, num_vars)
    print(f"预处理及标准化后样本数: {len(df_external)}")

    # 特征对齐
    for col in best_features:
        if col not in df_external.columns:
            df_external[col] = 0

    # 保存编码后的外部数据集
    df_external_encoded = df_external[[id_col, target_col] + best_features].copy()
    df_external_encoded.to_excel(os.path.join(encoded_data_path, "External_Complete_Encoded_Std.xlsx"), index=False)
    df_external_encoded.to_csv(os.path.join(encoded_data_path, "External_Complete_Encoded_Std.csv"), index=False,
                               encoding='utf-8-sig')

    X_ext = df_external[best_features].astype(float).values
    y_ext = df_external[target_col].astype(int).values
    ids_ext = df_external[id_col].values

    # 直接使用标准化后的数据
    X_ext_s = X_ext

    # 预测
    y_pred_ext = model.predict(X_ext_s)
    y_prob_ext = model.predict_proba(X_ext_s)

    # 保存预测结果
    pred_df_ext = pd.DataFrame({
        'ID': ids_ext,
        'True_Grade_Encoded': y_ext,
        'True_Grade': [labels[code] for code in y_ext],
        'Predicted_Grade_Encoded': y_pred_ext,
        'Predicted_Grade': [labels[code] for code in y_pred_ext],
        'Correct': (y_ext == y_pred_ext).astype(int)
    })
    for j, lab in enumerate(labels):
        pred_df_ext[f'Prob_{lab}'] = y_prob_ext[:, j]
    pred_df_ext.to_excel(os.path.join(results_path, "external_validation_predictions.xlsx"), index=False)

    # 混淆矩阵
    cm_ext = confusion_matrix(y_ext, y_pred_ext)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_ext, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('External Validation: Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(results_path, "external_confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ROC曲线
    y_ext_bin = label_binarize(y_ext, classes=[0, 1, 2])
    plt.figure(figsize=(10, 8))
    for i in range(len(labels)):
        fpr, tpr, _ = roc_curve(y_ext_bin[:, i], y_prob_ext[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{labels[i]} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('External Validation: Multi-class ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(results_path, "external_roc_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 评价指标
    external_metrics, external_macro_auc, external_weighted_auc = calculate_full_metrics(
        y_ext, y_pred_ext, y_prob_ext, labels
    )
    external_metrics.to_excel(os.path.join(results_path, "external_validation_metrics.xlsx"))

    # 置信区间
    print("  计算外部验证指标置信区间...")
    external_ci = bootstrap_metric_ci(y_ext, y_pred_ext, y_prob_ext, labels, BOOTSTRAP_N, CI_LEVEL)
    external_ci.to_excel(os.path.join(results_path, "external_validation_metrics_with_CI.xlsx"), index=False)

    print(f"外部验证完成")
    print(f"  Accuracy: {np.mean(y_ext == y_pred_ext):.4f}")
    print(f"  Macro AUC: {external_macro_auc:.4f}")
    print(f"  Weighted AUC: {external_weighted_auc:.4f}")

    # 外部SHAP分析
    print("\n" + "=" * 60)
    print("Step 8: 外部SHAP分析")
    print("=" * 60)

    try:
        shap_ext, mean_shap_ext = compute_shap_analysis(
            model, X_ext_s, best_features, labels,
            ids_ext.tolist(), "external", shap_output_path
        )
    except Exception as e:
        print(f"  外部SHAP分析失败: {e}")

else:
    print(f"警告：未找到外部验证数据文件: {external_data_file}")

# ==========================================
# 13. 生成综合报告
# ==========================================
print("\n" + "=" * 60)
print("Step 9: 生成综合报告")
print("=" * 60)

report_path = os.path.join(results_path, "Comprehensive_Results_Summary.xlsx")
with pd.ExcelWriter(report_path) as writer:
    # 特征列表
    pd.DataFrame({'Selected_Features': best_features}).to_excel(writer, sheet_name='特征列表', index=False)

    # 数据集信息
    dataset_info = pd.DataFrame({
        '数据集': ['内部训练集', '内部测试集', '外部验证集'],
        '样本数': [len(X_train), len(X_test), len(df_external) if 'df_external' in dir() else 0]
    })
    dataset_info.to_excel(writer, sheet_name='数据集信息', index=False)

    # 内部验证指标
    internal_ci.to_excel(writer, sheet_name='内部验证指标_CI', index=False)

    # 外部验证指标
    if 'external_ci' in dir():
        external_ci.to_excel(writer, sheet_name='外部验证指标_CI', index=False)

    # 模型系数
    coef_df.to_excel(writer, sheet_name='模型系数')

print(f"综合报告保存至: {report_path}")

# 打印输出文件清单
print("\n" + "=" * 60)
print("输出文件清单")
print("=" * 60)
print(f"""
主要输出目录: {results_path}

1. 编码后数据集 ({encoded_data_path}):
   - Internal_Complete_Encoded_Std.xlsx/csv
   - External_Complete_Encoded_Std.xlsx/csv

2. 模型文件:
   - complete_cases_model.pkl
   - Model_Coefficients.xlsx

3. 预测结果:
   - internal_test_predictions.xlsx
   - external_validation_predictions.xlsx

4. 评价指标:
   - internal_validation_metrics.xlsx
   - internal_validation_metrics_with_CI.xlsx
   - external_validation_metrics.xlsx
   - external_validation_metrics_with_CI.xlsx

5. 可视化:
   - internal_confusion_matrix.png
   - internal_roc_curve.png
   - external_confusion_matrix.png
   - external_roc_curve.png

6. SHAP分析 ({shap_output_path}):
   - internal_train_*.xlsx
   - internal_test_*.xlsx
   - external_*.xlsx
   - *_shap_summary_beeswarm.png

7. 综合报告:
   - Comprehensive_Results_Summary.xlsx
""")

print("\n=== IPMN完整数据集建模验证（外部参数标准化版）全部完成 ===")