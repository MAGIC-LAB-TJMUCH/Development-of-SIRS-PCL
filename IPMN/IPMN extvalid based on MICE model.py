# -*- coding: utf-8 -*-
"""
IPMN 风险分级模型 - 完整数据集外部验证脚本

修复版本：从Excel Sheet4读取特征列表

功能：
1. 从Excel Sheet4加载特征列表
2. 加载已训练的模型和标准化参数（从Excel文件）
3. 使用完整数据集进行外部验证
4. 生成SHAP值分析结果
5. 生成预测结果与AUC等评价指标的置信区间
6. 生成混淆矩阵和ROC曲线
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

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ==========================================
# 1. 路径配置
# ==========================================
# 模型和参数路径（来自训练脚本的输出）
model_base_path = r"G:\胰腺囊性病变分类与风险预测_1.2\IPMN风险结果\IPMN_new\IPMN_Grade_Std_ExternalParams"

# 完整验证数据集路径
complete_valid_path = r"E:\Pancreatic cancer\多中心胰腺囊性病变诊断与风险预测\完整数据集-验证\Data_IPMN_Pre_TRUE.xlsx"

# 标准化参数文件路径（推荐使用Excel）
std_params_path = r"E:\Pancreatic cancer\多中心胰腺囊性病变诊断与风险预测\完整数据集-五分类建模\MICE_model\Standardization_Params_IPMN\Standardization_Parameters.xlsx"

# ★ 特征选择结果文件路径 - 使用Sheet4
variable_selection_path = r"G:\胰腺囊性病变分类与风险预测_1.2\IPMN风险结果\IPMN_new\IPMN_Grade_Std_ExternalParams\IPMN_Variable_Selection_Results.xlsx"

# 输出路径
results_path = os.path.join(model_base_path, "Complete_PreExternal_Validation_all_v2")
os.makedirs(results_path, exist_ok=True)

shap_output_path = os.path.join(results_path, "SHAP_Analysis")
os.makedirs(shap_output_path, exist_ok=True)

# 基本配置
target_col = "Grade"
id_col = "key"
seed = 5156

# Bootstrap置信区间参数
BOOTSTRAP_N = 1000
CI_LEVEL = 0.95

# 类别标签
labels = ['Low Risk', 'Medium Risk', 'High Risk']

# 原始分类变量（用于预处理）
raw_categorical_vars = [
    "Cyst wall thickness", "Uniform Cyst wall", "Cyst wall enhancement",
    "Mural nodule status", "Mural nodule enhancement", "Solid component enhancement",
    "Intracystic septations", "Uniform Septations", "Intracystic septa enhancement",
    "Capsule", "Main PD communication", "Pancreatic parenchymal atrophy",
    "Mural nodule in MPD", "Common bile duct dilation",
    "Vascular abutment", "Enlarged lymph nodes", "Distant metastasis",
    "Tumor lesion", "Lesion_Head_neck", "Lesion_body_tail", "Diabetes", "Jaundice"
]

num_vars = ["Long diameter of lesion (mm)", "Short diameter of lesion (mm)", "Long diameter of solid component (mm)",
            "Short diameter of solid component (mm)", "Long diameter of largest mural nodule (mm)",
            "Short diameter of largest mural nodule (mm)", "Diameter of MPD (mm)",
            "CA_199", "CEA"]

whitelist = [target_col, id_col] + raw_categorical_vars + num_vars


# ==========================================
# 2. 特征加载函数
# ==========================================
def load_features_from_sheet4(filepath, sheet_name=3):
    """从Excel Sheet4加载特征列表

    参数：
        filepath: Excel文件路径
        sheet_name: Sheet名称或索引（默认3表示Sheet4，因为索引从0开始）

    返回：
        best_features: 特征列表
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"未找到特征选择结果文件: {filepath}")

    print(f"  正在从Excel Sheet4加载特征列表...")
    print(f"  文件路径: {filepath}")

    try:
        # 首先列出所有Sheet
        excel_file = pd.ExcelFile(filepath)
        print(f"  Excel文件中的所有Sheet: {excel_file.sheet_names}")

        # 读取Sheet4（索引为3）
        # 支持多种方式指定Sheet：'Sheet4'（名称）或3（索引）
        try:
            # 尝试用索引读取
            features_df = pd.read_excel(filepath, sheet_name=3)
            print(f"  ✓ 使用索引3读取Sheet4成功")
        except:
            # 尝试用名称读取
            try:
                features_df = pd.read_excel(filepath, sheet_name='Sheet4')
                print(f"  ✓ 使用名称'Sheet4'读取成功")
            except:
                # 列出sheet名称并提示
                print(f"  ✗ 无法读取Sheet4或索引3")
                print(f"  可用的Sheet: {excel_file.sheet_names}")
                raise

        print(f"  Sheet4的列名: {list(features_df.columns)}")
        print(f"  Sheet4的行数: {len(features_df)}")
        print(f"  Sheet4的前5行数据:")
        print(features_df.head())

        # 查找包含特征信息的列
        feature_col = None
        for col in ['Feature', 'feature_name', 'Feature_Name', 'Variable', 'variable', '特征', '变量']:
            if col in features_df.columns:
                feature_col = col
                print(f"  ✓ 找到特征列: {feature_col}")
                break

        if feature_col is None:
            # 如果找不到，使用第一列
            print(f"  ⚠️  未找到命名的特征列，使用第一列")
            feature_col = features_df.columns[0]

        # 读取特征列表
        # 检查是否有筛选列（最后保留的特征列表）
        if 'Final_Keep' in features_df.columns:
            best_features = features_df[features_df['Final_Keep'] == 1][feature_col].tolist()
            print(f"  ✓ 使用Final_Keep筛选，得到{len(best_features)}个特征")
        elif 'Keep' in features_df.columns:
            best_features = features_df[features_df['Keep'] == 1][feature_col].tolist()
            print(f"  ✓ 使用Keep筛选，得到{len(best_features)}个特征")
        elif 'Selected' in features_df.columns:
            best_features = features_df[features_df['Selected'] == 1][feature_col].tolist()
            print(f"  ✓ 使用Selected筛选，得到{len(best_features)}个特征")
        else:
            # 使用所有行
            best_features = features_df[feature_col].dropna().tolist()
            print(f"  ✓ 使用所有特征，共{len(best_features)}个")

        # 过滤掉空值
        best_features = [f for f in best_features if pd.notna(f) and str(f).strip() != '']

        print(f"  ✓ 成功加载{len(best_features)}个特征")
        print(f"  特征列表:")
        for i, feat in enumerate(best_features, 1):
            print(f"    {i}. {feat}")

        return best_features

    except Exception as e:
        print(f"  ✗ 读取特征列表失败: {e}")
        print(f"  错误详情: {type(e).__name__}")
        print(f"\n  将尝试使用备选方案...")
        raise


# ==========================================
# 3. 标准化工具函数
# ==========================================
def load_standardization_params(filepath):
    """加载标准化参数文件"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"未找到标准化参数文件: {filepath}")

    print(f"  正在加载标准化参数: {filepath}")
    try:
        params_df = pd.read_excel(filepath)

        # 检查必要的列
        required_cols = ['Feature', 'Mean', 'Std']
        if not all(col in params_df.columns for col in required_cols):
            raise ValueError(f"Excel文件缺少必要的列。需要：{required_cols}，实际：{list(params_df.columns)}")

        # 转换为字典格式
        params_dict = params_df.set_index('Feature')[['Mean', 'Std']].to_dict('index')
        print(f"  ✓ 成功加载{len(params_dict)}个特征的标准化参数")

        return params_dict
    except Exception as e:
        print(f"  ✗ 读取标准化参数文件失败: {e}")
        raise


def apply_external_standardization(df, params_dict, num_vars):
    """应用外部标准化参数 (Z-score)"""
    df_scaled = df.copy()
    processed_cols = []
    skipped_cols = []

    for col in num_vars:
        if col in df_scaled.columns and col in params_dict:
            mean_val = params_dict[col]['Mean']
            std_val = params_dict[col]['Std']

            if std_val == 0:
                print(f"  警告: 变量 {col} 标准差为0，跳过标准化。")
                skipped_cols.append(col)
            else:
                df_scaled[col] = df_scaled[col].astype(float)
                df_scaled[col] = (df_scaled[col] - mean_val) / std_val
                processed_cols.append(col)
        elif col in df_scaled.columns and col not in params_dict:
            print(f"  注意: 变量 {col} 未在参数文件中找到，保持原始值。")
            skipped_cols.append(col)

    print(f"  ✓ 标准化完成：{len(processed_cols)}个变量已处理，{len(skipped_cols)}个变量已跳过")
    return df_scaled


# ==========================================
# 4. 数据预处理函数
# ==========================================
def dynamic_preprocess_ipmn(df):
    """IPMN数据预处理函数"""
    print("  开始数据预处理...")

    # 过滤IPMN患者
    df_sub = df.copy()
    #print(f"    ✓ 过滤{filter_value}患者: {len(df_sub)}条记录")

    # 选择白名单中的列
    available_cols = [c for c in whitelist if c in df_sub.columns]
    df_clean = df_sub[available_cols].copy()
    print(f"    ✓ 选择特征: {len(available_cols)}/{len(whitelist)}列可用")

    # 编码目标变量（Grade）
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
        print(f"    ⚠️  丢弃{before_drop - after_drop}条无法识别的Grade标签样本")
    df_clean[target_col] = df_clean[target_col].astype(int)

    # 处理enhancement变量
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

    # 处理分类变量
    print("    处理分类变量...")
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

    print(f"  ✓ 预处理完成")
    return df_clean


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
                if sum(y_true_bin_boot[:, i]) > 0:
                    fpr, tpr, _ = roc_curve(y_true_bin_boot[:, i], y_prob_boot[:, i])
                    roc_auc = auc(fpr, tpr)
                    bootstrap_results['auc'][label].append(roc_auc)
            except:
                pass

        accuracy = np.mean(y_pred_boot == y_true_boot)
        bootstrap_results['accuracy'].append(accuracy)

        try:
            macro_auc = roc_auc_score(y_true_bin_boot, y_prob_boot, multi_class='ovr', average='macro')
            bootstrap_results['macro_auc'].append(macro_auc)
        except:
            pass

        try:
            weighted_auc = roc_auc_score(y_true_bin_boot, y_prob_boot, multi_class='ovr', average='weighted')
            bootstrap_results['weighted_auc'].append(weighted_auc)
        except:
            pass

    alpha = 1 - ci_level
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100

    results_list = []
    for label in class_labels:
        for metric_name in ['precision', 'sensitivity', 'specificity', 'f1', 'auc']:
            metric_values = bootstrap_results[metric_name][label]
            if metric_values:
                mean_val = np.mean(metric_values)
                lower = np.percentile(metric_values, lower_percentile)
                upper = np.percentile(metric_values, upper_percentile)
                results_list.append({
                    'Class': label,
                    'Metric': metric_name.upper(),
                    'Mean': mean_val,
                    'Lower_CI': lower,
                    'Upper_CI': upper
                })

    for metric_name in ['accuracy', 'macro_auc', 'weighted_auc']:
        metric_values = bootstrap_results[metric_name]
        if metric_values:
            mean_val = np.mean(metric_values)
            lower = np.percentile(metric_values, lower_percentile)
            upper = np.percentile(metric_values, upper_percentile)
            results_list.append({
                'Class': 'Overall',
                'Metric': metric_name.upper(),
                'Mean': mean_val,
                'Lower_CI': lower,
                'Upper_CI': upper
            })

    return pd.DataFrame(results_list)


def calculate_full_metrics(y_true, y_pred, y_prob, class_labels):
    """计算完整的评价指标"""
    n_classes = len(class_labels)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    results = []
    for i, label in enumerate(class_labels):
        cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

        results.append({
            'Class': label,
            'Precision': precision,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'F1_Score': f1
        })

    macro_auc = roc_auc_score(y_true_bin, y_prob, multi_class='ovr', average='macro')
    weighted_auc = roc_auc_score(y_true_bin, y_prob, multi_class='ovr', average='weighted')

    return pd.DataFrame(results), macro_auc, weighted_auc


# ==========================================
# SHAP分析函数
# ==========================================
def compute_shap_analysis(model, X, feature_names, class_labels, sample_ids, data_type="train", output_path="./"):
    """计算SHAP分析"""
    print(f"  计算{data_type}数据集的SHAP值...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap_explanation = explainer(X)

    X_df = pd.DataFrame(X, columns=feature_names)

    mean_abs_shap = []
    mean_raw_shap = []

    for i, class_name in enumerate(class_labels):
        if isinstance(shap_values, list):
            class_shap_vals = np.abs(shap_values[i])
            class_raw_shap = shap_values[i]
        else:
            class_shap_vals = np.abs(shap_values[:, :, i])
            class_raw_shap = shap_values[:, :, i]

        mean_abs = np.mean(class_shap_vals, axis=0)
        mean_raw = np.mean(class_raw_shap, axis=0)

        mean_abs_shap.append(mean_abs)
        mean_raw_shap.append(mean_raw)

    mean_abs_df = pd.DataFrame(mean_abs_shap, columns=feature_names, index=class_labels)
    mean_abs_df.to_excel(os.path.join(output_path, f'{data_type}_mean_absolute_shap.xlsx'))

    mean_raw_df = pd.DataFrame(mean_raw_shap, columns=feature_names, index=class_labels)
    mean_raw_df.to_excel(os.path.join(output_path, f'{data_type}_mean_raw_shap.xlsx'))

    print(f"  ✓ {data_type} SHAP分析完成")
    return shap_explanation, mean_abs_df


# ==========================================
# 主程序
# ==========================================
print("=" * 70)
print("IPMN 风险分级模型 - 完整数据集外部验证（从Sheet4读取特征版本）")
print("=" * 70)

# ==========================================
# Step 1: 加载模型和参数
# ==========================================
print("\n[Step 1] 加载训练好的模型和参数...")

# 加载模型
model_path = os.path.join(model_base_path, "final_ipmn_grade_model.pkl")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"找不到模型文件: {model_path}")
final_model = joblib.load(model_path)
print(f"  ✓ 模型加载成功")

# ★ 从Sheet4加载特征列表
print("\n  从Excel Sheet4加载特征列表...")
best_features = load_features_from_sheet4(variable_selection_path)
print(f"  ✓ 成功从Sheet4加载{len(best_features)}个特征")

# 验证特征数量
try:
    expected_n_features = final_model.n_features_in_
    print(f"  ✓ 模型期望的输入特征数: {expected_n_features}")

    if len(best_features) != expected_n_features:
        print(f"\n  ⚠️  警告：特征数不匹配！")
        print(f"     模型期望: {expected_n_features}, 加载到: {len(best_features)}")
        print(f"     可能需要检查Sheet4中是否有筛选条件")
except:
    print(f"  ⚠️  无法确定模型期望的特征数，继续执行...")

# ==========================================
# Step 2: 加载和预处理验证数据
# ==========================================
print("\n[Step 2] 加载和预处理完整验证数据集...")

if not os.path.exists(complete_valid_path):
    raise FileNotFoundError(f"找不到验证数据文件: {complete_valid_path}")

raw_valid_df = pd.read_excel(complete_valid_path)
print(f"  ✓ 原始数据加载成功，共{len(raw_valid_df)}条记录")

# 预处理
df_valid = dynamic_preprocess_ipmn(raw_valid_df)
print(f"  ✓ 预处理后IPMN样本数: {len(df_valid)}")

# 保存编码后的验证数据集
df_valid.to_excel(os.path.join(results_path, "Complete_Validation_Encoded.xlsx"), index=False)

# 对齐特征
print(f"\n  对齐特征...")
df_valid_aligned = df_valid.reindex(columns=best_features + [target_col, id_col], fill_value=0)
print(f"  ✓ 特征对齐完成")

# 提取特征和标签
X_valid = df_valid_aligned[best_features].values
y_valid = df_valid_aligned[target_col].values
ids_valid = df_valid_aligned[id_col].values

print(f"  ✓ 特征维度: {X_valid.shape}")
print(f"  ✓ 类别分布:")
for i, label in enumerate(labels):
    count = np.sum(y_valid == i)
    print(f"    - {label}: {count} ({count / len(y_valid) * 100:.1f}%)")

# ==========================================
# Step 3: 应用标准化参数
# ==========================================
print("\n[Step 3] 应用标准化参数...")

# 加载标准化参数
std_params_dict = load_standardization_params(std_params_path)

# 将X_valid转换为DataFrame以便应用标准化
df_valid_to_scale = pd.DataFrame(X_valid, columns=best_features)

# 应用标准化
df_valid_scaled = apply_external_standardization(df_valid_to_scale, std_params_dict, best_features)

# 转换回numpy数组
X_valid_scaled = df_valid_scaled[best_features].values

print(f"  ✓ 标准化完成，最终输入维度: {X_valid_scaled.shape}")

# ==========================================
# Step 4: 模型预测
# ==========================================
print("\n[Step 4] 执行模型预测...")

y_prob = final_model.predict_proba(X_valid_scaled)
y_pred = np.argmax(y_prob, axis=1)

# 保存预测结果
prediction_df = pd.DataFrame({
    'ID': ids_valid,
    'True_Grade': y_valid,
    'True_Grade_Label': [labels[int(y)] for y in y_valid],
    'Predicted_Grade': y_pred,
    'Predicted_Grade_Label': [labels[int(p)] for p in y_pred]
})
for j, lab in enumerate(labels):
    prediction_df[f'Prob_{lab}'] = y_prob[:, j]

prediction_df.to_excel(os.path.join(results_path, "Complete_Validation_Predictions.xlsx"), index=False)
print(f"  ✓ 预测结果已保存")

# ==========================================
# Step 5: 计算评价指标
# ==========================================
print("\n[Step 5] 计算评价指标...")

# 基础指标
basic_metrics, macro_auc, weighted_auc = calculate_full_metrics(y_valid, y_pred, y_prob, labels)
basic_metrics.to_excel(os.path.join(results_path, "Complete_Validation_Metrics.xlsx"))
print(f"  ✓ Macro AUC: {macro_auc:.4f}")
print(f"  ✓ Weighted AUC: {weighted_auc:.4f}")

# 计算置信区间
print("  计算Bootstrap置信区间...")
ci_df = bootstrap_metric_ci(y_valid, y_pred, y_prob, labels, n_bootstrap=BOOTSTRAP_N, ci_level=CI_LEVEL)
ci_df.to_excel(os.path.join(results_path, "Complete_Validation_Metrics_with_CI.xlsx"), index=False)
print("  ✓ 置信区间计算完成")

# ==========================================
# Step 6: 生成混淆矩阵
# ==========================================
print("\n[Step 6] 生成混淆矩阵...")

cm = confusion_matrix(y_valid, y_pred)

# 保存混淆矩阵数据
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
cm_df.to_excel(os.path.join(results_path, "Complete_Validation_Confusion_Matrix.xlsx"))

# 绘制混淆矩阵图
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('External Validation: Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(results_path, "Complete_Validation_Confusion_Matrix.png"), dpi=300, bbox_inches='tight')
plt.close()

# 归一化混淆矩阵
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(10, 8))
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('External Validation: Normalized Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(results_path, "Complete_Validation_Confusion_Matrix_Normalized.png"), dpi=300,
            bbox_inches='tight')
plt.close()

print("  ✓ 混淆矩阵已保存")

# ==========================================
# Step 7: 生成ROC曲线
# ==========================================
print("\n[Step 7] 生成ROC曲线...")

y_valid_bin = label_binarize(y_valid, classes=[0, 1, 2])

plt.figure(figsize=(10, 8))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for i, (label, color) in enumerate(zip(labels, colors)):
    fpr, tpr, _ = roc_curve(y_valid_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2, label=f'{label} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('External Validation: Multi-class ROC Curve', fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_path, "Complete_Validation_ROC_Curve.png"), dpi=300, bbox_inches='tight')
plt.close()

print("  ✓ ROC曲线已保存")

# ==========================================
# Step 8: SHAP分析
# ==========================================
print("\n[Step 8] 执行SHAP分析...")

try:
    shap_explanation, mean_shap_df = compute_shap_analysis(
        final_model, X_valid_scaled, best_features, labels,
        ids_valid.tolist() if hasattr(ids_valid, 'tolist') else list(ids_valid),
        data_type="external_validation", output_path=shap_output_path
    )
except Exception as e:
    print(f"  ✗ SHAP分析失败: {e}")

# ==========================================
# Step 9: 生成综合报告
# ==========================================
print("\n[Step 9] 生成综合报告...")

report_path = os.path.join(results_path, "Complete_Validation_Summary.xlsx")
with pd.ExcelWriter(report_path) as writer:
    # 基本信息
    info_df = pd.DataFrame({
        'Item': ['验证数据集', '样本数', '特征数', 'Macro AUC', 'Weighted AUC', '特征来源'],
        'Value': ['完整数据集', len(y_valid), len(best_features), f"{macro_auc:.4f}", f"{weighted_auc:.4f}", 'Sheet4']
    })
    info_df.to_excel(writer, sheet_name='基本信息', index=False)

    # 特征列表
    pd.DataFrame({'Features': best_features}).to_excel(writer, sheet_name='使用特征', index=False)

    # 评价指标（带置信区间）
    ci_df.to_excel(writer, sheet_name='评价指标_CI', index=False)

    # 混淆矩阵
    cm_df.to_excel(writer, sheet_name='混淆矩阵')

    # 类别分布
    dist_df = pd.DataFrame({
        'Class': labels,
        'True_Count': [np.sum(y_valid == i) for i in range(len(labels))],
        'Pred_Count': [np.sum(y_pred == i) for i in range(len(labels))]
    })
    dist_df.to_excel(writer, sheet_name='类别分布', index=False)

print(f"  ✓ 综合报告已保存: {report_path}")

# ==========================================
# 打印输出文件清单
# ==========================================
print("\n" + "=" * 70)
print("完成！输出文件清单")
print("=" * 70)
print(f"""
输出目录: {results_path}

✓ 预测结果:
  - Complete_Validation_Predictions.xlsx
  - Complete_Validation_Encoded.xlsx

✓ 评价指标:
  - Complete_Validation_Metrics.xlsx
  - Complete_Validation_Metrics_with_CI.xlsx

✓ 混淆矩阵:
  - Complete_Validation_Confusion_Matrix.xlsx
  - Complete_Validation_Confusion_Matrix.png
  - Complete_Validation_Confusion_Matrix_Normalized.png

✓ ROC曲线:
  - Complete_Validation_ROC_Curve.png

✓ SHAP分析: ({shap_output_path})
  - external_validation_mean_absolute_shap.xlsx
  - external_validation_mean_raw_shap.xlsx

✓ 综合报告:
  - Complete_Validation_Summary.xlsx
""")

print("=" * 70)
print("✓✓✓ 外部验证全部完成 ✓✓✓")
print("=" * 70)