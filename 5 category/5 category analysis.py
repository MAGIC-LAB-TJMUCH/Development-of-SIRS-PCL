# -*- coding: utf-8 -*-
"""
胰腺囊性病变五分类预测模型构建与验证
包含特征筛选、Rubin规则池化、SHAP解释、内外验证全流程

【定制修改版】
1. 数值型变量标准化：严格基于外部提供的 Parameters 文件进行 (Z-score转换)
2. AUC等评价指标的Bootstrap置信区间计算
3. 保存所有编码后的完整数据集（内部建模+外部验证）
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
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                             classification_report, roc_auc_score)
from sklearn.utils import resample
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib

# ===================== 全局设置 =====================
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示

# ===================== 路径与参数配置 =====================
base_path = r"\多中心胰腺囊性病变诊断与风险预测\完整数据集-五分类建模\MICE_model"
valid_base_path = r"\多中心胰腺囊性病变诊断与风险预测\完整数据集-验证\MICE Valid"

# 标准化参数文件路径
std_params_path = os.path.join(base_path, "Standardization_Params", "Standardization_Parameters.xlsx")

# 输出路径 (修改以区分版本)
results_path = os.path.join(base_path, "Final_Analysis_Results_Std_ExternalParams")
shap_output_path = os.path.join(results_path, "SHAP_Analysis")
encoded_data_path = os.path.join(results_path, "Encoded_Datasets")

# 创建结果目录
os.makedirs(results_path, exist_ok=True)
os.makedirs(shap_output_path, exist_ok=True)
os.makedirs(encoded_data_path, exist_ok=True)

output_file = os.path.join(results_path, "Variable_Selection_Results_Enhance.xlsx")

# 核心参数
target_col = "Dignosis"
id_col = "key"
m = 10
corr_threshold = 0.9
seed = 9477
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
    # 假设Excel包含 'Feature', 'Mean', 'Std' 列
    # 转换为字典格式: {'FeatureName': {'mean': x, 'std': y}}
    params_dict = params_df.set_index('Feature')[['Mean', 'Std']].to_dict('index')
    return params_dict


def apply_external_standardization(df, params_dict, num_vars):
    """应用外部标准化参数 (Z-score)"""
    df_scaled = df.copy()
    scaled_cols = []

    for col in num_vars:
        if col in df_scaled.columns and col in params_dict:
            mean_val = params_dict[col]['Mean']
            std_val = params_dict[col]['Std']

            # 避免除以0
            if std_val == 0:
                print(f"警告: 变量 {col} 标准差为0，跳过标准化。")
            else:
                # 执行标准化 (X - Mean) / Std
                df_scaled[col] = (df_scaled[col] - mean_val) / std_val
                scaled_cols.append(col)
        elif col in df_scaled.columns and col not in params_dict:
            print(f"注意: 变量 {col} 未在参数文件中找到，保持原始值。")

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
    # ===================== 0. 加载外部标准化参数 =====================
    print("=" * 50)
    print("Step 0: 加载数值型变量标准化参数")
    print("=" * 50)

    try:
        std_params = load_standardization_params(std_params_path)
        print("参数加载成功。")
    except Exception as e:
        print(f"错误: 无法加载标准化参数文件。请确认路径: {std_params_path}")
        print(f"详细错误: {e}")
        return

    # ===================== 1. 数据加载与特征对齐 =====================
    print("\n" + "=" * 50)
    print("Step 1: 读取并处理15组插补数据集 (含标准化)")
    print("=" * 50)

    processed_dfs = []
    full_column_set = set()

    for i in range(1, m + 1):
        file_path = os.path.join(base_path, f"df_model_imputed_{i}.xlsx")
        raw_df = pd.read_excel(file_path)

        # 1. 基础预处理
        proc, mapping = dynamic_preprocess(raw_df)

        # 2. 【关键步骤】应用外部标准化参数
        proc = apply_external_standardization(proc, std_params, num_vars)

        final_mapping.update(mapping)
        processed_dfs.append(proc)
        full_column_set.update(proc.columns)

    global labels
    labels = [final_mapping[i] for i in sorted(final_mapping.keys())]
    n_classes = len(labels)

    mapping_df = pd.DataFrame(list(final_mapping.items()), columns=['Code', 'Diagnosis_Label'])
    print("\n--- 结局类别映射对照表 ---")
    print(mapping_df.to_string(index=False))

    all_features_index = sorted(list(full_column_set - {target_col, id_col}))

    final_aligned_data = []
    for df in processed_dfs:
        for col in full_column_set:
            if col not in df.columns:
                df[col] = 0
        final_aligned_data.append(df)

    # ===================== 保存所有编码后的内部建模数据集 =====================
    print("\n" + "=" * 50)
    print("Step 1.5: 保存所有编码后的内部建模数据集 (已标准化)")
    print("=" * 50)

    all_encoded_internal = []
    for i, df_encoded in enumerate(final_aligned_data):
        encoded_file = os.path.join(encoded_data_path, f"Internal_Encoded_Dataset_{i + 1}.xlsx")
        df_encoded.to_excel(encoded_file, index=False)

        df_temp = df_encoded.copy()
        df_temp['Imputation_Set'] = i + 1
        all_encoded_internal.append(df_temp)

    combined_internal_encoded = pd.concat(all_encoded_internal, ignore_index=True)
    combined_internal_encoded.to_excel(
        os.path.join(encoded_data_path, "All_Internal_Encoded_Combined.xlsx"),
        index=False
    )
    print(f"  合并编码后内部数据集保存完成，共 {len(combined_internal_encoded)} 条记录")

    # ===================== 2. 稳定性特征筛选 =====================
    print("\n" + "=" * 50)
    print("Step 2: 执行稳定性特征筛选 (基于已标准化数据)")
    print("=" * 50)

    detailed_selection = pd.DataFrame(index=all_features_index)
    train_cache = []

    for i, df_ready in enumerate(final_aligned_data):
        current_seed = seed + i
        X = df_ready.drop(columns=[target_col, id_col], errors='ignore').astype(float)
        y = df_ready[target_col].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=current_seed, stratify=y
        )

        # 注意：此处 X_train 已经是标准化过的数据，不需要再 fit StandardScaler
        # 直接进行相关性剔除和特征筛选
        X_train_processed = X_train.copy()

        # 剔除高相关特征
        corr_matrix = X_train_processed.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop_corr = [col for col in upper_tri.columns if any(upper_tri[col] > corr_threshold)]

        X_train_s = X_train_processed.drop(columns=to_drop_corr, errors='ignore')

        # 1. 随机森林
        rf = RandomForestClassifier(n_estimators=100, random_state=current_seed).fit(X_train_s, y_train)
        t_sel = pd.Series(SelectFromModel(rf, prefit=True, threshold="median").get_support(),
                          index=X_train_s.columns)

        # 2. RFE
        rfe_base = LogisticRegression(max_iter=2000, solver='saga', tol=0.01, random_state=current_seed)
        rfe_selector = RFE(estimator=rfe_base, n_features_to_select=15).fit(X_train_s, y_train)
        r_sel = pd.Series(rfe_selector.get_support(), index=X_train_s.columns)

        # 3. L1正则化
        l1_lr = LogisticRegression(penalty='l1', solver='saga', C=0.5, max_iter=2000,
                                   random_state=current_seed, tol=0.01)
        l1_lr.fit(X_train_s, y_train)
        l_sel = pd.Series(SelectFromModel(l1_lr, prefit=True).get_support(), index=X_train_s.columns)

        selected_count = t_sel.astype(int) + r_sel.astype(int) + l_sel.astype(int)
        round_res = (selected_count >= 2).astype(int)
        round_res = round_res.reindex(detailed_selection.index, fill_value=0).astype(int)
        detailed_selection[f'Imputation_{i + 1}'] = round_res

        train_cache.append((X_train_s, y_train, X_test, y_test))

    # ===================== 3. 筛选结果汇总 =====================
    print("\n" + "=" * 50)
    print("Step 3: 特征筛选结果汇总")
    print("=" * 50)

    detailed_selection['Total_Frequency'] = detailed_selection.sum(axis=1)
    detailed_selection['Stability_Score'] = (detailed_selection['Total_Frequency'] / m).round(4)
    detailed_selection['Final_Keep'] = (detailed_selection['Total_Frequency'] >= 8).astype(int)

    detailed_selection = detailed_selection.sort_values(by='Total_Frequency', ascending=False)
    best_features = detailed_selection[detailed_selection['Final_Keep'] == 1].index.tolist()

    with pd.ExcelWriter(output_file) as writer:
        detailed_selection.to_excel(writer, sheet_name='Variable_Stability')
        mapping_df.to_excel(writer, sheet_name='Diagnosis_Mapping', index=False)

    joblib.dump(detailed_selection, os.path.join(results_path, "selector_results.pkl"))
    print(f"最终筛选出的特征数: {len(best_features)}")

    # ===================== 5. 模型构建与Rubin规则池化 =====================
    print("\n" + "=" * 50)
    print("Step 5: 模型构建与Rubin规则池化")
    print("=" * 50)

    all_intercepts = []
    all_coefs = []
    internal_test_sets = []
    internal_pred_results = []

    lr_model = LogisticRegression(
        penalty='l2', solver='lbfgs', multi_class='multinomial',
        max_iter=5000, random_state=seed
    )

    for i in range(m):
        df_ready = final_aligned_data[i]
        X = df_ready[best_features].astype(float)
        y = df_ready[target_col].astype(int)
        ids = df_ready[id_col]

        X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
            X, y, ids, test_size=0.3, random_state=120 + i, stratify=y
        )

        # X_train 已经是标准化的，直接训练
        lr_model.fit(X_train, y_train)

        all_intercepts.append(lr_model.intercept_)
        all_coefs.append(lr_model.coef_)

        y_pred_train = lr_model.predict(X_train)
        y_prob_train = lr_model.predict_proba(X_train)
        y_pred_test = lr_model.predict(X_test)
        y_prob_test = lr_model.predict_proba(X_test)

        internal_test_sets.append((X_test, y_test, ids_test))

        prob_cols = [f"Prob_{labels[j]}" for j in range(len(labels))]
        train_result = pd.DataFrame({
            'ID': ids_train, 'Set_Type': 'Train',
            'True_Label_Encoded': y_train,
            'True_Label': [labels[code] for code in y_train],
            'Predicted_Label_Encoded': y_pred_train,
            'Predicted_Label': [labels[code] for code in y_pred_train]
        })
        train_result[prob_cols] = y_prob_train

        test_result = pd.DataFrame({
            'ID': ids_test, 'Set_Type': 'Test',
            'True_Label_Encoded': y_test,
            'True_Label': [labels[code] for code in y_test],
            'Predicted_Label_Encoded': y_pred_test,
            'Predicted_Label': [labels[code] for code in y_pred_test]
        })
        test_result[prob_cols] = y_prob_test

        result_df = pd.concat([train_result, test_result], ignore_index=True)
        result_path = os.path.join(results_path, f"internal_prediction_imputation_{i + 1}.xlsx")
        result_df.to_excel(result_path, index=False)
        print(f"  完成第{i + 1}/{m}组模型训练")

    pooled_coef = np.mean(all_coefs, axis=0)
    pooled_intercept = np.mean(all_intercepts, axis=0)

    final_model = LogisticRegression(multi_class='multinomial', solver='saga')
    final_model.coef_ = pooled_coef
    final_model.intercept_ = pooled_intercept
    final_model.classes_ = np.unique(labels)  # Just placeholder
    final_model.classes_ = np.arange(len(labels))  # Correct indices

    joblib.dump(final_model, os.path.join(results_path, "final_model.pkl"))
    print("Rubin规则参数池化完成，最终模型已保存")

    # ===================== 6. 保存模型系数 =====================
    print("\n" + "=" * 50)
    print("Step 6: 保存模型系数")
    print("=" * 50)

    coef_df = pd.DataFrame(
        pooled_coef,
        columns=best_features,
        index=[f"Class_{labels[i]}" for i in range(len(labels))]
    )
    coef_df['Intercept'] = pooled_intercept
    coef_df.to_excel(os.path.join(results_path, "Pooled_Coefficients.xlsx"))

    # ===================== 7. 内部验证与SHAP分析 =====================
    print("\n" + "=" * 50)
    print("Step 7: 内部验证与SHAP分析")
    print("=" * 50)

    X_test_s_last, y_test_last, ids_test_last = internal_test_sets[-1]
    X_test_df = pd.DataFrame(X_test_s_last, columns=best_features)

    y_pred_internal = final_model.predict(X_test_s_last)
    y_prob_internal = final_model.predict_proba(X_test_s_last)

    # 混淆矩阵与ROC
    cm_internal = confusion_matrix(y_test_last, y_pred_internal)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_internal, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Internal Validation: Confusion Matrix')
    plt.savefig(os.path.join(results_path, "internal_confusion_matrix.png"), bbox_inches='tight')
    plt.close()

    y_test_bin_internal = label_binarize(y_test_last, classes=range(len(labels)))
    plt.figure(figsize=(10, 8))
    for i in range(len(labels)):
        fpr, tpr, _ = roc_curve(y_test_bin_internal[:, i], y_prob_internal[:, i])
        plt.plot(fpr, tpr, label=f'ROC {labels[i]} (AUC = {auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(results_path, "internal_roc_curve.png"), bbox_inches='tight')
    plt.close()

    print("  计算内部验证指标...")
    internal_metrics_basic, internal_ci = get_metrics_with_ci(
        np.array(y_test_last), y_pred_internal, y_prob_internal, labels,
        n_bootstrap=BOOTSTRAP_N, ci_level=CI_LEVEL
    )
    internal_ci.to_excel(os.path.join(results_path, "internal_metrics_with_CI.xlsx"), index=False)

    print("  计算SHAP值...")
    background_size = min(100, len(X_test_df))
    background_data = shap.kmeans(X_test_df, background_size)
    explainer = shap.KernelExplainer(final_model.predict_proba, background_data)
    shap_explanation = explainer(X_test_df)

    shap.summary_plot(shap_explanation, X_test_df, class_names=labels, show=False)
    plt.savefig(os.path.join(shap_output_path, "shap_summary_beeswarm.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 计算平均SHAP
    mean_shap_all_classes = pd.DataFrame(index=best_features)
    for class_idx, class_name in enumerate(labels):
        class_shap_values = shap_explanation.values[:, :, class_idx]
        mean_shap = np.mean(class_shap_values, axis=0)
        mean_shap_all_classes[f'Mean_SHAP_{class_name}'] = mean_shap

    mean_shap_all_classes.to_excel(os.path.join(shap_output_path, "mean_shap_table_all_classes.xlsx"))

    # ===================== 8. 外部验证 =====================
    print("\n" + "=" * 50)
    print("Step 8: 外部验证 (使用相同外部参数标准化)")
    print("=" * 50)

    all_encoded_external = []
    ext_probs = []
    ext_y_true = []

    for i in range(1, m + 1):
        valid_file = os.path.join(valid_base_path, f"df_valid_imputed_{i}.xlsx")
        df_valid_raw = pd.read_excel(valid_file)

        # 1. 基础预处理
        df_v, _ = dynamic_preprocess(df_valid_raw)

        # 2. 【关键步骤】应用同样的外部标准化参数
        df_v = apply_external_standardization(df_v, std_params, num_vars)

        df_v_save = df_v.copy()
        df_v_save['Imputation_Set'] = i
        all_encoded_external.append(df_v_save)
        df_v.to_excel(os.path.join(encoded_data_path, f"External_Encoded_Dataset_{i}.xlsx"), index=False)

        df_v = df_v.reindex(columns=best_features + [target_col, id_col], fill_value=0)
        X_v = df_v[best_features].values
        y_v = df_v[target_col].astype(int).values
        ids_v = df_v[id_col]

        y_prob_v = final_model.predict_proba(X_v)
        y_pred_v = final_model.predict(X_v)

        prob_cols = [f"Prob_{labels[j]}" for j in range(len(labels))]
        pred_df = pd.DataFrame({
            'ID': ids_v, 'True_Label_Encoded': y_v, 'Predicted_Label_Encoded': y_pred_v
        })
        pred_df[prob_cols] = y_prob_v
        pred_df.to_excel(os.path.join(results_path, f"external_predictions_imputation_{i}.xlsx"), index=False)

        ext_probs.append(y_prob_v)
        ext_y_true.append(y_v)

    combined_external_encoded = pd.concat(all_encoded_external, ignore_index=True)
    combined_external_encoded.to_excel(
        os.path.join(encoded_data_path, "All_External_Encoded_Combined.xlsx"), index=False
    )

    # 池化结果分析
    avg_ext_prob = np.mean(ext_probs, axis=0)
    avg_y_pred = np.argmax(avg_ext_prob, axis=1)
    y_v_final = ext_y_true[0]

    cm_pooled = confusion_matrix(y_v_final, avg_y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_pooled, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('External Validation: Pooled Confusion Matrix')
    plt.savefig(os.path.join(results_path, "external_pooled_confusion_matrix.png"), bbox_inches='tight')
    plt.close()

    y_v_bin_final = label_binarize(y_v_final, classes=range(len(labels)))
    plt.figure(figsize=(10, 8))
    for j in range(len(labels)):
        fpr, tpr, _ = roc_curve(y_v_bin_final[:, j], avg_ext_prob[:, j])
        plt.plot(fpr, tpr, label=f'ROC {labels[j]} (AUC = {auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(results_path, "external_pooled_roc_curve.png"), bbox_inches='tight')
    plt.close()

    print("  计算外部验证指标...")
    external_metrics_basic, external_ci = get_metrics_with_ci(
        y_v_final, avg_y_pred, avg_ext_prob, labels,
        n_bootstrap=BOOTSTRAP_N, ci_level=CI_LEVEL
    )
    external_ci.to_excel(os.path.join(results_path, "external_pooled_metrics_with_CI.xlsx"), index=False)

    # ===================== 9. 生成综合报告 =====================
    print("\n" + "=" * 50)
    print("Step 9: 生成综合结果报告")
    print("=" * 50)

    summary_report_path = os.path.join(results_path, "Comprehensive_Results_Summary.xlsx")
    with pd.ExcelWriter(summary_report_path) as writer:
        mapping_df.to_excel(writer, sheet_name='类别映射', index=False)
        pd.DataFrame({'Selected_Features': best_features}).to_excel(writer, sheet_name='最终特征', index=False)
        internal_ci.to_excel(writer, sheet_name='内部验证指标_CI', index=False)
        external_ci.to_excel(writer, sheet_name='外部验证指标_CI', index=False)
        coef_df.to_excel(writer, sheet_name='模型系数')
        mean_shap_all_classes.to_excel(writer, sheet_name='SHAP特征重要性')

    print(f"综合报告保存至: {summary_report_path}")
    print("\n=== 所有流程执行完毕 (Standardization Applied) ===")


if __name__ == "__main__":
    main()