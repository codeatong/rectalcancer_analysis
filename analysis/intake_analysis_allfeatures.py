import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


# 所有可能导致直肠癌的特征 每个特征具体分析的代码
# 忽略警告
warnings.filterwarnings("ignore")

# 设置绘图风格 (适配中文)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def analyze_single_feature_detailed(df, feature_name):
    """
    对单个特征进行全套深度分析：清洗 -> 线性回归 -> 逻辑回归 -> 独立绘图
    """
    print(f"\n{'=' * 30}")
    print(f"正在深入分析特征: 【 {feature_name} 】")
    print(f"{'=' * 30}")

    # 1. 针对该特征的数据清洗
    if feature_name not in df.columns:
        print(f"跳过: 列 {feature_name} 不存在")
        return

    # 剔除无效值 (-1, -3), 保留 0-5
    df_clean = df[df[feature_name] >= 0].copy()

    # 简单的缺失值填充 (控制变量)
    controls = ['age', 'gender', 'bmi', 'poverty']
    for col in controls:
        if col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # 公式
    formula = f"label ~ {feature_name} + age + C(gender) + bmi + poverty"

    # ------------------------------------------
    # A. 线性回归 (计算绝对风险 ATE)
    # ------------------------------------------
    try:
        model_linear = smf.ols(formula, data=df_clean).fit()
        beta = model_linear.params[feature_name]
        p_linear = model_linear.pvalues[feature_name]

        # 绝对风险增加 (假设从 Level 0 到 Level 5)
        ate_total = beta * 5
        print(f"1. [绝对风险] (Linear Regression)")
        print(f"   - 每级系数: {beta:.6f}")
        print(f"   - P值: {p_linear:.4f}")
        print(f"   - 结论: 相比从不吃(0)，每天吃(5) 绝对概率增加: {ate_total * 100:.3f}%")

    except Exception as e:
        print(f"   线性回归出错: {e}")

    # ------------------------------------------
    # B. 逻辑回归 (计算相对风险 OR - SCI标准)
    # ------------------------------------------
    try:
        model_logit = smf.logit(formula, data=df_clean).fit(disp=0)
        coef = model_logit.params[feature_name]
        p_logit = model_logit.pvalues[feature_name]
        conf_int = model_logit.conf_int().loc[feature_name]

        # 计算 OR
        or_per_unit = np.exp(coef)
        or_total = np.exp(coef * 5)  # 0 to 5

        # 置信区间 (Total)
        ci_lower_total = np.exp(conf_int[0] * 5)
        ci_upper_total = np.exp(conf_int[1] * 5)

        print(f"2. [相对风险] (Logistic Regression)")
        print(f"   - 单级 OR: {or_per_unit:.3f}")
        print(f"   - ★★★ 每天吃(5) vs 从不吃(0) 总OR: {or_total:.3f}")
        print(f"   - 95% 置信区间: [{ci_lower_total:.3f}, {ci_upper_total:.3f}]")
        print(f"   - 结论: 风险倍数是原来的 {or_total:.2f} 倍 (增加 {(or_total - 1) * 100:.1f}%)")

        # ------------------------------------------
        # C. 独立绘图 (Dose-Response Curve)
        # ------------------------------------------
        levels = np.arange(0, 6)
        risk_multipliers = [np.exp(coef * x) for x in levels]

        # 计算置信区间带
        ci_upper_band = [np.exp(conf_int[1] * x) for x in levels]
        ci_lower_band = [np.exp(conf_int[0] * x) for x in levels]

        plt.figure(figsize=(8, 6))

        # 绘制主曲线
        color = 'darkred' if or_total > 1 else 'green'  # 风险用红，保护用绿
        sns.lineplot(x=levels, y=risk_multipliers, marker='o', color=color, linewidth=2.5, markersize=8)

        # 绘制阴影区域
        plt.fill_between(levels, ci_lower_band, ci_upper_band, color=color, alpha=0.15, label='95% CI')

        # 标注末端数值
        plt.text(5.1, risk_multipliers[-1], f"{risk_multipliers[-1]:.2f}x",
                 color=color, fontweight='bold', va='center', fontsize=12)

        # 装饰
        plt.axhline(y=1, color='gray', linestyle='--', linewidth=1)
        plt.title(f'特征分析: {feature_name}\n风险趋势 (OR={or_total:.2f}, P={p_logit:.4f})', fontsize=14)
        plt.xlabel('摄入频率 (0=Never -> 5=Daily)', fontsize=12)
        plt.ylabel('相对风险倍数 (Odds Ratio)', fontsize=12)
        plt.xticks(levels, ['Never(0)', '<1/wk(1)', '1/wk(2)', '2-4/wk(3)', '5-6/wk(4)', 'Daily(5)'])
        plt.xlim(-0.5, 5.8)  # 留出一点空间给文字
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"   逻辑回归或绘图出错: {e}")


def main_analysis(file_path):
    print(f"读取文件: {file_path}")
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    # 1. 计算年龄
    if 'birth_year' in df.columns:
        df['age'] = 2025 - df['birth_year']
        print("-> 年龄计算完成")
    else:
        print("无法计算年龄，程序终止")
        return

    # 2. 定义要单独分析的特征列表
    features = [
        'pork_intake',
        'cooked_meat_intake',
        'beef_intake',
        'lamb_intake',
        'salt',
        'cheese_intake'
    ]

    # 3. 循环调用分析函数
    for feature in features:
        analyze_single_feature_detailed(df, feature)


if __name__ == "__main__":
    file_path = "/Users/tongan/tongan/project/ukbank_rectal cancer_analysis/dataset/merged_with_C18_flag_label_final.csv"
    main_analysis(file_path)