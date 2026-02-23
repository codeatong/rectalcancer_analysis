import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# 忽略一些不必要的警告
warnings.filterwarnings("ignore")

# 设置绘图风格
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

'''
只分析吃和不吃的增长关系
'''
def analyze_pork_intake(file_path):
    print("正在读取数据...")
    df = pd.read_csv(file_path)

    # 清除列名空格（防止 ' age' 这种错误）
    df.columns = df.columns.str.strip()

    # ==========================================
    # 1. 计算 age 变量
    # ==========================================
    if 'birth_year' in df.columns:
        df['age'] = 2025 - df['birth_year']
        print("-> 已成功从 birth_year 计算 age")
    elif 'age' not in df.columns:
        print("错误：CSV中既没有 age 也没有 birth_year，无法控制年龄变量！")
        return

    # ==========================================
    # 2. 数据清洗
    # ==========================================
    target_col = 'pork_intake'  # 目标变量是猪肉摄入量

    # 检查目标列是否存在
    if target_col not in df.columns:
        print(f"错误：未找到列 {target_col}，请检查拼写")
        return

    # 剔除无效回答 (-1, -3)
    df_clean = df[df[target_col] >= 0].copy()

    # 简单的缺失值填充 (针对公式里用到的变量)
    # 确保公式里的变量都在列名里
    needed_cols = ['label', 'pork_intake', 'age', 'gender', 'bmi', 'poverty']

    # 检查所有需要的列是否都在
    missing_cols = [c for c in needed_cols if c not in df_clean.columns]
    if missing_cols:
        print(f"错误：缺少以下列，无法运行回归：{missing_cols}")
        return

    # 填充缺失值，防止回归报错
    for col in needed_cols:
        if df_clean[col].dtype == 'object':
            # 如果是字符串(比如gender)，转成类别编码
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        else:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    print(f"清洗后样本量: {len(df_clean)} (剔除了无效问卷)")

    # ==========================================
    # 方法 A: 线性模型 (计算绝对风险 ARI)
    # ==========================================
    print("\n[1/2] 正在运行线性回归...")
    # C(gender) 表示把 gender 当作分类变量处理，防止报错
    formula_linear = "label ~ pork_intake + age + C(gender) + bmi + poverty"

    # formula_linear = "label ~ C(pork_intake, Treatment(reference='Never')) + age + C(gender) + bmi + poverty"
    # formula_linear = "label ~ C(pork_intake, Treatment(reference='Never')) + age + C(gender) + bmi + poverty"

    try:
        model_linear = smf.ols(formula_linear, data=df_clean).fit()
        beta_linear = model_linear.params['pork_intake']
        p_linear = model_linear.pvalues['pork_intake']

        absolute_risk_increase = beta_linear * 5  # 从Level 0 到 Level 5

        print(f"   -> 系数 (Beta): {beta_linear:.5f}")
        print(f"   -> P值: {p_linear:.4f}")
        print(f"   -> 结论：相比'从不吃'，'每天吃' 绝对概率增加: {absolute_risk_increase * 100:.3f}%")

    except Exception as e:
        print(f"线性回归出错: {e}")

    # ==========================================
    # 方法 B: 逻辑回归 (计算优势比 Odds Ratio)
    # ==========================================
    print("\n[2/2] 正在运行逻辑回归 (Logistic Regression)...")
    try:
        model_logit = smf.logit(formula_linear, data=df_clean).fit(disp=0)

        coef_logit = model_logit.params['pork_intake']
        conf_int = model_logit.conf_int().loc['pork_intake']

        or_per_unit = np.exp(coef_logit)
        or_daily_vs_never = np.exp(coef_logit * 5)  # 0到5级跨度为5

        print(f"   -> 单级 OR: {or_per_unit:.3f}")
        print(f"   -> ★★★ 每天吃 vs 从不吃 总 OR: {or_daily_vs_never:.3f}")
        print(f"   -> 解释: 风险是原来的 {or_daily_vs_never:.2f} 倍 (+{(or_daily_vs_never - 1) * 100:.1f}%)")

        # ==========================================
        # 可视化
        # ==========================================
        levels = np.arange(0, 6)
        risk_multipliers = [np.exp(coef_logit * x) for x in levels]

        plt.figure(figsize=(10, 6))
        sns.lineplot(x=levels, y=risk_multipliers, marker='o', color='darkred', linewidth=2)

        # 画置信区间
        ci_upper = [np.exp(conf_int[1] * x) for x in levels]
        ci_lower = [np.exp(conf_int[0] * x) for x in levels]
        plt.fill_between(levels, ci_lower, ci_upper, color='red', alpha=0.1)

        plt.text(5, risk_multipliers[-1], f"{risk_multipliers[-1]:.2f}x Risk",
                 va='bottom', ha='right', fontweight='bold', color='darkred')

        plt.xticks(levels, ['Never(0)', '<1/wk(1)', '1/wk(2)', '2-4/wk(3)', '5-6/wk(4)', 'Daily(5)'])
        plt.axhline(y=1, color='gray', linestyle='--')
        plt.title('猪肉摄入频率与直肠癌风险 (Odds Ratio)', fontsize=14)
        plt.ylabel('相对风险倍数 (Odds Ratio)', fontsize=12)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"逻辑回归出错: {e}")


# 分析每周猪肉不同摄入频率的影响
def analyze_pork_intake_categorical(file_path):
    print("正在读取数据...")
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    # 1. 变量计算与清洗
    if 'birth_year' in df.columns:
        df['age'] = 2025 - df['birth_year']
    elif 'age' not in df.columns:
        print(" 错误：缺少 age 或 birth_year")
        return

    target_col = 'pork_intake'
    df_clean = df[df[target_col].isin([0, 1, 2, 3, 4, 5])].copy()

    # === 分组合并 (如之前讨论，合并高频组以稳定结果) ===
    # 将 5 (Daily) 合并入 4 (5-6/wk)，统称为 Level 4 (>=5/wk)
    df_clean[target_col] = df_clean[target_col].replace({5: 4})
    df_clean[target_col] = df_clean[target_col].astype(int)

    # 填充缺失值
    needed_cols = ['label', 'pork_intake', 'age', 'gender', 'bmi', 'poverty']
    for col in needed_cols:
        if col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # 2. 运行逻辑回归
    print(f"\n[分析结果] 样本量: {len(df_clean)} (已合并 Daily 到 5-6/wk 组)")
    formula = "label ~ C(pork_intake) + age + C(gender) + bmi + poverty"

    try:
        model = smf.logit(formula, data=df_clean).fit(disp=0)

        # === 关键步骤：计算绝对概率差异 (Marginal Effects) ===
        # 我们需要计算：如果所有人都变成Level X，相比所有人都变成Level 0，概率平均增加了多少？

        # 1. 创建基准数据（假设所有人都是 Level 0: Never）
        df_base = df_clean.copy()
        df_base['pork_intake'] = 0
        prob_base = model.predict(df_base)  # 预测基准概率

        labels_map = {
            1: '<1/wk',
            2: '1/wk',
            3: '2-4/wk',
            4: '≥5/wk'
        }

        print("\n" + "=" * 50)
        print("不同猪肉摄入频率对直肠癌的绝对风险影响")
        print("=" * 50)

        # 提取模型系数和P值
        params = model.params
        pvalues = model.pvalues

        for i in range(1, 5):  # 遍历 Level 1 到 4
            level_name = labels_map[i]
            col_name = f'C(pork_intake)[T.{i}]'

            if col_name in params:
                # 获取 Beta 和 P值
                beta = params[col_name]
                p_val = pvalues[col_name]

                # 计算绝对概率增加 (AME方法)
                # 1. 假设所有人都是 Level i
                df_counterfactual = df_clean.copy()
                df_counterfactual['pork_intake'] = i
                prob_current = model.predict(df_counterfactual)

                # 2. 计算平均概率差值 (Level i 平均概率 - Level 0 平均概率)
                risk_diff = (prob_current - prob_base).mean()

                # 格式化输出
                print(f"\n【对比组：{level_name}】")
                print(f"   -> 系数 (Beta): {beta:.5f}")
                print(f"   -> P值: {p_val:.4f}")

                # 判断显著性并输出结论
                significance = "" if p_val < 0.05 else " (统计不显著)"
                print(f"   -> 结论：相比'从不吃'，'{level_name}' 绝对概率增加: {risk_diff * 100:.3f}%{significance}")

        print("\n" + "=" * 50)

    except Exception as e:
        print(f"分析出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    file_path = "/Users/tongan/tongan/project/ukbank_rectal cancer_analysis/dataset/merged_with_C18_flag_label_final.csv"
    analyze_pork_intake(file_path)
    # analyze_pork_intake_categorical(file_path)
