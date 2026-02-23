# 控制变量: 年龄 (age) + BMI (bmi) + 腰围 (waist) + 臀围 (hip) + 性别 (gender) + 贫困指数 (poverty)
# 建议使用这个

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def get_significance_star(p_val):
    """根据 P 值返回显著性星号"""
    if pd.isna(p_val): return ""
    if p_val < 0.001:
        return "***"
    elif p_val < 0.01:
        return "**"
    elif p_val < 0.05:
        return "*"
    else:
        return ""


def print_detailed_growth_analysis(res_df):
    """
    【新增功能】输出详细的增长率文字报告
    计算环比增长（比上一级）和定比增长（比不吸烟）
    """
    print("\n" + "=" * 50)
    print("阶梯式风险增长解读报告")
    print("=" * 50)

    # 这里的 prev_or 初始化为 1.0 (基准组的 OR)
    prev_or = 1.0
    prev_group_name = "None (不吸)"

    for index, row in res_df.iterrows():
        curr_group = row['Group']
        curr_or = row['OR']
        p_val = row['P-val']
        star = get_significance_star(p_val)

        # 跳过第一行（基准组）
        if index == 0:
            print(f"🔹 {curr_group} (基准组): 风险设定为 1.0 倍")
            continue

        # 1. 计算相对于【基准组(不吸)】的增长
        increase_vs_baseline = (curr_or - 1) * 100

        # 2. 计算相对于【前一组】的增长 (核心逻辑: 当前OR / 前一组OR - 1)
        increase_vs_prev = ((curr_or / prev_or) - 1) * 100

        print(f"\n组别: {curr_group} (OR={curr_or:.2f}{star})")

        # 打印对比不吸烟
        direction = "增加" if increase_vs_baseline > 0 else "降低"
        print(f" 1. 对比 [不吸烟]: 患癌风险{direction}了 {abs(increase_vs_baseline):.1f}%")

        # 打印对比上一组
        direction_prev = "增加" if increase_vs_prev > 0 else "降低"
        print(f" 2. 对比 [{prev_group_name}]: 风险进一步{direction_prev}了 {abs(increase_vs_prev):.1f}%")

        # 更新"前一组"的数据，供下一次循环使用
        prev_or = curr_or
        prev_group_name = curr_group.split(' ')[0]  # 简化名字，只取英文部分

    print("=" * 50 + "\n")

def analyze_smoking_by_groups(file_path):
    print(" 开始按照【分组对比法】进行分析...")

    # 1. 读取数据
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    target = 'pack_years_proportion'

    # 2. 核心清洗逻辑
    # 将 -1, -3 视为无效值 (NaN)
    # df[target] = df[target].replace({-1: np.nan, -3: np.nan})

    # 【关键修正】将 NaN 填充为 0 (假设空值即为不吸烟)
    df[target] = df[target].fillna(0)

    # 截断大于1的异常值
    df[target] = np.where(df[target] > 1, 1, df[target])

    # 3. 创建分组 (Binning)
    bins = [-0.1, 0, 0.2, 0.4, 1.1]
    labels = ['None (不吸)', 'Low (轻度)', 'Medium (中度)', 'High (重度)']

    df['smoking_group'] = pd.cut(df[target], bins=bins, labels=labels)

    print("各组样本量分布:")
    print(df['smoking_group'].value_counts().sort_index())

    # 4. 准备控制变量
    if 'age' not in df.columns and 'birth_year' in df.columns:
        df['age'] = 2025 - df['birth_year']

    # 填充控制变量缺失值
    for col in ['age', 'bmi', 'poverty', 'gender']:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())

    # 5. 运行逻辑回归
    formula = "label ~ C(smoking_group, Treatment(reference='None (不吸)')) + age + bmi + hip + waist"
    if 'gender' in df.columns: formula += " + C(gender)"
    if 'poverty' in df.columns: formula += " + poverty"

    try:
        model = smf.logit(formula, data=df).fit(disp=0)

        # 提取结果
        results = []
        # 基准组手动添加
        results.append({'Group': 'None (不吸)', 'OR': 1.0, 'Lower': 1.0, 'Upper': 1.0, 'P-val': 1.0})

        # 提取其他组
        for i, label in enumerate(labels[1:]):
            try:
                var_name = f"C(smoking_group, Treatment(reference='None (不吸)'))[T.{label}]"
                coef = model.params[var_name]
                conf = model.conf_int().loc[var_name]
                pval = model.pvalues[var_name]

                results.append({
                    'Group': label,
                    'OR': np.exp(coef),
                    'Lower': np.exp(conf[0]),
                    'Upper': np.exp(conf[1]),
                    'P-val': pval
                })
            except KeyError:
                print(f"警告: 组别 {label} 样本太少，无法计算")

        res_df = pd.DataFrame(results)

        print("\n" + "=" * 40)
        print("🩺 基础数据表")
        print("=" * 40)
        print(res_df[['Group', 'OR', 'P-val']].to_string(index=False))
        print("-" * 40)

        # 调用新增的文字分析函数
        print_detailed_growth_analysis(res_df)

        # === 绘图部分 ===
        # 1. 画原来的柱状图
        plot_or_results(res_df)
        # 2. 画新的趋势曲线图 (对比用)
        plot_trend_curve(res_df)

    except Exception as e:
        print(f"分析出错: {e}")


def plot_or_results(df):
    plt.figure(figsize=(10, 7))  # 稍微把图拉高一点

    yerr = [df['OR'] - df['Lower'], df['Upper'] - df['OR']]
    colors = ['gray' if p > 0.05 and g != 'None (不吸)' else '#d62728' for p, g in zip(df['P-val'], df['Group'])]
    colors[0] = '#2ca02c'

    bars = plt.bar(df['Group'], df['OR'], yerr=yerr, capsize=10, color=colors, alpha=0.8, width=0.6)

    plt.axhline(y=1, color='black', linestyle='--', linewidth=1)
    plt.ylabel('患直肠癌风险倍数 (Odds Ratio)', fontsize=12)
    plt.title('不同程度吸烟占比与患癌风险对比 (柱状图)', fontsize=14)

    # === 关键修改点 1: 动态设置 Y 轴上限，防止文字被切掉 ===
    # 获取整个数据中最高的点（可能是置信区间的上限）
    max_height = df['Upper'].max()
    # 让图表的顶端留出 15% 的空白空间给文字
    plt.ylim(0, max_height * 1.15)

    # === 关键修改点 2: 调整文字位置 ===
    # zip 中加入了 df['Upper']，我们要基于置信区间上限来定位
    for bar, or_val, p_val, upper_val, group in zip(bars, df['OR'], df['P-val'], df['Upper'], df['Group']):
        # text = f"{or_val:.2f}x"
        # star = get_significance_star(p_val) if group != 'None (不吸)' else ""
        text = f"{or_val:.2f}"
        if p_val < 0.05 and or_val != 1.0:
            text += "*"

        # 计算文字的 Y 坐标：
        # 取 (柱子高度 OR值) 和 (误差棒顶端 Upper) 的最大值
        # 然后再往上加一点点偏移量 (比如最大高度的 2%)
        text_y = max(or_val, upper_val) + (max_height * 0.02)

        plt.text(bar.get_x() + bar.get_width() / 2.,
                 text_y,
                 text,
                 ha='center',
                 va='bottom',
                 fontweight='bold',
                 fontsize=11)  # 字体稍微加大一点

    plt.tight_layout()
    plt.show()


def plot_trend_curve(df):
    """
    【新增函数】画趋势曲线图 (Line Plot with Confidence Band)
    """
    plt.figure(figsize=(10, 6))

    # 将组名映射为数字索引 (0, 1, 2, 3) 以便画线
    x_indices = range(len(df))

    # 1. 画主趋势线 (红色实线 + 圆点)
    plt.plot(x_indices, df['OR'], marker='o', markersize=8, color='#d62728', linewidth=2.5, label='Risk Trend')

    # 2. 画置信区间阴影 (红色半透明区域)
    plt.fill_between(x_indices, df['Lower'], df['Upper'], color='#d62728', alpha=0.15, label='95% CI')

    # 3. 辅助线
    plt.axhline(y=1, color='gray', linestyle='--', linewidth=1, label='Baseline (OR=1)')

    # 4. 标注数值
    for x, y, p_val, group in zip(x_indices, df['OR'], df['P-val'], df['Group']):
        # plt.text(x, y + 0.05, f"{y:.2f}x", ha='center', va='bottom', fontweight='bold', color='#d62728', fontsize=11)
        star = get_significance_star(p_val) if group != 'None (不吸)' else ""
        plt.text(x, y + 0.05, f"{y:.2f}{star}", ha='center', va='bottom', fontweight='bold', color='#d62728', fontsize=11)

    # 5. 美化图表
    plt.xticks(x_indices, df['Group'], fontsize=11)
    plt.yticks(fontsize=11)
    plt.ylabel('相对风险倍数 (Odds Ratio)', fontsize=12)
    plt.title('吸烟占比与直肠癌风险趋势 (剂量-反应关系)', fontsize=14)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)  # 添加网格线方便看数

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    fpath = "/Users/tongan/tongan/project/ukbank_rectal_cancer_analysis/dataset/merged_with_C18_flag_label_final.csv"
    analyze_smoking_by_groups(fpath)


