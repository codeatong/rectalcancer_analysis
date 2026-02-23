import pandas as pd
import numpy as np
import dowhy
from dowhy import CausalModel
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import os

# 这个是最终版本 修复了ate 红色新号位置问题
class CausalCancerAnalysis:
    def __init__(self, data_path):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"文件未找到: {data_path}")

        print(f"正在读取文件: {data_path}")
        self.df = pd.read_csv(data_path)
        self.preprocess_data()

    def preprocess_data(self):
        print("Step 1: 数据预处理...")

        self.df.columns = self.df.columns.str.strip()
        print(f"-> CSV实际包含列名(前5个): {list(self.df.columns)[:5]}...")

        if 'birth_year' in self.df.columns:
            self.df['age'] = 2025 - self.df['birth_year']
        elif 'age' not in self.df.columns:
            self.df['age'] = 50

        if 'eid' in self.df.columns:
            self.df.drop(columns=['eid'], inplace=True)

            # --- A. alcohol ---
            if 'alcohol_intake_frequency' in self.df.columns:
                print("   -> 正在修正 'alcohol_intake_frequency' 编码方向...")
                alcohol_map = {1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1, -3: np.nan}
                self.df['alcohol_intake_frequency'] = self.df['alcohol_intake_frequency'].map(alcohol_map)

            # --- B. insomnia ---
            if 'insomnia' in self.df.columns:
                self.df['insomnia'] = self.df['insomnia'].replace({-3: np.nan})

            # --- C. chronotype ---
            if 'chronotype' in self.df.columns:
                self.df['chronotype'] = self.df['chronotype'].replace({-1: np.nan, -3: np.nan})

            # --- D. smoking ---
            if 'smoking_packs' in self.df.columns:
                self.df['smoking_packs'] = self.df['smoking_packs'].replace({-1: np.nan, -10: np.nan})

        self.confounders_base = ['age', 'gender', 'finish_education_age', 'poverty']

        self.exposures = [
            'processed_meat_intake',
            'beef_intake',
            'lamb_intake',
            'pork_intake',
            'cooked_vegetable_intake',
            'raw_vegetable_intake',
            'cheese_intake',
            'grain_intake',
            'salt',
            'tea_intake',
            'coffee_intake',
            'sugary_drinks',
            'alcohol_intake_frequency',
            'insomnia',
            'chronotype',
            'exercise_weekly(>10min)',
            'watch_tv_time',
            'play_computer_time',
            'time_spent_driving',
            'smoking_packs',
            'pack_years_proportion'
        ]

        self.body_metrics = ['bmi', 'waist', 'body_fat_percentage', 'hip']

        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                self.df[col] = self.df[col].fillna('Unknown')
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
            else:
                self.df[col] = self.df[col].fillna(self.df[col].median())

    def run_analysis(self):
        results = []
        print(f"Step 2: 开始因果分析 (自动构建图模式)...")

        actual_treatments = [col for col in self.exposures if col in self.df.columns]
        if not actual_treatments:
            print("错误: 未找到有效特征列，请检查列名拼写！")
            return pd.DataFrame()

        for i, treatment in enumerate(actual_treatments):
            print(f"[{i + 1}/{len(actual_treatments)}] 分析中: {treatment}")

            current_confounders = [c for c in self.confounders_base if c in self.df.columns]

            try:
                model = CausalModel(
                    data=self.df,
                    treatment=treatment,
                    outcome='label',
                    common_causes=current_confounders,
                    logging_level="ERROR"
                )

                identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

                estimate = model.estimate_effect(
                    identified_estimand,
                    method_name="backdoor.linear_regression",
                    test_significance=True
                )

                refute = model.refute_estimate(
                    identified_estimand,
                    estimate,
                    method_name="placebo_treatment_refuter"
                )

                se = estimate.get_standard_error()
                ate = estimate.value

                p_val = estimate.test_stat_significance().get('p_value', np.nan)

                results.append({
                    'Feature': treatment,
                    'ATE': ate,
                    'CI_Low': ate - 1.96 * se,
                    'CI_High': ate + 1.96 * se,
                    'p_value': p_val,
                    'refutation_p': refute.refutation_result.get('p_value', np.nan)
                })

            except Exception as e:
                print(f"   -> 跳过 (原因: {str(e)[:120]}...)")
                continue

        return pd.DataFrame(results)

    #显著性与 CI 一致
    def plot_forest(self, results_df, sig_by_ci=True, alpha=0.05):
        if results_df.empty:
            print("无数据可绘图。")
            return

        print("\nStep 3: 生成森林图...")
        df_plot = results_df.sort_values(by='ATE', ascending=True).reset_index(drop=True)

        min_ci = df_plot['CI_Low'].min()
        max_ci = df_plot['CI_High'].max()
        data_range = max_ci - min_ci if max_ci > min_ci else 1.0

        star_offset = data_range * 0.015
        pad_left = data_range * 0.05
        pad_right = data_range * 0.12

        plt.figure(figsize=(14, max(6, len(df_plot) * 0.5)))

        y_pos = np.arange(len(df_plot))
        plt.errorbar(
            df_plot['ATE'], y_pos,
            xerr=[df_plot['ATE'] - df_plot['CI_Low'], df_plot['CI_High'] - df_plot['ATE']],
            fmt='o', color='black', ecolor='gray', capsize=5
        )

        plt.axvline(x=0, color='red', linestyle='--', linewidth=1)
        plt.yticks(y_pos, df_plot['Feature'])
        plt.xlabel('ATE (Average Treatment Effect)')
        plt.title("Results of causal effect analysis (adjusting for selected confounders)")

        #标星
        for i in range(len(df_plot)):
            if sig_by_ci:
                # 用 CI 判定显著性：CI 不跨 0
                sig = (df_plot.loc[i, 'CI_Low'] > 0) or (df_plot.loc[i, 'CI_High'] < 0)
            else:
                # 或者用 p_value 判定（如果你坚持用 p 值）
                sig = (df_plot.loc[i, 'p_value'] < alpha)

            if sig:
                ci_high = df_plot.loc[i, 'CI_High']
                plt.text(ci_high + star_offset, i, "*", va='center', fontsize=15, color='red')

        plt.xlim(min_ci - pad_left, max_ci + pad_right)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    file_path = "/Users/tongan/tongan/project/deeplearing/PycharmProjects/ukbank/merged_with_C18_flag_label_final.csv"

    try:
        analysis = CausalCancerAnalysis(data_path=file_path)
        results_df = analysis.run_analysis()

        if not results_df.empty:
            print("\n=== Causal Analysis Results (Sorted by Impact) ===")

            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)
            pd.set_option('display.float_format', '{:.6f}'.format)

            # 显著性标记（基于 p_value）
            results_df['Significance'] = results_df['p_value'].apply(
                lambda x: '***' if x < 0.001 else ('**' if x < 0.01 else ('*' if x < 0.05 else ''))
            )

            results_df['Abs_Effect'] = results_df['ATE'].abs()
            df_show = results_df.sort_values(by='Abs_Effect', ascending=False).drop(columns=['Abs_Effect', 'refutation_p'])

            cols_to_print = ['Feature', 'ATE', 'CI_Low', 'CI_High', 'p_value', 'Significance']
            print(df_show[cols_to_print])

            # 默认用 CI 判星号（最一致）
            analysis.plot_forest(results_df, sig_by_ci=True)

        else:
            print("\n结果为空。")

    except Exception as e:
        print(f"运行出错: {e}")
