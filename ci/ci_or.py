import pandas as pd
import numpy as np
from dowhy import CausalModel
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
import os


class CausalCancerORAnalysis:
    def __init__(self, data_path):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"文件未找到: {data_path}")

        print(f"正在读取文件: {data_path}")
        self.df = pd.read_csv(data_path)
        self.preprocess_data()

    def preprocess_data(self):
        print("Step 1: 数据预处理...")

        self.df.columns = self.df.columns.str.strip()

        # 年龄
        if 'birth_year' in self.df.columns:
            self.df['age'] = 2025 - self.df['birth_year']
        elif 'age' not in self.df.columns:
            self.df['age'] = 50

        # 去掉ID
        if 'eid' in self.df.columns:
            self.df.drop(columns=['eid'], inplace=True)

        # 变量清洗（保留你原来的）
        if 'alcohol_intake_frequency' in self.df.columns:
            alcohol_map = {1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1, -3: np.nan}
            self.df['alcohol_intake_frequency'] = self.df['alcohol_intake_frequency'].map(alcohol_map)

        if 'insomnia' in self.df.columns:
            self.df['insomnia'] = self.df['insomnia'].replace({-3: np.nan})

        if 'chronotype' in self.df.columns:
            self.df['chronotype'] = self.df['chronotype'].replace({-1: np.nan, -3: np.nan})

        if 'smoking_packs' in self.df.columns:
            self.df['smoking_packs'] = self.df['smoking_packs'].replace({-1: np.nan, -10: np.nan})

        # confounders 和 exposures
        self.confounders_base = ['age', 'gender', 'finish_education_age', 'poverty']

        self.exposures = [
            'processed_meat_intake', 'beef_intake', 'lamb_intake', 'pork_intake',
            'cooked_vegetable_intake', 'raw_vegetable_intake', 'cheese_intake',
            'grain_intake', 'salt', 'tea_intake', 'coffee_intake',
            'sugary_drinks', 'alcohol_intake_frequency', 'insomnia',
            'chronotype', 'exercise_weekly(>10min)', 'watch_tv_time',
            'play_computer_time', 'time_spent_driving', 'smoking_packs',
            'pack_years_proportion'
        ]

        # 缺失值填充与编码
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                self.df[col] = self.df[col].fillna('Unknown')
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
            else:
                self.df[col] = self.df[col].fillna(self.df[col].median())

        # 确保 label 是 0/1
        if 'label' in self.df.columns:
            self.df['label'] = self.df['label'].astype(int)

    def _fit_logit_or(self, tmp_df, treatment, confounders):
        """用 statsmodels Logit 拟合并返回 OR/CI/p"""
        X = tmp_df[[treatment] + confounders]
        X = sm.add_constant(X, has_constant='add')
        y = tmp_df['label']

        # Logit 拟合
        model = sm.Logit(y, X).fit(disp=0)

        beta = model.params[treatment]
        ci_beta = model.conf_int().loc[treatment]
        p_val = model.pvalues[treatment]

        OR = float(np.exp(beta))
        CI_low = float(np.exp(ci_beta[0]))
        CI_high = float(np.exp(ci_beta[1]))

        return OR, CI_low, CI_high, float(p_val)

    def run_analysis(self, do_placebo_refute=False):
        results = []
        print("Step 2: DoWhy 识别(backdoor) + Logistic 回归输出 OR ...")

        actual_treatments = [c for c in self.exposures if c in self.df.columns]
        if not actual_treatments:
            print("错误: 未找到有效暴露列！")
            return pd.DataFrame()

        confounders = [c for c in self.confounders_base if c in self.df.columns]
        if not confounders:
            print("错误: 未找到混杂变量列！")
            return pd.DataFrame()

        for i, treatment in enumerate(actual_treatments):
            print(f"[{i+1}/{len(actual_treatments)}] 分析中: {treatment}")

            cols_needed = [treatment, 'label'] + confounders
            tmp_df = self.df[cols_needed].dropna()
            if tmp_df.empty:
                print(f"   -> 跳过 {treatment} (数据为空)")
                continue

            try:
                # 1) DoWhy: 建模 + 识别（说明我们是在 backdoor 调整框架下）
                causal_model = CausalModel(
                    data=tmp_df,
                    treatment=treatment,
                    outcome='label',
                    common_causes=confounders,
                    logging_level="ERROR"
                )
                identified_estimand = causal_model.identify_effect(proceed_when_unidentifiable=True)

                # 2) 用 Logit 输出 OR（核心：你想要的 OR）
                OR, CI_low, CI_high, p_val = self._fit_logit_or(tmp_df, treatment, confounders)

                row = {
                    'Feature': treatment,
                    'OR': OR,
                    'CI_Low': CI_low,
                    'CI_High': CI_high,
                    'p_value': p_val,
                    'Identified': True,
                    'Estimand': str(identified_estimand.estimand_type) if hasattr(identified_estimand, "estimand_type") else "NA"
                }

                # 3) （可选）DoWhy refuter：注意它是针对 DoWhy 的 estimate（通常是风险差），不是 OR
                if do_placebo_refute:
                    try:
                        est = causal_model.estimate_effect(
                            identified_estimand,
                            method_name="backdoor.linear_regression",
                            test_significance=False
                        )
                        ref = causal_model.refute_estimate(
                            identified_estimand, est,
                            method_name="placebo_treatment_refuter"
                        )
                        row['placebo_refute_p'] = ref.refutation_result.get('p_value', np.nan)
                    except Exception:
                        row['placebo_refute_p'] = np.nan

                results.append(row)

            except Exception as e:
                print(f"   -> 跳过 (原因: {str(e)[:120]})")
                continue

        return pd.DataFrame(results)

    def plot_forest_or(self, results_df):
        if results_df.empty:
            print("无数据可绘图。")
            return

        print("\nStep 3: 生成 OR 森林图...")
        df_plot = results_df.sort_values(by='OR', ascending=True).reset_index(drop=True)

        min_ci = df_plot['CI_Low'].min()
        max_ci = df_plot['CI_High'].max()
        data_range = max_ci - min_ci if max_ci > min_ci else 1.0

        pad = data_range * 0.10
        star_offset = data_range * 0.02

        plt.figure(figsize=(14, max(6, len(df_plot) * 0.5)))
        y_pos = np.arange(len(df_plot))

        xerr_low = df_plot['OR'] - df_plot['CI_Low']
        xerr_high = df_plot['CI_High'] - df_plot['OR']

        plt.errorbar(
            df_plot['OR'], y_pos,
            xerr=[xerr_low, xerr_high],
            fmt='o', color='black', ecolor='gray', capsize=5,
            label='OR (95% CI)'
        )

        plt.axvline(x=1, color='red', linestyle='--', linewidth=1, label='No Effect (OR=1)')
        plt.yticks(y_pos, df_plot['Feature'])
        plt.xlabel('Odds Ratio (OR)')
        plt.title("Backdoor-adjusted association framed as causal (DoWhy identify + Logit OR)")
        plt.xlim(min_ci - pad, max_ci + pad)
        plt.legend()

        # 星号：CI 不跨 1
        for i in range(len(df_plot)):
            is_sig = (df_plot.loc[i, 'CI_Low'] > 1) or (df_plot.loc[i, 'CI_High'] < 1)
            if is_sig:
                plt.text(df_plot.loc[i, 'CI_High'] + star_offset, i, "*",
                         va='center', fontsize=15, color='red', fontweight='bold')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    file_path = "/Users/tongan/tongan/project/ukbank_rectal cancer_analysis/dataset/merged_with_C18_flag_label_final.csv"

    analysis = CausalCancerORAnalysis(file_path)
    results = analysis.run_analysis(do_placebo_refute=False)

    if not results.empty:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.float_format', '{:.4f}'.format)

        # 显著性列
        results['Significance'] = results['p_value'].apply(
            lambda x: '***' if x < 0.001 else ('**' if x < 0.01 else ('*' if x < 0.05 else ''))
        )

        # 按离 1 的距离排序
        results['Dist_from_1'] = (results['OR'] - 1).abs()
        show = results.sort_values('Dist_from_1', ascending=False).drop(columns=['Dist_from_1'])
        print(show[['Feature', 'OR', 'CI_Low', 'CI_High', 'p_value', 'Significance']])

        analysis.plot_forest_or(results)
    else:
        print("无结果。")
