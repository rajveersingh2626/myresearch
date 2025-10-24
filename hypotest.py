import pandas as pd
from scipy.stats import mannwhitneyu
import numpy as np

df = pd.read_csv('analysis_data_fully_processed.csv')

print("\n--- Loaded DataFrame Columns (Check for 'category') ---")
print(df.columns.tolist())

df.columns = df.columns.str.strip()

if 'category' not in df.columns:
    print("\nERROR: The 'category' column is still missing after cleaning. Please check your CSV file's header.")
else:
    print("\nSUCCESS: Proceeding with analysis.")
    political = df[df['category'] == 'Political']['engagement'].copy()
    entertainment = df[df['category'] == 'Entertainment']['engagement'].copy()

    desc_stats = df.groupby('category')[['engagement', 'polarity', 'flesch_kincaid_grade']].agg(['mean', 'median', 'std']).T
    print("\n--- 1. Descriptive Statistics by Category ---")
    print(desc_stats)

    u_stat, p_value = mannwhitneyu(political, entertainment, alternative='two-sided')

    def cliffs_delta(x, y):
        n_x = len(x)
        n_y = len(y)
        if n_x == 0 or n_y == 0: return np.nan
        num_greater = sum(1 for x_i in x for y_j in y if x_i > y_j)
        num_less = sum(1 for x_i in x for y_j in y if x_i < y_j)
        delta = (num_greater - num_less) / (n_x * n_y)
        return delta

    delta = cliffs_delta(political, entertainment)
with open('results_A_hypothesis_test.txt', 'w') as f:
    f.write("--- stats by category ---\n")
    f.write(desc_stats.to_string() + "\n\n")
    f.write("---  Mann-Whitney U Test Results ---\n")
    f.write(f"Median Engagement (Political):    {political.median():.2f}\n")
    f.write(f"Median Engagement (Entertainment):{entertainment.median():.2f}\n")
    f.write(f"P-value:     {p_value:.5f}\n")
    f.write(f"Cliff's Delta (Effect Size): {delta:.3f}\n")

print("saved")
