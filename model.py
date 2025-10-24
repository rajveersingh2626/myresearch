import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import glm
import numpy as np

df = pd.read_csv('analysis_data_fully_processed.csv')
df['is_political'] = (df['category'] == 'Political').astype(int)

print("\nnegative binomial regression")
formula_nb = 'engagement ~ is_political + polarity + subjectivity + flesch_kincaid_grade'

nb_model = glm(formula_nb, data=df, family=sm.families.NegativeBinomial()).fit()

with open('results_B_regression_summary.txt', 'w') as f:
    f.write("---Negative Binomial Regression Summary ---\n")
    f.write(nb_model.summary().as_text())

print("saved")
