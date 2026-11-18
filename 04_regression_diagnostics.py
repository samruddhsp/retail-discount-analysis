import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tools.tools import add_constant
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

df = pd.read_csv(
    PROJECT_DIR / "outputs" / "cleaned_retail_data.csv"
)

# Fit the same regression model
model = smf.ols(
    """
    Purchase_Amount ~ Discount + Quantity + Age
    + Previous_Purchases
    + C(Category) + C(Channel) + C(Gender)
    """,
    data=df
).fit()

# 1. Residual normality
jb = jarque_bera(model.resid)

print("\n========== NORMALITY TEST ==========")
print("Jarque-Bera statistic:", jb[0])
print("P-value:", jb[1])

# 2. Heteroscedasticity
bp = het_breuschpagan(model.resid, model.model.exog)

print("\n========== BREUSCH-PAGAN TEST ==========")
print("LM statistic:", bp[0])
print("P-value:", bp[1])

# 3. Multicollinearity
X = add_constant(model.model.exog)
vif = pd.DataFrame({
    "Variable": model.model.exog_names,
    "VIF": [
        variance_inflation_factor(X, i)
        for i in range(X.shape[1])
    ]
})

print("\n========== VIF ==========")
print(vif)

# 4. Residual plots
plt.figure(figsize=(8, 5))
plt.scatter(model.fittedvalues, model.resid, alpha=0.3)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted Purchase Amount")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Values")
plt.tight_layout()

plt.savefig(
    PROJECT_DIR / "outputs" / "residual_plot.png"
)
plt.close()

print("\nDiagnostics completed!")