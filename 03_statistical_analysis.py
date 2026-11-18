import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import kruskal
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "outputs" / "cleaned_retail_data.csv"

df = pd.read_csv(DATA_PATH)

# =====================================
# 1. MULTIPLE LINEAR REGRESSION
# =====================================

model = smf.ols(
    formula="""
    Purchase_Amount ~ Discount + Quantity + Age
    + Previous_Purchases
    + C(Category) + C(Channel) + C(Gender)
    """,
    data=df
).fit(
    cov_type="cluster",
    cov_kwds={"groups": df["Customer_ID"]}
)

print("\n========== MULTIPLE LINEAR REGRESSION ==========")
print(model.summary())

# Save coefficient estimates and confidence intervals
results = pd.DataFrame({
    "Coefficient": model.params,
    "P_Value": model.pvalues,
    "CI_Lower": model.conf_int()[0],
    "CI_Upper": model.conf_int()[1]
})

results.to_csv(
    PROJECT_DIR / "outputs" / "regression_results.csv"
)

# =====================================
# 2. KRUSKAL-WALLIS TEST
# =====================================

print("\n========== KRUSKAL-WALLIS TEST ==========")

groups_spending = [
    group["Purchase_Amount"].values
    for _, group in df.groupby("Discount_Band", observed=True)
]

groups_quantity = [
    group["Quantity"].values
    for _, group in df.groupby("Discount_Band", observed=True)
]

spending_test = kruskal(*groups_spending)
quantity_test = kruskal(*groups_quantity)

print("\nSpending across discount bands:")
print("H-statistic:", spending_test.statistic)
print("P-value:", spending_test.pvalue)

print("\nQuantity across discount bands:")
print("H-statistic:", quantity_test.statistic)
print("P-value:", quantity_test.pvalue)

# =====================================
# 3. GROUP-WISE DESCRIPTIVE STATISTICS
# =====================================

print("\n========== DISCOUNT BAND COMPARISON ==========")

comparison = df.groupby(
    "Discount_Band", observed=True
).agg(
    Transactions=("Transaction_ID", "count"),
    Mean_Spending=("Purchase_Amount", "mean"),
    Median_Spending=("Purchase_Amount", "median"),
    Mean_Quantity=("Quantity", "mean"),
    Median_Discount=("Discount", "median")
)

print(comparison)

comparison.to_csv(
    PROJECT_DIR / "outputs" / "discount_band_comparison.csv"
)

print("\nStatistical analysis completed!")