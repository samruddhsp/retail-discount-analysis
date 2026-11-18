import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

df = pd.read_csv(OUTPUT_DIR / "cleaned_retail_data.csv")

sns.set_theme(style="whitegrid")

# 1. Discount vs Purchase Amount
plt.figure(figsize=(8, 5))
sns.regplot(
    data=df, x="Discount", y="Purchase_Amount",
    scatter_kws={"alpha": 0.2},
    line_kws={"color": "red"}
)
plt.title("Discount vs Purchase Amount")
plt.savefig(OUTPUT_DIR / "01_discount_vs_spending.png")
plt.close()

# 2. Spending by Discount Band
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Discount_Band", y="Purchase_Amount")
plt.title("Purchase Amount Across Discount Bands")
plt.savefig(OUTPUT_DIR / "02_discount_band_boxplot.png")
plt.close()

# 3. Average Spending by Category
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="Category", y="Purchase_Amount", errorbar=None)
plt.title("Average Spending by Product Category")
plt.savefig(OUTPUT_DIR / "03_category_spending.png")
plt.close()

# 4. Model comparison
models = pd.read_csv(OUTPUT_DIR / "model_comparison.csv")

plt.figure(figsize=(8, 5))
sns.barplot(data=models, x="Model", y="Mean R2")
plt.title("Predictive Model Comparison (R²)")
plt.ylim(0, 1)
plt.savefig(OUTPUT_DIR / "04_model_comparison.png")
plt.close()

# 5. Monthly transaction trend
df["Purchase_Date"] = pd.to_datetime(df["Purchase_Date"])
monthly = df.groupby(
    df["Purchase_Date"].dt.to_period("M")
).size()

plt.figure(figsize=(10, 5))
monthly.index = monthly.index.astype(str)
monthly.plot(marker="o")
plt.title("Monthly Transaction Trend")
plt.ylabel("Number of Transactions")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_monthly_trend.png")
plt.close()

print("All five visualisations saved in outputs!")