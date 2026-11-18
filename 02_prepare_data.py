import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "retail_dataset.xlsx"
OUTPUT_DIR = PROJECT_DIR / "outputs"

df = pd.read_excel(DATA_PATH)

# Clean column names
df.rename(columns={
    "Purchase Amount (â‚¹)": "Purchase_Amount",
    "Shipping Charge (â‚¹)": "Shipping_Charge",
    "Transaction ID": "Transaction_ID",
    "Customer ID": "Customer_ID",
    "Purchase Date": "Purchase_Date",
    "Discount (%)": "Discount",
    "Online/Offline": "Channel",
    "Previous Purchases": "Previous_Purchases",
}, inplace=True)

# Create discount categories
df["Discount_Band"] = pd.cut(
    df["Discount"],
    bins=[-1, 5, 15, float("inf")],
    labels=["Low", "Medium", "High"]
)

# Extract time-based features
df["Year"] = df["Purchase_Date"].dt.year
df["Month"] = df["Purchase_Date"].dt.month
df["Year_Month"] = df["Purchase_Date"].dt.to_period("M").astype(str)

# Calculate total transaction value including shipping
df["Total_Value"] = df["Purchase_Amount"] + df["Shipping_Charge"]

# Save prepared dataset
OUTPUT_DIR.mkdir(exist_ok=True)
df.to_csv(OUTPUT_DIR / "cleaned_retail_data.csv", index=False)

print("========== DATA PREPARATION ==========")
print("\nDiscount distribution:")
print(df["Discount_Band"].value_counts().sort_index())

print("\nDiscount range:")
print(df["Discount"].describe())

print("\nPurchase amount range:")
print(df["Purchase_Amount"].describe())

print("\nTransaction ID duplicates:", df["Transaction_ID"].duplicated().sum())
print("Unique customers:", df["Customer_ID"].nunique())

print("\nChannel distribution:")
print(df["Channel"].value_counts())

print("\nPrepared dataset saved successfully!")