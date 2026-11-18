import pandas as pd
from pathlib import Path

# Locate the project and dataset
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "retail_dataset.xlsx"

# Load Excel dataset
df = pd.read_excel(DATA_PATH)

print("\n========== DATASET OVERVIEW ==========")

# 1. Dataset dimensions
print("\nRows and columns:", df.shape)

# 2. Column names
print("\nColumn names:")
print(df.columns.tolist())

# 3. Data types
print("\nData types:")
print(df.dtypes)

# 4. Missing values
print("\nMissing values:")
print(df.isnull().sum())

# 5. Duplicate records
print("\nDuplicate rows:", df.duplicated().sum())

# 6. First five records
print("\nFirst five records:")
print(df.head())

# 7. Numerical summary
print("\nNumerical summary:")
print(df.describe())

# 8. Categorical summary
print("\nCategorical summary:")
print(df.describe(include="object"))

# Save missing-value report
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

df.isnull().sum().to_csv(
    OUTPUT_DIR / "missing_values.csv",
    header=["Missing Count"]
)

print("\nInspection completed successfully!")