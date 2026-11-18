import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

PROJECT_DIR = Path(__file__).resolve().parent.parent

df = pd.read_csv(
    PROJECT_DIR / "outputs" / "cleaned_retail_data.csv"
)

# Target and predictors
target = "Purchase_Amount"

features = [
    "Discount",
    "Quantity",
    "Age",
    "Previous_Purchases",
    "Category",
    "Channel",
    "Gender"
]

X = df[features]
y = df[target]
groups = df["Customer_ID"]

numeric_features = [
    "Discount", "Quantity", "Age", "Previous_Purchases"
]

categorical_features = ["Category", "Channel", "Gender"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=10,
        min_samples_leaf=10,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=150,
        max_depth=15,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42
    )
}

cv = GroupKFold(n_splits=5)

results = []

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        groups=groups,
        scoring={
            "r2": "r2",
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error"
        },
        n_jobs=-1
    )

    results.append({
        "Model": name,
        "Mean R2": scores["test_r2"].mean(),
        "Mean MAE": -scores["test_mae"].mean(),
        "Mean RMSE": -scores["test_rmse"].mean()
    })

    print(f"\nCompleted: {name}")

results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

results_df.to_csv(
    PROJECT_DIR / "outputs" / "model_comparison.csv",
    index=False
)

print("\nMachine learning analysis completed!")