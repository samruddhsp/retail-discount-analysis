
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
from scipy.stats import kruskal

# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Retail Intelligence | ASSP",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"

# =========================================================
# CUSTOM THEME
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #100F10;
    color: #F2EAE6;
}

[data-testid="stSidebar"] {
    background: #0A090B;
    border-right: 1px solid #30252E;
}

[data-testid="stSidebar"] h1 {
    color: #E8C2AD;
    font-size: 22px;
}

[data-testid="stSidebar"] p {
    color: #B5A6B0;
}

.block-container {
    padding-top: 2.2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

h1, h2, h3 {
    color: #F5EDE9 !important;
    font-weight: 700 !important;
}

p, label {
    color: #BBAFB9;
}

[data-testid="stMetric"] {
    background: linear-gradient(135deg, #241B27, #19151B);
    border: 1px solid #49334F;
    padding: 20px 20px;
    border-radius: 14px;
}

[data-testid="stMetricLabel"] {
    color: #BBAABD !important;
}

[data-testid="stMetricValue"] {
    color: #F4E8F8 !important;
    font-weight: 700;
}

[data-testid="stMetricDelta"] {
    color: #C5A1DC !important;
}

div[data-testid="stPlotlyChart"] {
    background: #171417;
    border: 1px solid #332832;
    border-radius: 14px;
    padding: 10px;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #3C303D;
    border-radius: 12px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 18px;
    background: transparent;
}

.stTabs [data-baseweb="tab"] {
    color: #BBAFB9;
}

.stTabs [aria-selected="true"] {
    color: #D7A9F0 !important;
    border-bottom-color: #B66CE0 !important;
}

div.stButton > button {
    border-radius: 9px;
    border: 1px solid #805D91;
    background: #281D30;
    color: #EBD8F5;
}

div.stButton > button:hover {
    border-color: #C18AE0;
    color: white;
}

hr {
    border-color: #342B35;
}

.hero {
    background: linear-gradient(115deg, #241A29, #171317 65%, #30221E);
    border: 1px solid #503957;
    border-radius: 18px;
    padding: 27px 30px;
    margin-bottom: 25px;
}

.hero-label {
    color: #C99FE0;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
}

.hero-title {
    font-size: 31px;
    font-weight: 800;
    color: #F6EAF9;
    margin-top: 8px;
}

.hero-sub {
    color: #B9AAB8;
    font-size: 13px;
    margin-top: 8px;
}

.section-label {
    color: #D6A68C;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.5px;
    margin-bottom: 8px;
}

.insight {
    background: #1C171D;
    border: 1px solid #443248;
    border-left: 4px solid #A86BC4;
    border-radius: 10px;
    padding: 16px 18px;
    margin: 8px 0;
    color: #E9DCEB;
}

.small-note {
    color: #9E929E;
    font-size: 12px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    df = pd.read_csv(OUT / "cleaned_retail_data.csv")
    models = pd.read_csv(OUT / "model_comparison.csv")
    regression = pd.read_csv(
        OUT / "regression_results.csv",
        index_col=0
    )

    df["Purchase_Date"] = pd.to_datetime(df["Purchase_Date"])

    return df, models, regression


try:
    df, models, regression = load_data()
except Exception as e:
    st.error(f"Could not load project files: {e}")
    st.info("Check that the required CSV files exist inside the outputs folder.")
    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("# ◈ RETAIL INTELLIGENCE")
    st.caption("ASSP | STATISTICAL ANALYTICS")

    st.divider()

    page = st.radio(
        "WORKSPACE",
        [
            "Overview",
            "Exploratory Analysis",
            "Discount Analysis",
            "Statistical Tests",
            "Model Performance",
            "Conclusion"
        ],
        label_visibility="visible"
    )

    st.divider()

    st.markdown("### DATA FILTERS")

    min_date = df["Purchase_Date"].min().date()
    max_date = df["Purchase_Date"].max().date()

    date_range = st.date_input(
        "Purchase period",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    categories = sorted(df["Category"].dropna().unique())
    selected_categories = st.multiselect(
        "Product category",
        categories,
        default=categories
    )

    channels = sorted(df["Channel"].dropna().unique())
    selected_channels = st.multiselect(
        "Sales channel",
        channels,
        default=channels
    )

    st.divider()

    st.caption("RETAIL DISCOUNT RESEARCH")
    st.caption("Dataset period: 2023–2024")


# =========================================================
# FILTER DATA
# =========================================================

filtered = df.copy()

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range

    filtered = filtered[
        filtered["Purchase_Date"].dt.date.between(
            start_date, end_date
        )
    ]

if selected_categories:
    filtered = filtered[
        filtered["Category"].isin(selected_categories)
    ]
else:
    filtered = filtered.iloc[0:0]

if selected_channels:
    filtered = filtered[
        filtered["Channel"].isin(selected_channels)
    ]
else:
    filtered = filtered.iloc[0:0]


# =========================================================
# CHART THEME
# =========================================================

def chart_theme(fig, title=None):

    fig.update_layout(
        title=title,
        paper_bgcolor="#171417",
        plot_bgcolor="#171417",
        font=dict(
            family="DM Sans",
            color="#D9CBD8",
            size=12
        ),
        title_font=dict(
            size=16,
            color="#F0E5EF"
        ),
        margin=dict(l=25, r=25, t=55, b=25),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#CDBECD")
        ),
        hoverlabel=dict(
            bgcolor="#241C27",
            font_color="#F5EAF5"
        )
    )

    fig.update_xaxes(
        gridcolor="#332B35",
        zerolinecolor="#493B4B"
    )

    fig.update_yaxes(
        gridcolor="#332B35",
        zerolinecolor="#493B4B"
    )

    return fig


def heading(title, subtitle=None):
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)


# =========================================================
# HERO
# =========================================================

st.markdown("""
<div class="hero">
    <div class="hero-label">RETAIL ANALYTICS / ASSP PROJECT</div>
    <div class="hero-title">Do Bigger Discounts Mean Better Deals?</div>
    <div class="hero-sub">
        Understanding the relationship between discounts,
        customer behaviour and purchase spending.
    </div>
</div>
""", unsafe_allow_html=True)

if filtered.empty:
    st.warning("No transactions match the selected filters.")
    st.stop()


# =========================================================
# 1. OVERVIEW
# =========================================================

if page == "Overview":

    heading(
        "Business Overview",
        "A snapshot of retail transaction performance."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "TRANSACTIONS",
        f"{len(filtered):,}"
    )

    c2.metric(
        "CUSTOMERS",
        f"{filtered['Customer_ID'].nunique():,}"
    )

    c3.metric(
        "AVG. SPENDING",
        f"₹{filtered['Purchase_Amount'].mean():,.0f}"
    )

    c4.metric(
        "AVG. DISCOUNT",
        f"{filtered['Discount'].mean():.1f}%"
    )

    st.divider()

    left, right = st.columns([1.7, 1])

    with left:

        heading("Transaction Activity", "Monthly purchase volume")

        monthly = (
            filtered.groupby(
                filtered["Purchase_Date"].dt.to_period("M")
            )
            .size()
            .reset_index(name="Transactions")
        )

        monthly["Month"] = monthly.iloc[:, 0].astype(str)

        fig = px.area(
            monthly,
            x="Month",
            y="Transactions",
            markers=True
        )

        fig.update_traces(
            line_color="#B88ACD",
            fillcolor="rgba(184,138,205,0.13)"
        )

        st.plotly_chart(
            chart_theme(fig),
            use_container_width=True
        )

    with right:

        heading("Sales Channels")

        channel_data = filtered["Channel"].value_counts().reset_index()
        channel_data.columns = ["Channel", "Transactions"]

        fig = px.pie(
            channel_data,
            names="Channel",
            values="Transactions",
            hole=0.68,
            color_discrete_sequence=["#B88ACD", "#C49A80"]
        )

        st.plotly_chart(
            chart_theme(fig),
            use_container_width=True
        )

    st.divider()

    heading("Recent Transactions")

    st.dataframe(
        filtered[
            [
                "Transaction_ID",
                "Purchase_Date",
                "Category",
                "Discount",
                "Quantity",
                "Purchase_Amount",
                "Channel"
            ]
        ].head(15),
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# 2. EXPLORATORY ANALYSIS
# =========================================================

elif page == "Exploratory Analysis":

    heading(
        "Exploratory Data Analysis",
        "Explore customer behaviour, spending and product patterns."
    )

    c1, c2 = st.columns(2)

    with c1:

        category_data = (
            filtered.groupby("Category")["Purchase_Amount"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            category_data,
            x="Category",
            y="Purchase_Amount",
            color="Category",
            color_discrete_sequence=[
                "#B88ACD", "#C49A80", "#9273A8"
            ]
        )

        st.plotly_chart(
            chart_theme(fig, "Average Spending by Category"),
            use_container_width=True
        )

    with c2:

        fig = px.histogram(
            filtered,
            x="Purchase_Amount",
            nbins=45,
            color_discrete_sequence=["#B88ACD"]
        )

        st.plotly_chart(
            chart_theme(fig, "Purchase Amount Distribution"),
            use_container_width=True
        )

    c1, c2 = st.columns(2)

    with c1:

        fig = px.box(
            filtered,
            x="Channel",
            y="Purchase_Amount",
            color="Channel",
            color_discrete_sequence=["#B88ACD", "#C49A80"]
        )

        st.plotly_chart(
            chart_theme(fig, "Online vs Offline Spending"),
            use_container_width=True
        )

    with c2:

        gender_data = filtered["Gender"].value_counts().reset_index()
        gender_data.columns = ["Gender", "Transactions"]

        fig = px.pie(
            gender_data,
            names="Gender",
            values="Transactions",
            hole=0.6,
            color_discrete_sequence=[
                "#B88ACD", "#C49A80", "#9273A8"
            ]
        )

        st.plotly_chart(
            chart_theme(fig, "Transaction Distribution by Gender"),
            use_container_width=True
        )


# =========================================================
# 3. DISCOUNT ANALYSIS
# =========================================================

elif page == "Discount Analysis":

    heading(
        "Discount Intelligence",
        "Investigating how discount levels relate to purchase behaviour."
    )

    c1, c2 = st.columns(2)

    with c1:

        fig = px.scatter(
            filtered.sample(
                min(2500, len(filtered)),
                random_state=42
            ),
            x="Discount",
            y="Purchase_Amount",
            color="Category",
            opacity=0.65,
            color_discrete_sequence=[
                "#B88ACD", "#C49A80", "#9273A8"
            ]
        )

        st.plotly_chart(
            chart_theme(fig, "Discount vs Purchase Amount"),
            use_container_width=True
        )

    with c2:

        fig = px.box(
            filtered,
            x="Discount_Band",
            y="Purchase_Amount",
            color="Discount_Band",
            color_discrete_sequence=[
                "#C49A80", "#B88ACD", "#9273A8"
            ]
        )

        st.plotly_chart(
            chart_theme(fig, "Spending Across Discount Bands"),
            use_container_width=True
        )

    comparison = (
        filtered.groupby("Discount_Band", observed=True)
        .agg(
            Transactions=("Transaction_ID", "count"),
            Average_Spending=("Purchase_Amount", "mean"),
            Median_Spending=("Purchase_Amount", "median"),
            Average_Quantity=("Quantity", "mean"),
            Median_Discount=("Discount", "median")
        )
        .reset_index()
    )

    st.subheader("Discount Band Comparison")

    st.dataframe(
        comparison.style.format({
            "Average_Spending": "₹{:,.2f}",
            "Median_Spending": "₹{:,.2f}",
            "Average_Quantity": "{:.2f}",
            "Median_Discount": "{:.1f}%"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("""
    <div class="insight">
    <b>Research observation</b><br>
    Higher-discount transactions show higher average spending
    descriptively. This does not establish that discounts cause
    customers to spend more.
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# 4. STATISTICAL TESTS
# =========================================================

elif page == "Statistical Tests":

    heading(
        "Statistical Evidence",
        "Regression estimates, hypothesis tests and model diagnostics."
    )

    st.subheader("Multiple Linear Regression")

    st.caption(
        "Regression results below are from the full dataset, "
        "not recalculated using the sidebar filters."
    )

    st.dataframe(
        regression.style.format({
            "Coefficient": "{:,.3f}",
            "P_Value": "{:.4f}",
            "CI_Lower": "{:,.3f}",
            "CI_Upper": "{:,.3f}"
        }),
        use_container_width=True
    )

    discount = regression.loc["Discount"]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Discount Coefficient",
        f"₹{discount['Coefficient']:.2f}"
    )

    c2.metric(
        "P-value",
        f"{discount['P_Value']:.3f}"
    )

    c3.metric(
        "95% CI",
        f"₹{discount['CI_Lower']:.2f} to ₹{discount['CI_Upper']:.2f}"
    )

    st.divider()

    st.subheader("Kruskal–Wallis Tests")

    spending_groups = [
        group["Purchase_Amount"].values
        for _, group in df.groupby(
            "Discount_Band", observed=True
        )
    ]

    quantity_groups = [
        group["Quantity"].values
        for _, group in df.groupby(
            "Discount_Band", observed=True
        )
    ]

    spending_test = kruskal(*spending_groups)
    quantity_test = kruskal(*quantity_groups)

    c1, c2 = st.columns(2)

    c1.metric(
        "Spending p-value",
        f"{spending_test.pvalue:.4f}"
    )

    c2.metric(
        "Quantity p-value",
        f"{quantity_test.pvalue:.4f}"
    )

    st.markdown("""
    <div class="insight">
    <b>Interpretation</b><br>
    Both tests have p-values above 0.05. We do not find
    statistically significant evidence of differences in
    spending or quantity across discount bands.
    </div>
    """, unsafe_allow_html=True)

    st.warning(
        "Kruskal–Wallis assumes independent observations. "
        "Repeated customers mean these results should be treated "
        "as exploratory rather than definitive."
    )

    st.divider()

    st.subheader("Regression Diagnostics")

    d1, d2, d3 = st.columns(3)

    d1.metric("Jarque–Bera p-value", "0.0000")
    d2.metric("Breusch–Pagan p-value", "<0.001")
    d3.metric("Maximum VIF", "2.064")

    st.caption(
        "Residuals depart from normality and show heteroscedasticity. "
        "VIF values indicate no serious multicollinearity."
    )


# =========================================================
# 5. MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    heading(
        "Predictive Model Performance",
        "Comparing three regression algorithms using customer-grouped cross-validation."
    )

    best_r2 = models.loc[models["Mean R2"].idxmax()]
    best_mae = models.loc[models["Mean MAE"].idxmin()]

    c1, c2 = st.columns(2)

    c1.metric(
        "Highest R²",
        best_r2["Model"],
        f"{best_r2['Mean R2']:.3f}"
    )

    c2.metric(
        "Lowest MAE",
        best_mae["Model"],
        f"₹{best_mae['Mean MAE']:,.0f}"
    )

    st.divider()

    c1, c2 = st.columns(2)

    with c1:

        fig = px.bar(
            models,
            x="Model",
            y="Mean R2",
            color="Model",
            color_discrete_sequence=[
                "#B88ACD", "#C49A80", "#9273A8"
            ]
        )

        st.plotly_chart(
            chart_theme(fig, "Model R² Comparison"),
            use_container_width=True
        )

    with c2:

        fig = px.bar(
            models,
            x="Model",
            y="Mean MAE",
            color="Model",
            color_discrete_sequence=[
                "#B88ACD", "#C49A80", "#9273A8"
            ]
        )

        st.plotly_chart(
            chart_theme(fig, "Mean Absolute Error"),
            use_container_width=True
        )

    st.subheader("Detailed Model Results")

    st.dataframe(
        models.style.format({
            "Mean R2": "{:.4f}",
            "Mean MAE": "₹{:,.2f}",
            "Mean RMSE": "₹{:,.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "The three models perform similarly. Predictive accuracy "
        "does not demonstrate a causal relationship between discounts "
        "and purchase spending."
    )


# =========================================================
# 6. CONCLUSION
# =========================================================

elif page == "Conclusion":

    heading(
        "Research Findings & Conclusion",
        "What does the retail dataset tell us about discounts?"
    )

    st.markdown("""
    <div class="insight">
    <b>Central research question</b><br>
    Do bigger discounts mean customers spend more?
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Key Findings")

    findings = [
        (
            "Discount and spending",
            "The regression coefficient is positive, but its p-value "
            "is 0.163. The evidence is insufficient to establish an "
            "independent discount–spending association."
        ),
        (
            "Discount bands",
            "Kruskal–Wallis tests did not detect significant differences "
            "in spending or quantity across the three discount bands."
        ),
        (
            "Product category",
            "Product category has statistically significant associations "
            "with purchase amount in the multiple regression."
        ),
        (
            "Predictive performance",
            "Linear Regression and Random Forest performed similarly "
            "under customer-grouped cross-validation."
        ),
        (
            "Regression assumptions",
            "Residual normality and constant variance assumptions were "
            "not supported by the diagnostic tests."
        )
    ]

    for title, description in findings:
        st.markdown(
            f"""
            <div class="insight">
            <b>{title}</b><br>
            {description}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.subheader("Limitations")

    st.markdown("""
    - The dataset contains completed transactions only.
    - Repeated transactions belong to the same customers.
    - Observational associations cannot establish causation.
    - Regression residuals show non-normality and heteroscedasticity.
    - Kruskal–Wallis tests do not account for repeated customers.
    """)

    st.subheader("Overall Conclusion")

    st.markdown("""
    The analysis does not provide sufficient statistical evidence
    that larger discounts independently lead to higher purchase
    spending. Product category and quantity show stronger associations
    with purchase amount in the fitted regression model.

    Further research using customer-level repeated-measures methods
    and data on customers who did not purchase would provide a more
    complete understanding of discount effectiveness.
    """)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#887B88; font-size:11px;">
    RETAIL INTELLIGENCE STUDIO · ASSP STATISTICAL PROJECT<br>
    DATA PERIOD 2023–2024 · BUILT WITH STREAMLIT & PYTHON
    </div>
    """,
    unsafe_allow_html=True
)
