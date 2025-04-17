###### Customer Lifetime Value Prediction ######

### CLTV = (Customer Value / Churn rate) * Profit Margin
### Customer Value = Purchase Frequency * Average Order Value
### CLTV = Expected Number of Transactions * Expected Average Profit
### CLTV = BG/NBD Model * Gamma-Gamma Submodel

### BG/NBD (Beta Geometric / Negative Binomial Distribution) is used to estimate:
# - Transaction Rate (follows Gamma distribution)
# - Dropout Rate (follows Beta distribution)

### Step-by-step CLTV Calculation Workflow:

# 1. Data Preparation

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: "%.4f" % x)

# Reading data
df_ = pd.read_excel("/Users/erdinc/PycharmProjects/pythonProject4/RMF/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()

# Data cleaning
df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]  # Remove returns
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

# Calculate total price
df["TotalPrice"] = df["Quantity"] * df["Price"]

# Set analysis date
today_date = dt.datetime(2011, 12, 11)

# 2. Prepare the structure for Lifetime Value modeling
# recency: Time between first and last purchase (in weeks)
# T: Customer's age (time since first purchase, in weeks)
# frequency: Total number of repeat purchases (must be >1)
# monetary: Average revenue per purchase

cltv_df = df.groupby("Customer ID").agg({
    "InvoiceDate": [lambda date: (date.max() - date.min()).days,
                    lambda date: (today_date - date.min()).days],
    "Invoice": lambda num: num.nunique(),
    "TotalPrice": lambda price: price.sum()
})

cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ["recency", "T", "frequency", "monetary"]
cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
cltv_df = cltv_df[cltv_df["frequency"] > 1]
cltv_df["recency"] = cltv_df["recency"] / 7
cltv_df["T"] = cltv_df["T"] / 7

# 3. BG/NBD Model – Predict Expected Number of Transactions
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df["frequency"], cltv_df["recency"], cltv_df["T"])

# Predict transactions for the next 1 week
cltv_df["expected_purc_1_week"] = bgf.predict(
    1, cltv_df["frequency"], cltv_df["recency"], cltv_df["T"]
)

# Predict transactions for the next 1 month (4 weeks)
cltv_df["expected_purc_1_month"] = bgf.predict(
    4, cltv_df["frequency"], cltv_df["recency"], cltv_df["T"]
)

# Total expected transactions in 1 month
total_expected_1_month = cltv_df["expected_purc_1_month"].sum()

# Plotting actual vs predicted
plot_period_transactions(bgf)
plt.show()

# 4. Gamma-Gamma Model – Predict Expected Average Profit
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df["frequency"], cltv_df["monetary"])

# Calculate expected average profit
cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(
    cltv_df["frequency"], cltv_df["monetary"]
)

# Top 10 customers by expected average profit
top_customers = cltv_df.sort_values("expected_average_profit", ascending=False).head(10)

# 5. (Optional) Calculate Full CLTV and Segment Customers
# CLTV = Expected Transactions * Expected Profit

# 6. (Optional) Normalize CLTV and create customer segments using quantiles or custom rules
