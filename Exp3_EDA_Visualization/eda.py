"""
Experiment 3: Exploratory data analysis and visualisation of the
Rental Car Management System dataset (cleaned output of Experiment 2).

Figures are written to the figures/ folder.
"""
import calendar
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

pd.set_option("display.width", 110)
pd.set_option("display.max_columns", 12)
os.makedirs("figures", exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")

df = pd.read_csv("rental_car_clean.csv", parse_dates=["rental_date", "return_date"])

# Step 1: first look at the data
print("Shape:", df.shape)
print("\nColumns:", list(df.columns))
print("\nMissing values in the whole table:", int(df.drop(columns="addons").isnull().sum().sum()))
print("Date range:", df["rental_date"].min().date(), "to", df["rental_date"].max().date())

num_cols = ["customer_age", "rental_days", "km_driven", "daily_rate",
            "rental_amount", "late_fee", "customer_rating"]
cat_cols = ["car_type", "fuel_type", "booking_channel", "trip_type", "branch_city", "brand"]

# Step 2: measures of central tendency
central = pd.DataFrame({"Mean": df[num_cols].mean(), "Median": df[num_cols].median()}).round(2)
print("\n=== Central tendency (numeric) ===")
print(central.to_string())
print("\n=== Mode (categorical) ===")
for col in cat_cols:
    top = df[col].value_counts()
    print(f"{col:16s} {top.index[0]:12s} ({top.iloc[0]} rentals, {top.iloc[0] / len(df):.1%})")

# Step 3: measures of dispersion
q1, q3 = df[num_cols].quantile(0.25), df[num_cols].quantile(0.75)
dispersion = pd.DataFrame({
    "Std Dev": df[num_cols].std(),
    "Variance": df[num_cols].var(),
    "Min": df[num_cols].min(),
    "Max": df[num_cols].max(),
    "Range": df[num_cols].max() - df[num_cols].min(),
    "IQR": q3 - q1,
}).round(2)
print("\n=== Dispersion ===")
print(dispersion.to_string(float_format=lambda v: f"{v:,.2f}"))

iqr = q3 - q1
outliers = ((df[num_cols] < q1 - 1.5 * iqr) | (df[num_cols] > q3 + 1.5 * iqr)).sum()
print("\nValues outside 1.5 x IQR:")
print(outliers.to_string())

# Step 4: correlation between numeric attributes
corr = df[num_cols + ["km_per_day", "addon_count"]].corr()
print("\n=== Pearson correlation ===")
print(corr.round(2).to_string())

# Step 5: grouped summaries that matter for the rental business
print("\n=== Revenue and rate by car type ===")
by_type = df.groupby("car_type").agg(rentals=("rental_id", "count"),
                                     avg_daily_rate=("daily_rate", "mean"),
                                     avg_days=("rental_days", "mean"),
                                     revenue_lakh=("rental_amount", lambda s: s.sum() / 1e5))
print(by_type.round(1).sort_values("revenue_lakh", ascending=False).to_string())

print("\n=== Late return rate ===")
for col in ["trip_type", "booking_channel", "return_weekend"]:
    rate = df.groupby(col)["late_return"].apply(lambda s: (s == "Yes").mean() * 100).round(1)
    print(f"{col}: " + ", ".join(f"{k} {v}%" for k, v in rate.items()))

# Step 6: visualisations
# 6a. Histogram of rental amount
plt.figure(figsize=(9, 4.5))
sns.histplot(df["rental_amount"], bins=40, kde=True, color="#4c72b0")
plt.title("Distribution of Rental Amount")
plt.xlabel("Rental amount (INR)")
plt.ylabel("Number of rentals")
plt.tight_layout()
plt.savefig("figures/fig1_rental_amount_hist.png", dpi=150)
plt.close()

# 6b. Boxplot of daily rate by car type
order = ["Hatchback", "Sedan", "MUV", "SUV", "Luxury"]
plt.figure(figsize=(9, 4.5))
sns.boxplot(data=df, x="car_type", y="daily_rate", order=order, hue="car_type",
            hue_order=order, legend=False)
plt.title("Daily Rate by Car Type")
plt.xlabel("Car type")
plt.ylabel("Daily rate (INR)")
plt.tight_layout()
plt.savefig("figures/fig2_daily_rate_boxplot.png", dpi=150)
plt.close()

# 6c. Correlation heatmap
plt.figure(figsize=(8, 6.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True)
plt.title("Correlation Heatmap of Numeric Attributes")
plt.tight_layout()
plt.savefig("figures/fig3_correlation_heatmap.png", dpi=150)
plt.close()

# 6d. Scatter plot: distance against rental length
plt.figure(figsize=(9, 5))
sns.scatterplot(data=df, x="rental_days", y="km_driven", hue="trip_type", alpha=0.5, s=18)
plt.title("Kilometres Driven vs Rental Days")
plt.xlabel("Rental days")
plt.ylabel("Kilometres driven")
plt.tight_layout()
plt.savefig("figures/fig4_km_vs_days_scatter.png", dpi=150)
plt.close()

# 6e. Monthly rentals for both years
monthly = df.groupby([df["rental_date"].dt.year.rename("year"),
                      df["rental_date"].dt.month.rename("month")]).size().unstack(0)
monthly.index = [calendar.month_abbr[m] for m in monthly.index]
plt.figure(figsize=(9, 4.5))
for year in monthly.columns:
    plt.plot(monthly.index, monthly[year], marker="o", label=str(year))
plt.title("Rentals per Month")
plt.xlabel("Month")
plt.ylabel("Number of rentals")
plt.legend(title="Year")
plt.tight_layout()
plt.savefig("figures/fig5_monthly_rentals.png", dpi=150)
plt.close()

# 6f. Late return rate by trip type and booking channel
late = (df.assign(late=(df["late_return"] == "Yes") * 100)
          .groupby(["booking_channel", "trip_type"])["late"].mean().unstack())
late.plot(kind="bar", figsize=(9, 4.5), rot=0)
plt.title("Late Return Rate by Booking Channel and Trip Type")
plt.xlabel("Booking channel")
plt.ylabel("Late returns (%)")
plt.legend(title="Trip type")
plt.tight_layout()
plt.savefig("figures/fig6_late_return_rate.png", dpi=150)
plt.close()

# 6g. Revenue by branch city
rev = df.groupby("branch_city")["rental_amount"].sum().sort_values() / 1e5
plt.figure(figsize=(9, 4.5))
rev.plot(kind="barh", color="#55a868")
plt.title("Total Revenue by Branch City (2024 to 2025)")
plt.xlabel("Revenue (INR lakh)")
plt.ylabel("")
plt.tight_layout()
plt.savefig("figures/fig7_revenue_by_branch.png", dpi=150)
plt.close()

print("\nSaved 7 figures in figures/")
