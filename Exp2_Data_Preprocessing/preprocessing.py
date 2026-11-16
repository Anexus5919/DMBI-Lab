"""
Experiment 2: Data preprocessing for the Rental Car Management System.

Input : rental_car_raw.csv   (5,045 rental records with quality problems)
Output: rental_car_clean.csv        cleaned, readable data used in Exp 3 to 10
        rental_car_model_ready.csv  encoded and scaled copy for ML models
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

pd.set_option("display.width", 100)
pd.set_option("display.max_columns", 12)

# 1. Load the dataset
df = pd.read_csv("rental_car_raw.csv")
print("Original shape:", df.shape)
print(df.head(3).T.to_string())

# 2. Check missing values and duplicates
print("\nMissing values per column:")
print(df.isnull().sum()[df.isnull().sum() > 0].to_string())
print("(an empty addons cell means no add-ons were taken, so it is not really missing)")
print("\nExact duplicate rows:", df.duplicated().sum())
print("Repeated rental_id values:", df["rental_id"].duplicated().sum())

# 3. Remove duplicate bookings
df = df.drop_duplicates().reset_index(drop=True)
print("\nShape after removing duplicates:", df.shape)

# 4. Fix inconsistent text in categorical columns
print("\ncar_type values before cleaning:", sorted(df["car_type"].unique()))
df["car_type"] = df["car_type"].str.strip().str.title().replace({"Suv": "SUV", "Muv": "MUV"})
df["booking_channel"] = df["booking_channel"].str.strip().str.title().replace({"Walk-In": "Walk-in"})
print("car_type values after cleaning: ", sorted(df["car_type"].unique()))
print("booking_channel values after cleaning:", sorted(df["booking_channel"].dropna().unique()))

# 5. Treat impossible values as missing
bad_age = ~df["customer_age"].between(18, 80) & df["customer_age"].notna()
bad_km = ((df["km_driven"] < 0) | (df["km_driven"] > 1000 * df["rental_days"])) & df["km_driven"].notna()
print(f"\nInvalid ages (outside 18 to 80): {bad_age.sum()} -> {sorted(df.loc[bad_age, 'customer_age'].astype(int).unique().tolist())}")
print(f"Invalid km_driven (negative or over 1000 km a day): {bad_km.sum()}")
df.loc[bad_age, "customer_age"] = np.nan
df.loc[bad_km, "km_driven"] = np.nan

# 6. Handle missing values
# fuel_type is fixed for each car model, so it can be looked up instead of guessed
fuel_by_model = df.dropna(subset=["fuel_type"]).groupby("car_model")["fuel_type"].agg(lambda s: s.mode()[0])
df["fuel_type"] = df["fuel_type"].fillna(df["car_model"].map(fuel_by_model))

# km_driven: median km per day for that trip type, times the rental length
km_per_day = (df["km_driven"] / df["rental_days"]).groupby(df["trip_type"]).median()
df["km_driven"] = df["km_driven"].fillna(df["trip_type"].map(km_per_day) * df["rental_days"]).round()

df["customer_age"] = df["customer_age"].fillna(df["customer_age"].median())
df["customer_rating"] = df["customer_rating"].fillna(df["customer_rating"].median())
df["booking_channel"] = df["booking_channel"].fillna(df["booking_channel"].mode()[0])
df["addons"] = df["addons"].fillna("")   # an empty basket means no add-ons were taken

for col in ["customer_age", "km_driven", "customer_rating"]:
    df[col] = df[col].astype(int)
print("\nMissing values after imputation:", int(df.drop(columns="addons").isnull().sum().sum()))

# 7. Feature engineering
rent = pd.to_datetime(df["rental_date"])
ret = pd.to_datetime(df["return_date"])
df["rental_month"] = rent.dt.month
df["rental_weekday"] = rent.dt.day_name()
df["return_weekend"] = np.where(ret.dt.day_of_week >= 5, "Yes", "No")
df["season"] = np.where(rent.dt.month.isin([5, 6, 10, 11, 12]), "Peak", "Regular")
df["km_per_day"] = (df["km_driven"] / df["rental_days"]).round(1)
df["addon_count"] = df["addons"].apply(lambda s: 0 if s == "" else len(s.split(";")))
print("\nNew columns: rental_month, rental_weekday, return_weekend, season, km_per_day, addon_count")

df.to_csv("rental_car_clean.csv", index=False)
print("Saved rental_car_clean.csv with shape", df.shape)

# 8. Encode categorical data
model_df = df.drop(columns=["rental_id", "rental_date", "return_date", "customer_id", "car_id",
                            "customer_city", "branch_id", "branch_city", "car_model", "addons"])
model_df["late_return"] = LabelEncoder().fit_transform(model_df["late_return"])   # No = 0, Yes = 1
for col in ["return_weekend", "season"]:
    model_df[col] = (model_df[col] == ("Yes" if col == "return_weekend" else "Peak")).astype(int)
model_df = pd.get_dummies(model_df, columns=["brand", "car_type", "fuel_type", "booking_channel",
                                             "trip_type", "rental_weekday"], dtype=int)
print("\nShape after one-hot encoding:", model_df.shape)

# 9. Scale numeric features (mean 0, standard deviation 1)
scale_cols = ["customer_age", "rental_days", "km_driven", "daily_rate", "rental_amount",
              "late_fee", "customer_rating", "km_per_day", "addon_count"]
model_df[scale_cols] = StandardScaler().fit_transform(model_df[scale_cols])
print("\nMean and standard deviation after scaling:")
print(model_df[scale_cols].agg(["mean", "std"]).round(3).T.to_string())

model_df.to_csv("rental_car_model_ready.csv", index=False)
print("\nSaved rental_car_model_ready.csv with shape", model_df.shape)
print(model_df.iloc[:5, :8].round(3).to_string())
