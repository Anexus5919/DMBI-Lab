"""
Builds the rental car data warehouse (SQLite) used by the dashboard.

The tables follow the star schema designed in Experiment 1. Two extra
measures, Km_Driven and Is_Late, are kept in the fact table because the
dashboard reports on distance and late returns.

Run:  python build_warehouse.py
"""
import sqlite3
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DB = HERE / "rental_dw.db"

SCHEMA = """
DROP TABLE IF EXISTS Rental_Fact;
DROP TABLE IF EXISTS Dim_Date;
DROP TABLE IF EXISTS Dim_Car;
DROP TABLE IF EXISTS Dim_Customer;
DROP TABLE IF EXISTS Dim_Branch;

CREATE TABLE Dim_Date (
    Date_ID TEXT PRIMARY KEY, Date TEXT, Day INTEGER, Month TEXT,
    Month_No INTEGER, Quarter TEXT, Year INTEGER
);
CREATE TABLE Dim_Car (
    Car_ID TEXT PRIMARY KEY, Car_Model TEXT, Brand TEXT, Car_Type TEXT, Fuel_Type TEXT
);
CREATE TABLE Dim_Customer (
    Customer_ID TEXT PRIMARY KEY, Customer_Name TEXT, City TEXT, State TEXT, Country TEXT
);
CREATE TABLE Dim_Branch (
    Branch_ID TEXT PRIMARY KEY, Branch_Name TEXT, City TEXT, State TEXT
);
CREATE TABLE Rental_Fact (
    Rental_ID TEXT PRIMARY KEY,
    Date_ID TEXT REFERENCES Dim_Date(Date_ID),
    Car_ID TEXT REFERENCES Dim_Car(Car_ID),
    Customer_ID TEXT REFERENCES Dim_Customer(Customer_ID),
    Branch_ID TEXT REFERENCES Dim_Branch(Branch_ID),
    Rental_Days INTEGER, Rental_Amount REAL, Late_Fee REAL,
    Km_Driven INTEGER, Is_Late INTEGER
);
CREATE INDEX idx_fact_date ON Rental_Fact(Date_ID);
CREATE INDEX idx_fact_car ON Rental_Fact(Car_ID);
CREATE INDEX idx_fact_branch ON Rental_Fact(Branch_ID);
"""


def build():
    rentals = pd.read_csv(DATA / "rental_car_clean.csv")
    cars = pd.read_csv(DATA / "cars.csv")
    customers = pd.read_csv(DATA / "customers.csv")
    branches = pd.read_csv(DATA / "branches.csv")

    dates = pd.to_datetime(rentals["rental_date"].drop_duplicates().sort_values())
    dim_date = pd.DataFrame({
        "Date_ID": dates.dt.strftime("%Y%m%d"),
        "Date": dates.dt.strftime("%Y-%m-%d"),
        "Day": dates.dt.day,
        "Month": dates.dt.month_name(),
        "Month_No": dates.dt.month,
        "Quarter": "Q" + dates.dt.quarter.astype(str),
        "Year": dates.dt.year,
    })
    dim_car = cars.rename(columns={"car_id": "Car_ID", "car_model": "Car_Model", "brand": "Brand",
                                   "car_type": "Car_Type", "fuel_type": "Fuel_Type"})
    dim_car = dim_car[["Car_ID", "Car_Model", "Brand", "Car_Type", "Fuel_Type"]]
    dim_customer = customers.rename(columns={"customer_id": "Customer_ID", "customer_name": "Customer_Name",
                                             "city": "City", "state": "State", "country": "Country"})
    dim_customer = dim_customer[["Customer_ID", "Customer_Name", "City", "State", "Country"]]
    dim_branch = branches.rename(columns={"branch_id": "Branch_ID", "branch_name": "Branch_Name",
                                          "city": "City", "state": "State"})
    fact = pd.DataFrame({
        "Rental_ID": rentals["rental_id"],
        "Date_ID": pd.to_datetime(rentals["rental_date"]).dt.strftime("%Y%m%d"),
        "Car_ID": rentals["car_id"],
        "Customer_ID": rentals["customer_id"],
        "Branch_ID": rentals["branch_id"],
        "Rental_Days": rentals["rental_days"],
        "Rental_Amount": rentals["rental_amount"],
        "Late_Fee": rentals["late_fee"],
        "Km_Driven": rentals["km_driven"],
        "Is_Late": (rentals["late_return"] == "Yes").astype(int),
    })

    con = sqlite3.connect(DB)
    con.executescript(SCHEMA)
    for name, frame in [("Dim_Date", dim_date), ("Dim_Car", dim_car), ("Dim_Customer", dim_customer),
                        ("Dim_Branch", dim_branch), ("Rental_Fact", fact)]:
        frame.to_sql(name, con, if_exists="append", index=False)
        print(f"{name:13s} {len(frame):6d} rows")
    con.commit()
    con.close()
    print("Warehouse written to", DB.name)


if __name__ == "__main__":
    build()
