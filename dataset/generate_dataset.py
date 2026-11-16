"""
Generates the synthetic Rental Car Management System dataset used in
Experiments 2 to 10.

The tables follow the star schema from Experiment 1:
    Rental_Fact  -> rental_car_raw.csv (one row per rental, flattened)
    Dim_Branch   -> branches.csv
    Dim_Car      -> cars.csv
    Dim_Customer -> customers.csv
Dim_Date is derived from rental_date when the warehouse is built.

Each customer belongs to a hidden usage profile (business, family, young
budget, premium). The profile drives rental length, car choice, distance,
add-ons and late returns, so later experiments have real structure to find.
The profile itself is not written to any file.

A few data quality problems are added on purpose (duplicates, missing values,
mixed text casing, impossible values) so that Experiment 2 has work to do.

Run:  python generate_dataset.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 52
N_CUSTOMERS = 800
N_RENTALS = 5000
OUT = Path(__file__).resolve().parent

rng = np.random.default_rng(SEED)

# Dim_Branch: (branch_id, branch_name, city, state, airport branch?)
BRANCHES = [
    ("BR01", "Mumbai Andheri", "Mumbai", "Maharashtra", 0),
    ("BR02", "Mumbai Airport", "Mumbai", "Maharashtra", 1),
    ("BR03", "Thane Ghodbunder", "Thane", "Maharashtra", 0),
    ("BR04", "Pune Hinjewadi", "Pune", "Maharashtra", 0),
    ("BR05", "Bengaluru Koramangala", "Bengaluru", "Karnataka", 0),
    ("BR06", "Bengaluru Airport", "Bengaluru", "Karnataka", 1),
    ("BR07", "Delhi Aerocity", "New Delhi", "Delhi", 1),
    ("BR08", "Gurugram Cyber City", "Gurugram", "Haryana", 0),
    ("BR09", "Hyderabad Gachibowli", "Hyderabad", "Telangana", 0),
    ("BR10", "Goa Panaji", "Panaji", "Goa", 0),
    ("BR11", "Jaipur MI Road", "Jaipur", "Rajasthan", 0),
    ("BR12", "Chennai T Nagar", "Chennai", "Tamil Nadu", 0),
]

# Car models: (brand, model, car_type, fuel_type, base daily rate in INR)
MODELS = [
    ("Maruti Suzuki", "Swift", "Hatchback", "Petrol", 1600),
    ("Maruti Suzuki", "Baleno", "Hatchback", "Petrol", 1800),
    ("Hyundai", "i20", "Hatchback", "Petrol", 1900),
    ("Tata", "Tiago EV", "Hatchback", "Electric", 1900),
    ("Tata", "Altroz", "Hatchback", "Diesel", 1800),
    ("Maruti Suzuki", "Dzire", "Sedan", "CNG", 2000),
    ("Honda", "City", "Sedan", "Petrol", 2800),
    ("Hyundai", "Verna", "Sedan", "Petrol", 2900),
    ("Skoda", "Slavia", "Sedan", "Petrol", 3000),
    ("Hyundai", "Creta", "SUV", "Diesel", 3400),
    ("Kia", "Seltos", "SUV", "Petrol", 3400),
    ("Tata", "Nexon EV", "SUV", "Electric", 3200),
    ("Mahindra", "XUV700", "SUV", "Diesel", 4500),
    ("Mahindra", "Thar", "SUV", "Diesel", 4000),
    ("Mahindra", "Scorpio-N", "SUV", "Diesel", 4200),
    ("Maruti Suzuki", "Ertiga", "MUV", "CNG", 2600),
    ("Kia", "Carens", "MUV", "Petrol", 3200),
    ("Toyota", "Innova Crysta", "MUV", "Diesel", 4800),
    ("Toyota", "Fortuner", "Luxury", "Diesel", 7500),
    ("Mercedes-Benz", "C-Class", "Luxury", "Petrol", 11000),
    ("BMW", "3 Series", "Luxury", "Diesel", 12000),
    ("Audi", "Q3", "Luxury", "Petrol", 10500),
]

ADDONS = ["GPS_Navigation", "Child_Seat", "Additional_Driver", "Full_Insurance",
          "Roadside_Assistance", "FASTag", "Fuel_Prepaid", "Doorstep_Delivery",
          "Airport_Pickup", "Chauffeur", "WiFi_Hotspot", "Roof_Carrier"]

# Flat charges per rental for each add-on (INR)
ADDON_PRICE = {"GPS_Navigation": 300, "Child_Seat": 400, "Additional_Driver": 600,
               "Full_Insurance": 900, "Roadside_Assistance": 350, "FASTag": 500,
               "Fuel_Prepaid": 2500, "Doorstep_Delivery": 500, "Airport_Pickup": 700,
               "Chauffeur": 1500, "WiFi_Hotspot": 250, "Roof_Carrier": 450}

# Customer home cities and the branch closest to each
HOME_CITIES = [
    ("Mumbai", "Maharashtra", ["BR01", "BR02"], 0.20),
    ("Thane", "Maharashtra", ["BR03"], 0.07),
    ("Navi Mumbai", "Maharashtra", ["BR01", "BR03"], 0.05),
    ("Pune", "Maharashtra", ["BR04"], 0.10),
    ("Bengaluru", "Karnataka", ["BR05", "BR06"], 0.14),
    ("New Delhi", "Delhi", ["BR07"], 0.10),
    ("Gurugram", "Haryana", ["BR08"], 0.06),
    ("Noida", "Uttar Pradesh", ["BR07", "BR08"], 0.04),
    ("Hyderabad", "Telangana", ["BR09"], 0.08),
    ("Chennai", "Tamil Nadu", ["BR12"], 0.07),
    ("Jaipur", "Rajasthan", ["BR11"], 0.04),
    ("Ahmedabad", "Gujarat", ["BR01", "BR04"], 0.03),
    ("Panaji", "Goa", ["BR10"], 0.02),
]

FIRST = ["Aarav", "Vivaan", "Aditya", "Arjun", "Rohan", "Karan", "Siddharth", "Rahul",
         "Vikram", "Nikhil", "Pranav", "Harsh", "Yash", "Omkar", "Tanmay", "Kunal",
         "Ananya", "Diya", "Isha", "Kavya", "Meera", "Neha", "Pooja", "Riya",
         "Sneha", "Tanvi", "Shreya", "Aditi", "Priya", "Nisha", "Sakshi", "Anjali",
         "Farhan", "Imran", "Zoya", "Sana", "Joseph", "Maria", "Gurpreet", "Harleen"]
LAST = ["Sharma", "Verma", "Patil", "Kulkarni", "Deshmukh", "Iyer", "Nair", "Menon",
        "Reddy", "Rao", "Gupta", "Agarwal", "Mehta", "Shah", "Joshi", "Singh",
        "Chauhan", "Yadav", "Pillai", "Khan", "Shaikh", "D'Souza", "Fernandes",
        "Gill", "Bose", "Banerjee", "Mishra", "Pandey", "Jain", "Kapoor"]

PROFILES = {
    #            share  rentals/yr weight
    "business": (0.27, 1.9),
    "family":   (0.28, 0.8),
    "young":    (0.30, 1.0),
    "premium":  (0.15, 0.9),
}


def pick(options, probs):
    return options[rng.choice(len(options), p=np.array(probs) / np.sum(probs))]


def make_branches():
    return pd.DataFrame(BRANCHES, columns=["branch_id", "branch_name", "city", "state", "is_airport"])


def make_cars():
    # Each branch keeps a fleet: cheaper cars in larger numbers
    rows, n = [], 1
    for b in BRANCHES:
        for brand, model, ctype, fuel, rate in MODELS:
            copies = 2 if ctype in ("Hatchback", "Sedan") else 1
            if ctype == "Luxury" and rng.random() < 0.4:
                copies = 0
            for _ in range(copies):
                rows.append({"car_id": f"CAR{n:04d}", "branch_id": b[0], "brand": brand,
                             "car_model": model, "car_type": ctype, "fuel_type": fuel,
                             "base_rate": rate,
                             "registration_year": int(rng.integers(2019, 2026))})
                n += 1
    return pd.DataFrame(rows)


def make_customers():
    names = list(PROFILES)
    shares = [PROFILES[p][0] for p in names]
    rows = []
    for i in range(1, N_CUSTOMERS + 1):
        prof = pick(names, shares)
        if prof == "business":
            age = rng.normal(36, 7)
        elif prof == "family":
            age = rng.normal(41, 7)
        elif prof == "young":
            age = rng.normal(24, 2.5)
        else:
            age = rng.normal(46, 8)
        age = int(np.clip(round(age), 21 if prof == "young" else 24, 68))
        city = pick(HOME_CITIES, [c[3] for c in HOME_CITIES])
        country = "India"
        if rng.random() < 0.05:
            country = pick(["United Arab Emirates", "United States", "United Kingdom", "Singapore"],
                           [0.4, 0.25, 0.2, 0.15])
        rows.append({"customer_id": f"C{i:04d}",
                     "customer_name": f"{rng.choice(FIRST)} {rng.choice(LAST)}",
                     "age": age, "city": city[0], "state": city[1], "country": country,
                     "_profile": prof, "_home": city[2]})
    return pd.DataFrame(rows)


def rental_dates(n):
    # Two years of bookings with summer holiday and festive season peaks
    month_w = np.array([0.9, 0.85, 0.95, 1.1, 1.35, 1.25, 0.7, 0.7, 0.85, 1.15, 1.3, 1.45])
    days = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    w = month_w[days.month - 1] * np.where(days.year == 2025, 1.15, 1.0)
    return days, w / w.sum()


def make_rentals(branches, cars, customers):
    days, day_w = rental_dates(N_RENTALS)
    activity = customers["_profile"].map(lambda p: PROFILES[p][1]).to_numpy()
    activity = activity * rng.gamma(2.0, 0.5, len(customers))
    cust_idx = rng.choice(len(customers), size=N_RENTALS, p=activity / activity.sum())
    airport = set(branches.loc[branches.is_airport == 1, "branch_id"])
    rows = []
    for k, ci in enumerate(cust_idx, start=1):
        c = customers.iloc[ci]
        prof = c["_profile"]

        # Start date: business trips start on weekdays, leisure trips near weekends
        while True:
            d = days[rng.choice(len(days), p=day_w)]
            dow = d.day_of_week
            if prof == "business" and dow >= 5 and rng.random() < 0.8:
                continue
            if prof in ("family", "young") and dow < 4 and rng.random() < 0.55:
                continue
            break

        # Branch: usually near home, otherwise a travel destination
        if rng.random() < 0.72:
            branch = rng.choice(c["_home"])
            if prof == "business" and rng.random() < 0.5:
                near_air = [b for b in c["_home"] if b in airport]
                branch = near_air[0] if near_air else rng.choice(sorted(airport))
        else:
            dest_w = [1.0] * len(BRANCHES)
            if prof in ("young", "family"):
                dest_w[9] = 4.0   # Goa
                dest_w[10] = 2.5  # Jaipur
            if prof == "business":
                dest_w = [3.0 if b[4] else 1.0 for b in BRANCHES]
            branch = pick([b[0] for b in BRANCHES], dest_w)

        # Trip shape per profile
        if prof == "business":
            rdays = pick([1, 2, 3, 4], [0.42, 0.33, 0.18, 0.07])
            trip = pick(["Local", "Outstation"], [0.68, 0.32])
            ctype = pick(["Sedan", "SUV", "Hatchback", "Luxury"], [0.55, 0.28, 0.12, 0.05])
            channel = pick(["Corporate", "Mobile App", "Website"], [0.55, 0.3, 0.15])
            kmpd = rng.normal(95, 30)
        elif prof == "family":
            rdays = int(np.clip(round(rng.normal(5.2, 1.8)), 2, 12))
            trip = pick(["Local", "Outstation"], [0.2, 0.8])
            ctype = pick(["MUV", "SUV", "Sedan"], [0.5, 0.4, 0.1])
            channel = pick(["Website", "Mobile App", "Walk-in"], [0.45, 0.4, 0.15])
            kmpd = rng.normal(210, 55) if trip == "Outstation" else rng.normal(70, 20)
        elif prof == "young":
            rdays = pick([1, 2, 3, 4, 5], [0.38, 0.34, 0.16, 0.08, 0.04])
            trip = pick(["Local", "Outstation"], [0.55, 0.45])
            ctype = pick(["Hatchback", "SUV", "Sedan"], [0.68, 0.22, 0.10])
            channel = pick(["Mobile App", "Website", "Walk-in"], [0.75, 0.15, 0.10])
            kmpd = rng.normal(165, 55) if trip == "Outstation" else rng.normal(80, 25)
        else:
            rdays = pick([1, 2, 3, 4, 5], [0.3, 0.3, 0.2, 0.12, 0.08])
            trip = pick(["Local", "Outstation"], [0.62, 0.38])
            ctype = pick(["Luxury", "SUV"], [0.85, 0.15])
            channel = pick(["Website", "Mobile App", "Corporate", "Walk-in"], [0.4, 0.3, 0.2, 0.1])
            kmpd = rng.normal(110, 35)

        fleet = cars[(cars.branch_id == branch) & (cars.car_type == ctype)]
        if fleet.empty:  # some branches have no luxury cars
            fleet = cars[(cars.branch_id == branch) & (cars.car_type == "SUV")]
        if prof == "young" and ctype == "SUV" and rng.random() < 0.6:
            thar = fleet[fleet.car_model == "Thar"]
            fleet = thar if not thar.empty else fleet
        if prof == "premium" and ctype == "SUV":
            xuv = fleet[fleet.car_model == "XUV700"]
            fleet = xuv if not xuv.empty else fleet
        car = fleet.iloc[rng.integers(len(fleet))]

        km = int(max(15, kmpd) * rdays * rng.uniform(0.9, 1.1))
        peak = d.month in (5, 6, 10, 11, 12)
        rate = car.base_rate * (1.12 if peak else 1.0) * (1.05 if d.day_of_week >= 4 else 1.0)
        rate = int(round(rate * rng.uniform(0.95, 1.05), -1))

        # Add-on basket
        p = {a: 0.02 for a in ADDONS}
        p["Full_Insurance"] = 0.10
        if trip == "Outstation":
            p["FASTag"] += 0.70
            p["Roadside_Assistance"] += 0.52
            p["Fuel_Prepaid"] += 0.22
        if prof == "family":
            p["Child_Seat"] += 0.68
            p["Additional_Driver"] += 0.38 + (0.22 if trip == "Outstation" else 0)
            p["Roof_Carrier"] += 0.40 if rdays >= 4 else 0.08
        if prof == "business":
            p["Airport_Pickup"] += 0.30 + (0.35 if branch in airport else 0)
            p["WiFi_Hotspot"] += 0.58
            p["GPS_Navigation"] += 0.25
            p["Doorstep_Delivery"] += 0.12
        if prof == "premium":
            p["Chauffeur"] += 0.60
            p["Full_Insurance"] += 0.62
            p["Doorstep_Delivery"] += 0.50
        if prof == "young":
            p["GPS_Navigation"] += 0.40
            p["Fuel_Prepaid"] += 0.15
        basket = [a for a in ADDONS if rng.random() < min(p[a], 0.95)]
        if "Chauffeur" in basket and "Additional_Driver" in basket:
            basket.remove("Additional_Driver")
        addon_charges = sum(ADDON_PRICE[a] for a in basket)

        # Late return: longer, outstation, weekend-ending trips by younger drivers
        ret = d + pd.Timedelta(days=int(rdays))
        z = (-2.9 + 0.24 * rdays + 1.0 * (trip == "Outstation")
             + 1.35 * (prof == "young") - 1.1 * (channel == "Corporate")
             + 0.65 * (ret.day_of_week == 6) + 0.30 * peak - 0.6 * (ctype == "Luxury")
             + rng.normal(0, 0.4))
        late = rng.random() < 1 / (1 + np.exp(-z))
        late_fee = 0
        if late:
            hours = int(np.ceil(rng.exponential(4.5) + 1))
            late_fee = int(round(min(hours * rate / 8, rate * 1.5), -1))

        rating = int(np.clip(round(rng.normal(4.3, 0.65) - 0.5 * late), 1, 5))
        rows.append({
            "rental_id": "",
            "rental_date": d.date().isoformat(),
            "return_date": ret.date().isoformat(),
            "customer_id": c["customer_id"],
            "customer_age": c["age"],
            "customer_city": c["city"],
            "branch_id": branch,
            "branch_city": next(b[2] for b in BRANCHES if b[0] == branch),
            "car_id": car.car_id,
            "brand": car.brand,
            "car_model": car.car_model,
            "car_type": car.car_type,
            "fuel_type": car.fuel_type,
            "booking_channel": channel,
            "trip_type": trip,
            "rental_days": int(rdays),
            "km_driven": km,
            "daily_rate": rate,
            "addons": ";".join(basket),
            "rental_amount": int(rate * rdays + addon_charges),
            "late_return": "Yes" if late else "No",
            "late_fee": late_fee,
            "customer_rating": rating,
        })
    out = pd.DataFrame(rows).sort_values("rental_date", kind="stable").reset_index(drop=True)
    out["rental_id"] = [f"R{i:05d}" for i in range(1, len(out) + 1)]
    return out


def add_quality_problems(df):
    raw = df.copy()
    n = len(raw)

    # Missing values
    for col, frac in [("customer_age", 0.03), ("km_driven", 0.025), ("fuel_type", 0.015),
                      ("customer_rating", 0.04), ("booking_channel", 0.01)]:
        idx = rng.choice(n, int(n * frac), replace=False)
        raw[col] = raw[col].astype(object)
        raw.loc[idx, col] = np.nan

    # Inconsistent text entered at different branches
    idx = rng.choice(n, int(n * 0.08), replace=False)
    raw.loc[idx, "car_type"] = raw.loc[idx, "car_type"].map(
        lambda v: rng.choice([v.upper(), v.lower(), f" {v} ", v.capitalize() + " "]))
    idx = rng.choice(n, int(n * 0.05), replace=False)
    raw.loc[idx, "booking_channel"] = raw.loc[idx, "booking_channel"].map(
        lambda v: v if pd.isna(v) else rng.choice([v.lower(), v.upper(), f"{v} "]))

    # Impossible values from manual entry and odometer misreads
    ok = raw["customer_age"].notna()
    for val, cnt in [(0, 3), (150, 2), (7, 2)]:
        idx = rng.choice(raw.index[ok], cnt, replace=False)
        raw.loc[idx, "customer_age"] = val
    ok = raw["km_driven"].notna()
    idx = rng.choice(raw.index[ok], 4, replace=False)
    raw.loc[idx, "km_driven"] = 99999
    idx = rng.choice(raw.index[ok], 3, replace=False)
    raw.loc[idx, "km_driven"] = -rng.integers(20, 200, 3)

    # Bookings entered twice
    dup = raw.sample(45, random_state=SEED)
    raw = pd.concat([raw, dup]).sort_values(["rental_date", "rental_id"]).reset_index(drop=True)
    return raw


def main():
    branches = make_branches()
    cars = make_cars()
    customers = make_customers()
    rentals = make_rentals(branches, cars, customers)
    raw = add_quality_problems(rentals)

    branches.drop(columns="is_airport").to_csv(OUT / "branches.csv", index=False)
    cars.drop(columns="base_rate").to_csv(OUT / "cars.csv", index=False)
    customers.drop(columns=["_profile", "_home"]).to_csv(OUT / "customers.csv", index=False)
    raw.to_csv(OUT / "rental_car_raw.csv", index=False)

    print(f"branches: {len(branches)}  cars: {len(cars)}  customers: {len(customers)}")
    print(f"rentals: {len(rentals)} clean, {len(raw)} rows written with quality problems")
    print("late return share:", round((rentals.late_return == "Yes").mean(), 3))


if __name__ == "__main__":
    main()
