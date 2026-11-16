"""
Experiment 10: Rental Car BI Dashboard (Streamlit).

Brings the earlier experiments together in one decision support app:
    Overview            KPIs and trends
    Data Preparation    raw vs cleaned data (Experiment 2)
    Warehouse and OLAP  star schema queries (Experiment 1)
    Late Return Risk    decision tree classifier (Experiments 4 and 5)
    Customer Segments   k-means clustering (Experiments 6 and 7)
    Add-on Bundles      association rules (Experiments 8 and 9)

Run:  streamlit run app.py
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DB = HERE / "rental_dw.db"
TEMPLATE = "simple_white"
COLOURS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860"]

st.set_page_config(page_title="Rental Car BI Dashboard", layout="wide")


def rupees(v):
    if v >= 1e7:
        return f"₹{v / 1e7:.2f} Cr"
    if v >= 1e5:
        return f"₹{v / 1e5:.2f} L"
    return f"₹{v:,.0f}"


@st.cache_data
def load_rentals():
    return pd.read_csv(DATA / "rental_car_clean.csv", keep_default_na=False,
                       parse_dates=["rental_date", "return_date"])


@st.cache_data
def load_raw():
    return pd.read_csv(DATA / "rental_car_raw.csv")


def query(sql, params=()):
    if not DB.exists():
        import build_warehouse
        build_warehouse.build()
    with sqlite3.connect(DB) as con:
        return pd.read_sql_query(sql, con, params=params)


def show_sql(sql):
    with st.expander("SQL used"):
        st.code(sql.strip(), language="sql")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_overview(df):
    st.title("Overview")
    years = sorted(df["rental_date"].dt.year.unique())
    pick = st.radio("Year", ["Both"] + [str(y) for y in years], horizontal=True)
    d = df if pick == "Both" else df[df["rental_date"].dt.year == int(pick)]

    c = st.columns(5)
    c[0].metric("Rentals", f"{len(d):,}")
    c[1].metric("Revenue", rupees(d["rental_amount"].sum()))
    c[2].metric("Avg rental length", f"{d['rental_days'].mean():.2f} days")
    c[3].metric("Late returns", f"{(d['late_return'] == 'Yes').mean():.1%}")
    c[4].metric("Avg customer rating", f"{d['customer_rating'].mean():.2f} / 5")

    left, right = st.columns(2)
    monthly = (d.assign(month=d["rental_date"].dt.to_period("M").dt.to_timestamp())
                 .groupby("month")["rental_amount"].sum().reset_index())
    fig = px.line(monthly, x="month", y="rental_amount", markers=True, template=TEMPLATE,
                  labels={"month": "", "rental_amount": "Revenue (₹)"}, title="Monthly revenue")
    left.plotly_chart(fig, width="stretch")

    by_type = d.groupby("car_type")["rental_amount"].sum().sort_values().reset_index()
    fig = px.bar(by_type, x="rental_amount", y="car_type", orientation="h", template=TEMPLATE,
                 labels={"rental_amount": "Revenue (₹)", "car_type": ""}, title="Revenue by car type",
                 color_discrete_sequence=[COLOURS[0]])
    right.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    by_city = d.groupby("branch_city").size().sort_values().reset_index(name="rentals")
    fig = px.bar(by_city, x="rentals", y="branch_city", orientation="h", template=TEMPLATE,
                 labels={"branch_city": "", "rentals": "Rentals"}, title="Rentals by branch city",
                 color_discrete_sequence=[COLOURS[2]])
    left.plotly_chart(fig, width="stretch")

    late = (d.groupby(["booking_channel", "trip_type"])["late_return"]
              .apply(lambda s: (s == "Yes").mean() * 100).reset_index(name="late_pct"))
    fig = px.bar(late, x="booking_channel", y="late_pct", color="trip_type", barmode="group",
                 template=TEMPLATE, color_discrete_sequence=COLOURS,
                 labels={"booking_channel": "", "late_pct": "Late returns (%)", "trip_type": "Trip"},
                 title="Late return rate by channel and trip type")
    right.plotly_chart(fig, width="stretch")


def page_preparation(df):
    st.title("Data Preparation")
    st.write("How the raw booking export was cleaned before analysis (Experiment 2).")
    raw = load_raw()
    dups = int(raw.duplicated().sum())
    dedup = raw.drop_duplicates()
    bad_age = int((~dedup["customer_age"].between(18, 80) & dedup["customer_age"].notna()).sum())
    bad_km = int(((dedup["km_driven"] < 0) | (dedup["km_driven"] > 1000 * dedup["rental_days"])).sum())
    variants = dedup["car_type"].nunique()

    c = st.columns(4)
    c[0].metric("Raw rows", f"{len(raw):,}")
    c[1].metric("Duplicate bookings removed", dups)
    c[2].metric("Impossible values fixed", bad_age + bad_km)
    c[3].metric("Clean rows", f"{len(df):,}")

    left, right = st.columns(2)
    miss = raw.drop(columns="addons").isnull().sum()
    miss = miss[miss > 0].sort_values().reset_index()
    miss.columns = ["column", "missing"]
    fig = px.bar(miss, x="missing", y="column", orientation="h", template=TEMPLATE,
                 color_discrete_sequence=[COLOURS[3]], title="Missing values in the raw data",
                 labels={"column": "", "missing": "Missing cells"})
    left.plotly_chart(fig, width="stretch")

    steps = pd.DataFrame({
        "Problem": ["Duplicate bookings", "Mixed text in car_type and booking_channel",
                    "Impossible ages (0, 7, 150)", "Odometer misreads (negative or 99999 km)",
                    "Missing fuel_type", "Missing km_driven", "Missing age, rating, channel"],
        "Fix": ["Dropped exact duplicates", f"Trimmed and fixed case ({variants} spellings to 5 types)",
                "Set to missing, then median", "Set to missing, then imputed",
                "Looked up from car model", "Median km per day for the trip type x days",
                "Median or mode"],
    })
    right.subheader("Cleaning steps")
    right.dataframe(steps, hide_index=True, width="stretch")

    st.subheader("Cleaned data sample")
    st.dataframe(df.drop(columns=["return_date"]).head(50), width="stretch", height=300)


def page_olap():
    st.title("Warehouse and OLAP")
    st.write("Star schema from Experiment 1: Rental_Fact joined to Dim_Date, Dim_Car, "
             "Dim_Customer and Dim_Branch.")
    counts = query("""SELECT 'Rental_Fact' AS table_name, COUNT(*) AS rows FROM Rental_Fact
                      UNION ALL SELECT 'Dim_Date', COUNT(*) FROM Dim_Date
                      UNION ALL SELECT 'Dim_Car', COUNT(*) FROM Dim_Car
                      UNION ALL SELECT 'Dim_Customer', COUNT(*) FROM Dim_Customer
                      UNION ALL SELECT 'Dim_Branch', COUNT(*) FROM Dim_Branch""")
    st.dataframe(counts, hide_index=True)

    tabs = st.tabs(["Roll-up", "Drill-down", "Slice", "Dice", "Pivot"])

    with tabs[0]:
        level = st.radio("Roll up revenue to", ["Year", "Quarter", "Month"], horizontal=True)
        group = {"Year": "d.Year", "Quarter": "d.Year, d.Quarter",
                 "Month": "d.Year, d.Month_No, d.Month"}[level]
        sql = f"""SELECT {group}, COUNT(*) AS Rentals, SUM(f.Rental_Amount) AS Revenue
                  FROM Rental_Fact f JOIN Dim_Date d ON f.Date_ID = d.Date_ID
                  GROUP BY {group} ORDER BY {group}"""
        res = query(sql)
        res["Period"] = res.drop(columns=["Rentals", "Revenue", "Month_No"], errors="ignore") \
                           .astype(str).agg(" ".join, axis=1)
        fig = px.bar(res, x="Period", y="Revenue", template=TEMPLATE, color_discrete_sequence=[COLOURS[0]],
                     title=f"Revenue by {level.lower()}")
        st.plotly_chart(fig, width="stretch")
        show_sql(sql)

    with tabs[1]:
        states = query("SELECT DISTINCT State FROM Dim_Branch ORDER BY State")["State"].tolist()
        state = st.selectbox("Drill down from state to branch", states, index=states.index("Maharashtra"))
        sql = """SELECT b.City, b.Branch_Name, COUNT(*) AS Rentals,
                        SUM(f.Rental_Amount) AS Revenue, ROUND(AVG(f.Is_Late) * 100, 1) AS Late_Pct
                 FROM Rental_Fact f JOIN Dim_Branch b ON f.Branch_ID = b.Branch_ID
                 WHERE b.State = ? GROUP BY b.City, b.Branch_Name ORDER BY Revenue DESC"""
        res = query(sql, (state,))
        top = query("""SELECT b.State, SUM(f.Rental_Amount) AS Revenue FROM Rental_Fact f
                       JOIN Dim_Branch b ON f.Branch_ID = b.Branch_ID GROUP BY b.State
                       ORDER BY Revenue DESC""")
        left, right = st.columns(2)
        left.plotly_chart(px.bar(top, x="State", y="Revenue", template=TEMPLATE, title="Revenue by state",
                                 color_discrete_sequence=[COLOURS[0]]), width="stretch")
        right.plotly_chart(px.bar(res, x="Branch_Name", y="Revenue", color="City", template=TEMPLATE,
                                  title=f"Branches in {state}", color_discrete_sequence=COLOURS),
                           width="stretch")
        st.dataframe(res, hide_index=True)
        show_sql(sql)

    with tabs[2]:
        types = query("SELECT DISTINCT Car_Type FROM Dim_Car ORDER BY Car_Type")["Car_Type"].tolist()
        ctype = st.selectbox("Slice the cube on Car_Type =", types, index=types.index("SUV"))
        sql = """SELECT d.Year, d.Quarter, COUNT(*) AS Rentals, SUM(f.Rental_Amount) AS Revenue,
                        ROUND(AVG(f.Rental_Days), 2) AS Avg_Days
                 FROM Rental_Fact f JOIN Dim_Car c ON f.Car_ID = c.Car_ID
                 JOIN Dim_Date d ON f.Date_ID = d.Date_ID
                 WHERE c.Car_Type = ? GROUP BY d.Year, d.Quarter ORDER BY d.Year, d.Quarter"""
        res = query(sql, (ctype,))
        res["Period"] = res["Year"].astype(str) + " " + res["Quarter"]
        st.plotly_chart(px.line(res, x="Period", y="Revenue", markers=True, template=TEMPLATE,
                                title=f"{ctype} revenue by quarter"), width="stretch")
        st.dataframe(res.drop(columns="Period"), hide_index=True)
        show_sql(sql)

    with tabs[3]:
        cities = query("SELECT DISTINCT City FROM Dim_Branch ORDER BY City")["City"].tolist()
        a, b, c = st.columns(3)
        pick_types = a.multiselect("Car type", types, default=["SUV", "MUV"])
        pick_cities = b.multiselect("Branch city", cities, default=["Mumbai", "Pune", "Panaji"])
        pick_q = c.multiselect("Quarter", ["Q1", "Q2", "Q3", "Q4"], default=["Q2", "Q4"])
        if pick_types and pick_cities and pick_q:
            ph = lambda xs: ",".join("?" * len(xs))
            sql = f"""SELECT c.Car_Type, b.City, d.Quarter, COUNT(*) AS Rentals,
                             SUM(f.Rental_Amount) AS Revenue, ROUND(AVG(f.Is_Late) * 100, 1) AS Late_Pct
                      FROM Rental_Fact f
                      JOIN Dim_Car c ON f.Car_ID = c.Car_ID
                      JOIN Dim_Branch b ON f.Branch_ID = b.Branch_ID
                      JOIN Dim_Date d ON f.Date_ID = d.Date_ID
                      WHERE c.Car_Type IN ({ph(pick_types)}) AND b.City IN ({ph(pick_cities)})
                        AND d.Quarter IN ({ph(pick_q)})
                      GROUP BY c.Car_Type, b.City, d.Quarter ORDER BY Revenue DESC"""
            res = query(sql, tuple(pick_types + pick_cities + pick_q))
            st.dataframe(res, hide_index=True, width="stretch")
            show_sql(sql)
        else:
            st.info("Pick at least one value in each filter.")

    with tabs[4]:
        year = st.radio("Year", [2024, 2025], horizontal=True, key="pivot_year")
        sql = """SELECT c.Car_Type, d.Quarter, SUM(f.Rental_Amount) AS Revenue
                 FROM Rental_Fact f JOIN Dim_Car c ON f.Car_ID = c.Car_ID
                 JOIN Dim_Date d ON f.Date_ID = d.Date_ID
                 WHERE d.Year = ? GROUP BY c.Car_Type, d.Quarter"""
        res = query(sql, (year,))
        pivot = res.pivot(index="Car_Type", columns="Quarter", values="Revenue")
        fig = px.imshow(pivot / 1e5, text_auto=".1f", aspect="auto", color_continuous_scale="Blues",
                        labels={"color": "₹ lakh"}, title=f"Revenue (₹ lakh), car type x quarter, {year}")
        st.plotly_chart(fig, width="stretch")
        show_sql(sql)


FEATURES_NUM = ["customer_age", "rental_days", "km_driven", "daily_rate"]
FEATURES_CAT = ["car_type", "fuel_type", "booking_channel", "trip_type", "return_weekend", "season"]


@st.cache_resource
def train_late_model(df):
    X = pd.get_dummies(df[FEATURES_NUM + FEATURES_CAT], columns=FEATURES_CAT, dtype=int)
    y = (df["late_return"] == "Yes").astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = DecisionTreeClassifier(criterion="gini", max_depth=5, min_samples_leaf=10, random_state=42)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    metrics = {"accuracy": accuracy_score(y_te, pred), "precision": precision_score(y_te, pred),
               "recall": recall_score(y_te, pred), "baseline": 1 - y_te.mean(),
               "cm": confusion_matrix(y_te, pred)}
    return model, list(X.columns), metrics


def page_late(df):
    st.title("Late Return Risk")
    st.write("Decision tree (Gini, depth 5) trained on 80% of rentals and tested on the other 20%.")
    model, cols, m = train_late_model(df)
    c = st.columns(4)
    c[0].metric("Accuracy", f"{m['accuracy']:.1%}", f"{(m['accuracy'] - m['baseline']) * 100:+.1f} pts vs always 'on time'")
    c[1].metric("Precision (late)", f"{m['precision']:.1%}")
    c[2].metric("Recall (late)", f"{m['recall']:.1%}")
    c[3].metric("Test rentals", f"{int(m['cm'].sum()):,}")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Check a new booking")
        with st.form("booking"):
            a, b = st.columns(2)
            age = a.number_input("Customer age", 18, 80, 24)
            days = b.number_input("Rental days", 1, 15, 3)
            ctype = a.selectbox("Car type", ["Hatchback", "Sedan", "SUV", "MUV", "Luxury"], index=2)
            fuel = b.selectbox("Fuel type", ["Petrol", "Diesel", "CNG", "Electric"], index=1)
            channel = a.selectbox("Booking channel", ["Mobile App", "Website", "Walk-in", "Corporate"])
            trip = b.selectbox("Trip type", ["Outstation", "Local"])
            rate = a.number_input("Daily rate (₹)", 1000, 20000, 4000, step=100)
            km = b.number_input("Expected km", 10, 5000, 600, step=10)
            weekend = a.selectbox("Return on a weekend?", ["Yes", "No"])
            season = b.selectbox("Season", ["Peak", "Regular"])
            go = st.form_submit_button("Predict")
        row = pd.DataFrame([{"customer_age": age, "rental_days": days, "km_driven": km, "daily_rate": rate,
                             "car_type": ctype, "fuel_type": fuel, "booking_channel": channel,
                             "trip_type": trip, "return_weekend": weekend, "season": season}])
        x = pd.get_dummies(row, columns=FEATURES_CAT, dtype=int).reindex(columns=cols, fill_value=0)
        p = model.predict_proba(x)[0, 1]
        level = "High" if p >= 0.5 else "Medium" if p >= 0.25 else "Low"
        st.metric("Chance of a late return", f"{p:.0%}", level, delta_color="off")
        if level == "High":
            st.write("Suggested action: send a return reminder the evening before and the morning "
                     "of the return, and confirm the late fee policy at pickup.")
        elif level == "Medium":
            st.write("Suggested action: send one reminder on the return day.")
        else:
            st.write("No action needed.")

    with right:
        imp = (pd.Series(model.feature_importances_, index=cols).groupby(lambda c: next(
            (f for f in FEATURES_NUM + FEATURES_CAT if c == f or c.startswith(f + "_")), c)).sum()
            .sort_values().tail(8).reset_index())
        imp.columns = ["feature", "importance"]
        fig = px.bar(imp, x="importance", y="feature", orientation="h", template=TEMPLATE,
                     color_discrete_sequence=[COLOURS[0]], title="What drives the prediction",
                     labels={"feature": ""})
        st.plotly_chart(fig, width="stretch")
        cm = pd.DataFrame(m["cm"], index=["Actual on time", "Actual late"],
                          columns=["Predicted on time", "Predicted late"])
        st.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                                  title="Confusion matrix (test set)"), width="stretch")


SEG_FEATURES = ["customer_age", "total_rentals", "avg_rental_days", "avg_km_per_day",
                "avg_daily_rate", "outstation_share", "corporate_share"]


@st.cache_data
def segment_customers(df):
    r = df.assign(is_out=(df["trip_type"] == "Outstation").astype(int),
                  is_corp=(df["booking_channel"] == "Corporate").astype(int),
                  is_late=(df["late_return"] == "Yes").astype(int))
    prof = r.groupby("customer_id").agg(
        customer_age=("customer_age", "first"), total_rentals=("rental_id", "count"),
        avg_rental_days=("rental_days", "mean"), avg_km_per_day=("km_per_day", "mean"),
        avg_daily_rate=("daily_rate", "mean"), outstation_share=("is_out", "mean"),
        corporate_share=("is_corp", "mean"), late_return_rate=("is_late", "mean"),
        total_spend=("rental_amount", "sum")).reset_index()
    km = KMeans(n_clusters=4, n_init=10, random_state=42)
    prof["cluster"] = km.fit_predict(StandardScaler().fit_transform(prof[SEG_FEATURES]))
    means = prof.groupby("cluster")[SEG_FEATURES].mean()
    names = {means["avg_daily_rate"].idxmax(): "Premium renters",
             means["corporate_share"].idxmax(): "Business renters",
             means["avg_rental_days"].idxmax(): "Family road trippers"}
    for c in means.index:
        names.setdefault(c, "Young budget renters")
    prof["segment"] = prof["cluster"].map(names)
    return prof


def page_segments(df):
    st.title("Customer Segments")
    st.write("K-means (k = 4) on standardised customer profiles, as in Experiments 6 and 7.")
    prof = segment_customers(df)
    order = ["Young budget renters", "Family road trippers", "Business renters", "Premium renters"]
    cmap = dict(zip(order, COLOURS))

    c = st.columns(4)
    for col, seg in zip(c, order):
        part = prof[prof["segment"] == seg]
        col.metric(seg, f"{len(part)} customers", f"{rupees(part['total_spend'].sum())} revenue",
                   delta_color="off")

    left, right = st.columns(2)
    fig = px.scatter(prof, x="avg_rental_days", y="avg_daily_rate", color="segment", template=TEMPLATE,
                     color_discrete_map=cmap, category_orders={"segment": order}, opacity=0.7,
                     labels={"avg_rental_days": "Average rental length (days)",
                             "avg_daily_rate": "Average daily rate (₹)", "segment": ""},
                     title="Rental length vs daily rate")
    left.plotly_chart(fig, width="stretch")
    summary = prof.groupby("segment")[SEG_FEATURES + ["late_return_rate"]].mean().reindex(order)
    late = summary["late_return_rate"].mul(100).reset_index()
    fig = px.bar(late, x="segment", y="late_return_rate", color="segment", color_discrete_map=cmap,
                 template=TEMPLATE, labels={"segment": "", "late_return_rate": "Late returns (%)"},
                 title="Late return rate by segment")
    fig.update_layout(showlegend=False)
    right.plotly_chart(fig, width="stretch")

    st.subheader("Segment profiles (averages)")
    st.dataframe(summary.round(2).T, width="stretch")

    st.subheader("Look up a customer")
    cid = st.selectbox("Customer ID", prof["customer_id"].tolist(), index=0)
    row = prof[prof["customer_id"] == cid].iloc[0]
    st.write(f"**{cid}** belongs to **{row['segment']}**: {int(row['total_rentals'])} rentals, "
             f"{row['avg_rental_days']:.1f} days on average, {rupees(row['total_spend'])} spent.")
    history = df[df["customer_id"] == cid].assign(rental_date=lambda x: x["rental_date"].dt.date)
    st.dataframe(history[["rental_id", "rental_date", "branch_city", "car_model", "trip_type",
                          "rental_days", "rental_amount", "late_return"]],
                 hide_index=True, width="stretch")


@st.cache_data
def addon_matrix(df):
    baskets = df.loc[df["addons"] != "", "addons"]
    return baskets.str.get_dummies(sep=";").astype(bool)


def page_bundles(df):
    st.title("Add-on Bundles")
    st.write("Apriori on the add-ons taken in each rental, as in Experiments 8 and 9.")
    tx = addon_matrix(df)
    a, b = st.columns(2)
    sup = a.slider("Minimum support", 0.02, 0.20, 0.05, 0.01)
    conf = b.slider("Minimum confidence", 0.30, 0.90, 0.50, 0.05)
    freq = apriori(tx, min_support=sup, use_colnames=True)
    if freq.empty or freq["itemsets"].apply(len).max() < 2:
        st.info("No itemsets with two or more add-ons at this support. Lower the support.")
        return
    rules = association_rules(freq, metric="confidence", min_threshold=conf)
    rules = rules[rules["lift"] > 1].sort_values("lift", ascending=False)
    fmt = lambda s: ", ".join(sorted(s))
    table = pd.DataFrame({"If the customer takes": rules["antecedents"].apply(fmt),
                          "Offer": rules["consequents"].apply(fmt),
                          "Support": rules["support"].round(3),
                          "Confidence": rules["confidence"].round(3),
                          "Lift": rules["lift"].round(2)})
    c = st.columns(3)
    c[0].metric("Baskets", f"{len(tx):,}")
    c[1].metric("Frequent itemsets", len(freq))
    c[2].metric("Rules", len(rules))

    left, right = st.columns([3, 2])
    left.dataframe(table, hide_index=True, width="stretch", height=380)
    fig = px.scatter(rules.assign(rule=table["If the customer takes"] + " -> " + table["Offer"]),
                     x="support", y="confidence", size="lift", color="lift", hover_name="rule",
                     template=TEMPLATE, color_continuous_scale="Viridis", title="Support vs confidence")
    right.plotly_chart(fig, width="stretch")

    st.subheader("Bundle suggestion at the booking counter")
    cart = st.multiselect("Add-ons already in the booking", sorted(tx.columns),
                          default=["Child_Seat"])
    if cart:
        hits = rules[rules["antecedents"].apply(lambda s: s <= set(cart))
                     & rules["consequents"].apply(lambda s: not (s & set(cart)))]
        if hits.empty:
            st.write("No rule matches this basket at the current thresholds.")
        else:
            best = (hits.assign(offer=hits["consequents"].apply(fmt))
                        .sort_values("confidence", ascending=False)
                        .drop_duplicates("offer").head(3))
            for _, r in best.iterrows():
                st.write(f"Suggest **{r['offer']}**: taken in {r['confidence']:.0%} of similar "
                         f"bookings ({r['lift']:.1f} times more often than usual).")


PAGES = {
    "Overview": page_overview,
    "Data Preparation": page_preparation,
    "Warehouse and OLAP": lambda df: page_olap(),
    "Late Return Risk": page_late,
    "Customer Segments": page_segments,
    "Add-on Bundles": page_bundles,
}


def main():
    df = load_rentals()
    names = list(PAGES)
    wanted = st.query_params.get("page", "Overview")
    start = names.index(wanted) if wanted in names else 0
    st.sidebar.header("Rental Car BI")
    page = st.sidebar.radio("Go to", names, index=start)
    st.sidebar.caption(f"{len(df):,} rentals, {df['rental_date'].min():%b %Y} to "
                       f"{df['rental_date'].max():%b %Y}, 12 branches")
    PAGES[page](df)


main()
