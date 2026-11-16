"""
Experiment 6: K-Means clustering in Python.

Customers of the rental company are grouped by how they rent: how often,
for how long, how far they drive, what they pay per day, how often they
take the car out of the city and how often the booking comes through an
employer's corporate account. The clustering ignores late returns and
spend, which are used afterwards to describe each group.

Input : rental_car_clean.csv (output of Experiment 2)
Output: customer_profiles.csv (also used in Experiment 7), figures/
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 12)
os.makedirs("figures", exist_ok=True)

# Step 1: load rentals and build one profile row per customer
rentals = pd.read_csv("rental_car_clean.csv")
rentals["is_outstation"] = (rentals["trip_type"] == "Outstation").astype(int)
rentals["is_late"] = (rentals["late_return"] == "Yes").astype(int)
rentals["is_corporate"] = (rentals["booking_channel"] == "Corporate").astype(int)

profiles = rentals.groupby("customer_id").agg(
    customer_age=("customer_age", "first"),
    total_rentals=("rental_id", "count"),
    avg_rental_days=("rental_days", "mean"),
    avg_km_per_day=("km_per_day", "mean"),
    avg_daily_rate=("daily_rate", "mean"),
    outstation_share=("is_outstation", "mean"),
    corporate_share=("is_corporate", "mean"),
    avg_addons=("addon_count", "mean"),
    late_return_rate=("is_late", "mean"),
    total_spend=("rental_amount", "sum"),
).round(2).reset_index()
profiles.to_csv("customer_profiles.csv", index=False)
print("Rentals:", len(rentals), "| Customers:", len(profiles))
print(profiles.head().to_string(index=False))

# Step 2: choose the clustering features
features = ["customer_age", "total_rentals", "avg_rental_days", "avg_km_per_day",
            "avg_daily_rate", "outstation_share", "corporate_share"]
X = profiles[features]
print("\nFeatures used:", features)
print(X.describe().round(2).to_string())

# Step 3: standardise so that rupees and kilometres do not dominate the distance
X_scaled = StandardScaler().fit_transform(X)
print("\nAfter scaling, mean:", X_scaled.mean(axis=0).round(3), "std:", X_scaled.std(axis=0).round(3))

# Step 4: elbow method and silhouette score for k = 2 to 10
rows = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
    rows.append({"k": k, "inertia": round(km.inertia_, 1),
                 "silhouette": round(silhouette_score(X_scaled, km.labels_), 4)})
scores = pd.DataFrame(rows)
print("\n=== Choosing k ===")
print(scores.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].plot(scores["k"], scores["inertia"], marker="o")
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("Number of clusters (k)")
axes[0].set_ylabel("Inertia (within-cluster SSE)")
axes[1].plot(scores["k"], scores["silhouette"], marker="o", color="#c44e52")
axes[1].set_title("Silhouette Score")
axes[1].set_xlabel("Number of clusters (k)")
axes[1].set_ylabel("Silhouette score")
for ax in axes:
    ax.axvline(4, color="grey", linestyle="--", linewidth=1)
    ax.set_xticks(range(2, 11))
plt.tight_layout()
plt.savefig("figures/fig1_elbow_silhouette.png", dpi=150)
plt.close()

# Step 5: run K-Means with k = 4
K = 4
kmeans = KMeans(n_clusters=K, n_init=10, random_state=42)
profiles["cluster"] = kmeans.fit_predict(X_scaled)
print("\nCustomers per cluster:")
print(profiles["cluster"].value_counts().sort_index().to_string())

# Step 6: evaluate the clustering
print("\n=== Evaluation (k = 4) ===")
print("Inertia             :", round(kmeans.inertia_, 2))
print("Silhouette score    :", round(silhouette_score(X_scaled, profiles["cluster"]), 4))
print("Davies-Bouldin index:", round(davies_bouldin_score(X_scaled, profiles["cluster"]), 4))

# Step 7: describe each cluster in original units and give it a name
summary = profiles.groupby("cluster")[features + ["avg_addons", "late_return_rate", "total_spend"]].mean()
summary.insert(0, "customers", profiles["cluster"].value_counts().sort_index())
names = {}
names[summary["avg_daily_rate"].idxmax()] = "Premium renters"
names[summary["corporate_share"].idxmax()] = "Business renters"
names[summary["avg_rental_days"].idxmax()] = "Family road trippers"
for c in summary.index:
    names.setdefault(c, "Young budget renters")
summary.insert(0, "segment", summary.index.map(names))
profiles["segment"] = profiles["cluster"].map(names)
print("\n=== Cluster profiles (mean values) ===")
print(summary.round(2).T.to_string())

centres = pd.DataFrame(StandardScaler().fit(X).inverse_transform(kmeans.cluster_centers_),
                       columns=features).round(2)
centres.index.name = "cluster"
print("\nCluster centres in original units:")
print(centres.to_string())

# Step 8: visualise
colours = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]
plt.figure(figsize=(9, 5.5))
for c in range(K):
    part = profiles[profiles["cluster"] == c]
    plt.scatter(part["avg_rental_days"], part["avg_daily_rate"], s=22, alpha=0.7,
                color=colours[c], label=f"{c}: {names[c]}")
plt.xlabel("Average rental length (days)")
plt.ylabel("Average daily rate (INR)")
plt.title("Customer Clusters: Rental Length vs Daily Rate")
plt.legend()
plt.tight_layout()
plt.savefig("figures/fig2_clusters_days_vs_rate.png", dpi=150)
plt.close()

pca = PCA(n_components=2, random_state=42)
pts = pca.fit_transform(X_scaled)
plt.figure(figsize=(9, 5.5))
for c in range(K):
    m = profiles["cluster"].to_numpy() == c
    plt.scatter(pts[m, 0], pts[m, 1], s=22, alpha=0.7, color=colours[c], label=f"{c}: {names[c]}")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} of variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} of variance)")
plt.title("Customer Clusters on the First Two Principal Components")
plt.legend()
plt.tight_layout()
plt.savefig("figures/fig3_clusters_pca.png", dpi=150)
plt.close()

plt.figure(figsize=(9, 4.2))
(summary["late_return_rate"] * 100).plot(kind="bar", color=[colours[c] for c in summary.index], rot=0)
plt.xticks(range(K), [f"{c}: {names[c]}" for c in summary.index], fontsize=9)
plt.ylabel("Late returns (%)")
plt.title("Late Return Rate by Customer Segment")
plt.tight_layout()
plt.savefig("figures/fig4_late_rate_by_segment.png", dpi=150)
plt.close()

profiles.to_csv("customer_segments.csv", index=False)
print("\nSaved customer_profiles.csv, customer_segments.csv and 4 figures")
