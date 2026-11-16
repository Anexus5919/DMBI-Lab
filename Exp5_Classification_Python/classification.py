"""
Experiment 5: Decision Tree and Naive Bayes classification in Python, first
with scikit-learn and then with functions written from scratch.

Task: predict whether a rental will be returned late (late_return = Yes/No)
using details that are known when the booking is made.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

os.makedirs("figures", exist_ok=True)

# 1. Load the cleaned data from Experiment 2
df = pd.read_csv("rental_car_clean.csv")
print("Dataset shape:", df.shape)
print("\nTarget distribution:")
print(df["late_return"].value_counts().to_string())

# 2. Select features that are known at booking time
num_features = ["customer_age", "rental_days", "km_driven", "daily_rate"]
cat_features = ["car_type", "fuel_type", "booking_channel", "trip_type", "return_weekend", "season"]
features = num_features + cat_features

X = df[features].copy()
encoders = {}
for col in cat_features:
    encoders[col] = LabelEncoder().fit(X[col])
    X[col] = encoders[col].transform(X[col])
y = (df["late_return"] == "Yes").astype(int).to_numpy()   # 1 = late, 0 = on time

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTraining rows: {len(X_train)} | Testing rows: {len(X_test)}")

results = []


def report(name, y_pred):
    acc = accuracy_score(y_test, y_pred)
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=["On time", "Late"], digits=4))
    results.append({
        "Model": name,
        "Accuracy": round(acc * 100, 2),
        "Precision (Late)": round(precision_score(y_test, y_pred) * 100, 2),
        "Recall (Late)": round(recall_score(y_test, y_pred) * 100, 2),
        "F1 (Late)": round(f1_score(y_test, y_pred) * 100, 2),
    })
    return confusion_matrix(y_test, y_pred)


# ---------------------------------------------------------------------------
# Method 1: Decision Tree
# ---------------------------------------------------------------------------
# 3. Inbuilt Decision Tree (CART with Gini index)
dt = DecisionTreeClassifier(criterion="gini", max_depth=5, min_samples_leaf=10, random_state=42)
dt.fit(X_train, y_train)
cm_dt = report("Decision Tree (scikit-learn)", dt.predict(X_test))

print("Tree rules (first 3 levels):")
print(export_text(dt, feature_names=features, max_depth=2))

plt.figure(figsize=(18, 7))
plot_tree(dt, feature_names=features, class_names=["On time", "Late"], max_depth=2,
          filled=True, rounded=True, fontsize=9)
plt.title("Decision Tree for Late Return (top 3 levels)")
plt.tight_layout()
plt.savefig("figures/fig1_decision_tree.png", dpi=150)
plt.close()


# 4. Decision Tree written from scratch
def gini(labels):
    if len(labels) == 0:
        return 0.0
    p = labels.mean()
    return 1.0 - p ** 2 - (1 - p) ** 2


def best_split(Xv, yv, min_leaf):
    """Try every feature and every threshold, return the split with the lowest weighted Gini."""
    n = len(yv)
    best = (None, None, gini(yv))   # feature index, threshold, impurity
    for j in range(Xv.shape[1]):
        order = np.argsort(Xv[:, j], kind="mergesort")
        xs, ys = Xv[order, j], yv[order]
        left_pos = np.cumsum(ys)[:-1]            # late count on the left for each cut
        left_n = np.arange(1, n)
        right_pos = ys.sum() - left_pos
        right_n = n - left_n
        pl, pr = left_pos / left_n, right_pos / right_n
        g = (left_n * (1 - pl ** 2 - (1 - pl) ** 2) + right_n * (1 - pr ** 2 - (1 - pr) ** 2)) / n
        valid = (xs[1:] != xs[:-1]) & (left_n >= min_leaf) & (right_n >= min_leaf)
        if not valid.any():
            continue
        g = np.where(valid, g, np.inf)
        i = int(np.argmin(g))
        if g[i] < best[2] - 1e-12:
            best = (j, (xs[i] + xs[i + 1]) / 2, g[i])
    return best


def build_tree(Xv, yv, depth=0, max_depth=5, min_leaf=10):
    node = {"prediction": int(yv.mean() >= 0.5), "samples": len(yv)}
    if depth == max_depth or len(np.unique(yv)) == 1:
        return node
    j, threshold, _ = best_split(Xv, yv, min_leaf)
    if j is None:
        return node
    mask = Xv[:, j] <= threshold
    node.update({"feature": j, "threshold": threshold,
                 "left": build_tree(Xv[mask], yv[mask], depth + 1, max_depth, min_leaf),
                 "right": build_tree(Xv[~mask], yv[~mask], depth + 1, max_depth, min_leaf)})
    return node


def predict_tree(node, row):
    while "feature" in node:
        node = node["left"] if row[node["feature"]] <= node["threshold"] else node["right"]
    return node["prediction"]


my_tree = build_tree(X_train.to_numpy(float), y_train, max_depth=5, min_leaf=10)
y_pred_my_tree = np.array([predict_tree(my_tree, r) for r in X_test.to_numpy(float)])
cm_my_dt = report("Decision Tree (custom function)", y_pred_my_tree)
print("Root split of the custom tree:",
      f"{features[my_tree['feature']]} <= {my_tree['threshold']:.2f}")
print("Predictions identical to scikit-learn tree:",
      f"{(y_pred_my_tree == dt.predict(X_test)).mean() * 100:.2f}% of test rows")


# ---------------------------------------------------------------------------
# Method 2: Naive Bayes
# ---------------------------------------------------------------------------
# 5. Inbuilt Gaussian Naive Bayes (treats every column as a normal distribution)
nb = GaussianNB()
nb.fit(X_train, y_train)
cm_nb = report("Naive Bayes (scikit-learn GaussianNB)", nb.predict(X_test))


# 6. Naive Bayes written from scratch
#    Numeric columns use a normal distribution, categorical columns use
#    counts with Laplace smoothing, which suits label-encoded categories better.
def fit_naive_bayes(Xdf, yv, num_cols, cat_cols, alpha=1.0):
    model = {"classes": np.unique(yv), "prior": {}, "num": {}, "cat": {}}
    for c in model["classes"]:
        part = Xdf[yv == c]
        model["prior"][c] = len(part) / len(Xdf)
        model["num"][c] = {col: (part[col].mean(), part[col].std(ddof=0) + 1e-9) for col in num_cols}
        model["cat"][c] = {}
        for col in cat_cols:
            levels = np.unique(Xdf[col])
            counts = part[col].value_counts()
            model["cat"][c][col] = {v: (counts.get(v, 0) + alpha) / (len(part) + alpha * len(levels))
                                    for v in levels}
    return model


def predict_naive_bayes(model, Xdf, num_cols, cat_cols):
    scores = []
    for c in model["classes"]:
        log_p = np.full(len(Xdf), np.log(model["prior"][c]))
        for col in num_cols:
            mu, sd = model["num"][c][col]
            x = Xdf[col].to_numpy(float)
            log_p += -0.5 * np.log(2 * np.pi * sd ** 2) - (x - mu) ** 2 / (2 * sd ** 2)
        for col in cat_cols:
            table = model["cat"][c][col]
            log_p += np.log(Xdf[col].map(table).fillna(1e-6).to_numpy(float))
        scores.append(log_p)
    return model["classes"][np.argmax(np.vstack(scores), axis=0)]


my_nb = fit_naive_bayes(X_train, y_train, num_features, cat_features)
y_pred_my_nb = predict_naive_bayes(my_nb, X_test, num_features, cat_features)
cm_my_nb = report("Naive Bayes (custom function)", y_pred_my_nb)

print("Prior probabilities learned by the custom model:",
      {("Late" if k else "On time"): round(v, 4) for k, v in my_nb["prior"].items()})
print("P(trip_type = Outstation | Late)    =",
      round(my_nb["cat"][1]["trip_type"][encoders["trip_type"].transform(["Outstation"])[0]], 4))
print("P(trip_type = Outstation | On time) =",
      round(my_nb["cat"][0]["trip_type"][encoders["trip_type"].transform(["Outstation"])[0]], 4))

# 7. Confusion matrices and summary
fig, axes = plt.subplots(2, 2, figsize=(10, 8.5))
titles = ["Decision Tree (scikit-learn)", "Decision Tree (custom)",
          "Naive Bayes (GaussianNB)", "Naive Bayes (custom)"]
for ax, cm, title in zip(axes.ravel(), [cm_dt, cm_my_dt, cm_nb, cm_my_nb], titles):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["On time", "Late"], yticklabels=["On time", "Late"])
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("figures/fig2_confusion_matrices.png", dpi=150)
plt.close()

summary = pd.DataFrame(results)
print("\n=== Summary on the 1,000 test rentals ===")
print(summary.to_string(index=False))
summary.to_csv("results_summary.csv", index=False)
