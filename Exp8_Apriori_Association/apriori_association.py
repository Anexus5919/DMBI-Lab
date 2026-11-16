"""
Experiment 8: Association rule mining with the Apriori algorithm in Python.

Each rental with at least one add-on (GPS, child seat, FASTag, chauffeur and
so on) is treated as a shopping basket. Apriori finds add-ons that are often
taken together, which the company can turn into bundle offers.

Input : rental_car_clean.csv (output of Experiment 2)
Output: rental_addon_transactions.csv (also used in Experiment 9), figures/
"""
import os
from itertools import combinations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

pd.set_option("display.width", 120)
os.makedirs("figures", exist_ok=True)

MIN_SUPPORT = 0.05
MIN_CONFIDENCE = 0.50

# Step 1: build the transactions
rentals = pd.read_csv("rental_car_clean.csv", keep_default_na=False)
baskets = rentals.loc[rentals["addons"] != "", "addons"].str.split(";")
print("Rentals:", len(rentals), "| Rentals with at least one add-on:", len(baskets))
print("\nFirst five baskets:")
for rid, items in zip(rentals.loc[baskets.index[:5], "rental_id"], baskets[:5]):
    print(f"  {rid}: {items}")

# Step 2: one-hot transaction matrix (1 = add-on taken in that rental)
matrix = baskets.str.join(";").str.get_dummies(sep=";")
matrix = matrix[sorted(matrix.columns)]
matrix.to_csv("rental_addon_transactions.csv", index=False)
print("\nTransaction matrix shape:", matrix.shape)
print(matrix.head().to_string())

# Step 3: support of each single add-on
freq = pd.DataFrame({"count": matrix.sum(), "support": (matrix.mean()).round(4)})
freq = freq.sort_values("count", ascending=False)
print("\n=== Add-on frequency ===")
print(freq.to_string())

plt.figure(figsize=(9, 4.5))
plt.barh(freq.index[::-1], freq["support"][::-1] * 100, color="#4c72b0")
plt.axvline(MIN_SUPPORT * 100, color="#c44e52", linestyle="--", label="Minimum support (5%)")
plt.xlabel("Support (% of baskets)")
plt.title("How Often Each Add-on Is Taken")
plt.legend()
plt.tight_layout()
plt.savefig("figures/fig1_addon_support.png", dpi=150)
plt.close()


# Step 4: Apriori written by hand, to see the level-wise search
def my_apriori(tx, min_support):
    n = len(tx)
    sets = [frozenset(t) for t in tx]
    current = [frozenset([i]) for i in sorted({i for t in sets for i in t})]
    frequent, level = {}, 1
    while current:
        counts = {c: sum(1 for t in sets if c <= t) for c in current}
        kept = {c: cnt / n for c, cnt in counts.items() if cnt / n >= min_support}
        print(f"  level {level}: {len(current)} candidates, {len(kept)} frequent")
        frequent.update(kept)
        # join step: combine frequent k-itemsets that share k-1 items
        keys = list(kept)
        candidates = {a | b for a, b in combinations(keys, 2) if len(a | b) == level + 1}
        # prune step: every subset of a candidate must already be frequent
        current = [c for c in candidates
                   if all(frozenset(s) in kept for s in combinations(c, level))]
        level += 1
    return frequent


print("\n=== Apriori (own implementation) ===")
mine = my_apriori(baskets.tolist(), MIN_SUPPORT)
print("Frequent itemsets found:", len(mine))

# Step 5: Apriori with mlxtend
frequent = apriori(matrix.astype(bool), min_support=MIN_SUPPORT, use_colnames=True)
frequent["length"] = frequent["itemsets"].apply(len)
print("\n=== Apriori (mlxtend) ===")
print("Frequent itemsets found:", len(frequent))
print("Same itemsets as own implementation:", set(frequent["itemsets"]) == set(mine))
print(frequent["length"].value_counts().sort_index().rename("itemsets by size").to_string())
print("\nFrequent itemsets with two or more add-ons:")
print(frequent[frequent["length"] >= 2].sort_values("support", ascending=False)
      .assign(itemsets=lambda d: d["itemsets"].apply(lambda s: ", ".join(sorted(s))))
      [["itemsets", "support"]].round(4).to_string(index=False))

# Step 6: association rules
rules = association_rules(frequent, metric="confidence", min_threshold=MIN_CONFIDENCE)
rules = rules[rules["lift"] > 1].sort_values(["lift", "confidence"], ascending=False)
rules["rule"] = (rules["antecedents"].apply(lambda s: ", ".join(sorted(s))) + "  ->  "
                 + rules["consequents"].apply(lambda s: ", ".join(sorted(s))))
print(f"\n=== Association rules (confidence >= {MIN_CONFIDENCE}, lift > 1) ===")
print("Number of rules:", len(rules))
print(rules[["rule", "support", "confidence", "lift"]].round(4).to_string(index=False))
rules[["rule", "support", "confidence", "lift", "leverage", "conviction"]].round(4) \
    .to_csv("association_rules.csv", index=False)

# Step 7: plots of the rules
top = rules.head(10).iloc[::-1]
plt.figure(figsize=(10, 5))
plt.barh(top["rule"], top["lift"], color="#55a868")
plt.xlabel("Lift")
plt.title("Top Association Rules by Lift")
plt.tight_layout()
plt.savefig("figures/fig2_top_rules_lift.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
sc = plt.scatter(rules["support"], rules["confidence"], c=rules["lift"], cmap="viridis",
                 s=rules["lift"] * 40, alpha=0.8)
plt.colorbar(sc, label="Lift")
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.title("Rules: Support vs Confidence (size and colour = lift)")
plt.tight_layout()
plt.savefig("figures/fig3_rules_scatter.png", dpi=150)
plt.close()

best = rules.iloc[0]
print("\nStrongest rule by lift:", best["rule"])
print(f"  support {best['support']:.4f}, confidence {best['confidence']:.4f}, lift {best['lift']:.4f}")
