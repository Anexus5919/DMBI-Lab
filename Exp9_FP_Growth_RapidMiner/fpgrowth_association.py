"""
Experiment 9 (Python cross-check): FP-Growth on the add-on transactions.

The RapidMiner process (addon_fpgrowth.rmp) is the main part of this
experiment. This script runs FP-Growth on the same file with the same
thresholds, so the RapidMiner rules can be checked, and compares its speed
with Apriori.

Input: rental_addon_transactions.csv (made in Experiment 8)
"""
import time
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

pd.set_option("display.width", 120)
MIN_SUPPORT = 0.05
MIN_CONFIDENCE = 0.50

tx = pd.read_csv("rental_addon_transactions.csv").astype(bool)
print("Transactions:", tx.shape[0], "| Items:", tx.shape[1])

# Header table of the FP-tree: items in descending order of frequency
print("\nItem order used to build the FP-tree:")
print(tx.sum().sort_values(ascending=False).to_string())

start = time.perf_counter()
fp_sets = fpgrowth(tx, min_support=MIN_SUPPORT, use_colnames=True)
fp_time = time.perf_counter() - start

start = time.perf_counter()
ap_sets = apriori(tx, min_support=MIN_SUPPORT, use_colnames=True)
ap_time = time.perf_counter() - start

print(f"\nFP-Growth: {len(fp_sets)} frequent itemsets in {fp_time * 1000:.1f} ms")
print(f"Apriori  : {len(ap_sets)} frequent itemsets in {ap_time * 1000:.1f} ms")
print("Same itemsets:", set(fp_sets["itemsets"]) == set(ap_sets["itemsets"]))

rules = association_rules(fp_sets, metric="confidence", min_threshold=MIN_CONFIDENCE)
rules = rules[rules["lift"] > 1].sort_values("confidence", ascending=False)
rules["premises"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
rules["conclusion"] = rules["consequents"].apply(lambda s: ", ".join(sorted(s)))
print(f"\nRules with confidence >= {MIN_CONFIDENCE} and lift > 1: {len(rules)}")
print(rules[["premises", "conclusion", "support", "confidence", "lift"]].round(4).to_string(index=False))
