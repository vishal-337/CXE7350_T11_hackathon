"""
Disagreement-Based Active Learning Query Selection + Improved Predictions

1. Combines train.csv + queryGroundTruth1.csv + query2/q3 results CSVs
2. Trains 5 different models
3. Predicts test set with each
4. Picks 100 mutations where models disagree most → query2.txt
5. Outputs improved predictions.csv and top10.txt using ensemble average
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)

# ─── 1. Load data ───────────────────────────────────────────────────────────

with open("../Hackathon_data/sequence.fasta", "r") as f:
    sequence_wt = f.readlines()[1].strip()

df_train = pd.read_csv("../Hackathon_data/train.csv")
df_query1 = pd.read_csv("../Hackathon_data/queryGroundTruth1.csv")[["mutant", "DMS_score"]]
df_query2 = pd.read_csv("../queries/query2/q2_results.csv")[["mutant", "DMS_score"]]
df_query3 = pd.read_csv("../queries/query3/q3_results.csv")[["mutant", "DMS_score"]]
df_test = pd.read_csv("../Hackathon_data/test.csv")

df_train_full = pd.concat(
    [df_train, df_query1, df_query2, df_query3], ignore_index=True
).drop_duplicates(subset=["mutant"], keep="last")
print(
    f"Training size: {len(df_train)} + {len(df_query1)} query1 + {len(df_query2)} query2 "
    f"+ {len(df_query3)} query3 → {len(df_train_full)} after dedup"
)

# ─── 2. Feature encoding (same as MLB26_Hack.ipynb) ─────────────────────────

amino_acids = "ACDEFGHIKLMNPQRSTVWY"
aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}

blosum62_raw = {('W', 'F'): 1, ('L', 'R'): -2, ('S', 'P'): -1, ('V', 'T'): 0, ('Q', 'Q'): 5, ('N', 'A'): -2, ('Z', 'Y'): -2, ('W', 'R'): -3, ('Q', 'A'): -1, ('S', 'D'): 0, ('H', 'A'): -2, ('S', 'H'): -1, ('H', 'D'): -1, ('L', 'N'): -3, ('W', 'A'): -3, ('Y', 'M'): -1, ('G', 'R'): -2, ('Y', 'I'): -1, ('Y', 'E'): -2, ('B', 'Y'): -3, ('Y', 'A'): -2, ('V', 'D'): -3, ('B', 'S'): 0, ('Y', 'Y'): 7, ('G', 'N'): 0, ('E', 'C'): -4, ('Y', 'Q'): -1, ('Z', 'Z'): 4, ('V', 'A'): 0, ('C', 'C'): 9, ('M', 'R'): -1, ('V', 'E'): -2, ('T', 'N'): 0, ('P', 'P'): 7, ('V', 'I'): 3, ('V', 'S'): -2, ('Z', 'P'): -1, ('V', 'M'): 1, ('T', 'F'): -2, ('V', 'Q'): -2, ('K', 'K'): 5, ('P', 'D'): -1, ('I', 'H'): -3, ('I', 'D'): -3, ('T', 'R'): -1, ('P', 'L'): -3, ('K', 'G'): -2, ('M', 'N'): -2, ('P', 'H'): -2, ('F', 'Q'): -3, ('Z', 'G'): -2, ('X', 'L'): -1, ('T', 'M'): -1, ('Z', 'C'): -3, ('X', 'H'): -1, ('D', 'R'): -2, ('B', 'W'): -4, ('X', 'D'): -1, ('Z', 'T'): -1, ('F', 'A'): -2, ('Z', 'W'): -3, ('F', 'E'): -3, ('D', 'N'): 1, ('B', 'K'): 0, ('X', 'P'): -2, ('D', 'J'): -3, ('X', 'T'): 0, ('Z', 'S'): 0, ('F', 'I'): 0, ('Z', 'O'): -1, ('F', 'M'): 0, ('J', 'H'): -3, ('Z', 'K'): 1, ('J', 'D'): -3, ('F', 'Y'): 3, ('X', 'R'): -1, ('Z', 'Q'): 3, ('X', 'F'): -1, ('X', 'B'): -1, ('J', 'N'): -3, ('X', 'Z'): -1, ('X', 'Y'): -1, ('X', 'W'): -2, ('X', 'V'): -1, ('F', 'C'): -2, ('X', 'Q'): -1, ('X', 'N'): -1, ('J', 'A'): -3, ('X', 'J'): -1, ('K', 'R'): 2, ('B', 'F'): -3, ('V', 'N'): -3, ('P', 'C'): -3, ('Y', 'H'): 2, ('Y', 'D'): -3, ('Y', 'L'): -1, ('G', 'A'): 0, ('P', 'O'): -2, ('Y', 'C'): -2, ('P', 'E'): -1, ('P', 'A'): -1, ('N', 'N'): 6, ('Y', 'W'): 2, ('Y', 'S'): -2, ('Y', 'T'): -2, ('B', 'D'): 4, ('B', 'H'): 0, ('B', 'L'): -4, ('W', 'N'): -4, ('Y', 'R'): -2, ('B', 'P'): -2, ('G', 'E'): -2, ('B', 'T'): -1, ('W', 'J'): -3, ('G', 'I'): -4, ('W', 'F'): 1, ('G', 'M'): -3, ('W', 'B'): -4, ('G', 'Q'): -2, ('G', 'Y'): -3, ('G', 'U'): -3, ('G', 'W'): -2, ('Y', 'V'): -1, ('E', 'R'): 0, ('B', 'X'): -1, ('G', 'C'): -3, ('E', 'N'): 0, ('W', 'V'): -3, ('W', 'R'): -3, ('E', 'A'): -1, ('W', 'Z'): -3, ('E', 'J'): -3, ('T', 'A'): 0, ('T', 'E'): -1, ('S', 'R'): -1, ('T', 'I'): -1, ('T', 'C'): -1, ('T', 'O'): -1, ('T', 'Y'): -2, ('T', 'W'): -2, ('T', 'V'): 0, ('T', 'Q'): -1, ('B', 'E'): 1, ('B', 'A'): -2, ('B', 'I'): -3, ('S', 'N'): 1, ('S', 'J'): -2, ('B', 'M'): -3, ('T', 'K'): -1, ('B', 'Q'): 0, ('B', 'U'): -4, ('I', 'R'): -3, ('S', 'A'): 1, ('S', 'F'): -2, ('B', 'C'): -3, ('I', 'N'): -3, ('S', 'B'): 0, ('I', 'A'): -1, ('S', 'M'): -1, ('S', 'Z'): 0, ('I', 'J'): 3, ('S', 'U'): -3, ('S', 'Y'): -2, ('S', 'Q'): 0, ('I', 'C'): -1, ('I', 'W'): -3, ('I', 'U'): -3, ('I', 'Y'): -1, ('I', 'Q'): -3, ('L', 'F'): 0, ('W', 'S'): -3, ('M', 'F'): 0, ('L', 'B'): -4, ('W', 'O'): -2, ('M', 'B'): -3, ('L', 'J'): 2, ('W', 'K'): -3, ('M', 'J'): 2, ('W', 'H'): -2, ('W', 'D'): -4, ('W', 'L'): -2, ('M', 'A'): -1, ('L', 'Z'): -3, ('L', 'V'): 1, ('L', 'W'): -2, ('M', 'E'): -2, ('M', 'I'): 1, ('L', 'Q'): -2, ('M', 'C'): -1, ('W', 'P'): -4, ('M', 'U'): -2, ('W', 'T'): -2, ('M', 'Y'): -1, ('M', 'W'): -1, ('M', 'Q'): 0}
blosum62 = {}
for (k1, k2), v in blosum62_raw.items():
    blosum62[(k1, k2)] = v
    blosum62[(k2, k1)] = v

hydro = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2}
vol = {'A': 88.6, 'R': 173.4, 'N': 114.1, 'D': 111.1, 'C': 108.5, 'Q': 143.8, 'E': 138.4, 'G': 60.1, 'H': 153.2, 'I': 166.7, 'L': 166.7, 'K': 168.6, 'M': 162.9, 'F': 189.9, 'P': 112.7, 'S': 89.0, 'T': 116.1, 'W': 227.8, 'Y': 193.6, 'V': 140.0}
charge = {'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0, 'Q': 0, 'E': -1, 'G': 0, 'H': 0.1, 'I': 0, 'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0, 'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0}


def encode_mutation(mut, seq_length=656):
    wt_aa, pos, mut_aa = mut[0], int(mut[1:-1]), mut[-1]
    vec = [pos / seq_length]

    # Mutant one-hot
    mut_oh = [0] * 20
    if mut_aa in aa_to_idx:
        mut_oh[aa_to_idx[mut_aa]] = 1
    vec.extend(mut_oh)

    # Biological differences
    vec.append(blosum62.get((wt_aa, mut_aa), 0))
    vec.append(hydro.get(mut_aa, 0) - hydro.get(wt_aa, 0))
    vec.append(vol.get(mut_aa, 0) - vol.get(wt_aa, 0))
    vec.append(charge.get(mut_aa, 0) - charge.get(wt_aa, 0))

    return np.array(vec, dtype=np.float32)


# ─── 3. Encode all data ─────────────────────────────────────────────────────

X_train = np.stack([encode_mutation(m) for m in df_train_full["mutant"].values])
y_train = df_train_full["DMS_score"].values.astype(np.float32)

X_test = np.stack([encode_mutation(m) for m in df_test["mutant"].values])

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")

# ─── 4. Validation split to evaluate models ─────────────────────────────────

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=0
)

# ─── 5. Define and train 5 models ───────────────────────────────────────────

models = {
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        subsample=0.8, random_state=42,
    ),
    "HistGradientBoosting": HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_depth=4, random_state=42,
    ),
    "Ridge": Ridge(alpha=1.0),
    "ExtraTrees": ExtraTreesRegressor(
        n_estimators=300, max_depth=8, random_state=42,
    ),
    "RandomForest": RandomForestRegressor(
        n_estimators=300, max_depth=8, random_state=42,
    ),
}

# First train on split to check validation Spearman
print("\n── Validation scores (80/20 split) ──")
for name, model in models.items():
    model.fit(X_tr, y_tr)
    val_preds = model.predict(X_val)
    spearman = spearmanr(y_val, val_preds).correlation
    print(f"  {name:25s} Spearman: {spearman:.4f}")

# ─── 6. Retrain all models on FULL training data ────────────────────────────

print("\n── Retraining on full data (train + query1 + query2 + query3) ──")
test_predictions = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    test_predictions[name] = preds
    print(f"  {name} done")

# ─── 7. Compute disagreement (variance across models) ───────────────────────

all_preds = np.stack(list(test_predictions.values()))  # shape: (5, 11324)

# Normalize each model's predictions to 0-1 range so variance is comparable
all_preds_norm = np.zeros_like(all_preds)
for i in range(len(all_preds)):
    mn, mx = all_preds[i].min(), all_preds[i].max()
    all_preds_norm[i] = (all_preds[i] - mn) / (mx - mn)

variance = np.var(all_preds_norm, axis=0)  # shape: (11324,)

print(f"\nVariance stats:")
print(f"  Mean: {variance.mean():.6f}")
print(f"  Max:  {variance.max():.6f}")
print(f"  Min:  {variance.min():.6f}")

# ─── 8. Pick 100 with diversity + disagreement → query2.txt ──────────────────

# Exclude mutations already queried in query1 and query2
query1_mutants = set(df_query1["mutant"].values)
query2_mutants_prev = set(df_query2["mutant"].values)
query3_mutants_prev = set(df_query3["mutant"].values)
queried_mutants = query1_mutants | query2_mutants_prev | query3_mutants_prev
# Also exclude mutations at positions already in training data
train_positions = set(int(m[1:-1]) for m in df_train_full["mutant"].values)

test_mutants = df_test["mutant"].values
test_positions = np.array([int(m[1:-1]) for m in test_mutants])

# Mask out already-queried mutations
mask = np.array([m not in queried_mutants for m in test_mutants])
variance_masked = variance.copy()
variance_masked[~mask] = -1

# Strategy: divide protein into bins, pick top disagreement from each bin
# This ensures coverage across the whole protein
n_bins = 20  # 20 bins across 656 positions → ~33 positions per bin, 5 mutations per bin
bin_size = 656 / n_bins
query2_indices = []

print(f"\n── Query 2 (disagreement + diversity across {n_bins} bins) ──")

for b in range(n_bins):
    bin_start = b * bin_size
    bin_end = (b + 1) * bin_size

    # Find test mutations in this bin that haven't been queried
    in_bin = (test_positions >= bin_start) & (test_positions < bin_end) & mask

    if in_bin.sum() == 0:
        continue

    # Get variances for mutations in this bin
    bin_indices = np.where(in_bin)[0]
    bin_variances = variance[bin_indices]

    # Pick top 5 highest disagreement from this bin
    n_pick = min(5, len(bin_indices))
    top_in_bin = bin_indices[np.argsort(bin_variances)[-n_pick:][::-1]]
    query2_indices.extend(top_in_bin.tolist())

    print(f"  Bin {b:2d} (pos {int(bin_start):3d}-{int(bin_end):3d}): "
          f"picked {n_pick}, top variance {bin_variances.max():.6f}, "
          f"e.g. {test_mutants[top_in_bin[0]]}")

# If we have more than 100 (unlikely), trim by variance; if less, fill from remaining
if len(query2_indices) > 100:
    # Keep the 100 with highest variance
    idx_variances = [(i, variance[i]) for i in query2_indices]
    idx_variances.sort(key=lambda x: x[1], reverse=True)
    query2_indices = [i for i, v in idx_variances[:100]]
elif len(query2_indices) < 100:
    # Fill remaining from highest variance mutations not already selected
    remaining = 100 - len(query2_indices)
    selected = set(query2_indices)
    all_by_variance = np.argsort(variance_masked)[::-1]
    for idx in all_by_variance:
        if idx not in selected and mask[idx]:
            query2_indices.append(idx)
            selected.add(idx)
            if len(query2_indices) >= 100:
                break

query2_mutants = test_mutants[query2_indices]

# Show position distribution
q2_positions = [int(m[1:-1]) for m in query2_mutants]
print(f"\n  Total selected: {len(query2_mutants)}")
print(f"  Position range: {min(q2_positions)} to {max(q2_positions)}")
print(f"  Unique positions: {len(set(q2_positions))}")
print(f"  Variance range: {variance[query2_indices].min():.6f} to {variance[query2_indices].max():.6f}")

# Check how many are from positions NOT in training data (truly new info)
new_positions = [p for p in q2_positions if p not in train_positions]
print(f"  From unseen positions: {len(new_positions)}/100")

with open("../queries/query3/query3.txt", "w") as f:
    for m in query2_mutants:
        f.write(m + "\n")
print("  Saved to queries/query3/query3.txt")

# ─── 9. Ensemble prediction (average of top models, exclude Ridge) ───────────

# Use top 4 models (exclude Ridge) for final predictions, normalized average
model_names = list(models.keys())
top_model_idx = [i for i, name in enumerate(model_names) if name != "Ridge"]
print(f"\nEnsemble uses: {[model_names[i] for i in top_model_idx]}")
print(f"Excluded from ensemble (still used for disagreement): Ridge")

ensemble_preds = all_preds_norm[top_model_idx].mean(axis=0)

df_test["DMS_score_predicted"] = ensemble_preds

df_test[["mutant", "DMS_score_predicted"]].to_csv(
    "../predictions/ensemble/predictions.csv", index=False
)
print(f"\n── Ensemble predictions saved ──")

top10 = df_test.sort_values("DMS_score_predicted", ascending=False).head(10)
print(f"\n── Top 10 predicted mutations ──")
for _, row in top10.iterrows():
    print(f"  {row['mutant']:10s} score: {row['DMS_score_predicted']:.4f}")

with open("../predictions/ensemble/top10.txt", "w") as f:
    for m in top10["mutant"].values:
        f.write(m + "\n")
print("  Saved to predictions/ensemble/top10.txt")

# ─── 10. Also save individual model predictions for analysis ─────────────────

print(f"\n── Individual model prediction samples (first 5 test mutations) ──")
print(f"  {'Mutant':10s}", end="")
for name in models:
    print(f"  {name:>15s}", end="")
print(f"  {'Variance':>10s}")

for i in range(5):
    print(f"  {test_mutants[i]:10s}", end="")
    for j, name in enumerate(models):
        print(f"  {all_preds_norm[j][i]:15.4f}", end="")
    print(f"  {variance[i]:10.6f}")
