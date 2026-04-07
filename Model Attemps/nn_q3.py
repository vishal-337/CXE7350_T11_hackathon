""""
# MLP experiment aligned to the disagreement_query.py
# feature pipeline. Uses:
#   - train.csv
#   - queryGroundTruth1.csv
#   - queries/query2/q2_results.csv
#   - queries/query3/q3_results.csv
#
# Goal:
#   Try a lightweight neural-network baseline on the same
#   mutation encoding used in the ensemble/disagreement script.
"""

import os
import random
import math
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------
# Load WT sequence
# ------------------------------------------------------------
with open("../Hackathon_data/sequence.fasta", "r") as f:
    sequence_wt = f.readlines()[1].strip()

SEQ_LEN = len(sequence_wt)


# ------------------------------------------------------------
# Same feature encoding as disagreement_query.py
# ------------------------------------------------------------
amino_acids = "ACDEFGHIKLMNPQRSTVWY"
aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}

blosum62_raw = {
    ('W', 'F'): 1, ('L', 'R'): -2, ('S', 'P'): -1, ('V', 'T'): 0, ('Q', 'Q'): 5,
    ('N', 'A'): -2, ('Z', 'Y'): -2, ('W', 'R'): -3, ('Q', 'A'): -1, ('S', 'D'): 0,
    ('H', 'A'): -2, ('S', 'H'): -1, ('H', 'D'): -1, ('L', 'N'): -3, ('W', 'A'): -3,
    ('Y', 'M'): -1, ('G', 'R'): -2, ('Y', 'I'): -1, ('Y', 'E'): -2, ('B', 'Y'): -3,
    ('Y', 'A'): -2, ('V', 'D'): -3, ('B', 'S'): 0, ('Y', 'Y'): 7, ('G', 'N'): 0,
    ('E', 'C'): -4, ('Y', 'Q'): -1, ('Z', 'Z'): 4, ('V', 'A'): 0, ('C', 'C'): 9,
    ('M', 'R'): -1, ('V', 'E'): -2, ('T', 'N'): 0, ('P', 'P'): 7, ('V', 'I'): 3,
    ('V', 'S'): -2, ('Z', 'P'): -1, ('V', 'M'): 1, ('T', 'F'): -2, ('V', 'Q'): -2,
    ('K', 'K'): 5, ('P', 'D'): -1, ('I', 'H'): -3, ('I', 'D'): -3, ('T', 'R'): -1,
    ('P', 'L'): -3, ('K', 'G'): -2, ('M', 'N'): -2, ('P', 'H'): -2, ('F', 'Q'): -3,
    ('Z', 'G'): -2, ('X', 'L'): -1, ('T', 'M'): -1, ('Z', 'C'): -3, ('X', 'H'): -1,
    ('D', 'R'): -2, ('B', 'W'): -4, ('X', 'D'): -1, ('Z', 'T'): -1, ('F', 'A'): -2,
    ('Z', 'W'): -3, ('F', 'E'): -3, ('D', 'N'): 1, ('B', 'K'): 0, ('X', 'P'): -2,
    ('D', 'J'): -3, ('X', 'T'): 0, ('Z', 'S'): 0, ('F', 'I'): 0, ('Z', 'O'): -1,
    ('F', 'M'): 0, ('J', 'H'): -3, ('Z', 'K'): 1, ('J', 'D'): -3, ('F', 'Y'): 3,
    ('X', 'R'): -1, ('Z', 'Q'): 3, ('X', 'F'): -1, ('X', 'B'): -1, ('J', 'N'): -3,
    ('X', 'Z'): -1, ('X', 'Y'): -1, ('X', 'W'): -2, ('X', 'V'): -1, ('F', 'C'): -2,
    ('X', 'Q'): -1, ('X', 'N'): -1, ('J', 'A'): -3, ('X', 'J'): -1, ('K', 'R'): 2,
    ('B', 'F'): -3, ('V', 'N'): -3, ('P', 'C'): -3, ('Y', 'H'): 2, ('Y', 'D'): -3,
    ('Y', 'L'): -1, ('G', 'A'): 0, ('P', 'O'): -2, ('Y', 'C'): -2, ('P', 'E'): -1,
    ('P', 'A'): -1, ('N', 'N'): 6, ('Y', 'W'): 2, ('Y', 'S'): -2, ('Y', 'T'): -2,
    ('B', 'D'): 4, ('B', 'H'): 0, ('B', 'L'): -4, ('W', 'N'): -4, ('Y', 'R'): -2,
    ('B', 'P'): -2, ('G', 'E'): -2, ('B', 'T'): -1, ('W', 'J'): -3, ('G', 'I'): -4,
    ('G', 'M'): -3, ('W', 'B'): -4, ('G', 'Q'): -2, ('G', 'Y'): -3, ('G', 'U'): -3,
    ('G', 'W'): -2, ('Y', 'V'): -1, ('E', 'R'): 0, ('B', 'X'): -1, ('G', 'C'): -3,
    ('E', 'N'): 0, ('W', 'V'): -3, ('E', 'A'): -1, ('W', 'Z'): -3, ('E', 'J'): -3,
    ('T', 'A'): 0, ('T', 'E'): -1, ('S', 'R'): -1, ('T', 'I'): -1, ('T', 'C'): -1,
    ('T', 'O'): -1, ('T', 'Y'): -2, ('T', 'W'): -2, ('T', 'V'): 0, ('T', 'Q'): -1,
    ('B', 'E'): 1, ('B', 'A'): -2, ('B', 'I'): -3, ('S', 'N'): 1, ('S', 'J'): -2,
    ('B', 'M'): -3, ('T', 'K'): -1, ('B', 'Q'): 0, ('B', 'U'): -4, ('I', 'R'): -3,
    ('S', 'A'): 1, ('S', 'F'): -2, ('B', 'C'): -3, ('I', 'N'): -3, ('S', 'B'): 0,
    ('I', 'A'): -1, ('S', 'M'): -1, ('S', 'Z'): 0, ('I', 'J'): 3, ('S', 'U'): -3,
    ('S', 'Y'): -2, ('S', 'Q'): 0, ('I', 'C'): -1, ('I', 'W'): -3, ('I', 'U'): -3,
    ('I', 'Y'): -1, ('I', 'Q'): -3, ('L', 'F'): 0, ('W', 'S'): -3, ('M', 'F'): 0,
    ('L', 'B'): -4, ('W', 'O'): -2, ('M', 'B'): -3, ('L', 'J'): 2, ('W', 'K'): -3,
    ('M', 'J'): 2, ('W', 'H'): -2, ('W', 'D'): -4, ('W', 'L'): -2, ('M', 'A'): -1,
    ('L', 'Z'): -3, ('L', 'V'): 1, ('L', 'W'): -2, ('M', 'E'): -2, ('M', 'I'): 1,
    ('L', 'Q'): -2, ('M', 'C'): -1, ('W', 'P'): -4, ('M', 'U'): -2, ('W', 'T'): -2,
    ('M', 'Y'): -1, ('M', 'W'): -1, ('M', 'Q'): 0
}
blosum62 = {}
for (a, b), v in blosum62_raw.items():
    blosum62[(a, b)] = v
    blosum62[(b, a)] = v

hydro = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8,
    'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}
vol = {
    'A': 88.6, 'R': 173.4, 'N': 114.1, 'D': 111.1, 'C': 108.5, 'Q': 143.8, 'E': 138.4,
    'G': 60.1, 'H': 153.2, 'I': 166.7, 'L': 166.7, 'K': 168.6, 'M': 162.9, 'F': 189.9,
    'P': 112.7, 'S': 89.0, 'T': 116.1, 'W': 227.8, 'Y': 193.6, 'V': 140.0
}
charge = {
    'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0, 'Q': 0, 'E': -1, 'G': 0, 'H': 0.1,
    'I': 0, 'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0, 'S': 0, 'T': 0, 'W': 0,
    'Y': 0, 'V': 0
}


def encode_mutation(mut, seq_length=SEQ_LEN):
    wt_aa, pos, mut_aa = mut[0], int(mut[1:-1]), mut[-1]

    vec = [pos / seq_length]

    mut_oh = [0] * 20
    if mut_aa in aa_to_idx:
        mut_oh[aa_to_idx[mut_aa]] = 1
    vec.extend(mut_oh)

    vec.append(blosum62.get((wt_aa, mut_aa), 0))
    vec.append(hydro.get(mut_aa, 0) - hydro.get(wt_aa, 0))
    vec.append(vol.get(mut_aa, 0) - vol.get(wt_aa, 0))
    vec.append(charge.get(mut_aa, 0) - charge.get(wt_aa, 0))

    return np.array(vec, dtype=np.float32)


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------
class MutationDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------
class SimpleMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


# ------------------------------------------------------------
# Train / eval helpers
# ------------------------------------------------------------
def evaluate_spearman(model, loader, device):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb).cpu().numpy()
            preds.extend(out.tolist())
            targets.extend(yb.numpy().tolist())

    rho = spearmanr(targets, preds).correlation
    if rho is None or math.isnan(rho):
        return 0.0
    return float(rho)


def predict(model, loader, device):
    model.eval()
    preds = []

    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            out = model(xb).cpu().numpy()
            preds.extend(out.tolist())

    return np.array(preds)


def train_model(model, train_loader, val_loader, device, epochs=100, lr=1e-3, patience=15):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_state = None
    best_rho = -1e9
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        val_rho = evaluate_spearman(model, val_loader, device)
        train_loss = np.mean(losses) if losses else 0.0
        print(f"Epoch {epoch:03d} | Loss {train_loss:.4f} | Val Spearman {val_rho:.4f}")

        if val_rho > best_rho:
            best_rho = val_rho
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"Best validation Spearman: {best_rho:.4f}")
    return model


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    set_seed(42)

    df_train = pd.read_csv("../Hackathon_data/train.csv")[["mutant", "DMS_score"]]
    df_query1 = pd.read_csv("../Hackathon_data/queryGroundTruth1.csv")[["mutant", "DMS_score"]]
    df_query2 = pd.read_csv("../queries/query2/q2_results.csv")[["mutant", "DMS_score"]]
    df_query3 = pd.read_csv("../queries/query3/q3_results.csv")[["mutant", "DMS_score"]]
    df_test = pd.read_csv("../Hackathon_data/test.csv")[["mutant"]]

    df_train_full = pd.concat(
        [df_train, df_query1, df_query2, df_query3],
        ignore_index=True
    ).drop_duplicates(subset=["mutant"], keep="last")

    print(f"Train rows: {len(df_train)}")
    print(f"+ Query1:    {len(df_query1)}")
    print(f"+ Query2:    {len(df_query2)}")
    print(f"+ Query3:    {len(df_query3)}")
    print(f"= Total:     {len(df_train_full)}")

    X = np.stack([encode_mutation(m) for m in df_train_full["mutant"].values])
    y = df_train_full["DMS_score"].values.astype(np.float32)
    X_test = np.stack([encode_mutation(m) for m in df_test["mutant"].values])

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    train_ds = MutationDataset(X_tr, y_tr)
    val_ds = MutationDataset(X_val, y_val)
    test_ds = MutationDataset(X_test_scaled)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SimpleMLP(input_dim=X_tr.shape[1]).to(device)
    model = train_model(model, train_loader, val_loader, device)

    final_val_rho = evaluate_spearman(model, val_loader, device)
    print(f"Final validation Spearman: {final_val_rho:.4f}")

    test_preds = predict(model, test_loader, device)

    out_df = pd.DataFrame({
        "id": np.arange(len(df_test)),
        "mutant": df_test["mutant"].values,
        "DMS_score": test_preds
    })

    os.makedirs("../predictions/mlp_query3", exist_ok=True)
    out_path = "../predictions/mlp_query3/predictions.csv"
    out_df[["id", "DMS_score"]].to_csv(out_path, index=False)

    print(f"Saved predictions to: {out_path}")

    top10 = out_df.sort_values("DMS_score", ascending=False).head(10)
    print("\nTop 10 predicted mutations:")
    for _, row in top10.iterrows():
        print(f"{row['mutant']:10s} {row['DMS_score']:.4f}")

    with open("../predictions/mlp_query3/top10.txt", "w") as f:
        for m in top10["mutant"].values:
            f.write(m + "\n")

    print("Saved top10.txt")


if __name__ == "__main__":
    main()