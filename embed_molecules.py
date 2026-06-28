"""
embed_molecules.py
Step 1: Download ESOL dataset and encode SMILES with ChemBERTa (GPU).
Run once — saves embeddings.npy and solubility.npy for the main experiment.

Author: Moirangthem Gelson Singh
"""

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import os, sys

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ─── Download ESOL Dataset ─────────────────────────────────────────────────────
ESOL_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
print(f"Downloading ESOL dataset ...")
try:
    df = pd.read_csv(ESOL_URL)
    print(f"  Loaded {len(df)} molecules from {ESOL_URL}")
except Exception as e:
    print(f"  Network error: {e}. Trying alternative URL ...")
    try:
        ALT_URL = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"
        df = pd.read_csv(ALT_URL)
        print(f"  Loaded {len(df)} molecules (alternative URL)")
    except Exception as e2:
        print(f"  Both URLs failed. Error: {e2}")
        sys.exit(1)

print(f"  Columns: {list(df.columns)}")

# Identify SMILES and solubility columns
smiles_col = [c for c in df.columns if "smiles" in c.lower() or "SMILES" in c][0]
sol_col    = [c for c in df.columns if "solubility" in c.lower() or "measured" in c.lower()][0]
print(f"  SMILES column    : '{smiles_col}'")
print(f"  Solubility column: '{sol_col}'")

smiles_list  = df[smiles_col].tolist()
solubility   = df[sol_col].values.astype(float)

# ─── ChemBERTa Encoding ────────────────────────────────────────────────────────
MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
print(f"\nLoading ChemBERTa tokenizer and model: {MODEL_NAME}")

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()
print(f"  Model loaded on {DEVICE}")

def encode_smiles_batch(smiles_batch, tokenizer, model, device, max_length=128):
    """Encode a list of SMILES strings into CLS embeddings."""
    inputs = tokenizer(
        smiles_batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # Use CLS token embedding (first token of last hidden state)
    cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return cls_embeddings

BATCH_SIZE = 64
embeddings_list = []

print(f"\nEncoding {len(smiles_list)} molecules (batch size={BATCH_SIZE}) ...")
for i in range(0, len(smiles_list), BATCH_SIZE):
    batch = smiles_list[i: i + BATCH_SIZE]
    try:
        emb = encode_smiles_batch(batch, tokenizer, model, DEVICE)
    except Exception as e:
        print(f"  Batch {i//BATCH_SIZE} failed ({e}), encoding individually ...")
        emb = np.vstack([
            encode_smiles_batch([s], tokenizer, model, DEVICE)
            for s in batch
        ])
    embeddings_list.append(emb)
    done = min(i + BATCH_SIZE, len(smiles_list))
    print(f"  [{done:4d}/{len(smiles_list)}] encoded", end="\r")

embeddings = np.vstack(embeddings_list)
print(f"\n  Embeddings shape: {embeddings.shape}")

# ─── Save ──────────────────────────────────────────────────────────────────────
np.save("embeddings.npy", embeddings)
np.save("solubility.npy", solubility)
df[[smiles_col, sol_col]].to_csv("molecules.csv", index=False)

print(f"\nSaved:")
print(f"  embeddings.npy  — {embeddings.shape} float32 array")
print(f"  solubility.npy  — {len(solubility)} solubility values")
print(f"  molecules.csv   — SMILES + solubility")
print(f"\nRun drug_discovery_poc.py next.")
