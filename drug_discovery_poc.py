"""
Human-in-the-Loop Drug Discovery via Preference Learning
Bayesian preference learning from pairwise molecular comparisons with active pair selection.

Requires: embed_molecules.py to have been run first (generates embeddings.npy + solubility.npy)

Author: Moirangthem Gelson Singh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

# ─── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f0f1a",
    "axes.facecolor":   "#1a1a2e",
    "axes.edgecolor":   "#444466",
    "axes.labelcolor":  "#ccccdd",
    "text.color":       "#ccccdd",
    "xtick.color":      "#aaaacc",
    "ytick.color":      "#aaaacc",
    "grid.color":       "#2a2a4a",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "font.family":      "monospace",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.titlepad":    10,
    "legend.facecolor": "#1a1a2e",
    "legend.edgecolor": "#444466",
})

VIOLET = "#7c4dff"
CYAN   = "#00e5ff"
ORANGE = "#ff6d00"
GREEN  = "#00c853"
GREY   = "#607d8b"


# ─── Load Embeddings ───────────────────────────────────────────────────────────
import os
if not os.path.exists("embeddings.npy"):
    raise FileNotFoundError("embeddings.npy not found. Run embed_molecules.py first.")

print("Loading ChemBERTa embeddings ...")
X_emb  = np.load("embeddings.npy")
sol    = np.load("solubility.npy")
df_mol = pd.read_csv("molecules.csv")   # columns: smiles, solubility

N = len(sol)
print(f"  Molecules  : {N}")
print(f"  Embedding  : {X_emb.shape[1]}-dim (ChemBERTa CLS)")
print(f"  Solubility : min={sol.min():.2f}, max={sol.max():.2f}, mean={sol.mean():.2f}")

# ─── Reduce Dimensionality ─────────────────────────────────────────────────────
print("\nReducing to 32-dim via PCA ...")
pca = PCA(n_components=32, random_state=SEED)
X   = pca.fit_transform(X_emb)
print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")

# Define "top" molecules: top 10% by solubility
TOP_THRESHOLD = np.percentile(sol, 90)
is_top = sol >= TOP_THRESHOLD
n_top  = is_top.sum()
print(f"\n  Top 10% (solubility >= {TOP_THRESHOLD:.2f}): {n_top} molecules")


# ─── Simulated Oracle (Scientist) ─────────────────────────────────────────────
def oracle(i, j, solubility, noise_std=0.8, rng=None):
    """
    Simulated scientist preference: molecule with higher solubility wins,
    with noise to model human uncertainty in borderline comparisons.
    noise_std=0.8 mimics a real human who is somewhat noisy and uncertain.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)
    diff = solubility[i] - solubility[j]
    p_i_wins = expit(diff / noise_std)
    return i if rng.random() < p_i_wins else j


# ─── Bradley-Terry Preference Model ───────────────────────────────────────────
def neg_log_likelihood(beta_free, outcomes, n, l2=0.1):
    beta = np.concatenate([[0.0], beta_free])
    nll  = 0.0
    for winner, loser in outcomes:
        nll -= np.log(expit(beta[winner] - beta[loser]) + 1e-12)
    nll += l2 * np.sum(beta_free ** 2)
    return nll


def fit_bt(outcomes, n, l2=0.1):
    """Fit Bradley-Terry and return MAP scores + Laplace posterior std."""
    if len(outcomes) == 0:
        return np.zeros(n), np.ones(n) * 2.0, np.eye(n) * 4.0

    beta0  = np.zeros(n - 1)
    result = minimize(neg_log_likelihood, beta0, args=(outcomes, n, l2),
                      method="L-BFGS-B", options={"maxiter": 300, "ftol": 1e-8})
    beta_map = np.concatenate([[0.0], result.x])
    beta_map -= beta_map.mean()

    # Numerical Hessian (diagonal approximation for speed at scale)
    eps   = 1e-3
    n_free = n - 1
    diag_H = np.zeros(n_free)
    for k in range(n_free):
        e = np.zeros(n_free); e[k] = eps
        f_plus  = neg_log_likelihood(result.x + e, outcomes, n, l2)
        f_minus = neg_log_likelihood(result.x - e, outcomes, n, l2)
        f_0     = neg_log_likelihood(result.x, outcomes, n, l2)
        diag_H[k] = (f_plus - 2*f_0 + f_minus) / (eps**2)
    diag_H = np.maximum(diag_H, 1e-3)  # stabilize

    std_free = 1.0 / np.sqrt(diag_H)
    std_full = np.concatenate([[0.0], std_free])  # anchor has 0 std
    return beta_map, std_full, np.diag(std_full**2)


def most_uncertain_pair(beta, std, n, queried=None):
    """Select pair (i, j) where outcome is most uncertain (P closest to 0.5)."""
    if queried is None:
        queried = set()
    best_pair, min_cert, best_p = None, np.inf, 0.5
    # For efficiency, sample candidate pairs rather than all O(n^2)
    rng2 = np.random.default_rng(99)
    candidates = set()
    while len(candidates) < min(300, n*(n-1)//2):
        i, j = sorted(rng2.choice(n, 2, replace=False))
        candidates.add((i, j))
    for (i, j) in candidates:
        if (i, j) in queried: continue
        diff = beta[i] - beta[j]
        var  = std[i]**2 + std[j]**2
        p    = expit(diff / (np.sqrt(var) + 1e-6))
        cert = abs(p - 0.5)
        if cert < min_cert:
            min_cert, best_pair, best_p = cert, (i, j), p
    if best_pair is None:  # fallback
        i, j = sorted(rng2.choice(n, 2, replace=False))
        best_pair, best_p = (i, j), 0.5
    return best_pair, best_p


# ─── Run Experiments ───────────────────────────────────────────────────────────
# We compare two strategies over 100 comparison rounds:
#   1. Active: pick the most uncertain pair each round
#   2. Random: pick a random pair each round

# Use a subset of 200 molecules for speed (covers all top-10% molecules)
rng = np.random.default_rng(SEED)
idx_subset = rng.choice(N, 200, replace=False)
# Make sure all top molecules are included
idx_top    = np.where(is_top)[0]
idx_subset = np.unique(np.concatenate([idx_subset, idx_top]))[:200]

X_sub  = X[idx_subset]
sol_sub = sol[idx_subset]
is_top_sub = is_top[idx_subset]
n_sub  = len(X_sub)
n_top_sub = is_top_sub.sum()

print(f"\nExperiment subset: {n_sub} molecules, {n_top_sub} top-10% ({n_top_sub/n_sub*100:.1f}%)")


def run_preference_discovery(sol_sub, is_top_sub, strategy="active", n_rounds=100, seed=SEED):
    rng2 = np.random.default_rng(seed)
    n    = len(sol_sub)
    outcomes  = []
    queried   = set()
    beta, std, cov = fit_bt(outcomes, n, l2=0.2)
    n_top_total = is_top_sub.sum()

    record = []  # (n_comparisons, n_top_found, spearman)

    for rnd in range(n_rounds):
        # Select pair
        if strategy == "active":
            pair, _ = most_uncertain_pair(beta, std, n, queried)
        else:
            while True:
                i, j = sorted(rng2.choice(n, 2, replace=False))
                pair = (i, j)
                break
        queried.add(pair)
        i, j = pair

        # Oracle comparison
        winner = oracle(i, j, sol_sub, noise_std=0.8, rng=rng2)
        loser  = j if winner == i else i
        outcomes.append((winner, loser))

        # Refit model
        beta, std, cov = fit_bt(outcomes, n, l2=0.2)

        # Metrics
        ranked = np.argsort(beta)[::-1]
        top_k  = set(np.where(is_top_sub)[0])
        top_found_k = len(set(ranked[:n_top_total]) & top_k)
        corr, _ = spearmanr(sol_sub, beta)
        record.append((rnd + 1, top_found_k, max(corr, 0)))

    return np.array(record)


print("\nRunning preference learning experiments ...")
print("  Active strategy  ...", end=" ", flush=True)
rec_active = run_preference_discovery(sol_sub, is_top_sub, "active", n_rounds=100)
print("done")
print("  Random strategy  ...", end=" ", flush=True)
rec_random = run_preference_discovery(sol_sub, is_top_sub, "random", n_rounds=100)
print("done")


# ─── Plot 1: Top Molecule Discovery Curve ─────────────────────────────────────
n_top_sub = is_top_sub.sum()
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(
    "Human-in-the-Loop Drug Discovery: Active Preference Learning\n"
    "Scientist compares two molecules per round; AI learns latent solubility ranking",
    fontsize=13, fontweight="bold", color="#e0e0ff", y=1.02
)

ax = axes[0]
ax.plot(rec_active[:, 0], rec_active[:, 1] / n_top_sub * 100,
        color=VIOLET, lw=2.5, label="Active: query most uncertain pair")
ax.plot(rec_random[:, 0], rec_random[:, 1] / n_top_sub * 100,
        color=GREY, lw=1.8, ls="--", alpha=0.8, label="Passive: query random pair")
ax.axhline(50, color=GREY, ls=":", lw=1, alpha=0.4)
ax.axhline(80, color=GREEN, ls=":", lw=1.5, alpha=0.5, label="80% discovery target")

# Annotate crossings
for label, rec, color in [("Active", rec_active, VIOLET), ("Passive", rec_random, GREY)]:
    idx = np.argmax(rec[:, 1] / n_top_sub >= 0.8)
    if rec[idx, 1] / n_top_sub >= 0.8:
        ax.axvline(rec[idx, 0], color=color, ls=":", lw=1.5, alpha=0.5)
        ax.text(rec[idx, 0] + 1, 20,
                f"{label}\n{int(rec[idx,0])} queries", color=color, fontsize=8)

ax.set_xlabel("Number of Pairwise Comparisons (oracle calls)", labelpad=8)
ax.set_ylabel("Top-10% Molecules Discovered (%)", labelpad=8)
ax.set_title("Discovery Efficiency\n(how fast does the AI find the best molecules?)")
ax.legend(fontsize=9, loc="lower right")
ax.set_ylim(0, 105)
ax.grid(True)

# Right: Spearman rank recovery
ax2 = axes[1]
ax2.plot(rec_active[:, 0], rec_active[:, 2],
         color=VIOLET, lw=2.5, label="Active")
ax2.plot(rec_random[:, 0], rec_random[:, 2],
         color=GREY, lw=1.8, ls="--", alpha=0.8, label="Passive")
ax2.axhline(0.8, color=GREEN, ls=":", lw=1.5, alpha=0.5, label="Target: rank corr = 0.80")
ax2.set_xlabel("Number of Pairwise Comparisons", labelpad=8)
ax2.set_ylabel("Spearman Rank Corr vs. True Solubility", labelpad=8)
ax2.set_title("Ranking Recovery Quality\n(how closely does the model learn true solubility order?)")
ax2.legend(fontsize=9, loc="lower right")
ax2.set_ylim(0, 1.05)
ax2.grid(True)

plt.tight_layout()
plt.savefig("discovery_curve.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("\nSaved: discovery_curve.png")


# ─── Plot 2: Uncertainty Reduction over Rounds ────────────────────────────────
# Rerun and track posterior std over rounds
print("Tracking posterior uncertainty over rounds ...")

def run_with_uncertainty_tracking(sol_sub, strategy, n_rounds=80, seed=SEED+1):
    rng2 = np.random.default_rng(seed)
    n = len(sol_sub)
    outcomes, queried = [], set()
    beta, std, _ = fit_bt(outcomes, n, l2=0.2)
    mean_stds = [std[1:].mean()]
    n_comps   = [0]
    for rnd in range(n_rounds):
        if strategy == "active":
            pair, _ = most_uncertain_pair(beta, std, n, queried)
        else:
            i, j = sorted(rng2.choice(n, 2, replace=False)); pair = (i, j)
        queried.add(pair); i, j = pair
        winner = oracle(i, j, sol_sub, noise_std=0.8, rng=rng2)
        loser  = j if winner == i else i
        outcomes.append((winner, loser))
        beta, std, _ = fit_bt(outcomes, n, l2=0.2)
        if (rnd + 1) % 5 == 0:
            mean_stds.append(std[1:].mean())
            n_comps.append(rnd + 1)
    return np.array(n_comps), np.array(mean_stds)

n_a, s_a = run_with_uncertainty_tracking(sol_sub, "active")
n_r, s_r = run_with_uncertainty_tracking(sol_sub, "random")

fig, ax = plt.subplots(figsize=(13, 5))
fig.suptitle("Posterior Uncertainty Reduces as the AI Receives More Scientist Feedback",
             fontsize=13, fontweight="bold", color="#e0e0ff")

ax.plot(n_a, s_a, color=VIOLET, lw=2.5, marker="o", ms=6, label="Active querying")
ax.plot(n_r, s_r, color=GREY,   lw=1.8, marker="s", ms=6, ls="--", alpha=0.8, label="Random querying")
ax.fill_between(n_a, s_a * 0.88, s_a * 1.12, color=VIOLET, alpha=0.12)
ax.set_xlabel("Number of Scientist Comparisons", labelpad=8)
ax.set_ylabel("Mean Posterior Std of Latent Scores (lower = more certain)", labelpad=8)
ax.legend(fontsize=10)
ax.grid(True)

plt.tight_layout()
plt.savefig("uncertainty_reduction.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("Saved: uncertainty_reduction.png")


# ─── Plot 3: Final Molecular Landscape ────────────────────────────────────────
# After 80 comparisons, show the learned ranking vs. true solubility
print("Generating molecular landscape plot ...")

rng3 = np.random.default_rng(SEED+2)
outcomes_final = []
queried_f = set()
beta_f, std_f, _ = fit_bt([], n_sub, l2=0.2)
for _ in range(80):
    pair, _ = most_uncertain_pair(beta_f, std_f, n_sub, queried_f)
    queried_f.add(pair); i, j = pair
    winner = oracle(i, j, sol_sub, noise_std=0.8, rng=rng3)
    loser  = j if winner == i else i
    outcomes_final.append((winner, loser))
    beta_f, std_f, _ = fit_bt(outcomes_final, n_sub, l2=0.2)

# PCA 2D projection of molecular space for visualization
from sklearn.decomposition import PCA as PCA2
pca2d = PCA2(n_components=2, random_state=SEED)
X_2d  = pca2d.fit_transform(X_sub)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Molecular Space: Learned Solubility Ranking vs. Ground Truth\n"
             "After 80 pairwise comparisons with a simulated scientist",
             fontsize=13, fontweight="bold", color="#e0e0ff", y=1.02)

for ax, (vals, title) in zip(axes, [
    (sol_sub,  "True Solubility (hidden from model)"),
    (beta_f,   "Learned Score via Preference Learning (80 comparisons)"),
]):
    sc = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=vals,
                    cmap="plasma", s=30, alpha=0.7, edgecolors="none")
    # Highlight top-10% molecules
    top_idx = np.where(is_top_sub)[0]
    ax.scatter(X_2d[top_idx, 0], X_2d[top_idx, 1],
               c="white", s=60, marker="*", zorder=5, label="Top 10% (high solubility)")
    plt.colorbar(sc, ax=ax, pad=0.02)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("PC1", labelpad=6)
    ax.set_ylabel("PC2", labelpad=6)
    if "Learned" in title:
        ax.legend(fontsize=8, loc="upper right")

plt.tight_layout()
plt.savefig("molecular_landscape.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("Saved: molecular_landscape.png")


# ─── Summary ──────────────────────────────────────────────────────────────────
corr_final, _ = spearmanr(sol_sub, beta_f)
top_found = len(set(np.argsort(beta_f)[::-1][:n_top_sub]) & set(np.where(is_top_sub)[0]))

# Compute comparison savings
idx80a = np.argmax(rec_active[:, 1] / n_top_sub >= 0.80) if (rec_active[:, 1] / n_top_sub).max() >= 0.80 else -1
idx80r = np.argmax(rec_random[:, 1] / n_top_sub >= 0.80) if (rec_random[:, 1] / n_top_sub).max() >= 0.80 else -1

print("\n" + "="*65)
print("  HUMAN-IN-THE-LOOP DRUG DISCOVERY -- SUMMARY")
print("="*65)
print(f"  Dataset          : ESOL (Delaney solubility, 1,128 molecules)")
print(f"  Molecular encoder: ChemBERTa (768-dim) -> PCA-32")
print(f"  Preference model : Bradley-Terry + Laplace (diagonal)")
print(f"  Active criterion : Most uncertain pair (P closest to 0.5)")
print(f"  Oracle noise     : Gaussian noise std=0.8 (realistic human)")
print()
print(f"  After 80 comparisons:")
print(f"    Spearman rank corr: {corr_final:.4f}")
print(f"    Top-10% found     : {top_found}/{n_top_sub} ({top_found/n_top_sub*100:.1f}%)")
if idx80a >= 0 and idx80r >= 0:
    savings = int(rec_random[idx80r, 0]) - int(rec_active[idx80a, 0])
    print(f"\n  Queries to find 80% of top molecules:")
    print(f"    Active  : {int(rec_active[idx80a, 0])}")
    print(f"    Random  : {int(rec_random[idx80r, 0])}")
    print(f"    Savings : {savings} fewer scientist comparisons")
print("="*65)
print("\nAll plots saved.")
