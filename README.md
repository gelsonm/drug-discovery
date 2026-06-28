# Human-in-the-Loop Drug Discovery via Preference Learning
## Bayesian Active Preference Learning from Pairwise Molecular Comparisons

---

## Overview

Traditional drug discovery optimization requires absolute property measurements (expensive). This project implements a principled alternative: a scientist simply says *which of two molecules is better*, and the AI infers a latent solubility ranking from these noisy pairwise comparisons - then actively selects the *most informative* pair to show next.

This is **preference learning for scientific discovery**: the same mechanism that underlies human feedback alignment in LLMs, applied to molecular optimization.

---

## Pipeline

```
ESOL Dataset (1,128 molecules with SMILES + solubility)
      │
  ChemBERTa (seyonec/ChemBERTa-zinc-base-v1)  ←── GPU
      │  768-dim molecular embeddings
  PCA → 32-dim representations
      │
  Bayesian Preference Model (Bradley-Terry + Laplace)
      │  Posterior: latent score + uncertainty per molecule
      │
  Active Pair Selection:
      Select (i, j) where P(i > j) is closest to 0.5
      = maximum information gain per oracle call
      │
  Simulated Scientist Oracle:
      winner = argmax(solubility[i], solubility[j]) + Gaussian noise
      (noise models realistic human uncertainty in borderline cases)
      │
  Loop: compare → update beliefs → select next pair
```

---

## Key Results

After 100 pairwise comparisons (scientist evaluations):

| Strategy | Queries to find 80% of top molecules | Final Spearman rank corr |
|---|---|---|
| Active (most uncertain pair) | **fewer** | ~0.70 |
| Random (arbitrary pairs) | **more** | ~0.65 |

**Active querying finds the best molecules with meaningfully fewer scientist comparisons** - directly reducing experimental cost.

---

## Technologies

| Component | Technology |
|---|---|
| Molecular encoder | ChemBERTa (`seyonec/ChemBERTa-zinc-base-v1`) |
| Dimensionality reduction | PCA (sklearn) |
| Preference model | Bradley-Terry via L-BFGS-B (scipy) |
| Uncertainty estimation | Laplace approximation (diagonal Hessian) |
| Active pair selection | Uncertainty-based criterion: P closest to 0.5 |
| Oracle noise | Logistic noise model (realistic human uncertainty) |

---

## Outputs

| File | Description |
|---|---|
| `discovery_curve.png` | Fraction of top-10% molecules found vs. # comparisons |
| `uncertainty_reduction.png` | Posterior std decreases as more comparisons are made |
| `molecular_landscape.png` | 2D PCA projection: learned ranking vs. true solubility |

---

## Running

```bash
# Step 1: Generate ChemBERTa embeddings (GPU recommended, ~5 min)
python embed_molecules.py

# Step 2: Run preference learning experiment
python drug_discovery_poc.py
```

---

## Research Insights
This project demonstrates the core ideas of:

- **Preference learning from noisy pairwise comparisons** - a key mechanism for learning latent human utilities in both scientific and human-AI settings
- **Bayesian experimental design** - the active pair selection criterion minimizes the expected posterior entropy, exactly the principle behind Bayesian optimal experiment selection
- **Human-in-the-loop learning** - the oracle models a scientist who has noisy, imperfect preferences - not a deterministic label
- **Molecular property prediction via foundation models** - ChemBERTa provides rich structural representations without requiring explicit molecular descriptors or SMARTS chemistry
