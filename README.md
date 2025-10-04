# SurvivalLVQ

Prototype-based survival analysis with a learnable Mahalanobis metric and IPCW-Brier training.

This folder contains a PyTorch implementation of Survival Learning Vector Quantization (LVQ) in `Models/SurvivalLVQ.py`, utilities for evaluation in `utils.py`, and an optional preprocessing transformer in `SkewTransformer.py`. A usage notebook is provided in `example analysis.ipynb`.

### Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Key dependencies: `torch`, `scikit-survival`, `scikit-learn`, `scipy`, `numpy`, `matplotlib`.

### What is SurvivalLVQ?

- **Prototypes**: Learns `n_prototypes` in feature space. Each prototype carries a survival curve label computed via a weighted Kaplan–Meier estimator.
- **Learnable metric**: Learns a relevance matrix $\Lambda = \Omega^T\Omega$ that defines a Mahalanobis distance. Rank can be limited via `n_omega_rows`.
- **Soft assignment to closest two prototypes**: For each sample, compute distances to prototypes, then a two-prototype probability `q_ij` used to mix prototype survival curves.
- **Training objective**: Minimizes an IPCW-weighted Brier score over timepoints (censoring handled via Kaplan–Meier of censoring distribution).
- **Predictions**: Risk score is negative area under the predicted survival curve; survival functions are time-indexed mixtures of prototype curves.

See details in `Models/SurvivalLVQ.py` and notes in `Models/SurvivalLVQ.md` and `Models/LVQ.md`.

## Quickstart

```python
import numpy as np
from sksurv.util import Surv
from sklearn.model_selection import train_test_split

# If running from this folder (recommended): `cd Survival_Tree/SurvivalLVQ`
from Models.SurvivalLVQ import SurvivalLVQ
from SkewTransformer import SkewTransformer
import utils
# Otherwise, add this folder to sys.path or install as a package.

# 1) Prepare data
# X: (n_samples, n_features) float array
# y: survival labels as sequence of (event, time) pairs or a scikit-survival structured array
X = ...  # your numpy array
D = ...  # boolean or {0,1} event indicator (1=event, 0=censored)
T = ...  # durations/time-to-event (float)
y = Surv.from_arrays(event=D.astype(bool), time=T)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Optional preprocessing (min-max, selective log transform for skew, standardize)
pre = SkewTransformer().fit(X_train)
X_train = pre.transform(X_train)
X_test = pre.transform(X_test)

# 2) Train SurvivalLVQ
model = SurvivalLVQ(
    n_prototypes=2,
    n_omega_rows=None,     # or set < n_features for low-rank metric
    batch_size=128,
    init='kmeans',         # or 'random'
    device='cpu',          # or torch.device('cuda') if available
    lr=1e-3,
    epochs=50,
    verbose=True,
)
model.fit(X_train, y_train)

# 3) Predict risk scores (higher = higher risk)
risk_scores = model.predict(X_test)

# 4) Predict survival functions
def predict_curve_grid(model, X, times):
    fs = model.predict_survival_function(X)
    return np.vstack([[f(t) for t in times] for f in fs])

# 5) Evaluate
ci = utils.score_CI(model, X_test, y_test)
ci_ipcw = utils.score_CI_ipcw(model, X_test, y_train, y_test)
ibs = utils.score_brier(model, X_test, y_train, y_test)
print({'c_index': ci, 'c_index_ipcw': ci_ipcw, 'integrated_brier': ibs})

# 6) Visualize (projection, prototype curves, relevance matrix)
# y is a structured array; extract arrays for plotting
D_test, T_test = map(np.array, zip(*y_test))
model.vis(X_test, D_test, T_test)
```

## Data format

- **Features `X`**: `numpy.ndarray` of shape `(n_samples, n_features)`, dtype float. Handle categorical variables with one-hot encoding beforehand.
- **Labels `y`**: A sequence of `(event, time)` or a scikit-survival structured array, e.g. `Surv.from_arrays(event=D, time=T)`. In code, the model consumes `y` via `D, T = map(np.array, zip(*y))`, so both formats work.
- **Missing values**: During KMeans initialization, median imputation is applied. During training/inference, `NaN` feature differences are set to zero inside the distance function (a "NaN-LVQ" behavior). Prefer imputing upstream when possible.

## Key API

- `SurvivalLVQ(n_prototypes=2, n_omega_rows=None, batch_size=128, init='kmeans', device='cpu', lr=1e-3, epochs=50, verbose=True)`
  - **n_prototypes**: number of prototypes.
  - **n_omega_rows**: rows of `Omega`; if `None`, defaults to full rank. Set smaller than `n_features` for low-rank relevance.
  - **batch_size**: mini-batch size; batches are filled from a random sampler if the last batch is short.
  - **init**: `'kmeans'` (cluster centers on median-imputed data) or `'random'`.
  - **device**: `'cpu'` or `'cuda'`.
  - **lr**: learning rate. Internally, `Omega` uses `lr * 0.1` for stability.
  - **epochs**: training epochs.
  - **verbose**: print epoch loss.

- `fit(X, y)`: Train model, updating prototypes `w` and metric `Omega` using IPCW-Brier loss with Adam.
- `predict(X, closest=False)`:
  - If `closest=False` (default): returns risk score = negative AUC of predicted survival curve.
  - If `closest=True`: returns index of closest prototype per sample.
- `predict_closest(X)`: same as `predict(X, closest=True)`.
- `predict_survival_function(X)`: returns a list of callables `f_i(t)` mapping times to survival probabilities for each sample.
- `vis(X, D, T, print_variance_covered=True, true_eigen=False)`: visualization of 2D projection (top eigenvectors of $\Lambda$), prototype survival curves, relevance matrix, and eigenvalues.

## How it works (high level)

- **Initialization** (`_init_model`):
  - Sets time grid to unique event times (subsamples if >1000).
  - Initializes `Omega` to identity (full-rank) or a pseudo-inverse when `n_omega_rows < n_features`.
  - Initializes prototypes via KMeans or random.
  - Estimates censoring survival `G(t)` via Kaplan–Meier for IPCW.
  - Precomputes IPCW weights and initial prototype labels.

- **Metric and distance** (`lambda_mat`, `_omega_distance`):
  - $\Lambda = \Omega^T\Omega$. Distances are quadratic forms $(x-w)^T \Lambda (x-w)$.
  - `normalize_trace()` enforces `trace(ΩᵀΩ)=1` after each optimizer step.

- **Soft assignment to two closest prototypes** (`_q_ij`):
  - For each sample, take the two smallest distances `d_j, d_k` and set `p_j = d_k / (d_j + d_k)`, `p_k = d_j / (d_j + d_k)`; others 0.

- **Prototype survival labels** (`_calc_labels`):
  - Winner-take-all assignment to a single closest prototype for labeling.
  - For each prototype with at least 2 distinct times assigned, compute weighted Kaplan–Meier; interpolate to common grid of `timepoints`.

- **Loss** (`loss_brier`):
  - IPCW-weighted Brier score over all timepoints; supports full-batch and mini-batch modes.

- **Training loop** (`fit`):
  - Each epoch: recompute prototype labels from current assignments; iterate mini-batches, compute loss, backpropagate, step Adam on `w` and `Ω` (smaller LR for `Ω`), and renormalize `Ω`.

## Applying to new datasets

1) Prepare `X`, `D`, `T` as described above. Ensure floats, no strings; one-hot encode categoricals; consider scaling (e.g., `SkewTransformer`).

2) Create labels:
```python
from sksurv.util import Surv
y = Surv.from_arrays(event=D.astype(bool), time=T)
```

3) Fit the model and tune hyperparameters:
- **`n_prototypes`**: start with 2–4; increase if heterogeneous subgroups are expected.
- **`n_omega_rows`**: set below `n_features` for low-rank relevance to regularize and emphasize key directions.
- **`lr`, `epochs`, `batch_size`**: typical values are `1e-3`, `50–200`, `64–256`.
- **`init`**: `'kmeans'` is recommended.

4) Evaluate with provided utilities in `utils.py`:
```python
import utils
ci = utils.score_CI(model, X_test, y_test)
ci_ipcw = utils.score_CI_ipcw(model, X_test, y_train, y_test)
ibs = utils.score_brier(model, X_test, y_train, y_test)
```

5) Interpret:
- Use `model.vis(...)` to view data/prototypes in the top-2 eigenvector projection of $\Lambda$, prototype survival curves, feature relevance (`diag(\Lambda))`), and eigenvalues.
- `predict_closest(X)` yields the closest prototype index per sample.

## Practical notes

- Times are automatically restricted to unique observed event times and capped at 1000 grid points for efficiency.
- If the largest observed time is censored, the last IPCW weights can be ill-defined; the implementation guards with `nan_to_num`.
- GPU acceleration: pass `device=torch.device('cuda')` (and ensure tensors are on the same device).
- Saving/loading: you can `torch.save(model.state_dict(), path)` and reload to a freshly constructed instance with identical hyperparameters, or serialize the estimator with `joblib` if preferred.

## Files

- `Models/SurvivalLVQ.py`: main implementation.
- `Models/SurvivalLVQ.md`, `Models/LVQ.md`: conceptual notes.
- `SkewTransformer.py`: preprocessing helper (min-max scaling, selective log-transform, standardization).
- `utils.py`: evaluation helpers (C-index, IPCW C-index, integrated Brier score).
- `example analysis.ipynb`: end-to-end walkthrough.
