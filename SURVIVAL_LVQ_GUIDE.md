# SurvivalLVQ Implementation and Usage Guide

## Overview

**SurvivalLVQ** is a prototype-based survival analysis method that combines Learning Vector Quantization (LVQ) with survival analysis. It learns a small number of representative prototypes in feature space, where each prototype has an associated survival curve. The model uses a learnable Mahalanobis distance metric and is trained using an IPCW-weighted Brier score loss.

---

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Implementation Architecture](#implementation-architecture)
3. [Installation](#installation)
4. [Data Preparation](#data-preparation)
5. [Basic Usage](#basic-usage)
6. [Complete Example with Real Dataset](#complete-example-with-real-dataset)
7. [Model Parameters](#model-parameters)
8. [Understanding the Output](#understanding-the-output)
9. [Evaluation Metrics](#evaluation-metrics)
10. [PyTorch Infrastructure Details](#pytorch-infrastructure-details)
11. [Advanced Topics](#advanced-topics)

---

## Core Concepts

### What is SurvivalLVQ?

- **Prototypes**: The model learns `n_prototypes` representative points in feature space. Each prototype represents a "typical patient profile" with an associated survival curve.

- **Learnable Distance Metric**: Instead of using Euclidean distance, the model learns a relevance matrix $\Lambda = \Omega^T \cdot \Omega$ that defines a Mahalanobis distance. This metric learns which features (or feature combinations) are most important for survival prediction.

- **Soft Assignment**: Each new patient is assigned probabilistically to the two closest prototypes based on learned distances.

- **Survival Curves**: Each prototype has a survival curve computed via weighted Kaplan-Meier estimation from training data assigned to it.

- **IPCW Weighting**: Inverse Probability of Censoring Weighting handles right-censored data properly during training.

### Key Advantages

1. **Interpretability**: Visual inspection of prototypes and their survival curves
2. **Feature Relevance**: Automatic feature importance via learned relevance matrix
3. **Handles Missing Data**: NaN-LVQ behavior (missing features → zero difference in distance)
4. **Handles Censoring**: Proper statistical treatment via IPCW

---

## Implementation Architecture

### Class Structure

```python
class SurvivalLVQ(torch.nn.Module, BaseEstimator):
    """
    Inherits from:
    - torch.nn.Module: PyTorch neural network module for automatic differentiation
    - BaseEstimator: scikit-learn estimator interface for compatibility
    """
```

### Key Components

1. **Learnable Parameters** (updated during training via backpropagation):
   - `self.w`: Prototype locations in feature space, shape `(n_prototypes, n_features)`
   - `self.omega`: Relevance matrix component, shape `(n_omega_rows, n_features)`

2. **Fixed Components** (computed from data):
   - `self.c`: Prototype survival curves, shape `(n_prototypes, n_timepoints)`
   - `self.timepoints`: Grid of unique event times
   - `self.IPCW_weights`: Pre-computed censoring weights

3. **Distance Function**:
   ```
   d²(x, w) = (x - w)^T · Λ · (x - w)
   where Λ = Ω^T · Ω
   ```

---

## Installation

### Requirements

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Key Dependencies

- `torch`: PyTorch for automatic differentiation and GPU acceleration
- `scikit-survival`: Survival analysis utilities (Surv, metrics)
- `scikit-learn`: ML utilities (KMeans, StandardScaler, etc.)
- `scipy`: Scientific computing (interpolation, statistics)
- `numpy`: Numerical arrays
- `matplotlib`: Visualization

---

## Data Preparation

### Input Format

**Features (X)**:
- Type: `numpy.ndarray`
- Shape: `(n_samples, n_features)`
- Dtype: `float32` or `float64`
- Handle categorical variables via one-hot encoding before passing to model

**Labels (y)**:
- Type: Structured array from scikit-survival
- Created using: `y = Surv.from_arrays(event=D, time=T)`
- Where:
  - `D`: Event indicator (1 = event occurred, 0 = censored)
  - `T`: Time to event or censoring (positive floats)

### Example Data Preparation

```python
import numpy as np
import pandas as pd
from sksurv.util import Surv
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Load your data
# df should have features + 'event' column (0/1) + 'time' column (floats)
df = pd.read_csv('your_data.csv')

# Separate features and outcomes
feature_cols = ['age', 'tumor_size', 'num_nodes', ...]  # your feature names
X = df[feature_cols].values

D = df['event'].values.astype(bool)  # Event indicator
T = df['time'].values.astype(float)  # Time

# Handle categorical variables (one-hot encoding)
if you_have_categorical_features:
    df_processed = pd.get_dummies(df, columns=['categorical_col1', 'categorical_col2'])
    X = df_processed[feature_cols].values

# Create survival labels
y = Surv.from_arrays(event=D, time=T)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=D
)

# Handle missing values (if any)
imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)
```

---

## Basic Usage

### Minimal Example

```python
from Models.SurvivalLVQ import SurvivalLVQ
import torch

# Create model
model = SurvivalLVQ(
    n_prototypes=2,      # Number of prototypes to learn
    epochs=50,           # Training epochs
    lr=1e-3,             # Learning rate
    verbose=True         # Print training progress
)

# Train model
model.fit(X_train, y_train)

# Predict risk scores (higher = higher risk)
risk_scores = model.predict(X_test)

# Predict survival functions
survival_functions = model.predict_survival_function(X_test)
# survival_functions[i] is a callable: f(time) -> survival_probability
```

---

## Complete Example with Real Dataset

### Using GBSG2 (German Breast Cancer Study)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sksurv.datasets import load_gbsg2
from sksurv.util import Surv

# Import SurvivalLVQ components
from Models.SurvivalLVQ import SurvivalLVQ
from SkewTransformer import SkewTransformer
import utils

# ============================================
# 1. LOAD DATA
# ============================================
X_df, y = load_gbsg2()
print(f"Dataset shape: {X_df.shape}")  # (686, 8)
print(f"Features: {X_df.columns.tolist()}")
# Features: ['age', 'estrec', 'horTh', 'menostat', 'pnodes', 'progrec', 'tgrade', 'tsize']

# ============================================
# 2. PREPROCESS FEATURES
# ============================================

# Identify categorical and numerical features
categorical_features = ['horTh', 'menostat', 'tgrade']
numerical_features = ['age', 'estrec', 'pnodes', 'progrec', 'tsize']

# Get column indices before one-hot encoding (for later use)
num_col_ids = [X_df.columns.get_loc(col) for col in numerical_features]

# One-hot encode categorical variables
X_processed = pd.get_dummies(X_df, columns=categorical_features)
print(f"After encoding: {X_processed.shape}")  # (686, 12)

# Convert to numpy array
X = X_processed.values.astype(float)

# Extract event indicator and time from structured array
D = y['cens'].astype(bool)   # Event indicator
T = y['time'].astype(float)  # Survival time

# ============================================
# 3. TRAIN/TEST SPLIT
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=D  # Stratify by event status
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

# ============================================
# 4. IMPUTE MISSING VALUES (if any)
# ============================================
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# ============================================
# 5. SCALE NUMERICAL FEATURES
# ============================================
# SkewTransformer: applies min-max scaling, selective log-transform
# for skewed features, then standardization

# Update num_col_ids for one-hot encoded data
# (In this example, numerical features stay in same positions)
num_col_ids_new = list(range(5))  # First 5 columns are numerical

scaler = SkewTransformer()
scaler.fit(X_train[:, num_col_ids_new])
X_train[:, num_col_ids_new] = scaler.transform(X_train[:, num_col_ids_new])
X_test[:, num_col_ids_new] = scaler.transform(X_test[:, num_col_ids_new])

# ============================================
# 6. TRAIN SURVIVALVQ MODEL
# ============================================
import torch

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = SurvivalLVQ(
    n_prototypes=2,          # Learn 2 representative patient profiles
    n_omega_rows=None,       # Full-rank relevance matrix (None = auto)
    batch_size=128,          # Mini-batch size
    init='kmeans',           # Initialize prototypes using K-means
    device=device,           # Use GPU if available
    lr=1e-2,                 # Learning rate
    epochs=50,               # Number of training epochs
    verbose=True             # Print progress
)

# Fit the model
model.fit(X_train, y_train)

# ============================================
# 7. MAKE PREDICTIONS
# ============================================

# Risk scores (higher = higher risk of event)
risk_scores_train = model.predict(X_train)
risk_scores_test = model.predict(X_test)

# Survival functions (callable objects)
surv_funcs_test = model.predict_survival_function(X_test)

# Example: predict survival probability at t=1000 days for first test patient
survival_prob_at_1000 = surv_funcs_test[0](1000)
print(f"Patient 0 survival probability at t=1000: {survival_prob_at_1000:.3f}")

# ============================================
# 8. EVALUATE MODEL
# ============================================

# Concordance index (C-index): measures ranking ability
ci = utils.score_CI(model, X_test, y_test)
print(f"C-index: {ci:.4f}")

# IPCW-adjusted C-index: accounts for censoring
ci_ipcw = utils.score_CI_ipcw(model, X_test, y_train, y_test)
print(f"C-index (IPCW): {ci_ipcw:.4f}")

# Integrated Brier Score: measures calibration
ibs = utils.score_brier(model, X_test, y_train, y_test)
print(f"Integrated Brier Score: {ibs:.4f}")

# ============================================
# 9. VISUALIZE RESULTS
# ============================================

# Extract event indicators and times for visualization
D_train, T_train = map(np.array, zip(*y_train))
D_test, T_test = map(np.array, zip(*y_test))

# Create comprehensive visualization
# - 2D projection of data and prototypes
# - Prototype survival curves
# - Feature relevance matrix
# - Feature importance (diagonal of relevance matrix)
# - Eigenvalues of relevance matrix
model.vis(X_train, D_train, T_train, print_variance_covered=True)

# ============================================
# 10. INSPECT LEARNED PROTOTYPES
# ============================================

# Get prototype locations and their survival curves
prototypes = model.w.detach().cpu().numpy()  # Shape: (n_prototypes, n_features)
prototype_curves = model.c.detach().cpu().numpy()  # Shape: (n_prototypes, n_timepoints)
timepoints = model.timepoints.numpy()

print(f"\nPrototype locations shape: {prototypes.shape}")
print(f"Prototype survival curves shape: {prototype_curves.shape}")

# Feature relevance (diagonal of Lambda matrix)
relevance_matrix = model.lambda_mat().detach().cpu().numpy()
feature_relevance = np.diag(relevance_matrix)

print(f"\nFeature relevance (sum={feature_relevance.sum():.3f}):")
for i, rel in enumerate(feature_relevance):
    print(f"  Feature {i}: {rel:.4f}")

# ============================================
# 11. ASSIGN TEST PATIENTS TO PROTOTYPES
# ============================================

# Find closest prototype for each test patient
closest_prototypes = model.predict_closest(X_test)
print(f"\nPrototype assignment for test patients:")
print(f"  Assigned to prototype 0: {np.sum(closest_prototypes == 0)}")
print(f"  Assigned to prototype 1: {np.sum(closest_prototypes == 1)}")
```

### Expected Output

```
Dataset shape: (686, 8)
Training samples: 548
Test samples: 138
Using device: cpu

Epoch: 1 / 50 | Loss: 0.276056
Epoch: 2 / 50 | Loss: 0.275815
...
Epoch: 50 / 50 | Loss: 0.250510

Patient 0 survival probability at t=1000: 0.723
C-index: 0.6523
C-index (IPCW): 0.6489
Integrated Brier Score: 0.1876

variance covered by projection: 78.3%

Feature relevance (sum=1.000):
  Feature 0: 0.0823
  Feature 1: 0.1245
  ...
```

---

## Model Parameters

### Constructor Parameters

```python
SurvivalLVQ(
    n_prototypes=2,       # Number of prototypes to learn
    n_omega_rows=None,    # Rank of relevance matrix (None=full rank)
    batch_size=128,       # Mini-batch size for training
    init='kmeans',        # Initialization: 'kmeans' or 'random'
    device='cpu',         # torch.device: 'cpu' or 'cuda'
    lr=1e-3,              # Learning rate for Adam optimizer
    epochs=50,            # Number of training epochs
    verbose=True          # Print training progress
)
```

#### Parameter Details

**`n_prototypes`** (int, default=2)
- Number of representative patient profiles to learn
- Start with 2-4 for interpretability
- Increase if you suspect multiple distinct survival subgroups
- More prototypes = more flexibility but less interpretability

**`n_omega_rows`** (int or None, default=None)
- Rank of the relevance matrix Ω
- If `None`: uses full rank = min(n_samples, n_features)
- If < n_features: low-rank metric learning (regularization)
- Lower rank emphasizes fewer feature directions, can prevent overfitting

**`batch_size`** (int, default=128)
- Number of samples per mini-batch
- Smaller = more stochastic updates, potentially better generalization
- Larger = more stable gradients, faster training
- Typical range: 64-256

**`init`** (str, default='kmeans')
- Prototype initialization method
- `'kmeans'`: Use K-means cluster centers (recommended)
- `'random'`: Random initialization near data mean
- K-means typically provides better starting positions

**`device`** (torch.device or str, default='cpu')
- Computation device
- `'cpu'`: Use CPU (slower but always available)
- `'cuda'` or `torch.device('cuda')`: Use GPU (much faster for large datasets)
- Check availability: `torch.cuda.is_available()`

**`lr`** (float, default=1e-3)
- Learning rate for Adam optimizer
- Prototypes use `lr`, relevance matrix uses `lr * 0.1` (more stable)
- Typical range: 1e-4 to 1e-2
- Too high: unstable training, too low: slow convergence

**`epochs`** (int, default=50)
- Number of passes through training data
- Monitor loss: if still decreasing, increase epochs
- Typical range: 50-200

**`verbose`** (bool, default=True)
- Print loss at each epoch for monitoring

---

## Understanding the Output

### 1. Risk Scores

```python
risk_scores = model.predict(X_test)
```

- **Higher score = higher risk of event**
- Computed as: `-∫S(t)dt / t_max` (negative area under survival curve)
- Use for ranking patients by risk
- Compatible with scikit-survival evaluation functions

### 2. Survival Functions

```python
survival_functions = model.predict_survival_function(X_test)
# Returns list of interpolation functions

# Get survival probability at specific time for patient i
prob_at_time = survival_functions[i](time)

# Example: 5-year survival for first patient
prob_5yr = survival_functions[0](5 * 365)  # Assuming time in days
```

- Each function is a `scipy.interpolate.interp1d` object
- Returns survival probability S(t) for any time t
- S(t) = probability of surviving beyond time t

### 3. Closest Prototype

```python
closest_proto = model.predict_closest(X_test)
```

- Returns prototype index (0, 1, ..., n_prototypes-1) for each sample
- Useful for assigning patients to risk groups
- Based on learned Mahalanobis distance

### 4. Visualization Output

```python
D_train, T_train = map(np.array, zip(*y_train))
model.vis(X_train, D_train, T_train)
```

Creates 5 plots:

**Plot 1: 2D Projection**
- Data points projected onto top 2 eigenvectors of Λ
- Color = survival time (observed events)
- White with black edge = censored observations
- Large numbered circles = prototypes
- Shows how prototypes partition feature space

**Plot 2: Prototype Survival Curves**
- Survival probability S(t) vs. time for each prototype
- Shows distinct survival patterns learned
- Prototype 1 (solid line), Prototype 2 (dashed), etc.

**Plot 3: Relevance Matrix**
- Heatmap of Λ = Ω^T·Ω
- Diagonal = individual feature relevance
- Off-diagonal = feature interactions
- Brighter = more relevant

**Plot 4: Feature Relevance**
- Bar plot of diagonal(Λ)
- Shows which features are most important
- Sum of all bars = 1.0 (normalized)

**Plot 5: Eigenvalues**
- Eigenvalues of Λ in descending order
- Shows dimensionality of learned metric
- Large first few eigenvalues = low-dimensional structure

---

## Evaluation Metrics

### Concordance Index (C-index)

```python
from utils import score_CI

ci = score_CI(model, X_test, y_test)
# Returns: float in [0, 1]
```

- Measures discrimination: ability to rank patients by risk
- 0.5 = random, 1.0 = perfect ranking
- Interpretation:
  - < 0.55: Poor
  - 0.55-0.65: Fair
  - 0.65-0.75: Good
  - \> 0.75: Excellent

### IPCW-adjusted C-index

```python
from utils import score_CI_ipcw

ci_ipcw = score_CI_ipcw(model, X_test, y_train, y_test)
```

- Accounts for censoring via IPCW
- More robust than standard C-index for heavy censoring
- Uses training data to estimate censoring distribution

### Integrated Brier Score (IBS)

```python
from utils import score_brier

ibs = score_brier(model, X_test, y_train, y_test)
# Returns: float in [0, 1]
```

- Measures calibration: accuracy of predicted survival probabilities
- Lower is better (0 = perfect, 0.25 = random)
- Integrated over time points in test data range
- Interpretation:
  - < 0.15: Excellent
  - 0.15-0.20: Good
  - 0.20-0.25: Fair
  - \> 0.25: Poor

---

## PyTorch Infrastructure Details

### Why PyTorch?

1. **Automatic Differentiation**: Computes gradients for backpropagation automatically
2. **GPU Acceleration**: Easy transfer to CUDA for fast training on large datasets
3. **Dynamic Computation Graphs**: Flexible model definition
4. **Optimization**: Built-in optimizers (Adam) with learning rate scheduling

### Key PyTorch Concepts Used

#### 1. Tensors

```python
# NumPy array → PyTorch tensor
X_torch = torch.tensor(X, dtype=torch.float32)

# Tensor → NumPy array
X_numpy = X_torch.numpy()

# Move tensor to device (CPU/GPU)
X_torch = X_torch.to(device)
```

#### 2. Parameters (nn.Parameter)

```python
self.w = torch.nn.Parameter(self.w)         # Prototypes
self.omega = torch.nn.Parameter(self.omega) # Relevance matrix
```

- Automatically tracked for gradient computation
- Updated by optimizer during training
- Registered in model's `parameters()`

#### 3. Gradient Computation

```python
# Forward pass (compute loss)
loss = self.loss_brier(x_batch, t_batch, d_batch)

# Backward pass (compute gradients via autograd)
loss.backward()  # ∂loss/∂w and ∂loss/∂omega computed automatically

# Update parameters
optimizer.step()  # w ← w - lr·∂loss/∂w, omega ← omega - lr·∂loss/∂omega
```

#### 4. No-Gradient Context

```python
with torch.no_grad():
    # Operations here don't build computation graph
    predictions = model.predict(X_test)
```

- Used during inference (prediction) to save memory
- Disables gradient tracking for operations

#### 5. Detaching from Computation Graph

```python
w_local = self.w.detach()  # Copy of w, not tracked for gradients
```

- Used when computing labels (avoids interfering with backprop)
- Creates a "view" that doesn't require gradients

### Training Loop Structure

```python
# Pseudo-code of what happens in fit()

# 1. Convert NumPy → Torch tensors
X_torch = torch.tensor(X, dtype=torch.float32)

# 2. Initialize model (_init_model)
#    - Initialize prototypes (w) and omega
#    - Make them nn.Parameters
#    - Compute IPCW weights, timepoints, etc.

# 3. Create DataLoader for mini-batching
dataset = torch.utils.data.TensorDataset(X, T, D)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

# 4. Create optimizer
optimizer = torch.optim.Adam([
    {'params': self.w},                      # prototypes, lr=1e-3
    {'params': self.omega, 'lr': 1e-4}       # omega, smaller lr
], lr=1e-3)

# 5. Training loop
for epoch in range(epochs):
    # Re-compute prototype labels from current positions
    self.fit_labels()

    for x_batch, t_batch, d_batch in dataloader:
        optimizer.zero_grad()                     # Reset gradients
        loss = self.loss_brier(x_batch, t_batch, d_batch)  # Forward pass
        loss.backward()                           # Backward pass (compute gradients)
        optimizer.step()                          # Update parameters
        self.normalize_trace()                    # Renormalize omega
```

### GPU Usage

```python
# Check GPU availability
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    print("Using CPU")

# Create model on GPU
model = SurvivalLVQ(n_prototypes=2, device=device, epochs=50)

# Train (data is automatically moved to GPU inside fit())
model.fit(X_train, y_train)
```

**Performance tip**: GPU acceleration is most beneficial when:
- Dataset has > 1000 samples
- Feature dimension > 50
- Many epochs (> 100)

---

## Advanced Topics

### 1. Low-Rank Metric Learning

```python
# Use low-rank relevance matrix for regularization
model = SurvivalLVQ(
    n_prototypes=2,
    n_omega_rows=5,  # Force 5-dimensional metric (even if n_features > 5)
    epochs=100
)
```

**When to use**:
- High-dimensional data (n_features > 50)
- Suspected low intrinsic dimensionality
- Prevent overfitting to noisy features

### 2. Handling Missing Data

SurvivalLVQ has built-in NaN handling:

```python
# During distance computation, NaN differences are set to 0
diff = torch.nan_to_num(diff)
```

**Best practice**: Impute missing values upstream when possible:

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')  # or 'mean', 'most_frequent'
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)
```

### 3. Hyperparameter Tuning

Use cross-validation to select hyperparameters:

```python
from sklearn.model_selection import KFold
from sksurv.metrics import concordance_index_censored

# Parameters to tune
n_prototypes_list = [2, 3, 4, 5]
lr_list = [1e-4, 1e-3, 1e-2]
n_omega_rows_list = [None, 5, 10, 20]

best_ci = 0
best_params = {}

for n_proto in n_prototypes_list:
    for lr in lr_list:
        for n_omega in n_omega_rows_list:
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            ci_scores = []

            for train_idx, val_idx in kf.split(X_train):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]

                model = SurvivalLVQ(
                    n_prototypes=n_proto,
                    n_omega_rows=n_omega,
                    lr=lr,
                    epochs=50,
                    verbose=False
                )
                model.fit(X_tr, y_tr)

                ci = utils.score_CI(model, X_val, y_val)
                ci_scores.append(ci)

            mean_ci = np.mean(ci_scores)
            if mean_ci > best_ci:
                best_ci = mean_ci
                best_params = {
                    'n_prototypes': n_proto,
                    'lr': lr,
                    'n_omega_rows': n_omega
                }

print(f"Best params: {best_params}, CV C-index: {best_ci:.4f}")
```

### 4. Saving and Loading Models

```python
import torch

# Save model state
torch.save({
    'model_state_dict': model.state_dict(),
    'n_prototypes': model.n_prototypes,
    'n_omega_rows': model.n_omega_rows,
    'n_features': model.n_features,
    # ... other hyperparameters
}, 'survival_lvq_model.pth')

# Load model
checkpoint = torch.load('survival_lvq_model.pth')

# Create new model with same architecture
model_loaded = SurvivalLVQ(
    n_prototypes=checkpoint['n_prototypes'],
    n_omega_rows=checkpoint['n_omega_rows'],
    # ... other hyperparameters
)

# Need to initialize model first (fit on any data to create parameters)
# Then load saved weights
model_loaded.load_state_dict(checkpoint['model_state_dict'])
```

**Note**: Due to dynamic initialization in `_init_model`, you need to fit once on a small batch to initialize all parameters before loading saved weights.

**Alternative**: Use `joblib` for pickling entire estimator:

```python
import joblib

# Save
joblib.dump(model, 'survival_lvq_model.pkl')

# Load
model_loaded = joblib.load('survival_lvq_model.pkl')
```

### 5. Feature Engineering Tips

**Standardization**: Always scale features
```python
from SkewTransformer import SkewTransformer

scaler = SkewTransformer()  # Handles skewness + standardization
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

**Categorical Encoding**: One-hot encoding
```python
X_df = pd.get_dummies(X_df, columns=['categorical_col1', 'categorical_col2'])
```

**Feature Interactions**: Manually add if domain knowledge suggests
```python
X_df['age_tumor_interaction'] = X_df['age'] * X_df['tumor_size']
```

### 6. Interpreting Learned Relevance Matrix

```python
# Get relevance matrix
Lambda = model.lambda_mat().detach().cpu().numpy()

# Individual feature importance
feature_importance = np.diag(Lambda)

# Feature interactions (off-diagonal elements)
interaction_strength = Lambda - np.diag(np.diag(Lambda))

# Most important feature
most_important_feature_idx = np.argmax(feature_importance)

# Top interacting feature pairs
off_diag_abs = np.abs(interaction_strength)
np.fill_diagonal(off_diag_abs, 0)
i, j = np.unravel_index(np.argmax(off_diag_abs), off_diag_abs.shape)
print(f"Strongest interaction: feature {i} <-> feature {j}")
```

### 7. Computational Complexity

**Training**:
- Time per epoch: O(n_samples × n_prototypes × n_features²)
- Memory: O(n_features² + n_prototypes × n_features)

**Prediction**:
- Time: O(n_test × n_prototypes × n_features²)
- Memory: O(n_test × n_timepoints)

**Scaling tips**:
- Use GPU for n_samples > 1000 or n_features > 50
- Use low-rank omega (n_omega_rows < n_features) for high-dimensional data
- Subsample timepoints (done automatically if > 1000 unique event times)

---

## Common Issues and Solutions

### Issue 1: Poor C-index (< 0.55)

**Solutions**:
- Increase number of prototypes
- Increase epochs (check if loss still decreasing)
- Try different learning rate (1e-4 to 1e-2)
- Check feature scaling (use SkewTransformer)
- Verify data quality (outliers, missing values)

### Issue 2: Training loss not decreasing

**Solutions**:
- Decrease learning rate
- Check for NaN/Inf in data
- Try different initialization ('kmeans' vs 'random')
- Increase batch size for more stable gradients

### Issue 3: GPU out of memory

**Solutions**:
- Decrease batch_size
- Use low-rank omega (n_omega_rows < n_features)
- Move model to CPU: `device='cpu'`

### Issue 4: Prototypes collapse (all in same location)

**Solutions**:
- Increase learning rate for prototypes
- Ensure features are scaled
- Try different initialization
- Increase number of prototypes

---

## Summary Checklist

### Data Preparation
- [ ] Features are numeric (one-hot encode categoricals)
- [ ] Features are scaled/standardized
- [ ] Missing values handled (imputation)
- [ ] Survival labels in correct format: `Surv.from_arrays(event, time)`

### Model Training
- [ ] Choose appropriate n_prototypes (2-5 for interpretability)
- [ ] Set device ('cpu' or 'cuda')
- [ ] Choose learning rate (1e-3 is good default)
- [ ] Set enough epochs (monitor loss convergence)

### Evaluation
- [ ] Compute C-index on test set
- [ ] Visualize prototypes and survival curves
- [ ] Inspect feature relevance
- [ ] Validate predictions make clinical sense

### Interpretation
- [ ] Examine prototype survival curves for distinct patterns
- [ ] Check feature relevance for clinical plausibility
- [ ] Visualize 2D projection to understand patient clustering
- [ ] Assign patients to prototypes for risk stratification

---

## Further Reading

- **Original LVQ**: Kohonen, T. (1995). "Learning Vector Quantization"
- **Survival Analysis**: Klein & Moeschberger, "Survival Analysis: Techniques for Censored and Truncated Data"
- **IPCW**: van der Laan & Robins, "Unified Methods for Censored Longitudinal Data and Causality"
- **PyTorch Documentation**: https://pytorch.org/docs/stable/index.html
- **scikit-survival**: https://scikit-survival.readthedocs.io/

---

## Contact and Support

For issues or questions:
- Check existing issues: https://github.com/[your-repo]/issues
- Review example notebooks: `example_breast_cancer.ipynb`, `example_veteran.ipynb`
- Read code documentation in `Models/SurvivalLVQ.py`