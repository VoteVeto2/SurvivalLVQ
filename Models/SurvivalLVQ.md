# SurvivalLVQ: Learning Vector Quantization for Survival Analysis

## Overview

SurvivalLVQ is a **prototype-based machine learning algorithm** that combines Learning Vector Quantization (LVQ) with survival analysis techniques. It learns representative prototypes in the feature space and associates each prototype with a survival curve, enabling both patient clustering and survival prediction.

## Core Concepts

### 1. **Prototypes ($\mathbf{w}$)**
- Representative points in the feature space that capture different patient subgroups
- Each prototype has an associated survival curve learned from assigned patients
- Initialized using K-means clustering for better starting positions

### 2. **Omega Matrix ($\Omega$)**
- A learnable transformation matrix that defines feature relevance and relationships
- Creates a metric for measuring distances in the transformed space
- Lambda matrix ($\Lambda = \Omega^T \Omega$) determines feature importance and interactions

### 3. **Distance Metric**
- Custom distance function: $d(\mathbf{x}, \mathbf{w}) = (\mathbf{x} - \mathbf{w})^T \Lambda (\mathbf{x} - \mathbf{w})$
- Allows the model to learn which features are most important for survival prediction
- Enables feature selection and dimensionality reduction

## Algorithm Architecture

```
Input Data (X) → Feature Transformation (Ω) → Distance Calculation → Prototype Assignment → Survival Prediction
                                                        ↓
                                               Kaplan-Meier Estimation per Prototype
```

## Step-by-Step Implementation Analysis

### **Step 1: Initialization (`__init__` and `_init_model`)**

```python
def __init__(self, n_prototypes=2, n_omega_rows=None, batch_size=128, ...):
```

**Purpose**: Set up model parameters and architecture

**Key Parameters**:
- `n_prototypes`: Number of patient subgroups to discover
- `n_omega_rows`: Dimensionality of the transformation matrix
- `batch_size`: Training batch size for gradient descent
- `lr`: Learning rate for optimization

**Initialization Process**:
1. **Data Setup**: Store survival times ($T$) and censoring indicators ($D$)
2. **Timepoint Selection**: Extract unique event times for survival curve evaluation
3. **Omega Matrix**: Initialize transformation matrix (identity or pseudo-inverse)
4. **Prototype Placement**: Use K-means to find initial prototype positions
5. **IPCW Weights**: Calculate Inverse Probability of Censoring Weights for bias correction

### **Step 2: Distance Function (`_omega_distance`)**

```python
def _omega_distance(self, x1, w, omega):
    diff = x1 - w[:, None, :]
    diff = torch.nan_to_num(diff)  # Handle missing values
    dists = (diff @ omega.transpose(dim0=-2, dim1=-1) @ omega)[:, :, None, :] @ diff[:, :, :, None]
    return dists.squeeze(-1).squeeze(-1)
```

**Purpose**: Compute learned distance between data points and prototypes

**Mathematical Formula**: $d(\mathbf{x}, \mathbf{w}_i) = (\mathbf{x} - \mathbf{w}_i)^T \Omega^T \Omega (\mathbf{x} - \mathbf{w}_i)$

**Key Features**:
- **Learnable**: The $\Omega$ matrix is optimized during training
- **NaN Handling**: Automatically handles missing values by setting them to 0
- **Feature Weighting**: Automatically learns which features are most important

### **Step 3: Prototype Assignment (`_q_ij`)**

```python
def _q_ij(self, d):
    top2_val, top2_i = d.topk(2, axis=0, largest=False)
    p = torch.flipud(top2_val) / top2_val.sum(dim=0)[None, :]
    res = torch.scatter(torch.zeros_like(d), dim=0, index=top2_i, src=p)
    return res
```

**Purpose**: Assign each patient to prototypes based on distance

**Strategy**: 
- Uses **soft assignment** to the two closest prototypes
- Assignment probability inversely proportional to distance
- Formula: $q_{ij} = \frac{d_k}{d_j + d_k}$ where $j, k$ are closest prototypes

**Benefits**:
- Prevents hard clustering artifacts
- Allows gradual transitions between patient groups
- Provides uncertainty quantification

### **Step 4: Survival Curve Learning (`_calc_labels`)**

```python
def _calc_labels(self):
    # ... distance calculation and assignment ...
    
    # Weighted Kaplan-Meier per prototype
    for i in range(w_local.size(0)):
        x, y = self.estimate_kaplan_meier(self.D[proto_assign[i, :]], 
                                         self.T[proto_assign[i, :]], 
                                         weights=q_ij[i, proto_assign[i, :]])
```

**Purpose**: Assign survival curves("label") for each prototype. 

**Process**:
1. **Patient Assignment**: Determine which patients belong to each prototype
2. **Weighted Kaplan-Meier**: Estimate survival curves using soft assignment weights
3. **Curve Storage**: Store survival functions for each prototype at predetermined timepoints

**Innovation**: Uses weighted Kaplan-Meier where weights come from prototype assignment probabilities

### **Step 5: Loss Function (`loss_brier`)**

```python
def loss_brier(self, x, t=None, d=None):
    pred = self.predict_curves(x)
    mu = IPCW_weights * (pred - (individual_curves * 1.0)) ** 2
    mu = mu.sum(dim=1)
    return mu.mean()
```

**Purpose**: Optimize model using Brier score with IPCW correction

**Components**:
- **Brier Score**: Measures accuracy of survival probability predictions
- **IPCW Weighting**: Corrects for censoring bias using Inverse Probability of Censoring Weights
- **Individual Curves**: True survival indicators for each patient at each timepoint

**Formula**:
$$
\mathrm{BS} = \sum w(t,\delta) \cdot [S(t|x) - I(T > t)]^2
$$

Where:
- $S(t|x)$: Predicted survival probability
- $I(T > t)$: True survival indicator
- $w(t,\delta)$: IPCW weights

### **Step 6: Training Loop (`fit`)**

```python
def fit(self, X, y):
    # ... initialization ...
    
    for epoch in range(self.epochs):
        self.fit_labels()  # Update prototype survival curves
        for batch in dataloader:
            optimizer_w.zero_grad()
            batch_loss = self.loss_brier(x, t, d)
            batch_loss.backward()
            optimizer_w.step()
            self.normalize_trace()  # Regularize Omega matrix
```

**Training Strategy**:
1. **Alternating Optimization**: 
   - Update prototype survival curves (Kaplan-Meier)
   - Update prototypes and $\Omega$ matrix (gradient descent)
2. **Batch Processing**: Use mini-batches for scalability
3. **Regularization**: Normalize trace of $\Lambda$ matrix to prevent overfitting

### **Step 7: Prediction (`predict`, `forward`)**

```python
def forward(self, x):
    pred = self.predict_curves(x)
    res = -torch.trapezoid(pred, self.timepoints) / self.timepoints.max()
    return res
```

**Prediction Types**:
1. **Risk Score**: Negative area under survival curve (higher = higher risk)
2. **Survival Curves**: Full survival function $S(t|x)$ for each patient
3. **Prototype Assignment**: Closest prototype for interpretability

**Process**:
1. Calculate distances to all prototypes
2. Compute soft assignment probabilities
3. Weight prototype survival curves by assignment probabilities
4. Return weighted average survival curve

## Key Innovations

### 1. **Learnable Distance Metric**
- Traditional LVQ uses Euclidean distance
- SurvivalLVQ learns optimal feature weighting through $\Omega$ matrix
- Enables automatic feature selection and interaction modeling

### 2. **IPCW Integration**
- Addresses censoring bias in survival data
- Uses Kaplan-Meier estimation for censoring distribution
- Properly weights training examples based on censoring probability

### 3. **Soft Prototype Assignment**
- Prevents hard clustering artifacts
- Allows uncertainty quantification
- Enables smooth transitions between patient groups

### 4. **Survival-Specific Loss Function**
- Uses Brier score appropriate for survival analysis
- Handles right-censored data correctly
- Optimizes for calibration and discrimination simultaneously

## Visualization Capabilities (`vis`)

The model provides comprehensive visualization:

1. **2D Projection**: Project data onto first two eigenvectors of $\Lambda$ matrix
2. **Prototype Survival Curves**: Show learned survival functions per prototype
3. **Feature Relevance Matrix**: Visualize feature interactions and importance
4. **Feature Importance**: Individual feature relevance scores
5. **Eigenvalue Spectrum**: Show dimensionality reduction effectiveness

## Mathematical Foundation

### Distance Function
$$
d(\mathbf{x}, \mathbf{w}_i) = (\mathbf{x} - \mathbf{w}_i)^T \Lambda (\mathbf{x} - \mathbf{w}_i), \quad \text{where} \ \Lambda = \Omega^T \Omega
$$

### Assignment Probabilities
$$
q_{ij} = \frac{d_k}{d_j + d_k}
$$
for closest prototypes $j, k$

### Risk Score
$$
\text{Risk} = -\frac{\int S(t|x) \, dt}{t_{\max}}
$$

### Brier Score Loss
$$
L = \sum_i \sum_t w_{it} \left[S(t|x_i) - I(T_i > t)\right]^2
$$

## Advantages

1. **Interpretability**: Prototype-based approach allows understanding of patient subgroups
2. **Feature Learning**: Automatically discovers relevant features and interactions
3. **Handles Missing Data**: Built-in NaN handling in distance calculations
4. **Scalable**: Mini-batch training for large datasets
5. **Comprehensive**: Provides risk scores, survival curves, and clustering
6. **Statistically Sound**: Uses proper survival analysis techniques (Kaplan-Meier, IPCW)

## Use Cases

1. **Patient Stratification**: Identify distinct survival patterns
2. **Biomarker Discovery**: Find important features through $\Omega$ matrix
3. **Risk Prediction**: Generate individual risk scores
4. **Clinical Decision Support**: Provide interpretable survival predictions
5. **Cohort Analysis**: Understand population-level survival patterns

## Technical Requirements

- **PyTorch**: For automatic differentiation and GPU acceleration
- **scikit-survival**: For survival analysis utilities
- **scikit-learn**: For preprocessing and evaluation
- **NumPy/SciPy**: For numerical computations
- **matplotlib**: For visualization

This implementation represents a sophisticated fusion of classical machine learning (LVQ) with modern survival analysis techniques, providing both high predictive accuracy and clinical interpretability.
