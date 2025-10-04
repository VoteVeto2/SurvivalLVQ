# Learning Vector Quantization

### Part 1 – *LVQ for a 15‑year‑old*

Imagine you’re arranging a big box of mixed LEGO bricks.
You don’t want to look at every single brick each time you build, so you drop a **few “sample bricks”** on the table—say one red, one blue, one yellow.
Now you repeatedly do two things:

1. **Pick up a random brick from the box.**
2. **Slide the nearest sample brick toward it** if the colors match, or **push it away** if the colors don’t.

After many rounds, each sample brick ends up sitting in the **heart of the pile** of bricks with its own color.
Whenever a new brick appears, you simply ask *“Which sample brick is it closest to?”*—and you instantly know its color group.

Learning Vector Quantization (LVQ) works almost exactly like that for data:

* **Bricks → data points** (numbers, images, sensor readings, …)
* **Sample bricks → prototypes** (tiny “representatives” the algorithm moves around)
* **Distance on the table → mathematical distance** (usually Euclidean)

By gently nudging the prototypes toward good matches and away from bad ones, LVQ learns where each **type** of data lives in the big “data space,” so it can sort new items very quickly.

---

### Part 2 – *LVQ for professional data scientists*

#### 1. Core idea

LVQ is a **prototype‑based, winner‑takes‑all** scheme originating from Kohonen’s work.
Given a training set $\{(\mathbf{x}_i, y_i)\}_{i=1}^{N}\subset\mathbb{R}^d\times\{1,\dots,C\}$, the model maintains $M$ labeled prototypes $\{\mathbf{w}_j, c_j\}_{j=1}^{M}$.
At inference, class prediction is $\hat{y} = c_{j^*}$ with

$$
j^*=\arg\min_j \|\mathbf{x}-\mathbf{w}_j\|_2.
$$

#### 2. Classical LVQ‑1 update

For a sample $(\mathbf{x},y)$:

* Identify winner $w^+$ with $c=w^+, y=c$ and nearest “impostor” $w^-$ with $c\neq y$.
* Learning rate $\alpha(t)$ (monotone decreasing).
* Update

  $$
  \mathbf{w}^+\leftarrow \mathbf{w}^+ +\alpha(t)\,(\mathbf{x}-\mathbf{w}^+),\qquad
  \mathbf{w}^- \leftarrow \mathbf{w}^- -\alpha(t)\,(\mathbf{x}-\mathbf{w}^-).
  $$

This creates piece‑wise linear decision boundaries (Voronoï cells).

#### 3. Modern formulations (GLVQ & metric learning)

Generalized LVQ (GLVQ) recasts learning as explicit cost minimization([NeurIPS Proceedings][1]):

$$
E = \sum_i \phi\!\bigl(\mu_i\bigr),\quad
\mu_i=\frac{d^+_i - d^-_i}{d^+_i + d^-_i},
$$

with $d^\pm$ the squared distances to $w^\pm$.
Stochastic gradient descent updates both prototypes **and possibly a relevance matrix** $\Lambda\succ0$ (GMLVQ) to learn a Mahalanobis metric, delivering intrinsic feature weighting and interpretability.

#### 4. From classification to clustering

Although vanilla LVQ is supervised, the machinery adapts naturally to unsupervised clustering:

| Strategy                       | Sketch                                                                                                                             |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| **“Blind” VQ**                 | Ignore labels; treat prototype index as cluster ID. Equivalent to competitive learning / online k‑means (a.k.a. Linde–Buzo–Gray).  |
| **Granule‑based LVQ**          | Use neighborhood cohesion terms so nearby prototypes cooperate, improving compactness and separation([ACM Digital Library][2]).    |
| **Hierarchical VQ (HVQ)**      | Stack two LVQ stages to refine coarse clusters into subclusters, recently used for video‑segment clustering([arXiv][3]).           |
| **Pseudo‑label bootstrapping** | Alternate k‑means label assignment and GLVQ metric learning; prototypes evolve and metric sharpens, akin to deep clustering loops. |

**Pseudocode (unsupervised LVQ variant)**

```
initialize M prototypes w_j ← random samples
repeat for epoch = 1…T
    for x ∈ data (shuffle):
        j* ← argmin_j ||x - w_j||
        w_j* ← w_j* + α(t) (x - w_j*)
    decay α(t)
until convergence
assign cluster(x) = argmin_j ||x - w_j||
```

#### 5. Practical insights & hyper‑parameters

| Element                    | Impact                                                                             | Heuristics                                               |
| -------------------------- | ---------------------------------------------------------------------------------- | -------------------------------------------------------- |
| **Prototype count $M$**    | Model capacity ↔ over‑fitting                                                      | Start with $M\approx10\sqrt{C}$ and prune/merge.         |
| **Learning‑rate schedule** | Convergence speed/stability                                                        | Exponential decay: $\alpha_t=\alpha_0 (1+\beta t)^{-1}$. |
| **Metric learning**        | Feature relevance, anisotropic clusters                                            | Enable Λ updates once prototypes stabilise.              |
| **Initialization**         | Strongly affects final cells                                                       | k‑means++ or stratified sampling within classes.         |
| **Cost monitoring**        | Quantization error (unsupervised) or GLVQ cost (supervised) guides early‑stopping. |                                                          |

#### 6. Why choose LVQ for clustering projects?

* **Prototype interpretability** – prototypes are in input space; easy to inspect or visualise.
* **Online/streaming friendly** – single‑pass updates $\mathcal{O}(d)$.
* **Metric learning plug‑in** – gives built‑in feature selection.
* **Voronoï tiling** – crisp, geometrically meaningful partitions; decision boundaries can be exported as polytopes.
* **Hybrid potential** – combine with SOM for topology preservation or embed inside deep auto‑encoders for discrete latent codes (VQ‑VAE).

#### 7. Recent research pointers

* LVQ variants for big data compression and recommender systems([arXiv][4]).
* Log‑Euclidean GLVQ for manifold‑valued data (e.g., SPD matrices)([PubMed][5]).
* Meta‑Quantization frameworks marrying VQ with meta‑learning for discrete latent models([ICML][6]).

---

### Take‑away

For your clustering study, start with **online k‑means‑style LVQ** to set prototypes, then iterate with **GLVQ + relevance learning** to refine clusters and highlight important features. The algorithm remains lightweight, interpretable, and easy to deploy both in batch and streaming pipelines while offering room for sophisticated metric‑learning extensions.

[1]: https://papers.neurips.cc/paper/1113-generalized-learning-vector-quantization.pdf?utm_source=chatgpt.com "[PDF] Generalized Learning Vector Quantization"
[2]: https://dl.acm.org/doi/10.3233/JIFS-220092?utm_source=chatgpt.com "An LVQ clustering algorithm based on neighborhood granules"
[3]: https://arxiv.org/abs/2412.17640?utm_source=chatgpt.com "Hierarchical Vector Quantization for Unsupervised Action Segmentation"
[4]: https://arxiv.org/abs/2405.03110?utm_source=chatgpt.com "Vector Quantization for Recommender Systems: A Review and Outlook"
[5]: https://pubmed.ncbi.nlm.nih.gov/35700257/?utm_source=chatgpt.com "Generalized Learning Vector Quantization With Log-Euclidean ..."
[6]: https://icml.cc/virtual/2025/poster/43509?utm_source=chatgpt.com "Learning to Quantize for Training Vector-Quantized Networks"
