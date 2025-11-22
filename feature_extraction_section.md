# 3.1.2 Feature Extraction

Feature extraction transforms the complex 3D geometric state of a bin-packing problem into a compact representation suitable for decision-making, whether by hand-crafted heuristics or learned neural models. The choice of features directly impacts solution quality: too sparse and critical spatial relationships are lost; too rich and computational cost becomes prohibitive. This section reviews how state-of-the-art methods—both traditional and learning-based—represent packing states to guide placement decisions.

## Geometric Features from Candidate Heuristics

Extreme point (EP) and empty maximal space (EMS) heuristics (Section 3.1.1) inherently define feature spaces through their geometric primitives. Each EP or EMS encodes **local spatial information**:

- **Position** $(x, y, z)$: Location within the container
- **Available volume**: For EMS, the dimensions $(w, d, h)$ of the free rectangular region; for EP, the clearance to nearest obstacles in each direction
- **Support characteristics**: Contact area with placed items below, distance to floor
- **Accessibility**: Distance to container walls or loading doors (relevant for LIFO constraints)

These geometric descriptors are used directly by constructive heuristics to rank candidates. For example, the EP-FFD heuristic [CPT08] selects placements lexicographically prioritizing lower $(x, y, z)$ coordinates, implicitly favoring compact, bottom-left arrangements. EMS-based methods [PÁVTO08, GR13] often prefer larger spaces to minimize fragmentation.


## Heightmap Representation

A **heightmap** is a 2D grid representation of the maximum occupied height at each $(x, y)$ position in the container. Originally developed for 2D bin packing with gravity constraints, heightmaps have been adapted for 3D nesting to efficiently encode vertical occupation and support surfaces [ARCO22].

### Construction and Usage

Given a container of dimensions $W \times D \times H$, a heightmap discretizes the base $(x, y)$ plane into a grid of resolution $r_x \times r_y$. Each cell $(i, j)$ stores the maximum $z$-coordinate of any item occupying that horizontal position:

$$
h[i, j] = \max\left\{z + d_z^k \mid \text{item } k \text{ occupies cell } (i, j)\right\}
$$

where $d_z^k$ is the height of item $k$ in its current orientation. If no item occupies cell $(i, j)$, then $h[i, j] = 0$.

**Key Advantages:**

1. **Efficient support detection**: An item placed at $(x, y, z)$ is supported if its footprint aligns with heightmap values $h[i, j] \approx z$ within tolerance
2. **Void identification**: Large flat regions in the heightmap indicate stable placement zones
3. **Gravity simulation**: Items can be "dropped" onto the heightmap by setting $z = \max_{(i,j) \in \text{footprint}} h[i, j]$
4. **Fixed representation size**: Unlike EP/EMS lists that grow with the number of placements, heightmaps maintain constant $O(r_x \times r_y)$ memory

**Limitations:**

- **Resolution trade-off**: Fine grids capture detail but increase computational cost; coarse grids may miss small voids or create artificial overlaps
- **Loss of 3D structure**: Overhangs and internal cavities cannot be represented; only the topmost surface is recorded
- **Discretization artifacts**: Continuous item positions must be mapped to discrete grid cells, potentially introducing small gaps or overlaps

### Integration with Learning

Heightmaps provide a natural **visual encoding** for convolutional neural networks (CNNs). Xiong et al. (2024) [XGP+24] use heightmaps as input to a Packing Transformer in their GOPT system for online 3D bin packing with robotic arms. The heightmap is treated as a single-channel image, processed by convolutional layers to extract spatial features such as flat regions, edge proximity, and height gradients. This representation enables the network to learn placement policies that generalize across different bin sizes—since the heightmap's spatial structure is preserved regardless of absolute container dimensions.

By combining heightmaps with item embeddings (size, weight, orientation constraints), GOPT learns to:
- Identify stable placement zones (flat, well-supported regions)
- Avoid fragmentation by favoring placements that preserve large contiguous free areas
- Respect physical constraints (support, load-bearing) implicitly encoded in the height profile


## Handcrafted Features in Traditional Methods

Classical heuristics and metaheuristics rely on explicitly engineered features to evaluate placement quality. These features typically decompose into three categories:

### Item-Level Features
Describe properties of the item to be placed:
- **Dimensions** $(w, h, d)$ in current orientation
- **Volume** and **aspect ratios** (e.g., $w/h$, capturing item "shape")
- **Weight** and **fragility class** (for load-bearing and stacking constraints)
- **Orientation constraints**: Set of allowed rotations $R_i \subseteq S_3$

### Placement-Level Features
Quantify the quality of placing an item at a specific candidate position:
- **Residual void**: Volume of empty space created around the item after placement
  $$
  V_{\text{void}} = V_{\text{EMS}} - V_{\text{item}}
  $$
  Minimizing void reduces fragmentation [PÁVTO08].

- **Support area**: Footprint overlap with items below or container floor. Higher support improves stability [GOGL16].

- **Wall contact**: Number of container faces the item touches. Maximizing contact tends to consolidate items [CPT08].

- **Clearance to edges**: Distance to container boundaries, relevant for accessibility constraints [BTC23].

- **Height penalty**: Vertical position $z$; lower placements typically preferred to maintain a compact profile.

### Global State Features
Capture the overall packing configuration:
- **Current utilization**: $\sum_{k \in \text{placed}} V_k / V_{\text{container}}$
- **Number of items remaining** to be packed
- **Bin count** (for multi-bin problems): penalizes opening new bins
- **Center-of-mass deviation**: Distance from container centroid, used for stability constraints [GOGL16]
- **Fragmentation metrics**: Number of disjoint empty regions or variance in heightmap values

These features are combined into **scoring functions** via weighted sums or lexicographic ordering. For example, Gonçalves and Resende (2013) [GR13] rank EMS candidates by:
$$
\text{score}(EMS, item) = \lambda_1 \cdot V_{\text{EMS}} - \lambda_2 \cdot V_{\text{void}} + \lambda_3 \cdot \text{support}(item, EMS)
$$
where weights $\lambda_i$ are hand-tuned or evolved via genetic algorithms.


## Learned Features in Neural Network Approaches

Neural network methods replace hand-crafted scoring with **end-to-end learned representations**. The network architecture and training objective implicitly determine which spatial patterns are important.

### Feature Inputs to Neural Scoring Models

Modern learning-based systems (Section 2.4) typically feed neural networks with a combination of:

1. **Item embeddings**: Normalized dimensions, weight, orientation flags, binary fragility indicators
2. **Candidate embeddings**: Normalized position $(x, y, z)$, available volume (for EMS), support area estimates
3. **Global context**: Container utilization, number of items packed/remaining, bin index
4. **Spatial encodings**: Heightmap (as image), voxelized occupancy grid, or graph-based representations

For example, the neural scoring model proposed in Section 2.4 accepts:

- **Item vector** $\mathbf{v}_{\text{item}} \in \mathbb{R}^d$: $[w, h, d, \text{weight}, \text{fragile}, \ldots]$
- **Placement vector** $\mathbf{v}_{\text{place}} \in \mathbb{R}^p$: $[x, y, z, V_{\text{void}}, A_{\text{support}}, \ldots]$
- **State vector** $\mathbf{v}_{\text{state}} \in \mathbb{R}^s$: $[\text{utilization}, n_{\text{remaining}}, \ldots]$

These are concatenated and passed through a multi-layer perceptron (MLP) to produce a scalar placement score. During training (via reinforcement learning or evolutionary search), the network learns which feature combinations correlate with high final utilization or low constraint violations.

### Attention-Based Context Aggregation

Beyond simple MLPs, recent work incorporates **attention mechanisms** [XGP+24] to model interactions between:

- **Items**: Which item dimensions are complementary for tight packing?
- **Candidates**: Which placements leave the most favorable residual space for future items?
- **Placed items**: Does the current configuration block high-value placements?

Transformer architectures process sets of item and EMS embeddings, computing attention weights that highlight relevant spatial relationships. For instance, GOPT [XGP+24] uses:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

where queries $Q$ are candidate EMS embeddings, keys $K$ and values $V$ are item embeddings. The attention scores reveal which items are "compatible" with which empty spaces, guiding the placement policy without explicit geometric reasoning.


## Convolutional and Graph-Based Spatial Encodings

When packing state is represented as a 2D heightmap or 3D voxel grid, **convolutional neural networks (CNNs)** extract spatial features hierarchically:

- **Low-level**: Edge detectors identify container boundaries and item interfaces
- **Mid-level**: Texture patterns capture repeating structures (e.g., uniform stacking)
- **High-level**: Spatial pooling aggregates information across the entire container, detecting global fragmentation or imbalance

Xiong et al. [XGP+24] apply 2D convolutions to heightmaps, followed by spatial pooling to produce a fixed-size feature vector. This vector is concatenated with item embeddings and processed by a Transformer to predict placement scores.

Alternatively, **graph neural networks (GNNs)** represent items and empty spaces as nodes in a spatial graph, with edges encoding proximity or containment relationships. GNNs propagate information through the graph structure, learning to reason about:

- **Support chains**: Which items depend on which for stability?
- **Blocking relationships**: Does placing item $i$ at position $p$ prevent placing item $j$ elsewhere?
- **Fragmentation**: Are empty regions disconnected or contiguous?

While promising, GNN-based approaches for 3D nesting remain less common than CNNs or attention-based methods due to the computational cost of graph construction and message-passing at each packing step.


## Non-Neural Feature-Based Methods

Not all effective feature extraction requires deep learning. Several state-of-the-art heuristics use simple, interpretable features with strong performance:

### Best-Fit and Worst-Fit Strategies

- **Best-Fit EMS**: Select the smallest EMS that can contain the item, minimizing wasted space [PÁVTO08]. Feature: $V_{\text{EMS}} - V_{\text{item}}$ (minimize).
- **Worst-Fit EMS**: Select the largest EMS, preserving large contiguous regions for future items [FB10]. Feature: $V_{\text{EMS}}$ (maximize).

These simple geometric features—available volume and residual void—are sufficient for competitive results on standard benchmarks when combined with effective item ordering (e.g., largest-volume-first).

### Support and Stability Metrics

Galrão Ramos et al. (2016) [GOGL16] introduce a **static equilibrium constraint**: an item is feasible only if its center of mass projects onto a support polygon with sufficient area. The support polygon is computed geometrically from the footprint overlap with items below. This binary feasibility check—rather than a continuous score—eliminates unstable placements without machine learning.

### Accessibility and LIFO Constraints

Bonet et al. (2023) [BTC23] model soft unloading constraints by assigning each item a **delivery priority** and penalizing placements that block earlier-priority items. The accessibility feature is a geometric reachability test: can item $i$ be removed without moving item $j$? This is computed via axis-aligned bounding box intersection tests, yielding a binary accessibility matrix used to filter candidates or add penalty terms to the objective function.


## Comparison: Handcrafted vs. Learned Features

| **Aspect**               | **Handcrafted Features**                          | **Learned Features (Neural)**                      |
|--------------------------|--------------------------------------------------|---------------------------------------------------|
| **Interpretability**      | High: each feature has clear geometric meaning   | Low: learned representations are opaque           |
| **Generalization**        | Brittle: fixed rules may not transfer across constraint sets | Strong: networks adapt to different distributions via training |
| **Computational Cost**    | Low: simple arithmetic on geometric primitives   | Moderate to High: forward passes through deep networks |
| **Data Requirements**     | None: features are manually designed             | High: requires large datasets or extensive simulation |
| **Flexibility**           | Limited: adding new constraints requires redesigning features | High: retraining with new reward function adapts automatically |

Hybrid approaches—using geometric features as input to lightweight neural scoring models—offer a promising middle ground, combining the efficiency and interpretability of handcrafted features with the adaptability of learned decision-making.


## Summary

Feature extraction in 3D bin packing serves to compress the complex spatial state into representations suitable for placement evaluation. Traditional methods rely on interpretable geometric features (void volume, support area, wall contact) combined via weighted scoring functions or lexicographic rules. Modern learning-based approaches embed items, candidates, and global state into neural networks—often augmented with heightmaps for spatial reasoning or attention mechanisms for relational context. Heightmaps in particular bridge the gap between discrete geometric heuristics and continuous visual encodings, enabling convolutional architectures to learn spatial patterns that generalize across problem instances.

The feature extraction framework proposed in this thesis (Chapter 4) combines compact geometric primitives from EMS-based candidate generation with a neural scoring model evolved via neurogenesis. This design aims to preserve the feasibility guarantees and computational efficiency of classical heuristics while leveraging the adaptability of evolved neural architectures to discover problem-specific inductive biases.
