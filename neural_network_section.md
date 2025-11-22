# 3.2 Neural Network Architectures for Bin Packing

The application of deep learning to combinatorial optimization has transformed bin packing from a domain of hand-crafted heuristics to one where neural networks learn placement strategies directly from experience. This section reviews the key neural network architectures employed in modern bin packing systems, with particular focus on deep reinforcement learning approaches that treat packing as a sequential decision-making problem.

## 3.2.1 Deep Q-Networks (DQN)

Deep Q-Networks revolutionized reinforcement learning by combining Q-learning with deep neural networks, enabling agents to learn policies directly from high-dimensional state representations [MKS+15]. For bin packing, DQN provides a natural framework: the agent learns to select placements (actions) that maximize long-term packing efficiency (cumulative reward).

### The Q-Learning Foundation

Q-learning seeks to learn an action-value function $Q(s, a)$ that estimates the expected cumulative reward of taking action $a$ in state $s$ and following the optimal policy thereafter. The optimal Q-function satisfies the **Bellman equation**:

$$
Q^*(s, a) = \mathbb{E}_{s'}\left[r + \gamma \max_{a'} Q^*(s', a')\right]
$$

where $r$ is the immediate reward, $\gamma \in [0, 1]$ is the discount factor, and $s'$ is the next state. Traditional Q-learning uses a table to store $Q(s, a)$ for every state-action pair, which is infeasible for large or continuous state spaces.

### Neural Function Approximation

DQN replaces the Q-table with a neural network $Q(s, a; \theta)$ parameterized by weights $\theta$. The network is trained to minimize the **temporal difference (TD) error**:

$$
\mathcal{L}(\theta) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{D}}\left[\left(r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta)\right)^2\right]
$$

where $\mathcal{D}$ is a **replay buffer** storing past experiences $(s, a, r, s')$, and $\theta^-$ are the parameters of a **target network** that is updated periodically to stabilize training.

**Key Innovations:**

1. **Experience Replay**: Past transitions $(s, a, r, s')$ are stored in a buffer and randomly sampled during training. This breaks temporal correlations in sequential data and improves sample efficiency.

2. **Target Network**: A separate network with frozen weights $\theta^-$ is used to compute the TD target $r + \gamma \max_{a'} Q(s', a'; \theta^-)$. The target network is updated every $C$ steps by copying $\theta \to \theta^-$, preventing divergence caused by the "moving target" problem.

3. **$\varepsilon$-greedy Exploration**: The agent selects the greedy action $a = \arg\max_a Q(s, a; \theta)$ with probability $1 - \varepsilon$, and a random action with probability $\varepsilon$. Epsilon is annealed from $\varepsilon_{\text{start}}$ to $\varepsilon_{\text{end}}$ over training.

### Application to Bin Packing

In the bin packing domain, the DQN framework is instantiated as follows:

- **State $s$**: The current packing configuration, encoded as feature vectors (Section 3.1.2), heightmaps, or voxel grids. Includes:
  - Container state: occupied positions, utilization, heightmap
  - Current item: dimensions, weight, orientation constraints
  - Remaining items: count, volume distribution

- **Action $a$**: A discrete choice of placement candidate. For $K$ candidate positions (e.g., extreme points or EMS), $a \in \{1, \ldots, K\}$. Some systems also include orientation selection as part of the action space.

- **Reward $r$**: Immediate feedback signal. Common reward structures:
  - **Utilization increment**: $r = V_{\text{item}} / V_{\text{container}}$ (reward for packing volume)
  - **Sparse completion reward**: $r = 0$ during packing, $r = \text{final utilization}$ at episode end
  - **Penalty-based**: $r = -1$ per bin opened (minimizes bin count)
  - **Shaped rewards**: Additional terms for support quality, fragmentation, or constraint violations

- **Episode Termination**: When all items are placed or no feasible placements remain for the current item.

**Network Architecture:**

The Q-network typically consists of:

1. **Input encoding**: State features $\mathbf{s} \in \mathbb{R}^d$ (item + container + global features)
2. **Shared trunk**: Multi-layer perceptron (MLP) or convolutional layers to process spatial representations
3. **Action-value heads**: Outputs $Q(s, a)$ for each candidate action $a$

For $K$ candidate placements, one can use:

- **Separate forward passes**: Encode each $(state, candidate_k)$ pair and evaluate $Q(s, a_k)$ separately
- **Single forward pass**: Encode state once, then score all candidates via attention or separate heads

Example network structure (from packing_with_dqncore2_enhanced.py):

```
Input: [item_features, container_state, candidate_position]
  ↓
FC layers with ReLU: [512, 512, 256]
  ↓
Output: Q(s, a) ∈ ℝ
```

### Advantages

- **End-to-end learning**: No need to manually design scoring functions; the network learns what features correlate with good outcomes
- **Generalization**: Networks trained on diverse problem instances can generalize to unseen items and container sizes
- **Implicit constraint handling**: Through negative rewards, the agent learns to avoid infeasible placements

### Limitations

- **Overestimation bias**: The $\max$ operator in the Bellman update introduces positive bias, leading to overoptimistic Q-values and unstable learning (addressed by Double DQN)
- **Discrete action spaces**: DQN requires discretization of continuous placement positions, limiting precision
- **Sample inefficiency**: Requires many episodes to learn effective policies, especially for large combinatorial action spaces
- **Hyperparameter sensitivity**: Learning rate, replay buffer size, target update frequency, and epsilon schedule require careful tuning


## 3.2.2 Double DQN

Standard DQN suffers from **overestimation bias**: the $\max$ operator in the Bellman target tends to select overestimated action values, leading to inflated Q-values and unstable learning [HGS16]. Double DQN addresses this by decoupling action selection from action evaluation.

### The Overestimation Problem

In DQN, the target is computed as:

$$
y_{\text{DQN}} = r + \gamma \max_{a'} Q(s', a'; \theta^-)
$$

The same network (or target network) is used both to **select** the best action ($\arg\max$) and **evaluate** its value ($Q$). If the Q-function has estimation errors, the $\max$ will preferentially select actions with **positive errors**, causing systematic overestimation.

### Double Q-Learning Solution

Double DQN uses the **online network** to select the action and the **target network** to evaluate it:

$$
y_{\text{Double}} = r + \gamma Q(s', \arg\max_{a'} Q(s', a'; \theta); \theta^-)
$$

**Breakdown:**

1. **Action selection**: Use current weights $\theta$ to find the best action:
   $$a^* = \arg\max_{a'} Q(s', a'; \theta)$$

2. **Action evaluation**: Use target weights $\theta^-$ to evaluate that action:
   $$Q(s', a^*; \theta^-)$$

This separation reduces overestimation because positive errors in the online network (selecting $a^*$) are unlikely to correlate with positive errors in the target network (evaluating $a^*$).

### Implementation

Double DQN requires only a **single-line change** to DQN:

```python
# Standard DQN target
target = reward + gamma * q_target(next_state).max(dim=1)[0]

# Double DQN target
best_actions = q_online(next_state).argmax(dim=1)  # Select with online
target = reward + gamma * q_target(next_state).gather(1, best_actions.unsqueeze(1)).squeeze(1)  # Evaluate with target
```

### Impact on Bin Packing

For bin packing, Double DQN provides:

- **More stable Q-values**: Especially important when reward signals are sparse (e.g., only at episode end)
- **Better convergence**: Reduced oscillation in learned policies during training
- **Improved sample efficiency**: Faster learning with fewer episodes

Empirical results [HGS16] show that Double DQN often achieves higher final performance and more consistent learning curves compared to standard DQN, particularly in domains with large action spaces—such as selecting among hundreds of candidate placements in a complex 3D packing scenario.


## 3.2.3 Transformer Architectures

Transformers, originally developed for natural language processing [VSP+17], have become a dominant architecture for modeling relational structures in combinatorial optimization [KW19, XGP+24]. Unlike convolutional or recurrent networks, Transformers process **sets** of elements (items, placements, spatial regions) via **self-attention**, learning which relationships matter for decision-making.

### The Attention Mechanism

The core operation is **scaled dot-product attention**:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

where:
- $Q \in \mathbb{R}^{n \times d_k}$: **Queries** (what we're looking for)
- $K \in \mathbb{R}^{m \times d_k}$: **Keys** (what's available)
- $V \in \mathbb{R}^{m \times d_v}$: **Values** (content to aggregate)
- $d_k$: Dimension of keys/queries (scaling prevents gradient vanishing)

**Interpretation**: For each query $q_i$, compute attention weights over all keys $k_j$ via softmax of dot products, then aggregate the corresponding values $v_j$. High $q_i \cdot k_j$ means element $i$ "attends to" element $j$.

### Multi-Head Attention

To capture diverse relationships, Transformers use **multiple attention heads**:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O
$$

where each head learns different attention patterns:

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

Different heads might focus on:
- Spatial proximity (items near each other)
- Size compatibility (items with similar dimensions)
- Constraint relationships (fragile items avoiding heavy items above)

### Transformer Encoder

A full Transformer encoder layer consists of:

1. **Multi-head self-attention**: Elements attend to each other
2. **Feed-forward network**: Per-element MLP
3. **Residual connections** and **layer normalization** for stable training

$$
\begin{align}
\mathbf{Z} &= \text{LayerNorm}(\mathbf{X} + \text{MultiHead}(\mathbf{X}, \mathbf{X}, \mathbf{X})) \\
\mathbf{H} &= \text{LayerNorm}(\mathbf{Z} + \text{FFN}(\mathbf{Z}))
\end{align}
$$

Multiple encoder layers are stacked to form deep representations.

### Application to Bin Packing

Transformers are well-suited to bin packing because packing involves **relational reasoning** over sets:

- **Items**: Which items pack well together?
- **Placements**: Which candidate positions create favorable configurations?
- **Constraints**: Which items are blocked by current placements?

**Example: GOPT [XGP+24]**

Xiong et al. (2024) propose a **Packing Transformer** for online 3D bin packing with robotic arms:

1. **Item Encoding**: Each item $i$ is embedded as $\mathbf{e}_i \in \mathbb{R}^d$ (dimensions, weight, orientation)

2. **EMS Encoding**: Each empty maximal space $j$ is embedded as $\mathbf{c}_j \in \mathbb{R}^d$ (position, available volume)

3. **Cross-Attention**: Items attend to EMS candidates:
   $$
   \mathbf{h}_j = \text{Attention}(Q=\mathbf{c}_j, K=\{\mathbf{e}_i\}, V=\{\mathbf{e}_i\})
   $$

   This computes, for each candidate position $j$, which items are relevant (compatible size, waiting to be packed).

4. **Heightmap Integration**: A 2D heightmap is processed by a CNN to extract spatial features $\mathbf{f}_{\text{spatial}}$, then concatenated with item/EMS embeddings.

5. **Policy Head**: The attended representations are passed to an MLP that outputs placement probabilities or Q-values.

**Benefits for Bin Packing:**

- **Permutation invariance**: Attention is invariant to the order of items/candidates, making the network robust to different input sequences
- **Variable-length sets**: Can process different numbers of items or candidates without architectural changes
- **Learned relationships**: Discovers which spatial patterns matter (e.g., "place large items first near corners")

### Comparison with CNNs and MLPs

| **Architecture** | **Strengths**                                      | **Weaknesses**                                   |
|------------------|---------------------------------------------------|--------------------------------------------------|
| **MLP**          | Simple, fast, good for fixed-size feature vectors | Cannot model spatial structure or variable sets  |
| **CNN**          | Excellent for grid-based representations (heightmaps) | Requires spatial discretization, limited relational reasoning |
| **Transformer**  | Handles variable sets, learns relational patterns | Computationally expensive ($O(n^2)$ in sequence length), needs large datasets |


## 3.2.4 Set Transformers

Standard Transformers are permutation-equivariant (output order matches input order) but produce one output per input element. For combinatorial optimization, we often need a **permutation-invariant** summary of a set, followed by set-to-element or set-to-set operations. **Set Transformers** [LSK+19] extend Transformers with specialized attention mechanisms for set-structured data.

### Motivation

In bin packing, we often need to:

1. **Aggregate item information**: Summarize all remaining items into a global context vector
2. **Score variable-size sets**: Evaluate sets of candidates of different sizes
3. **Maintain permutation invariance**: The set $\{\text{item}_1, \text{item}_2\}$ should be processed identically to $\{\text{item}_2, \text{item}_1\}$

Set Transformers provide building blocks for these operations.

### Induced Set Attention Block (ISAB)

Standard self-attention has $O(n^2)$ complexity for a set of size $n$. **ISAB** reduces this to $O(nm)$ by introducing $m$ **inducing points** $\mathbf{I} \in \mathbb{R}^{m \times d}$ (learnable parameters):

1. Aggregate input set $\mathbf{X}$ into inducing points:
   $$
   \mathbf{H} = \text{Attention}(Q=\mathbf{I}, K=\mathbf{X}, V=\mathbf{X})
   $$

2. Project inducing points back to output:
   $$
   \mathbf{Y} = \text{Attention}(Q=\mathbf{X}, K=\mathbf{H}, V=\mathbf{H})
   $$

This is analogous to bottleneck layers in CNNs: compress information through inducing points, then expand.

### Pooling by Multihead Attention (PMA)

To produce a **fixed-size** output from a variable-size set, PMA uses $k$ learnable **seed vectors** $\mathbf{S} \in \mathbb{R}^{k \times d}$:

$$
\mathbf{Z} = \text{Attention}(Q=\mathbf{S}, K=\mathbf{X}, V=\mathbf{X})
$$

The seeds attend to the entire input set, producing $k$ summary vectors. For $k=1$, this gives a **single global representation** of the set, similar to global average pooling but learned.

### Set-to-Set Functions

A Set Transformer encoder is built by stacking ISAB blocks, followed by PMA:

$$
\mathbf{X} \xrightarrow{\text{ISAB}} \mathbf{H}_1 \xrightarrow{\text{ISAB}} \mathbf{H}_2 \xrightarrow{\text{PMA}} \mathbf{Z}
$$

The output $\mathbf{Z}$ is a fixed-size representation of the input set, invariant to permutations.

### Application to Bin Packing

**Item Set Encoding**: Given a set of items $\{\mathbf{v}_i\}$ to be packed:

1. Embed each item: $\mathbf{e}_i = \text{Embed}(\mathbf{v}_i)$ (dimensions, weight, etc.)
2. Apply ISAB layers: Learn inter-item relationships (e.g., clustering by size)
3. Pool with PMA: Produce a global item set representation $\mathbf{z}_{\text{items}}$

**Candidate Scoring**: Given a set of candidate placements $\{\mathbf{p}_j\}$:

1. Embed each candidate: $\mathbf{c}_j = \text{Embed}(\mathbf{p}_j)$ (position, available volume)
2. Attend to item set:
   $$
   \mathbf{h}_j = \text{Attention}(Q=\mathbf{c}_j, K=\mathbf{z}_{\text{items}}, V=\mathbf{z}_{\text{items}})
   $$
3. Score candidates: $Q(s, a_j) = \text{MLP}(\mathbf{h}_j)$

**Advantages:**

- **Handles variable-size inputs**: Number of items or candidates can vary
- **Scalable**: ISAB reduces quadratic complexity to linear in the number of inducing points
- **Interpretable attention**: Attention weights reveal which items influence which placements

**Limitations:**

- **Inducing point tuning**: Choosing the number of inducing points $m$ requires experimentation
- **Less common**: Fewer pre-trained models or established best practices compared to standard Transformers


## 3.2.5 Single-Problem Neural Networks

Most neural approaches aim to learn **general policies** that generalize across problem instances. An alternative paradigm is **single-problem learning**: training a neural network (or evolving an architecture) specifically for **one fixed problem instance** [ACG21, VMP20].

### Motivation

For recurring packing tasks (e.g., warehouses packing the same set of product SKUs daily, or shipping containers with similar item distributions), a specialized network may outperform a general one by exploiting instance-specific structure.

### Training Paradigm

Given a fixed bin packing instance $I = (W, D, H, \{v_1, \ldots, v_n\})$:

1. **Network initialization**: Random weights or a pre-trained general model
2. **Episode generation**: Sample random item orderings (or learn the ordering jointly)
3. **Reinforcement learning**: Train via DQN, policy gradient, or evolutionary methods to maximize utilization on $I$
4. **Convergence**: Train until performance plateaus (often 1000s of episodes)

**Key Difference from General Learning:**

- **No generalization pressure**: Network can overfit to $I$, learning idiosyncrasies (e.g., "item 3 always goes in the bottom-left corner")
- **Faster convergence**: Smaller effective hypothesis space → quicker training
- **Optimal for static problems**: If the item set never changes, generalization is unnecessary

### Use Cases

1. **Static packing problems**: Manufacturing with fixed product lines
2. **Warm-starting for similar problems**: Train on a representative instance, then fine-tune for variations
3. **Benchmarking**: Measure the performance ceiling of neural approaches on standard test instances

### Example: Evolved Architectures for Single Instances

The neurogenesis approach (Chapter 4) can be applied in single-problem mode:

1. **Genetic algorithm**: Evolve network architectures (layer sizes, connections) optimized for a specific dataset
2. **Fitness evaluation**: Train each candidate architecture on the target instance for $N$ episodes, measure final utilization
3. **Selection**: Breed high-performing architectures

The resulting network is highly specialized but may discover inductive biases (e.g., specific attention patterns or feature combinations) that happen to generalize to similar instances.

### Limitations

- **No generalization**: Catastrophic failure on different item sets or container sizes
- **Training cost**: Full RL training per instance can be expensive
- **Overfitting risk**: Network may exploit spurious correlations specific to $I$ (e.g., item ordering artifacts)


## 3.2.6 Graph Neural Networks (GNNs)

Graph Neural Networks provide a natural framework for encoding the **relational structure** of bin packing: items, empty spaces, and spatial relationships can be represented as nodes and edges in a graph [SZL+22, ZWL+24].

### Graph Representation of Packing State

A packing configuration can be modeled as a graph $G = (V, E)$ where:

**Nodes $V$:**
- **Item nodes**: Each placed item $i$ with features $\mathbf{x}_i = [w_i, h_i, d_i, x_i, y_i, z_i, \text{weight}_i]$
- **EMS nodes**: Each empty maximal space $j$ with features $\mathbf{x}_j = [x_j, y_j, z_j, w_j, h_j, d_j]$
- **Container node**: Global state $\mathbf{x}_{\text{bin}} = [\text{utilization}, \text{num\_items\_placed}]$
- **Item-to-place node**: The current item awaiting placement

**Edges $E$:**
- **Support edges**: $(i, j)$ if item $i$ supports item $j$ (geometric contact)
- **Proximity edges**: $(i, j)$ if items $i$ and $j$ are within distance $\epsilon$
- **Containment edges**: $(i, \text{EMS}_k)$ if placing item $i$ in EMS $k$ is feasible
- **Global edges**: All nodes connected to the container node

### Message Passing Framework

GNNs learn node representations by iteratively **aggregating information** from neighbors. At each layer $\ell$:

1. **Message computation**: For each edge $(i, j)$:
   $$
   \mathbf{m}_{ij}^{(\ell)} = \text{Message}(\mathbf{h}_i^{(\ell-1)}, \mathbf{h}_j^{(\ell-1)}, \mathbf{e}_{ij})
   $$
   where $\mathbf{h}_i$ is the hidden state of node $i$, and $\mathbf{e}_{ij}$ are edge features (e.g., distance, contact area).

2. **Aggregation**: Collect messages from all neighbors:
   $$
   \mathbf{a}_i^{(\ell)} = \text{Aggregate}\left(\{\mathbf{m}_{ji}^{(\ell)} : j \in \mathcal{N}(i)\}\right)
   $$
   Common aggregators: sum, mean, max, or attention-weighted sum.

3. **Update**: Combine aggregated messages with current state:
   $$
   \mathbf{h}_i^{(\ell)} = \text{Update}(\mathbf{h}_i^{(\ell-1)}, \mathbf{a}_i^{(\ell)})
   $$
   Typically a neural network (MLP or GRU).

After $L$ layers, each node has a representation $\mathbf{h}_i^{(L)}$ that aggregates information from its $L$-hop neighborhood.

### GNN Variants for Bin Packing

**Graph Convolutional Networks (GCN)** [KW17]:
$$
\mathbf{h}_i^{(\ell)} = \sigma\left(W^{(\ell)} \sum_{j \in \mathcal{N}(i) \cup \{i\}} \frac{\mathbf{h}_j^{(\ell-1)}}{\sqrt{|\mathcal{N}(i)||\mathcal{N}(j)|}}\right)
$$

Simple and efficient, but fixed aggregation (mean).

**Graph Attention Networks (GAT)** [VCC+18]:
$$
\mathbf{h}_i^{(\ell)} = \sigma\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij} W^{(\ell)} \mathbf{h}_j^{(\ell-1)}\right)
$$

where $\alpha_{ij}$ are learned attention weights. Allows the network to focus on relevant neighbors (e.g., items that directly support the current item).

**Relational GCN (R-GCN)** [SKB+18]:

Uses different weight matrices for different edge types:
$$
\mathbf{h}_i^{(\ell)} = \sigma\left(\sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)} \frac{1}{|\mathcal{N}_r(i)|} W_r^{(\ell)} \mathbf{h}_j^{(\ell-1)}\right)
$$

where $\mathcal{R}$ is the set of relation types (support, proximity, containment). Useful when different relationships have different semantics.

### Application to Placement Selection

Given a graph representation of the current packing state:

1. **Graph construction**: Build $G$ from placed items, EMS candidates, and the item to place
2. **Message passing**: Run $L$ GNN layers to compute node embeddings $\{\mathbf{h}_i^{(L)}\}$
3. **Readout**:
   - **Node-level**: Score each EMS node: $Q(s, a_j) = \text{MLP}(\mathbf{h}_j^{(L)})$
   - **Graph-level**: Pool all EMS embeddings and decode: $Q(s, a) = \text{MLP}(\text{Pool}(\{\mathbf{h}_j^{(L)}\}))$

**Advantages:**

- **Explicit relational structure**: Encodes support, adjacency, and blocking relationships directly
- **Inductive bias**: Message passing naturally captures spatial dependencies (e.g., "supported items depend on supporters")
- **Interpretability**: Attention weights or message values reveal which relationships drive decisions

**Challenges:**

- **Graph construction overhead**: Building and updating the graph at each packing step is computationally expensive
- **Dynamic graphs**: Adding/removing nodes (items/EMS) requires re-indexing and re-initializing embeddings
- **Scalability**: Large packing problems (100s of items) lead to dense graphs with $O(n^2)$ edges
- **Limited adoption**: GNNs for bin packing remain less common than CNNs or Transformers, with fewer established architectures

### Comparison with Other Architectures

| **Architecture** | **Relational Reasoning** | **Spatial Structure** | **Complexity**       | **Adoption in Packing** |
|------------------|-------------------------|-----------------------|----------------------|-------------------------|
| **MLP**          | None (flat features)    | None                  | $O(d)$               | High (simple baselines)|
| **CNN**          | Local (via convolutions)| Explicit (grid-based) | $O(WHD)$             | Medium (heightmaps)    |
| **Transformer**  | Global (via attention)  | Learned (positional)  | $O(n^2)$             | Growing (GOPT, etc.)   |
| **GNN**          | Explicit (graph edges)  | Graph topology        | $O(n^2)$ (edges)     | Low (emerging)         |


## 3.2.7 Deep Reinforcement Learning Context

The neural architectures discussed above are typically integrated within **deep reinforcement learning (DRL)** frameworks, where the network serves as a function approximator for value functions or policies. This section briefly contextualizes the role of neural networks in the broader DRL pipeline.

### Value-Based vs. Policy-Based Methods

**Value-Based (DQN, Double DQN)**:

- Learn $Q(s, a)$ or $V(s)$
- Policy is derived implicitly: $\pi(s) = \arg\max_a Q(s, a)$
- Off-policy: Can learn from experiences collected by different policies
- Sample efficient but limited to discrete action spaces

**Policy-Based (REINFORCE, PPO, A3C)**:

- Learn $\pi(a|s; \theta)$ directly
- Sample actions from the policy: $a \sim \pi(\cdot|s)$
- On-policy: Requires recent experience from the current policy
- Works with continuous action spaces but higher variance

**Actor-Critic (A2C, PPO)**:

- Combine both: Learn $\pi(a|s)$ (actor) and $V(s)$ or $Q(s, a)$ (critic)
- Critic reduces variance of policy gradient estimates
- State-of-the-art for many RL tasks

### Policy Gradient Methods

For bin packing with large or continuous action spaces (e.g., precise $(x, y, z)$ positioning), policy gradients may be preferred. The **policy gradient theorem** provides a gradient estimator:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot G_t\right]
$$

where $G_t = \sum_{t'=t}^T \gamma^{t'-t} r_{t'}$ is the return from time $t$. Modern algorithms like **Proximal Policy Optimization (PPO)** [SWD+17] add clipping or trust regions to stabilize updates.

**Application to Packing:**

- **Continuous placement**: Output a probability distribution over $(x, y, z, \text{orientation})$
- **Item ordering**: Learn a policy over item permutations (e.g., via pointer networks)
- **Multi-agent**: Multiple robots packing in parallel, each with a local policy

### Sample Efficiency and Curriculum Learning

DRL often requires millions of environment steps to converge. For bin packing:

- **Simulated environments**: Fast physics-free collision checks enable rapid data collection
- **Curriculum learning**: Train on easy instances (few items, simple shapes) before hard ones
- **Transfer learning**: Pre-train on synthetic data, fine-tune on real-world distributions

### Model-Based RL

Most packing systems use **model-free** RL (learn $Q$ or $\pi$ directly from experience). An alternative is **model-based RL**:

1. Learn a dynamics model: $\hat{s}_{t+1} = f(s_t, a_t)$
2. Plan using the model (e.g., Monte Carlo Tree Search)
3. Update the model from real experience

**Challenges for Packing:**

- High-dimensional state (3D geometry) makes model learning difficult
- Discrete item placements lead to discontinuous dynamics

Model-based methods remain rare in bin packing, though they show promise for long-horizon planning [KSH+20].


## 3.3 Hybrid Architectures and Integration

Modern packing systems often **combine** multiple neural components:

**CNN + Transformer (GOPT [XGP+24])**:
- CNN processes heightmap → spatial features
- Transformer processes item/EMS embeddings → relational features
- Concatenate and feed to policy/value head

**GNN + DQN**:
- GNN encodes packing state as graph
- DQN uses GNN embeddings to score actions
- Experience replay and target networks stabilize training

**Attention + Heuristics**:
- Neural network scores candidates
- Heuristic filters infeasible placements
- Combine scores: $\text{final\_score} = \alpha \cdot \text{NN\_score} + (1-\alpha) \cdot \text{heuristic\_score}$

This hybrid approach leverages the strengths of both learned and hand-crafted methods: neural networks adapt to data distributions, while heuristics guarantee feasibility and provide inductive biases.


## Summary

Neural network architectures for bin packing have evolved from simple MLPs to sophisticated models incorporating attention, graph structure, and deep reinforcement learning. **DQN** provides a foundational framework for learning placement policies via value function approximation, while **Double DQN** addresses overestimation bias for more stable training. **Transformers** and **Set Transformers** enable relational reasoning over variable-size sets of items and candidates, learning which spatial patterns matter without explicit geometric programming. **Single-problem networks** specialize to individual instances, trading generalization for instance-specific performance. **Graph Neural Networks** encode explicit relational structure (support, proximity, blocking), though at higher computational cost.

The choice of architecture depends on the problem characteristics:

- **Small, fixed state spaces**: MLP-based DQN
- **Spatial grid representations**: CNN + DQN
- **Variable-size item/candidate sets**: Transformer or Set Transformer
- **Explicit relational constraints**: GNN
- **Recurring static problems**: Single-problem specialized networks

Hybrid architectures combining multiple paradigms (e.g., CNN for spatial processing + Transformer for relational reasoning) represent the current state-of-the-art, balancing expressiveness, computational cost, and sample efficiency. The neurogenesis framework proposed in this thesis (Chapter 4) extends this landscape by evolving neural architectures tailored to the specific inductive biases of 3D bin packing, searching over network topologies to discover effective combinations of spatial, relational, and value-based processing.
