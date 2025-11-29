# Chapter 4: Proposed Method

This chapter presents our proposed method for solving the 3D bin packing problem with complex constraints. We begin by providing an overview of the system architecture, followed by detailed explanations of each component and the rationale behind our design choices.

## 4.1 Overview

Our approach combines geometric heuristics with deep reinforcement learning to address the multi-bin 3D packing problem defined in Section 2.1. The system follows a two-stage decision process:

1. **Candidate Generation**: The Empty Maximal Spaces (EMS) heuristic generates a set of feasible placement positions for each item-bin pair.
2. **Candidate Scoring**: A Deep Q-Network (DQN) scores each candidate placement, and the highest-scoring valid action is selected.

This hybrid architecture leverages the strengths of both approaches: EMS provides geometric reasoning and constraint filtering, while the neural network learns to optimize long-term packing quality through reinforcement learning.

The overall system architecture is illustrated in Figure 4.1. At each decision step, the environment enumerates all valid actions (bin, item, EMS, rotation combinations), extracts state features and heightmap patches for each action, and passes them through the Q-network to obtain Q-values. The action with the highest Q-value is executed, placing the item and updating the bin state.

## 4.2 Reinforcement Learning Framework

### 4.2.1 Why Reinforcement Learning?

Unlike supervised learning approaches that require large labeled datasets of optimal solutions (as discussed in Section 3.3), the 3D bin packing problem lacks such training data. The problem is NP-hard, and computing optimal solutions is intractable for realistic problem sizes. Even near-optimal solutions from heuristics are expensive to generate and may not generalize well across different problem instances.

Reinforcement learning addresses this challenge by learning directly from interaction with the environment. As described in Section 2.3, RL agents learn through trial and error, receiving rewards based on the quality of their packing decisions. This eliminates the need for pre-computed optimal solutions and allows the agent to discover effective strategies through self-play.

Furthermore, the sequential nature of bin packing—where each placement decision affects future possibilities—aligns naturally with the Markov Decision Process (MDP) framework that underlies reinforcement learning. The agent learns to maximize cumulative reward over the entire packing sequence, not just immediate placement quality.

### 4.2.2 Deep Q-Networks

We employ Deep Q-Networks (DQN), a value-based reinforcement learning algorithm introduced by Mnih et al. (2015) and discussed in Section 3.4.2. DQN learns a Q-function Q(s, a) that estimates the expected cumulative reward of taking action a in state s and following the optimal policy thereafter.

The key advantages of DQN for our problem include:

- **Sample Efficiency**: Experience replay allows the agent to learn from past experiences multiple times, improving sample efficiency.
- **Stability**: Target networks provide stable learning targets, preventing the oscillations common in Q-learning.
- **Generalization**: Deep neural networks can generalize across similar states, learning patterns rather than memorizing individual configurations.

### 4.2.3 Double DQN

We adopt the Double DQN variant (van Hasselt et al., 2016) to address Q-value overestimation. Standard DQN uses the max operator for both action selection and evaluation, which introduces a positive bias:

```
Q(s, a) ← r + γ max_{a'} Q_target(s', a')
```

Double DQN decouples these operations, using the online network for action selection and the target network for evaluation:

```
Q(s, a) ← r + γ Q_target(s', argmax_{a'} Q_online(s', a'))
```

This reduces overestimation and leads to more accurate Q-value estimates, particularly important in our domain where many actions may have similar immediate rewards but different long-term consequences.

## 4.3 State and Action Space

### 4.3.1 State Representation

The state representation captures both global packing progress and local geometric information:

**Global State Features** (scalar values):
- Current bin utilization
- Number of items placed
- Number of items remaining
- Current bin index
- Total number of bins
- Bin dimensions and weight capacity

These features provide context about overall packing efficiency and progress, allowing the agent to adjust its strategy based on how much of the problem remains.

**Item Features** (per-item vectors):
- Dimensions (width, depth, height)
- Weight
- Incompatibility flags
- Affinity group ID
- Relative positioning constraints

The item features are processed through a permutation-invariant encoder (described in Section 4.5.2) to handle the variable number of remaining items.

### 4.3.2 Action Space and EMS Heuristic

**Why Empty Maximal Spaces?**

As discussed in Section 3.2.1, two main geometric heuristics exist for 3D bin packing: Extreme Points (EP) and Empty Maximal Spaces (EMS). We choose EMS for several critical advantages:

1. **Collision-Free by Construction**: Each EMS represents a maximal empty rectangular volume. Any item placement within an EMS is guaranteed not to collide with existing items, eliminating the need for collision checking at decision time.

2. **Efficient Updates**: When an item is placed, we only need to update the EMS set once—splitting intersected spaces and removing dominated ones. With Extreme Points, collision checks must be performed for every candidate action at every decision step, which is computationally expensive.

3. **Bounded Action Space**: EMS naturally prunes infeasible placements. If an item is too large to fit in any EMS, it is automatically excluded from the action set. This keeps the action space manageable.

4. **Quality Metric**: The size and shape of available EMS spaces provides a geometric quality metric (described in Section 4.4.3) that can guide learning through reward shaping.

The computational efficiency of EMS is particularly important for reinforcement learning, which requires millions of environment interactions during training. The collision-free guarantee and single-update-per-placement property significantly accelerate training.

**Action Encoding**

Each action is represented as an 8-tuple:
```
(bin_idx, item_idx, ems_idx, rotation_idx, ems, size, weight, item_id)
```

We extract a 25-dimensional feature vector for each action:
- EMS corner position (x, y, z) normalized by container dimensions
- EMS dimensions (w, d, h) normalized by container dimensions
- Item dimensions (w, d, h) in the selected rotation, normalized
- Item weight normalized by bin capacity
- Bin utilization after placement
- Fit metrics: (item_width / ems_width, item_depth / ems_depth, item_height / ems_height)
- Space efficiency: (item_volume / ems_volume)
- Flags: is_corner_placement, is_bottom_placement, is_wall_adjacent
- Number of supporting boxes below
- Center of mass distance from bin center
- Remaining bin weight capacity after placement

These features capture both geometric properties (position, fit, support) and global context (utilization, capacity), enabling the network to reason about placement quality.

### 4.3.3 Heightmap Representation

In addition to the action features, we extract a 7×7 heightmap patch around each placement position. The heightmap is a 2D grid where each cell stores the maximum occupied height at that (x, y) location in the bin.

**Why Heightmaps?**

Heightmaps provide crucial spatial context that scalar features cannot capture:

1. **Local Geometry**: A patch around the placement position reveals the terrain—whether the location is in a corner, against a wall, in a valley, or on a plateau.
2. **Support Patterns**: The height variation indicates how well the new item will be supported and whether it might create unstable configurations.
3. **Future Packing Potential**: Placing an item in a valley creates a flat surface for future items, while placing on a peak creates fragmentation.

This local spatial information complements the global action features, allowing the network to make placement decisions based on both immediate geometric fit and long-term packing potential.

The 7×7 patch size provides sufficient context (±3 cells around the placement) while keeping the computational cost manageable. Patches are normalized by container height to ensure values are in the [0, 1] range.

## 4.4 Reward Design

### 4.4.1 The Sparse Reward Problem

Bin packing is naturally a sparse reward problem. The primary objective—minimizing the number of bins—only provides feedback at the end of an episode when all items are placed. Intermediate placement decisions receive no signal about their quality.

This sparsity creates severe learning challenges:
- **Credit Assignment**: Which of the hundreds of placement decisions led to needing an extra bin?
- **Exploration**: Random exploration rarely produces good solutions, providing little learning signal.
- **Slow Learning**: The agent may take millions of episodes to discover reward-producing behaviors.

As discussed in Section 3.4.3, reward shaping addresses this by providing intermediate rewards that guide learning while preserving the optimal policy.

### 4.4.2 Potential-Based Reward Shaping

We employ potential-based reward shaping (Ng et al., 1999), which augments the environment reward with a shaping function based on state potential:

```
r_shaped(s, a, s') = r_env(s, a, s') + γ · Φ(s') - Φ(s)
```

where Φ(s) is the potential function. This formulation guarantees that the optimal policy under the shaped reward is identical to the optimal policy under the original reward, as proven by Ng et al.

**Our Potential Function**

We define the potential as a combination of bin utilization and EMS quality:

```
Φ(s) = bin_utilization(s) + 0.3 · ems_quality(s)
```

- **Bin Utilization**: The ratio of occupied volume to total bin capacity. This encourages tight packing and efficient space usage.

- **EMS Quality**: A measure of how well-shaped the remaining empty spaces are (detailed below). This discourages fragmentation.

The coefficient 0.3 was chosen empirically to balance immediate packing efficiency (utilization) with future packing potential (EMS quality).

### 4.4.3 EMS Quality Metric

Fragmentation—creating many small, irregularly-shaped empty spaces—is a major challenge in bin packing. To discourage this, we define an EMS quality metric:

```
ems_quality(bin) = (average_volume_top3_largest_ems)^(1/3) / bin_height
```

We use the cube root to convert volume back to a linear scale comparable to height. By focusing on the top-3 largest spaces rather than all spaces, we emphasize maintaining a few large packing opportunities rather than many tiny crevices.

This metric provides a continuous signal: placements that preserve large, usable spaces receive higher rewards than those that create fragmentation.

### 4.4.4 Terminal Reward

Upon successfully packing all items, the agent receives a terminal reward based on the number of bins used:

```
r_terminal = 100 - 20 · (num_bins_used - 1)
```

This provides a strong signal for bin minimization while still rewarding valid solutions. The shaped intermediate rewards guide the agent toward configurations that achieve this terminal goal.

## 4.5 Neural Network Architecture

Our Q-network architecture processes three types of inputs: global state features, per-action features, and per-action heightmap patches. These are combined through specialized modules and an attention mechanism to produce Q-values for each candidate action.

### 4.5.1 Heightmap CNN

**Why Convolutional Neural Networks?**

As discussed in Section 3.3.1, Convolutional Neural Networks excel at extracting spatial features from grid-structured data. The heightmap patches are precisely such data: a 2D grid where spatial patterns (corners, edges, valleys, plateaus) indicate geometric properties.

Standard fully-connected networks would not exploit the spatial structure—they treat each grid cell independently and must learn from scratch that adjacent cells are related. CNNs, through their local receptive fields and weight sharing, inherently capture spatial patterns.

**Architecture**

Our heightmap CNN processes each 7×7 patch through the following layers:

```
Input: (1, 7, 7) single-channel heightmap patch
  ↓
Conv2d(1 → 32, kernel=3x3, padding=1) + ReLU
  ↓
Conv2d(32 → 64, kernel=3x3, padding=1) + ReLU
  ↓
AdaptiveAvgPool2d(1x1)  # Global average pooling
  ↓
Flatten → Linear(64 → 64) + ReLU
  ↓
Output: 64-dimensional embedding
```

The two convolutional layers extract increasingly abstract spatial features: the first layer detects basic patterns (edges, gradients), while the second combines these into higher-level structures (corners, valleys). Global average pooling aggregates spatial information into a fixed-size representation, and the final linear layer produces a 64-dimensional embedding.

This design is lightweight yet effective, processing each heightmap patch in parallel (one per action) with minimal computational cost.

### 4.5.2 Set Transformer for Actions

**Why Attention Mechanisms?**

The number of valid actions varies dramatically across states—from a few dozen to several hundred depending on the number of remaining items, available EMS spaces, and constraint satisfaction. Traditional approaches either:

1. **Pad to fixed size**: Wastes computation on padding tokens.
2. **Process sequentially**: Loses parallelism and inter-action relationships.

As discussed in Section 3.3.2, Transformer architectures solve this through self-attention, which allows each action to attend to all others and learn relational patterns. In bin packing, this is crucial: the value of placing item A in position 1 depends on whether item B has better alternative placements.

**Set Transformer**

We employ the Set Transformer architecture (Lee et al., 2019), specifically the Induced Set Attention Block (ISAB), which achieves permutation invariance—essential since action order is arbitrary.

The ISAB uses a set of learnable inducing points to reduce computational complexity from O(N²) to O(NM), where M is the number of inducing points (we use M=32). This allows efficient processing of large action sets.

**Architecture**

```
Input: (A, d_action) action features, where A varies
  ↓
ISAB(d_action, num_heads=4, num_inducing=32)
  ↓
ISAB(d_action, num_heads=4, num_inducing=32)
  ↓
Output: (A, d_action) contextual action embeddings
```

Each action's embedding is updated based on all other actions, allowing the network to reason about action interdependencies and competition for resources.

### 4.5.3 Complete Q-Network Architecture

The complete architecture integrates all components:

```
1. State Encoder:
   Global state → MLP(3 layers, [128, 128, 64]) → state_embedding (64-dim)

2. Action Encoder:
   Action features (25-dim) → MLP(2 layers, [128, 128]) → action_embedding (128-dim)

3. Heightmap Processing:
   For each action: heightmap_patch (7×7) → CNN → heightmap_embedding (64-dim)

4. Feature Fusion:
   Concatenate: [action_embedding, heightmap_embedding, state_embedding]
   → combined_features (128 + 64 + 64 = 256-dim per action)

5. Attention:
   combined_features (A, 256) → Set Transformer ISABs → contextualized_features (A, 256)

6. Q-Value Head:
   For each action: contextualized_features → MLP(2 layers, [128, 1]) → Q(s, a)
```

This architecture allows the network to:
- Extract spatial patterns from heightmaps (CNN)
- Reason about action relationships (Set Transformer)
- Integrate global state context (state embedding broadcast to all actions)
- Produce Q-values that account for both local geometry and global optimization

## 4.6 Training Strategy

### 4.6.1 Experience Replay

We use experience replay with a buffer size of 100,000 transitions. Each transition stores:
```
(state, action, reward, next_state, done, action_mask)
```

The action_mask is crucial for handling variable action spaces: it indicates which actions are valid in the next state, ensuring the target Q-value computation only considers feasible actions.

Replay provides two benefits:
1. **Sample Efficiency**: Each environment interaction can be reused for multiple gradient updates.
2. **Decorrelation**: Sampling random minibatches breaks temporal correlations, stabilizing learning.

### 4.6.2 N-Step Returns

Standard Q-learning uses 1-step returns:
```
Q(s, a) ← r + γ max_{a'} Q(s', a')
```

We employ n-step returns (n=15) to propagate terminal rewards more rapidly:
```
Q(s, a) ← Σ_{i=0}^{n-1} γ^i r_{t+i} + γ^n max_{a'} Q(s_{t+n}, a')
```

This is particularly important given the sparse terminal reward for bin minimization. With 1-step returns, the terminal reward would require hundreds of episodes to propagate back to early decisions. N-step returns accelerate this credit assignment, allowing the agent to learn which early placements lead to bin savings.

### 4.6.3 Target Network Updates

The target network is updated every 1,000 training steps using a soft update with τ=0.005:
```
θ_target ← τ θ_online + (1 - τ) θ_target
```

This provides stable learning targets while allowing gradual incorporation of improvements from the online network.

### 4.6.4 Exploration Strategy

We use ε-greedy exploration with exponential decay:
- Initial ε = 1.0 (pure exploration)
- Final ε = 0.05 (5% exploration)
- Decay over 500,000 steps

Early exploration helps discover diverse packing strategies, while later exploitation refines the learned policy.

### 4.6.5 Curriculum Learning

Training begins with simpler problem instances (fewer items, fewer constraints) and gradually increases difficulty. This curriculum approach helps the agent master basic packing before tackling complex constraint satisfaction.

## 4.7 Neural Architecture Search with Neurogenesis

### 4.7.1 Motivation

The neural architecture described in Section 4.5 contains numerous hyperparameters: network depths, layer widths, number of attention heads, CNN channels, activation functions, and more. Manual hyperparameter tuning is time-consuming and may miss better configurations.

As discussed in Section 3.5, Neural Architecture Search (NAS) automates this process. We employ an evolutionary approach inspired by neurogenesis—the growth of new neurons in biological neural networks.

### 4.7.2 Genome Encoding

Each candidate architecture is encoded as a genome—a dictionary of hyperparameters:

```python
genome = {
    # State encoder
    'state_depth': [2, 3, 4],
    'state_width': [64, 128, 256],

    # Action encoder
    'action_depth': [1, 2, 3],
    'action_width': [64, 128, 256],

    # Heightmap CNN
    'cnn_channels': [[16, 32], [32, 64], [64, 128]],

    # Attention
    'num_heads': [2, 4, 8],
    'num_inducing': [16, 32, 64],
    'num_isab_blocks': [1, 2, 3],

    # Q-value head
    'q_depth': [1, 2, 3],
    'q_width': [64, 128, 256],

    # Training
    'learning_rate': [1e-4, 3e-4, 1e-3],
    'gamma': [0.95, 0.99, 0.995],
    'n_step': [1, 5, 15],

    # Activation function
    'activation': ['relu', 'elu', 'leaky_relu']
}
```

### 4.7.3 Evolutionary Algorithm

The evolutionary process follows these steps:

1. **Initialization**: Generate a population of N random genomes.
2. **Evaluation**: Train each genome's architecture for K episodes and evaluate on a validation set.
3. **Selection**: Select the top M performers as parents.
4. **Crossover**: Create offspring by combining parent genomes.
5. **Mutation**: Randomly modify offspring genomes with probability p_mut.
6. **Iteration**: Repeat for G generations.

This process explores the architecture space efficiently, leveraging successful configurations through crossover and discovering novel designs through mutation.

### 4.7.4 Fitness Function

Genomes are evaluated on multiple criteria:
- **Packing performance**: Average number of bins used on validation instances.
- **Training stability**: Variance in episode returns during training.
- **Computational cost**: Number of parameters and inference time.

The fitness function balances these objectives, favoring architectures that pack efficiently without excessive complexity.

## 4.8 System Integration

The complete system integrates all components described above:

1. **Problem instances** are loaded with items, bins, and constraints.
2. The **EMS-based environment** generates valid actions at each step.
3. **Heightmap patches** are extracted around each action's placement position.
4. The **Q-network** scores all actions based on features, heightmaps, and attention.
5. The **highest-scoring valid action** is executed.
6. The **reward shaping function** provides learning signals.
7. **Experience replay** stores transitions for training.
8. **Double DQN with n-step returns** updates the Q-network.
9. **Neurogenesis** searches for optimal architectures.

This integration creates a complete learning system that addresses the challenges of 3D bin packing: constraint satisfaction (EMS + environment), long-term optimization (RL + reward shaping), spatial reasoning (heightmap CNN), and action relationships (Set Transformer).

## 4.9 Summary

Our proposed method combines classical geometric heuristics (EMS) with modern deep reinforcement learning (DQN, Transformers) to solve constrained 3D bin packing. The key innovations include:

1. **EMS for efficiency**: Collision-free candidate generation with single-update-per-placement.
2. **Reward shaping for learning**: Potential-based shaping using utilization and EMS quality to address sparse rewards.
3. **Heightmap CNN for spatial reasoning**: Convolutional networks extract local geometric patterns.
4. **Set Transformer for action relationships**: Attention mechanisms model interdependencies between placements.
5. **N-step returns for credit assignment**: Multi-step bootstrapping propagates terminal rewards rapidly.
6. **Neurogenesis for architecture search**: Evolutionary optimization discovers effective network designs.

Each design choice is motivated by specific problem characteristics identified in Chapter 2 and builds upon state-of-the-art methods reviewed in Chapter 3. The result is a system that learns effective packing strategies without labeled training data, handles complex real-world constraints, and optimizes for long-term bin minimization.
