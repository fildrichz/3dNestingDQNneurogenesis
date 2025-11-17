# Extended Thesis Description

This thesis proposes a learning-guided system for 3D nesting of convex items that optimizes utilization under non-overlap and container-bound constraints. It aims to explore the use of neurogenesis in order to appraise its viability for this class of problems.

## Architecture and Approach

The system employs **Deep Q-Network (DQN)** reinforcement learning combined with **Empty Maximal Spaces (EMS)** heuristics to enumerate feasible loading plans. The neural architecture, designated **QNetworkEnhanced**, consists of five key components:

1. **State Encoder**: An MLP processing an 8-dimensional observation vector capturing global packing state (packed volume ratio, items remaining, maximum EMS capacities, average heightmap height, weight utilization, bins used)

2. **Heightmap CNN**: A novel 2-layer convolutional network that extracts spatial features from 7×7 patches of the 2D heightmap around each placement position, learning local topology patterns (corners, walls, valleys) to inform placement decisions

3. **Action Encoder**: An MLP processing 25-dimensional action features representing EMS-based placements (item dimensions, rotated dimensions, EMS position and size, slack space, utilization deltas, tightness metrics, and heightmap features)

4. **Attention Mechanism**: Either standard Transformer or Set Transformer architecture enabling the network to reason about relationships between candidate actions, with configurable attention heads and inducing points

5. **Q-Value Head**: A multi-layer perceptron combining state and action embeddings to output scalar Q-values for ranking placement candidates

## Neurogenesis via Genetic Algorithm

Model architectures are evolved using a genetic algorithm based on Ding et al. (2010). The **NetworkGenome** encodes 11 evolvable architectural parameters including:

- **Multiplicative genes** (powers of 2): hidden dimensions (64-1024), encoder layers (1-8), attention heads (2-16), inducing points (8-128), heightmap patch size (3-15)
- **Discrete genes**: attention type (standard/set_transformer/none), CNN channel configurations, dropout rates (0.0-0.2), activation functions (ReLU/GELU/SiLU)

The fitness function is multi-objective:
```
Fitness = 0.70 × Utilization + 0.20 × (1 - Bins_Penalty) + 0.10 × (1 - Complexity_Penalty)
```

Evolution proceeds with tournament selection (size 3), elitism (top 2 preserved), uniform crossover, and adaptive mutation (20% base rate with diversity boosting). Typical configurations use population size 20 over 10 generations, with each genome evaluated over 20 training episodes using reduced replay buffers (50k transitions) for memory efficiency.

## State Representation and Heuristics

### EMS-Based State Exploration

The system employs the **Empty Maximal Spaces (EMS)** heuristic to guarantee collision-free placements while efficiently reducing the search space. For each bin, the top-1000 largest EMS are maintained and sorted by volume. After each placement, EMS are updated via 6-way difference computation (creating left, right, front, back, bottom, top slices) with dominated space pruning to remove fully-contained volumes.

For each state, candidate actions are enumerated across all bins, EMS, remaining items, and 6 possible orientations, with constraint checking for:
- **Physical constraints**: EMS fit, container bounds, gravity simulation
- **Weight capacity**: Maximum weight per bin (800-1000 kg)
- **Incompatibility**: Items that cannot share bins
- **Positive affinity**: Items that must be packed together (with proactive checking)
- **Relative positioning**: Heavy items cannot be placed atop lighter items
- **Center of mass**: Constraint satisfaction for stability

When actions exceed the maximum (128), random sampling ensures diverse exploration without computational bottleneck.

### Feature Engineering

**Items, containers, and candidates** are represented with compact local and global features:

- **State features (8D)**: Packed volume ratio, items remaining, maximum EMS capacities (x,y,z), average heightmap height, weight utilization, bins used
- **Action features (25D)**: Original and rotated item dimensions, EMS corner position, EMS dimensions, slack space, delta utilization, tightness, container aspect ratios, heightmap features (position height, local average), item weight, remaining capacity, bin context
- **Heightmap patches (7×7)**: Normalized height values extracted around placement positions with adaptive resolution (container_width // 20) and zero-padding boundaries

## Reward Function and Training

The system uses **potential-based reward shaping** with per-bin potential:
```
Φ(s) = bin_utilization + 0.3 × EMS_quality
Reward = γ × Φ(s') - Φ(s)
```

where `EMS_quality = (average_of_top_3_EMS_volumes / bin_volume)^(1/3)` provides intermediate feedback about space fragmentation.

Additional bonuses incentivize complete packing (+0.3), bin efficiency (+0.2 × (1 - bins/max_bins)), and bin balancing (+0.01 for below-average item counts). Penalties apply for incomplete packing (-0.2 × items_left/total) and constraint violations.

**Training employs Double DQN with n-step returns** (n=15) for improved credit assignment in sparse reward settings:
- Discount factor γ=0.992
- Learning rate 1e-4 (Adam optimizer)
- Batch size 128
- Replay buffer 200k-400k transitions
- Target network hard updates every 500 steps
- ε-greedy exploration: ε=1.0 → 0.05-0.15 with linear decay
- Gradient clipping at 1.0
- Smooth L1 (Huber) loss

## Benchmark and Evaluation

We **survey the state of the art and build a reproducible benchmark** using:

- **Real-world 3D bin packing problems** (3dBPP_1 through 3dBPP_12): 38-54 items per problem with varying constraint combinations in 1500×1500×1500mm containers
- **Thpack dataset**: 9 instances
- **BR dataset**: 10 instances
- **Elhedhli dataset**: Additional benchmark instances

**Evaluation reports**:
- **Primary metrics**: Utilization (total_volume_packed / (bins_used × bin_volume)), bin count, items placed
- **Secondary metrics**: Per-bin utilization and weight distributions, constraint satisfaction rate, episode return, training loss
- **Runtime**: Episode execution time and training convergence
- **GA-specific**: Fitness scores, network complexity (millions of parameters), population diversity

## Key Contributions

This work makes several novel contributions to learning-based 3D bin packing:

1. **First application of heightmap CNNs** to 3D bin packing, enabling learned spatial reasoning about local topology
2. **Attention mechanisms over action sets** for reasoning about placement alternatives
3. **Proactive constraint handling** preventing affinity violations before placement (not reactive)
4. **Genetic algorithm for architecture search** automating network structure optimization beyond manual hyperparameter tuning
5. **Potential-based reward shaping** with EMS quality metric for theoretically sound intermediate feedback
6. **N-step returns (n=15)** addressing sparse reward challenges in sequential packing
7. **Comprehensive real-world constraint handling**: weight capacity, incompatibility, affinity, relative positioning, center of mass

The system demonstrates the viability of combining classical heuristics (EMS) with modern deep reinforcement learning (DQN + attention) and automated architecture search (neurogenesis) for constrained 3D bin packing problems.
