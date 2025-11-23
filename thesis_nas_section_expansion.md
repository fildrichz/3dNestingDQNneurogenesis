# Neural Architecture Search - Expanded Section 3.3

## 3.3 Neural Architecture Search

Neural Architecture Search (NAS) automates the design of neural network architectures through systematic exploration of the architecture space. This section examines the evolution from foundational genetic algorithm approaches to modern efficient neural architecture search methods, with particular emphasis on genome representation strategies and their relevance to combinatorial optimization problems such as 3D bin packing.

### 3.3.1 Foundational Approaches: Genetic Algorithms for Neural Network Optimization

#### Ding, Xu, and Su (2010): Using Genetic Algorithms to Optimize Artificial Neural Networks

The foundational work by **Ding, Xu, and Su (2010)** [DXS10] establishes core principles for applying genetic algorithms (GAs) to neural network optimization. Their approach addresses both architectural search and hyperparameter tuning through evolutionary computation, providing a template that subsequent NAS methods have built upon.

**Core Methodology.** Ding et al. propose a comprehensive framework where both the network structure and training parameters are encoded as genes within a chromosome. Each individual in the population represents a complete neural network specification, including:

1. **Topology genes**: Number of hidden layers, neurons per layer, connectivity patterns
2. **Activation function genes**: Selection from available nonlinear functions (sigmoid, tanh, ReLU)
3. **Learning parameter genes**: Learning rate, momentum coefficient, weight decay
4. **Training strategy genes**: Batch size, number of epochs, early stopping criteria

The fitness function evaluates each candidate network based on validation performance, incorporating both accuracy and complexity penalties to avoid over fitting and excessive computational cost.

#### Genome Representation Types

Ding et al. identify three primary encoding schemes for representing neural network architectures genetically:

**1. Direct Encoding (Binary String Representation)**

The simplest approach uses binary strings where each bit or group of bits directly specifies a network component. For example:
- Bits 0-7: Number of hidden neurons in layer 1 (0-255)
- Bits 8-9: Activation function (00=sigmoid, 01=tanh, 10=ReLU)
- Bits 10-15: Learning rate exponent (10^-6 to 10^-1)

*Advantages*: Straightforward encoding and decoding, easy to implement standard GA operators.

*Disadvantages*: Fixed-length chromosomes limit flexibility; difficult to represent variable-depth architectures; high encoding redundancy.

**2. Graph-Based Encoding**

Networks are represented as directed acyclic graphs where nodes represent computational units (neurons/layers) and edges represent connections. The chromosome encodes the adjacency matrix or edge list along with node properties.

For a network with n potential nodes, the encoding includes:
- Connectivity matrix: n×n binary matrix or compressed edge list
- Node type vector: n-dimensional vector specifying neuron activation functions
- Edge weight initialization: Connection strength parameters

*Advantages*: Natural representation of neural network topology; supports variable architecture sizes; enables representation of skip connections and complex topologies.

*Disadvantages*: Variable-length genomes complicate crossover operations; larger search space; potential for invalid graphs requiring repair mechanisms.

**3. Parametric Encoding (Real-Valued Vectors)**

Continuous parameters are encoded as real-valued genes, particularly suitable for hyperparameters. A typical chromosome might be:

```
[n_layers, n_units_1, n_units_2, learning_rate, dropout_rate, ...]
```

where each gene is a floating-point number within a specified range. Discrete choices (e.g., activation functions) are encoded as categorical indices or one-hot vectors.

*Advantages*: Efficient for continuous hyperparameter optimization; allows gradient-like exploration through arithmetic crossover; natural representation for layer sizes and learning rates.

*Disadvantages*: Requires careful normalization and scaling; discrete architectural choices less naturally represented; crossover may produce invalid intermediate architectures.

#### Evolutionary Mechanisms in Ding et al.

The evolutionary process follows these steps:

**Initialization.** The initial population is generated using:
- *Random sampling*: Uniform or Gaussian sampling from valid parameter ranges
- *Heuristic seeding*: Including known good architectures (e.g., standard 3-layer MLPs)
- *Diversity enforcement*: Ensuring population covers different regions of the search space

**Selection.** Ding et al. employ tournament selection with elitism:
1. Randomly sample k individuals (typically k=3-7)
2. Select the fittest individual from the tournament
3. Repeat until parent pool is filled
4. Always preserve top e individuals (elite size e ≈ 0.05-0.10 × population size)

This balances exploration (through probabilistic selection) with exploitation (via elitism), preventing premature convergence while maintaining best solutions.

**Crossover Operators.** Different crossover strategies are applied based on gene type:

*Uniform Crossover* (for binary/discrete genes):
```
Parent 1: [1, 0, 1, 1, 0, 1]
Parent 2: [0, 1, 1, 0, 1, 0]
Mask:     [1, 0, 1, 0, 1, 0]
Offspring:[1, 1, 1, 0, 0, 0]
```

*Arithmetic Crossover* (for continuous genes):
```
Offspring = α × Parent1 + (1-α) × Parent2
```
where α ∈ [0,1] is randomly chosen or set to 0.5 for uniform blending.

*Structural Crossover* (for graph-based representations):
- Randomly select a crossover point in each parent graph
- Exchange subgraphs while maintaining connectivity
- Apply repair operators to ensure valid topologies

**Mutation Operators.** Mutations introduce variation and prevent stagnation:

1. **Structural mutations**:
   - Add/remove hidden layer (with probability p_add_layer ≈ 0.05)
   - Add/remove neurons from existing layers (p_modify_neurons ≈ 0.1)
   - Modify connectivity patterns (for graph encodings, p_modify_edges ≈ 0.1)

2. **Parametric mutations**:
   - Gaussian perturbation for continuous genes: gene' = gene + N(0, σ)
   - Random reset: Replace gene value with new random sample (p_reset ≈ 0.05)
   - Swap mutation for categorical genes: Change activation function, optimizer type

3. **Adaptive mutation rates**:
   Ding et al. note that mutation probability should decrease over generations:
   ```
   p_mutation(t) = p_init × exp(-λt)
   ```
   where t is the generation number, enabling broader exploration early and fine-tuning later.

**Fitness Evaluation.** Each candidate network is trained on a training set and evaluated on a validation set. The fitness function typically combines multiple objectives:

```
Fitness = w_acc × Accuracy - w_comp × Complexity - w_time × TrainingTime
```

where:
- Accuracy: Validation set performance (classification accuracy, MSE, etc.)
- Complexity: Number of parameters, FLOPs, or network depth
- TrainingTime: Actual wall-clock training time

The weights (w_acc, w_comp, w_time) are problem-dependent. For resource-constrained applications like embedded 3D bin packing solvers, w_comp and w_time would be set higher.

### 3.3.2 Genome Representation for Bin Packing Neural Scoring Models

Building upon Ding et al.'s foundation, we now specify genome encoding for the neural scoring models used in our 3D nesting framework. Recall from Section 2.2.2 and Table 2.1 that our scoring model genome must encode:

#### Compact Genome Structure

Our genome uses a **mixed-encoding strategy**, combining discrete choices with continuous hyperparameters:

```
Genome = [
  // Architecture genes (discrete)
  depth ∈ {2, 3, 4, 5, 6},
  hidden_width ∈ {64, 128, 256, 512},
  activation ∈ {ReLU, GELU, SiLU},
  normalization ∈ {none, LayerNorm, BatchNorm},
  residuals ∈ {off, on},
  context_aggregation ∈ {none, MLP_pooling, small_GNN},

  // Feature engineering genes (binary mask)
  feature_subset: BitArray[n_features],

  // Training hyperparameters (continuous)
  dropout ∈ [0.0, 0.3],
  weight_decay ∈ [0.0, 0.1],
  learning_rate_schedule ∈ {cosine, step, warmup}
]
```

**Encoding Details:**

1. **Discrete architectural choices** are encoded as integer indices:
   - `depth = 3` → 3 hidden layers
   - `activation = 1` → GELU (index into {ReLU=0, GELU=1, SiLU=2})

2. **Feature subset selection** uses a binary mask of length equal to the total number of engineered features:
   ```
   feature_subset = [1,1,0,1,0,0,1,1,...]
                     ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑
                     | | | | | | | └─ wall_contact
                     | | | | | | └─── support_area
                     | | | | | └───── residual_void
                     | | | | └─────── clearance
                     | | | └───────── item_height
                     | | └─────────── item_width
                     | └───────────── item_depth
                     └─────────────── EMS_volume
   ```

3. **Continuous hyperparameters** are stored as floating-point values within specified bounds.

#### Specialized Crossover for Mixed Genomes

Given the heterogeneous gene types, we apply different crossover strategies to different genome segments:

**Algorithm: Mixed-Type Crossover**
```
function CROSSOVER(parent1, parent2):
  offspring = new Genome()

  // 1. Uniform crossover for discrete genes
  for gene in [depth, hidden_width, activation, ...]:
    offspring.gene = RANDOM_CHOICE([parent1.gene, parent2.gene])

  // 2. Arithmetic crossover for continuous genes
  for gene in [dropout, weight_decay]:
    alpha = UNIFORM(0.3, 0.7)  # Biased towards center
    offspring.gene = alpha * parent1.gene + (1-alpha) * parent2.gene

  // 3. Two-point crossover for feature mask
  p1, p2 = RANDOM_TWO_POINTS(len(feature_subset))
  offspring.feature_subset[0:p1] = parent1.feature_subset[0:p1]
  offspring.feature_subset[p1:p2] = parent2.feature_subset[p1:p2]
  offspring.feature_subset[p2:end] = parent1.feature_subset[p2:end]

  return offspring
```

This hybrid approach respects the structure of different gene types: discrete architectural genes inherit complete specifications from one parent (avoiding invalid intermediate architectures), continuous parameters blend smoothly, and the feature mask undergoes block exchange to preserve feature co-occurrence patterns.

#### Specialized Mutation Operators

Mutations are applied probabilistically based on gene type:

**1. Architectural Mutations** (probability p_arch = 0.15):
```
if RANDOM() < p_arch:
  // Randomly select one architectural gene
  gene = RANDOM_CHOICE([depth, hidden_width, activation, ...])

  if gene == depth:
    offspring.depth = CLAMP(offspring.depth + RANDOM_CHOICE([-1, +1]), 2, 6)
  elif gene == hidden_width:
    offspring.hidden_width = RANDOM_CHOICE([64, 128, 256, 512])
  elif gene == activation:
    offspring.activation = RANDOM_CHOICE([ReLU, GELU, SiLU])
  // ... similar for other discrete genes
```

**2. Feature Subset Mutations** (probability p_feature = 0.10):
```
if RANDOM() < p_feature:
  // Flip 1-3 random bits in feature mask
  n_flips = RANDOM_INT(1, 3)
  for i in 1..n_flips:
    idx = RANDOM_INT(0, len(feature_subset)-1)
    offspring.feature_subset[idx] = NOT offspring.feature_subset[idx]
```

**3. Hyperparameter Mutations** (probability p_hyper = 0.20):
```
if RANDOM() < p_hyper:
  // Gaussian perturbation with adaptive step size
  for gene in [dropout, weight_decay]:
    if RANDOM() < 0.5:  # 50% chance per continuous gene
      sigma = (gene.max - gene.min) * 0.1  # 10% of range
      offspring.gene = CLAMP(
        offspring.gene + NORMAL(0, sigma),
        gene.min, gene.max
      )
```

**Mutation Rate Adaptation.** Following Ding et al.'s recommendation, mutation probabilities decrease over generations:

```
p_arch(t) = p_arch_init × max(0.5, exp(-0.01t))
p_feature(t) = p_feature_init × max(0.3, exp(-0.015t))
p_hyper(t) = p_hyper_init × max(0.7, exp(-0.005t))
```

This schedule allows broad architectural exploration early in evolution while emphasizing hyperparameter fine-tuning in later generations.

### 3.3.3 Evolution Beyond Ding et al.: Refinements and Extensions

While Ding et al. established foundational principles, subsequent work has addressed key limitations and inefficiencies in genetic algorithm-based NAS:

#### NeuroEvolution of Augmenting Topologies (NEAT)

**Stanley and Miikkulainen (2002)** [SM02] introduced NEAT, which addresses the competing conventions problem in neuroevolution—the issue that identical networks can have different genetic encodings, making crossover disruptive.

**Key Innovations:**

1. **Historical Markings**: Each gene (connection) is assigned a global innovation number when it first appears through mutation. During crossover, genes with matching innovation numbers are aligned, while disjoint and excess genes are inherited from the fitter parent.

2. **Incremental Growth**: Networks begin minimally (input → output only) and evolve complexity through:
   - *Add connection*: Create new edge between existing nodes
   - *Add node*: Split an existing connection, inserting a new node

3. **Speciation**: Population is divided into species based on genetic distance. Fitness sharing within species protects innovation:
   ```
   adjusted_fitness(i) = fitness(i) / Σ_j sh(distance(i,j))
   ```
   where sh(d) = 1 if d < δ_threshold, else 0.

**Relevance to Bin Packing:** NEAT's incremental complexification aligns well with our goal of evolving lightweight scoring models. Starting with simple architectures (single hidden layer) and selectively adding complexity based on packing performance prevents over-engineering while discovering problem-specific inductive biases.

#### Efficient Neural Architecture Search (ENAS)

While NEAT focuses on topology evolution for relatively small networks, **Efficient Neural Architecture Search (ENAS)** methods scale evolutionary and reinforcement learning approaches to deep learning architectures.

**Pham et al. (2018)** [PCZ+18] introduce parameter sharing across candidate architectures, dramatically reducing computational cost:

**Algorithm: ENAS Framework**
```
1. Train a large "super-network" containing all possible operations
2. Controller (RNN or RL agent) samples child architectures from super-network
3. Child architectures inherit weights from super-network (no training from scratch)
4. Measure child validation performance quickly
5. Update controller to favor high-performing architectures
```

Instead of training each candidate network independently (requiring thousands of GPU-hours), ENAS trains once and samples many times. This reduces search cost from ~3000 GPU-days (NAS with RL) to ~0.5 GPU-days.

**Real et al. (2017)** [RMS+17] demonstrate large-scale evolution of image classifiers using simple evolutionary strategies:
- Tournament selection with mutation-only (no crossover)
- Aging mechanism: Remove old individuals to prevent stagnation
- Binary tournament: Sample 2 individuals, keep fitter, mutate, replace worse

Their ablation studies show that **mutation-only evolution with aging** outperforms more complex evolutionary strategies, suggesting simpler is often better for NAS.

**Relevance to Bin Packing:** Parameter sharing is less applicable to our domain (each genome defines a unique small network), but the mutation-only + aging strategy aligns well with our population-based search over compact scoring models.

#### Multi-Objective Neural Architecture Search

**Booysen and Bosman (2024)** [BB24] extend evolutionary NAS to multi-objective optimization, balancing:
1. Task accuracy (primary objective)
2. Model size (parameter count, memory footprint)
3. Inference time (FLOPs, wall-clock latency)

They use **non-dominated sorting** (from NSGA-II) to maintain a Pareto front of solutions, allowing practitioners to select architectures based on deployment constraints.

**Multi-Objective Fitness for Bin Packing:** Our fitness function similarly balances multiple objectives:

```
Pareto-dominance criteria:
  f1 = Packing utilization (maximize)
  f2 = Bin count (minimize)
  f3 = Constraint violations (minimize)
  f4 = Inference time per placement (minimize)

A solution x dominates y if:
  x is no worse than y in all objectives, AND
  x is strictly better than y in at least one objective
```

Maintaining a Pareto front allows selecting architectures tailored to specific deployment scenarios:
- **Online packing** (time-critical): Prefer fast, lightweight models even if slightly less accurate
- **Offline optimization** (batch processing): Accept slower, larger models for maximum packing quality

#### Crossover Operators for Real-Coded GAs

**Takahashi and Kita (2001)** [TK01] introduce the **BLX-α crossover** for real-valued genes, which we apply to continuous hyperparameters:

```
function BLX_ALPHA_CROSSOVER(parent1.gene, parent2.gene, alpha=0.5):
  c_min = min(parent1.gene, parent2.gene)
  c_max = max(parent1.gene, parent2.gene)
  range = c_max - c_min

  // Offspring sampled from expanded interval
  offspring.gene = UNIFORM(
    c_min - alpha * range,
    c_max + alpha * range
  )

  // Clamp to valid bounds
  return CLAMP(offspring.gene, gene.min_value, gene.max_value)
```

The α parameter controls exploration: α=0 gives uniform crossover (offspring within parent range), α>0 allows extrapolation beyond parents, enabling discovery of better values not present in either parent.

**Empirical studies** [TK01] show α=0.5 provides good balance between exploitation (staying near parents) and exploration (venturing beyond current population) for continuous optimization problems.

### 3.3.4 Best Practices from NAS Literature

**White et al. (2023)** [WSS+23] synthesize insights from analyzing 1000 NAS papers, distilling evidence-based recommendations:

**1. Compact, Problem-Specific Search Spaces Outperform General Ones**

Rather than searching over all possible deep learning architectures, restricting the search space to designs appropriate for the problem domain dramatically improves both search efficiency and final performance.

*Implication for Bin Packing:* Our genome (Table 2.1) deliberately excludes irrelevant architectural choices (e.g., very deep networks, exotic attention mechanisms) and focuses on:
- Shallow-to-medium depth networks (2-6 layers) suitable for candidate scoring
- Feature engineering genes to discover which geometric/spatial features matter most
- Compact hidden sizes (64-512 units) appropriate for real-time placement decisions

**2. Multi-Fidelity Evaluation Accelerates Search**

Evaluating candidates on:
- Smaller proxy tasks (fewer items, smaller containers)
- Shorter training budgets (fewer epochs)
- Downsampled validation sets

provides noisy but faster fitness estimates early in search. High-fidelity evaluation is reserved for promising candidates in later generations.

*Implementation Strategy:*
```
Generation 1-10:  Evaluate on small instances (5-10 items), 50 training steps
Generation 11-25: Evaluate on medium instances (15-25 items), 200 steps
Generation 26+:   Evaluate on full benchmark suite, full training budget
```

**3. Population Diversity is Critical**

Maintaining diversity prevents premature convergence to local optima. Effective strategies include:
- *Explicit diversity metrics*: Penalize genomes too similar to existing population members
- *Niching and speciation*: Divide population into subpopulations optimized for different objective trade-offs
- *Crowding distance*: In multi-objective optimization, favor solutions in less-crowded regions of objective space

**4. Warm-Starting with Heuristic Designs**

Rather than pure random initialization, seeding the initial population with hand-designed architectures (e.g., standard MLPs, successful architectures from related tasks) accelerates convergence and provides a strong baseline.

*For Bin Packing:* We seed with:
- Simple 3-layer MLP using all available features
- Shallow wide network (2 layers, 512 units) for rapid inference
- Deep narrow network (5 layers, 128 units) for capacity-constrained deployment
- Architecture from [XGP+24] adapted to our feature set

### 3.3.5 Challenges in Applying NAS to Discrete Optimization

While NAS has achieved remarkable success in supervised learning tasks (image classification, language modeling), applying these methods to discrete combinatorial optimization introduces unique challenges:

**1. Sparse, Delayed Reward Signal**

Unlike supervised learning where every prediction receives immediate gradient feedback, reinforcement learning for bin packing only provides:
- **Terminal rewards**: Packing utilization known only at episode end
- **Infrequent intermediate rewards**: Constraint violations discovered only when they occur
- **High variance**: Stochastic packing policies lead to high variance in episode returns

*Mitigation Strategies:*
- Use multiple packing episodes per fitness evaluation and average returns
- Implement reward shaping: Small rewards for geometrically sensible placements (e.g., low centers of gravity, high wall contact)
- Multi-episode curriculum: Gradually increase problem difficulty (item count, container complexity)

**2. Architecture Sensitivity to Problem Structure**

The optimal neural architecture depends heavily on:
- **Item distribution**: Homogeneous vs. heterogeneous item sets favor different feature importances
- **Container geometry**: Tall narrow containers vs. wide flat containers affect spatial reasoning requirements
- **Constraint profile**: Stability-constrained packing may benefit from graph neural networks to model item support relationships

This sensitivity means a single architecture may not generalize across all packing scenarios. Our neurogenesis approach can evolve:
- **Specialist architectures** optimized for specific problem classes (e.g., dense homogeneous packing)
- **Generalist architectures** with good average performance across problem distributions
- **Mixture-of-experts** where multiple specialized models are evolved and a gating mechanism selects the appropriate one per instance

**3. Computational Budget Constraints**

Each fitness evaluation requires:
1. Instantiating and initializing the candidate network
2. Training on a set of packing instances (expensive)
3. Evaluating on validation instances
4. Computing fitness metrics

With population size P=50 and G=30 generations, this requires 1500 network training runs—potentially prohibitive if each training run takes hours.

*Mitigation via Surrogate Models:* [WSS+23, EMH19]
- After initial random search (generations 1-5), train a surrogate model (e.g., Gaussian process, random forest) to predict fitness from genome
- Use surrogate to pre-screen candidates: only evaluate high-predicted-fitness genomes
- Periodically retrain surrogate with newly evaluated candidates
- Reduces evaluations by 50-70% while maintaining search quality

**4. Evaluation of Transfer Learnability**

A candidate architecture that achieves high fitness on one set of packing instances may fail to transfer to:
- Larger problem sizes (more items)
- Different item size distributions
- Additional constraints not present during evolution

*Robustness Evaluation Protocol:*
- Test evolved architectures on held-out instance distributions
- Measure performance degradation as problem characteristics shift
- Report not just best fitness, but confidence intervals and worst-case performance

### 3.3.6 Gap Analysis: Neurogenesis for 3D Bin Packing

Current NAS literature exhibits several gaps when applied to constrained combinatorial optimization:

**Gap 1: Limited Work on NAS for Discrete Optimization**

The overwhelming majority of NAS research targets supervised learning (image classification, object detection, segmentation). Relatively few works [LZYZ23, ZLZ+21] apply architecture search to combinatorial problems, and none specifically combine neurogenesis with deep reinforcement learning for 3D nesting with realistic constraints.

**Gap 2: Lack of Constraint-Aware Architecture Design**

Standard NAS methods optimize for accuracy and computational efficiency but do not explicitly incorporate problem constraints into the search process. For bin packing:
- Stability constraints → May benefit from architectures that model center-of-mass, support polygons
- Accessibility constraints (LIFO) → May require sequence/temporal modeling (RNN/LSTM components)
- Multi-bin optimization → May benefit from hierarchical or graph-structured policies

Our work will investigate **constraint-conditioned architecture search**: allowing the genome to include constraint-specific modules (e.g., stability checking subnetworks) that are activated only when those constraints are present.

**Gap 3: Absence of Evolved Feature Subsets**

Existing learning-based bin packing methods [XGP+24, WFH+23] use fixed, hand-engineered feature sets. While Table 3.1 compares handcrafted vs. learned features at the representation level, no prior work evolves which subset of engineered features to use.

Our feature subset genome (binary mask in Table 2.1) addresses this gap, enabling the evolutionary process to discover:
- Which geometric features (residual void, support area, wall contact) are most predictive
- Which global state features (current utilization, remaining items) improve placement decisions
- Whether item-specific features (dimensions, weight, fragility) should be included or are redundant

**Gap 4: Neurogenesis at Multiple Timescales**

Ding et al. and subsequent NAS methods operate at a single timescale: evolve architecture → train → evaluate → next generation. For reinforcement learning problems, we can consider evolution at multiple timescales:

- **Inner loop (fast)**: RL training updates network weights via gradient descent (TD learning, policy gradients)
- **Middle loop (medium)**: Hyperparameter tuning adjusts learning rate, exploration, batch size
- **Outer loop (slow)**: Architectural evolution via genetic algorithm

**Lamarckian evolution** (trained weights inherited) vs. **Darwinian evolution** (weights reset each generation) represents a key design choice:
- *Lamarckian*: Offspring inherit parent's trained weights, then continue training → Faster convergence, risk of losing diversity
- *Darwinian*: Each offspring trains from scratch → Maintains diversity, expensive evaluation

Most modern NAS uses Darwinian evolution, but for RL where training is expensive, **partial Lamarckian inheritance** (transfer weights from parent, reinitialize final layers) may offer a beneficial middle ground. This remains an open research question.

### 3.3.7 Summary and Relevance to This Work

The progression from Ding et al.'s foundational genetic algorithm framework through NEAT's incremental complexification to modern ENAS methods establishes a rich toolkit for automated architecture design. Key takeaways for our neurogenesis-based 3D nesting system:

1. **Genome Encoding**: Our mixed-type representation (discrete architectural genes, continuous hyperparameters, binary feature mask) follows Ding et al.'s parametric encoding philosophy while incorporating domain-specific structure.

2. **Evolutionary Operators**: We adopt specialized crossover (uniform for discrete, arithmetic for continuous, two-point for feature mask) and mutation strategies (adaptive rates, gene-type-specific perturbations) proven effective in prior NAS literature.

3. **Multi-Objective Optimization**: Following [BB24], we maintain a Pareto front balancing packing quality, inference speed, and model complexity—enabling deployment-specific architecture selection.

4. **Efficiency Mechanisms**: Multi-fidelity evaluation, surrogate-assisted search, and mutation-only evolution (per [RMS+17]) are employed to manage computational budget while maintaining search effectiveness.

5. **Problem-Specific Search Space**: Informed by [WSS+23]'s best practices, we restrict our architectural search space to designs appropriate for scoring candidate placements rather than exploring all possible deep learning architectures.

The next chapter describes our proposed method, which instantiates these neurogenesis principles within a complete learning-guided 3D nesting framework combining EMS-based candidate generation, neural scoring models with evolved architectures, and Deep Q-Network reinforcement learning.

---

## References for Expanded Section

[DXS10] Shifei Ding, Li Xu, and Chunyang Su, "Using Genetic Algorithms to Optimize Artificial Neural Networks," Journal of Convergence Information Technology, vol. 5, no. 8, pp. 54-62, 2010.

[SM02] Kenneth O. Stanley and Risto Miikkulainen, "Evolving Neural Networks through Augmenting Topologies," Evolutionary Computation, vol. 10, no. 2, pp. 99-127, 2002.

[PCZ+18] Hieu Pham, Melody Y. Guan, Barret Zoph, Quoc V. Le, and Jeff Dean, "Efficient Neural Architecture Search via Parameter Sharing," ICML 2018.

(Additional references already cited in the original thesis text: [EMH19], [WSS+23], [LSX+20], [RMS+17], [BB24], [TK01], [XGP+24], [WFH+23], [LZYZ23], [ZLZ+21])
