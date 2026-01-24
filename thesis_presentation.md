# Thesis Defense Presentation
## 3D Bin Packing with Deep Reinforcement Learning and Neuroevolution

**Slides: 7 main + appendix | Time: 5-8 minutes**

---

# Slide 1: Title

## 3D Bin Packing with Deep Reinforcement Learning and Neuroevolution

**Diploma Thesis Defense**

*Combining DQN, Genetic Algorithms, and Neural Architecture Search for Constrained 3D Packing*

> **Speaker Notes:**
> - Introduce yourself and thesis topic
> - Mention this combines multiple AI techniques for a classic optimization problem
> - Preview: problem definition, methodology, results

---

# Slide 2: Problem Definition & Motivation

## The 3D Bin Packing Problem

**Definition:** Given a set of rectangular items and bins of fixed dimensions, pack all items into the minimum number of bins while respecting constraints.

**Nesting:** The optimization of item placement considering geometry, orientations (6 rotations), and spatial relationships between items.

### Why This Problem Matters
| Application | Industry |
|------------|----------|
| Container loading | Logistics & Shipping |
| Warehouse storage | E-commerce & Retail |
| Cutting stock | Manufacturing |

### Challenges
- **NP-Hard** - no polynomial-time algorithm exists
- **Large action space** - 128+ valid placements per step
- **Complex constraints** - weight, incompatibilities, stability

> **Speaker Notes:**
> - Clarify nesting = optimizing spatial arrangement (reviewer noted missing definition)
> - Emphasize practical importance - billions saved in logistics annually
> - NP-Hard means exact solutions are infeasible for large instances

---

# Slide 3: Methodology - Hybrid Approach

## DQN + Neuroevolution (Neurogenesis)

### The Key Insight
*Instead of manually designing a neural network, let evolution discover the optimal architecture for this specific problem.*

### Two-Level Optimization

```
┌─────────────────────────────────────────────────────┐
│         OUTER LOOP: Genetic Algorithm               │
│   Evolves network architecture (17 genes)           │
│   ┌─────────────────────────────────────────────┐   │
│   │     INNER LOOP: DQN Training                │   │
│   │   Learns packing policy for given arch.     │   │
│   └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### Multi-Objective Fitness
```
Fitness = 0.70 × Utilization + 0.20 × Bin_Efficiency + 0.10 × Complexity_Penalty
```

> **Speaker Notes:**
> - "Neurogenesis" = growing/developing neural architectures through evolution
> - Novel contribution: co-evolve architecture AND hyperparameters (LR, batch size, gamma)
> - Parsimony pressure prevents overly complex networks

---

# Slide 4: System Architecture

## Neural Network Components

```
INPUT                      PROCESSING                    OUTPUT
─────                      ──────────                    ──────

State (8-dim) ──────┐     ┌─────────────────┐
  - utilization     │     │   Combined      │
  - bins used       ├────▶│   Features      │
  - items remaining │     │                 │
                    │     │  ┌───────────┐  │
Action (25-dim) ────┤     │  │ Attention │  │────▶ Q(s,a)
  - bin index       │     │  │ Mechanism │  │       (action
  - rotation        │     │  └───────────┘  │        value)
  - position        │     │                 │
                    │     └─────────────────┘
Heightmap ──────────┘
  (7×7 patch)             CNN extracts local
   via CNN                spatial patterns
```

### Evolvable Genes (17 Parameters)
| Category | Parameters |
|----------|------------|
| **Architecture** | hidden_dim (8-512), enc_layers (1-8), attention_type |
| **Spatial** | patch_size (3-15), CNN channels |
| **Regularization** | dropout, activation function |
| **Hyperparameters** | learning_rate, batch_size, gamma |

> **Speaker Notes:**
> - Heightmap CNN learns to recognize corners, valleys, walls - where items fit well
> - Set Transformer provides permutation-invariant action reasoning
> - 17 genes create huge search space - GA efficiently explores it

---

# Slide 5: Experimental Scenarios

## Specialist vs. Generalist Training

### Scenario A: Problem-Specific (Specialist)
```
Single Problem ──▶ GA Evolution ──▶ Optimized Architecture ──▶ Best performance
                   (50 gens)        for THAT problem          on target
```

### Scenario B: Multi-Problem (Generalist)
```
9 Training    ──▶ GA Evolution ──▶ General Architecture ──▶ Test on 3
Problems          (round-robin)    for ALL problems         new problems
```

### Dataset Split
- **Training:** 3dBPP_2, 3, 5, 6, 7, 8, 9, 10, 11 (9 problems)
- **Test:** 3dBPP_1, 4, 12 (3 held-out problems)

### Research Question
*Can evolved architectures generalize across different problem instances, or does each problem need its own specialized network?*

> **Speaker Notes:**
> - Leave-one-out cross-validation measures generalization gap
> - Trade-off: specialists are better but don't transfer
> - Both scenarios run on HPC (Metacentrum) - 48h per experiment

---

# Slide 6: Results

## Evolved Architecture Performance

### Best Evolved Architecture
| Gene | Evolved Value | Interpretation |
|------|---------------|----------------|
| hidden_dim | 64 | Compact representation |
| enc_layers | 2 | Shallow network sufficient |
| attention_type | none | Not needed for this problem size |
| patch_size | 3 | Small local patterns important |
| dropout | 0.1 | Light regularization |
| learning_rate | 0.001 | Standard value |
| **Parameters** | **~45K** | Very lightweight model |

### Performance Metrics
| Metric | Value |
|--------|-------|
| Best Utilization | **49.9%** |
| Average Utilization | 41.9% |
| Bins Used | 1.0 (optimal) |
| Training Time | 78 seconds/evaluation |

### Key Observations
1. Evolution converges quickly (2-5 generations)
2. Simpler architectures often outperform complex ones
3. Patch size 3 preferred over larger sizes (local patterns matter)

**Visualizations available:** `output_data/ga_evolution/evolution_plot.png`, `output_data/best_bin_1_filled.png`

> **Speaker Notes:**
> - Point to the evolution plot showing fitness improvement
> - Show 3D packing visualization if time permits
> - Highlight that evolved architecture is surprisingly simple

---

# Slide 7: Conclusions & Future Work

## Summary

### Contributions
1. **Novel integration** of DQN + genetic algorithm + NAS for 3D bin packing
2. **Co-evolution** of network architecture with hyperparameters
3. **Systematic comparison** of specialist vs. generalist approaches
4. **Constraint handling** for realistic industrial scenarios

### Limitations (Addressed in Review)
- No comparison with traditional heuristics/baselines
- Computational cost of evolution process

### Future Directions
- Add baseline algorithms (first-fit, best-fit heuristics) for comparison
- Apply to real-world logistics datasets
- Explore transfer learning between problem domains
- GPU-accelerated parallel fitness evaluation

> **Speaker Notes:**
> - Acknowledge reviewer's point about missing baselines
> - Emphasize novelty in method combination
> - Mention potential for publication (reviewer noted this)

---

# Questions?

## Anticipated Questions from Review

**Q: What is nesting exactly?**
A: Nesting refers to optimizing the spatial arrangement of items within bins, considering geometry, rotations (6 orientations), and the relationships between items to maximize space utilization.

**Q: Why no baseline algorithm comparison?**
A: Future work will include first-fit decreasing (FFD), best-fit heuristics, and potentially genetic algorithm baselines to provide difficulty assessment.

**Q: How hard are the problem instances?**
A: Each problem has 50-80 items with dimensions 127-545 units, multiple constraints (weight, incompatibilities), and requires fitting into 1200³ containers. Optimal solutions are unknown for these NP-hard instances.

---

# Appendix: Technical Details

## Dataset Characteristics (3dBPP_1 to 3dBPP_12)
- **Container:** 1200 × 1200 × 1200 cubic units
- **Items per problem:** 50-80 rectangular boxes
- **Item dimensions:** 127-545 in each dimension
- **Constraints:** Weight limits, incompatibilities, affinities, center-of-mass

## GA Parameters
- Population size: 30
- Generations: 50
- Episodes per evaluation: 100
- Selection: Tournament (size=3)
- Mutation: Adaptive rate (0.3→0.1)
- Elitism: Top 3 preserved

## DQN Enhancements
- Double DQN with target network
- N-step returns (n=3)
- Experience replay (200K capacity)
- Episode-based epsilon decay

---

# Appendix: Figures to Include

1. **Architecture Diagram** - Neural network structure (create from slide 4)
2. **Evolution Plot** - `main/output_data/ga_evolution/evolution_plot.png`
3. **3D Packing Visualization** - `main/output_data/best_bin_1_filled.png`
4. **Problem Instance** - `main/nesting/inputData/.../Output/3dBPP_1.png`
5. **Specialist vs Generalist Comparison** - Create from experiment results

---
