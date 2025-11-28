"""
Genetic Algorithm for Neural Architecture Evolution

Based on "Using Genetic Algorithms to Optimize Artificial Neural Networks"
(Ding et al., 2010), Section 3.2: Optimizing Network Architecture.

Implements the complete GA loop:
1. Initialize population
2. Evaluate fitness (train each architecture)
3. Selection (keep best performers)
4. Crossover (create offspring)
5. Mutation (introduce variation)
6. Repeat until convergence

EXTENDED with modern improvements:
- Adaptive population sizing (2024) - dynamic exploration/exploitation phases
- Multi-objective fitness with parsimony pressure
- Adaptive mutation rates with diversity tracking
- Co-evolution of architecture and hyperparameters (Neuvo NAS+ 2025)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import json
import time
from pathlib import Path

from genome import NetworkGenome, create_initial_population, tournament_selection
from dqn_core.dqn_enhanced import DQNAgentEnhanced
from nesting.dataset_loader import load_problem, BinPackingProblem


def get_adaptive_population_size(generation: int,
                                 total_generations: int,
                                 base_population_size: int,
                                 exploration_ratio: float = 1.5,
                                 exploitation_ratio: float = 0.75) -> int:
    """
    Calculate dynamic population size based on evolution phase (2024 improvement).

    Strategy:
    - Early phase (first 1/3): Larger population for exploration (exploration_ratio × base)
    - Middle phase (middle 1/3): Normal population (1.0 × base)
    - Late phase (final 1/3): Smaller population for exploitation (exploitation_ratio × base)

    This adapts computational resources to the search phase:
    - More diversity early when exploring the space
    - Focus on refining best solutions later

    Args:
        generation: Current generation (0-indexed)
        total_generations: Total number of generations
        base_population_size: Base population size parameter
        exploration_ratio: Multiplier for early phase (default 1.5)
        exploitation_ratio: Multiplier for late phase (default 0.75)

    Returns:
        Adjusted population size (integer)
    """
    progress = generation / max(total_generations - 1, 1)  # 0.0 to 1.0

    if progress < 1.0 / 3.0:  # Early: exploration
        ratio = exploration_ratio
    elif progress > 2.0 / 3.0:  # Late: exploitation
        ratio = exploitation_ratio
    else:  # Middle: stable
        ratio = 1.0

    return int(base_population_size * ratio)


def evaluate_genome_fitness(genome: NetworkGenome,
                           env,
                           items: List,
                           episodes: int = 20,
                           verbose: bool = False) -> Tuple[float, Dict]:
    """
    Evaluate fitness of a genome by training its architecture.

    Fitness = Performance - Complexity Penalty

    Args:
        genome: NetworkGenome to evaluate
        env: MultiBinPackingEnv instance
        items: List of items to pack
        episodes: Number of training episodes
        verbose: Print detailed progress

    Returns:
        (fitness_score, metrics_dict)
    """
    import torch
    import gc
    from packing_with_dqncore2_enhanced import evaluate_agent_on_problem

    # Build network from genome (with reduced buffer for GA phase)
    cfg = genome.to_dqn_config(
        obs_dim=8,  # Fixed for this environment
        action_feat_dim=25,  # Fixed
        max_actions=env.max_actions,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Reduce buffer size during GA to save memory (override fixed params)
    cfg.buffer_size = 50_000  # Reduced from default 200k

    agent = DQNAgentEnhanced(cfg)

    try:
        # Train and evaluate
        metrics = evaluate_agent_on_problem(
            agent=agent,
            env=env,
            items=items,
            episodes=episodes,
            patch_size=genome.genes['patch_size'],
            train_freq=1,
            num_train_steps=1,
            verbose=verbose
        )
    finally:
        # Clean up memory
        del agent
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    # Calculate fitness (multi-objective)
    # Objectives:
    # 1. Maximize utilization (main objective)
    # 2. Minimize bins used (secondary)
    # 3. Minimize network complexity (parsimony)

    avg_util = metrics['avg_utilization']
    avg_bins = metrics.get('avg_bins_used', 1.0)
    complexity = genome.get_network_complexity()

    # Normalize bins (assume max 5 bins based on problem)
    max_bins = 5.0
    bins_penalty = min(avg_bins / max_bins, 1.0)  # Cap at 1.0

    # Normalize complexity (typical range: 0.5M - 5M params)
    # Use sigmoid-like normalization to handle outliers
    max_complexity = 5.0  # 5M params
    complexity_penalty = min(complexity / max_complexity, 1.0)  # Cap at 1.0

    # Multi-objective fitness with weighted components
    # Weights: 70% utilization, 20% bins efficiency, 10% complexity
    fitness = (
        0.70 * avg_util +                      # Maximize utilization (0-1)
        0.20 * (1.0 - bins_penalty) +          # Minimize bins used (0-1)
        0.10 * (1.0 - complexity_penalty)      # Parsimony (smaller = better)
    )

    # Store metrics in genome
    genome.fitness = fitness
    genome.metrics = metrics
    genome.metrics['fitness_components'] = {
        'utilization': avg_util,
        'bins_efficiency': 1.0 - bins_penalty,
        'parsimony': 1.0 - complexity_penalty,  # Fixed: now matches actual fitness calculation
        'complexity_params': complexity
    }

    return fitness, metrics


def evolve_architecture(problem,
                       population_size: int = 20,
                       generations: int = 10,
                       episodes_per_eval: int = 20,
                       elite_size: int = 2,
                       mutation_rate: float = 0.2,
                       adaptive_mutation: bool = True,
                       adaptive_population: bool = True,
                       exploration_ratio: float = 1.5,
                       exploitation_ratio: float = 0.75,
                       crossover_method: str = 'uniform',
                       tournament_size: int = 3,
                       seed: Optional[int] = None,
                       save_dir: Optional[str] = None,
                       verbose: bool = True) -> Tuple[NetworkGenome, List[NetworkGenome]]:
    """
    Main GA loop for architecture evolution.

    Args:
        problem: BinPackingProblem instance
        population_size: Base population size (will be dynamically adjusted if adaptive_population=True)
        generations: Number of generations to evolve
        episodes_per_eval: Episodes to train each architecture
        elite_size: Number of top genomes to preserve (elitism)
        mutation_rate: Initial probability of gene mutation
        adaptive_mutation: If True, decay mutation rate over generations
        adaptive_population: If True, dynamically adjust population size across generations (2024)
        exploration_ratio: Population multiplier for early exploration phase (default 1.5)
        exploitation_ratio: Population multiplier for late exploitation phase (default 0.75)
        crossover_method: 'uniform' or 'single_point'
        tournament_size: Size of tournament for selection
        seed: Random seed for reproducibility (None = random)
        save_dir: Directory to save results (None = don't save)
        verbose: Print progress

    Returns:
        (best_genome, final_population)
    """
    # Setup
    from packing_with_dqncore2_enhanced import load_problem_as_items, MultiBinPackingEnv
    import random

    # Set random seeds for reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        if verbose:
            print(f"Random seed set to: {seed}")

    items = load_problem_as_items(problem)
    W, D, H = problem.bin_dimensions
    
    env = MultiBinPackingEnv(
        W, D, H,
        items=items,
        max_actions=128,
        topk_eps=1000,
        seed=42,
        gamma=0.992,
        problem=problem
    )
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"GA-BASED NEURAL ARCHITECTURE EVOLUTION")
        print(f"{'='*80}")
        print(f"Problem: {W}×{D}×{H} container, {len(items)} items")
        print(f"Base population size: {population_size}")
        if adaptive_population:
            print(f"Adaptive population: ENABLED (exploration={exploration_ratio}×, exploitation={exploitation_ratio}×)")
        else:
            print(f"Adaptive population: DISABLED (fixed size)")
        print(f"Generations: {generations}")
        print(f"Episodes per evaluation: {episodes_per_eval}")
        print(f"Elite size: {elite_size}")
        print(f"Mutation rate: {mutation_rate} (adaptive: {adaptive_mutation})")
        print(f"{'='*80}\n")

    # Initialize population with starting size
    # For adaptive population, start with exploration size
    if adaptive_population:
        initial_size = get_adaptive_population_size(
            0, generations, population_size, exploration_ratio, exploitation_ratio
        )
    else:
        initial_size = population_size

    population = create_initial_population(initial_size)

    if verbose and adaptive_population:
        print(f"Initial population size: {initial_size} (exploration phase)")
        print()
    
    # Track best genome across all generations
    best_genome_ever = None
    best_fitness_ever = -np.inf
    
    # Track evolution history
    history = {
        'generation': [],
        'population_size': [],  # Track actual population size per generation
        'best_fitness': [],
        'avg_fitness': [],
        'std_fitness': [],
        'best_utilization': [],
        'avg_utilization': [],
        'best_bins': [],
        'avg_complexity': [],
        'mutation_rate': [],
        'diversity_score': []
    }
    
    # Main GA loop
    for gen in range(generations):
        gen_start_time = time.time()
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"GENERATION {gen + 1}/{generations}")
            print(f"{'='*80}")
        
        # Evaluate all genomes
        fitness_scores = []
        
        for i, genome in enumerate(population):
            if verbose:
                print(f"\n[{i+1}/{population_size}] Evaluating genome {genome.genome_id}...")
                print(f"  Architecture: hidden={genome.genes['hidden_dim']}, "
                      f"layers={genome.genes['enc_layers']}, "
                      f"head={genome.genes['head_hidden']}")
                print(f"  Attention: type={genome.genes['attention_type']}, "
                      f"heads={genome.genes['attention_heads']}, "
                      f"inds={genome.genes['num_inducing_points']}")
                print(f"  Features: patch={genome.genes['patch_size']}, "
                      f"cnn={genome.genes['cnn_channels']}, "
                      f"act={genome.genes['activation']}, "
                      f"drop={genome.genes['dropout']}")
            
            fitness, metrics = evaluate_genome_fitness(
                genome=genome,
                env=env,
                items=items.copy(),
                episodes=episodes_per_eval,
                verbose=False
            )
            
            fitness_scores.append(fitness)
            
            if verbose:
                print(f"  → Fitness: {fitness:.4f} | "
                      f"Util: {metrics['avg_utilization']:.3f} | "
                      f"Bins: {metrics['avg_bins_used']:.1f} | "
                      f"Complexity: {genome.get_network_complexity():.3f}M params")
        
        # Sort population by fitness
        population.sort(key=lambda g: g.fitness, reverse=True)
        
        # Track best
        gen_best = population[0]
        if gen_best.fitness > best_fitness_ever:
            best_fitness_ever = gen_best.fitness
            best_genome_ever = gen_best
            if verbose:
                print(f"\n✨ NEW BEST GENOME FOUND!")
                print(f"   Fitness: {best_fitness_ever:.4f}")
                print(f"   Utilization: {gen_best.metrics['avg_utilization']:.3f}")
                print(gen_best)
        
        # Statistics
        avg_fitness = np.mean(fitness_scores)
        std_fitness = np.std(fitness_scores)
        avg_util = np.mean([g.metrics['avg_utilization'] for g in population])
        avg_complexity = np.mean([g.get_network_complexity() for g in population])

        # Diversity score (based on gene variation)
        diversity_score = 0.0
        for gene_name in NetworkGenome.GENE_SPACES.keys():
            gene_values = [str(g.genes[gene_name]) for g in population]
            unique_values = len(set(gene_values))
            total_possible = len(NetworkGenome.GENE_SPACES[gene_name])
            diversity_score += unique_values / total_possible
        diversity_score /= len(NetworkGenome.GENE_SPACES)  # Normalize

        history['generation'].append(gen + 1)
        history['population_size'].append(len(population))
        history['best_fitness'].append(gen_best.fitness)
        history['avg_fitness'].append(avg_fitness)
        history['std_fitness'].append(std_fitness)
        history['best_utilization'].append(gen_best.metrics['avg_utilization'])
        history['avg_utilization'].append(avg_util)
        history['best_bins'].append(gen_best.metrics.get('avg_bins_used', 0))
        history['avg_complexity'].append(avg_complexity)
        history['mutation_rate'].append(mutation_rate)  # Will be updated below if adaptive
        history['diversity_score'].append(diversity_score)
        
        gen_time = time.time() - gen_start_time
        
        if verbose:
            print(f"\n{'─'*80}")
            print(f"Generation {gen + 1} Summary:")
            print(f"  Population size: {len(population)}")
            print(f"  Best fitness: {gen_best.fitness:.4f}")
            print(f"  Avg fitness: {avg_fitness:.4f} ± {std_fitness:.4f}")
            print(f"  Best utilization: {gen_best.metrics['avg_utilization']:.3f}")
            print(f"  Avg utilization: {avg_util:.3f}")
            print(f"  Avg complexity: {avg_complexity:.3f}M params")
            print(f"  Diversity: {diversity_score:.2%}")
            print(f"  Time: {gen_time:.1f}s")
            print(f"{'─'*80}")
        
        # Save generation results
        if save_dir:
            gen_file = save_dir / f"generation_{gen+1:03d}.json"
            gen_data = {
                'generation': gen + 1,
                'population': [g.to_dict() for g in population],
                'best_fitness': gen_best.fitness,
                'avg_fitness': avg_fitness,
            }
            with open(gen_file, 'w') as f:
                json.dump(gen_data, f, indent=2)
        
        # Create next generation
        if gen < generations - 1:  # Don't create new generation on last iteration
            # Calculate target population size for next generation
            if adaptive_population:
                target_size = get_adaptive_population_size(
                    gen + 1, generations, population_size,
                    exploration_ratio, exploitation_ratio
                )
            else:
                target_size = population_size

            next_population = []

            # Elitism: Keep top performers (but not more than target size)
            num_elites = min(elite_size, target_size)
            next_population.extend(population[:num_elites])

            # Adaptive mutation rate
            current_mutation_rate = mutation_rate
            if adaptive_mutation:
                # Decay mutation rate: high early (exploration), low late (exploitation)
                # Linear decay from mutation_rate to mutation_rate/4
                progress = (gen + 1) / generations
                current_mutation_rate = mutation_rate * (1.0 - 0.75 * progress)

                # Boost mutation if population diversity is low
                # Use relative threshold based on fitness range
                fitness_std = np.std([g.fitness for g in population])
                fitness_mean = np.mean([g.fitness for g in population])
                # Relative diversity: coefficient of variation
                relative_diversity = fitness_std / max(abs(fitness_mean), 0.01)
                if relative_diversity < 0.05:  # Low diversity threshold (5% CoV)
                    current_mutation_rate = min(mutation_rate * 1.5, 0.5)
                    if verbose:
                        print(f"  ⚠ Low diversity detected (CoV={relative_diversity:.4f}), "
                              f"boosting mutation to {current_mutation_rate:.3f}")

            # Fill rest of population with offspring
            while len(next_population) < target_size:
                # Tournament selection
                parent1 = tournament_selection(population, tournament_size)
                parent2 = tournament_selection(population, tournament_size)

                # Crossover
                child = NetworkGenome.crossover(parent1, parent2, method=crossover_method)

                # Mutation
                child = child.mutate(mutation_rate=current_mutation_rate)

                # Assign ID
                child.genome_id = len(next_population) + gen * max(target_size, population_size)

                next_population.append(child)

            if verbose:
                if adaptive_mutation:
                    print(f"  Mutation rate: {current_mutation_rate:.3f}")
                if adaptive_population and target_size != len(population):
                    print(f"  Population size: {len(population)} → {target_size}")

            # Update mutation rate in history
            history['mutation_rate'][-1] = current_mutation_rate

            population = next_population
    
    # Final summary
    if verbose:
        print(f"\n{'='*80}")
        print(f"EVOLUTION COMPLETE")
        print(f"{'='*80}")
        print(f"\nBest genome found:")
        print(best_genome_ever)
        print(f"\nArchitecture details:")
        for gene, value in best_genome_ever.genes.items():
            print(f"  {gene}: {value}")
        print(f"\nPerformance:")
        print(f"  Fitness: {best_genome_ever.fitness:.4f}")
        print(f"  Utilization: {best_genome_ever.metrics['avg_utilization']:.3f}")
        print(f"  Bins used: {best_genome_ever.metrics['avg_bins_used']:.1f}")
        print(f"  Complexity: {best_genome_ever.get_network_complexity():.3f}M params")
        print(f"{'='*80}\n")
    
    # Save final results
    if save_dir:
        # Save best genome
        best_file = save_dir / "best_genome.json"
        with open(best_file, 'w') as f:
            json.dump(best_genome_ever.to_dict(), f, indent=2)
        
        # Save history
        history_file = save_dir / "evolution_history.json"
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        if verbose:
            print(f"Results saved to: {save_dir}")
    
    return best_genome_ever, population


def load_best_genome(save_dir: str) -> NetworkGenome:
    """Load best genome from saved results."""
    best_file = Path(save_dir) / "best_genome.json"
    with open(best_file, 'r') as f:
        data = json.load(f)
    return NetworkGenome.from_dict(data)


def plot_evolution_history(save_dir: str, output_file: Optional[str] = None):
    """
    Plot evolution history (requires matplotlib).
    
    Args:
        save_dir: Directory with evolution_history.json
        output_file: Save plot to file (None = display)
    """
    import matplotlib.pyplot as plt
    
    history_file = Path(save_dir) / "evolution_history.json"
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Fitness over generations
    axes[0, 0].plot(history['generation'], history['best_fitness'], 'b-o', label='Best')
    axes[0, 0].plot(history['generation'], history['avg_fitness'], 'r--', label='Average')
    axes[0, 0].set_xlabel('Generation')
    axes[0, 0].set_ylabel('Fitness')
    axes[0, 0].set_title('Fitness Evolution')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Utilization over generations
    axes[0, 1].plot(history['generation'], history['best_utilization'], 'g-o')
    axes[0, 1].set_xlabel('Generation')
    axes[0, 1].set_ylabel('Best Utilization')
    axes[0, 1].set_title('Utilization Evolution')
    axes[0, 1].grid(True)
    
    # Complexity over generations
    axes[1, 0].plot(history['generation'], history['avg_complexity'], 'm-o')
    axes[1, 0].set_xlabel('Generation')
    axes[1, 0].set_ylabel('Avg Complexity (M params)')
    axes[1, 0].set_title('Network Complexity Evolution')
    axes[1, 0].grid(True)
    
    # Fitness vs Complexity (final generation)
    axes[1, 1].scatter(history['avg_complexity'][-1], history['best_fitness'][-1], s=100)
    axes[1, 1].set_xlabel('Complexity (M params)')
    axes[1, 1].set_ylabel('Fitness')
    axes[1, 1].set_title('Final Generation: Fitness vs Complexity')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()