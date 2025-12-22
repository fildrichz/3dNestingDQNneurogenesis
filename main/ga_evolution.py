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
                                 exploration_ratio: float = 1.25,
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
                           verbose: bool = False,
                           item_fraction: float = 1.0) -> Tuple[float, Dict]:
    """
    Evaluate fitness of a genome by training its architecture.

    Fitness = Performance - Complexity Penalty

    Args:
        genome: NetworkGenome to evaluate
        env: MultiBinPackingEnv instance
        items: List of items to pack
        episodes: Number of training episodes
        verbose: Print detailed progress
        item_fraction: Fraction of items to use (0.0-1.0) for curriculum learning

    Returns:
        (fitness_score, metrics_dict)
    """
    import torch
    import gc
    from packing_with_dqncore2_enhanced import evaluate_agent_on_problem

    # Apply curriculum learning: use subset of items if requested
    if item_fraction < 1.0:
        num_items = max(1, int(len(items) * item_fraction))
        items_subset = items[:num_items]
    else:
        items_subset = items

    # Build network from genome (with reduced buffer for GA phase)
    # Epsilon decay is calculated automatically based on total_episodes
    cfg = genome.to_dqn_config(
        obs_dim=8,  # Fixed for this environment
        action_feat_dim=25,  # Fixed
        max_actions=env.max_actions,
        device="cuda" if torch.cuda.is_available() else "cpu",
        total_episodes=episodes  # Automatic episode-based epsilon decay
    )

    # Reduce buffer size during GA to save memory (override fixed params)
    cfg.buffer_size = 15_000  # Reduced from default 200k (was 50k)

    agent = DQNAgentEnhanced(cfg)

    try:
        # Train and evaluate
        metrics = evaluate_agent_on_problem(
            agent=agent,
            env=env,
            items=items_subset,
            episodes=episodes,
            patch_size=genome.genes['patch_size'],
            train_freq=5,
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
        'complexity_params': complexity,
        'utilization_contribution': 0.70 * avg_util,
        'bins_contribution': 0.20 * (1.0 - bins_penalty),
        'parsimony_contribution': 0.10 * (1.0 - complexity_penalty)
    }
    genome.metrics['items_packed'] = metrics.get('avg_items_packed', 0)
    genome.metrics['training_reward'] = metrics.get('avg_reward', 0.0)
    genome.metrics['episode_length'] = metrics.get('avg_episode_length', 0)

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
                       verbose: bool = True,
                       curriculum_schedule: Optional[Dict] = None,
                       resume_from: Optional[str] = None) -> Tuple[NetworkGenome, List[NetworkGenome]]:
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
        curriculum_schedule: Optional dict with 'generations', 'item_fractions', 'episodes'
                            for progressive difficulty training
        resume_from: Optional path to checkpoint file to resume from
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
            print(f"Adaptive population: ENABLED (exploration={exploration_ratio}×, middle=(1x), exploitation={exploitation_ratio}×)")
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
        'avg_bins': [],
        'std_bins': [],
        'avg_complexity': [],
        'std_complexity': [],
        'mutation_rate': [],
        'diversity_score': [],
        # Items packed metrics
        'best_items_packed': [],
        'avg_items_packed': [],
        'packing_success_rate': [],
        # Fitness components
        'best_utilization_component': [],
        'best_bins_component': [],
        'best_parsimony_component': [],
        'avg_utilization_component': [],
        'avg_bins_component': [],
        'avg_parsimony_component': [],
        # Convergence metrics
        'fitness_improvement': [],
        'stagnation_counter': [],
        # Training performance
        'avg_training_reward': [],
        'best_training_reward': [],
        'avg_episode_length': [],
        # Gene-level diversity
        'gene_entropy': {}  # Will store per-gene entropy
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
                print(f"  -> Fitness: {fitness:.4f} | "
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
        std_complexity = np.std([g.get_network_complexity() for g in population])

        # Bins statistics
        avg_bins = np.mean([g.metrics.get('avg_bins_used', 0) for g in population])
        std_bins = np.std([g.metrics.get('avg_bins_used', 0) for g in population])

        # Items packed statistics
        avg_items = np.mean([g.metrics.get('items_packed', 0) for g in population])
        best_items = gen_best.metrics.get('items_packed', 0)
        total_items = len(items)
        packing_rate = avg_items / max(total_items, 1)

        # Fitness components statistics
        best_components = gen_best.metrics.get('fitness_components', {})
        avg_util_component = np.mean([g.metrics.get('fitness_components', {}).get('utilization_contribution', 0) for g in population])
        avg_bins_component = np.mean([g.metrics.get('fitness_components', {}).get('bins_contribution', 0) for g in population])
        avg_parsimony_component = np.mean([g.metrics.get('fitness_components', {}).get('parsimony_contribution', 0) for g in population])

        # Training performance
        avg_reward = np.mean([g.metrics.get('training_reward', 0) for g in population])
        best_reward = gen_best.metrics.get('training_reward', 0)
        avg_ep_length = np.mean([g.metrics.get('episode_length', 0) for g in population])

        # Convergence metrics
        if gen == 0:
            fitness_improvement = 0.0
            stagnation_count = 0
        else:
            fitness_improvement = gen_best.fitness - history['best_fitness'][-1]
            if fitness_improvement <= 1e-6:  # No significant improvement
                stagnation_count = history['stagnation_counter'][-1] + 1
            else:
                stagnation_count = 0

        # Diversity score (based on gene variation)
        diversity_score = 0.0
        gene_entropy_dict = {}
        for gene_name in NetworkGenome.GENE_SPACES.keys():
            gene_values = [str(g.genes[gene_name]) for g in population]
            unique_values = len(set(gene_values))
            total_possible = len(NetworkGenome.GENE_SPACES[gene_name])
            diversity_score += unique_values / total_possible

            # Calculate Shannon entropy for this gene
            value_counts = {}
            for val in gene_values:
                value_counts[val] = value_counts.get(val, 0) + 1
            entropy = 0.0
            for count in value_counts.values():
                p = count / len(gene_values)
                if p > 0:
                    entropy -= p * np.log2(p)
            gene_entropy_dict[gene_name] = entropy

        diversity_score /= len(NetworkGenome.GENE_SPACES)  # Normalize

        # Update history
        history['generation'].append(gen + 1)
        history['population_size'].append(len(population))
        history['best_fitness'].append(gen_best.fitness)
        history['avg_fitness'].append(avg_fitness)
        history['std_fitness'].append(std_fitness)
        history['best_utilization'].append(gen_best.metrics['avg_utilization'])
        history['avg_utilization'].append(avg_util)
        history['best_bins'].append(gen_best.metrics.get('avg_bins_used', 0))
        history['avg_bins'].append(avg_bins)
        history['std_bins'].append(std_bins)
        history['avg_complexity'].append(avg_complexity)
        history['std_complexity'].append(std_complexity)
        history['mutation_rate'].append(mutation_rate)  # Will be updated below if adaptive
        history['diversity_score'].append(diversity_score)
        # Items packed
        history['best_items_packed'].append(best_items)
        history['avg_items_packed'].append(avg_items)
        history['packing_success_rate'].append(packing_rate)
        # Fitness components
        history['best_utilization_component'].append(best_components.get('utilization_contribution', 0))
        history['best_bins_component'].append(best_components.get('bins_contribution', 0))
        history['best_parsimony_component'].append(best_components.get('parsimony_contribution', 0))
        history['avg_utilization_component'].append(avg_util_component)
        history['avg_bins_component'].append(avg_bins_component)
        history['avg_parsimony_component'].append(avg_parsimony_component)
        # Convergence
        history['fitness_improvement'].append(fitness_improvement)
        history['stagnation_counter'].append(stagnation_count)
        # Training performance
        history['avg_training_reward'].append(avg_reward)
        history['best_training_reward'].append(best_reward)
        history['avg_episode_length'].append(avg_ep_length)
        # Gene entropy (store as nested dict)
        for gene_name, entropy_val in gene_entropy_dict.items():
            if gene_name not in history['gene_entropy']:
                history['gene_entropy'][gene_name] = []
            history['gene_entropy'][gene_name].append(entropy_val)
        
        gen_time = time.time() - gen_start_time
        
        if verbose:
            print(f"\n{'-'*80}")
            print(f"Generation {gen + 1} Summary:")
            print(f"  Population size: {len(population)}")
            print(f"  Best fitness: {gen_best.fitness:.4f} (improvement: {fitness_improvement:+.4f})")
            print(f"  Avg fitness: {avg_fitness:.4f} ± {std_fitness:.4f}")
            print(f"  Best utilization: {gen_best.metrics['avg_utilization']:.3f}")
            print(f"  Avg utilization: {avg_util:.3f}")
            print(f"  Best items packed: {best_items}/{total_items} ({best_items/max(total_items,1):.1%})")
            print(f"  Avg items packed: {avg_items:.1f} (success rate: {packing_rate:.1%})")
            print(f"  Bins used: {gen_best.metrics.get('avg_bins_used', 0):.1f} (avg: {avg_bins:.1f}±{std_bins:.1f})")
            print(f"  Complexity: {gen_best.get_network_complexity():.3f}M params (avg: {avg_complexity:.3f}±{std_complexity:.3f}M)")
            print(f"  Diversity: {diversity_score:.2%}")
            if stagnation_count > 0:
                print(f"  Stagnation: {stagnation_count} generation(s)")
            print(f"  Time: {gen_time:.1f}s")
            print(f"{'-'*80}")
        
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

            # Also save as latest checkpoint for resume functionality
            checkpoint_file = save_dir / "checkpoint_latest.json"
            with open(checkpoint_file, 'w') as f:
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
                        print(f"  WARNING Low diversity detected (CoV={relative_diversity:.4f}), "
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
                    print(f"  Population size: {len(population)} -> {target_size}")

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


def export_metrics_to_csv(save_dir: str, output_file: Optional[str] = None):
    """
    Export evolution metrics to CSV for external analysis.

    Args:
        save_dir: Directory with evolution_history.json
        output_file: Output CSV file path (default: evolution_metrics.csv)
    """
    import csv

    history_file = Path(save_dir) / "evolution_history.json"
    with open(history_file, 'r') as f:
        history = json.load(f)

    if output_file is None:
        output_file = Path(save_dir) / "evolution_metrics.csv"

    # Flatten gene_entropy into columns
    gene_entropy_cols = {}
    if 'gene_entropy' in history and history['gene_entropy']:
        for gene_name, values in history['gene_entropy'].items():
            gene_entropy_cols[f'entropy_{gene_name}'] = values

    # Prepare CSV rows
    num_generations = len(history['generation'])

    with open(output_file, 'w', newline='') as csvfile:
        # Determine all column names
        base_cols = ['generation', 'population_size', 'best_fitness', 'avg_fitness', 'std_fitness',
                     'best_utilization', 'avg_utilization', 'best_bins', 'avg_bins', 'std_bins',
                     'best_items_packed', 'avg_items_packed', 'packing_success_rate',
                     'avg_complexity', 'std_complexity', 'mutation_rate', 'diversity_score',
                     'best_utilization_component', 'best_bins_component', 'best_parsimony_component',
                     'avg_utilization_component', 'avg_bins_component', 'avg_parsimony_component',
                     'fitness_improvement', 'stagnation_counter',
                     'avg_training_reward', 'best_training_reward', 'avg_episode_length']

        all_cols = base_cols + list(gene_entropy_cols.keys())

        writer = csv.DictWriter(csvfile, fieldnames=all_cols)
        writer.writeheader()

        for i in range(num_generations):
            row = {}
            for col in base_cols:
                if col in history:
                    row[col] = history[col][i] if i < len(history[col]) else ''
                else:
                    row[col] = ''

            for col, values in gene_entropy_cols.items():
                row[col] = values[i] if i < len(values) else ''

            writer.writerow(row)

    print(f"Metrics exported to: {output_file}")


def plot_evolution_dashboard(save_dir: str, output_file: Optional[str] = None):
    """
    Create comprehensive evolution overview dashboard.

    Args:
        save_dir: Directory with evolution_history.json
        output_file: Save plot to file (None = display)
    """
    import matplotlib.pyplot as plt

    history_file = Path(save_dir) / "evolution_history.json"
    with open(history_file, 'r') as f:
        history = json.load(f)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Evolution Overview Dashboard', fontsize=16, fontweight='bold')

    gens = history['generation']

    # 1. Fitness Evolution
    ax = axes[0, 0]
    ax.plot(gens, history['best_fitness'], 'b-o', label='Best', linewidth=2, markersize=4)
    ax.plot(gens, history['avg_fitness'], 'r--', label='Average', linewidth=1.5)
    if 'std_fitness' in history:
        avg = np.array(history['avg_fitness'])
        std = np.array(history['std_fitness'])
        ax.fill_between(gens, avg - std, avg + std, alpha=0.2, color='red')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    ax.set_title('Fitness Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Utilization Evolution
    ax = axes[0, 1]
    ax.plot(gens, history['best_utilization'], 'g-o', label='Best', linewidth=2, markersize=4)
    ax.plot(gens, history['avg_utilization'], 'orange', linestyle='--', label='Average', linewidth=1.5)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Utilization')
    ax.set_title('Utilization Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    # 3. Bins Usage
    ax = axes[0, 2]
    ax.plot(gens, history['best_bins'], 'purple', marker='o', label='Best', linewidth=2, markersize=4)
    if 'avg_bins' in history:
        ax.plot(gens, history['avg_bins'], 'purple', linestyle='--', label='Average', alpha=0.6, linewidth=1.5)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Bins Used')
    ax.set_title('Bins Usage Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Items Packed
    ax = axes[1, 0]
    if 'best_items_packed' in history and 'avg_items_packed' in history:
        ax.plot(gens, history['best_items_packed'], 'cyan', marker='o', label='Best', linewidth=2, markersize=4)
        ax.plot(gens, history['avg_items_packed'], 'cyan', linestyle='--', label='Average', alpha=0.6, linewidth=1.5)
        ax.set_ylabel('Items Packed')
        ax.legend()
    ax.set_xlabel('Generation')
    ax.set_title('Items Packed Evolution')
    ax.grid(True, alpha=0.3)

    # 5. Complexity Evolution
    ax = axes[1, 1]
    ax.plot(gens, history['avg_complexity'], 'm-o', linewidth=2, markersize=4)
    if 'std_complexity' in history:
        avg_comp = np.array(history['avg_complexity'])
        std_comp = np.array(history['std_complexity'])
        ax.fill_between(gens, avg_comp - std_comp, avg_comp + std_comp, alpha=0.2, color='magenta')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Complexity (M params)')
    ax.set_title('Network Complexity Evolution')
    ax.grid(True, alpha=0.3)

    # 6. Packing Success Rate
    ax = axes[1, 2]
    if 'packing_success_rate' in history:
        ax.plot(gens, history['packing_success_rate'], 'brown', marker='o', linewidth=2, markersize=4)
        ax.set_ylim([0, 1])
    ax.set_xlabel('Generation')
    ax.set_ylabel('Packing Success Rate')
    ax.set_title('Average Packing Success Rate')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Dashboard saved to: {output_file}")
    else:
        plt.show()

    plt.close()


def plot_fitness_components(save_dir: str, output_file: Optional[str] = None):
    """
    Plot fitness components breakdown over generations.

    Args:
        save_dir: Directory with evolution_history.json
        output_file: Save plot to file (None = display)
    """
    import matplotlib.pyplot as plt

    history_file = Path(save_dir) / "evolution_history.json"
    with open(history_file, 'r') as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Fitness Components Analysis', fontsize=16, fontweight='bold')

    gens = history['generation']

    # 1. Stacked Area Chart - Best Genome Components
    ax = axes[0]
    if all(k in history for k in ['best_utilization_component', 'best_bins_component', 'best_parsimony_component']):
        util_comp = history['best_utilization_component']
        bins_comp = history['best_bins_component']
        pars_comp = history['best_parsimony_component']

        ax.stackplot(gens, util_comp, bins_comp, pars_comp,
                     labels=['Utilization (70%)', 'Bins Efficiency (20%)', 'Parsimony (10%)'],
                     alpha=0.7, colors=['green', 'blue', 'purple'])
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness Contribution')
        ax.set_title('Best Genome - Fitness Components (Stacked)')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

    # 2. Line Plot - Average Population Components
    ax = axes[1]
    if all(k in history for k in ['avg_utilization_component', 'avg_bins_component', 'avg_parsimony_component']):
        ax.plot(gens, history['avg_utilization_component'], 'g-o', label='Utilization (70%)', linewidth=2, markersize=3)
        ax.plot(gens, history['avg_bins_component'], 'b-s', label='Bins Efficiency (20%)', linewidth=2, markersize=3)
        ax.plot(gens, history['avg_parsimony_component'], 'purple', marker='^', label='Parsimony (10%)', linewidth=2, markersize=3)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Average Fitness Contribution')
        ax.set_title('Population Average - Fitness Components')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Fitness components plot saved to: {output_file}")
    else:
        plt.show()

    plt.close()


def plot_convergence_analysis(save_dir: str, output_file: Optional[str] = None):
    """
    Plot convergence metrics and improvement over time.

    Args:
        save_dir: Directory with evolution_history.json
        output_file: Save plot to file (None = display)
    """
    import matplotlib.pyplot as plt

    history_file = Path(save_dir) / "evolution_history.json"
    with open(history_file, 'r') as f:
        history = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Convergence Analysis', fontsize=16, fontweight='bold')

    gens = history['generation']

    # 1. Fitness Improvement per Generation
    ax = axes[0, 0]
    if 'fitness_improvement' in history:
        improvements = history['fitness_improvement']
        colors = ['green' if x > 0 else 'red' for x in improvements]
        ax.bar(gens, improvements, color=colors, alpha=0.6)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness Improvement (Δ)')
        ax.set_title('Fitness Improvement per Generation')
        ax.grid(True, alpha=0.3, axis='y')

    # 2. Cumulative Fitness Improvement
    ax = axes[0, 1]
    best_fitness = history['best_fitness']
    if len(best_fitness) > 0:
        cumulative_improvement = [best_fitness[i] - best_fitness[0] for i in range(len(best_fitness))]
        ax.plot(gens, cumulative_improvement, 'b-o', linewidth=2, markersize=4)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Cumulative Improvement')
        ax.set_title('Cumulative Fitness Improvement')
        ax.grid(True, alpha=0.3)

    # 3. Stagnation Periods
    ax = axes[1, 0]
    if 'stagnation_counter' in history:
        stagnation = history['stagnation_counter']
        ax.plot(gens, stagnation, 'r-o', linewidth=2, markersize=4)
        ax.fill_between(gens, stagnation, alpha=0.3, color='red')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Generations Without Improvement')
        ax.set_title('Stagnation Counter')
        ax.grid(True, alpha=0.3)

    # 4. Best Fitness with Plateau Highlighting
    ax = axes[1, 1]
    ax.plot(gens, best_fitness, 'b-o', linewidth=2, markersize=4, label='Best Fitness')
    # Highlight plateaus (stagnation > 1)
    if 'stagnation_counter' in history:
        for i, stag in enumerate(history['stagnation_counter']):
            if stag > 1:
                ax.axvspan(gens[i] - 0.5, gens[i] + 0.5, alpha=0.2, color='red')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Best Fitness')
    ax.set_title('Best Fitness (Red = Stagnation)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Convergence analysis saved to: {output_file}")
    else:
        plt.show()

    plt.close()


def plot_population_dynamics(save_dir: str, output_file: Optional[str] = None):
    """
    Plot population dynamics including size, mutation, and diversity.

    Args:
        save_dir: Directory with evolution_history.json
        output_file: Save plot to file (None = display)
    """
    import matplotlib.pyplot as plt

    history_file = Path(save_dir) / "evolution_history.json"
    with open(history_file, 'r') as f:
        history = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Population Dynamics', fontsize=16, fontweight='bold')

    gens = history['generation']

    # 1. Population Size Over Time
    ax = axes[0, 0]
    if 'population_size' in history:
        ax.plot(gens, history['population_size'], 'b-o', linewidth=2, markersize=4)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Population Size')
        ax.set_title('Population Size Evolution')
        ax.grid(True, alpha=0.3)

    # 2. Mutation Rate Evolution
    ax = axes[0, 1]
    if 'mutation_rate' in history:
        ax.plot(gens, history['mutation_rate'], 'r-o', linewidth=2, markersize=4)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Mutation Rate')
        ax.set_title('Adaptive Mutation Rate')
        ax.grid(True, alpha=0.3)

    # 3. Diversity Score
    ax = axes[1, 0]
    if 'diversity_score' in history:
        ax.plot(gens, history['diversity_score'], 'g-o', linewidth=2, markersize=4)
        ax.axhline(y=0.3, color='orange', linestyle='--', label='Low Diversity Threshold', linewidth=1)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Diversity Score')
        ax.set_title('Population Diversity')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

    # 4. Gene Entropy Heatmap
    ax = axes[1, 1]
    if 'gene_entropy' in history and history['gene_entropy']:
        gene_names = list(history['gene_entropy'].keys())
        entropy_matrix = []
        for gene_name in gene_names:
            entropy_matrix.append(history['gene_entropy'][gene_name])

        if entropy_matrix:
            im = ax.imshow(entropy_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
            ax.set_yticks(range(len(gene_names)))
            ax.set_yticklabels([name.replace('_', ' ').title() for name in gene_names], fontsize=8)
            ax.set_xlabel('Generation')
            ax.set_title('Gene Entropy Heatmap')
            plt.colorbar(im, ax=ax, label='Entropy')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Population dynamics plot saved to: {output_file}")
    else:
        plt.show()

    plt.close()


def plot_architecture_evolution(save_dir: str, output_file: Optional[str] = None):
    """
    Plot architecture characteristics evolution.

    Args:
        save_dir: Directory with evolution_history.json
        output_file: Save plot to file (None = display)
    """
    import matplotlib.pyplot as plt

    history_file = Path(save_dir) / "evolution_history.json"
    with open(history_file, 'r') as f:
        history = json.load(f)

    # Load all generations to analyze gene distributions
    generations_dir = Path(save_dir)
    gen_files = sorted(generations_dir.glob("generation_*.json"))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Architecture Evolution', fontsize=16, fontweight='bold')

    gens = history['generation']

    # 1. Network Complexity
    ax = axes[0, 0]
    ax.plot(gens, history['avg_complexity'], 'm-o', linewidth=2, markersize=4, label='Average')
    if 'std_complexity' in history:
        avg_comp = np.array(history['avg_complexity'])
        std_comp = np.array(history['std_complexity'])
        ax.fill_between(gens, avg_comp - std_comp, avg_comp + std_comp, alpha=0.2, color='magenta')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Complexity (M params)')
    ax.set_title('Network Complexity Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Hidden Dim Distribution (from generation files)
    ax = axes[0, 1]
    if gen_files:
        hidden_dims_over_time = []
        gen_nums = []
        for gen_file in gen_files[::max(1, len(gen_files)//10)]:  # Sample every ~10%
            with open(gen_file, 'r') as f:
                gen_data = json.load(f)
            hidden_dims = [g['genes']['hidden_dim'] for g in gen_data['population']]
            hidden_dims_over_time.append(hidden_dims)
            gen_nums.append(gen_data['generation'])

        if hidden_dims_over_time:
            ax.boxplot(hidden_dims_over_time, positions=gen_nums, widths=max(1, len(gens) * 0.05))
            ax.set_xlabel('Generation')
            ax.set_ylabel('Hidden Dimension')
            ax.set_title('Hidden Dimension Distribution')
            ax.grid(True, alpha=0.3, axis='y')

    # 3. Layer Count Distribution
    ax = axes[1, 0]
    if gen_files:
        layers_over_time = []
        gen_nums = []
        for gen_file in gen_files[::max(1, len(gen_files)//10)]:
            with open(gen_file, 'r') as f:
                gen_data = json.load(f)
            layers = [g['genes']['enc_layers'] for g in gen_data['population']]
            layers_over_time.append(layers)
            gen_nums.append(gen_data['generation'])

        if layers_over_time:
            ax.boxplot(layers_over_time, positions=gen_nums, widths=max(1, len(gens) * 0.05))
            ax.set_xlabel('Generation')
            ax.set_ylabel('Number of Layers')
            ax.set_title('Encoder Layers Distribution')
            ax.grid(True, alpha=0.3, axis='y')

    # 4. Attention Type Usage (from last generation)
    ax = axes[1, 1]
    if gen_files:
        with open(gen_files[-1], 'r') as f:
            final_gen = json.load(f)

        attention_types = [g['genes']['attention_type'] for g in final_gen['population']]
        type_counts = {}
        for att_type in attention_types:
            type_counts[att_type] = type_counts.get(att_type, 0) + 1

        if type_counts:
            ax.bar(type_counts.keys(), type_counts.values(), color='skyblue', alpha=0.7)
            ax.set_xlabel('Attention Type')
            ax.set_ylabel('Count')
            ax.set_title(f'Attention Types (Final Gen)')
            ax.grid(True, alpha=0.3, axis='y')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Architecture evolution plot saved to: {output_file}")
    else:
        plt.show()

    plt.close()


def plot_pareto_analysis(save_dir: str, output_file: Optional[str] = None):
    """
    Plot Pareto frontier analysis (fitness/utilization vs complexity).

    Args:
        save_dir: Directory with evolution_history.json
        output_file: Save plot to file (None = display)
    """
    import matplotlib.pyplot as plt

    history_file = Path(save_dir) / "evolution_history.json"
    with open(history_file, 'r') as f:
        history = json.load(f)

    # Load final generation for scatter plot
    generations_dir = Path(save_dir)
    gen_files = sorted(generations_dir.glob("generation_*.json"))

    if not gen_files:
        print("No generation files found for Pareto analysis")
        return

    with open(gen_files[-1], 'r') as f:
        final_gen = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Pareto Analysis (Final Generation)', fontsize=16, fontweight='bold')

    # Extract genome data
    complexities = []
    fitnesses = []
    utilizations = []

    for genome in final_gen['population']:
        # Calculate complexity
        from genome import NetworkGenome
        temp_genome = NetworkGenome.from_dict(genome)
        complexities.append(temp_genome.get_network_complexity())
        fitnesses.append(genome.get('fitness', 0))
        utilizations.append(genome.get('metrics', {}).get('avg_utilization', 0))

    # 1. Fitness vs Complexity
    ax = axes[0]
    scatter = ax.scatter(complexities, fitnesses, c=utilizations, cmap='RdYlGn',
                        s=100, alpha=0.6, edgecolors='black', linewidth=0.5)

    # Highlight Pareto front
    pareto_indices = []
    for i in range(len(complexities)):
        is_pareto = True
        for j in range(len(complexities)):
            if i != j:
                # Dominated if: lower fitness AND higher/equal complexity
                if fitnesses[j] > fitnesses[i] and complexities[j] <= complexities[i]:
                    is_pareto = False
                    break
        if is_pareto:
            pareto_indices.append(i)

    if pareto_indices:
        pareto_comp = [complexities[i] for i in pareto_indices]
        pareto_fit = [fitnesses[i] for i in pareto_indices]
        # Sort for line plot
        sorted_pairs = sorted(zip(pareto_comp, pareto_fit))
        ax.plot([p[0] for p in sorted_pairs], [p[1] for p in sorted_pairs],
               'r--', linewidth=2, label='Pareto Front')

    ax.set_xlabel('Complexity (M params)')
    ax.set_ylabel('Fitness')
    ax.set_title('Fitness vs Complexity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Utilization')

    # 2. Utilization vs Complexity
    ax = axes[1]
    scatter = ax.scatter(complexities, utilizations, c=fitnesses, cmap='viridis',
                        s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Complexity (M params)')
    ax.set_ylabel('Utilization')
    ax.set_title('Utilization vs Complexity')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Fitness')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Pareto analysis saved to: {output_file}")
    else:
        plt.show()

    plt.close()


def generate_summary_report(save_dir: str, output_file: Optional[str] = None):
    """
    Generate a comprehensive markdown summary report.

    Args:
        save_dir: Directory with evolution_history.json
        output_file: Output markdown file (default: summary_report.md)
    """
    history_file = Path(save_dir) / "evolution_history.json"
    with open(history_file, 'r') as f:
        history = json.load(f)

    if output_file is None:
        output_file = Path(save_dir) / "summary_report.md"

    # Load best genome
    best_file = Path(save_dir) / "best_genome.json"
    if best_file.exists():
        with open(best_file, 'r') as f:
            best_genome_data = json.load(f)
    else:
        best_genome_data = None

    with open(output_file, 'w') as f:
        f.write("# Neurogenesis Evolution Summary Report\n\n")
        f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Evolution Configuration\n\n")
        f.write(f"- **Generations:** {len(history['generation'])}\n")
        if 'population_size' in history:
            f.write(f"- **Population Size:** {history['population_size'][0]} -> {history['population_size'][-1]}\n")
        f.write("\n")

        f.write("## Performance Summary\n\n")
        f.write(f"- **Best Fitness:** {max(history['best_fitness']):.4f}\n")
        f.write(f"- **Best Utilization:** {max(history['best_utilization']):.4f}\n")
        if 'best_bins' in history:
            f.write(f"- **Best Bins Used:** {min([b for b in history['best_bins'] if b > 0]):.1f}\n")
        if 'best_items_packed' in history:
            f.write(f"- **Best Items Packed:** {max(history['best_items_packed'])}\n")
        f.write("\n")

        f.write("## Convergence Analysis\n\n")
        total_improvement = history['best_fitness'][-1] - history['best_fitness'][0]
        f.write(f"- **Total Fitness Improvement:** {total_improvement:+.4f}\n")

        if 'stagnation_counter' in history:
            max_stagnation = max(history['stagnation_counter'])
            f.write(f"- **Max Stagnation Period:** {max_stagnation} generations\n")

        if 'fitness_improvement' in history:
            positive_improvements = sum(1 for x in history['fitness_improvement'] if x > 0)
            f.write(f"- **Generations with Improvement:** {positive_improvements}/{len(history['fitness_improvement'])}\n")
        f.write("\n")

        f.write("## Best Genome Details\n\n")
        if best_genome_data:
            f.write(f"**Fitness:** {best_genome_data.get('fitness', 'N/A'):.4f}\n\n")
            f.write("### Architecture\n\n")
            genes = best_genome_data.get('genes', {})
            f.write(f"- Hidden Dimension: {genes.get('hidden_dim', 'N/A')}\n")
            f.write(f"- Encoder Layers: {genes.get('enc_layers', 'N/A')}\n")
            f.write(f"- Attention Type: {genes.get('attention_type', 'N/A')}\n")
            f.write(f"- Attention Heads: {genes.get('attention_heads', 'N/A')}\n")
            f.write(f"- Patch Size: {genes.get('patch_size', 'N/A')}\n")
            f.write(f"- Activation: {genes.get('activation', 'N/A')}\n")
            f.write(f"- Dropout: {genes.get('dropout', 'N/A')}\n")
            f.write("\n")

            if 'metrics' in best_genome_data:
                metrics = best_genome_data['metrics']
                f.write("### Performance Metrics\n\n")
                f.write(f"- Utilization: {metrics.get('avg_utilization', 'N/A'):.3f}\n")
                f.write(f"- Bins Used: {metrics.get('avg_bins_used', 'N/A'):.1f}\n")
                f.write(f"- Items Packed: {metrics.get('items_packed', 'N/A')}\n")
                f.write("\n")

        f.write("## Population Statistics\n\n")
        f.write(f"- **Average Complexity:** {np.mean(history['avg_complexity']):.3f}M params\n")
        f.write(f"- **Average Diversity:** {np.mean(history['diversity_score']):.2%}\n")
        if 'mutation_rate' in history:
            f.write(f"- **Mutation Rate Range:** {min(history['mutation_rate']):.3f} - {max(history['mutation_rate']):.3f}\n")
        f.write("\n")

        f.write("## Recommendations\n\n")

        # Check for low diversity
        if np.mean(history['diversity_score'][-3:]) < 0.3:
            f.write("- ⚠️ **Low diversity detected** in final generations. Consider increasing mutation rate or population size.\n")

        # Check for early convergence
        if 'stagnation_counter' in history and max(history['stagnation_counter']) > 5:
            f.write("- ⚠️ **Extended stagnation detected**. Evolution may have converged early.\n")

        # Check complexity trends
        if len(history['avg_complexity']) > 3:
            if history['avg_complexity'][-1] > history['avg_complexity'][0] * 1.5:
                f.write("- 📈 Network complexity increased significantly. Consider stronger parsimony pressure.\n")

        f.write("\n---\n")
        f.write("*Report generated by ga_evolution.py*\n")

    print(f"Summary report saved to: {output_file}")


def generate_all_visualizations(save_dir: str, create_plots_subdir: bool = True):
    """
    Generate all visualization plots and reports in one call.

    Args:
        save_dir: Directory with evolution_history.json
        create_plots_subdir: If True, create a 'plots' subdirectory for outputs
    """
    save_path = Path(save_dir)

    if create_plots_subdir:
        plots_dir = save_path / "plots"
        plots_dir.mkdir(exist_ok=True)
    else:
        plots_dir = save_path

    print(f"\n{'='*80}")
    print("Generating Evolution Visualizations")
    print(f"{'='*80}\n")

    # Generate all plots
    plot_evolution_dashboard(save_dir, output_file=str(plots_dir / "overview_dashboard.png"))
    plot_fitness_components(save_dir, output_file=str(plots_dir / "fitness_components.png"))
    plot_convergence_analysis(save_dir, output_file=str(plots_dir / "convergence_analysis.png"))
    plot_population_dynamics(save_dir, output_file=str(plots_dir / "population_dynamics.png"))
    plot_architecture_evolution(save_dir, output_file=str(plots_dir / "architecture_evolution.png"))
    plot_pareto_analysis(save_dir, output_file=str(plots_dir / "pareto_analysis.png"))

    # Generate exports
    export_metrics_to_csv(save_dir, output_file=str(save_path / "evolution_metrics.csv"))
    generate_summary_report(save_dir, output_file=str(save_path / "summary_report.md"))

    print(f"\n{'='*80}")
    print("All visualizations generated successfully!")
    print(f"Location: {plots_dir}")
    print(f"{'='*80}\n")