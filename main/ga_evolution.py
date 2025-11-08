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
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import json
import time
from pathlib import Path

from genome import NetworkGenome, create_initial_population, tournament_selection
from dqn_core.dqn_enhanced import DQNAgentEnhanced
from nesting.dataset_loader import load_problem, BinPackingProblem


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
    from packing_with_dqncore2_enhanced import evaluate_agent_on_problem
    
    # Build network from genome
    cfg = genome.to_dqn_config(
        obs_dim=8,  # Fixed for this environment
        action_feat_dim=25,  # Fixed
        max_actions=env.max_actions,
        device="cuda" if __import__("torch").cuda.is_available() else "cpu"
    )
    
    agent = DQNAgentEnhanced(cfg)
    
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
    
    # Calculate fitness
    # Main objective: maximize utilization
    # Secondary objective: minimize network complexity (parsimony)
    avg_util = metrics['avg_utilization']
    complexity = genome.get_network_complexity()
    
    # Fitness with parsimony pressure
    # Weight complexity penalty to prefer smaller networks when performance is equal
    fitness = avg_util - (0.01 * complexity)
    
    # Store metrics in genome
    genome.fitness = fitness
    genome.metrics = metrics
    
    return fitness, metrics


def evolve_architecture(problem,
                       population_size: int = 20,
                       generations: int = 10,
                       episodes_per_eval: int = 20,
                       elite_size: int = 2,
                       mutation_rate: float = 0.2,
                       crossover_method: str = 'uniform',
                       tournament_size: int = 3,
                       save_dir: Optional[str] = None,
                       verbose: bool = True) -> Tuple[NetworkGenome, List[NetworkGenome]]:
    """
    Main GA loop for architecture evolution.
    
    Args:
        problem: BinPackingProblem instance
        population_size: Number of genomes per generation
        generations: Number of generations to evolve
        episodes_per_eval: Episodes to train each architecture
        elite_size: Number of top genomes to preserve (elitism)
        mutation_rate: Probability of gene mutation
        crossover_method: 'uniform' or 'single_point'
        tournament_size: Size of tournament for selection
        save_dir: Directory to save results (None = don't save)
        verbose: Print progress
        
    Returns:
        (best_genome, final_population)
    """
    # Setup
    from packing_with_dqncore2_enhanced import load_problem_as_items, MultiBinPackingEnv
    
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
        print(f"Population size: {population_size}")
        print(f"Generations: {generations}")
        print(f"Episodes per evaluation: {episodes_per_eval}")
        print(f"Elite size: {elite_size}")
        print(f"Mutation rate: {mutation_rate}")
        print(f"{'='*80}\n")
    
    # Initialize population
    population = create_initial_population(population_size)
    
    # Track best genome across all generations
    best_genome_ever = None
    best_fitness_ever = -np.inf
    
    # Track evolution history
    history = {
        'generation': [],
        'best_fitness': [],
        'avg_fitness': [],
        'best_utilization': [],
        'avg_complexity': []
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
                      f"attention={genome.genes['use_attention']}, "
                      f"patch={genome.genes['patch_size']}")
            
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
        avg_util = np.mean([g.metrics['avg_utilization'] for g in population])
        avg_complexity = np.mean([g.get_network_complexity() for g in population])
        
        history['generation'].append(gen + 1)
        history['best_fitness'].append(gen_best.fitness)
        history['avg_fitness'].append(avg_fitness)
        history['best_utilization'].append(gen_best.metrics['avg_utilization'])
        history['avg_complexity'].append(avg_complexity)
        
        gen_time = time.time() - gen_start_time
        
        if verbose:
            print(f"\n{'─'*80}")
            print(f"Generation {gen + 1} Summary:")
            print(f"  Best fitness: {gen_best.fitness:.4f}")
            print(f"  Avg fitness: {avg_fitness:.4f}")
            print(f"  Best utilization: {gen_best.metrics['avg_utilization']:.3f}")
            print(f"  Avg complexity: {avg_complexity:.3f}M params")
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
            next_population = []
            
            # Elitism: Keep top performers
            next_population.extend(population[:elite_size])
            
            # Fill rest of population with offspring
            while len(next_population) < population_size:
                # Tournament selection
                parent1 = tournament_selection(population, tournament_size)
                parent2 = tournament_selection(population, tournament_size)
                
                # Crossover
                child = NetworkGenome.crossover(parent1, parent2, method=crossover_method)
                
                # Mutation
                child = child.mutate(mutation_rate=mutation_rate)
                
                # Assign ID
                child.genome_id = len(next_population) + gen * population_size
                
                next_population.append(child)
            
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
