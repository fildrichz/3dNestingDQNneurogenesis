"""
Experiment 2: Leave-One-Out Generalization Testing

This experiment tests architecture generalization by:
1. Selecting one problem as held-out test set
2. Evolving architecture on all OTHER problems (N-1 training set)
3. Training and evaluating on the held-out problem
4. Comparing with problem-specific architecture

This tests whether neuroevolution can find general architectures that
work across different problem instances, or if problem-specific tuning
is necessary.

Supports:
- Curriculum learning (progressive difficulty during evolution)
- Multi-problem fitness aggregation
- Checkpoint/resume capability
- Comprehensive result logging
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import numpy as np
import torch

from nesting.dataset_loader import load_problem, BinPackingProblem
from ga_evolution import evaluate_genome_fitness
from example_ga_training import train_with_evolved_genome
from genome import NetworkGenome, create_initial_population, tournament_selection
from packing_with_dqncore2_enhanced import load_problem_as_items, MultiBinPackingEnv


def get_problem_files(dataset_dir: str, exclude_pattern: Optional[str] = None) -> List[Path]:
    """
    Get all problem files from dataset directory, optionally excluding one.

    Args:
        dataset_dir: Path to directory containing problem .txt files
        exclude_pattern: Pattern to exclude (e.g., "3dBPP_12")

    Returns:
        List of Path objects for problem files
    """
    dataset_path = Path(dataset_dir)
    problem_files = sorted(dataset_path.glob("3dBPP_*.txt"))

    # Exclude solution files
    problem_files = [f for f in problem_files if not f.name.endswith("_sol.txt")]

    # Exclude target if specified
    if exclude_pattern:
        problem_files = [f for f in problem_files if exclude_pattern not in f.stem]

    return problem_files


def evaluate_genome_multi_problem(
    genome: NetworkGenome,
    problems: List[Tuple[BinPackingProblem, Path]],
    episodes_per_problem: int = 10,
    item_fraction: float = 1.0,
    verbose: bool = False
) -> Tuple[float, Dict]:
    """
    Evaluate genome fitness across multiple problems.

    Args:
        genome: NetworkGenome to evaluate
        problems: List of (problem, problem_file_path) tuples
        episodes_per_problem: Episodes to train on each problem
        item_fraction: Fraction of items to use for curriculum
        verbose: Print progress

    Returns:
        (average_fitness, metrics_dict)
    """
    fitness_scores = []
    all_metrics = []

    for problem, problem_path in problems:
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

        fitness, metrics = evaluate_genome_fitness(
            genome=genome,
            env=env,
            items=items,
            episodes=episodes_per_problem,
            verbose=False,
            item_fraction=item_fraction
        )

        fitness_scores.append(fitness)
        all_metrics.append({
            'problem': problem_path.stem,
            'fitness': fitness,
            'utilization': metrics['avg_utilization'],
            'bins': metrics.get('avg_bins_used', 0)
        })

        if verbose:
            print(f"  {problem_path.stem}: fitness={fitness:.4f}, "
                  f"util={metrics['avg_utilization']:.3f}")

    # Aggregate fitness (mean across problems)
    avg_fitness = np.mean(fitness_scores)

    aggregate_metrics = {
        'avg_fitness': avg_fitness,
        'std_fitness': np.std(fitness_scores),
        'min_fitness': np.min(fitness_scores),
        'max_fitness': np.max(fitness_scores),
        'avg_utilization': np.mean([m['utilization'] for m in all_metrics]),
        'avg_bins_used': np.mean([m['bins'] for m in all_metrics]),
        'per_problem': all_metrics
    }

    return avg_fitness, aggregate_metrics


def evolve_multi_problem_architecture(
    problems: List[Tuple[BinPackingProblem, Path]],
    population_size: int = 20,
    generations: int = 15,
    episodes_per_problem: int = 10,
    adaptive_population: bool = True,
    elite_size: int = 2,
    mutation_rate: float = 0.2,
    seed: Optional[int] = None,
    save_dir: Optional[str] = None,
    verbose: bool = True,
    curriculum_schedule: Optional[Dict] = None,
    resume_from: Optional[str] = None
) -> Tuple[NetworkGenome, List[NetworkGenome]]:
    """
    Evolve architecture across multiple problems.

    Args:
        problems: List of (problem, problem_file_path) tuples
        population_size: Base population size
        generations: Number of generations to evolve
        episodes_per_problem: Episodes per problem for fitness evaluation
        adaptive_population: Enable adaptive population sizing (1.5x early, 0.75x late)
        elite_size: Number of top genomes to preserve
        mutation_rate: Initial probability of gene mutation
        seed: Random seed for reproducibility
        save_dir: Directory to save results
        verbose: Print progress
        curriculum_schedule: Curriculum learning schedule
        resume_from: Checkpoint file to resume from

    Returns:
        (best_genome, final_population)
    """
    import random
    from pathlib import Path

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Check for checkpoint resume
    start_generation = 0
    population = None
    history = None
    best_genome_ever = None
    best_fitness_ever = -np.inf

    if resume_from and Path(resume_from).exists():
        if verbose:
            print(f"\nResuming from checkpoint: {resume_from}")
        with open(resume_from, 'r') as f:
            checkpoint = json.load(f)
        start_generation = checkpoint['generation']
        population = [NetworkGenome(genes=g) for g in checkpoint['population']]
        history = checkpoint['history']
        best_genome_ever = NetworkGenome(genes=checkpoint['best_genome'])
        best_fitness_ever = checkpoint['best_fitness']
        if verbose:
            print(f"Resuming from generation {start_generation}/{generations}")
            print(f"Best fitness so far: {best_fitness_ever:.4f}\n")

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n{'='*80}")
        print(f"MULTI-PROBLEM ARCHITECTURE EVOLUTION")
        print(f"{'='*80}")
        print(f"Training problems: {len(problems)}")
        for prob, path in problems:
            print(f"  - {path.stem}: {prob.bin_dimensions}, {len(prob.items)} item types")
        print(f"Base population size: {population_size}")
        if adaptive_population:
            print(f"Adaptive population: enabled (1.5x early -> 1.0x mid -> 0.75x late)")
        else:
            print(f"Adaptive population: disabled (fixed size)")
        print(f"Generations: {generations}")
        print(f"Episodes per problem: {episodes_per_problem}")
        if curriculum_schedule:
            print(f"Curriculum learning: enabled")
        print(f"{'='*80}\n")

    # Initialize population (if not resuming)
    if population is None:
        from ga_evolution import get_adaptive_population_size
        if adaptive_population:
            initial_size = get_adaptive_population_size(0, generations, population_size, 1.5, 0.75)
        else:
            initial_size = population_size
        population = create_initial_population(initial_size)

    # Initialize tracking (if not resuming)
    if history is None:
        history = {
            'generation': [],
            'best_fitness': [],
            'avg_fitness': [],
            'std_fitness': [],
            'best_utilization': [],
            'avg_utilization': [],
            'mutation_rate': [],
        }

    # Main evolution loop
    for gen in range(start_generation, generations):
        gen_start_time = time.time()

        if verbose:
            print(f"\n{'='*80}")
            print(f"GENERATION {gen + 1}/{generations}")
            print(f"{'='*80}")

        # Determine curriculum parameters
        current_item_fraction = 1.0
        current_episodes = episodes_per_problem

        if curriculum_schedule:
            gen_thresholds = curriculum_schedule.get('generations', [])
            item_fractions = curriculum_schedule.get('item_fractions', [1.0])
            episode_counts = curriculum_schedule.get('episodes', [episodes_per_problem])

            for i, threshold in enumerate(gen_thresholds):
                if gen >= threshold:
                    current_item_fraction = item_fractions[i]
                    current_episodes = episode_counts[i]

            if verbose:
                print(f"Curriculum: {current_item_fraction*100:.0f}% items, "
                      f"{current_episodes} episodes per problem")

        # Evaluate all genomes
        fitness_scores = []

        for i, genome in enumerate(population):
            if verbose:
                print(f"\n[{i+1}/{population_size}] Evaluating genome {genome.genome_id}...")

            fitness, metrics = evaluate_genome_multi_problem(
                genome=genome,
                problems=problems,
                episodes_per_problem=current_episodes,
                item_fraction=current_item_fraction,
                verbose=verbose
            )

            fitness_scores.append(fitness)
            genome.fitness = fitness
            genome.metrics = metrics

            if verbose:
                print(f"  -> Avg fitness: {fitness:.4f}, "
                      f"Avg util: {metrics['avg_utilization']:.3f}, "
                      f"Complexity: {genome.get_network_complexity():.3f}M params")

        # Sort population by fitness
        population.sort(key=lambda g: g.fitness, reverse=True)

        # Track best
        gen_best = population[0]
        if gen_best.fitness > best_fitness_ever:
            best_fitness_ever = gen_best.fitness
            best_genome_ever = gen_best
            if verbose:
                print(f"\n*** NEW BEST GENOME FOUND ***")
                print(f"   Fitness: {best_fitness_ever:.4f}")
                print(f"   Avg utilization: {gen_best.metrics['avg_utilization']:.3f}")

        # Statistics
        avg_fitness = np.mean(fitness_scores)
        std_fitness = np.std(fitness_scores)
        avg_util = np.mean([g.metrics['avg_utilization'] for g in population])

        history['generation'].append(gen + 1)
        history['best_fitness'].append(gen_best.fitness)
        history['avg_fitness'].append(avg_fitness)
        history['std_fitness'].append(std_fitness)
        history['best_utilization'].append(gen_best.metrics['avg_utilization'])
        history['avg_utilization'].append(avg_util)
        history['mutation_rate'].append(mutation_rate)

        gen_time = time.time() - gen_start_time

        if verbose:
            print(f"\n{'-'*80}")
            print(f"Generation {gen + 1} Summary:")
            print(f"  Best fitness: {gen_best.fitness:.4f}")
            print(f"  Avg fitness: {avg_fitness:.4f} +/- {std_fitness:.4f}")
            print(f"  Best avg utilization: {gen_best.metrics['avg_utilization']:.3f}")
            print(f"  Time: {gen_time:.1f}s")
            print(f"{'-'*80}")

        # Save checkpoint
        if save_dir:
            checkpoint_file = save_dir / "checkpoint_latest.json"
            checkpoint_data = {
                'generation': gen + 1,
                'population': [g.to_dict() for g in population],
                'history': history,
                'best_genome': best_genome_ever.to_dict(),
                'best_fitness': best_fitness_ever,
            }
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)

        # Create next generation
        if gen < generations - 1:
            # Determine target population size for next generation
            from ga_evolution import get_adaptive_population_size
            if adaptive_population:
                target_size = get_adaptive_population_size(gen + 1, generations, population_size, 1.5, 0.75)
            else:
                target_size = population_size

            next_population = []

            # Elitism
            next_population.extend(population[:elite_size])

            # Adaptive mutation
            current_mutation_rate = mutation_rate
            progress = (gen + 1) / generations
            current_mutation_rate = mutation_rate * (1.0 - 0.75 * progress)

            # Fill with offspring
            while len(next_population) < target_size:
                parent1 = tournament_selection(population, 3)
                parent2 = tournament_selection(population, 3)
                child = NetworkGenome.crossover(parent1, parent2, method='uniform')
                child = child.mutate(mutation_rate=current_mutation_rate)
                child.genome_id = len(next_population) + gen * population_size
                next_population.append(child)

            population = next_population

    # Final summary
    if verbose:
        print(f"\n{'='*80}")
        print(f"EVOLUTION COMPLETE")
        print(f"{'='*80}")
        print(f"\nBest genome:")
        print(best_genome_ever)
        print(f"\nPerformance:")
        print(f"  Fitness: {best_genome_ever.fitness:.4f}")
        print(f"  Avg utilization: {best_genome_ever.metrics['avg_utilization']:.3f}")
        print(f"  Complexity: {best_genome_ever.get_network_complexity():.3f}M params")
        print(f"{'='*80}\n")

    # Save final results
    if save_dir:
        best_file = save_dir / "best_genome.json"
        with open(best_file, 'w') as f:
            json.dump(best_genome_ever.to_dict(), f, indent=2)

        history_file = save_dir / "evolution_history.json"
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)

    return best_genome_ever, population


def run_leave_one_out_experiment(
    dataset_dir: str,
    target_problem: str,
    results_dir: str,
    population_size: int = 20,
    generations: int = 15,
    episodes_per_problem: int = 10,
    training_episodes: int = 300,
    use_curriculum: bool = True,
    adaptive_population: bool = True,
    elite_size: int = 2,
    mutation_rate: float = 0.2,
    seed: Optional[int] = None,
    verbose: bool = True,
    resume: bool = True
):
    """
    Run leave-one-out generalization experiment.

    Args:
        dataset_dir: Directory containing problem files
        target_problem: Problem to hold out (e.g., "3dBPP_12")
        results_dir: Directory to save results
        population_size: GA base population size
        generations: Number of GA generations
        episodes_per_problem: Episodes per problem for fitness evaluation
        training_episodes: Episodes for final training on target
        use_curriculum: Enable curriculum learning
        adaptive_population: Enable adaptive population sizing (1.5x early, 0.75x late)
        elite_size: Number of elite genomes
        mutation_rate: Initial mutation rate
        seed: Random seed
        verbose: Print progress
        resume: Resume from checkpoint
    """
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    # Get training and test problems
    training_files = get_problem_files(dataset_dir, exclude_pattern=target_problem)
    target_file = list(Path(dataset_dir).glob(f"{target_problem}.txt"))

    if not target_file:
        raise ValueError(f"Target problem {target_problem} not found in {dataset_dir}")

    target_file = target_file[0]
    target_problem_obj = load_problem(str(target_file))

    training_problems = [(load_problem(str(f)), f) for f in training_files]

    if verbose:
        print("="*80)
        print("EXPERIMENT 2: LEAVE-ONE-OUT GENERALIZATION TEST")
        print("="*80)
        print(f"Dataset directory: {dataset_dir}")
        print(f"Results directory: {results_dir}")
        print(f"\nTarget (held-out) problem: {target_problem}")
        print(f"  Dimensions: {target_problem_obj.bin_dimensions}")
        print(f"  Items: {len(target_problem_obj.items)} types")
        print(f"\nTraining problems: {len(training_problems)}")
        for prob, path in training_problems:
            print(f"  - {path.stem}")
        print(f"\nGA Configuration:")
        print(f"  Base population size: {population_size}")
        if adaptive_population:
            print(f"  Adaptive population: enabled (1.5x early -> 1.0x mid -> 0.75x late)")
        else:
            print(f"  Adaptive population: disabled (fixed size)")
        print(f"  Generations: {generations}")
        print(f"  Episodes per problem: {episodes_per_problem}")
        print(f"  Curriculum learning: {use_curriculum}")
        print("="*80)
        print()

    # Curriculum schedule
    curriculum_schedule = None
    if use_curriculum:
        curriculum_schedule = {
            'generations': [0, generations // 3, 2 * generations // 3],
            'item_fractions': [0.3, 0.6, 1.0],
            'episodes': [
                max(5, episodes_per_problem // 2),
                max(7, int(episodes_per_problem * 0.75)),
                episodes_per_problem
            ]
        }

    # Phase 1: Evolve on training problems
    evolution_dir = results_path / "evolution"
    checkpoint_file = evolution_dir / "checkpoint_latest.json"
    resume_from = str(checkpoint_file) if (resume and checkpoint_file.exists()) else None

    if verbose:
        print(f"\n[Phase 1/2] Evolving architecture on {len(training_problems)} training problems...")

    best_genome, final_population = evolve_multi_problem_architecture(
        problems=training_problems,
        population_size=population_size,
        generations=generations,
        episodes_per_problem=episodes_per_problem,
        adaptive_population=adaptive_population,
        elite_size=elite_size,
        mutation_rate=mutation_rate,
        seed=seed,
        save_dir=str(evolution_dir),
        verbose=verbose,
        curriculum_schedule=curriculum_schedule,
        resume_from=resume_from
    )

    # Save evolved genome
    genome_file = results_path / "evolved_genome.json"
    with open(genome_file, 'w') as f:
        json.dump(best_genome.to_dict(), f, indent=2)

    # Phase 2: Train and evaluate on target problem
    if verbose:
        print(f"\n[Phase 2/3] Training on target problem: {target_problem}...")

    model_save_path = results_path / "trained_model.pth"

    agent = train_with_evolved_genome(
        problem_path=str(target_file),
        genome=best_genome,
        episodes=training_episodes,
        save_path=None  # Don't save full checkpoint
    )

    # Save minimal checkpoint (weights + architecture + performance only)
    # Get performance metrics from agent's training history
    from packing_with_dqncore2_enhanced import MultiBinPackingEnv, load_problem_as_items
    temp_items = load_problem_as_items(target_problem_obj)
    temp_env = MultiBinPackingEnv(
        target_problem_obj.bin_dimensions[0], target_problem_obj.bin_dimensions[1], target_problem_obj.bin_dimensions[2],
        items=temp_items, max_actions=128, topk_eps=1000, seed=42, gamma=0.992, problem=target_problem_obj
    )

    # Run quick evaluation to get best metrics
    best_bins_count = float('inf')
    best_items_count = 0
    best_util = 0.0

    for _ in range(5):
        obs = temp_env.reset(items=temp_items.copy())
        done = False
        steps = 0
        agent._eps = 0.0

        while not done and steps < 1000:
            from packing_with_dqncore2_enhanced import build_action_features
            from nesting.heightmap_utils import extract_patches_for_actions
            actions, mask = temp_env.action_space()
            if len(actions) == 0:
                break
            feats = build_action_features(temp_env, actions)
            patches = extract_patches_for_actions(temp_env, actions, patch_size=best_genome.genes['patch_size'])
            act_idx = agent.select_action(obs, feats, patches, mask)
            if act_idx is None or act_idx >= len(actions):
                break
            obs, rew, done, info = temp_env.step(actions[act_idx])
            steps += 1

        bins_used = len([b for b in temp_env.bins if len(b.placed) > 0])
        items_packed = sum(len(b.placed) for b in temp_env.bins)
        total_vol = sum(sum(it.w * it.d * it.h for it in b.placed) for b in temp_env.bins)
        utilization = total_vol / (temp_env.bin_volume * bins_used) if bins_used > 0 else 0

        if items_packed > best_items_count or (items_packed == best_items_count and bins_used < best_bins_count):
            best_bins_count = bins_used
            best_items_count = items_packed
            best_util = utilization

    del temp_env

    # Save minimal checkpoint
    torch.save({
        'model_state_dict': agent.q.state_dict(),
        'genome': best_genome.to_dict(),
        'best_bins': best_bins_count,
        'best_items': best_items_count,
        'best_util': best_util
    }, str(model_save_path), pickle_protocol=4)

    # Phase 3: Visualize best packing solution
    if verbose:
        print(f"\n[Phase 3/3] Generating visualizations for {target_problem}...")

    # Create visualization directory
    viz_dir = results_path / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    # Reload environment to get best solution
    from packing_with_dqncore2_enhanced import MultiBinPackingEnv, load_problem_as_items
    items = load_problem_as_items(target_problem_obj)
    W, D, H = target_problem_obj.bin_dimensions

    env = MultiBinPackingEnv(
        W, D, H,
        items=items,
        max_actions=128,
        topk_eps=1000,
        seed=42,
        gamma=0.992,
        problem=target_problem_obj
    )

    # Run one episode with trained agent to get solution
    agent._eps = 0.0  # Greedy evaluation (set internal variable directly)
    obs = env.reset(items=items.copy())
    done = False
    step_count = 0

    while not done and step_count < 1000:
        from packing_with_dqncore2_enhanced import build_action_features
        from nesting.heightmap_utils import extract_patches_for_actions

        actions, mask_short = env.action_space()
        if len(actions) == 0:
            break

        feats = build_action_features(env, actions)
        patches = extract_patches_for_actions(env, actions, patch_size=best_genome.genes['patch_size'])

        act_idx = agent.select_action(obs, feats, patches, mask_short)
        if act_idx is None or act_idx >= len(actions):
            break

        obs, rew, done, info = env.step(actions[act_idx])
        step_count += 1

    # Visualize all bins with items
    for i, bin_obj in enumerate(env.bins):
        if len(bin_obj.placed) > 0:
            bin_util = sum(b.w * b.d * b.h for b in bin_obj.placed) / env.bin_volume

            # Save filled version (clean)
            bin_obj.plot3d_filled(
                title=f"{target_problem} - Bin {i+1} ({len(bin_obj.placed)} items, util:{bin_util:.3f})",
                save_path=str(viz_dir / f"bin_{i+1}_filled.png"),
                show=False
            )

            # Save version with EMS (for analysis)
            bin_obj.plot3d(
                title=f"{target_problem} - Bin {i+1} ({len(bin_obj.placed)} items, util:{bin_util:.3f}) [with EMS]",
                save_path=str(viz_dir / f"bin_{i+1}_with_ems.png"),
                show=False
            )

    if verbose:
        print(f"  Visualizations saved to: {viz_dir}")

    # Clean up
    del agent
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load and record results
    checkpoint = torch.load(str(model_save_path), map_location='cpu', weights_only=False)

    results = {
        'experiment_type': 'leave_one_out',
        'target_problem': target_problem,
        'training_problems': [p.stem for _, p in training_problems],
        'evolution': {
            'generations': generations,
            'population_size': population_size,
            'best_fitness': best_genome.fitness,
            'training_metrics': best_genome.metrics,
        },
        'target_evaluation': {
            'training_episodes': training_episodes,
            'best_utilization': checkpoint.get('best_util', 0.0),
            'best_bins': checkpoint.get('best_bins', 0),
            'best_items': checkpoint.get('best_items', 0),
        },
        'architecture': {
            'hidden_dim': best_genome.genes['hidden_dim'],
            'enc_layers': best_genome.genes['enc_layers'],
            'attention_type': best_genome.genes['attention_type'],
            'complexity': best_genome.get_network_complexity(),
            'genome': best_genome.to_dict()
        }
    }

    # Save results
    results_file = results_path / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(f"\n{'='*80}")
        print("EXPERIMENT COMPLETE")
        print(f"{'='*80}")
        print(f"Target problem: {target_problem}")
        print(f"Best utilization: {results['target_evaluation']['best_utilization']:.3f}")
        print(f"Best bins: {results['target_evaluation']['best_bins']}")
        print(f"Architecture complexity: {results['architecture']['complexity']:.3f}M params")
        print(f"\nResults saved to: {results_file}")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run leave-one-out generalization experiment"
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        required=True,
        help='Directory containing problem .txt files'
    )
    parser.add_argument(
        '--target-problem',
        type=str,
        required=True,
        help='Problem to hold out for testing (e.g., "3dBPP_12")'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default=None,
        help='Directory to save results (default: results/experiment_loo_{target})'
    )
    parser.add_argument(
        '--population-size',
        type=int,
        default=20,
        help='GA population size'
    )
    parser.add_argument(
        '--generations',
        type=int,
        default=15,
        help='Number of GA generations'
    )
    parser.add_argument(
        '--episodes-per-problem',
        type=int,
        default=10,
        help='Episodes per problem for fitness evaluation'
    )
    parser.add_argument(
        '--training-episodes',
        type=int,
        default=300,
        help='Episodes for final training on target'
    )
    parser.add_argument(
        '--no-curriculum',
        action='store_true',
        help='Disable curriculum learning'
    )
    parser.add_argument(
        '--no-adaptive-population',
        action='store_true',
        help='Disable adaptive population sizing (use fixed population)'
    )
    parser.add_argument(
        '--elite-size',
        type=int,
        default=2,
        help='Number of elite genomes'
    )
    parser.add_argument(
        '--mutation-rate',
        type=float,
        default=0.2,
        help='Initial mutation rate'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Start from scratch'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Minimal output'
    )

    args = parser.parse_args()

    # Default results directory
    if args.results_dir is None:
        args.results_dir = f"results/experiment_leave_one_out_{args.target_problem}"

    run_leave_one_out_experiment(
        dataset_dir=args.dataset_dir,
        target_problem=args.target_problem,
        results_dir=args.results_dir,
        population_size=args.population_size,
        generations=args.generations,
        episodes_per_problem=args.episodes_per_problem,
        training_episodes=args.training_episodes,
        use_curriculum=not args.no_curriculum,
        adaptive_population=not args.no_adaptive_population,
        elite_size=args.elite_size,
        mutation_rate=args.mutation_rate,
        seed=args.seed,
        verbose=not args.quiet,
        resume=not args.no_resume
    )


if __name__ == "__main__":
    main()
