"""
Experiment: Multi-Problem Neural Network Training

This experiment evolves a neural architecture that learns to solve multiple
different bin packing problems simultaneously. The architecture is evolved
using a genetic algorithm where genomes are evaluated by training on multiple
training problems in a round-robin fashion, then tested on separate test problems.

Workflow:
1. Load training problem datasets (multiple problem IDs)
2. Load test problem datasets (separate problem IDs)
3. Run genetic algorithm (GA):
   - Each genome is trained using round-robin cycling through training datasets
   - After training, evaluate genome at eps=0 on all training datasets
   - Average utilization across training datasets = genome fitness
4. Train final evolved architecture (2000 episodes, round-robin)
5. Evaluate on test datasets

Supports:
- Multi-problem round-robin training
- Curriculum learning (progressive difficulty during evolution)
- Checkpoint/resume capability for long Metacentrum runs
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

import matplotlib
matplotlib.use("Agg")

from nesting.dataset_loader import load_problem, BinPackingProblem
from genome import NetworkGenome, create_initial_population, tournament_selection
from example_ga_training import train_with_evolved_genome


def get_problem_files_multi(dataset_dir: str, problem_ids: List[int]) -> List[Path]:
    """
    Get all problem files for specified problem IDs.

    Args:
        dataset_dir: Path to directory containing problem .txt files
        problem_ids: List of problem IDs to load (e.g., [1, 2, 3])

    Returns:
        List of Path objects for problem files
    """
    dataset_path = Path(dataset_dir)
    problem_files = []

    for problem_id in problem_ids:
        files = sorted(dataset_path.glob(f"3dBPP_{problem_id}.txt"))
        # Exclude solution files
        files = [f for f in files if not f.name.endswith("_sol.txt")]
        problem_files.extend(files)

    return problem_files


def evaluate_genome_multi_problem(
    genome: NetworkGenome,
    training_problems: List[Tuple[BinPackingProblem, Path]],
    episodes: int = 100,
    item_fraction: float = 1.0,
    verbose: bool = False
) -> Tuple[float, Dict]:
    """
    Evaluate genome fitness by training on multiple problems in round-robin fashion,
    then evaluating at eps=0 on all training problems.

    Training Strategy:
    - Cycles through training problems episode by episode (round-robin)
    - Episode 1: problem 0, Episode 2: problem 1, ..., Episode N: problem 0 again
    - After training, evaluate at eps=0 on ALL training problems
    - Average utilization across all problems = final fitness

    Args:
        genome: NetworkGenome to evaluate
        training_problems: List of (problem, problem_file_path) tuples
        episodes: Total training episodes (distributed across problems)
        item_fraction: Fraction of items to use (for curriculum learning)
        verbose: Print detailed progress

    Returns:
        (fitness_score, metrics_dict)
    """
    import torch
    import gc
    from packing_with_dqncore2_enhanced import (
        MultiBinPackingEnv, load_problem_as_items, build_action_features,
        pad_feats_mask, ACTION_FEAT_DIM, _build_constraint_cache
    )
    from nesting.heightmap_utils import extract_patches_for_actions, pad_patches

    if verbose:
        print(f"  Training on {len(training_problems)} problems for {episodes} episodes (round-robin)...")

    # Load all problem environments
    envs_and_items = []
    for problem, problem_path in training_problems:
        items = load_problem_as_items(problem)

        # Apply curriculum learning
        if item_fraction < 1.0:
            num_items = max(1, int(len(items) * item_fraction))
            items = items[:num_items]

        W, D, H = problem.bin_dimensions
        env = MultiBinPackingEnv(
            W, D, H,
            items=items.copy(),
            max_actions=128,
            topk_eps=1000,
            seed=42,
            gamma=0.992,
            problem=problem
        )
        # Pre-compute constraint cache once per problem
        constraint_cache = _build_constraint_cache(problem)
        envs_and_items.append((env, items, problem_path, constraint_cache))

    # Build network from genome
    first_env = envs_and_items[0][0]
    cfg = genome.to_dqn_config(
        obs_dim=8,
        action_feat_dim=ACTION_FEAT_DIM,
        max_actions=first_env.max_actions,
        device="cuda" if torch.cuda.is_available() else "cpu",
        total_episodes=episodes
    )

    # Reduce buffer size during GA to save memory
    cfg.buffer_size = 15_000

    from dqn_core.dqn_enhanced import DQNAgentEnhanced
    agent = DQNAgentEnhanced(cfg)

    try:
        # Round-robin training across all problems
        num_problems = len(envs_and_items)

        for episode in range(episodes):
            # Select which problem to train on this episode (round-robin)
            problem_idx = episode % num_problems
            env, items, problem_path, constraint_cache = envs_and_items[problem_idx]

            # Reset environment for this problem
            obs = env.reset(items=items.copy())
            done = False
            step_count = 0
            episode_reward = 0.0

            patch_size = genome.genes['patch_size']

            while not done and step_count < 1000:
                # Get valid actions for current state
                actions, mask = env.action_space()
                if len(actions) == 0:
                    break

                # Build features for current state
                feats = build_action_features(env, actions, constraint_cache) if len(actions) > 0 else np.zeros((0, ACTION_FEAT_DIM), np.float32)
                patches = extract_patches_for_actions(env, actions, patch_size=patch_size) if len(actions) > 0 else np.zeros((0, patch_size, patch_size), np.float32)

                # Pad current features
                currF, currM = pad_feats_mask(
                    feats if feats.shape[0] > 0 else np.zeros((0, ACTION_FEAT_DIM), np.float32),
                    mask if mask.shape[0] > 0 else np.zeros((0,), np.float32),
                    env.max_actions
                )
                currP = pad_patches(
                    patches if patches.shape[0] > 0 else np.zeros((0, patch_size, patch_size), np.float32),
                    env.max_actions,
                    patch_size
                )

                # Select action (use unpadded features for selection)
                act_idx = agent.select_action(
                    obs,
                    feats if feats.shape[0] > 0 else np.zeros((1, ACTION_FEAT_DIM), np.float32),
                    patches if patches.shape[0] > 0 else np.zeros((1, patch_size, patch_size), np.float32),
                    mask if mask.shape[0] > 0 else np.zeros((1,), np.float32)
                )

                if act_idx is None or act_idx >= len(actions):
                    break

                # Step environment
                next_obs, reward, done, info = env.step(actions[act_idx])
                episode_reward += reward

                # Get next state actions and features
                next_actions, next_mask = env.action_space()
                next_feats = build_action_features(env, next_actions, constraint_cache) if len(next_actions) > 0 else np.zeros((0, ACTION_FEAT_DIM), np.float32)
                next_patches = extract_patches_for_actions(env, next_actions, patch_size=patch_size) if len(next_actions) > 0 else np.zeros((0, patch_size, patch_size), np.float32)

                # Pad next features
                nextF, nextM = pad_feats_mask(
                    next_feats,
                    next_mask if next_mask.shape[0] > 0 else np.zeros((0,), np.float32),
                    env.max_actions
                )
                nextP = pad_patches(
                    next_patches if next_patches.shape[0] > 0 else np.zeros((0, patch_size, patch_size), np.float32),
                    env.max_actions,
                    patch_size
                )

                # Store transition
                agent.store(obs, act_idx, reward, next_obs, done,
                           curr_action_feats=currF, curr_mask=currM, curr_patches=currP,
                           next_action_feats=nextF, next_mask=nextM, next_patches=nextP)

                obs = next_obs
                step_count += 1

                # Train agent periodically
                if episode > 0 and step_count % 5 == 0:
                    agent.train_step()

            # Update episode counter for epsilon decay
            agent.on_episode_end()

        # Evaluation phase: Evaluate at eps=0 on ALL training problems
        if verbose:
            print(f"  Evaluating at eps=0 on {num_problems} training problems...")

        eval_metrics = []
        for env, items, problem_path in envs_and_items:
            # Run 5 evaluation episodes per problem at eps=0
            problem_utils = []
            problem_bins = []

            for _ in range(5):
                obs = env.reset(items=items.copy())
                done = False
                step_count = 0
                agent._eps = 0.0  # Greedy evaluation

                while not done and step_count < 1000:
                    actions, mask = env.action_space()
                    if len(actions) == 0:
                        break

                    feats = build_action_features(env, actions)
                    patches = extract_patches_for_actions(env, actions, patch_size=genome.genes['patch_size'])

                    act_idx = agent.select_action(obs, feats, patches, mask)
                    if act_idx is None or act_idx >= len(actions):
                        break

                    obs, reward, done, info = env.step(actions[act_idx])
                    step_count += 1

                # Calculate metrics
                bins_used = len([b for b in env.bins if len(b.placed) > 0])
                items_packed = sum(len(b.placed) for b in env.bins)
                total_vol = sum(sum(it.w * it.d * it.h for it in b.placed) for b in env.bins)
                utilization = total_vol / (env.bin_volume * bins_used) if bins_used > 0 else 0.0

                problem_utils.append(utilization)
                problem_bins.append(bins_used)

            # Average metrics for this problem
            avg_util = np.mean(problem_utils)
            avg_bins = np.mean(problem_bins)

            eval_metrics.append({
                'problem': problem_path.stem,
                'utilization': avg_util,
                'bins': avg_bins
            })

            if verbose:
                print(f"    {problem_path.stem}: util={avg_util:.3f}, bins={avg_bins:.1f}")

        # Calculate overall fitness (average utilization across all training problems)
        avg_utilization = np.mean([m['utilization'] for m in eval_metrics])
        avg_bins_used = np.mean([m['bins'] for m in eval_metrics])

        # Fitness = utilization - small complexity penalty
        complexity = genome.get_network_complexity()
        complexity_penalty = 0.01 * complexity  # Small penalty for large networks
        fitness = avg_utilization - complexity_penalty

        genome.fitness = fitness
        genome.metrics = {
            'avg_utilization': avg_utilization,
            'avg_bins_used': avg_bins_used,
            'complexity': complexity,
            'per_problem': eval_metrics
        }

        if verbose:
            print(f"  Overall: fitness={fitness:.4f}, avg_util={avg_utilization:.3f}, "
                  f"complexity={complexity:.3f}M params")

        return fitness, genome.metrics

    finally:
        # Clean up
        del agent
        for env, _, _ in envs_and_items:
            del env
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def evolve_multi_problem_architecture(
    training_problems: List[Tuple[BinPackingProblem, Path]],
    population_size: int = 30,
    generations: int = 50,
    episodes_per_eval: int = 100,
    elite_size: int = 3,
    mutation_rate: float = 0.2,
    adaptive_population: bool = True,
    seed: Optional[int] = 42,
    save_dir: Optional[str] = None,
    verbose: bool = True,
    curriculum_schedule: Optional[Dict] = None,
    resume_from: Optional[str] = None
) -> Tuple[NetworkGenome, List[NetworkGenome]]:
    """
    Evolve neural architecture using genetic algorithm on multiple problems.

    Args:
        training_problems: List of (problem, problem_file_path) tuples
        population_size: Base GA population size
        generations: Number of GA generations
        episodes_per_eval: Episodes per genome evaluation
        elite_size: Number of elite genomes to preserve
        mutation_rate: Initial mutation rate
        adaptive_population: Enable adaptive population sizing
        seed: Random seed for reproducibility
        save_dir: Directory to save results
        verbose: Print detailed progress
        curriculum_schedule: Curriculum learning schedule
        resume_from: Checkpoint file to resume from

    Returns:
        (best_genome, final_population)
    """
    import random
    from pathlib import Path
    from ga_evolution import get_adaptive_population_size

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

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
        # Restore full population with fitness and metrics
        population = [NetworkGenome.from_dict(g) for g in checkpoint['population']]
        history = checkpoint['history']
        # Restore best genome with all data
        best_genome_ever = NetworkGenome.from_dict(checkpoint['best_genome'])
        best_fitness_ever = checkpoint['best_fitness']
        if verbose:
            print(f"Resuming from generation {start_generation}/{generations}")
            print(f"Population size: {len(population)}")
            print(f"Best fitness so far: {best_fitness_ever:.4f}")
            print(f"Skipping re-evaluation of loaded population\n")

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n{'='*80}")
        print(f"MULTI-PROBLEM ARCHITECTURE EVOLUTION")
        print(f"{'='*80}")
        print(f"Training problems: {len(training_problems)}")
        for prob, path in training_problems:
            print(f"  - {path.stem}: {prob.bin_dimensions}, {len(prob.items)} item types")
        print(f"Base population size: {population_size}")
        if adaptive_population:
            print(f"Adaptive population: enabled (1.5x early -> 1.0x mid -> 0.75x late)")
        else:
            print(f"Adaptive population: disabled (fixed size)")
        print(f"Generations: {generations}")
        print(f"Episodes per eval: {episodes_per_eval}")
        if curriculum_schedule:
            print(f"Curriculum learning: enabled")
        print(f"{'='*80}\n")

    # Initialize population (if not resuming)
    if population is None:
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
            'best_bins': [],
            'avg_bins': [],
            'std_bins': [],
            'avg_complexity': [],
            'std_complexity': [],
            'mutation_rate': [],
            'diversity_score': [],
            'best_items_packed': [],
            'avg_items_packed': [],
            'packing_success_rate': [],
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
        current_episodes = episodes_per_eval

        if curriculum_schedule:
            gen_thresholds = curriculum_schedule.get('generations', [])
            item_fractions = curriculum_schedule.get('item_fractions', [1.0])
            episode_counts = curriculum_schedule.get('episodes', [episodes_per_eval])

            for i, threshold in enumerate(gen_thresholds):
                if gen >= threshold:
                    current_item_fraction = item_fractions[i]
                    current_episodes = episode_counts[i]

            if verbose:
                print(f"Curriculum: {current_item_fraction*100:.0f}% items, "
                      f"{current_episodes} episodes")

        # Evaluate all genomes
        # Skip evaluation if population was just loaded from checkpoint (already evaluated)
        fitness_scores = []
        needs_evaluation = any(g.fitness is None for g in population)

        if needs_evaluation:
            for i, genome in enumerate(population):
                if verbose:
                    print(f"\n[{i+1}/{len(population)}] Evaluating genome {genome.genome_id}...")

                fitness, metrics = evaluate_genome_multi_problem(
                    genome=genome,
                    training_problems=training_problems,
                    episodes=current_episodes,
                    item_fraction=current_item_fraction,
                    verbose=verbose
                )

                fitness_scores.append(fitness)
                genome.fitness = fitness
                genome.metrics = metrics
        else:
            # Population already evaluated (loaded from checkpoint)
            if verbose:
                print(f"\nPopulation already evaluated (loaded from checkpoint), skipping evaluation...")
            fitness_scores = [g.fitness for g in population]

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
        avg_bins = np.mean([g.metrics.get('avg_bins_used', 0) for g in population])
        std_bins = np.std([g.metrics.get('avg_bins_used', 0) for g in population])
        avg_complexity = np.mean([g.get_network_complexity() for g in population])
        std_complexity = np.std([g.get_network_complexity() for g in population])

        # Calculate diversity (unique gene combinations)
        gene_signatures = set()
        for g in population:
            sig = tuple(sorted(g.genes.items(), key=lambda x: x[0]))
            gene_signatures.add(str(sig))
        diversity_score = len(gene_signatures) / len(population)

        history['generation'].append(gen + 1)
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
        history['mutation_rate'].append(mutation_rate)
        history['diversity_score'].append(diversity_score)
        # For items packed, we'll use a placeholder since multi-problem doesn't track this
        history['best_items_packed'].append(0)
        history['avg_items_packed'].append(0)
        history['packing_success_rate'].append(1.0)

        gen_time = time.time() - gen_start_time

        if verbose:
            print(f"\n{'-'*80}")
            print(f"Generation {gen + 1} Summary:")
            print(f"  Best fitness: {gen_best.fitness:.4f}")
            print(f"  Avg fitness: {avg_fitness:.4f} +/- {std_fitness:.4f}")
            print(f"  Best avg utilization: {gen_best.metrics['avg_utilization']:.3f}")
            print(f"  Avg utilization: {avg_util:.3f}")
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

            # Also save generation-specific file
            gen_file = save_dir / f"generation_{gen+1:03d}.json"
            with open(gen_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)

        # Create next generation
        if gen < generations - 1:
            # Determine target population size for next generation
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


def run_multi_problem_experiment(
    dataset_dir: str,
    results_dir: str,
    training_problem_ids: List[int],
    test_problem_ids: List[int],
    population_size: int = 30,
    generations: int = 50,
    episodes_per_eval: int = 100,
    training_episodes: int = 2000,
    use_curriculum: bool = True,
    adaptive_population: bool = True,
    elite_size: int = 3,
    mutation_rate: float = 0.2,
    seed: Optional[int] = 42,
    verbose: bool = True,
    resume: bool = False
):
    """
    Run multi-problem neural architecture evolution experiment.

    Args:
        dataset_dir: Directory containing problem files
        results_dir: Directory to save results
        training_problem_ids: List of problem IDs for training (e.g., [1, 2, 3])
        test_problem_ids: List of problem IDs for testing (e.g., [12, 13])
        population_size: GA base population size
        generations: Number of GA generations
        episodes_per_eval: Episodes for genome fitness evaluation
        training_episodes: Episodes for final training
        use_curriculum: Enable curriculum learning
        adaptive_population: Enable adaptive population sizing
        elite_size: Number of elite genomes to preserve
        mutation_rate: Initial mutation rate
        seed: Random seed for reproducibility
        verbose: Print detailed progress
        resume: Resume from checkpoint if available
    """
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    # Load training and test problems
    training_files = get_problem_files_multi(dataset_dir, training_problem_ids)
    test_files = get_problem_files_multi(dataset_dir, test_problem_ids)

    if len(training_files) == 0:
        raise ValueError(f"No training files found for problem IDs: {training_problem_ids}")
    if len(test_files) == 0:
        raise ValueError(f"No test files found for problem IDs: {test_problem_ids}")

    training_problems = [(load_problem(str(f)), f) for f in training_files]
    test_problems = [(load_problem(str(f)), f) for f in test_files]

    if verbose:
        print("="*80)
        print("EXPERIMENT: MULTI-PROBLEM NEURAL NETWORK TRAINING")
        print("="*80)
        print(f"Dataset directory: {dataset_dir}")
        print(f"Results directory: {results_dir}")
        print(f"\nTraining problems: {len(training_files)}")
        for prob, path in training_problems:
            print(f"  - {path.stem}")
        print(f"\nTest problems: {len(test_files)}")
        for prob, path in test_problems:
            print(f"  - {path.stem}")
        print(f"\nGA Configuration:")
        print(f"  Base population size: {population_size}")
        if adaptive_population:
            print(f"  Adaptive population: enabled (1.5x early -> 1.0x mid -> 0.75x late)")
        else:
            print(f"  Adaptive population: disabled (fixed size)")
        print(f"  Generations: {generations}")
        print(f"  Episodes per eval: {episodes_per_eval}")
        print(f"  Curriculum learning: {use_curriculum}")
        print(f"\nTraining Configuration:")
        print(f"  Final training episodes: {training_episodes}")
        print("="*80)
        print()

    # Curriculum schedule (if enabled)
    curriculum_schedule = None
    if use_curriculum:
        curriculum_schedule = {
            'generations': [0, generations // 3, 2 * generations // 3],
            'item_fractions': [0.3, 0.6, 1.0],
            'episodes': [
                max(10, episodes_per_eval // 2),
                max(15, int(episodes_per_eval * 0.75)),
                episodes_per_eval
            ]
        }
        if verbose:
            print("Curriculum Schedule:")
            for i, gen_start in enumerate(curriculum_schedule['generations']):
                print(f"  Gen {gen_start}+: "
                      f"{curriculum_schedule['item_fractions'][i]*100:.0f}% items, "
                      f"{curriculum_schedule['episodes'][i]} episodes")
            print()

    # Phase 1: Evolve architecture
    evolution_dir = results_path / "evolution"
    checkpoint_file = evolution_dir / "checkpoint_latest.json"
    resume_from = str(checkpoint_file) if (resume and checkpoint_file.exists()) else None

    if verbose:
        print(f"\n[Phase 1/3] Evolving architecture on {len(training_problems)} training problems...")

    experiment_start_time = time.time()

    best_genome, final_population = evolve_multi_problem_architecture(
        training_problems=training_problems,
        population_size=population_size,
        generations=generations,
        episodes_per_eval=episodes_per_eval,
        elite_size=elite_size,
        mutation_rate=mutation_rate,
        adaptive_population=adaptive_population,
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

    if verbose:
        print(f"\nBest genome saved to: {genome_file}")
        print(f"Evolution fitness: {best_genome.fitness:.4f}")

    # Generate evolution visualizations
    if verbose:
        print(f"\nGenerating evolution visualizations...")

    try:
        from ga_evolution import generate_all_visualizations
        generate_all_visualizations(str(evolution_dir), create_plots_subdir=True)
    except Exception as viz_error:
        if verbose:
            print(f"  Warning: Visualization generation failed: {viz_error}")

    # Phase 2: Train final model on training problems
    if verbose:
        print(f"\n[Phase 2/3] Training final model with {training_episodes} episodes (round-robin on training problems)...")

    # For final training, we'll use a simple approach: round-robin training similar to evaluation
    # but with more episodes, then save the trained model
    from packing_with_dqncore2_enhanced import (
        MultiBinPackingEnv, load_problem_as_items, build_action_features,
        pad_feats_mask, ACTION_FEAT_DIM, _build_constraint_cache
    )
    from nesting.heightmap_utils import extract_patches_for_actions, pad_patches
    from dqn_core.dqn_enhanced import DQNAgentEnhanced

    # Build agent with evolved architecture
    first_problem, _ = training_problems[0]
    items = load_problem_as_items(first_problem)
    W, D, H = first_problem.bin_dimensions
    temp_env = MultiBinPackingEnv(
        W, D, H,
        items=items,
        max_actions=128,
        topk_eps=1000,
        seed=seed,
        gamma=0.992,
        problem=first_problem
    )

    cfg = best_genome.to_dqn_config(
        obs_dim=8,
        action_feat_dim=ACTION_FEAT_DIM,
        max_actions=temp_env.max_actions,
        device="cuda" if torch.cuda.is_available() else "cpu",
        total_episodes=training_episodes
    )

    agent = DQNAgentEnhanced(cfg)

    # Prepare all training environments
    training_envs = []
    for problem, path in training_problems:
        items = load_problem_as_items(problem)
        W, D, H = problem.bin_dimensions
        env = MultiBinPackingEnv(
            W, D, H,
            items=items,
            max_actions=128,
            topk_eps=1000,
            seed=seed,
            gamma=0.992,
            problem=problem
        )
        constraint_cache = _build_constraint_cache(problem)
        training_envs.append((env, items, path, constraint_cache))

    # Round-robin training
    num_training_problems = len(training_envs)
    patch_size = best_genome.genes['patch_size']

    for episode in range(training_episodes):
        problem_idx = episode % num_training_problems
        env, items, path, constraint_cache = training_envs[problem_idx]

        obs = env.reset(items=items.copy())
        done = False
        step_count = 0

        while not done and step_count < 1000:
            # Current state actions and features
            actions, mask = env.action_space()
            if len(actions) == 0:
                break

            feats = build_action_features(env, actions, constraint_cache) if len(actions) > 0 else np.zeros((0, ACTION_FEAT_DIM), np.float32)
            patches = extract_patches_for_actions(env, actions, patch_size=patch_size) if len(actions) > 0 else np.zeros((0, patch_size, patch_size), np.float32)

            # Pad current features
            currF, currM = pad_feats_mask(
                feats if feats.shape[0] > 0 else np.zeros((0, ACTION_FEAT_DIM), np.float32),
                mask if mask.shape[0] > 0 else np.zeros((0,), np.float32),
                env.max_actions
            )
            currP = pad_patches(
                patches if patches.shape[0] > 0 else np.zeros((0, patch_size, patch_size), np.float32),
                env.max_actions,
                patch_size
            )

            # Select action
            act_idx = agent.select_action(
                obs,
                feats if feats.shape[0] > 0 else np.zeros((1, ACTION_FEAT_DIM), np.float32),
                patches if patches.shape[0] > 0 else np.zeros((1, patch_size, patch_size), np.float32),
                mask if mask.shape[0] > 0 else np.zeros((1,), np.float32)
            )

            if act_idx is None or act_idx >= len(actions):
                break

            # Step environment
            next_obs, reward, done, info = env.step(actions[act_idx])

            # Next state actions and features
            next_actions, next_mask = env.action_space()
            next_feats = build_action_features(env, next_actions, constraint_cache) if len(next_actions) > 0 else np.zeros((0, ACTION_FEAT_DIM), np.float32)
            next_patches = extract_patches_for_actions(env, next_actions, patch_size=patch_size) if len(next_actions) > 0 else np.zeros((0, patch_size, patch_size), np.float32)

            # Pad next features
            nextF, nextM = pad_feats_mask(
                next_feats,
                next_mask if next_mask.shape[0] > 0 else np.zeros((0,), np.float32),
                env.max_actions
            )
            nextP = pad_patches(
                next_patches if next_patches.shape[0] > 0 else np.zeros((0, patch_size, patch_size), np.float32),
                env.max_actions,
                patch_size
            )

            # Store transition
            agent.store(obs, act_idx, reward, next_obs, done,
                       curr_action_feats=currF, curr_mask=currM, curr_patches=currP,
                       next_action_feats=nextF, next_mask=nextM, next_patches=nextP)

            obs = next_obs
            step_count += 1

            if episode > 0 and step_count % 5 == 0:
                agent.train_step()

        # Update episode counter for epsilon decay
        agent.on_episode_end()

        if verbose and (episode + 1) % 100 == 0:
            print(f"  Episode {episode+1}/{training_episodes} completed")

    # Phase 3: Evaluate trained model on test problems and generate visualizations
    if verbose:
        print(f"\n[Phase 3/3] Evaluating on {len(test_problems)} test problems...")

    test_results = []
    test_visualizations = {}  # Store best episode state for each problem

    for problem, path in test_problems:
        items = load_problem_as_items(problem)
        W, D, H = problem.bin_dimensions

        env = MultiBinPackingEnv(
            W, D, H,
            items=items,
            max_actions=128,
            topk_eps=1000,
            seed=seed,
            gamma=0.992,
            problem=problem
        )

        # Run multiple evaluation episodes, track best for visualization
        problem_utils = []
        problem_bins = []
        problem_items = []
        best_util = 0.0
        best_env_state = None

        for eval_ep in range(10):
            obs = env.reset(items=items.copy())
            done = False
            step_count = 0
            agent._eps = 0.0

            while not done and step_count < 1000:
                actions, mask = env.action_space()
                if len(actions) == 0:
                    break

                feats = build_action_features(env, actions)
                patches = extract_patches_for_actions(env, actions, patch_size=best_genome.genes['patch_size'])

                act_idx = agent.select_action(obs, feats, patches, mask)
                if act_idx is None or act_idx >= len(actions):
                    break

                obs, reward, done, info = env.step(actions[act_idx])
                step_count += 1

            bins_used = len([b for b in env.bins if len(b.placed) > 0])
            items_packed = sum(len(b.placed) for b in env.bins)
            total_vol = sum(sum(it.w * it.d * it.h for it in b.placed) for b in env.bins)
            utilization = total_vol / (env.bin_volume * bins_used) if bins_used > 0 else 0.0

            problem_utils.append(utilization)
            problem_bins.append(bins_used)
            problem_items.append(items_packed)

            # Track best episode for visualization
            if utilization > best_util:
                best_util = utilization
                # Store a copy of the bins for visualization
                import copy
                best_env_state = copy.deepcopy(env.bins)

        test_results.append({
            'problem': path.stem,
            'avg_utilization': np.mean(problem_utils),
            'avg_bins': np.mean(problem_bins),
            'avg_items': np.mean(problem_items)
        })

        # Store best episode state for visualization
        test_visualizations[path.stem] = {
            'bins': best_env_state,
            'bin_volume': env.bin_volume,
            'utilization': best_util
        }

        if verbose:
            print(f"  {path.stem}: util={np.mean(problem_utils):.3f}, "
                  f"bins={np.mean(problem_bins):.1f}, items={np.mean(problem_items):.1f}")

    # Calculate average test performance
    avg_test_util = np.mean([r['avg_utilization'] for r in test_results])
    avg_test_bins = np.mean([r['avg_bins'] for r in test_results])

    # Save trained model
    model_save_path = results_path / "trained_model.pth"
    torch.save({
        'model_state_dict': agent.q.state_dict(),
        'genome': best_genome.to_dict(),
        'test_utilization': avg_test_util,
        'test_bins': avg_test_bins,
        'test_results': test_results
    }, str(model_save_path), pickle_protocol=4)

    if verbose:
        print(f"\nTrained model saved to: {model_save_path}")

    # Clean up agent
    del agent
    for env, _, _ in training_envs:
        del env
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Generate bin packing visualizations for each test problem
    if verbose:
        print(f"\nGenerating bin packing visualizations for test problems...")

    for problem_name, viz_data in test_visualizations.items():
        viz_dir = results_path / f"visualizations_{problem_name}"
        viz_dir.mkdir(exist_ok=True)

        bins = viz_data['bins']
        bin_volume = viz_data['bin_volume']

        if bins is not None:
            for i, bin_obj in enumerate(bins):
                if len(bin_obj.placed) > 0:
                    bin_util = sum(b.w * b.d * b.h for b in bin_obj.placed) / bin_volume

                    # Save filled version (clean)
                    bin_obj.plot3d_filled(
                        title=f"{problem_name} - Bin {i+1} ({len(bin_obj.placed)} items, util:{bin_util:.3f})",
                        save_path=str(viz_dir / f"bin_{i+1}_filled.png"),
                        show=False
                    )

                    # Save version with EMS (for analysis)
                    bin_obj.plot3d(
                        title=f"{problem_name} - Bin {i+1} ({len(bin_obj.placed)} items, util:{bin_util:.3f}) [with EMS]",
                        save_path=str(viz_dir / f"bin_{i+1}_with_ems.png"),
                        show=False
                    )

            if verbose:
                print(f"  {problem_name}: Saved {len([b for b in bins if len(b.placed) > 0])} bin visualizations to {viz_dir}")

    # Save final results
    experiment_time = time.time() - experiment_start_time

    results = {
        'experiment_type': 'multi_problem',
        'training_problem_ids': training_problem_ids,
        'test_problem_ids': test_problem_ids,
        'training_problems': [path.stem for _, path in training_problems],
        'test_problems': [path.stem for _, path in test_problems],
        'evolution': {
            'generations': generations,
            'base_population_size': population_size,
            'adaptive_population': adaptive_population,
            'best_fitness': best_genome.fitness,
            'best_genome': best_genome.to_dict()
        },
        'training': {
            'episodes': training_episodes,
            'avg_training_utilization': best_genome.metrics['avg_utilization'],
        },
        'testing': {
            'avg_test_utilization': avg_test_util,
            'avg_test_bins': avg_test_bins,
            'per_problem': test_results
        },
        'architecture': {
            'hidden_dim': best_genome.genes['hidden_dim'],
            'enc_layers': best_genome.genes['enc_layers'],
            'attention_type': best_genome.genes['attention_type'],
            'complexity': best_genome.get_network_complexity()
        },
        'time_seconds': experiment_time
    }

    results_file = results_path / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(f"\n{'='*80}")
        print("EXPERIMENT COMPLETE")
        print(f"{'='*80}")
        print(f"Training problems: {len(training_problems)}")
        print(f"Test problems: {len(test_problems)}")
        print(f"Evolution fitness: {best_genome.fitness:.4f}")
        print(f"Training avg utilization: {best_genome.metrics['avg_utilization']:.3f}")
        print(f"Test avg utilization: {avg_test_util:.3f}")
        print(f"Test avg bins: {avg_test_bins:.1f}")
        print(f"Architecture complexity: {best_genome.get_network_complexity():.3f}M params")
        print(f"Total time: {experiment_time:.1f}s")
        print(f"\nResults saved to: {results_file}")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-problem neural architecture evolution experiment"
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='nesting/inputData/Benchmark dataset and instance generator for Real-World 3dBPP/Input',
        help='Directory containing problem .txt files'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results/experiment_multi_problem',
        help='Directory to save results'
    )
    parser.add_argument(
        '--training-problem-ids',
        type=int,
        nargs='+',
        required=True,
        help='Problem IDs for training (e.g., 1 2 3 5 7)'
    )
    parser.add_argument(
        '--test-problem-ids',
        type=int,
        nargs='+',
        required=True,
        help='Problem IDs for testing (e.g., 12 13)'
    )
    parser.add_argument(
        '--population-size',
        type=int,
        default=30,
        help='GA population size'
    )
    parser.add_argument(
        '--generations',
        type=int,
        default=50,
        help='Number of GA generations'
    )
    parser.add_argument(
        '--episodes-per-eval',
        type=int,
        default=100,
        help='Episodes for fitness evaluation'
    )
    parser.add_argument(
        '--training-episodes',
        type=int,
        default=2000,
        help='Episodes for final training'
    )
    parser.add_argument(
        '--no-curriculum',
        action='store_true',
        help='Disable curriculum learning'
    )
    parser.add_argument(
        '--no-adaptive-population',
        action='store_true',
        help='Disable adaptive population sizing'
    )
    parser.add_argument(
        '--elite-size',
        type=int,
        default=3,
        help='Number of elite genomes to preserve'
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
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Start from scratch (ignore checkpoints)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Minimal output'
    )

    args = parser.parse_args()

    run_multi_problem_experiment(
        dataset_dir=args.dataset_dir,
        results_dir=args.results_dir,
        training_problem_ids=args.training_problem_ids,
        test_problem_ids=args.test_problem_ids,
        population_size=args.population_size,
        generations=args.generations,
        episodes_per_eval=args.episodes_per_eval,
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