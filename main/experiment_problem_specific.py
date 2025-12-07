"""
Experiment 1: Problem-Specific Architecture Evolution

This experiment evolves specialized neural architectures for each individual
problem instance. For each problem:
1. Run neuroevolution (GA) to find optimal architecture
2. Train final model with discovered architecture
3. Evaluate and save results

Supports:
- Curriculum learning (progressive difficulty during evolution)
- Checkpoint/resume capability
- Comprehensive result logging
- Parallel-ready for cluster environments (Metacentrum)
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import numpy as np
import torch

from nesting.dataset_loader import load_problem, BinPackingProblem
from ga_evolution import evolve_architecture
from example_ga_training import train_with_evolved_genome
from genome import NetworkGenome


def get_problem_files(dataset_dir: str) -> List[Path]:
    """
    Get all problem files from dataset directory.

    Args:
        dataset_dir: Path to directory containing problem .txt files

    Returns:
        List of Path objects for problem files
    """
    dataset_path = Path(dataset_dir)
    problem_files = sorted(dataset_path.glob("3dBPP_*.txt"))

    # Exclude solution files
    problem_files = [f for f in problem_files if not f.name.endswith("_sol.txt")]

    return problem_files


def load_experiment_state(results_dir: Path) -> Dict:
    """
    Load experiment state for resume capability.

    Args:
        results_dir: Directory where results are saved

    Returns:
        Dictionary with completed problems and their results
    """
    state_file = results_dir / "experiment_state.json"

    if state_file.exists():
        with open(state_file, 'r') as f:
            return json.load(f)

    return {
        'completed_problems': [],
        'results': {},
        'start_time': datetime.now().isoformat(),
        'last_update': None
    }


def save_experiment_state(results_dir: Path, state: Dict):
    """
    Save experiment state for resume capability.

    Args:
        results_dir: Directory where results are saved
        state: State dictionary to save
    """
    state_file = results_dir / "experiment_state.json"
    state['last_update'] = datetime.now().isoformat()

    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)


def run_problem_specific_experiment(
    dataset_dir: str,
    results_dir: str,
    population_size: int = 20,
    generations: int = 15,
    episodes_per_eval: int = 20,
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
    Run problem-specific architecture evolution experiment.

    Args:
        dataset_dir: Directory containing problem files
        results_dir: Directory to save results
        population_size: GA base population size
        generations: Number of GA generations
        episodes_per_eval: Episodes for fitness evaluation
        training_episodes: Episodes for final training
        use_curriculum: Enable curriculum learning
        adaptive_population: Enable adaptive population sizing (1.5x early, 0.75x late)
        elite_size: Number of elite genomes to preserve
        mutation_rate: Initial mutation rate
        seed: Random seed for reproducibility
        verbose: Print detailed progress
        resume: Resume from checkpoint if available
    """
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    # Load experiment state
    state = load_experiment_state(results_path) if resume else {
        'completed_problems': [],
        'results': {},
        'start_time': datetime.now().isoformat(),
        'last_update': None
    }

    # Get all problem files
    problem_files = get_problem_files(dataset_dir)

    if verbose:
        print("="*80)
        print("EXPERIMENT 1: PROBLEM-SPECIFIC ARCHITECTURE EVOLUTION")
        print("="*80)
        print(f"Dataset directory: {dataset_dir}")
        print(f"Results directory: {results_dir}")
        print(f"Total problems: {len(problem_files)}")
        print(f"Completed: {len(state['completed_problems'])}")
        print(f"Remaining: {len(problem_files) - len(state['completed_problems'])}")
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

    # Process each problem
    for problem_file in problem_files:
        problem_name = problem_file.stem

        # Skip if already completed
        if problem_name in state['completed_problems']:
            if verbose:
                print(f"Skipping {problem_name} (already completed)")
            continue

        if verbose:
            print(f"\n{'='*80}")
            print(f"Processing: {problem_name}")
            print(f"{'='*80}\n")

        problem_start_time = time.time()

        try:
            # Load problem
            problem = load_problem(str(problem_file))

            # Create problem-specific result directory
            problem_dir = results_path / problem_name
            problem_dir.mkdir(parents=True, exist_ok=True)

            # Check for evolution checkpoint
            evolution_dir = problem_dir / "evolution"
            checkpoint_file = evolution_dir / "checkpoint_latest.json" if evolution_dir.exists() else None
            resume_from = str(checkpoint_file) if (resume and checkpoint_file and checkpoint_file.exists()) else None

            # Phase 1: Evolve architecture
            if verbose:
                print(f"\n[Phase 1/2] Evolving architecture for {problem_name}...")

            best_genome, final_population = evolve_architecture(
                problem=problem,
                population_size=population_size,
                generations=generations,
                episodes_per_eval=episodes_per_eval,
                elite_size=elite_size,
                mutation_rate=mutation_rate,
                adaptive_mutation=True,
                adaptive_population=adaptive_population,
                crossover_method='uniform',
                tournament_size=3,
                seed=seed,
                save_dir=str(evolution_dir),
                verbose=verbose,
                curriculum_schedule=curriculum_schedule,
                resume_from=resume_from
            )

            # Save evolved genome
            genome_file = problem_dir / "evolved_genome.json"
            with open(genome_file, 'w') as f:
                json.dump(best_genome.to_dict(), f, indent=2)

            if verbose:
                print(f"\nBest genome saved to: {genome_file}")
                print(f"Evolution fitness: {best_genome.fitness:.4f}")

            # Phase 2: Train final model
            if verbose:
                print(f"\n[Phase 2/2] Training final model for {problem_name}...")

            model_save_path = problem_dir / "trained_model.pth"

            agent = train_with_evolved_genome(
                problem_path=str(problem_file),
                genome=best_genome,
                episodes=training_episodes,
                save_path=str(model_save_path)
            )

            # Phase 3: Visualize best packing solution
            if verbose:
                print(f"\n[Phase 3/3] Generating visualizations for {problem_name}...")

            # Create visualization directory
            viz_dir = problem_dir / "visualizations"
            viz_dir.mkdir(exist_ok=True)

            # Reload environment to get best solution
            from packing_with_dqncore2_enhanced import MultiBinPackingEnv, load_problem_as_items
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
                print(f"  Visualizations saved to: {viz_dir}")

            # Clean up agent
            del agent
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Load and record final results
            checkpoint = torch.load(str(model_save_path), map_location='cpu', weights_only=False)

            problem_results = {
                'problem_file': str(problem_file),
                'problem_name': problem_name,
                'evolution': {
                    'generations': generations,
                    'base_population_size': population_size,
                    'adaptive_population': adaptive_population,
                    'best_fitness': best_genome.fitness,
                    'best_genome': best_genome.to_dict()
                },
                'training': {
                    'episodes': training_episodes,
                    'best_utilization': checkpoint.get('best_util', 0.0),
                    'best_bins': checkpoint.get('best_bins', 0),
                    'best_items': checkpoint.get('best_items', 0)
                },
                'architecture': {
                    'hidden_dim': best_genome.genes['hidden_dim'],
                    'enc_layers': best_genome.genes['enc_layers'],
                    'attention_type': best_genome.genes['attention_type'],
                    'complexity': best_genome.get_network_complexity()
                },
                'time_seconds': time.time() - problem_start_time
            }

            # Save problem-specific results
            result_file = problem_dir / "results.json"
            with open(result_file, 'w') as f:
                json.dump(problem_results, f, indent=2)

            # Update global state
            state['completed_problems'].append(problem_name)
            state['results'][problem_name] = problem_results
            save_experiment_state(results_path, state)

            if verbose:
                print(f"\n{'='*80}")
                print(f"COMPLETED: {problem_name}")
                print(f"{'='*80}")
                print(f"Time: {problem_results['time_seconds']:.1f}s")
                print(f"Best utilization: {problem_results['training']['best_utilization']:.3f}")
                print(f"Best bins: {problem_results['training']['best_bins']}")
                print(f"Architecture complexity: {problem_results['architecture']['complexity']:.3f}M params")
                print(f"{'='*80}\n")

        except Exception as e:
            if verbose:
                print(f"\nERROR processing {problem_name}: {e}")
                import traceback
                traceback.print_exc()

            # Save error state
            state['results'][problem_name] = {
                'error': str(e),
                'time_seconds': time.time() - problem_start_time
            }
            save_experiment_state(results_path, state)

            # Continue with next problem
            continue

    # Generate final summary
    if verbose:
        print(f"\n{'='*80}")
        print("EXPERIMENT COMPLETE")
        print(f"{'='*80}")
        print(f"Total problems: {len(problem_files)}")
        print(f"Completed: {len(state['completed_problems'])}")
        print(f"Failed: {len([r for r in state['results'].values() if 'error' in r])}")

    # Generate summary report
    summary = {
        'experiment_type': 'problem_specific',
        'configuration': {
            'base_population_size': population_size,
            'adaptive_population': adaptive_population,
            'generations': generations,
            'episodes_per_eval': episodes_per_eval,
            'training_episodes': training_episodes,
            'use_curriculum': use_curriculum,
            'curriculum_schedule': curriculum_schedule
        },
        'problems': state['results'],
        'summary_statistics': {
            'total_problems': len(problem_files),
            'completed': len(state['completed_problems']),
            'failed': len([r for r in state['results'].values() if 'error' in r]),
            'avg_utilization': np.mean([
                r['training']['best_utilization']
                for r in state['results'].values()
                if 'training' in r
            ]) if state['completed_problems'] else 0.0,
            'avg_complexity': np.mean([
                r['architecture']['complexity']
                for r in state['results'].values()
                if 'architecture' in r
            ]) if state['completed_problems'] else 0.0
        },
        'start_time': state['start_time'],
        'end_time': datetime.now().isoformat()
    }

    summary_file = results_path / "experiment_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"\nSummary saved to: {summary_file}")
        print(f"Average utilization: {summary['summary_statistics']['avg_utilization']:.3f}")
        print(f"Average complexity: {summary['summary_statistics']['avg_complexity']:.3f}M params")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run problem-specific architecture evolution experiment"
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        required=True,
        help='Directory containing problem .txt files'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results/experiment_problem_specific',
        help='Directory to save results'
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
        '--episodes-per-eval',
        type=int,
        default=20,
        help='Episodes for fitness evaluation'
    )
    parser.add_argument(
        '--training-episodes',
        type=int,
        default=300,
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
        help='Disable adaptive population sizing (use fixed population)'
    )
    parser.add_argument(
        '--elite-size',
        type=int,
        default=2,
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
        default=None,
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

    run_problem_specific_experiment(
        dataset_dir=args.dataset_dir,
        results_dir=args.results_dir,
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
