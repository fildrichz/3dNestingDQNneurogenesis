#!/usr/bin/env python3
"""
Test script for evaluating trained models on specific 3D bin packing problems.

Usage:
    python test_model.py --model-path results/experiment_multi_problem/trained_model.pth \
                         --genome-path results/experiment_multi_problem/evolved_genome.json \
                         --problem-file nesting/inputData/Thpack/Input/3dBPP_1.txt \
                         --num-episodes 10 \
                         --visualize
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import random

# Import project modules
from genome import NetworkGenome
from dqn_core.dqn_enhanced import DQNAgentEnhanced
from packing_with_dqncore2_enhanced import (
    MultiBinPackingEnv,
    build_action_features,
    load_problem_as_items
)
from nesting.dataset_loader import load_problem
from nesting.heightmap_utils import extract_patches_for_actions


def load_trained_model(model_path: str, genome_path: str, device: str = "cpu") -> Tuple[DQNAgentEnhanced, NetworkGenome]:
    """Load a trained model and its genome configuration.

    Prefer the genome stored inside the checkpoint (trained_model.pth) if present,
    otherwise fall back to an external genome JSON.

    Args:
        model_path: Path to the saved model weights (.pth file)
        genome_path: Path to the saved genome configuration (.json file), used as fallback
        device: Device to load model onto

    Returns:
        (agent, genome)
    """
    print(f"Loading model weights from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Extract state dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model_state = checkpoint["model_state_dict"]
    else:
        # Some checkpoints may save the raw state_dict directly
        model_state = checkpoint

    # Load genome: prefer checkpoint, fallback to JSON
    if isinstance(checkpoint, dict) and "genome" in checkpoint:
        genome_data = checkpoint["genome"]
        print("Loaded genome from checkpoint.")
    else:
        print(f"Loading genome from JSON: {genome_path}")
        with open(genome_path, "r") as f:
            genome_data = json.load(f)

    genome = NetworkGenome()

    # best_genome.to_dict() is expected to contain {"genes": {...}}
    if isinstance(genome_data, dict) and "genes" in genome_data:
        genome.genes = genome_data["genes"]
    elif isinstance(genome_data, dict):
        genome.genes = genome_data
    else:
        raise ValueError("Unsupported genome format in checkpoint/JSON.")

    print("Loaded genome with configuration:")
    print(f"  Hidden dim: {genome.genes.get('hidden_dim')}")
    print(f"  Encoder layers: {genome.genes.get('enc_layers')}")
    print(f"  Attention type: {genome.genes.get('attention_type')}")
    print(f"  Patch size: {genome.genes.get('patch_size')}")
    print(f"  Learning rate: {genome.genes.get('lr')}")

    cfg = genome.to_dqn_config(
        obs_dim=8,
        action_feat_dim=25,
        max_actions=128,
        device=device,
        total_episodes=1
    )

    agent = DQNAgentEnhanced(cfg)
    agent.q.load_state_dict(model_state, strict=True)
    agent.q.eval()
    agent._eps = 0.0

    print("Model loaded successfully!")
    return agent, genome


def evaluate_model_on_problem(
    agent: DQNAgentEnhanced,
    genome: NetworkGenome,
    problem_path: str,
    num_episodes: int = 10,
    visualize: bool = False,
    output_dir: str = None,
    max_steps: int = 1000,
    verbose: bool = True,
    seed: int = 42,
    topk_eps: int = 1000,
    gamma: float = 0.992
) -> Dict:
    """
    Evaluate a trained model on a specific bin packing problem.

    Args:
        agent: Trained DQN agent
        genome: Network genome configuration
        problem_path: Path to the problem file
        num_episodes: Number of evaluation episodes
        visualize: Whether to generate visualizations
        output_dir: Directory to save outputs
        max_steps: Maximum steps per episode
        verbose: Whether to print detailed information
        seed: Random seed for environment and RNGs
        topk_eps: Environment topk_eps (align with experiment_multi_problem)
        gamma: Environment gamma (align with experiment_multi_problem)

    Returns:
        Dictionary with evaluation results
    """
    # Load problem
    print(f"\nLoading problem: {problem_path}")
    problem = load_problem(problem_path)

    # Extract bin dimensions
    W, D, H = problem.bin_size
    print(f"Bin dimensions: {W} x {D} x {H}")

    # Print problem info
    print(f"Problem info:")
    print(f"  Number of item types: {len(problem.items)}")
    print(f"  Total items to pack: {sum(item.quantity for item in problem.items)}")
    print(f"  Incompatibilities: {len(problem.incompatibilities)}")
    print(f"  Positive affinities: {len(problem.positive_affinities)}")
    print(f"  Relative positioning constraints: {len(problem.relative_pos)}")

    # Prepare items list (expand quantities) - use correct format: (length, width, height, weight, item_id)
    items = load_problem_as_items(problem)

    # Get patch size from genome
    patch_size = genome.genes.get('patch_size', 7)

    # Create environment (align with experiment_multi_problem)
    env = MultiBinPackingEnv(
        W=W, D=D, H=H,
        items=items,
        max_actions=128,
        topk_eps=topk_eps,
        seed=seed,
        gamma=gamma,
        problem=problem
    )

    # Results tracking
    episode_results = []
    total_utilization = 0.0
    total_bins_used = 0
    total_items_packed = 0
    successful_episodes = 0

    # Track best episode for visualization
    best_utilization = 0.0
    best_env_state = None

    # Run evaluation episodes
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")

        # Reset environment
        obs = env.reset(items=items.copy())
        done = False
        step_count = 0

        # Force greedy
        agent._eps = 0.0

        # Episode tracking
        episode_items_placed = 0

        with torch.no_grad():
            while not done and step_count < max_steps:
                # Get feasible actions
                actions = env.enumerate_actions()
                if not actions:
                    if verbose:
                        print("No feasible actions available")
                    break

                # Build action features
                feats = build_action_features(env, actions)

                # Build patches (heightmap patches)
                patches = extract_patches_for_actions(env, actions, patch_size=patch_size)

                # Select action
                action_idx, _ = agent.select_action(obs, feats, patches=patches)
                action = actions[action_idx]

                # Take step
                obs, reward, done, info = env.step(action)
                step_count += 1

                if info.get('placed_item', False):
                    episode_items_placed += 1

                if verbose and (step_count % 10 == 0 or done):
                    print(f"Step {step_count}: reward={reward:.3f}, items_placed={episode_items_placed}, bins={len(env.bins)}")

        # Calculate episode results
        bins_used = len(env.bins)
        packed_volume = sum(
            sum(box.w * box.d * box.h for box in b.placed) for b in env.bins
        )
        utilization = packed_volume / (bins_used * W * D * H) if bins_used > 0 else 0.0

        episode_result = {
            "episode": episode + 1,
            "steps": step_count,
            "bins_used": bins_used,
            "items_placed": episode_items_placed,
            "utilization": utilization,
            "success": done
        }

        episode_results.append(episode_result)

        # Accumulate totals
        total_utilization += utilization
        total_bins_used += bins_used
        total_items_packed += episode_items_placed
        if done:
            successful_episodes += 1

        print(f"Episode {episode + 1} results:")
        print(f"  Steps: {step_count}")
        print(f"  Bins used: {bins_used}")
        print(f"  Items placed: {episode_items_placed}/{len(items)}")
        print(f"  Utilization: {utilization:.4f}")
        print(f"  Success: {done}")

        # Track best for visualization
        if utilization > best_utilization:
            best_utilization = utilization
            if visualize and output_dir:
                import copy
                best_env_state = copy.deepcopy(env)

    # Calculate averages
    avg_utilization = total_utilization / num_episodes if num_episodes else 0.0
    avg_bins_used = total_bins_used / num_episodes if num_episodes else 0.0
    avg_items_packed = total_items_packed / num_episodes if num_episodes else 0.0
    success_rate = successful_episodes / num_episodes if num_episodes else 0.0

    # Print summary
    print(f"\n=== Evaluation Summary ===")
    print(f"Problem: {Path(problem_path).name}")
    print(f"Episodes: {num_episodes}")
    print(f"Average utilization: {avg_utilization:.4f}")
    print(f"Average bins used: {avg_bins_used:.2f}")
    print(f"Average items packed: {avg_items_packed:.2f}/{len(items)}")
    print(f"Success rate: {success_rate:.2%}")
    print(f"Best utilization: {best_utilization:.4f}")

    # Generate visualizations from best episode
    if visualize and output_dir and best_env_state is not None:
        visualize_solution(best_env_state, problem_path, output_dir, total_items=len(items))

    # Return results
    results = {
        "problem_name": Path(problem_path).stem,
        "problem_file": str(problem_path),
        "bin_dimensions": [W, D, H],
        "num_item_types": len(problem.items),
        "total_items": len(items),
        "evaluation_params": {
            "num_episodes": num_episodes,
            "max_steps": max_steps,
            "seed": seed,
            "topk_eps": topk_eps,
            "gamma": gamma
        },
        "episode_results": episode_results,
        "summary": {
            "avg_utilization": avg_utilization,
            "avg_bins_used": avg_bins_used,
            "avg_items_packed": avg_items_packed,
            "success_rate": success_rate,
            "best_utilization": best_utilization
        }
    }

    return results


def visualize_solution(env: MultiBinPackingEnv, problem_path: str, output_dir: str, total_items: int = None):
    """
    Generate 3D visualizations of the packing solution.

    Args:
        env: Environment with packed bins
        problem_path: Path to problem file (for naming)
        output_dir: Directory to save visualizations
        total_items: Total items in the instance (for title)
    """
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        problem_name = Path(problem_path).stem
        viz_dir = output_path / f"visualizations_{problem_name}"
        viz_dir.mkdir(exist_ok=True)

        print(f"Generating visualizations in: {viz_dir}")

        for i, bin in enumerate(env.bins):
            if len(bin.placed) > 0:
                # Filled bin visualization
                plot_path = viz_dir / f"bin_{i+1}_filled.png"

                # Compute per-bin utilization and item count for the title
                placed_vol = sum(b.w * b.d * b.h for b in bin.placed)

                bin_vol = getattr(env, "bin_volume", None)
                if bin_vol is None:
                    W = getattr(env, "W", None)
                    D = getattr(env, "D", None)
                    H = getattr(env, "H", None)
                    bin_vol = (W * D * H) if (W is not None and D is not None and H is not None) else 0.0

                bin_util = (placed_vol / bin_vol) if bin_vol else 0.0
                n_items = len(bin.placed)
                if total_items is None:
                    extra = f"{n_items} items, util:{bin_util:.3f}"
                else:
                    extra = f"{n_items}/{total_items} items, util:{bin_util:.3f}"

                bin.plot3d_filled(save_path=str(plot_path), title=f"{problem_name} - Bin {i+1} ({extra}) - Filled")
                print(f"  Saved: {plot_path}")

                # Bin with EMS visualization
                plot_path = viz_dir / f"bin_{i+1}_with_ems.png"
                bin.plot3d(save_path=str(plot_path), title=f"{problem_name} - Bin {i+1} ({extra}) - With EMS")
                print(f"  Saved: {plot_path}")

        print("Visualizations complete!")

    except Exception as e:
        print(f"Error generating visualizations: {e}")


def main():
    """Main function for testing trained models."""
    # Default paths
    default_model_path = Path("results/experiment_multi_problem/trained_model.pth")
    default_genome_path = Path("results/experiment_multi_problem/evolved_genome.json")

    parser = argparse.ArgumentParser(
        description="Test trained 3D bin packing models on specific problems",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model/genome paths (now optional with defaults)
    parser.add_argument(
        '--model-path',
        type=str,
        default=str(default_model_path),
        help='Path to trained model weights (.pth file)'
    )
    parser.add_argument(
        '--genome-path',
        type=str,
        default=str(default_genome_path),
        help='Path to saved genome configuration (.json file)'
    )

    # Problem selection (updated for batch evaluation)
    parser.add_argument(
        '--problem-file',
        type=str,
        default=None,
        help='Path to specific problem file (overrides --problem-id)'
    )
    parser.add_argument(
        '--problem-id',
        type=int,
        default=1,
        help='Problem ID to test (used if --problem-file not provided)'
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default="nesting/inputData/Thpack/Input",
        help='Directory containing problem files'
    )

    # Evaluation parameters
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=10,
        help='Number of evaluation episodes'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=1000,
        help='Maximum steps per episode'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for environment and RNGs'
    )
    parser.add_argument(
        '--topk-eps',
        type=int,
        default=1000,
        help='Environment topk_eps (align with experiment)'
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.992,
        help='Environment gamma (align with experiment)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate 3D visualizations of the solution'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='test_results',
        help='Directory to save results and visualizations'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda'],
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run model on'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed step-by-step information'
    )
    parser.add_argument(
        '--save-results',
        action='store_true',
        help='Save results to JSON file'
    )

    args = parser.parse_args()

    # Set RNG seeds for reproducibility (align with experiment)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Determine problem file
    if args.problem_file is None:
        # Use problem ID to construct path
        problem_file = Path(args.dataset_dir) / f"3dBPP_{args.problem_id}.txt"
        print(f"No problem file provided, using problem ID {args.problem_id}: {problem_file}")
    else:
        problem_file = Path(args.problem_file)

    # Validate paths
    if not Path(args.model_path).exists():
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)

    if not Path(args.genome_path).exists():
        print(f"Error: Genome file not found: {args.genome_path}")
        sys.exit(1)

    if not problem_file.exists():
        print(f"Error: Problem file not found: {problem_file}")
        sys.exit(1)

    # Create output directory
    if args.visualize or args.save_results:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    # Load trained model
    agent, genome = load_trained_model(
        model_path=args.model_path,
        genome_path=args.genome_path,
        device=args.device
    )

    # Evaluate model
    results = evaluate_model_on_problem(
        agent=agent,
        genome=genome,
        problem_path=str(problem_file),
        num_episodes=args.num_episodes,
        visualize=args.visualize,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        verbose=args.verbose,
        seed=args.seed,
        topk_eps=args.topk_eps,
        gamma=args.gamma
    )

    # Save results if requested
    if args.save_results:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results_file = output_path / f"test_results_{results['problem_name']}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {results_file}")

    print("\nTest complete!")


if __name__ == '__main__':
    main()
