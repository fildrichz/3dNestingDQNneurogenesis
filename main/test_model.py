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
    """
    Load a trained model and its genome configuration.

    Args:
        model_path: Path to the saved model weights (.pth file)
        genome_path: Path to the saved genome configuration (.json file)
        device: Device to load the model on ('cpu' or 'cuda')

    Returns:
        Tuple of (agent, genome)
    """
    print(f"Loading genome from: {genome_path}")
    with open(genome_path, 'r') as f:
        genome_data = json.load(f)

    # Reconstruct genome
    genome = NetworkGenome()
    genome.genes = genome_data['genes']

    print(f"Loaded genome with configuration:")
    print(f"  Hidden dim: {genome.genes['hidden_dim']}")
    print(f"  Encoder layers: {genome.genes['enc_layers']}")
    print(f"  Attention type: {genome.genes['attention_type']}")
    print(f"  Patch size: {genome.genes['patch_size']}")
    print(f"  Learning rate: {genome.genes['lr']}")

    # Create DQN configuration
    cfg = genome.to_dqn_config(
        obs_dim=8,              # State features
        action_feat_dim=25,     # Action features
        max_actions=128,        # Max actions to consider
        device=device,
        total_episodes=1        # Not training, so minimal epsilon decay
    )

    # Create agent
    agent = DQNAgentEnhanced(cfg)

    # Load trained weights
    print(f"Loading model weights from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Extract model state dict from checkpoint (checkpoint contains multiple keys)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
    else:
        model_state = checkpoint

    agent.q.load_state_dict(model_state)
    agent.q.eval()  # Set to evaluation mode

    # Disable exploration (use greedy policy)
    agent._eps = 0.0  # Set internal epsilon to 0 for greedy policy

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
    verbose: bool = True
) -> Dict:
    """
    Evaluate a trained model on a specific problem.

    Args:
        agent: Trained DQN agent
        genome: Genome configuration
        problem_path: Path to the problem file
        num_episodes: Number of evaluation episodes
        visualize: Whether to generate 3D visualizations
        output_dir: Directory to save visualizations
        max_steps: Maximum steps per episode
        verbose: Print detailed episode information

    Returns:
        Dictionary with evaluation results
    """
    print(f"\n{'='*80}")
    print(f"Evaluating on problem: {problem_path}")
    print(f"{'='*80}\n")

    # Load problem
    problem = load_problem(problem_path)
    W, D, H = problem.bin_dimensions

    print(f"Problem details:")
    print(f"  Bin dimensions: {W} x {D} x {H}")
    print(f"  Max bins: {problem.max_bins}")
    print(f"  Max weight: {problem.max_weight}")
    print(f"  Number of items: {len(problem.items)}")
    print(f"  Total items to pack: {sum(item.quantity for item in problem.items)}")
    print(f"  Incompatibilities: {len(problem.incompatibilities)}")
    print(f"  Positive affinities: {len(problem.positive_affinities)}")
    print(f"  Relative positioning constraints: {len(problem.relative_pos)}")

    # Prepare items list (expand quantities) - use correct format: (length, width, height, weight, item_id)
    items = load_problem_as_items(problem)

    # Get patch size from genome
    patch_size = genome.genes.get('patch_size', 7)

    # Create environment
    env = MultiBinPackingEnv(
        W=W, D=D, H=H,
        items=items,
        max_actions=128,
        problem=problem
    )

    # Results tracking
    episode_results = []
    total_utilization = 0.0
    total_bins_used = 0
    total_items_packed = 0
    successful_episodes = 0

    # Run evaluation episodes
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")

        # Reset environment
        obs = env.reset(items=items.copy())
        done = False
        step_count = 0
        episode_reward = 0.0

        while not done and step_count < max_steps:
            # Get feasible actions
            actions, mask = env.action_space()

            if len(actions) == 0:
                if verbose:
                    print(f"  Step {step_count}: No feasible actions remaining")
                break

            # Build action features (25-dimensional)
            action_features = build_action_features(env, actions)

            # Extract heightmap patches for CNN
            patches = extract_patches_for_actions(env, actions, patch_size=patch_size)

            # Agent selects action (greedy, no exploration)
            action_idx = agent.select_action(obs, action_features, patches, mask)

            # Execute action
            next_obs, reward, done, info = env.step(actions[action_idx])

            episode_reward += reward
            step_count += 1

            if verbose and step_count % 10 == 0:
                items_left = sum(1 for item in env.items if item is not None)
                print(f"  Step {step_count}: Items remaining: {items_left}, Reward: {reward:.4f}")

            obs = next_obs

        # Calculate episode metrics
        bins_used = sum(1 for bin in env.bins if len(bin.placed) > 0)
        items_packed = sum(len(bin.placed) for bin in env.bins)
        items_left = sum(1 for item in env.items if item is not None)

        # Calculate utilization
        # Items are tuples: (length, width, height, weight, item_id)
        total_item_volume = sum(
            item[0] * item[1] * item[2]  # length * width * height
            for item in items
        )
        packed_volume = sum(
            box.w * box.d * box.h
            for bin in env.bins
            for box in bin.placed
        )
        utilization = packed_volume / (bins_used * W * D * H) if bins_used > 0 else 0.0

        all_packed = items_left == 0
        if all_packed:
            successful_episodes += 1

        episode_result = {
            'episode': episode + 1,
            'bins_used': bins_used,
            'items_packed': items_packed,
            'items_left': items_left,
            'utilization': utilization,
            'total_reward': episode_reward,
            'steps': step_count,
            'all_packed': all_packed
        }
        episode_results.append(episode_result)

        total_utilization += utilization
        total_bins_used += bins_used
        total_items_packed += items_packed

        print(f"  Results:")
        print(f"    Bins used: {bins_used}")
        print(f"    Items packed: {items_packed}/{len(items)}")
        print(f"    Utilization: {utilization:.4f}")
        print(f"    Total reward: {episode_reward:.4f}")
        print(f"    All items packed: {'YES' if all_packed else 'NO'}")

        # Generate visualization for first episode if requested
        if visualize and episode == 0 and output_dir:
            visualize_solution(env, problem_path, output_dir)

    # Calculate averages
    avg_utilization = total_utilization / num_episodes
    avg_bins_used = total_bins_used / num_episodes
    avg_items_packed = total_items_packed / num_episodes
    success_rate = successful_episodes / num_episodes

    results = {
        'problem_path': problem_path,
        'problem_name': Path(problem_path).stem,
        'num_episodes': num_episodes,
        'avg_utilization': avg_utilization,
        'avg_bins_used': avg_bins_used,
        'avg_items_packed': avg_items_packed,
        'total_items': len(items),
        'success_rate': success_rate,
        'episodes': episode_results
    }

    print(f"\n{'='*80}")
    print(f"SUMMARY RESULTS")
    print(f"{'='*80}")
    print(f"Average utilization: {avg_utilization:.4f}")
    print(f"Average bins used: {avg_bins_used:.2f}")
    print(f"Average items packed: {avg_items_packed:.2f}/{len(items)}")
    print(f"Success rate: {success_rate:.2%} ({successful_episodes}/{num_episodes})")
    print(f"{'='*80}\n")

    return results


def visualize_solution(env: MultiBinPackingEnv, problem_path: str, output_dir: str):
    """
    Generate 3D visualizations of the packing solution.

    Args:
        env: Environment with packed bins
        problem_path: Path to problem file (for naming)
        output_dir: Directory to save visualizations
    """
    try:
        from nesting.packing_core_enhanced import plot_bin_3d

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        problem_name = Path(problem_path).stem
        viz_dir = output_path / f"visualizations_{problem_name}"
        viz_dir.mkdir(exist_ok=True)

        print(f"\nGenerating visualizations in: {viz_dir}")

        for i, bin in enumerate(env.bins):
            if len(bin.placed) > 0:
                # Filled bin
                plot_path = viz_dir / f"bin_{i+1}_filled.png"
                plot_bin_3d(bin, show_ems=False, title=f"Bin {i+1} - Filled", save_path=str(plot_path))
                print(f"  Saved: {plot_path}")

                # Bin with EMS
                plot_path = viz_dir / f"bin_{i+1}_with_ems.png"
                plot_bin_3d(bin, show_ems=True, title=f"Bin {i+1} - With EMS", save_path=str(plot_path))
                print(f"  Saved: {plot_path}")

        print("Visualizations complete!")

    except ImportError as e:
        print(f"Warning: Could not generate visualizations - {e}")
        print("Make sure matplotlib and required dependencies are installed.")


def main():
    # Get script directory for relative paths
    script_dir = Path(__file__).parent

    # Default paths (relative to script directory)
    default_dataset_dir = script_dir / "nesting/inputData/Benchmark dataset and instance generator for Real-World 3dBPP/Input"
    default_model_path = script_dir / "trainedModels/trained_model.pth"
    default_genome_path = script_dir / "trainedModels/evolved_genome.json"

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
        help='Path to evolved genome configuration (.json file)'
    )

    # Problem selection (can use either --problem-id or --problem-file)
    parser.add_argument(
        '--problem-id',
        type=int,
        default=1,
        help='Problem ID to test (e.g., 1 for 3dBPP_1.txt). Ignored if --problem-file is specified.'
    )
    parser.add_argument(
        '--problem-file',
        type=str,
        default=None,
        help='Path to specific problem file (overrides --problem-id)'
    )

    # Optional arguments
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

    # Determine problem file path
    if args.problem_file is None:
        # Use problem ID to construct path
        problem_file = default_dataset_dir / f"3dBPP_{args.problem_id}.txt"
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

    # Load model
    agent, genome = load_trained_model(
        model_path=args.model_path,
        genome_path=args.genome_path,
        device=args.device
    )

    # Evaluate on problem
    results = evaluate_model_on_problem(
        agent=agent,
        genome=genome,
        problem_path=str(problem_file),
        num_episodes=args.num_episodes,
        visualize=args.visualize,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        verbose=args.verbose
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
