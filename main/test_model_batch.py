#!/usr/bin/env python3
"""
Batch testing script for evaluating trained models on multiple problems.

Usage:
    python test_model_batch.py --model-path results/experiment_multi_problem/trained_model.pth \
                                --genome-path results/experiment_multi_problem/evolved_genome.json \
                                --dataset-dir nesting/inputData/Thpack/Input \
                                --problem-ids 1 2 3 4 5 \
                                --num-episodes 10
"""

import argparse
import json
import pandas as pd
from pathlib import Path
import sys
from typing import List

from test_model import load_trained_model, evaluate_model_on_problem


def get_problem_files(dataset_dir: str, problem_ids: List[int]) -> List[Path]:
    """
    Get problem file paths from dataset directory.

    Args:
        dataset_dir: Directory containing problem files
        problem_ids: List of problem IDs

    Returns:
        List of problem file paths
    """
    dataset_path = Path(dataset_dir)
    problem_files = []

    for pid in problem_ids:
        problem_file = dataset_path / f"3dBPP_{pid}.txt"
        if problem_file.exists():
            problem_files.append(problem_file)
        else:
            print(f"Warning: Problem file not found: {problem_file}")

    return problem_files


def main():
    # Get script directory for relative paths
    script_dir = Path(__file__).parent

    # Default paths (relative to script directory)
    default_dataset_dir = script_dir / "nesting/inputData/Benchmark dataset and instance generator for Real-World 3dBPP/Input"
    default_model_path = script_dir / "trainedModels/trained_model.pth"
    default_genome_path = script_dir / "trainedModels/evolved_genome.json"

    parser = argparse.ArgumentParser(
        description="Batch test trained models on multiple 3D bin packing problems",
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
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default=str(default_dataset_dir),
        help='Directory containing problem files'
    )
    parser.add_argument(
        '--problem-ids',
        type=int,
        nargs='+',
        default=[1, 2, 3, 4, 5],
        help='List of problem IDs to test (e.g., 1 2 3 4 5)'
    )

    # Optional arguments
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=10,
        help='Number of evaluation episodes per problem'
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
        help='Generate 3D visualizations (only for first problem)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='test_results_batch',
        help='Directory to save results'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda'],
        default='auto',
        help='Device to run model on'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed information'
    )

    args = parser.parse_args()

    # Auto-detect device
    if args.device == 'auto':
        import torch
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Validate paths
    if not Path(args.model_path).exists():
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)

    if not Path(args.genome_path).exists():
        print(f"Error: Genome file not found: {args.genome_path}")
        sys.exit(1)

    if not Path(args.dataset_dir).exists():
        print(f"Error: Dataset directory not found: {args.dataset_dir}")
        sys.exit(1)

    # Get problem files
    problem_files = get_problem_files(args.dataset_dir, args.problem_ids)

    if len(problem_files) == 0:
        print("Error: No valid problem files found")
        sys.exit(1)

    print(f"Found {len(problem_files)} problem files to test")

    # Load model once
    agent, genome = load_trained_model(
        model_path=args.model_path,
        genome_path=args.genome_path,
        device=args.device
    )

    # Test on each problem
    all_results = []

    for i, problem_file in enumerate(problem_files):
        print(f"\n{'#'*80}")
        print(f"Testing problem {i+1}/{len(problem_files)}: {problem_file.name}")
        print(f"{'#'*80}")

        # Visualize only first problem
        visualize = args.visualize and i == 0

        results = evaluate_model_on_problem(
            agent=agent,
            genome=genome,
            problem_path=str(problem_file),
            num_episodes=args.num_episodes,
            visualize=visualize,
            output_dir=args.output_dir,
            max_steps=args.max_steps,
            verbose=args.verbose
        )

        all_results.append(results)

    # Create summary
    print(f"\n{'='*80}")
    print("BATCH TEST SUMMARY")
    print(f"{'='*80}\n")

    summary_data = []
    for result in all_results:
        summary_data.append({
            'Problem': result['problem_name'],
            'Avg Utilization': f"{result['avg_utilization']:.4f}",
            'Avg Bins Used': f"{result['avg_bins_used']:.2f}",
            'Avg Items Packed': f"{result['avg_items_packed']:.1f}/{result['total_items']}",
            'Success Rate': f"{result['success_rate']:.2%}"
        })

    df = pd.DataFrame(summary_data)
    print(df.to_string(index=False))

    # Calculate overall statistics
    overall_utilization = sum(r['avg_utilization'] for r in all_results) / len(all_results)
    overall_bins = sum(r['avg_bins_used'] for r in all_results) / len(all_results)
    overall_success = sum(r['success_rate'] for r in all_results) / len(all_results)

    print(f"\n{'='*80}")
    print(f"Overall Average Utilization: {overall_utilization:.4f}")
    print(f"Overall Average Bins Used: {overall_bins:.2f}")
    print(f"Overall Success Rate: {overall_success:.2%}")
    print(f"{'='*80}\n")

    # Save results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    results_file = output_path / "batch_test_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'model_path': args.model_path,
            'genome_path': args.genome_path,
            'dataset_dir': args.dataset_dir,
            'problem_ids': args.problem_ids,
            'num_episodes': args.num_episodes,
            'overall_avg_utilization': overall_utilization,
            'overall_avg_bins_used': overall_bins,
            'overall_success_rate': overall_success,
            'results': all_results
        }, f, indent=2)

    print(f"Detailed results saved to: {results_file}")

    # Save summary CSV
    csv_file = output_path / "batch_test_summary.csv"
    df.to_csv(csv_file, index=False)
    print(f"Summary CSV saved to: {csv_file}")

    print("\nBatch test complete!")


if __name__ == '__main__':
    main()
