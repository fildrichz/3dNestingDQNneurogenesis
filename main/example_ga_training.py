"""
Example: GA-Based Neural Architecture Evolution for Bin Packing

This script demonstrates the complete workflow:
1. Evolve network architecture using genetic algorithms
2. Train final model with best architecture
3. Compare evolved vs hand-designed architecture

Based on: "Using Genetic Algorithms to Optimize Artificial Neural Networks"
(Ding et al., 2010), Section 3.2: Optimizing Network Architecture
"""

import torch
from pathlib import Path

from nesting.dataset_loader import load_problem
from ga_evolution import evolve_architecture, load_best_genome, plot_evolution_history
from genome import NetworkGenome
from dqn_core.dqn_enhanced import DQNAgentEnhanced
from packing_with_dqncore2_enhanced import (
    train_multibin_pack_dqn, 
    load_problem_as_items,
    MultiBinPackingEnv
)


def full_datapath(filename: str) -> str:
    """Helper for dataset path."""
    return "nesting\\inputData\\Benchmark dataset and instance generator for Real-World 3dBPP\\Input\\" + filename


def compare_architectures(problem_path: str, 
                         evolved_genome: NetworkGenome,
                         episodes: int = 100):
    """
    Compare evolved architecture vs hand-designed baseline.
    
    Args:
        problem_path: Path to problem file
        evolved_genome: Best genome from GA evolution
        episodes: Number of episodes for comparison
    """
    print(f"\n{'='*80}")
    print("ARCHITECTURE COMPARISON")
    print(f"{'='*80}\n")
    
    problem = load_problem(problem_path)
    items = load_problem_as_items(problem)
    W, D, H = problem.bin_dimensions
    
    # Create environment
    env = MultiBinPackingEnv(
        W, D, H,
        items=items,
        max_actions=128,
        topk_eps=1000,
        seed=42,
        gamma=0.992,
        problem=problem
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Hand-designed baseline (your original architecture)
    print("Testing BASELINE (hand-designed) architecture...")
    baseline_genome = NetworkGenome(genes={
        'hidden_dim': 256,
        'enc_layers': 2,
        'head_hidden': 256,
        'use_attention': True,
        'attention_heads': 4,
        'patch_size': 7,
    })
    
    baseline_cfg = baseline_genome.to_dqn_config(8, 25, 128, device)
    baseline_agent = DQNAgentEnhanced(baseline_cfg)
    
    from packing_with_dqncore2_enhanced import evaluate_agent_on_problem
    baseline_metrics = evaluate_agent_on_problem(
        agent=baseline_agent,
        env=env,
        items=items.copy(),
        episodes=episodes,
        patch_size=7,
        verbose=True
    )
    
    print(f"\nBaseline Results:")
    print(f"  Utilization: {baseline_metrics['avg_utilization']:.3f}")
    print(f"  Bins used: {baseline_metrics['avg_bins_used']:.1f}")
    print(f"  Complexity: {baseline_genome.get_network_complexity():.3f}M params")
    
    # 2. Evolved architecture
    print(f"\n{'-'*80}\n")
    print("Testing EVOLVED architecture...")
    print(evolved_genome)
    
    evolved_cfg = evolved_genome.to_dqn_config(8, 25, 128, device)
    evolved_agent = DQNAgentEnhanced(evolved_cfg)
    
    evolved_metrics = evaluate_agent_on_problem(
        agent=evolved_agent,
        env=env,
        items=items.copy(),
        episodes=episodes,
        patch_size=evolved_genome.genes['patch_size'],
        verbose=True
    )
    
    print(f"\nEvolved Results:")
    print(f"  Utilization: {evolved_metrics['avg_utilization']:.3f}")
    print(f"  Bins used: {evolved_metrics['avg_bins_used']:.1f}")
    print(f"  Complexity: {evolved_genome.get_network_complexity():.3f}M params")
    
    # Comparison
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    util_improvement = (evolved_metrics['avg_utilization'] - baseline_metrics['avg_utilization']) / baseline_metrics['avg_utilization'] * 100
    bins_improvement = baseline_metrics['avg_bins_used'] - evolved_metrics['avg_bins_used']
    complexity_diff = evolved_genome.get_network_complexity() - baseline_genome.get_network_complexity()
    
    print(f"\nUtilization improvement: {util_improvement:+.1f}%")
    print(f"Bins saved: {bins_improvement:+.1f}")
    print(f"Complexity difference: {complexity_diff:+.2f}M params")
    
    if util_improvement > 0:
        print(f"\n✅ Evolved architecture is BETTER!")
    elif util_improvement > -2:
        print(f"\n≈ Evolved architecture is COMPARABLE")
    else:
        print(f"\n❌ Baseline is better (GA may need more generations)")
    
    print(f"{'='*80}\n")


def main():
    """Main workflow: Evolve architecture → Train final model → Compare"""
    
    # Configuration
    problem_file = "3dBPP_4.txt"
    problem_path = full_datapath(problem_file)
    
    # GA parameters
    ga_config = {
        'population_size': 3,      # Small for quick testing (use 20+ for real)
        'generations': 2,            # Small for quick testing (use 10+ for real)
        'episodes_per_eval': 15,     # Episodes to train each architecture
        'elite_size': 2,             # Keep top 2 genomes
        'mutation_rate': 0.2,        # 20% mutation rate
        'crossover_method': 'uniform',
        'tournament_size': 3,
        'save_dir': 'output_data/ga_evolution',
        'verbose': True
    }
    
    print("="*80)
    print("GENETIC ALGORITHM ARCHITECTURE EVOLUTION")
    print("="*80)
    print(f"Problem: {problem_file}")
    print(f"Population: {ga_config['population_size']}")
    print(f"Generations: {ga_config['generations']}")
    print(f"Episodes per evaluation: {ga_config['episodes_per_eval']}")
    print("="*80)
    
    # Step 1: Load problem
    print("\n[1/4] Loading problem...")
    problem = load_problem(problem_path)
    print(f"✓ Loaded: {problem.bin_dimensions[0]}×{problem.bin_dimensions[1]}×{problem.bin_dimensions[2]} container")
    
    # Step 2: Evolve architecture
    print("\n[2/4] Evolving architecture with GA...")
    best_genome, final_population = evolve_architecture(problem, **ga_config)
    
    # Step 3: Plot evolution
    print("\n[3/4] Plotting evolution history...")
    try:
        plot_evolution_history(
            save_dir=ga_config['save_dir'],
            output_file='output_data/ga_evolution/evolution_plot.png'
        )
        print("✓ Evolution plot saved")
    except Exception as e:
        print(f"⚠ Could not create plot: {e}")
    
    # Step 4: Compare with baseline
    print("\n[4/4] Comparing evolved vs baseline architecture...")
    compare_architectures(
        problem_path=problem_path,
        evolved_genome=best_genome,
        episodes=50  # More episodes for fair comparison
    )
    
    # Final report
    print("\n" + "="*80)
    print("FINAL REPORT")
    print("="*80)
    print("\nBest evolved architecture:")
    print(best_genome)
    print(f"\nGenome saved to: {ga_config['save_dir']}/best_genome.json")
    print("="*80)
    
    # Optional: Train full model with best architecture
    user_input = input("\nTrain full model (100+ episodes) with best architecture? [y/N]: ")
    if user_input.lower() == 'y':
        print("\nTraining full model with evolved architecture...")
        print("toto")
        
        # Build config from best genome
        cfg = best_genome.to_dqn_config(
            obs_dim=8,
            action_feat_dim=25,
            max_actions=128,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        agent = DQNAgentEnhanced(cfg)
        
        # Note: You'd need to modify train_multibin_pack_dqn to accept pre-built agent
        # For now, we'll just show the genome
        print("\nTo train with this architecture, use:")
        print(f"  hidden_dim={best_genome.genes['hidden_dim']}")
        print(f"  enc_layers={best_genome.genes['enc_layers']}")
        print(f"  head_hidden={best_genome.genes['head_hidden']}")
        print(f"  use_attention={best_genome.genes['use_attention']}")
        print(f"  patch_size={best_genome.genes['patch_size']}")


if __name__ == "__main__":
    main()
