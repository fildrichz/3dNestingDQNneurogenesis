"""
Example: Neuroevolution for 3D Bin Packing
===========================================

Demonstrates GA-based architecture search for DQN agents.

Usage:
    python train_neuroevolution.py
"""

from neuroevolution_dqn import NeuroEvolutionDQN, ArchitectureGene
from packing_with_dqncore2 import (
    MultiBinPackingEnv, 
    build_action_features,
    pad_feats_mask,
    load_problem_as_items,
    ACTION_FEAT_DIM
)
from nesting.dataset_loader import load_problem
from dqn_core.dqn_base import DQNConfig


def full_datapath(filename: str) -> str:
    """Helper for dataset path"""
    return "nesting\\inputData\\Benchmark dataset and instance generator for Real-World 3dBPP\\Input\\" + filename


def train_with_neuroevolution(
    problem_path: str,
    num_generations: int = 5,
    population_size: int = 8,
    episodes_per_gen: int = 50,
    seed: int = 42
):
    """
    Train using neuroevolution approach:
    
    1. Initialize population of N different architectures
    2. For each generation:
       a. Train each architecture for M episodes
       b. Evaluate performance (fitness)
       c. Select best architectures (elitism)
       d. Create offspring via crossover + mutation
    3. Return best architecture found
    """
    
    print(f"\n{'='*80}")
    print(f"NEUROEVOLUTION FOR 3D BIN PACKING")
    print(f"{'='*80}")
    print(f"Problem: {problem_path}")
    print(f"Generations: {num_generations}")
    print(f"Population size: {population_size}")
    print(f"Episodes per generation: {episodes_per_gen}")
    print(f"{'='*80}\n")
    
    # Load problem
    problem = load_problem(problem_path)
    items = load_problem_as_items(problem)
    W, D, H = problem.bin_dimensions
    
    print(f" Problem Details:")
    print(f"   Container: {W}×{D}×{H}")
    print(f"   Max bins: {problem.max_bins}")
    print(f"   Max weight: {problem.max_weight}")
    print(f"   Items: {len(items)}")
    print(f"   Total volume: {sum(i[0]*i[1]*i[2] for i in items):,}")
    
    # Create environment
    max_actions = 128
    env = MultiBinPackingEnv(
        W, D, H,
        items=items,
        max_actions=max_actions,
        topk_eps=1000,
        seed=seed,
        gamma=0.992,
        problem=problem
    )
    obs = env.reset()
    OBS_DIM = obs.shape[0]
    
    # Base DQN config (will be modified by evolution)
    base_config = DQNConfig(
        obs_dim=OBS_DIM,
        action_feat_dim=ACTION_FEAT_DIM,
        max_actions=max_actions,
        device="cuda" if __import__("torch").cuda.is_available() else "cpu",
        
        # These will be evolved:
        hidden=256,
        enc_layers=2,
        head_hidden=256,
        lr=1e-4,
        gamma=0.992,
        n_step=3,
        batch_size=128,
        
        # Fixed parameters
        buffer_size=400_000,
        eps_start=1.0,
        eps_end=0.15,
        eps_decay_steps=episodes_per_gen * num_generations * 30,
        target_update_interval=500,
        double_dqn=True,
        warmup_steps=1500,
    )
    
    # Initialize neuroevolution
    neuro_evo = NeuroEvolutionDQN(
        base_config=base_config,
        population_size=population_size,
        elite_size=2,  # Keep top 2
        mutation_rate=0.2,
        crossover_rate=0.7
    )
    
    # Initialize population
    neuro_evo.initialize_population()
    
    # Evolution loop
    print(f"\n Starting evolution for {num_generations} generations...")
    
    for gen in range(num_generations):
        stats = neuro_evo.evolve_generation(
            env=env,
            items=items,
            episodes_per_individual=episodes_per_gen,
            build_features_fn=build_action_features,
            pad_feats_mask_fn=pad_feats_mask,
            max_actions=max_actions,
            action_feat_dim=ACTION_FEAT_DIM,
            train_freq=1
        )
        
        # Save best after each generation
        neuro_evo.save_best(f"output_data/neuro_best_gen{gen+1}.pth")
    
    # Final results
    print(f"\n{'='*80}")
    print(f"EVOLUTION COMPLETE!")
    print(f"{'='*80}")
    
    best = neuro_evo.best_individual
    print(f"\n BEST ARCHITECTURE FOUND:")
    print(f"   Fitness: {best.gene.fitness:.4f}")
    print(f"   Utilization: {best.avg_utilization:.3f}")
    print(f"   Bins used: {best.avg_bins_used:.2f}")
    print(f"   Items placed: {best.avg_items_placed:.1%}")
    print(f"\n   Architecture:")
    print(f"   - Hidden dim: {best.gene.hidden_dim}")
    print(f"   - Encoder layers: {best.gene.enc_layers}")
    print(f"   - Head hidden: {best.gene.head_hidden}")
    print(f"   - Learning rate: {best.gene.lr:.0e}")
    print(f"   - Gamma: {best.gene.gamma}")
    print(f"   - N-step: {best.gene.n_step}")
    print(f"   - Batch size: {best.gene.batch_size}")
    print(f"   - Model complexity: ~{best.gene.complexity():,} parameters")
    
    # Save final best
    neuro_evo.save_best("output_data/neuro_best_final.pth")
    
    # Plot evolution
    neuro_evo.plot_evolution("output_data/evolution_progress.png")
    
    print(f"\n{'='*80}\n")
    
    return neuro_evo, best


def compare_architectures(problem_path: str, episodes: int = 100):
    """
    Quick comparison: manually designed vs. evolved architecture.
    
    This demonstrates the value of neuroevolution!
    """
    
    print(f"\n{'='*80}")
    print(f"ARCHITECTURE COMPARISON")
    print(f"{'='*80}\n")
    
    # Load problem
    problem = load_problem(problem_path)
    items = load_problem_as_items(problem)
    W, D, H = problem.bin_dimensions
    
    # Create environment
    env = MultiBinPackingEnv(
        W, D, H, items=items, max_actions=128, 
        topk_eps=1000, seed=42, gamma=0.992, problem=problem
    )
    obs = env.reset()
    OBS_DIM = obs.shape[0]
    
    # Architecture 1: Manual design (current default)
    from dqn_core.dqn_base import DQNAgent
    config1 = DQNConfig(
        obs_dim=OBS_DIM, action_feat_dim=ACTION_FEAT_DIM,
        max_actions=128, hidden=256, enc_layers=2, head_hidden=256,
        lr=1e-4, gamma=0.992, device="cpu"
    )
    agent1 = DQNAgent(config1)
    
    # Architecture 2: Hypothetical evolved architecture
    # (In practice, we will load this from neuroevolution results)
    config2 = DQNConfig(
        obs_dim=OBS_DIM, action_feat_dim=ACTION_FEAT_DIM,
        max_actions=128, hidden=128, enc_layers=3, head_hidden=512,
        lr=2e-4, gamma=0.985, device="cpu"
    )
    agent2 = DQNAgent(config2)
    
    # Train and compare
    # (Training code would go here - similar to evaluate_individual)
    
    print("Architecture 1 (Manual):")
    print(f"  Hidden: {config1.hidden}, Layers: {config1.enc_layers}, Head: {config1.head_hidden}")
    print(f"  Params: ~{config1.hidden**2 * config1.enc_layers:,}")
    
    print("\nArchitecture 2 (Evolved):")
    print(f"  Hidden: {config2.hidden}, Layers: {config2.enc_layers}, Head: {config2.head_hidden}")
    print(f"  Params: ~{config2.hidden**2 * config2.enc_layers:,}")
    
    print("\n(Full comparison requires training both agents)")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import os
    os.makedirs("output_data", exist_ok=True)
    
    # Example 1: Small-scale neuroevolution
    print(" EXAMPLE 1: Small-scale neuroevolution")
    problem = "3dBPP_4.txt"
    
    neuro_evo, best = train_with_neuroevolution(
        problem_path=full_datapath(problem),
        num_generations=2,      # Quick demo - use 10-20 for real runs
        population_size=4,      # Small population - use 10-20 for real runs
        episodes_per_gen=30,    # Short training - use 50-100 for real runs
        seed=42
    )
    
    # Example 2: Architecture comparison
    print("\n🔬 EXAMPLE 2: Architecture comparison")
    compare_architectures(full_datapath(problem), episodes=50)
    
    print("\n All examples complete!")
    print("\nTo run full neuroevolution:")
    print("  - Increase num_generations to 10-20")
    print("  - Increase population_size to 10-20")
    print("  - Increase episodes_per_gen to 50-100")
    print("  - Consider using GPU (device='cuda')")
    print("\nExpected runtime:")
    print("  - Small scale (above): ~30 minutes")
    print("  - Full scale: ~5-10 hours on GPU")
