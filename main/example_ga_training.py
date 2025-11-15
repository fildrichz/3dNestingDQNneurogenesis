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
        'attention_type': 'standard',  # Changed from use_attention: True
        'attention_heads': 4,
        'num_inducing_points': 32,     # Required for set_transformer
        'patch_size': 7,
        'cnn_channels': [16, 32],      # Default CNN channels
        'dropout': 0.0,                # No dropout in baseline
        'activation': 'relu',          # ReLU activation
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
        print(f"\n Evolved architecture is BETTER!")
    elif util_improvement > -2:
        print(f"\n Evolved architecture is COMPARABLE")
    else:
        print(f"\n Baseline is better (GA may need more generations)")
    
    print(f"{'='*80}\n")


def train_with_evolved_genome(problem_path: str,
                              genome: NetworkGenome,
                              episodes: int = 200,
                              save_path: str = None):
    """
    Train a full model using the evolved genome architecture.

    Args:
        problem_path: Path to problem file
        genome: Evolved genome with best architecture
        episodes: Number of training episodes
        save_path: Path to save trained model
    """
    import collections
    from pathlib import Path

    print(f"\n{'='*80}")
    print("FULL TRAINING WITH EVOLVED ARCHITECTURE")
    print(f"{'='*80}\n")

    # Load problem
    problem = load_problem(problem_path)
    items = load_problem_as_items(problem)
    W, D, H = problem.bin_dimensions

    print("Problem Specification:")
    print(f"  Container: {W}×{D}×{H} (volume: {W*D*H:,})")
    print(f"  Max weight per bin: {problem.max_weight}")
    print(f"  Max bins available: {problem.max_bins}")
    print(f"  Items to pack: {len(items)}")
    print(f"\nEvolved Architecture:")
    print(genome)
    print(f"  Complexity: {genome.get_network_complexity():.3f}M params")
    print(f"\nTraining Configuration:")
    print(f"  Episodes: {episodes}")
    print(f"  Save path: {save_path}")
    print(f"{'='*80}\n")

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

    obs = env.reset()
    OBS_DIM = obs.shape[0]
    ACTION_FEAT_DIM = 25  # Standard action feature dimension

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build config from evolved genome
    cfg = genome.to_dqn_config(
        obs_dim=OBS_DIM,
        action_feat_dim=ACTION_FEAT_DIM,
        max_actions=128,
        device=device
    )

    # Override epsilon decay for longer training
    cfg.eps_decay_steps = episodes * 30

    # Create agent
    agent = DQNAgentEnhanced(cfg)

    print(f"Agent initialized on {device}")
    print(f"  Heightmap patches: {cfg.heightmap_patch_size}×{cfg.heightmap_patch_size}")
    print(f"  Attention: {cfg.attention_type if cfg.use_attention else 'None'}")
    print(f"Starting training...\n")

    # Training tracking
    util_hist = collections.deque(maxlen=50)
    bins_hist = collections.deque(maxlen=50)
    items_hist = collections.deque(maxlen=50)
    returns_hist = collections.deque(maxlen=50)
    best_bins = float('inf')
    best_items = 0
    best_util = 0.0

    # Training loop
    for ep in range(episodes):
        obs = env.reset(items=items.copy())
        ep_ret = 0.0
        steps = 0
        losses = []

        while True:
            # Get available actions
            action_feats, action_heightmaps, action_metadata = env.get_action_features(
                patch_size=genome.genes['patch_size']
            )

            if len(action_feats) == 0:
                break

            # Select action
            action_idx = agent.select_action(obs, action_feats, action_heightmaps)

            # Execute action
            next_obs, reward, done, info = env.step(action_idx, action_metadata)

            # Store transition
            agent.store_transition(
                obs, action_feats, action_heightmaps, action_idx,
                reward, next_obs, done
            )

            # Train
            if len(agent.memory) > agent.config.warmup_steps and steps % 1 == 0:
                loss = agent.train_step()
                if loss is not None:
                    losses.append(loss)

            ep_ret += reward
            obs = next_obs
            steps += 1

            if done:
                break

        # Episode statistics
        util_hist.append(info['utilization'])
        bins_hist.append(info['bins_used'])
        items_hist.append(info['items_packed'])
        returns_hist.append(ep_ret)

        # Track best solution
        if info['items_packed'] > best_items or \
           (info['items_packed'] == best_items and info['bins_used'] < best_bins):
            best_items = info['items_packed']
            best_bins = info['bins_used']
            best_util = info['utilization']

        # Logging
        if (ep + 1) % 10 == 0:
            avg_util = sum(util_hist) / len(util_hist) if util_hist else 0
            avg_bins = sum(bins_hist) / len(bins_hist) if bins_hist else 0
            avg_items = sum(items_hist) / len(items_hist) if items_hist else 0
            avg_return = sum(returns_hist) / len(returns_hist) if returns_hist else 0
            avg_loss = sum(losses) / len(losses) if losses else 0

            print(f"Ep {ep+1:4d}/{episodes} | "
                  f"Items: {avg_items:.1f}/{len(items)} | "
                  f"Bins: {avg_bins:.1f} | "
                  f"Util: {avg_util:.3f} | "
                  f"Return: {avg_return:+.2f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"ε: {agent.epsilon:.3f}")

    # Final results
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Best Performance:")
    print(f"  Items packed: {best_items}/{len(items)}")
    print(f"  Bins used: {best_bins}")
    print(f"  Utilization: {best_util:.3f}")
    print(f"\nFinal 50-episode average:")
    print(f"  Items packed: {sum(items_hist)/len(items_hist):.1f}/{len(items)}")
    print(f"  Bins used: {sum(bins_hist)/len(bins_hist):.1f}")
    print(f"  Utilization: {sum(util_hist)/len(util_hist):.3f}")

    # Save model
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': agent.policy_net.state_dict(),
            'genome': genome.to_dict(),
            'config': cfg.__dict__,
            'best_bins': best_bins,
            'best_items': best_items,
            'best_util': best_util
        }, save_path)
        print(f"\nModel saved to: {save_path}")

    print(f"{'='*80}\n")

    return agent


def main():
    """Main workflow: Evolve architecture → Train final model → Compare"""
    
    # Configuration
    problem_file = "3dBPP_4.txt"
    problem_path = full_datapath(problem_file)
    
    # GA parameters
    ga_config = {
        'population_size': 10,      # Small for quick testing (use 20+ for real)
        'generations': 5,            # Small for quick testing (use 10+ for real)
        'episodes_per_eval': 50,     # Episodes to train each architecture
        'elite_size': 2,             # Keep top 2 genomes
        'mutation_rate': 0.2,        # Initial mutation rate (will decay if adaptive)
        'adaptive_mutation': True,   # Enable adaptive mutation rate
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
        train_with_evolved_genome(
            problem_path=problem_path,
            genome=best_genome,
            episodes=200,  # Full training run
            save_path='output_data/evolved_model.pth'
        )


if __name__ == "__main__":
    main()