"""
Hyperparameter Configuration Guide for DQN 3D Bin Packing
=========================================================

This guide provides recommended hyperparameter configurations for different
scenarios and container sizes.
"""

# =============================================================================
# SCENARIO 1: Small Container, Quick Training (20x20x20, 50 items)
# =============================================================================
SMALL_FAST = {
    "episodes": 100,
    "W": 20, "D": 20, "H": 20,
    "n_items": 50,
    "max_actions": 50,
    "topk_eps": 100,
    "train_freq": 1,
    "num_train_steps": 1,
    
    "dqn_config": {
        "gamma": 0.992,
        "lr": 3e-4,              # Higher LR for faster learning
        "batch_size": 64,
        "buffer_size": 50_000,
        "eps_start": 1.0,
        "eps_end": 0.05,
        "eps_decay_steps": 1500,  # Decay over ~15 episodes
        "target_update_interval": 300,
        "n_step": 3,
        "warmup_steps": 500,
    },
    
    "description": "Fast training for small problems. Good for debugging."
}

# =============================================================================
# SCENARIO 2: Medium Container, Balanced (40x40x40, 100 items)
# =============================================================================
MEDIUM_BALANCED = {
    "episodes": 300,
    "W": 40, "D": 40, "H": 40,
    "n_items": 100,
    "max_actions": 128,
    "topk_eps": 200,
    "train_freq": 1,
    "num_train_steps": 2,     # 2 updates per step
    
    "dqn_config": {
        "gamma": 0.995,
        "lr": 2e-4,
        "batch_size": 128,
        "buffer_size": 150_000,
        "eps_start": 1.0,
        "eps_end": 0.05,
        "eps_decay_steps": 6000,  # Decay over 20 episodes
        "target_update_interval": 500,
        "n_step": 3,
        "warmup_steps": 1000,
    },
    
    "description": "Balanced approach for medium-sized problems."
}

# =============================================================================
# SCENARIO 3: Large Container, High Quality (60x60x60, 200 items)
# =============================================================================
LARGE_QUALITY = {
    "episodes": 500,
    "W": 60, "D": 60, "H": 60,
    "n_items": 200,
    "max_actions": 200,
    "topk_eps": 300,
    "train_freq": 1,
    "num_train_steps": 4,     # 4 updates per step
    
    "dqn_config": {
        "gamma": 0.997,           # Higher gamma for longer horizons
        "lr": 1e-4,               # Lower LR for stability
        "batch_size": 256,
        "buffer_size": 300_000,
        "eps_start": 1.0,
        "eps_end": 0.02,          # Lower final epsilon
        "eps_decay_steps": 10000,
        "target_update_interval": 1000,
        "n_step": 5,              # Longer n-step
        "warmup_steps": 2000,
    },
    
    "description": "High-quality solutions for large problems. Slower training."
}

# =============================================================================
# SCENARIO 4: Fixed Items Overfitting (for analysis)
# =============================================================================
FIXED_OVERFIT = {
    "episodes": 200,
    "W": 30, "D": 30, "H": 30,
    "n_items": 80,
    "max_actions": 100,
    "topk_eps": 150,
    "train_freq": 1,
    "num_train_steps": 2,
    
    "dqn_config": {
        "gamma": 0.995,
        "lr": 5e-4,               # Higher LR to overfit quickly
        "batch_size": 64,
        "buffer_size": 100_000,
        "eps_start": 1.0,
        "eps_end": 0.01,          # Very low final epsilon
        "eps_decay_steps": 4000,
        "target_update_interval": 400,
        "n_step": 3,
        "warmup_steps": 800,
    },
    
    "description": "Overfit on specific item set for analysis/debugging."
}

# =============================================================================
# SCENARIO 5: Exploration-Heavy (for diverse datasets)
# =============================================================================
EXPLORATION_HEAVY = {
    "episodes": 400,
    "W": 40, "D": 40, "H": 40,
    "n_items": 120,
    "max_actions": 150,
    "topk_eps": 250,
    "train_freq": 1,
    "num_train_steps": 2,
    
    "dqn_config": {
        "gamma": 0.995,
        "lr": 2e-4,
        "batch_size": 128,
        "buffer_size": 200_000,
        "eps_start": 1.0,
        "eps_end": 0.1,           # Higher final epsilon
        "eps_decay_steps": 12000, # Slower decay
        "target_update_interval": 600,
        "n_step": 3,
        "warmup_steps": 1500,
    },
    
    "description": "More exploration for diverse/challenging problems."
}

# =============================================================================
# Usage Examples
# =============================================================================

def get_config(scenario="medium"):
    """Get configuration for a specific scenario"""
    configs = {
        "small": SMALL_FAST,
        "medium": MEDIUM_BALANCED,
        "large": LARGE_QUALITY,
        "fixed": FIXED_OVERFIT,
        "explore": EXPLORATION_HEAVY,
    }
    return configs.get(scenario, MEDIUM_BALANCED)


def train_with_scenario(scenario="medium", seed=42):
    """
    Train with a predefined scenario configuration
    
    Args:
        scenario: One of ["small", "medium", "large", "fixed", "explore"]
        seed: Random seed
    """
    from packing_with_dqncore_potential_fixed import train_pack_dqn, train_pack_dqn_fixed_items
    from dqn_fixed import DQNConfig
    
    config = get_config(scenario)
    print(f"\n{'='*70}")
    print(f"Training with {scenario.upper()} configuration")
    print(f"Description: {config['description']}")
    print(f"{'='*70}\n")
    
    # Determine which training function to use
    use_fixed = (scenario == "fixed")
    train_fn = train_pack_dqn_fixed_items if use_fixed else train_pack_dqn
    
    # Train
    agent, env = train_fn(
        episodes=config["episodes"],
        W=config["W"],
        D=config["D"],
        H=config["H"],
        n_items=config["n_items"],
        max_actions=config["max_actions"],
        topk_eps=config["topk_eps"],
        train_freq=config["train_freq"],
        num_train_steps=config["num_train_steps"],
        seed=seed,
        log_interval=10,
        save_path=f"/mnt/user-data/outputs/dqn_packing_{scenario}.pt"
    )
    
    return agent, env


# =============================================================================
# Hyperparameter Tuning Tips
# =============================================================================

TUNING_TIPS = """
HYPERPARAMETER TUNING GUIDE
============================

1. LEARNING RATE (lr)
   Problem: Loss oscillates wildly
   Solution: Decrease lr by 2-5x (e.g., 2e-4 → 5e-5)
   
   Problem: No improvement after many episodes
   Solution: Increase lr by 2-3x (e.g., 2e-4 → 5e-4)

2. EPSILON DECAY (eps_decay_steps)
   Problem: Agent stops exploring too early
   Solution: Increase eps_decay_steps by 2x
   
   Problem: Agent explores too much at the end
   Solution: Decrease eps_decay_steps or lower eps_end

3. TARGET UPDATE INTERVAL
   Problem: Training very unstable
   Solution: Increase target_update_interval (e.g., 500 → 1000)
   
   Problem: Learning too slow
   Solution: Decrease target_update_interval (e.g., 500 → 250)

4. BATCH SIZE
   Problem: Not enough GPU memory
   Solution: Decrease batch_size (e.g., 128 → 64)
   
   Problem: Noisy gradients
   Solution: Increase batch_size (e.g., 64 → 128)

5. N-STEP RETURNS
   Problem: Slow credit assignment
   Solution: Increase n_step (e.g., 3 → 5)
   
   Problem: High variance in returns
   Solution: Decrease n_step (e.g., 5 → 3 or 1)

6. MAX ACTIONS
   Problem: Takes too long per step
   Solution: Decrease max_actions (faster but less optimal)
   
   Problem: Missing good actions
   Solution: Increase max_actions (slower but more thorough)

7. TRAIN FREQUENCY & NUM STEPS
   Problem: Training too slow
   Solution: Increase num_train_steps (e.g., 1 → 2 or 4)
   
   Problem: Training unstable
   Solution: Decrease train_freq to train less often

GENERAL DEBUGGING TIPS
======================

1. Start with SMALL_FAST config to verify everything works
2. Monitor MA50 - should increase and stabilize
3. Watch epsilon - should decay smoothly
4. Check loss - should decrease then stabilize
5. If utilization plateaus early:
   - Try higher learning rate
   - Increase exploration (eps_end)
   - Check reward function
6. If training is unstable:
   - Lower learning rate
   - Increase target update interval
   - Increase batch size
"""

def print_tuning_tips():
    """Print hyperparameter tuning guide"""
    print(TUNING_TIPS)


# =============================================================================
# Quick Start
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("DQN 3D BIN PACKING - CONFIGURATION GUIDE")
    print("="*70 + "\n")
    
    print("Available Scenarios:")
    print("-" * 70)
    for name, cfg in [("small", SMALL_FAST), 
                      ("medium", MEDIUM_BALANCED),
                      ("large", LARGE_QUALITY),
                      ("fixed", FIXED_OVERFIT),
                      ("explore", EXPLORATION_HEAVY)]:
        print(f"\n{name.upper()}:")
        print(f"  Container: {cfg['W']}x{cfg['D']}x{cfg['H']}")
        print(f"  Items: {cfg['n_items']}")
        print(f"  Episodes: {cfg['episodes']}")
        print(f"  Description: {cfg['description']}")
    
    print("\n" + "="*70)
    print("USAGE EXAMPLES")
    print("="*70 + "\n")
    
    print("Example 1 - Train with small config:")
    print('  agent, env = train_with_scenario("small")')
    print()
    
    print("Example 2 - Train with large config:")
    print('  agent, env = train_with_scenario("large")')
    print()
    
    print("Example 3 - Manual configuration:")
    print('  config = get_config("medium")')
    print('  # Modify config as needed')
    print('  agent, env = train_pack_dqn(**config)')
    print()
    
    print("Example 4 - Print tuning tips:")
    print('  print_tuning_tips()')
    print()
    
    print("="*70 + "\n")
    
    # Uncomment to run a quick test
    # agent, env = train_with_scenario("small", seed=42)
