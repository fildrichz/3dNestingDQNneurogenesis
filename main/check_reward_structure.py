"""
Analyze the reward structure to understand learning signals.

This checks if potential-based shaping is dominating the learning signal,
which could make n-step irrelevant.
"""

import numpy as np

def analyze_reward_structure(env, items, num_episodes=10):
    """
    Analyze reward magnitudes and temporal structure.
    """
    print("\n" + "="*80)
    print("REWARD STRUCTURE ANALYSIS")
    print("="*80)

    step_rewards = []
    terminal_rewards = []
    episode_returns = []
    reward_sequences = []

    for ep in range(num_episodes):
        env.reset(items=items.copy())
        done = False
        step = 0
        episode_rewards = []

        while not done and step < 100:
            actions, mask_short = env.action_space()
            if len(actions) == 0 or mask_short.sum() == 0:
                break

            # Take random action
            valid = np.where(mask_short > 0.5)[0]
            if len(valid) == 0:
                break

            act_idx = np.random.choice(valid)
            act = actions[act_idx]

            _, reward, done, info = env.step(act)
            episode_rewards.append(reward)

            if not done:
                step_rewards.append(reward)
            else:
                terminal_rewards.append(reward)

            step += 1

        episode_returns.append(sum(episode_rewards))
        reward_sequences.append(episode_rewards)

    print(f"\nAnalyzed {num_episodes} episodes")

    # Step rewards (non-terminal)
    if len(step_rewards) > 0:
        print(f"\nStep Rewards (potential-based shaping):")
        print(f"  Count: {len(step_rewards)}")
        print(f"  Mean: {np.mean(step_rewards):.4f}")
        print(f"  Std: {np.std(step_rewards):.4f}")
        print(f"  Range: [{np.min(step_rewards):.4f}, {np.max(step_rewards):.4f}]")
        print(f"  Median: {np.median(step_rewards):.4f}")

    # Terminal rewards
    if len(terminal_rewards) > 0:
        print(f"\nTerminal Rewards (completion bonus):")
        print(f"  Count: {len(terminal_rewards)}")
        print(f"  Mean: {np.mean(terminal_rewards):.4f}")
        print(f"  Std: {np.std(terminal_rewards):.4f}")
        print(f"  Range: [{np.min(terminal_rewards):.4f}, {np.max(terminal_rewards):.4f}]")

    # Episode returns
    if len(episode_returns) > 0:
        print(f"\nEpisode Returns:")
        print(f"  Mean: {np.mean(episode_returns):.4f}")
        print(f"  Std: {np.std(episode_returns):.4f}")
        print(f"  Range: [{np.min(episode_returns):.4f}, {np.max(episode_returns):.4f}]")

    # Analyze signal strength
    if len(step_rewards) > 0 and len(terminal_rewards) > 0:
        avg_step_reward = np.mean(np.abs(step_rewards))
        avg_terminal_reward = np.mean(np.abs(terminal_rewards))
        avg_episode_length = np.mean([len(seq) for seq in reward_sequences])

        total_step_signal = avg_step_reward * avg_episode_length
        terminal_signal = avg_terminal_reward

        print(f"\n" + "-"*80)
        print("SIGNAL STRENGTH ANALYSIS")
        print("-"*80)
        print(f"Average episode length: {avg_episode_length:.1f} steps")
        print(f"Avg step reward magnitude: {avg_step_reward:.4f}")
        print(f"Avg terminal reward magnitude: {avg_terminal_reward:.4f}")
        print(f"\nTotal signal from step rewards: {total_step_signal:.4f}")
        print(f"Signal from terminal reward: {terminal_signal:.4f}")
        print(f"\nRatio (terminal / step total): {terminal_signal / total_step_signal:.2f}")

        if terminal_signal < total_step_signal:
            print(f"\n⚠️  FINDING: Step rewards (potential-based) provide MORE signal than terminal reward!")
            print(f"   This means n-step is less important because rewards are already DENSE.")
            print(f"   The network can learn primarily from 1-step returns.")
        else:
            print(f"\n✓ Terminal reward is dominant. N-step returns are important for credit assignment.")

    # Check temporal structure
    print(f"\n" + "-"*80)
    print("TEMPORAL STRUCTURE")
    print("-"*80)

    for i, seq in enumerate(reward_sequences[:3]):  # Show first 3 episodes
        print(f"\nEpisode {i+1} reward sequence ({len(seq)} steps):")
        if len(seq) <= 10:
            print(f"  {[f'{r:.3f}' for r in seq]}")
        else:
            print(f"  First 5: {[f'{r:.3f}' for r in seq[:5]]}")
            print(f"  Last 5:  {[f'{r:.3f}' for r in seq[-5:]]}")
            print(f"  Terminal: {seq[-1]:.3f}")

    # Check for any negative rewards (penalties)
    negative_rewards = [r for r in step_rewards + terminal_rewards if r < 0]
    if len(negative_rewards) > 0:
        print(f"\n⚠️  Found {len(negative_rewards)} negative rewards (penalties)")
        print(f"   This could cause instability in Q-value learning.")

    print("="*80 + "\n")

if __name__ == "__main__":
    from packing_with_dqncore2_enhanced import MultiBinPackingEnv, load_problem, load_problem_as_items

    # Load a test problem
    problem_path = "nesting/inputData/Benchmark dataset and instance generator for Real-World 3dBPP/Input/3dBPP_4.txt"
    problem = load_problem(problem_path)
    items = load_problem_as_items(problem)

    W, D, H = problem.bin_dimensions
    env = MultiBinPackingEnv(W, D, H, items=items, max_actions=128, topk_eps=1000,
                              seed=42, gamma=0.992, problem=problem)

    analyze_reward_structure(env, items, num_episodes=20)
