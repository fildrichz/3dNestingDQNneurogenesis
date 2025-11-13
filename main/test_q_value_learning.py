"""
Test if Q-network is actually learning meaningful action discrimination.

This script checks:
1. Do Q-values differ significantly between actions? (action discrimination)
2. Do higher Q-value actions lead to better outcomes? (Q-value accuracy)
3. Is greedy policy better than random? (learning effectiveness)
"""

import numpy as np
from collections import defaultdict

def test_q_value_discrimination(agent, env, items, num_episodes=10):
    """
    Test if Q-values meaningfully discriminate between actions.

    If Q-values for all actions are similar, the network hasn't learned
    meaningful action selection.
    """
    print("\n" + "="*80)
    print("TEST 1: Q-VALUE DISCRIMINATION")
    print("="*80)

    q_value_spreads = []
    q_value_ranges = []

    for ep in range(num_episodes):
        env.reset(items=items.copy())
        done = False
        step = 0

        while not done and step < 100:
            actions, mask_short = env.action_space()
            if len(actions) == 0 or mask_short.sum() == 0:
                break

            # Get Q-values for all actions
            from packing_with_dqncore2_enhanced import build_action_features
            from nesting.heightmap_utils import extract_patches_for_actions

            obs = env._obs()
            feats = build_action_features(env, actions)
            patches = extract_patches_for_actions(env, actions, patch_size=7)

            # Forward pass
            import torch
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(agent.device)
            feats_t = torch.from_numpy(feats).float().unsqueeze(0).to(agent.device)
            patches_t = torch.from_numpy(patches).float().unsqueeze(0).to(agent.device)
            mask_t = torch.from_numpy(mask_short).float().unsqueeze(0).to(agent.device)

            with torch.no_grad():
                q_values = agent.q(obs_t, feats_t, patches_t, mask_t)[0].cpu().numpy()

            # Get valid Q-values
            valid_q = q_values[mask_short > 0.5]

            if len(valid_q) > 1:
                q_spread = np.std(valid_q)
                q_range = np.max(valid_q) - np.min(valid_q)
                q_value_spreads.append(q_spread)
                q_value_ranges.append(q_range)

            # Take action (don't care about outcome for this test)
            act_idx = agent.select_action(obs, feats, patches, mask_short)
            if act_idx is None:
                break

            act = actions[act_idx]
            _, _, done, _ = env.step(act)
            step += 1

    print(f"\nTested {len(q_value_spreads)} decision points across {num_episodes} episodes")
    if len(q_value_spreads) > 0:
        print(f"\nQ-value spread (std dev):")
        print(f"  Mean: {np.mean(q_value_spreads):.4f}")
        print(f"  Median: {np.median(q_value_spreads):.4f}")
        print(f"  Range: [{np.min(q_value_spreads):.4f}, {np.max(q_value_spreads):.4f}]")

        print(f"\nQ-value range (max - min):")
        print(f"  Mean: {np.mean(q_value_ranges):.4f}")
        print(f"  Median: {np.median(q_value_ranges):.4f}")
        print(f"  Range: [{np.min(q_value_ranges):.4f}, {np.max(q_value_ranges):.4f}]")

        # Interpretation
        avg_spread = np.mean(q_value_spreads)
        if avg_spread < 0.01:
            print(f"\n❌ PROBLEM: Q-values are nearly identical across actions!")
            print(f"   Network is NOT learning meaningful action discrimination.")
            print(f"   All actions look the same to the network.")
        elif avg_spread < 0.1:
            print(f"\n⚠️  WARNING: Q-values have low variance.")
            print(f"   Network may be learning slowly or action features are insufficient.")
        else:
            print(f"\n✓ Q-values show meaningful differences between actions.")

    print("="*80)

def test_q_value_accuracy(agent, env, items, num_episodes=5):
    """
    Test if higher Q-value actions actually lead to better outcomes.

    Compares actual returns with predicted Q-values.
    """
    print("\n" + "="*80)
    print("TEST 2: Q-VALUE ACCURACY")
    print("="*80)

    q_predictions = []
    actual_returns = []

    for ep in range(num_episodes):
        env.reset(items=items.copy())
        done = False
        step = 0
        episode_return = 0

        while not done and step < 100:
            actions, mask_short = env.action_space()
            if len(actions) == 0 or mask_short.sum() == 0:
                break

            from packing_with_dqncore2_enhanced import build_action_features
            from nesting.heightmap_utils import extract_patches_for_actions

            obs = env._obs()
            feats = build_action_features(env, actions)
            patches = extract_patches_for_actions(env, actions, patch_size=7)

            # Get Q-value prediction
            import torch
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(agent.device)
            feats_t = torch.from_numpy(feats).float().unsqueeze(0).to(agent.device)
            patches_t = torch.from_numpy(patches).float().unsqueeze(0).to(agent.device)
            mask_t = torch.from_numpy(mask_short).float().unsqueeze(0).to(agent.device)

            with torch.no_grad():
                q_values = agent.q(obs_t, feats_t, patches_t, mask_t)[0].cpu().numpy()

            # Select action greedily
            q_values[mask_short < 0.5] = -np.inf
            act_idx = int(np.argmax(q_values))
            predicted_q = q_values[act_idx]

            act = actions[act_idx]
            _, reward, done, _ = env.step(act)

            q_predictions.append(predicted_q)
            episode_return += reward * (agent.cfg.gamma ** step)

            step += 1

        # Propagate actual return back to all steps
        for i in range(len(q_predictions) - len(actual_returns), len(q_predictions)):
            actual_returns.append(episode_return)

    if len(q_predictions) > 0:
        correlation = np.corrcoef(q_predictions, actual_returns)[0, 1]

        print(f"\nCompared {len(q_predictions)} predictions across {num_episodes} episodes")
        print(f"\nQ-value vs Actual Return correlation: {correlation:.3f}")

        print(f"\nPredicted Q-values:")
        print(f"  Mean: {np.mean(q_predictions):.3f}")
        print(f"  Range: [{np.min(q_predictions):.3f}, {np.max(q_predictions):.3f}]")

        print(f"\nActual returns:")
        print(f"  Mean: {np.mean(actual_returns):.3f}")
        print(f"  Range: [{np.min(actual_returns):.3f}, {np.max(actual_returns):.3f}]")

        # Interpretation
        if abs(correlation) < 0.1:
            print(f"\n❌ PROBLEM: Q-values have no correlation with actual returns!")
            print(f"   Network predictions are meaningless.")
        elif abs(correlation) < 0.3:
            print(f"\n⚠️  WARNING: Weak correlation between Q-values and returns.")
            print(f"   Network predictions are poor.")
        else:
            print(f"\n✓ Q-values correlate with actual returns.")

    print("="*80)

def test_greedy_vs_random(agent, env, items, num_episodes=10):
    """
    Compare greedy policy performance vs random policy.

    If greedy is not better than random, network hasn't learned.
    """
    print("\n" + "="*80)
    print("TEST 3: GREEDY VS RANDOM POLICY")
    print("="*80)

    greedy_utils = []
    random_utils = []

    # Test greedy policy
    print("\nTesting greedy policy...")
    for ep in range(num_episodes):
        env.reset(items=items.copy())
        done = False
        step = 0

        while not done and step < 100:
            actions, mask_short = env.action_space()
            if len(actions) == 0 or mask_short.sum() == 0:
                break

            from packing_with_dqncore2_enhanced import build_action_features
            from nesting.heightmap_utils import extract_patches_for_actions

            obs = env._obs()
            feats = build_action_features(env, actions)
            patches = extract_patches_for_actions(env, actions, patch_size=7)

            # Greedy action selection
            import torch
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(agent.device)
            feats_t = torch.from_numpy(feats).float().unsqueeze(0).to(agent.device)
            patches_t = torch.from_numpy(patches).float().unsqueeze(0).to(agent.device)
            mask_t = torch.from_numpy(mask_short).float().unsqueeze(0).to(agent.device)

            with torch.no_grad():
                q_values = agent.q(obs_t, feats_t, patches_t, mask_t)[0].cpu().numpy()

            q_values[mask_short < 0.5] = -np.inf
            act_idx = int(np.argmax(q_values))

            act = actions[act_idx]
            _, _, done, info = env.step(act)
            step += 1

        util = info.get('utilization', 0.0)
        greedy_utils.append(util)

    # Test random policy
    print("Testing random policy...")
    for ep in range(num_episodes):
        env.reset(items=items.copy())
        done = False
        step = 0

        while not done and step < 100:
            actions, mask_short = env.action_space()
            if len(actions) == 0 or mask_short.sum() == 0:
                break

            # Random action selection
            valid = np.where(mask_short > 0.5)[0]
            if len(valid) == 0:
                break

            act_idx = np.random.choice(valid)
            act = actions[act_idx]
            _, _, done, info = env.step(act)
            step += 1

        util = info.get('utilization', 0.0)
        random_utils.append(util)

    print(f"\nResults after {num_episodes} episodes each:")
    print(f"\nGreedy policy:")
    print(f"  Mean utilization: {np.mean(greedy_utils):.3f} ± {np.std(greedy_utils):.3f}")

    print(f"\nRandom policy:")
    print(f"  Mean utilization: {np.mean(random_utils):.3f} ± {np.std(random_utils):.3f}")

    improvement = np.mean(greedy_utils) - np.mean(random_utils)
    print(f"\nGreedy improvement over random: {improvement:+.3f} ({improvement/np.mean(random_utils)*100:+.1f}%)")

    # Statistical test
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(greedy_utils, random_utils)
    print(f"T-test p-value: {p_value:.4f}")

    # Interpretation
    if improvement < 0.01:
        print(f"\n❌ PROBLEM: Greedy policy is NO BETTER than random!")
        print(f"   Network has NOT learned a useful policy.")
    elif improvement < 0.05:
        print(f"\n⚠️  WARNING: Greedy policy only slightly better than random.")
        print(f"   Network may be learning slowly or Q-values are inaccurate.")
    else:
        print(f"\n✓ Greedy policy significantly better than random.")
        if p_value < 0.05:
            print(f"   Improvement is statistically significant.")

    print("="*80)

if __name__ == "__main__":
    print("Q-Value Learning Test Suite")
    print("This requires a trained agent. Please run from your training script.")
