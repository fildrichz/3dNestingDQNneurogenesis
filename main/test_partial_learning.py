"""
Check if network is learning PARTIAL improvement even without completing episodes.

This tests:
1. Does utilization improve over training? (0.3 → 0.4 → 0.5?)
2. Do more items get packed over training? (20 → 25 → 30?)
3. Is greedy policy better than random for partial packing?

If YES: Network is learning but stuck at local optimum (can't complete)
If NO: Network isn't learning at all (fundamental architecture issue)
"""

import numpy as np
import torch

def test_partial_learning(agent, env, items, num_episodes=20):
    """
    Test if agent is learning to pack MORE items, even if not all.
    """
    print("\n" + "="*80)
    print("PARTIAL LEARNING TEST")
    print("="*80)

    # Test greedy policy
    print("\nTesting GREEDY policy (network learned)...")
    greedy_utils = []
    greedy_items_packed = []
    greedy_items_remaining = []

    for ep in range(num_episodes):
        env.reset(items=items.copy())
        done = False
        step = 0

        while not done and step < 200:
            actions, mask_short = env.action_space()
            if len(actions) == 0 or mask_short.sum() == 0:
                break

            from packing_with_dqncore2_enhanced import build_action_features
            from nesting.heightmap_utils import extract_patches_for_actions

            obs = env._obs()
            feats = build_action_features(env, actions)
            patches = extract_patches_for_actions(env, actions, patch_size=7)

            # Greedy action
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

        # Record statistics at episode end (successful or stuck)
        total_items = len(items)
        items_packed = total_items - len(env.items)
        items_remaining = len(env.items)
        util = env.total_placed_volume / (env.max_bins * env.bin_volume)

        greedy_utils.append(util)
        greedy_items_packed.append(items_packed)
        greedy_items_remaining.append(items_remaining)

    # Test random policy
    print("Testing RANDOM policy (baseline)...")
    random_utils = []
    random_items_packed = []
    random_items_remaining = []

    for ep in range(num_episodes):
        env.reset(items=items.copy())
        done = False
        step = 0

        while not done and step < 200:
            actions, mask_short = env.action_space()
            if len(actions) == 0 or mask_short.sum() == 0:
                break

            # Random action
            valid = np.where(mask_short > 0.5)[0]
            if len(valid) == 0:
                break

            act_idx = np.random.choice(valid)
            act = actions[act_idx]
            _, _, done, info = env.step(act)
            step += 1

        total_items = len(items)
        items_packed = total_items - len(env.items)
        items_remaining = len(env.items)
        util = env.total_placed_volume / (env.max_bins * env.bin_volume)

        random_utils.append(util)
        random_items_packed.append(items_packed)
        random_items_remaining.append(items_remaining)

    # Print results
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")

    print(f"\nTotal items in problem: {len(items)}")

    print(f"\nGREEDY POLICY (learned):")
    print(f"  Utilization: {np.mean(greedy_utils):.3f} ± {np.std(greedy_utils):.3f}")
    print(f"  Items packed: {np.mean(greedy_items_packed):.1f} ± {np.std(greedy_items_packed):.1f}")
    print(f"  Items remaining: {np.mean(greedy_items_remaining):.1f} ± {np.std(greedy_items_remaining):.1f}")
    print(f"  Packing rate: {np.mean(greedy_items_packed)/len(items)*100:.1f}%")

    print(f"\nRANDOM POLICY (baseline):")
    print(f"  Utilization: {np.mean(random_utils):.3f} ± {np.std(random_utils):.3f}")
    print(f"  Items packed: {np.mean(random_items_packed):.1f} ± {np.std(random_items_packed):.1f}")
    print(f"  Items remaining: {np.mean(random_items_remaining):.1f} ± {np.std(random_items_remaining):.1f}")
    print(f"  Packing rate: {np.mean(random_items_packed)/len(items)*100:.1f}%")

    # Improvement
    util_improvement = np.mean(greedy_utils) - np.mean(random_utils)
    items_improvement = np.mean(greedy_items_packed) - np.mean(random_items_packed)

    print(f"\n{'='*80}")
    print("IMPROVEMENT (Greedy vs Random)")
    print(f"{'='*80}")
    print(f"  Utilization: {util_improvement:+.3f} ({util_improvement/np.mean(random_utils)*100:+.1f}%)")
    print(f"  Items packed: {items_improvement:+.1f} ({items_improvement/np.mean(random_items_packed)*100:+.1f}%)")

    # Statistical significance
    from scipy import stats
    t_util, p_util = stats.ttest_ind(greedy_utils, random_utils)
    t_items, p_items = stats.ttest_ind(greedy_items_packed, random_items_packed)

    print(f"\nStatistical significance:")
    print(f"  Utilization: p={p_util:.4f} {'(significant)' if p_util < 0.05 else '(not significant)'}")
    print(f"  Items packed: p={p_items:.4f} {'(significant)' if p_items < 0.05 else '(not significant)'}")

    # Diagnosis
    print(f"\n{'='*80}")
    print("DIAGNOSIS")
    print(f"{'='*80}")

    if util_improvement < 0.01 and abs(items_improvement) < 1:
        print("❌ CRITICAL: Network has NOT learned anything!")
        print("   Greedy policy is no better than random.")
        print("   This indicates:")
        print("   - Architecture cannot learn from the reward signal")
        print("   - OR gradients are not flowing properly")
        print("   - OR learning rate is too small")
        print("   - OR exploration never discovers better actions")

    elif util_improvement < 0.05:
        print("⚠️  WARNING: Network shows minimal learning.")
        print("   Greedy policy is slightly better than random but improvement is small.")
        print("   This suggests:")
        print("   - Reward signal is too weak")
        print("   - More training needed")
        print("   - Or problem is very hard for this architecture")

    else:
        print("✓ Network IS learning partial improvement!")
        print(f"  Greedy packs {np.mean(greedy_items_packed):.0f}/{len(items)} items vs random {np.mean(random_items_packed):.0f}/{len(items)}")
        print(f"  Network has learned to pack {items_improvement:.0f} more items than random.")
        print()
        print("  However, it's NOT completing episodes (packing all items).")
        print("  This is a SPARSE REWARD problem - network stuck at local optimum.")
        print()
        print("  Solutions:")
        print("  1. Curriculum learning: Start with easier problems (fewer items)")
        print("  2. Better intermediate rewards: Bonus for each N items packed")
        print("  3. Hindsight Experience Replay: Learn from 'failed' episodes")
        print("  4. Much more training: Network needs to accidentally discover completion")

    print(f"{'='*80}\n")

if __name__ == "__main__":
    print("This test requires a trained agent.")
    print("Run this from your training script after agent has trained.")
