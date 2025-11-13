"""
Check if episodes are completing successfully (all items packed).

This determines why terminal rewards are missing.
"""

import numpy as np

def check_episode_completion(env, items, num_episodes=20):
    """
    Check how many episodes successfully pack all items.
    """
    print("\n" + "="*80)
    print("EPISODE COMPLETION ANALYSIS")
    print("="*80)

    completion_stats = {
        'full_completion': 0,
        'no_actions_stuck': 0,
        'invalid_action': 0,
        'items_packed': [],
        'items_remaining': [],
        'final_rewards': [],
        'completion_rewards': [],
    }

    for ep in range(num_episodes):
        env.reset(items=items.copy())
        done = False
        step = 0
        last_reward = None

        while not done and step < 100:
            actions, mask_short = env.action_space()

            # Check if stuck (no valid actions)
            if len(actions) == 0 or mask_short.sum() == 0:
                completion_stats['no_actions_stuck'] += 1
                break

            # Take random action
            valid = np.where(mask_short > 0.5)[0]
            if len(valid) == 0:
                completion_stats['no_actions_stuck'] += 1
                break

            act_idx = np.random.choice(valid)
            act = actions[act_idx]

            _, reward, done, info = env.step(act)
            last_reward = reward
            step += 1

            if done:
                # Check completion type
                items_remaining = info.get('items_remaining', len(env.items))
                items_packed = info.get('items_placed', 0)

                completion_stats['items_packed'].append(items_packed)
                completion_stats['items_remaining'].append(items_remaining)
                completion_stats['final_rewards'].append(reward)

                if items_remaining == 0:
                    completion_stats['full_completion'] += 1
                    completion_stats['completion_rewards'].append(reward)
                elif 'invalid' in info:
                    completion_stats['invalid_action'] += 1

    # Print results
    total_episodes = num_episodes
    print(f"\nAnalyzed {total_episodes} episodes:")
    print(f"\n✓ Full completion (all items packed): {completion_stats['full_completion']} ({completion_stats['full_completion']/total_episodes*100:.1f}%)")
    print(f"⚠ Stuck (no valid actions): {completion_stats['no_actions_stuck']} ({completion_stats['no_actions_stuck']/total_episodes*100:.1f}%)")
    print(f"❌ Invalid action error: {completion_stats['invalid_action']} ({completion_stats['invalid_action']/total_episodes*100:.1f}%)")

    if len(completion_stats['items_packed']) > 0:
        print(f"\nItems packed when episode ended:")
        print(f"  Mean: {np.mean(completion_stats['items_packed']):.1f}")
        print(f"  Range: [{np.min(completion_stats['items_packed'])}, {np.max(completion_stats['items_packed'])}]")
        print(f"  Total items available: {len(items)}")

    if len(completion_stats['items_remaining']) > 0:
        print(f"\nItems remaining when episode ended:")
        print(f"  Mean: {np.mean(completion_stats['items_remaining']):.1f}")
        print(f"  Range: [{np.min(completion_stats['items_remaining'])}, {np.max(completion_stats['items_remaining'])}]")

    if len(completion_stats['final_rewards']) > 0:
        print(f"\nFinal step rewards (all episodes):")
        print(f"  Mean: {np.mean(completion_stats['final_rewards']):.4f}")
        print(f"  Range: [{np.min(completion_stats['final_rewards']):.4f}, {np.max(completion_stats['final_rewards']):.4f}]")

    if len(completion_stats['completion_rewards']) > 0:
        print(f"\nCompletion bonus (successful episodes only):")
        print(f"  Mean: {np.mean(completion_stats['completion_rewards']):.4f}")
        print(f"  Range: [{np.min(completion_stats['completion_rewards']):.4f}, {np.max(completion_stats['completion_rewards']):.4f}]")
        print(f"  Expected: 0.3 to 0.5 (0.3 base + 0.0-0.2 bins efficiency)")
    else:
        print(f"\n❌ NO EPISODES COMPLETED SUCCESSFULLY!")
        print(f"   This means the terminal reward bonus is NEVER applied.")
        print(f"   This is why rewards are so small and learning is ineffective!")

    # Diagnosis
    print(f"\n" + "="*80)
    print("DIAGNOSIS")
    print("="*80)

    if completion_stats['full_completion'] == 0:
        print("❌ CRITICAL PROBLEM: Episodes never complete successfully!")
        print("   Agent is getting stuck before packing all items.")
        print("   This means:")
        print("   1. No terminal reward bonus (0.3-0.5) is ever applied")
        print("   2. All learning signal comes from tiny step rewards (~0.003)")
        print("   3. Network has no strong signal for 'success'")
        print()
        print("   CAUSE: Agent is likely hitting invalid states or running out of valid actions")
        print("   due to:")
        print("   - Constraint violations blocking placements")
        print("   - Poor action enumeration")
        print("   - Fragmented EMS preventing placement of remaining items")
    elif completion_stats['full_completion'] < total_episodes * 0.5:
        print("⚠️  WARNING: Less than 50% of episodes complete successfully.")
        print(f"   Only {completion_stats['full_completion']}/{total_episodes} episodes pack all items.")
        print("   Learning signal is weak because terminal bonus rarely applied.")
    else:
        print("✓ Episodes are completing successfully.")
        print("   If learning is still poor, the problem is elsewhere.")

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

    check_episode_completion(env, items, num_episodes=50)
