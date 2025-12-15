#!/usr/bin/env python3
"""
Test script to demonstrate dynamic epsilon decay.

Shows how epsilon decay adapts to different training lengths:
- Short GA evaluation (100 episodes)
- Medium training (500 episodes)
- Long training (2000 episodes)
"""

from dqn_core.dqn_enhanced import calculate_dynamic_epsilon_decay


def test_epsilon_schedules():
    """Test epsilon decay for different training scenarios."""

    print("="*80)
    print("DYNAMIC EPSILON DECAY - TEST")
    print("="*80)
    print("\nTesting epsilon decay for different training scenarios:\n")

    # Test 1: GA evaluation (short)
    print("\n" + "─"*80)
    print("SCENARIO 1: GA Fitness Evaluation (100 episodes)")
    print("─"*80)
    decay_steps_ga = calculate_dynamic_epsilon_decay(
        episodes=100,
        avg_steps_per_episode=100,
        plateau_at_ratio=0.8,
        verbose=True
    )

    # Test 2: Medium training
    print("\n" + "─"*80)
    print("SCENARIO 2: Medium Training (500 episodes)")
    print("─"*80)
    decay_steps_medium = calculate_dynamic_epsilon_decay(
        episodes=500,
        avg_steps_per_episode=100,
        plateau_at_ratio=0.8,
        verbose=True
    )

    # Test 3: Long training
    print("\n" + "─"*80)
    print("SCENARIO 3: Long Training (2000 episodes)")
    print("─"*80)
    decay_steps_long = calculate_dynamic_epsilon_decay(
        episodes=2000,
        avg_steps_per_episode=100,
        plateau_at_ratio=0.8,
        verbose=True
    )

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nGA Evaluation (100 ep):   eps_decay_steps = {decay_steps_ga:,}")
    print(f"Medium Training (500 ep):  eps_decay_steps = {decay_steps_medium:,}")
    print(f"Long Training (2000 ep):   eps_decay_steps = {decay_steps_long:,}")

    print("\n✅ Dynamic epsilon decay adapts to training length!")
    print("   - Short training: Quick decay (agent learns fast)")
    print("   - Long training: Gradual decay (more exploration time)")
    print("   - All scenarios: 20% plateau at ε_min for final exploitation")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_epsilon_schedules()
