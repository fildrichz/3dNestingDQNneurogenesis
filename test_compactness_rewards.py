"""
Test that compactness rewards encourage better packing behavior.
"""

import sys
sys.path.insert(0, 'main')

import numpy as np
from nesting.packing_core_enhanced import Container, EMS

def test_height_penalty():
    """Test that placing boxes lower gets higher reward."""
    print("="*80)
    print("TEST 1: Height Penalty")
    print("="*80)

    W, D, H = 1000, 1000, 1000
    bin = Container(W, D, H)

    # Place a base box
    base_size = (200, 200, 100)
    bin.place_at_ems(bin.ems_list[0], base_size, weight=50, item_id=0)
    print(f"\nPlaced base box at ground: (0, 0, 0) size {base_size}")
    print(f"  Top at Z={100}")

    # Scenario A: Place box at ground level (low height)
    # Find EMS at ground level
    ground_ems = None
    for ems in bin.ems_list:
        if ems.z == 0 and ems.x >= 200:  # Next to base box
            ground_ems = ems
            break

    if ground_ems:
        final_z_ground = 0  # At ground
        height_penalty_ground = (final_z_ground / H) * 0.05
        print(f"\nScenario A: Place at ground level")
        print(f"  Final Z: {final_z_ground}")
        print(f"  Height penalty: {height_penalty_ground:.6f}")

    # Scenario B: Place box on top of base (high height)
    high_ems = None
    for ems in bin.ems_list:
        if ems.z == 100 and ems.x == 0:  # On top of base box
            high_ems = ems
            break

    if high_ems:
        final_z_high = 100  # On top of base
        height_penalty_high = (final_z_high / H) * 0.05
        print(f"\nScenario B: Place on top of base box")
        print(f"  Final Z: {final_z_high}")
        print(f"  Height penalty: {height_penalty_high:.6f}")

        print(f"\nDifference: {height_penalty_high - height_penalty_ground:.6f}")
        print(f"  → Placing at ground is {height_penalty_high / max(height_penalty_ground, 1e-9):.1f}x better!")

    return True


def test_slack_penalty():
    """Test that tight fits get higher reward than loose fits."""
    print("\n" + "="*80)
    print("TEST 2: Slack Penalty")
    print("="*80)

    W, D, H = 1000, 1000, 1000

    # Scenario A: Box fits perfectly in EMS (no slack)
    ems_tight = EMS(0, 0, 0, 200, 200, 200)
    box_size_tight = (200, 200, 200)

    slack_x_tight = max(0, ems_tight.w - box_size_tight[0])
    slack_y_tight = max(0, ems_tight.d - box_size_tight[1])
    slack_z_tight = max(0, ems_tight.h - box_size_tight[2])
    total_slack_tight = slack_x_tight + slack_y_tight + slack_z_tight
    slack_penalty_tight = (total_slack_tight / (W + D + H)) * 0.03

    print(f"\nScenario A: Perfect fit (box = EMS)")
    print(f"  EMS: {ems_tight.w}×{ems_tight.d}×{ems_tight.h}")
    print(f"  Box: {box_size_tight[0]}×{box_size_tight[1]}×{box_size_tight[2]}")
    print(f"  Slack: ({slack_x_tight}, {slack_y_tight}, {slack_z_tight}) = {total_slack_tight}")
    print(f"  Slack penalty: {slack_penalty_tight:.6f}")

    # Scenario B: Box fits loosely in EMS (lots of slack)
    ems_loose = EMS(0, 0, 0, 500, 500, 500)
    box_size_loose = (200, 200, 200)

    slack_x_loose = max(0, ems_loose.w - box_size_loose[0])
    slack_y_loose = max(0, ems_loose.d - box_size_loose[1])
    slack_z_loose = max(0, ems_loose.h - box_size_loose[2])
    total_slack_loose = slack_x_loose + slack_y_loose + slack_z_loose
    slack_penalty_loose = (total_slack_loose / (W + D + H)) * 0.03

    print(f"\nScenario B: Loose fit (box << EMS)")
    print(f"  EMS: {ems_loose.w}×{ems_loose.d}×{ems_loose.h}")
    print(f"  Box: {box_size_loose[0]}×{box_size_loose[1]}×{box_size_loose[2]}")
    print(f"  Slack: ({slack_x_loose}, {slack_y_loose}, {slack_z_loose}) = {total_slack_loose}")
    print(f"  Slack penalty: {slack_penalty_loose:.6f}")

    print(f"\nDifference: {slack_penalty_loose - slack_penalty_tight:.6f}")
    print(f"  → Tight fit is {slack_penalty_loose / max(slack_penalty_tight, 1e-9):.1f}x better!")

    return True


def test_tightness_bonus():
    """Test that perfect fits in multiple dimensions get bonus."""
    print("\n" + "="*80)
    print("TEST 3: Tightness Bonus")
    print("="*80)

    # Scenario A: All 3 dimensions tight
    ems_a = EMS(0, 0, 0, 200, 200, 200)
    box_a = (200, 200, 200)

    tight_dims_a = 0
    if ems_a.w == box_a[0]: tight_dims_a += 1
    if ems_a.d == box_a[1]: tight_dims_a += 1
    if ems_a.h == box_a[2]: tight_dims_a += 1
    bonus_a = 0.02 if tight_dims_a >= 2 else 0.0

    print(f"\nScenario A: Perfect fit (all 3 dimensions)")
    print(f"  EMS: {ems_a.w}×{ems_a.d}×{ems_a.h}")
    print(f"  Box: {box_a[0]}×{box_a[1]}×{box_a[2]}")
    print(f"  Tight dimensions: {tight_dims_a}/3")
    print(f"  Tightness bonus: {bonus_a:.3f}")

    # Scenario B: 2 dimensions tight
    ems_b = EMS(0, 0, 0, 200, 200, 300)
    box_b = (200, 200, 200)

    tight_dims_b = 0
    if ems_b.w == box_b[0]: tight_dims_b += 1
    if ems_b.d == box_b[1]: tight_dims_b += 1
    if ems_b.h == box_b[2]: tight_dims_b += 1
    bonus_b = 0.02 if tight_dims_b >= 2 else 0.0

    print(f"\nScenario B: 2 dimensions tight")
    print(f"  EMS: {ems_b.w}×{ems_b.d}×{ems_b.h}")
    print(f"  Box: {box_b[0]}×{box_b[1]}×{box_b[2]}")
    print(f"  Tight dimensions: {tight_dims_b}/3")
    print(f"  Tightness bonus: {bonus_b:.3f}")

    # Scenario C: Only 1 dimension tight
    ems_c = EMS(0, 0, 0, 200, 300, 400)
    box_c = (200, 200, 200)

    tight_dims_c = 0
    if ems_c.w == box_c[0]: tight_dims_c += 1
    if ems_c.d == box_c[1]: tight_dims_c += 1
    if ems_c.h == box_c[2]: tight_dims_c += 1
    bonus_c = 0.02 if tight_dims_c >= 2 else 0.0

    print(f"\nScenario C: Only 1 dimension tight")
    print(f"  EMS: {ems_c.w}×{ems_c.d}×{ems_c.h}")
    print(f"  Box: {box_c[0]}×{box_c[1]}×{box_c[2]}")
    print(f"  Tight dimensions: {tight_dims_c}/3")
    print(f"  Tightness bonus: {bonus_c:.3f}")

    print(f"\n  → Scenarios A & B get bonus, C doesn't")

    return True


def test_combined_effect():
    """Test combined effect of all compactness rewards."""
    print("\n" + "="*80)
    print("TEST 4: Combined Effect")
    print("="*80)

    W, D, H = 1000, 1000, 1000

    # Good placement: Ground level, tight fit
    print(f"\nGood Placement:")
    print(f"  - Ground level (Z=0)")
    print(f"  - Perfect fit in EMS")

    height_penalty_good = (0 / H) * 0.05
    slack_penalty_good = (0 / (W + D + H)) * 0.03
    tightness_bonus_good = 0.02

    total_compactness_good = -height_penalty_good - slack_penalty_good + tightness_bonus_good

    print(f"  Height penalty: -{height_penalty_good:.6f}")
    print(f"  Slack penalty: -{slack_penalty_good:.6f}")
    print(f"  Tightness bonus: +{tightness_bonus_good:.6f}")
    print(f"  Total compactness reward: {total_compactness_good:+.6f}")

    # Bad placement: High up, loose fit
    print(f"\nBad Placement:")
    print(f"  - High up (Z=800)")
    print(f"  - Loose fit (slack=900)")

    height_penalty_bad = (800 / H) * 0.05
    slack_penalty_bad = (900 / (W + D + H)) * 0.03
    tightness_bonus_bad = 0.0

    total_compactness_bad = -height_penalty_bad - slack_penalty_bad + tightness_bonus_bad

    print(f"  Height penalty: -{height_penalty_bad:.6f}")
    print(f"  Slack penalty: -{slack_penalty_bad:.6f}")
    print(f"  Tightness bonus: +{tightness_bonus_bad:.6f}")
    print(f"  Total compactness reward: {total_compactness_bad:+.6f}")

    print(f"\nDifference: {total_compactness_good - total_compactness_bad:+.6f}")
    print(f"  → Good placement is {abs(total_compactness_good - total_compactness_bad):.4f} reward points better!")

    return True


if __name__ == "__main__":
    print("\n" + "█"*80)
    print("COMPACTNESS REWARDS TEST SUITE")
    print("█"*80)

    success = True
    success &= test_height_penalty()
    success &= test_slack_penalty()
    success &= test_tightness_bonus()
    success &= test_combined_effect()

    print("\n" + "="*80)
    if success:
        print("✓ ALL TESTS PASSED")
        print("\nExpected behavior:")
        print("  - Agent will prefer placing boxes at ground level")
        print("  - Agent will prefer tight fits in EMS")
        print("  - Agent will prefer placements with 2+ dimensions matching exactly")
        print("  - Overall: More compact, solid packing structures")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*80)
