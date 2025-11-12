#!/usr/bin/env python3
"""
Test if EMS quality provides meaningful differentiation for local decisions.
"""
import sys
sys.path.insert(0, 'main')
from nesting.packing_core_enhanced import Container, EMS

def compute_ems_quality(bin, bin_volume):
    """Same as in packing_with_dqncore2_enhanced.py - UPDATED to use average"""
    if not bin.ems_list:
        return 0.0

    volumes = sorted([ems.volume() for ems in bin.ems_list], reverse=True)
    top_3 = volumes[:min(3, len(volumes))]
    avg_top_3 = sum(top_3) / len(top_3)
    normalized_avg = avg_top_3 / bin_volume
    ems_quality = normalized_avg ** (1.0/3.0)
    return ems_quality

print("="*80)
print("EMS QUALITY ANALYSIS: Does it differentiate good vs bad placements?")
print("="*80)

W, D, H = 1000, 1000, 1000
bin_volume = W * D * H

# Test 1: Empty bin
print("\nTEST 1: Empty bin")
bin = Container(W, D, H)
ems_quality = compute_ems_quality(bin, bin_volume)
print(f"  EMS quality: {ems_quality:.4f}")
print(f"  Expected: ∛1 = 1.0")
print(f"  Number of EMS: {len(bin.ems_list)}")

# Test 2: Place item in CORNER (good - preserves large spaces)
print("\nTEST 2: Place 200×200×200 item in CORNER (0,0,0)")
bin_corner = Container(W, D, H)
corner_ems = bin_corner.ems_list[0]  # Should be (0,0,0)
bin_corner.place_at_ems(corner_ems, (200, 200, 200), weight=10, item_id=0)

ems_corner = compute_ems_quality(bin_corner, bin_volume)
bin_vol_corner = sum(b.w * b.d * b.h for b in bin_corner.placed)
bin_util_corner = bin_vol_corner / bin_volume

print(f"  Bin utilization: {bin_util_corner:.4f}")
print(f"  EMS quality: {ems_corner:.4f}")
print(f"  Number of EMS: {len(bin_corner.ems_list)}")
top_3_corner = sorted([ems.volume() for ems in bin_corner.ems_list], reverse=True)[:3]
print(f"  Top-3 EMS volumes: {[f'{v/1e6:.1f}M' for v in top_3_corner]}")
potential_corner = bin_util_corner + 0.3 * ems_corner
print(f"  Potential: {potential_corner:.4f}")

# Test 3: Place item in CENTER (bad - fragments space)
print("\nTEST 3: Place 200×200×200 item in CENTER (400,400,0)")
bin_center = Container(W, D, H)
# Find or create EMS at center
center_ems = EMS(400, 400, 0, 200, 200, 1000)
bin_center.place_at_ems(center_ems, (200, 200, 200), weight=10, item_id=0)

ems_center = compute_ems_quality(bin_center, bin_volume)
bin_vol_center = sum(b.w * b.d * b.h for b in bin_center.placed)
bin_util_center = bin_vol_center / bin_volume

print(f"  Bin utilization: {bin_util_center:.4f}")
print(f"  EMS quality: {ems_center:.4f}")
print(f"  Number of EMS: {len(bin_center.ems_list)}")
top_3_center = sorted([ems.volume() for ems in bin_center.ems_list], reverse=True)[:3]
print(f"  Top-3 EMS volumes: {[f'{v/1e6:.1f}M' for v in top_3_center]}")
potential_center = bin_util_center + 0.3 * ems_center
print(f"  Potential: {potential_center:.4f}")

# Compare
print("\n" + "="*80)
print("COMPARISON: Corner vs Center placement")
print("="*80)
print(f"Both have same utilization: {bin_util_corner:.4f}")
print(f"Corner EMS quality: {ems_corner:.4f}")
print(f"Center EMS quality: {ems_center:.4f}")
print(f"Difference: {ems_corner - ems_center:.4f} ({(ems_corner - ems_center)/ems_center*100:.1f}%)")
print()

# Compute reward difference
gamma = 0.99
reward_corner = (gamma * potential_corner) - 1.0  # Starting from empty bin
reward_center = (gamma * potential_center) - 1.0

print(f"Reward for corner placement: {reward_corner:.5f}")
print(f"Reward for center placement: {reward_center:.5f}")
print(f"Difference: {reward_corner - reward_center:.5f}")
print()

if reward_corner > reward_center:
    print("✓ GOOD: Corner placement gets higher reward")
    print(f"  Agent incentivized to place in corners by {(reward_corner - reward_center)*1000:.2f} millipoints")
else:
    print("✗ BAD: EMS quality not differentiating well enough!")

# Test magnitude of EMS contribution
print("\n" + "="*80)
print("MAGNITUDE ANALYSIS: Is EMS term significant?")
print("="*80)
util_contribution = bin_util_corner
ems_contribution = 0.3 * ems_corner
print(f"Utilization term: {util_contribution:.5f}")
print(f"EMS term (β=0.3): {ems_contribution:.5f}")
print(f"Ratio: Util is {util_contribution / ems_contribution:.1f}x larger than EMS")
print()
print(f"EMS difference between placements: {0.3 * (ems_corner - ems_center):.5f}")
print(f"Typical util increase per item: {bin_util_corner:.5f}")
print(f"EMS difference is {0.3 * (ems_corner - ems_center) / bin_util_corner * 100:.1f}% of util increase")
print()

if 0.3 * (ems_corner - ems_center) < bin_util_corner * 0.1:
    print("⚠️  WARNING: EMS term is <10% of utilization increase")
    print("   Consider increasing β (currently 0.3) to 0.5-1.0 for stronger signal")
else:
    print("✓ EMS term provides meaningful differentiation")
