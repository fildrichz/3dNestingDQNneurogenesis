#!/usr/bin/env python3
"""
Test that relative positioning constraint works based on XY overlap only.
Z position should be irrelevant.
"""
import sys
sys.path.insert(0, 'main')

from nesting.packing_core_enhanced import Container, EMS

print("="*80)
print("RELATIVE POSITIONING: XY OVERLAP CONSTRAINT TEST")
print("="*80)
print("\nConstraint: Heavy items cannot occupy ANY XY position where light items exist")
print("Z position is IRRELEVANT\n")

W, D, H = 1000, 1000, 1000
bin = Container(W, D, H)

# Set constraint: item 9 (heavy) cannot overlap with item 7 (light) in XY
bin.set_constraints(relative_pos={0: [(7, 9)]})

print("Constraint: Item 9 (heavy) cannot overlap XY footprint of item 7 (light)")
print(f"Internal format: {bin.relative_pos}\n")

# Place light item 7 at ground
light_ems = EMS(100, 100, 0, 100, 100, 100)
bin.place_at_ems(light_ems, (100, 100, 100), weight=50, item_id=7)
light = bin.placed[-1]

print(f"Light item 7 placed:")
print(f"  Position: ({light.x}, {light.y}, {light.z})")
print(f"  Size: ({light.w}, {light.d}, {light.h})")
print(f"  XY footprint: X=[{light.x}, {light.x + light.w}], Y=[{light.y}, {light.y + light.d}]")
print(f"  Z range: [{light.z}, {light.z + light.h}]\n")

# Test 1: Heavy ABOVE light (XY overlap) - should REJECT
print("="*80)
print("TEST 1: Heavy item ABOVE light (XY overlap)")
print("="*80)
heavy_pos_above = (150, 150, 500)  # High up, XY overlaps with (100-200, 100-200)
heavy_size = (50, 50, 100)
allowed = bin.check_relative_positioning(9, heavy_pos_above, heavy_size)
print(f"Heavy position: {heavy_pos_above}")
print(f"Heavy XY footprint: X=[150, 200], Y=[150, 200]")
print(f"XY overlap with light: YES")
print(f"Heavy is above light: YES (Z=500 vs Z=0-100)")
print(f"Result: {'ALLOWED' if allowed else 'REJECTED'}")
print(f"Expected: REJECTED (XY overlap)")
if not allowed:
    print("✓ CORRECT\n")
else:
    print("✗ WRONG - should reject XY overlap!\n")

# Test 2: Heavy BELOW light (XY overlap) - should REJECT
print("="*80)
print("TEST 2: Heavy item BELOW light (XY overlap)")
print("="*80)

# Need to place light higher first - create new bin
bin2 = Container(W, D, H)
bin2.set_constraints(relative_pos={0: [(7, 9)]})

# Place supports to elevate light
support_ems = EMS(100, 100, 0, 100, 100, 500)
bin2.place_at_ems(support_ems, (100, 100, 500), weight=100, item_id=99)

# Place light on top of support
light_ems2 = EMS(100, 100, 500, 100, 100, 100)
bin2.place_at_ems(light_ems2, (100, 100, 100), weight=50, item_id=7)
light2 = bin2.placed[-1]

print(f"Light item 7 placed at:")
print(f"  Position: ({light2.x}, {light2.y}, {light2.z})")
print(f"  XY footprint: X=[{light2.x}, {light2.x + light2.w}], Y=[{light2.y}, {light2.y + light2.d}]")
print(f"  Z range: [{light2.z}, {light2.z + light2.h}]")

heavy_pos_below = (150, 150, 0)  # Ground level, XY overlaps
allowed2 = bin2.check_relative_positioning(9, heavy_pos_below, heavy_size)
print(f"\nHeavy position: {heavy_pos_below}")
print(f"Heavy XY footprint: X=[150, 200], Y=[150, 200]")
print(f"XY overlap with light: YES")
print(f"Heavy is below light: YES (Z=0 vs Z={light2.z}-{light2.z + light2.h})")
print(f"Result: {'ALLOWED' if allowed2 else 'REJECTED'}")
print(f"Expected: REJECTED (XY overlap)")
if not allowed2:
    print("✓ CORRECT\n")
else:
    print("✗ WRONG - should reject XY overlap!\n")

# Test 3: Heavy at same Z (XY overlap) - should REJECT
print("="*80)
print("TEST 3: Heavy item at SAME Z (XY overlap)")
print("="*80)
heavy_pos_same = (150, 150, 50)  # Same Z range, XY overlaps
allowed3 = bin.check_relative_positioning(9, heavy_pos_same, heavy_size)
print(f"Heavy position: {heavy_pos_same}")
print(f"Heavy XY footprint: X=[150, 200], Y=[150, 200]")
print(f"XY overlap with light: YES")
print(f"Heavy Z overlaps with light: YES (Z=50-150 vs Z=0-100)")
print(f"Result: {'ALLOWED' if allowed3 else 'REJECTED'}")
print(f"Expected: REJECTED (XY overlap)")
if not allowed3:
    print("✓ CORRECT\n")
else:
    print("✗ WRONG - should reject XY overlap!\n")

# Test 4: No XY overlap - should ALLOW (regardless of Z)
print("="*80)
print("TEST 4: No XY overlap (different XY position)")
print("="*80)
heavy_pos_no_overlap = (500, 500, 50)  # Different XY, same Z range
allowed4 = bin.check_relative_positioning(9, heavy_pos_no_overlap, heavy_size)
print(f"Heavy position: {heavy_pos_no_overlap}")
print(f"Heavy XY footprint: X=[500, 550], Y=[500, 550]")
print(f"Light XY footprint: X=[{light.x}, {light.x + light.w}], Y=[{light.y}, {light.y + light.d}]")
print(f"XY overlap with light: NO")
print(f"Result: {'ALLOWED' if allowed4 else 'REJECTED'}")
print(f"Expected: ALLOWED (no XY overlap)")
if allowed4:
    print("✓ CORRECT\n")
else:
    print("✗ WRONG - should allow when no XY overlap!\n")

# Summary
print("="*80)
print("SUMMARY")
print("="*80)
all_correct = (not allowed and not allowed2 and not allowed3 and allowed4)
if all_correct:
    print("✓ ALL TESTS PASSED")
    print("\nConstraint correctly enforced:")
    print("  - Rejects ANY XY overlap (regardless of Z position)")
    print("  - Allows placements with no XY overlap")
else:
    print("✗ SOME TESTS FAILED")
    print("\nConstraint is NOT working correctly!")
print("="*80)
