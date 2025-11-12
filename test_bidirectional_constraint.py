#!/usr/bin/env python3
"""
Test that relative positioning constraint works bidirectionally:
1. Placing heavy item -> check no light items with XY overlap
2. Placing light item -> check no heavy items with XY overlap
"""
import sys
sys.path.insert(0, 'main')

from nesting.packing_core_enhanced import Container, EMS

print("="*80)
print("BIDIRECTIONAL RELATIVE POSITIONING CONSTRAINT TEST")
print("="*80)

W, D, H = 1000, 1000, 1000

# Constraint: (light=7, heavy=9) means heavy 9 cannot overlap XY with light 7
constraint = {0: [(7, 9)]}

# ===========================================================================
# TEST 1: Place LIGHT first, then try to place HEAVY
# ===========================================================================
print("\nTEST 1: Place LIGHT item first, then try HEAVY at same XY")
print("="*80)

bin1 = Container(W, D, H)
bin1.set_constraints(relative_pos=constraint)

print(f"Constraint mappings:")
print(f"  Heavy->Light: {bin1.relative_pos}")
print(f"  Light->Heavy: {bin1.relative_pos_reverse}")

# Place light item 7 at (100, 100)
print(f"\n1. Placing LIGHT item 7 at (100, 100, 0)")
light_ems = EMS(100, 100, 0, 100, 100, 100)
result1 = bin1.place_at_ems(light_ems, (100, 100, 100), weight=50, item_id=7)
print(f"   Result: {'SUCCESS' if result1 else 'FAILED'}")
print(f"   Light item 7 XY footprint: X=[100, 200], Y=[100, 200]")

# Try to place heavy item 9 at overlapping XY position
print(f"\n2. Trying to place HEAVY item 9 at (150, 150, 0) - overlaps with light")
heavy_pos = (150, 150, 0)
allowed = bin1.check_relative_positioning(9, heavy_pos, (50, 50, 100))
print(f"   Heavy XY footprint: X=[150, 200], Y=[150, 200]")
print(f"   XY overlap: YES")
print(f"   check_relative_positioning: {allowed}")
print(f"   Expected: False (should reject)")

if not allowed:
    print("   ✓ CORRECT: Heavy placement rejected")
else:
    print("   ✗ WRONG: Should reject heavy on light!")

# ===========================================================================
# TEST 2: Place HEAVY first, then try to place LIGHT
# ===========================================================================
print("\n\nTEST 2: Place HEAVY item first, then try LIGHT at same XY")
print("="*80)

bin2 = Container(W, D, H)
bin2.set_constraints(relative_pos=constraint)

# Place heavy item 9 at (100, 100)
print(f"\n1. Placing HEAVY item 9 at (100, 100, 0)")
heavy_ems = EMS(100, 100, 0, 100, 100, 100)
result2 = bin2.place_at_ems(heavy_ems, (100, 100, 100), weight=100, item_id=9)
print(f"   Result: {'SUCCESS' if result2 else 'FAILED'}")
print(f"   Heavy item 9 XY footprint: X=[100, 200], Y=[100, 200]")

# Try to place light item 7 at overlapping XY position
print(f"\n2. Trying to place LIGHT item 7 at (150, 150, 0) - overlaps with heavy")
light_pos = (150, 150, 0)
allowed2 = bin2.check_relative_positioning(7, light_pos, (50, 50, 100))
print(f"   Light XY footprint: X=[150, 200], Y=[150, 200]")
print(f"   XY overlap: YES")
print(f"   check_relative_positioning: {allowed2}")
print(f"   Expected: False (should reject)")

if not allowed2:
    print("   ✓ CORRECT: Light placement rejected (bidirectional check works!)")
else:
    print("   ✗ WRONG: Should reject light under heavy!")

# ===========================================================================
# TEST 3: No overlap - both should be allowed
# ===========================================================================
print("\n\nTEST 3: No XY overlap - placements should be allowed")
print("="*80)

bin3 = Container(W, D, H)
bin3.set_constraints(relative_pos=constraint)

print(f"\n1. Placing LIGHT item 7 at (100, 100, 0)")
bin3.place_at_ems(EMS(100, 100, 0, 100, 100, 100), (100, 100, 100), weight=50, item_id=7)

print(f"2. Trying to place HEAVY item 9 at (500, 500, 0) - NO overlap")
no_overlap_allowed = bin3.check_relative_positioning(9, (500, 500, 0), (100, 100, 100))
print(f"   XY overlap: NO")
print(f"   check_relative_positioning: {no_overlap_allowed}")
print(f"   Expected: True (should allow)")

if no_overlap_allowed:
    print("   ✓ CORRECT: Placement allowed when no overlap")
else:
    print("   ✗ WRONG: Should allow when no overlap!")

# ===========================================================================
# SUMMARY
# ===========================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

all_pass = (not allowed and not allowed2 and no_overlap_allowed)

if all_pass:
    print("✓ ALL TESTS PASSED")
    print("\nBidirectional constraint working correctly:")
    print("  1. Heavy cannot be placed where light exists ✓")
    print("  2. Light cannot be placed where heavy exists ✓")
    print("  3. Non-overlapping placements allowed ✓")
else:
    print("✗ SOME TESTS FAILED")
    results = []
    if allowed:
        results.append("  - Heavy on light: FAILED")
    if allowed2:
        results.append("  - Light under heavy: FAILED")
    if not no_overlap_allowed:
        results.append("  - No overlap: FAILED")
    for r in results:
        print(r)

print("="*80)
