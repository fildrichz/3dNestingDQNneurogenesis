"""
Test that relative positioning constraint correctly handles gravity.

The bug: Old code rejected placements where heavy item ENDS UP below light item.
The fix: Now checks final position after gravity, not initial EMS position.
"""

import sys
sys.path.insert(0, 'main')

from nesting.packing_core_enhanced import Container

def test_heavy_below_light():
    """
    Test: Heavy item placed below light item should be ALLOWED.

    Scenario:
    - Place support at (0, 0) to elevate light without blocking (150, 150)
    - Place light on support at overlapping position
    - Place heavy at ground level with XY overlap with light
    - Heavy is below light -> ALLOWED
    """
    print("="*80)
    print("TEST 1: Heavy Item Below Light Item (Should ALLOW)")
    print("="*80)

    W, D, H = 1000, 1000, 1000
    bin = Container(W, D, H)

    # Set up constraint: item 9 (heavy) cannot be on top of item 7 (light)
    # Format: {grouping: [(light, heavy), ...]}
    bin.set_constraints(relative_pos={0: [(7, 9)]})

    print(f"\nConstraint: Item 9 (heavy) cannot be on top of Item 7 (light)")
    print(f"Relative pos dict: {bin.relative_pos}")

    # Place support at (0, 0) with size (50, 50, 200) - doesn't block (150, 150)
    from nesting.packing_core_enhanced import EMS
    support_ems = EMS(0, 0, 0, 50, 50, 200)
    bin.place_at_ems(support_ems, (50, 50, 200), weight=100, item_id=98)
    support1 = bin.placed[-1]

    # Place another support at (0, 100) to create platform for light
    support_ems2 = EMS(0, 100, 0, 50, 50, 200)
    bin.place_at_ems(support_ems2, (50, 50, 200), weight=100, item_id=97)

    # Place another support at (100, 0)
    support_ems3 = EMS(100, 0, 0, 50, 50, 200)
    bin.place_at_ems(support_ems3, (50, 50, 200), weight=100, item_id=96)

    # Place another support at (100, 100)
    support_ems4 = EMS(100, 100, 0, 50, 50, 200)
    bin.place_at_ems(support_ems4, (50, 50, 200), weight=100, item_id=95)

    print(f"\nPlaced 4 support pillars at corners to elevate to Z=200")

    # Place light item at (25, 25, 200) - sits on the supports
    light_ems = EMS(25, 25, 200, 100, 100, 100)
    bin.place_at_ems(light_ems, (100, 100, 100), weight=50, item_id=7)
    light = bin.placed[-1]

    print(f"\nPlaced light item 7:")
    print(f"  Position: ({light.x}, {light.y}, {light.z})")
    print(f"  Size: ({light.w}, {light.d}, {light.h})")
    print(f"  Occupies Z: {light.z} to {light.z + light.h}")

    # Now try to place heavy at ground level with XY overlap
    # Heavy at (50, 50, 0) overlaps with light at (25, 25, 200)
    heavy_pos = (50, 50, 0)
    heavy_size = (50, 50, 100)

    print(f"\nAttempt to place heavy item 9:")
    print(f"  Initial position: {heavy_pos}")
    print(f"  Size: {heavy_size}")

    # Check XY overlap manually
    heavy_x_range = (50, 100)
    heavy_y_range = (50, 100)
    light_x_range = (light.x, light.x + light.w)
    light_y_range = (light.y, light.y + light.d)

    x_overlap = not (heavy_x_range[1] <= light_x_range[0] or light_x_range[1] <= heavy_x_range[0])
    y_overlap = not (heavy_y_range[1] <= light_y_range[0] or light_y_range[1] <= heavy_y_range[0])

    print(f"  Heavy XY: {heavy_x_range} x {heavy_y_range}")
    print(f"  Light XY: {light_x_range} x {light_y_range}")
    print(f"  XY overlaps: {x_overlap and y_overlap}")

    # Check relative positioning
    allowed = bin.check_relative_positioning(item_id=9, ep=heavy_pos, size=heavy_size)

    # Verify what gravity does
    final_z = bin.apply_gravity(heavy_pos[0], heavy_pos[1], heavy_pos[2], *heavy_size)
    print(f"  After gravity: Z = {final_z}")
    print(f"  Heavy occupies Z: {final_z} to {final_z + heavy_size[2]}")
    print(f"  Light occupies Z: {light.z} to {light.z + light.h}")

    if final_z + heavy_size[2] <= light.z:
        print(f"  Heavy is BELOW light")
        expected = True
    else:
        print(f"  Heavy would be on/above light")
        expected = False

    if allowed == expected:
        print(f"\n  ✓ PASS: Correct behavior (allowed={allowed})")
        return True
    else:
        print(f"\n  ✗ FAIL: Wrong behavior (allowed={allowed}, expected={expected})")
        return False


def test_heavy_above_light():
    """
    Test: Heavy item placed above light item should be REJECTED.

    Scenario:
    - Light item 7 at Z=0 (ground)
    - Try to place heavy item 9 at high position
    - Gravity drops heavy to Z=100 (on top of light)
    - Should be REJECTED (violates constraint)
    """
    print("\n" + "="*80)
    print("TEST 2: Heavy Item Above Light Item (Should REJECT)")
    print("="*80)

    W, D, H = 1000, 1000, 1000
    bin = Container(W, D, H)

    # Set up constraint: item 9 (heavy) cannot be on top of item 7 (light)
    # Format: {grouping: [(light, heavy), ...]}
    bin.set_constraints(relative_pos={0: [(7, 9)]})

    print(f"\nConstraint: Item 9 (heavy) cannot be on top of Item 7 (light)")

    # Place light item 7 at ground level
    light_pos = (100, 100, 0)
    light_size = (100, 100, 100)

    from nesting.packing_core_enhanced import EMS
    temp_ems = EMS(light_pos[0], light_pos[1], light_pos[2], light_size[0], light_size[1], light_size[2])
    bin.place_at_ems(temp_ems, light_size, weight=50, item_id=7)
    light = bin.placed[-1]

    print(f"\nPlaced light item 7:")
    print(f"  Position: ({light.x}, {light.y}, {light.z})")
    print(f"  Size: ({light.w}, {light.d}, {light.h})")
    print(f"  Occupies Z: {light.z} to {light.z + light.h}")

    # Try to place heavy item 9 at high position (XY overlaps)
    # Gravity will drop it onto the light item
    heavy_pos = (100, 100, 500)  # High up, XY overlaps
    heavy_size = (100, 100, 100)

    print(f"\nAttempt to place heavy item 9:")
    print(f"  Initial position: {heavy_pos}")
    print(f"  Size: {heavy_size}")
    print(f"  XY overlaps with light item 7: YES")

    # Check relative positioning
    allowed = bin.check_relative_positioning(item_id=9, ep=heavy_pos, size=heavy_size)

    # Verify what gravity does
    final_z = bin.apply_gravity(heavy_pos[0], heavy_pos[1], heavy_pos[2], *heavy_size)
    print(f"  After gravity: Z = {final_z}")
    print(f"  Heavy occupies Z: {final_z} to {final_z + heavy_size[2]}")
    print(f"  Light occupies Z: {light.z} to {light.z + light.h}")

    if not allowed:
        print(f"\n  ✓ PASS: Placement REJECTED (heavy would be on top of light)")
        return True
    else:
        print(f"\n  ✗ FAIL: Placement ALLOWED (bug: should reject heavy on top of light!)")
        return False


def test_no_xy_overlap():
    """
    Test: No XY overlap = always allowed regardless of Z.

    Scenario:
    - Light item 7 at (100, 100, Z)
    - Heavy item 9 at (500, 500, Z) - no XY overlap
    - Should always be ALLOWED
    """
    print("\n" + "="*80)
    print("TEST 3: No XY Overlap (Should ALLOW)")
    print("="*80)

    W, D, H = 1000, 1000, 1000
    bin = Container(W, D, H)

    bin.set_constraints(relative_pos={0: [(7, 9)]})

    print(f"\nConstraint: Item 9 (heavy) cannot be on top of Item 7 (light)")

    # Place light item 7
    light_pos = (100, 100, 0)
    light_size = (100, 100, 100)

    from nesting.packing_core_enhanced import EMS
    temp_ems = EMS(light_pos[0], light_pos[1], light_pos[2], light_size[0], light_size[1], light_size[2])
    bin.place_at_ems(temp_ems, light_size, weight=50, item_id=7)

    print(f"\nPlaced light item 7:")
    print(f"  Position: {light_pos}")
    print(f"  Size: {light_size}")

    # Place heavy item 9 at different XY position
    heavy_pos = (500, 500, 0)  # No XY overlap
    heavy_size = (100, 100, 100)

    print(f"\nAttempt to place heavy item 9:")
    print(f"  Position: {heavy_pos}")
    print(f"  Size: {heavy_size}")
    print(f"  XY overlaps with light item 7: NO")

    allowed = bin.check_relative_positioning(item_id=9, ep=heavy_pos, size=heavy_size)

    if allowed:
        print(f"\n  ✓ PASS: Placement ALLOWED (no XY overlap)")
        return True
    else:
        print(f"\n  ✗ FAIL: Placement REJECTED (no XY overlap, should be allowed!)")
        return False


if __name__ == "__main__":
    print("\n" + "█"*80)
    print("RELATIVE POSITIONING FIX VERIFICATION")
    print("█"*80)
    print("\nBug: Old code rejected placements where heavy item ends up BELOW light item")
    print("Fix: Now checks final Z position after gravity")

    results = []
    results.append(test_heavy_below_light())
    results.append(test_heavy_above_light())
    results.append(test_no_xy_overlap())

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if all(results):
        print("\n✓ ALL TESTS PASSED")
        print("\nRelative positioning now correctly:")
        print("  - Allows heavy items placed BELOW light items")
        print("  - Rejects heavy items placed ON TOP OF light items")
        print("  - Considers final position after gravity, not initial EMS position")
    else:
        print("\n✗ SOME TESTS FAILED")
        print("\nFailed tests indicate the fix may not be working correctly")

    print("="*80)
