"""
Simple test for relative positioning bug fix.
"""

import sys
sys.path.insert(0, 'main')

from nesting.packing_core_enhanced import Container

def test_scenario():
    """
    Recreate the scenario from user's visualization:
    - Item 9 (heavy) placed on top of item 7 (light) - should be REJECTED
    """
    print("="*80)
    print("TEST: Reproducing User's Bug Report")
    print("="*80)

    W, D, H = 1100, 1100, 1100
    bin = Container(W, D, H)

    # Set up constraint from 3dBPP_4: items 0,1,9 are heavy
    # They cannot be placed on top of items 2-8 (light)
    constraints = {
        0: [(2,0), (3,0), (4,0), (5,0), (6,0), (7,0), (8,0)],
        1: [(2,1), (3,1), (4,1), (5,1), (6,1), (7,1), (8,1)],
        2: [(2,9), (3,9), (4,9), (5,9), (6,9), (7,9), (8,9)]
    }
    bin.set_constraints(relative_pos=constraints)

    print(f"\nConstraints set up:")
    print(f"  Heavy items: 0, 1, 9")
    print(f"  Light items: 2, 3, 4, 5, 6, 7, 8")
    print(f"  Heavy cannot be on top of light")
    print(f"\nInternal format: {bin.relative_pos}")

    # Place light item 7 at ground
    print(f"\nStep 1: Place light item 7 at ground")
    ems = bin.ems_list[0]
    success = bin.place_at_ems(ems, (100, 100, 100), weight=50, item_id=7)
    light_box = bin.placed[-1]
    print(f"  Placed at: ({light_box.x}, {light_box.y}, {light_box.z})")
    print(f"  Size: ({light_box.w}, {light_box.d}, {light_box.h})")
    print(f"  Occupies Z: {light_box.z} to {light_box.z + light_box.h}")

    # Try to place heavy item 9 in XY overlap position
    print(f"\nStep 2: Try to place heavy item 9 at overlapping XY")
    # Find an EMS that overlaps XY with light item 7
    ems_high = None
    for ems in bin.ems_list:
        if ems.z > 0:  # EMS above ground
            # Check if it overlaps XY with light item 7
            x_overlap = not (ems.x >= light_box.x + light_box.w or ems.x + ems.w <= light_box.x)
            y_overlap = not (ems.y >= light_box.y + light_box.d or ems.y + ems.d <= light_box.y)
            if x_overlap and y_overlap:
                ems_high = ems
                break

    if ems_high:
        print(f"  Found EMS at: ({ems_high.x}, {ems_high.y}, {ems_high.z})")
        print(f"  EMS size: ({ems_high.w}, {ems_high.d}, {ems_high.h})")

        # Check if this placement is allowed
        heavy_size = (100, 100, 100)
        allowed = bin.check_relative_positioning(
            item_id=9,
            ep=(ems_high.x, ems_high.y, ems_high.z),
            size=heavy_size
        )

        # Calculate where gravity would place it
        final_z = bin.apply_gravity(ems_high.x, ems_high.y, ems_high.z, *heavy_size)
        print(f"\n  Gravity analysis:")
        print(f"    Initial Z: {ems_high.z}")
        print(f"    After gravity: Z = {final_z}")
        print(f"    Heavy would occupy: Z {final_z} to {final_z + heavy_size[2]}")
        print(f"    Light occupies: Z {light_box.z} to {light_box.z + light_box.h}")

        if final_z >= light_box.z + light_box.h:
            print(f"    → Heavy would be ON TOP OF light (violation!)")
            expected_result = False
        else:
            print(f"    → Heavy would be BELOW light (allowed)")
            expected_result = True

        print(f"\n  check_relative_positioning result: {allowed}")
        print(f"  Expected result: {expected_result}")

        if allowed == expected_result:
            print(f"\n  ✓ CORRECT BEHAVIOR")
            return True
        else:
            print(f"\n  ✗ BUG DETECTED")
            return False
    else:
        print(f"  No high EMS found (unusual)")
        return False


def test_direct_violation():
    """
    Direct test: Place heavy directly on top of light.
    """
    print("\n" + "="*80)
    print("TEST: Direct Violation (Heavy Directly On Top)")
    print("="*80)

    W, D, H = 1000, 1000, 1000
    bin = Container(W, D, H)

    # Constraint: 9 cannot be on top of 7
    bin.set_constraints(relative_pos={0: [(7, 9)]})
    print(f"Constraint: Item 9 (heavy) cannot be on top of item 7 (light)")
    print(f"Internal format: {bin.relative_pos}")

    # Place light item 7 at ground
    ems = bin.ems_list[0]
    bin.place_at_ems(ems, (100, 100, 100), weight=50, item_id=7)
    light = bin.placed[-1]

    print(f"\nLight item 7 placed:")
    print(f"  Position: ({light.x}, {light.y}, {light.z})")
    print(f"  Top: Z = {light.z + light.h}")

    # Find EMS on top of light item
    ems_on_top = None
    for ems in bin.ems_list:
        if ems.z == light.z + light.h and ems.x == light.x and ems.y == light.y:
            ems_on_top = ems
            break

    if ems_on_top:
        print(f"\nEMS on top of light found at Z = {ems_on_top.z}")

        # Check if placing heavy here is allowed
        heavy_size = (100, 100, 100)
        allowed = bin.check_relative_positioning(9, (ems_on_top.x, ems_on_top.y, ems_on_top.z), heavy_size)

        final_z = bin.apply_gravity(ems_on_top.x, ems_on_top.y, ems_on_top.z, *heavy_size)

        print(f"Gravity would place heavy at Z = {final_z}")
        print(f"Light top at Z = {light.z + light.h}")
        print(f"check_relative_positioning: {allowed}")

        if not allowed:
            print(f"\n✓ CORRECT: Placement rejected (heavy on top of light)")
            return True
        else:
            print(f"\n✗ BUG: Placement allowed (should be rejected!)")
            return False
    else:
        print(f"No EMS on top found (checking all EMS)")
        for i, ems in enumerate(bin.ems_list):
            print(f"  EMS {i}: ({ems.x},{ems.y},{ems.z}) size ({ems.w},{ems.d},{ems.h})")
        return False


if __name__ == "__main__":
    print("\n" + "█"*80)
    print("RELATIVE POSITIONING BUG FIX VERIFICATION")
    print("█"*80)
    print("\nBug: Item 9 (heavy) was placed on top of item 7 (light)")
    print("Fix: check_relative_positioning now uses final Z after gravity\n")

    result1 = test_scenario()
    result2 = test_direct_violation()

    print("\n" + "="*80)
    if result1 and result2:
        print("✓ ALL TESTS PASSED - Bug is fixed!")
    else:
        print("✗ TESTS FAILED - Bug may still exist")
    print("="*80)
