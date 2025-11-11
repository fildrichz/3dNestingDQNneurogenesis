"""
Test script to demonstrate constraint checking and color legend functionality.

This script tests:
1. Relative positioning constraints (heavy items cannot be on top of light items)
2. Incompatibility constraints
3. Affinity constraints
4. Color legend for visualization
"""

import sys
sys.path.insert(0, 'main')

from nesting.packing_core_enhanced import Container
import numpy as np

def test_relative_positioning():
    """Test that relative positioning prevents heavy items from being on top of light items."""
    print("\n" + "="*80)
    print("TEST 1: RELATIVE POSITIONING CONSTRAINT")
    print("="*80)

    # Create container
    container = Container(100, 100, 100)

    # Set constraint: Item 1 (heavy) cannot be placed on top of Item 7 (light)
    relative_pos = {1: [(7, 0), (7, 1)]}
    container.set_constraints(relative_pos=relative_pos)

    print("\nConstraint: Item 1 cannot be placed on top of Item 7")

    # First, place item 7 (light) at bottom
    print("\n1. Placing Item 7 (light) at (10, 10, 0) with size (20, 20, 10)")
    success = container.place_at_ems(
        container.ems_list[0],
        size=(20, 20, 10),
        weight=5,
        item_id=7
    )
    print(f"   Result: {'SUCCESS' if success else 'FAILED'}")

    # Try to place item 1 (heavy) on top of item 7 - should FAIL
    print("\n2. Trying to place Item 1 (heavy) at (10, 10, 10) - directly on top of Item 7")
    print("   This should FAIL due to relative positioning constraint")

    # Find EMS at position (10, 10, 10) or close to it
    # After placing first box, we should have EMS that overlaps with it in XY
    found_ems = None
    for ems in container.ems_list:
        # Check if this EMS is at or above the light box and overlaps in XY
        if ems.x <= 15 and ems.x + ems.w >= 15 and ems.y <= 15 and ems.y + ems.h >= 15:
            if ems.z >= 10:  # At or above the light box
                found_ems = ems
                break

    if found_ems:
        success = container.place_at_ems(
            found_ems,
            size=(15, 15, 10),
            weight=10,
            item_id=1
        )
        print(f"   Result: {'FAILED (Expected!)' if not success else 'SUCCESS (Unexpected!)'}")

        if not success:
            print("   ✓ Constraint working correctly - heavy item blocked from being on top of light item")
        else:
            print("   ✗ ERROR: Constraint failed - heavy item was placed on top of light item!")
    else:
        print("   (Could not find suitable EMS for test)")

    # Try to place item 1 (heavy) beside item 7 - should SUCCEED
    print("\n3. Placing Item 1 (heavy) at (50, 50, 0) - beside Item 7, not on top")
    print("   This should SUCCEED")

    # Find EMS that doesn't overlap with item 7 in XY
    found_ems = None
    for ems in container.ems_list:
        if ems.x >= 40 and ems.y >= 40 and ems.z == 0:
            found_ems = ems
            break

    if found_ems:
        success = container.place_at_ems(
            found_ems,
            size=(15, 15, 10),
            weight=10,
            item_id=1
        )
        print(f"   Result: {'SUCCESS (Expected!)' if success else 'FAILED (Unexpected!)'}")

        if success:
            print("   ✓ Heavy item can be placed beside light item")
        else:
            print("   ✗ ERROR: Should allow placement beside (not on top)")

    print("\n" + "="*80)
    return container


def test_incompatibility():
    """Test incompatibility constraints."""
    print("\n" + "="*80)
    print("TEST 2: INCOMPATIBILITY CONSTRAINT")
    print("="*80)

    container = Container(100, 100, 100)

    # Items 4 and 7 cannot be in same bin
    incompatibilities = [(4, 7)]
    container.set_constraints(incompatibilities=incompatibilities)

    print("\nConstraint: Items 4 and 7 cannot be in same bin")

    # Place item 4
    print("\n1. Placing Item 4 at (10, 10, 0)")
    success = container.place_at_ems(
        container.ems_list[0],
        size=(20, 20, 10),
        weight=5,
        item_id=4
    )
    print(f"   Result: {'SUCCESS' if success else 'FAILED'}")

    # Try to place item 7 - should FAIL
    print("\n2. Trying to place Item 7 in same bin - should FAIL")
    success = container.place_at_ems(
        container.ems_list[0],
        size=(20, 20, 10),
        weight=5,
        item_id=7
    )
    print(f"   Result: {'FAILED (Expected!)' if not success else 'SUCCESS (Unexpected!)'}")

    if not success:
        print("   ✓ Incompatibility constraint working correctly")
    else:
        print("   ✗ ERROR: Incompatible items were placed in same bin!")

    # Try to place item 3 - should SUCCEED
    print("\n3. Placing Item 3 (no incompatibility) - should SUCCEED")
    success = container.place_at_ems(
        container.ems_list[0],
        size=(20, 20, 10),
        weight=5,
        item_id=3
    )
    print(f"   Result: {'SUCCESS (Expected!)' if success else 'FAILED (Unexpected!)'}")

    print("\n" + "="*80)
    return container


def test_color_legend():
    """Test color legend functionality."""
    print("\n" + "="*80)
    print("TEST 3: COLOR LEGEND AND CONSTRAINT INFO")
    print("="*80)

    container = Container(200, 200, 100)

    # Set various constraints
    container.set_constraints(
        incompatibilities=[(4, 7), (2, 5)],
        positive_affinities=[(1, 3)],
        relative_pos={6: [(1, 0), (7, 0)]},
    )

    # Place several items
    print("\nPlacing various items...")
    items_to_place = [
        (1, (30, 30, 20), (10, 10, 0)),
        (3, (30, 30, 20), (50, 10, 0)),
        (6, (25, 25, 15), (100, 10, 0)),
        (7, (20, 20, 10), (150, 10, 0)),
        (1, (30, 30, 20), (10, 50, 0)),  # Another instance of item 1
    ]

    for item_id, size, position in items_to_place:
        # Find EMS near desired position
        found_ems = None
        for ems in container.ems_list:
            if abs(ems.x - position[0]) < 5 and abs(ems.y - position[1]) < 5 and ems.z == position[2]:
                found_ems = ems
                break

        if found_ems:
            success = container.place_at_ems(found_ems, size, weight=10, item_id=item_id)
            print(f"  Item {item_id}: {'Placed' if success else 'Failed'}")

    # Print constraint information
    container.print_constraint_info()

    # Print color legend
    container.print_item_legend()

    print("Note: You can call bin.plot3d() to visualize the packing with these colors")
    print("="*80)
    return container


if __name__ == "__main__":
    # Run all tests
    container1 = test_relative_positioning()
    container2 = test_incompatibility()
    container3 = test_color_legend()

    print("\n" + "="*80)
    print("ALL TESTS COMPLETED")
    print("="*80)
    print("\nSummary of new features:")
    print("  1. Fixed relative positioning constraint (removed off-by-one error)")
    print("  2. Added bin.print_item_legend() - shows item ID to color mapping")
    print("  3. Added bin.print_constraint_info() - shows all active constraints")
    print("  4. Added bin.get_item_color_mapping() - programmatic access to color map")
    print("\nThese features help verify that constraints are working correctly!")
    print("="*80 + "\n")
