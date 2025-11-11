"""
Test gravity functionality to verify boxes aren't floating.
"""

import sys
sys.path.insert(0, 'main')

from nesting.packing_core_enhanced import Container

def test_gravity():
    """Test that gravity prevents floating boxes."""
    print("\n" + "="*80)
    print("TEST: GRAVITY FUNCTIONALITY")
    print("="*80)

    container = Container(200, 200, 100)

    # Test 1: Place first box at ground level
    print("\nTest 1: Place box at ground")
    print("  Placing 40x40x10 box...")
    success = container.place_at_ems(
        container.ems_list[0],  # Should be at (0, 0, 0)
        size=(40, 40, 10),
        weight=10,
        item_id=1
    )
    print(f"  Result: {'SUCCESS' if success else 'FAILED'}")

    if success and container.placed:
        box1 = container.placed[0]
        print(f"  Box position: ({box1.x}, {box1.y}, {box1.z})")
        print(f"  Expected z=0, Got z={box1.z}")
        assert box1.z == 0, f"Box should be at ground (z=0), but is at z={box1.z}"
        print("  ✓ Box correctly at ground level")

    # Test 2: Place second box - should stack on first box
    print("\nTest 2: Place box that should stack on first box")
    print("  Placing 30x30x10 box...")

    # Find EMS that overlaps with first box in XY
    target_ems = None
    for ems in container.ems_list:
        # Check if EMS overlaps with box1 in XY
        if ems.x < 40 and ems.y < 40:
            target_ems = ems
            print(f"  Using EMS at ({ems.x}, {ems.y}, {ems.z})")
            break

    if target_ems:
        success = container.place_at_ems(
            target_ems,
            size=(30, 30, 10),
            weight=5,
            item_id=2
        )
        print(f"  Result: {'SUCCESS' if success else 'FAILED'}")

        if success and len(container.placed) >= 2:
            box2 = container.placed[-1]
            print(f"  Box position: ({box2.x}, {box2.y}, {box2.z})")
            expected_z = 10  # Should be on top of first box
            print(f"  Expected z=10 (on top of first box), Got z={box2.z}")
            assert box2.z == expected_z, f"Box should be at z={expected_z}, but is at z={box2.z}"
            print("  ✓ Box correctly stacked on first box")
    else:
        print("  Could not find suitable EMS for test")

    # Test 3: Place box away from others - should drop to ground
    print("\nTest 3: Place box away from others (should drop to ground)")
    print("  Placing 30x30x10 box far from existing boxes...")

    # Find EMS that doesn't overlap with existing boxes
    target_ems = None
    for ems in container.ems_list:
        if ems.x >= 100 and ems.y >= 100:
            target_ems = ems
            print(f"  Using EMS at ({ems.x}, {ems.y}, {ems.z})")
            break

    if target_ems:
        success = container.place_at_ems(
            target_ems,
            size=(30, 30, 10),
            weight=5,
            item_id=3
        )
        print(f"  Result: {'SUCCESS' if success else 'FAILED'}")

        if success:
            box3 = container.placed[-1]
            print(f"  Box position: ({box3.x}, {box3.y}, {box3.z})")
            print(f"  Expected z=0 (ground), Got z={box3.z}")
            assert box3.z == 0, f"Box should drop to ground (z=0), but is at z={box3.z}"
            print("  ✓ Box correctly dropped to ground")
    else:
        print("  Could not find suitable EMS for test")

    # Test 4: Verify apply_gravity directly
    print("\nTest 4: Direct test of apply_gravity function")
    print("  Testing apply_gravity(100, 100, 50, 20, 20, 10)")
    print("  Should drop from z=50 to z=0 (no obstacles)")
    final_z = container.apply_gravity(100, 100, 50, 20, 20, 10)
    print(f"  Result: z={final_z}")
    assert final_z == 0, f"Should drop to z=0, but got z={final_z}"
    print("  ✓ apply_gravity correctly drops to ground")

    # Test 5: apply_gravity with obstacle
    print("\nTest 5: apply_gravity with obstacle below")
    # Place a box manually for testing
    container.placed.append(container.placed[0])  # Use first box as reference
    print(f"  Obstacle at z=0-10")
    print("  Testing apply_gravity(10, 10, 50, 20, 20, 10)")
    print("  Should drop from z=50 to z=10 (on top of obstacle)")
    final_z = container.apply_gravity(10, 10, 50, 20, 20, 10)
    print(f"  Result: z={final_z}")
    expected_z = 10
    assert final_z == expected_z, f"Should drop to z={expected_z}, but got z={final_z}"
    print(f"  ✓ apply_gravity correctly stops at z={expected_z}")

    # Print all box positions
    print("\n" + "="*80)
    print("FINAL BOX POSITIONS:")
    print("="*80)
    for i, box in enumerate(container.placed):
        print(f"  Box {i}: item_id={box.item_id}, pos=({box.x}, {box.y}, {box.z}), "
              f"size=({box.w}, {box.d}, {box.h}), z_top={box.z + box.h}")
    print("="*80)

    # Generate visualization
    print("\nGenerating visualization...")
    container.plot3d_filled(save_path="test_gravity.png", show=False)
    print("  ✓ Saved: test_gravity.png")
    print("\nPlease check test_gravity.png to visually verify no boxes are floating!")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_gravity()
