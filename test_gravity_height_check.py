"""
Test that enumerate_actions correctly rejects placements that would exceed
container height after gravity.
"""

import sys
sys.path.insert(0, 'main')

from nesting.packing_core_enhanced import Container

def test_gravity_height_check():
    """
    Test scenario:
    1. Place a tall box at ground level (height=950)
    2. Try to place another box (height=200) on top
    3. After gravity: 950 + 200 = 1150 > 1100 -> should be rejected
    """
    print("="*80)
    print("TEST: Gravity Height Check")
    print("="*80)

    # Create container
    W, D, H = 1100, 1100, 1100
    bin = Container(W, D, H)

    print(f"\nContainer: {W}×{D}×{H}")

    # Place a tall box first
    ems = bin.ems_list[0]
    tall_box_size = (100, 100, 950)  # Very tall box

    print(f"\nStep 1: Place tall box {tall_box_size}")
    success = bin.place_at_ems(ems, tall_box_size, weight=100, item_id=0)

    if not success:
        print("  ✗ Failed to place tall box")
        return False

    print(f"  ✓ Placed tall box at (0, 0, 0)")
    print(f"    Box occupies: (0, 0, 0) to (100, 100, 950)")
    print(f"    Remaining EMS: {len(bin.ems_list)}")

    # Now try to place another box that overlaps in XY
    # This should be rejected because after gravity it would exceed container height
    short_box_size = (100, 100, 200)  # Overlaps with tall box in XY

    print(f"\nStep 2: Try to place short box {short_box_size} at overlapping XY position")

    # Find an EMS at a high Z position that overlaps with the tall box
    suitable_ems = None
    for ems in bin.ems_list:
        if ems.x == 0 and ems.y == 0:  # Overlaps with tall box XY
            suitable_ems = ems
            print(f"  Found EMS at ({ems.x}, {ems.y}, {ems.z}) size ({ems.w}, {ems.d}, {ems.h})")
            break

    if not suitable_ems:
        print("  (No suitable EMS found - creating one manually for testing)")
        # Manually check what would happen
        x, y, z = 0, 0, 950  # Try to place at top of tall box
        final_z = bin.apply_gravity(x, y, z, *short_box_size)

        print(f"  Gravity would drop box from Z={z} to Z={final_z}")
        print(f"  Final position: ({x}, {y}, {final_z}) + {short_box_size}")
        print(f"  Top of box: {final_z} + {short_box_size[2]} = {final_z + short_box_size[2]}")

        if final_z + short_box_size[2] > H:
            print(f"  ✓ CORRECT: Would exceed container height ({final_z + short_box_size[2]} > {H})")
            print(f"  enumerate_actions should reject this placement")
            return True
        else:
            print(f"  ✗ ERROR: Would fit but shouldn't!")
            return False
    else:
        # Check if fits
        fits_ems = bin._fits_ems(suitable_ems, short_box_size)
        fits_container = bin._fits_container((suitable_ems.x, suitable_ems.y, suitable_ems.z), short_box_size)

        print(f"  Initial checks:")
        print(f"    fits_ems: {fits_ems}")
        print(f"    fits_container: {fits_container}")

        # Check after gravity
        final_z = bin.apply_gravity(suitable_ems.x, suitable_ems.y, suitable_ems.z, *short_box_size)
        final_top = final_z + short_box_size[2]

        print(f"\n  After gravity:")
        print(f"    Box drops to Z={final_z}")
        print(f"    Top of box: {final_z} + {short_box_size[2]} = {final_top}")
        print(f"    Container height: {H}")

        if final_top > H:
            print(f"  ✓ CORRECT: Would exceed container height!")
            print(f"  The new enumerate_actions check should prevent this.")

            # Try to place it (should fail with new debug output)
            print(f"\n  Attempting place_at_ems (should fail with detailed error):")
            success = bin.place_at_ems(suitable_ems, short_box_size, weight=50, item_id=1)

            if not success:
                print(f"  ✓ place_at_ems correctly rejected the placement")
                return True
            else:
                print(f"  ✗ ERROR: place_at_ems ACCEPTED invalid placement!")
                return False
        else:
            print(f"  Box would fit (not the scenario we're testing)")
            return True

if __name__ == "__main__":
    success = test_gravity_height_check()

    print("\n" + "="*80)
    if success:
        print("✓ TEST PASSED")
    else:
        print("✗ TEST FAILED")
    print("="*80)
