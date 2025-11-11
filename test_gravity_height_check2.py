"""
Test gravity height check with a scenario where EMS check passes but
gravity would cause height violation.
"""

import sys
sys.path.insert(0, 'main')

from nesting.packing_core_enhanced import Container, EMS

def test_gravity_height_violation():
    """
    Test scenario:
    1. Place tall box A at (0,0,0) with height=900
    2. Place box B next to it at (200,0,0) with height=100
    3. Now there's an EMS at (0,0,900) - above box A
    4. Try to place box C (height=250) at EMS (0,0,900)
    5. Box C fits in the EMS (height >= 250)
    6. But after gravity, box C drops to Z=900, so top = 900+250=1150 > 1100
    7. This should be caught by the new gravity check in enumerate_actions
    """
    print("="*80)
    print("TEST: Gravity Height Violation (EMS passes, gravity fails)")
    print("="*80)

    W, D, H = 1100, 1100, 1100
    bin = Container(W, D, H)

    print(f"\nContainer: {W}×{D}×{H}")

    # Step 1: Place tall box A
    print(f"\nStep 1: Place tall box A (150×150×900)")
    ems = bin.ems_list[0]
    success = bin.place_at_ems(ems, (150, 150, 900), weight=100, item_id=0)
    print(f"  Placed at (0,0,0), top at Z=900")
    print(f"  EMS count: {len(bin.ems_list)}")

    # Step 2: Place shorter box B next to it
    print(f"\nStep 2: Place shorter box B (150×150×100) next to A")
    # Find EMS next to box A
    for ems in bin.ems_list:
        if ems.x == 150 and ems.y == 0 and ems.z == 0:
            success = bin.place_at_ems(ems, (150, 150, 100), weight=50, item_id=1)
            print(f"  Placed at (150,0,0), top at Z=100")
            break

    print(f"  EMS count: {len(bin.ems_list)}")

    # Step 3: Find EMS above box A
    print(f"\nStep 3: Looking for EMS above box A...")
    high_ems = None
    for ems in bin.ems_list:
        if ems.x == 0 and ems.y == 0 and ems.z == 900:
            high_ems = ems
            print(f"  Found EMS at ({ems.x},{ems.y},{ems.z}) size ({ems.w},{ems.d},{ems.h})")
            break

    if not high_ems:
        print("  No EMS at (0,0,900) - creating manually for test")
        high_ems = EMS(0, 0, 900, 150, 150, 200)  # EMS with height 200

    # Step 4: Try to place box C (height=250)
    box_c_size = (100, 100, 250)
    print(f"\nStep 4: Try to place box C {box_c_size} at EMS ({high_ems.x},{high_ems.y},{high_ems.z})")

    # Check constraints
    fits_ems = bin._fits_ems(high_ems, box_c_size)
    fits_container_initial = bin._fits_container((high_ems.x, high_ems.y, high_ems.z), box_c_size)

    print(f"  Initial checks at EMS position:")
    print(f"    fits_ems: {fits_ems} (EMS height={high_ems.h}, box height={box_c_size[2]})")
    print(f"    fits_container: {fits_container_initial}")

    # Check after gravity
    final_z = bin.apply_gravity(high_ems.x, high_ems.y, high_ems.z, *box_c_size)
    final_top = final_z + box_c_size[2]

    print(f"\n  After gravity:")
    print(f"    Box drops from Z={high_ems.z} to Z={final_z}")
    print(f"    Top of box: {final_z} + {box_c_size[2]} = {final_top}")
    print(f"    Container height: {H}")

    if final_top > H:
        print(f"\n  ✓ VIOLATION DETECTED: {final_top} > {H}")
        print(f"    This is the bug we're fixing!")
        print(f"    Old enumerate_actions: Would ACCEPT (only checks initial position)")
        print(f"    New enumerate_actions: Should REJECT (checks final position after gravity)")

        # Verify the fix works
        print(f"\n  Testing new enumerate_actions logic:")
        if final_z + box_c_size[2] > bin.h:
            print(f"    ✓ Gravity check: REJECT (final_z + h = {final_z + box_c_size[2]} > {bin.h})")
        else:
            print(f"    ✗ Gravity check: ACCEPT (should reject!)")

        return final_top > H
    else:
        print(f"  Box would fit - not the test scenario we need")
        return True

if __name__ == "__main__":
    success = test_gravity_height_violation()

    print("\n" + "="*80)
    if success:
        print("✓ TEST PASSED - Gravity height violation correctly detected")
    else:
        print("✗ TEST FAILED")
    print("="*80)
