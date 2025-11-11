"""
Diagnose why placement fails at constraint re-check after gravity.
"""

import sys
sys.path.insert(0, 'main')

import numpy as np
from nesting.dataset_loader import load_problem
from nesting.packing_core_enhanced import Container

def test_placement_failure():
    """Test what fails during placement."""
    print("="*80)
    print("DIAGNOSING PLACEMENT FAILURE AT CONSTRAINT RE-CHECK")
    print("="*80)

    # Load problem
    problem_path = 'main/nesting/inputData/Benchmark dataset and instance generator for Real-World 3dBPP/Input/3dBPP_4.txt'
    problem = load_problem(problem_path)

    W, D, H = problem.bin_dimensions
    print(f"\nContainer: {W}×{D}×{H}")

    # Create container with constraints
    bin = Container(W, D, H)
    bin.set_constraints(
        incompatibilities=problem.incompatibilities,
        positive_affinities=problem.positive_affinities,
        center_of_mass=None,
        relative_pos=problem.relative_pos
    )

    print(f"Relative positioning constraints: {bin.relative_pos}")

    # Convert items
    items = []
    for i, item in enumerate(problem.items):
        for _ in range(item.quantity):
            items.append((item.length, item.width, item.height, item.weight, i))

    print(f"\nItems: {len(items)}")

    # Try placing items manually
    print("\n" + "="*80)
    print("PLACING ITEMS ONE BY ONE")
    print("="*80)

    for idx in range(min(5, len(items))):
        item = items[idx]
        w, d, h, weight, item_id = item

        print(f"\n--- Item {idx}: id={item_id}, size=({w},{d},{h}), weight={weight} ---")

        if not bin.ems_list:
            print("  ✗ No EMS available")
            break

        ems = bin.ems_list[0]
        print(f"  EMS: ({ems.x},{ems.y},{ems.z}) size ({ems.w},{ems.d},{ems.h})")

        # Try original orientation
        size = (w, d, h)
        ep = (ems.x, ems.y, ems.z)

        print(f"\n  Initial checks at EMS position {ep}:")
        fits_ems = bin._fits_ems(ems, size)
        fits_container = bin._fits_container(ep, size)
        weight_ok = bin.check_weight_constraint(weight)
        incomp_ok = bin.check_incompatibility(item_id)
        relpos_ok = bin.check_relative_positioning(item_id, ep, size)

        print(f"    fits_ems: {fits_ems}")
        print(f"    fits_container: {fits_container}")
        print(f"    weight: {weight_ok}")
        print(f"    incomp: {incomp_ok}")
        print(f"    relpos: {relpos_ok}")

        if not (fits_ems and fits_container and weight_ok and incomp_ok and relpos_ok):
            print("  ✗ Failed initial checks, trying next item...")
            continue

        # Apply gravity
        x, y, z = ep
        final_z = bin.apply_gravity(x, y, z, w, d, h)
        final_ep = (x, y, final_z)

        print(f"\n  After gravity: {ep} -> {final_ep} (dropped {z - final_z})")

        # Re-check at final position
        print(f"\n  Final checks at position {final_ep}:")
        final_fits_container = bin._fits_container(final_ep, size)
        final_relpos_ok = bin.check_relative_positioning(item_id, final_ep, size)

        print(f"    fits_container: {final_fits_container}")
        if not final_fits_container:
            print(f"      Container bounds: x+w={x}+{w}={x+w} <= {W}, y+d={y}+{d}={y+d} <= {D}, z+h={final_z}+{h}={final_z+h} <= {H}")

        print(f"    relpos: {final_relpos_ok}")
        if not final_relpos_ok:
            print(f"      Item {item_id} is heavy (cannot be on top of light items)")
            print(f"      Light items to avoid: {bin.relative_pos.get(item_id, [])}")
            print(f"      Items currently in bin: {bin.item_ids_in_bin}")

            # Check which light item is blocking
            if item_id in bin.relative_pos:
                light_items = bin.relative_pos[item_id]
                for light_id in light_items:
                    if light_id in bin.item_positions:
                        for light_box in bin.item_positions[light_id]:
                            x_overlap = not (x + w <= light_box.x or light_box.x + light_box.w <= x)
                            y_overlap = not (y + d <= light_box.y or light_box.y + light_box.d <= y)
                            if x_overlap and y_overlap:
                                print(f"      BLOCKING: Light item {light_id} at ({light_box.x},{light_box.y},{light_box.z}) size ({light_box.w},{light_box.d},{light_box.h})")
                                print(f"                XY overlap detected!")

        if final_fits_container and final_relpos_ok:
            print("  ✓ All checks passed, placing item...")
            success = bin.place_at_ems(ems, size, weight=weight, item_id=item_id)

            if success:
                print(f"  ✓ PLACED successfully at {final_ep}")
                print(f"    Bin now has {len(bin.placed)} items, {len(bin.ems_list)} EMS")
            else:
                print(f"  ✗ place_at_ems FAILED despite checks passing!")
                return False
        else:
            print(f"  ✗ Final checks failed")
            print(f"\n  This is the BUG! Initial checks passed but final checks failed.")
            print(f"  enumerate_actions would approve this but place_at_ems rejects it!")
            return False

    print("\n" + "="*80)
    print("✓ All placements successful")
    print("="*80)
    return True

if __name__ == "__main__":
    test_placement_failure()
