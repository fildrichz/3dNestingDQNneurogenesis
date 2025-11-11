"""
Debug why enumerate_actions is returning 0 actions.
"""

import sys
sys.path.insert(0, 'main')

import numpy as np
from nesting.packing_core_enhanced import Container
from nesting.dataset_loader import load_problem

def diagnose_enumerate_actions():
    """Diagnose why no actions are being found."""
    print("="*80)
    print("DIAGNOSING ENUMERATE_ACTIONS")
    print("="*80)

    # Load problem
    problem_path = 'main/nesting/inputData/Benchmark dataset and instance generator for Real-World 3dBPP/Input/3dBPP_4.txt'
    problem = load_problem(problem_path)

    W, D, H = problem.bin_dimensions
    items = [(item.length, item.width, item.height, item.weight, i)
             for i, item in enumerate(problem.items)
             for _ in range(item.quantity)]

    print(f"\nProblem: {W}×{D}×{H}, {len(items)} items")
    print(f"First item: {items[0]}")

    # Create bin with constraints
    bin = Container(W, D, H, max_weight=problem.max_weight)
    bin.set_constraints(
        incompatibilities=problem.incompatibilities,
        positive_affinities=problem.positive_affinities,
        relative_pos=problem.relative_pos
    )

    print(f"\nBin state:")
    print(f"  EMS count: {len(bin.ems_list)}")
    print(f"  Placed: {len(bin.placed)}")
    print(f"  item_positions: {bin.item_positions}")

    # Take first item
    w, d, h, weight, item_id = items[0]
    print(f"\nChecking item 0: id={item_id}, size=({w},{d},{h}), weight={weight}")

    # Check if it has constraints
    if item_id in bin.relative_pos:
        print(f"  Has constraint: cannot be on top of {bin.relative_pos[item_id]}")
    else:
        print(f"  No relative positioning constraint")

    # Try first EMS, first rotation
    ems = bin.ems_list[0]
    size = (w, d, h)
    ep = (ems.x, ems.y, ems.z)

    print(f"\nTrying EMS 0: pos=({ems.x}, {ems.y}, {ems.z}), size=({ems.w}, {ems.d}, {ems.h})")
    print(f"Placement position: {ep}, size: {size}")

    # Check each constraint step by step
    print("\nConstraint checks:")

    # 1. Fits EMS
    fits_ems = bin._fits_ems(ems, size)
    print(f"  1. _fits_ems: {fits_ems}")
    if not fits_ems:
        print(f"     Failed: {w} <= {ems.w}? {w <= ems.w}, {d} <= {ems.d}? {d <= ems.d}, {h} <= {ems.h}? {h <= ems.h}")

    # 2. Fits container
    fits_container = bin._fits_container(ep, size)
    print(f"  2. _fits_container: {fits_container}")

    # 3. Weight
    weight_ok = bin.check_weight_constraint(weight)
    print(f"  3. check_weight_constraint: {weight_ok}")

    # 4. Incompatibility
    incompat_ok = bin.check_incompatibility(item_id)
    print(f"  4. check_incompatibility: {incompat_ok}")

    # 5. Relative positioning
    rel_pos_ok = bin.check_relative_positioning(item_id, ep, size)
    print(f"  5. check_relative_positioning: {rel_pos_ok}")

    if not rel_pos_ok:
        print(f"     FAILED! Checking why...")
        if item_id in bin.relative_pos:
            light_items = bin.relative_pos[item_id]
            print(f"     Light items this cannot be on: {light_items}")
            for light_id in light_items:
                if light_id in bin.item_positions:
                    print(f"       Light item {light_id} IS in bin!")
                    for light_box in bin.item_positions[light_id]:
                        print(f"         Position: ({light_box.x}, {light_box.y}, {light_box.z})")
                else:
                    print(f"       Light item {light_id} not in bin (OK)")

    # Final result
    all_ok = fits_ems and fits_container and weight_ok and incompat_ok and rel_pos_ok
    print(f"\n  ALL CHECKS PASS: {all_ok}")

    if all_ok:
        print("\n✓ Action should be valid!")
        print("  If enumerate_actions still returns 0, there's a bug in enumerate_actions itself")
    else:
        print("\n✗ Action blocked by constraints")

    # Now actually try to enumerate actions using the bin directly
    print("\n" + "-"*80)
    print("TESTING ACTUAL ENUMERATE (simplified version):")
    print("-"*80)

    actions = []
    for ems_idx, ems in enumerate(bin.ems_list[:1]):  # Just first EMS
        for item_idx, item in enumerate([items[0]]):  # Just first item
            w, d, h, weight, item_id = item
            for rot_idx, size in enumerate([(w,d,h)]):  # Just first rotation
                ep = (ems.x, ems.y, ems.z)

                if (bin._fits_ems(ems, size) and
                    bin._fits_container(ep, size) and
                    bin.check_weight_constraint(weight) and
                    bin.check_incompatibility(item_id) and
                    bin.check_relative_positioning(item_id, ep, size)):

                    actions.append((0, item_idx, ems_idx, rot_idx, ems, size, weight, item_id))
                    print(f"  ✓ Action created!")

    print(f"\nActions found: {len(actions)}")

    if len(actions) == 0:
        print("\n❌ PROBLEM CONFIRMED: No actions being created!")
    else:
        print("\n✓ Actions being created correctly")


if __name__ == "__main__":
    diagnose_enumerate_actions()
