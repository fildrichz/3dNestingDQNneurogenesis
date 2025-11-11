"""
Test action enumeration in MultiBinPackingEnv to diagnose why no actions are available.
"""

import sys
sys.path.insert(0, 'main')

# Import without torch dependencies
import numpy as np
from nesting.packing_core_enhanced import Container
from nesting.dataset_loader import load_problem

def test_action_enumeration():
    """Test if actions can be enumerated."""
    print("="*80)
    print("TESTING ACTION ENUMERATION")
    print("="*80)

    # Load problem
    problem_path = 'main/nesting/inputData/Benchmark dataset and instance generator for Real-World 3dBPP/Input/3dBPP_4.txt'
    problem = load_problem(problem_path)

    W, D, H = problem.bin_dimensions
    print(f"\nProblem: {W}×{D}×{H}")

    # Get items
    items = [(item.length, item.width, item.height, item.weight, i)
             for i, item in enumerate(problem.items)
             for _ in range(item.quantity)]

    print(f"Total items: {len(items)}")

    # Create single bin with constraints
    bin = Container(W, D, H, max_weight=problem.max_weight)
    bin.set_constraints(
        incompatibilities=problem.incompatibilities,
        positive_affinities=problem.positive_affinities,
        relative_pos=problem.relative_pos
    )

    print(f"\nInitial state:")
    print(f"  EMS count: {len(bin.ems_list)}")
    print(f"  Placed: {len(bin.placed)}")

    if bin.ems_list:
        ems = bin.ems_list[0]
        print(f"  First EMS: pos=({ems.x}, {ems.y}, {ems.z}), size=({ems.w}, {ems.d}, {ems.h})")

    # Manual action enumeration (simplified from enumerate_actions)
    print("\n" + "-"*80)
    print("ENUMERATING ACTIONS (first item only)...")
    print("-"*80)

    actions_found = []

    # Take first item
    if items:
        w, d, h, weight, item_id = items[0]
        print(f"\nItem 0: id={item_id}, size=({w}, {d}, {h}), weight={weight}")

        # Try all EMS
        for ems_idx, ems in enumerate(bin.ems_list[:5]):  # Check first 5 EMS
            print(f"\n  EMS {ems_idx}: pos=({ems.x}, {ems.y}, {ems.z}), size=({ems.w}, {ems.d}, {ems.h})")

            # Try all rotations
            rots = [(w,d,h), (w,h,d), (d,w,h), (d,h,w), (h,w,d), (h,d,w)]

            for rot_idx, size in enumerate(rots):
                rw, rd, rh = size

                # Check if fits in EMS
                if not (rw <= ems.w and rd <= ems.d and rh <= ems.h):
                    continue

                ep = (ems.x, ems.y, ems.z)

                # Check container fit
                if not bin._fits_container(ep, size):
                    print(f"    Rot {rot_idx} ({rw}×{rd}×{rh}): ✗ Container fit")
                    continue

                # Check weight
                if not bin.check_weight_constraint(weight):
                    print(f"    Rot {rot_idx} ({rw}×{rd}×{rh}): ✗ Weight")
                    continue

                # Check incompatibility
                if not bin.check_incompatibility(item_id):
                    print(f"    Rot {rot_idx} ({rw}×{rd}×{rh}): ✗ Incompatibility")
                    continue

                # Apply gravity
                final_z = bin.apply_gravity(ems.x, ems.y, ems.z, rw, rd, rh)

                # Check relative positioning at final position
                if not bin.check_relative_positioning(item_id, (ems.x, ems.y, final_z), size):
                    print(f"    Rot {rot_idx} ({rw}×{rd}×{rh}): ✗ Relative pos (z {ems.z}→{final_z})")
                    continue

                print(f"    Rot {rot_idx} ({rw}×{rd}×{rh}): ✓ VALID ACTION")
                actions_found.append((ems_idx, rot_idx, size))

    print("\n" + "="*80)
    print(f"RESULT: Found {len(actions_found)} valid actions for first item")
    print("="*80)

    if len(actions_found) == 0:
        print("\n❌ PROBLEM: No valid actions found!")
        print("\nThis explains why packing fails immediately.")
        print("\nPossible causes:")
        print("  1. All rotations blocked by constraints")
        print("  2. EMS dimensions too small")
        print("  3. Bug in constraint checking")

        # Try placing without constraints as sanity check
        print("\n" + "-"*80)
        print("SANITY CHECK: Trying without constraints...")
        print("-"*80)

        test_bin = Container(W, D, H)
        w, d, h, weight, item_id = items[0]

        success = test_bin.place_at_ems(
            test_bin.ems_list[0],
            size=(w, d, h),
            weight=weight,
            item_id=item_id
        )

        if success:
            print("✓ Placement works WITHOUT constraints")
            print("→ Problem is constraint checking, not basic placement")
        else:
            print("✗ Placement fails even WITHOUT constraints")
            print("→ Bug in basic placement logic!")
    else:
        print("\n✓ Actions available - environment should work")

    return len(actions_found) > 0


if __name__ == "__main__":
    test_action_enumeration()
