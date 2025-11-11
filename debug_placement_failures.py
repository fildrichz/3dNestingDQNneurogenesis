"""
Debug why placements are failing with new gravity implementation.
"""

import sys
sys.path.insert(0, 'main')

from nesting.packing_core_enhanced import Container
from nesting.dataset_loader import load_problem

def test_first_box_placement():
    """Test placing the very first box - this should NEVER fail."""
    print("="*80)
    print("TEST 1: FIRST BOX PLACEMENT")
    print("="*80)

    container = Container(1000, 1000, 1000)

    print(f"\nInitial state:")
    print(f"  EMS count: {len(container.ems_list)}")
    print(f"  Placed boxes: {len(container.placed)}")
    print(f"  First EMS: ({container.ems_list[0].x}, {container.ems_list[0].y}, {container.ems_list[0].z}), "
          f"size ({container.ems_list[0].w}, {container.ems_list[0].d}, {container.ems_list[0].h})")

    # Try to place first box
    size = (100, 100, 100)
    print(f"\nAttempting to place box with size {size}...")

    success = container.place_at_ems(
        container.ems_list[0],
        size=size,
        weight=10,
        item_id=1
    )

    print(f"Result: {'SUCCESS' if success else 'FAILED'}")

    if success:
        box = container.placed[0]
        print(f"  Placed at: ({box.x}, {box.y}, {box.z})")
        print(f"  Expected z=0 (ground)")
        assert box.z == 0, f"First box should be at ground, but is at z={box.z}"
        print("  ✓ First box correctly at ground level")
    else:
        print("  ✗ CRITICAL BUG: First box placement failed!")
        print("  This should NEVER happen!")
        return False

    return True


def test_with_constraints():
    """Test with actual problem constraints."""
    print("\n" + "="*80)
    print("TEST 2: PLACEMENT WITH CONSTRAINTS")
    print("="*80)

    # Load real problem
    problem = load_problem("main/nesting/inputData/Benchmark dataset and instance generator for Real-World 3dBPP/Input/3dBPP_4.txt")

    W, D, H = problem.bin_dimensions
    print(f"\nProblem: 3dBPP_4")
    print(f"  Container: {W}×{D}×{H}")
    print(f"  Max weight: {problem.max_weight}")
    print(f"  Items: {len(problem.items)} types")

    container = Container(W, D, H, max_weight=problem.max_weight)
    container.set_constraints(
        incompatibilities=problem.incompatibilities,
        positive_affinities=problem.positive_affinities,
        relative_pos=problem.relative_pos
    )

    container.print_constraint_info()

    # Try placing first few items
    print("\nAttempting to place first 5 items...")

    items = [(item.length, item.width, item.height, item.weight, i)
             for i, item in enumerate(problem.items)
             for _ in range(item.quantity)]

    placement_attempts = 0
    successful_placements = 0
    failed_by_reason = {
        'no_ems': 0,
        'ems_fit': 0,
        'container_fit': 0,
        'weight': 0,
        'incompatibility': 0,
        'relative_pos': 0,
        'unknown': 0
    }

    for i, (w, d, h, weight, item_id) in enumerate(items[:10]):
        print(f"\n  Item {i}: id={item_id}, size=({w}, {d}, {h}), weight={weight}")

        # Try all EMS
        placed = False
        for ems_idx, ems in enumerate(container.ems_list[:5]):  # Try first 5 EMS
            placement_attempts += 1

            # Check each rotation
            for rot_idx, (rw, rd, rh) in enumerate([(w,d,h), (w,h,d), (d,w,h), (d,h,w), (h,w,d), (h,d,w)]):
                size = (rw, rd, rh)

                # Manual step-by-step checking
                x, y, z = ems.x, ems.y, ems.z

                # Check 1: EMS fit
                if not (rw <= ems.w and rd <= ems.d and rh <= ems.h):
                    continue

                # Check 2: Container fit
                if not (x + rw <= W and y + rd <= D and z + rh <= H):
                    failed_by_reason['container_fit'] += 1
                    continue

                # Check 3: Weight
                if problem.max_weight and container.current_weight + weight > problem.max_weight:
                    failed_by_reason['weight'] += 1
                    continue

                # Check 4: Incompatibility
                if not container.check_incompatibility(item_id):
                    failed_by_reason['incompatibility'] += 1
                    continue

                # Apply gravity
                final_z = container.apply_gravity(x, y, z, rw, rd, rh)

                # Check 5: Relative positioning at final position
                if not container.check_relative_positioning(item_id, (x, y, final_z), size):
                    failed_by_reason['relative_pos'] += 1
                    print(f"    ✗ Rotation {rot_idx}: BLOCKED by relative positioning constraint")
                    print(f"      After gravity: z={z} → {final_z}")

                    # Show what's below
                    for other in container.placed:
                        ox_overlap = not (x + rw <= other.x or other.x + other.w <= x)
                        oy_overlap = not (y + rd <= other.y or other.y + other.d <= y)
                        if ox_overlap and oy_overlap:
                            print(f"      Box below: item_id={other.item_id}, z={other.z} to {other.z+other.h}")
                    continue

                # Try placement
                success = container.place_at_ems(ems, size, weight=weight, item_id=item_id)

                if success:
                    placed = True
                    successful_placements += 1
                    box = container.placed[-1]
                    print(f"    ✓ Placed at ({box.x}, {box.y}, {box.z})")
                    break
                else:
                    failed_by_reason['unknown'] += 1

            if placed:
                break

        if not placed:
            print(f"    ✗ Failed to place item {i} anywhere")

    print("\n" + "="*80)
    print("PLACEMENT SUMMARY")
    print("="*80)
    print(f"Attempts: {placement_attempts}")
    print(f"Successful: {successful_placements}")
    print(f"Failure reasons:")
    for reason, count in failed_by_reason.items():
        if count > 0:
            print(f"  {reason}: {count}")

    if failed_by_reason['relative_pos'] > 0:
        print("\n⚠️  RELATIVE POSITIONING is blocking many placements!")
        print("This is correct behavior, but may indicate the constraint is too restrictive.")

    return successful_placements > 0


if __name__ == "__main__":
    # Test 1: First box (should always work)
    test1_pass = test_first_box_placement()

    if not test1_pass:
        print("\n❌ CRITICAL: First box placement failing - gravity bug!")
        sys.exit(1)

    # Test 2: With constraints
    test2_pass = test_with_constraints()

    if test2_pass:
        print("\n✓ Placement working, but may be constrained by problem rules")
    else:
        print("\n⚠️  No successful placements - investigate constraints")
