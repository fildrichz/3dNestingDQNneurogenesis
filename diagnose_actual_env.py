"""
Diagnose the actual MultiBinPackingEnv to see why enumerate_actions returns 0.
"""

import sys
sys.path.insert(0, 'main')

import numpy as np

# Import the actual environment
from packing_with_dqncore2_enhanced import MultiBinPackingEnv, load_problem_as_items
from nesting.dataset_loader import load_problem

def test_actual_environment():
    """Test the actual MultiBinPackingEnv."""
    print("="*80)
    print("DIAGNOSING ACTUAL MULTI-BIN PACKING ENVIRONMENT")
    print("="*80)

    # Load problem (Linux path)
    problem_path = 'main/nesting/inputData/Benchmark dataset and instance generator for Real-World 3dBPP/Input/3dBPP_4.txt'
    problem = load_problem(problem_path)
    items = load_problem_as_items(problem)

    W, D, H = problem.bin_dimensions

    print(f"\nProblem: {W}×{D}×{H}, {len(items)} items")
    print(f"Max bins: {problem.max_bins}")
    print(f"Max weight: {problem.max_weight}")

    # Create environment EXACTLY like in training
    env = MultiBinPackingEnv(
        W, D, H,
        items=items,
        max_actions=128,
        topk_eps=1000,
        seed=42,
        gamma=0.992,
        problem=problem
    )

    print(f"\nInitializing environment...")
    obs = env.reset(items=items.copy())

    print(f"\nEnvironment state after reset:")
    print(f"  Number of bins: {len(env.bins)}")
    print(f"  Items to pack: {len(env.items)}")
    print(f"  Observation shape: {obs.shape}")

    for i, bin in enumerate(env.bins):
        print(f"\n  Bin {i}:")
        print(f"    EMS count: {len(bin.ems_list)}")
        print(f"    Placed items: {len(bin.placed)}")
        if bin.ems_list:
            ems = bin.ems_list[0]
            print(f"    First EMS: pos=({ems.x}, {ems.y}, {ems.z}), size=({ems.w}, {ems.d}, {ems.h})")
        print(f"    Constraints:")
        print(f"      Incompatibilities: {len(bin.incompatibilities) if bin.incompatibilities else 0}")
        print(f"      Positive affinities: {len(bin.positive_affinities) if bin.positive_affinities else 0}")
        print(f"      Relative pos: {len(bin.relative_pos) if bin.relative_pos else 0}")
        if bin.relative_pos:
            print(f"      Relative pos dict: {bin.relative_pos}")

    # Try to enumerate actions
    print("\n" + "="*80)
    print("ENUMERATING ACTIONS...")
    print("="*80)

    actions = env.enumerate_actions()

    print(f"\nActions found: {len(actions)}")

    if len(actions) == 0:
        print("\n❌ NO ACTIONS FOUND!")
        print("\nDEBUGGING FIRST ITEM IN FIRST BIN:")

        bin = env.bins[0]
        item = env.items[0]
        w, d, h, weight, item_id = item

        print(f"\n  Item: id={item_id}, size=({w},{d},{h}), weight={weight}")
        print(f"  First EMS: {bin.ems_list[0] if bin.ems_list else 'NONE'}")

        if bin.ems_list:
            ems = bin.ems_list[0]
            print(f"    EMS: pos=({ems.x},{ems.y},{ems.z}), size=({ems.w},{ems.d},{ems.h})")

            # Try all rotations
            rots = [(w,d,h), (w,h,d), (d,w,h), (d,h,w), (h,w,d), (h,d,w)]

            print(f"\n  Testing rotations:")
            for rot_idx, size in enumerate(rots):
                rw, rd, rh = size
                ep = (ems.x, ems.y, ems.z)

                fits_ems = bin._fits_ems(ems, size)
                fits_container = bin._fits_container(ep, size)
                weight_ok = bin.check_weight_constraint(weight)
                incomp_ok = bin.check_incompatibility(item_id)
                relpos_ok = bin.check_relative_positioning(item_id, ep, size)
                affinity_ok = env.check_affinity_placement(item_id, 0)

                all_ok = fits_ems and fits_container and weight_ok and incomp_ok and relpos_ok and affinity_ok

                status = "✓" if all_ok else "✗"
                print(f"    Rot {rot_idx} ({rw}×{rd}×{rh}): {status}")
                if not all_ok:
                    print(f"      fits_ems: {fits_ems}")
                    print(f"      fits_container: {fits_container}")
                    print(f"      weight: {weight_ok}")
                    print(f"      incompatibility: {incomp_ok}")
                    print(f"      relative_pos: {relpos_ok}")
                    print(f"      affinity: {affinity_ok}")

        return False
    else:
        print(f"\n✓ Actions available!")
        print(f"  First action: {actions[0]}")

        # Try to execute first action
        print("\n" + "="*80)
        print("EXECUTING FIRST ACTION...")
        print("="*80)

        bin_idx, item_idx, ems_idx, rot_idx, ems, size, weight, item_id = actions[0]

        print(f"\n  Action details:")
        print(f"    Bin: {bin_idx}")
        print(f"    Item: {item_idx} (id={item_id})")
        print(f"    EMS: {ems_idx} at ({ems.x}, {ems.y}, {ems.z})")
        print(f"    Rotation: {rot_idx} → {size}")

        # Execute
        obs, reward, done, info = env.step(actions[0])

        if "invalid" in info:
            print(f"\n  ✗ PLACEMENT FAILED!")
            print(f"    enumerate_actions approved but place_at_ems rejected")
            return False
        else:
            print(f"\n  ✓ PLACEMENT SUCCESSFUL!")
            print(f"    Items remaining: {len(env.items)}")
            print(f"    Reward: {reward}")
            print(f"    Done: {done}")

            # Try second action
            print("\n" + "-"*80)
            print("Checking if more actions available...")
            print("-"*80)

            actions2 = env.enumerate_actions()
            print(f"\n  Actions available after first placement: {len(actions2)}")

            return True

if __name__ == "__main__":
    success = test_actual_environment()

    print("\n" + "="*80)
    if success:
        print("✓ ENVIRONMENT IS WORKING")
    else:
        print("❌ ENVIRONMENT HAS ISSUES")
    print("="*80)
