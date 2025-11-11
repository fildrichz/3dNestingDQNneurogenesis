"""
Full environment test mimicking actual DQN training.
"""

import sys
sys.path.insert(0, 'main')

import numpy as np
from nesting.packing_core_enhanced import Container
from nesting.dataset_loader import load_problem

# Minimal environment simulation
class TestEnv:
    def __init__(self, W, D, H, items, problem):
        self.W, self.D, self.H = W, D, H
        self.items = items.copy()
        self.bins = [Container(W, D, H, max_weight=problem.max_weight)]
        self.bins[0].set_constraints(
            incompatibilities=problem.incompatibilities,
            positive_affinities=problem.positive_affinities,
            relative_pos=problem.relative_pos
        )
        self.positive_affinities = problem.positive_affinities or []
        self.topk_eps = 1000
        self.max_bins = 1

    def check_affinity_placement(self, item_id, target_bin_idx):
        if not self.positive_affinities:
            return True

        affinity_partners = set()
        for a, b in self.positive_affinities:
            if item_id == a:
                affinity_partners.add(b)
            if item_id == b:
                affinity_partners.add(a)

        if not affinity_partners:
            return True

        for bin_idx, bin in enumerate(self.bins):
            if bin_idx == target_bin_idx:
                continue

            for partner_id in affinity_partners:
                if partner_id in bin.item_ids_in_bin:
                    return False

        return True

    def enumerate_actions(self):
        actions = []

        for bin_idx, bin in enumerate(self.bins):
            ems_sorted = sorted(bin.ems_list, key=lambda e: (-e.volume(), e.z, e.y, e.x))[:self.topk_eps]

            for ems_idx, ems in enumerate(ems_sorted):
                for item_idx, item in enumerate(self.items):
                    w, d, h, weight, item_id = item
                    rots = ((w,d,h), (w,h,d), (d,w,h), (d,h,w), (h,w,d), (h,d,w))

                    for rot_idx, size in enumerate(rots):
                        ep = (ems.x, ems.y, ems.z)

                        if (bin._fits_ems(ems, size) and
                            bin._fits_container(ep, size) and
                            bin.check_weight_constraint(weight) and
                            bin.check_incompatibility(item_id) and
                            bin.check_relative_positioning(item_id, ep, size) and
                            self.check_affinity_placement(item_id, bin_idx)):

                            actions.append((bin_idx, item_idx, ems_idx, rot_idx, ems, size, weight, item_id))

        return actions

    def step(self, action):
        if action is None:
            return None, -1.0, True, {}

        bin_idx, item_idx, ems_idx, rot_idx, ems, size, weight, item_id = action
        target_bin = self.bins[bin_idx]

        ok = target_bin.place_at_ems(ems, size, weight=weight, item_id=item_id)

        if not ok:
            print(f"  ✗ place_at_ems FAILED for item {item_id}")
            return None, -1.0, True, {"invalid": True}

        del self.items[item_idx]
        return None, 1.0, len(self.items) == 0, {}


def test_full_environment():
    """Test full environment flow."""
    print("="*80)
    print("FULL ENVIRONMENT TEST")
    print("="*80)

    # Load problem
    problem_path = 'main/nesting/inputData/Benchmark dataset and instance generator for Real-World 3dBPP/Input/3dBPP_4.txt'
    problem = load_problem(problem_path)

    W, D, H = problem.bin_dimensions
    items = [(item.length, item.width, item.height, item.weight, i)
             for i, item in enumerate(problem.items)
             for _ in range(item.quantity)]

    print(f"\nProblem: {W}×{D}×{H}, {len(items)} items")

    # Create environment
    env = TestEnv(W, D, H, items, problem)

    print(f"\nInitial state:")
    print(f"  Bins: {len(env.bins)}")
    print(f"  Items: {len(env.items)}")
    print(f"  Bin 0 EMS: {len(env.bins[0].ems_list)}")

    # Try to enumerate actions
    print("\n" + "-"*80)
    print("ENUMERATING ACTIONS...")
    print("-"*80)

    actions = env.enumerate_actions()

    print(f"\nActions found: {len(actions)}")

    if len(actions) == 0:
        print("\n❌ NO ACTIONS FOUND!")
        print("  This is the problem - environment can't place anything")

        # Debug first item specifically
        print("\n  Debugging first item:")
        bin = env.bins[0]
        item = env.items[0]
        w, d, h, weight, item_id = item
        ems = bin.ems_list[0]

        print(f"    Item: id={item_id}, size=({w},{d},{h})")
        print(f"    EMS: pos=({ems.x},{ems.y},{ems.z}), size=({ems.w},{ems.d},{ems.h})")

        ep = (ems.x, ems.y, ems.z)
        size = (w, d, h)

        print(f"\n    Checks:")
        print(f"      _fits_ems: {bin._fits_ems(ems, size)}")
        print(f"      _fits_container: {bin._fits_container(ep, size)}")
        print(f"      check_weight: {bin.check_weight_constraint(weight)}")
        print(f"      check_incompatibility: {bin.check_incompatibility(item_id)}")
        print(f"      check_relative_pos: {bin.check_relative_positioning(item_id, ep, size)}")
        print(f"      check_affinity: {env.check_affinity_placement(item_id, 0)}")

        return False
    else:
        print(f"\n✓ Actions available: {len(actions)}")
        print(f"  First action: bin={actions[0][0]}, item={actions[0][1]}, rot={actions[0][3]}")

        # Try to execute first action
        print("\n" + "-"*80)
        print("EXECUTING FIRST ACTION...")
        print("-"*80)

        action = actions[0]
        bin_idx, item_idx, ems_idx, rot_idx, ems, size, weight, item_id = action

        print(f"  Action: place item {item_id} in bin {bin_idx}")
        print(f"    Size: {size}, EMS: ({ems.x},{ems.y},{ems.z})")

        _, reward, done, info = env.step(action)

        if "invalid" in info:
            print(f"  ✗ PLACEMENT FAILED")
            print(f"    This means enumerate_actions allowed it but place_at_ems rejected it")
            return False
        else:
            print(f"  ✓ PLACEMENT SUCCESSFUL")
            print(f"    Items remaining: {len(env.items)}")
            return True


if __name__ == "__main__":
    success = test_full_environment()

    print("\n" + "="*80)
    if success:
        print("✓ ENVIRONMENT WORKING CORRECTLY")
    else:
        print("❌ ENVIRONMENT HAS ISSUES")
    print("="*80)
