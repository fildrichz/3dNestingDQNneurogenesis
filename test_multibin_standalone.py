"""
Standalone test of multi-bin environment without torch dependency.
"""

import sys
sys.path.insert(0, 'main')

import numpy as np
from typing import List, Tuple, Optional
from nesting.packing_core_enhanced import Container
from nesting.dataset_loader import load_problem, BinPackingProblem

def load_problem_as_items(problem: BinPackingProblem) -> List[Tuple[int,int,int,int,int]]:
    """Convert problem items to list format."""
    items = []
    for i, item in enumerate(problem.items):
        for _ in range(item.quantity):
            items.append((item.length, item.width, item.height, item.weight, i))
    return items

# Simplified MultiBinPackingEnv without torch dependencies
class TestMultiBinEnv:
    def __init__(self, W, D, H, items, max_actions=128, topk_eps=1000,
                 seed=0, gamma=0.992, problem=None):
        self.rng = np.random.default_rng(seed)
        self.bin_size = (int(W), int(D), int(H))
        self.bin_volume = int(W*D*H)
        self.gamma = float(gamma)
        self.max_actions = int(max_actions)
        self.topk_eps = int(topk_eps)

        # Store problem constraints
        if problem is not None:
            self.max_bins = problem.max_bins
            self.max_weight = problem.max_weight
            self.incompatibilities = problem.incompatibilities
            self.positive_affinities = problem.positive_affinities
            self.relative_pos = problem.relative_pos
            self.center_of_mass_constraint = getattr(problem, 'center_of_mass', None)
        else:
            self.max_bins = 1
            self.max_weight = None
            self.incompatibilities = []
            self.positive_affinities = []
            self.relative_pos = {}
            self.center_of_mass_constraint = None

        self.initial_items = list(items) if items else []
        self.items = list(items) if items else []
        self.n_items = len(self.items)

        # Create bins
        self.bins = [self._create_bin() for _ in range(self.max_bins)]
        self.total_placed_volume = 0
        self.done = False

    def _create_bin(self):
        W, D, H = self.bin_size
        C = Container(W, D, H, max_weight=self.max_weight)
        C.set_constraints(
            incompatibilities=self.incompatibilities,
            positive_affinities=self.positive_affinities,
            center_of_mass=self.center_of_mass_constraint,
            relative_pos=self.relative_pos
        )
        return C

    def reset(self, items=None):
        """Reset environment."""
        if items is not None:
            self.items = list(items)
        else:
            self.items = list(self.initial_items)

        self.n_items = len(self.items)
        self.bins = [self._create_bin() for _ in range(self.max_bins)]
        self.total_placed_volume = 0
        self.done = False

    def check_affinity_placement(self, item_id: int, target_bin_idx: int) -> bool:
        """Check if placing item in target bin would violate affinity."""
        if not self.positive_affinities:
            return True

        # Find all affinity partners
        affinity_partners = set()
        for a, b in self.positive_affinities:
            if item_id == a:
                affinity_partners.add(b)
            if item_id == b:
                affinity_partners.add(a)

        if not affinity_partners:
            return True

        # Check if any partners are in other bins
        for bin_idx, bin in enumerate(self.bins):
            if bin_idx == target_bin_idx:
                continue

            for partner_id in affinity_partners:
                if partner_id in bin.item_ids_in_bin:
                    return False

        return True

    def enumerate_actions(self):
        """Enumerate all valid actions across all bins."""
        actions = []

        for bin_idx, bin in enumerate(self.bins):
            # Sort EMS by volume and take top-k
            ems_sorted = sorted(bin.ems_list, key=lambda e: (-e.volume(), e.z, e.y, e.x))[:self.topk_eps]

            for ems_idx, ems in enumerate(ems_sorted):
                for item_idx, item in enumerate(self.items):
                    w, d, h, weight, item_id = item
                    rots = ((w,d,h), (w,h,d), (d,w,h), (d,h,w), (h,w,d), (h,d,w))

                    for rot_idx, size in enumerate(rots):
                        ep = (ems.x, ems.y, ems.z)

                        # Check all constraints
                        if (bin._fits_ems(ems, size) and
                            bin._fits_container(ep, size) and
                            bin.check_weight_constraint(weight) and
                            bin.check_incompatibility(item_id) and
                            bin.check_relative_positioning(item_id, ep, size) and
                            self.check_affinity_placement(item_id, bin_idx)):

                            actions.append((bin_idx, item_idx, ems_idx, rot_idx, ems, size, weight, item_id))

        return actions

    def step(self, action):
        """Execute action."""
        if action is None:
            return None, -1.0, True, {"items_placed": self.n_items - len(self.items)}

        bin_idx, item_idx, ems_idx, rot_idx, ems, size, weight, item_id = action
        target_bin = self.bins[bin_idx]

        ok = target_bin.place_at_ems(ems, size, weight=weight, item_id=item_id)

        if not ok:
            return None, -1.0, True, {"invalid": True, "items_placed": self.n_items - len(self.items)}

        # Update
        v = int(size[0] * size[1] * size[2])
        self.total_placed_volume += v
        del self.items[item_idx]

        done = len(self.items) == 0
        reward = 1.0 if done else 0.0

        return None, reward, done, {"items_placed": self.n_items - len(self.items)}


def test_multibin_standalone():
    """Test the multi-bin environment."""
    print("="*80)
    print("TESTING MULTI-BIN ENVIRONMENT (STANDALONE)")
    print("="*80)

    # Load problem
    problem_path = 'main/nesting/inputData/Benchmark dataset and instance generator for Real-World 3dBPP/Input/3dBPP_4.txt'
    problem = load_problem(problem_path)
    items = load_problem_as_items(problem)

    W, D, H = problem.bin_dimensions

    print(f"\nProblem: {W}×{D}×{H}, {len(items)} items")
    print(f"Max bins: {problem.max_bins}")
    print(f"Max weight: {problem.max_weight}")

    # Create environment
    env = TestMultiBinEnv(
        W, D, H,
        items=items,
        max_actions=128,
        topk_eps=1000,
        seed=42,
        gamma=0.992,
        problem=problem
    )

    env.reset(items=items.copy())

    print(f"\nEnvironment state:")
    print(f"  Bins: {len(env.bins)}")
    print(f"  Items: {len(env.items)}")

    for i, bin in enumerate(env.bins):
        print(f"\n  Bin {i}:")
        print(f"    EMS: {len(bin.ems_list)}")
        print(f"    Placed: {len(bin.placed)}")
        if bin.ems_list:
            ems = bin.ems_list[0]
            print(f"    First EMS: ({ems.x},{ems.y},{ems.z}) size ({ems.w},{ems.d},{ems.h})")
        print(f"    Relative pos constraints: {bin.relative_pos}")

    # Enumerate actions
    print("\n" + "="*80)
    print("ENUMERATING ACTIONS...")
    print("="*80)

    actions = env.enumerate_actions()

    print(f"\nActions found: {len(actions)}")

    if len(actions) == 0:
        print("\n❌ NO ACTIONS - DEBUGGING FIRST ITEM")

        bin = env.bins[0]
        item = env.items[0]
        w, d, h, weight, item_id = item

        print(f"\n  Item: id={item_id}, size=({w},{d},{h}), weight={weight}")

        if bin.ems_list:
            ems = bin.ems_list[0]
            print(f"  EMS: ({ems.x},{ems.y},{ems.z}) size ({ems.w},{ems.d},{ems.h})")

            rots = [(w,d,h), (w,h,d), (d,w,h), (d,h,w), (h,w,d), (h,d,w)]

            for rot_idx, size in enumerate(rots):
                ep = (ems.x, ems.y, ems.z)

                fits_ems = bin._fits_ems(ems, size)
                fits_container = bin._fits_container(ep, size)
                weight_ok = bin.check_weight_constraint(weight)
                incomp_ok = bin.check_incompatibility(item_id)
                relpos_ok = bin.check_relative_positioning(item_id, ep, size)
                affinity_ok = env.check_affinity_placement(item_id, 0)

                all_ok = fits_ems and fits_container and weight_ok and incomp_ok and relpos_ok and affinity_ok

                print(f"\n    Rot {rot_idx} {size}: {'✓' if all_ok else '✗'}")
                if not all_ok:
                    print(f"      fits_ems: {fits_ems}")
                    print(f"      fits_container: {fits_container}")
                    print(f"      weight: {weight_ok}")
                    print(f"      incomp: {incomp_ok}")
                    print(f"      relpos: {relpos_ok}")
                    print(f"      affinity: {affinity_ok}")

        return False
    else:
        print(f"✓ {len(actions)} actions available")

        # Execute actions until done
        placed = 0
        max_steps = 10

        for step in range(max_steps):
            actions = env.enumerate_actions()
            if not actions:
                break

            action = actions[0]
            _, reward, done, info = env.step(action)

            if "invalid" in info:
                print(f"\n✗ Step {step+1}: PLACEMENT FAILED")
                return False

            placed += 1
            print(f"  Step {step+1}: Placed item (total: {placed}/{len(items)})")

            if done:
                print(f"\n✓ COMPLETED: All {placed} items placed!")
                return True

        print(f"\n  Stopped after {max_steps} steps ({placed} items placed)")
        return True

if __name__ == "__main__":
    success = test_multibin_standalone()

    print("\n" + "="*80)
    if success:
        print("✓ ENVIRONMENT WORKING")
    else:
        print("❌ ENVIRONMENT HAS ISSUES")
    print("="*80)
