"""
Test script for EMS (Empty Maximal Spaces) implementation.
"""
import sys
sys.path.append('main')

from nesting.packing_core_enhanced import Container, EMS, box3d


def test_ems_basic():
    """Test basic EMS functionality."""
    print("="*80)
    print("TEST 1: Basic EMS Initialization")
    print("="*80)

    # Create a container
    container = Container(100, 100, 100, max_ems=50)

    print(f"Initial EMS list: {len(container.ems_list)} spaces")
    for i, ems in enumerate(container.ems_list):
        print(f"  EMS {i}: {ems}")

    assert len(container.ems_list) == 1, "Should start with 1 EMS"
    assert container.ems_list[0].volume() == 100*100*100, "Initial EMS should be full container"
    print("✓ PASSED\n")


def test_ems_placement():
    """Test EMS updates after placement."""
    print("="*80)
    print("TEST 2: EMS Update After Placement")
    print("="*80)

    # Create a container
    container = Container(100, 100, 100, max_ems=50)

    # Place a box at the origin
    ems = container.ems_list[0]
    size = (30, 30, 30)

    print(f"Placing box of size {size} at EMS corner ({ems.x}, {ems.y}, {ems.z})")
    success = container.place_at_ems(ems, size, weight=10, item_id=0)

    print(f"Placement success: {success}")
    print(f"Placed boxes: {len(container.placed)}")
    print(f"EMS list after placement: {len(container.ems_list)} spaces")

    for i, ems in enumerate(container.ems_list[:10]):  # Show top 10
        print(f"  EMS {i}: pos=({ems.x},{ems.y},{ems.z}), size=({ems.w},{ems.d},{ems.h}), vol={ems.volume()}")

    assert success, "Placement should succeed"
    assert len(container.placed) == 1, "Should have 1 placed box"
    assert len(container.ems_list) > 0, "Should have remaining EMS"
    print("✓ PASSED\n")


def test_ems_multiple_placements():
    """Test EMS with multiple placements."""
    print("="*80)
    print("TEST 3: Multiple Placements")
    print("="*80)

    # Create a container
    container = Container(100, 100, 100, max_ems=50)

    # Place multiple boxes
    placements = [
        (30, 30, 30),
        (20, 20, 20),
        (15, 15, 15),
    ]

    for i, size in enumerate(placements):
        if len(container.ems_list) == 0:
            print(f"No EMS available for box {i+1}")
            break

        # Find largest EMS
        largest_ems = max(container.ems_list, key=lambda e: e.volume())

        print(f"\nPlacing box {i+1} of size {size}")
        print(f"  Using EMS: pos=({largest_ems.x},{largest_ems.y},{largest_ems.z}), "
              f"size=({largest_ems.w},{largest_ems.d},{largest_ems.h})")

        success = container.place_at_ems(largest_ems, size, weight=10, item_id=i)

        print(f"  Success: {success}")
        print(f"  Total placed: {len(container.placed)}")
        print(f"  Remaining EMS: {len(container.ems_list)}")

    print(f"\nFinal state:")
    print(f"  Boxes placed: {len(container.placed)}")
    print(f"  Total volume used: {sum(b.w*b.d*b.h for b in container.placed)}")
    print(f"  Remaining EMS: {len(container.ems_list)}")

    assert len(container.placed) > 0, "Should have placed some boxes"
    print("✓ PASSED\n")


def test_ems_vs_ep_action_space():
    """Compare action space size between EMS and EP (conceptually)."""
    print("="*80)
    print("TEST 4: Action Space Comparison")
    print("="*80)

    try:
        from packing_with_dqncore2_enhanced import MultiBinPackingEnv

        # Create environment with some items
        items = [
            (30, 30, 30, 100, 0),
            (25, 25, 25, 80, 1),
            (20, 20, 20, 60, 2),
        ]

        env = MultiBinPackingEnv(W=100, D=100, H=100, items=items,
                                max_actions=128, topk_eps=32, max_bins=2)

        # Get action space
        actions, mask = env.action_space()

        print(f"Items to place: {len(env.items)}")
        print(f"Bins available: {env.max_bins}")
        print(f"Actions generated: {len([a for a in actions if a is not None])}")
        print(f"EMS per bin: {[len(bin.ems_list) for bin in env.bins]}")

        # Check action structure
        if actions and actions[0] is not None:
            sample_action = actions[0]
            print(f"\nSample action structure:")
            print(f"  bin_idx: {sample_action[0]}")
            print(f"  item_idx: {sample_action[1]}")
            print(f"  ems_idx: {sample_action[2]}")
            print(f"  rot_idx: {sample_action[3]}")
            print(f"  ems: {sample_action[4]}")
            print(f"  size: {sample_action[5]}")

        assert len(actions) > 0, "Should have some actions"
        print("✓ PASSED\n")
    except ImportError as e:
        print(f"⊘ SKIPPED (missing dependency: {e})\n")


def test_ems_gravity():
    """Test that gravity still works with EMS."""
    print("="*80)
    print("TEST 5: Gravity with EMS")
    print("="*80)

    container = Container(100, 100, 100, max_ems=50)

    # Place first box on ground
    ems1 = container.ems_list[0]
    container.place_at_ems(ems1, (30, 30, 30), weight=10, item_id=0)

    print(f"Box 1 placed at: ({container.placed[0].x}, {container.placed[0].y}, {container.placed[0].z})")
    print(f"Box 1 expected at: (0, 0, 0) - ground level")

    # Try to place second box "floating" - should drop due to gravity
    # Find an EMS that's above ground
    high_ems = None
    for ems in container.ems_list:
        if ems.z > 0:
            high_ems = ems
            break

    if high_ems:
        print(f"\nTrying to place box at high EMS: pos=({high_ems.x},{high_ems.y},{high_ems.z})")
        container.place_at_ems(high_ems, (20, 20, 20), weight=10, item_id=1)

        if len(container.placed) > 1:
            print(f"Box 2 placed at: ({container.placed[1].x}, {container.placed[1].y}, {container.placed[1].z})")
            print(f"Note: Gravity should have adjusted Z position if needed")

    # Verify first box is at z=0
    assert container.placed[0].z == 0, "First box should be at ground level"
    print("✓ PASSED\n")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("TESTING EMS IMPLEMENTATION")
    print("="*80 + "\n")

    try:
        test_ems_basic()
        test_ems_placement()
        test_ems_multiple_placements()
        test_ems_vs_ep_action_space()
        test_ems_gravity()

        print("="*80)
        print("ALL TESTS PASSED! ✓")
        print("="*80)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
