"""
Check for floating boxes in placed containers.
A box is floating if its bottom (z position) is above ground
and it has no supporting box beneath it.
"""

import sys
sys.path.insert(0, 'main')

from nesting.packing_core_enhanced import Container

def check_box_has_support(box, all_boxes):
    """
    Check if a box has support beneath it.
    Returns: (has_support, support_z, supporting_boxes)
    """
    if box.z == 0:
        return True, 0, ["ground"]

    # Find all boxes that could potentially support this box
    supporting_boxes = []

    for other in all_boxes:
        if other is box:
            continue

        # Check if other box is below this box
        if other.z + other.h > box.z:
            continue  # Other box extends above our bottom

        # Check if they overlap in XY plane
        x_overlap = not (box.x + box.w <= other.x or other.x + other.w <= box.x)
        y_overlap = not (box.y + box.d <= other.y or other.y + other.d <= box.y)

        if x_overlap and y_overlap:
            # This box overlaps in XY and is below us
            # Check if it's directly supporting us (top touches our bottom)
            if other.z + other.h == box.z:
                supporting_boxes.append(other)

    if supporting_boxes:
        support_z = max(b.z + b.h for b in supporting_boxes)
        return True, support_z, supporting_boxes
    else:
        return False, -1, []

def analyze_container_for_floating(container):
    """Analyze a container and report any floating boxes."""
    print("\n" + "="*80)
    print("FLOATING BOX ANALYSIS")
    print("="*80)

    floating_boxes = []
    grounded_boxes = 0
    supported_boxes = 0

    for i, box in enumerate(container.placed):
        has_support, support_z, supporters = check_box_has_support(box, container.placed)

        if box.z == 0:
            grounded_boxes += 1
        elif has_support:
            supported_boxes += 1
        else:
            floating_boxes.append((i, box, support_z))

    print(f"\nTotal boxes: {len(container.placed)}")
    print(f"  - Grounded (z=0): {grounded_boxes}")
    print(f"  - Supported by other boxes: {supported_boxes}")
    print(f"  - FLOATING: {len(floating_boxes)}")

    if floating_boxes:
        print("\n" + "="*80)
        print("FLOATING BOXES DETECTED!")
        print("="*80)
        for i, box, support_z in floating_boxes:
            print(f"\nBox #{i}:")
            print(f"  Item ID: {box.item_id}")
            print(f"  Position: ({box.x}, {box.y}, {box.z})")
            print(f"  Size: ({box.w}, {box.d}, {box.h})")
            print(f"  Bottom at z={box.z}, Top at z={box.z + box.h}")
            print(f"  ⚠️  NO SUPPORT BENEATH THIS BOX!")

            # Check what should be supporting it
            expected_z = container.find_support_surface(box.x, box.y, box.w, box.d)
            print(f"  Expected z position: {expected_z} (gap: {box.z - expected_z})")
    else:
        print("\n✓ All boxes are properly supported!")

    print("="*80 + "\n")
    return len(floating_boxes) > 0

# Load and analyze existing output
if __name__ == "__main__":
    # Try to load from saved state or create test
    print("Checking for floating boxes in output data...")

    # For now, create a test case
    print("\nCreating test container with potential floating box issue...")
    container = Container(1000, 1000, 1000)

    # Manually create a scenario that would cause floating if gravity fails
    # Place box at ground
    container.place_at_ems(
        container.ems_list[0],
        size=(200, 200, 800),
        weight=10,
        item_id=1
    )

    print(f"Placed box 1 at: ({container.placed[0].x}, {container.placed[0].y}, {container.placed[0].z})")
    print(f"EMS count after box 1: {len(container.ems_list)}")

    # Now try to place a box using an EMS that might be floating
    # Find an EMS with high z value
    high_ems = None
    for ems in sorted(container.ems_list, key=lambda e: -e.z):
        if ems.z > 100:
            high_ems = ems
            break

    if high_ems:
        print(f"\nTrying to place using high EMS at: ({high_ems.x}, {high_ems.y}, {high_ems.z})")
        print(f"  EMS size: ({high_ems.w}, {high_ems.d}, {high_ems.h})")

        success = container.place_at_ems(
            high_ems,
            size=(100, 100, 100),
            weight=5,
            item_id=2
        )

        if success:
            box2 = container.placed[-1]
            print(f"✓ Placed box 2 at: ({box2.x}, {box2.y}, {box2.z})")
        else:
            print("✗ Failed to place box 2")

    # Analyze for floating boxes
    has_floating = analyze_container_for_floating(container)

    if has_floating:
        print("GRAVITY IS NOT WORKING CORRECTLY!")
        print("Generating visualization...")
        container.plot3d_filled(save_path="floating_test.png", show=False)
        print("✓ Saved: floating_test.png")
    else:
        print("Gravity appears to be working correctly in this test.")
