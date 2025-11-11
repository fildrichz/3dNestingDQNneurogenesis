"""
Analyze saved packing state to check for floating boxes.
This will help us verify if gravity is working in actual DQN runs.
"""

import sys
sys.path.insert(0, 'main')

import pickle
from nesting.packing_core_enhanced import Container

def analyze_floating_boxes(container, name="Container"):
    """Check for floating boxes."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {name}")
    print(f"{'='*80}")

    floating = []
    grounded = 0
    supported = 0

    for i, box in enumerate(container.placed):
        if box.z == 0:
            grounded += 1
            continue

        # Check if box has support beneath it
        has_support = False
        for other in container.placed:
            if other is box:
                continue

            # Check if other box overlaps in XY and is directly below
            x_overlap = not (box.x + box.w <= other.x or other.x + other.w <= box.x)
            y_overlap = not (box.y + box.d <= other.y or other.y + other.d <= box.y)

            if x_overlap and y_overlap:
                # Other box overlaps in XY
                if other.z + other.h == box.z:
                    # Directly supports this box
                    has_support = True
                    break

        if has_support:
            supported += 1
        else:
            floating.append((i, box))

    print(f"Total boxes: {len(container.placed)}")
    print(f"  Grounded (z=0): {grounded}")
    print(f"  Supported: {supported}")
    print(f"  FLOATING: {len(floating)}")

    if floating:
        print(f"\n⚠️  FLOATING BOXES DETECTED:")
        for i, box in floating[:10]:  # Show first 10
            print(f"  Box {i}: item_id={box.item_id}, pos=({box.x},{box.y},{box.z}), size=({box.w},{box.d},{box.h})")

            # Calculate expected Z
            expected_z = 0
            for other in container.placed:
                if other is box:
                    continue
                x_overlap = not (box.x + box.w <= other.x or other.x + other.w <= box.x)
                y_overlap = not (box.y + box.d <= other.y or other.y + other.d <= box.y)
                if x_overlap and y_overlap:
                    expected_z = max(expected_z, other.z + other.h)

            if expected_z != box.z:
                print(f"    Expected z={expected_z}, got z={box.z}, gap={box.z - expected_z}")
    else:
        print("\n✓ All boxes properly supported!")

    return len(floating)


# Test with fresh code
print("="*80)
print("TESTING CURRENT EMS-BASED CODE")
print("="*80)

# Create a realistic test scenario
container = Container(1000, 1000, 1200)

# Place several boxes to create a complex scenario
test_placements = [
    # Ground level boxes
    ((200, 200, 100), 10, 1),
    ((300, 300, 150), 15, 2),
    ((100, 600, 200), 20, 3),
    ((700, 200, 180), 12, 4),

    # These should stack or drop to ground
    ((50, 50, 100), 8, 5),
    ((350, 350, 120), 10, 6),
    ((150, 650, 100), 9, 7),
]

print("\nPlacing test boxes...")
for size, weight, item_id in test_placements:
    # Find best EMS
    best_ems = None
    for ems in sorted(container.ems_list, key=lambda e: -e.volume()):
        if ems.w >= size[0] and ems.d >= size[1] and ems.h >= size[2]:
            best_ems = ems
            break

    if best_ems:
        success = container.place_at_ems(best_ems, size, weight=weight, item_id=item_id)
        if success:
            box = container.placed[-1]
            print(f"  ✓ Placed item {item_id} at ({box.x}, {box.y}, {box.z})")
        else:
            print(f"  ✗ Failed to place item {item_id}")

# Analyze
num_floating = analyze_floating_boxes(container, "Test Container")

if num_floating > 0:
    print("\n⚠️  GRAVITY BUG CONFIRMED IN CURRENT CODE!")
    container.plot3d_filled(save_path="current_code_test.png", show=False)
    print("  Saved visualization: current_code_test.png")
else:
    print("\n✓ Gravity working correctly in current code!")
    print("  The floating boxes in output_data/*.png are from the old EP-based code.")

print("="*80)
