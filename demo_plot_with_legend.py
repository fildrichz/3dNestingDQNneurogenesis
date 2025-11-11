"""
Demonstrate the updated plotting functionality with color legend.

The plots now show:
- Which color corresponds to which item ID
- Count of boxes for each item ID
- EMS visualization (if present)
"""

import sys
sys.path.insert(0, 'main')

from nesting.packing_core_enhanced import Container

def demo_plot_with_legend():
    """Create a packing and show plot with color legend."""
    print("\n" + "="*80)
    print("DEMONSTRATION: PLOTTING WITH COLOR LEGEND")
    print("="*80)

    # Create container
    container = Container(200, 200, 100)

    # Place various items
    print("\nPlacing items...")
    items_to_place = [
        # (item_id, size, weight, description)
        (1, (40, 40, 20), 10, "Item 1 - Large box"),
        (1, (40, 40, 20), 10, "Item 1 - Another large box"),
        (3, (30, 30, 15), 5, "Item 3 - Medium box"),
        (3, (30, 30, 15), 5, "Item 3 - Another medium box"),
        (3, (30, 30, 15), 5, "Item 3 - Third medium box"),
        (5, (25, 25, 10), 3, "Item 5 - Small box"),
        (7, (20, 20, 8), 2, "Item 7 - Tiny box"),
        (7, (20, 20, 8), 2, "Item 7 - Another tiny box"),
    ]

    placed_count = 0
    for item_id, size, weight, desc in items_to_place:
        # Try to find a suitable EMS
        best_ems = None
        for ems in container.ems_list:
            if ems.w >= size[0] and ems.d >= size[1] and ems.h >= size[2]:
                best_ems = ems
                break

        if best_ems:
            success = container.place_at_ems(best_ems, size, weight=weight, item_id=item_id)
            if success:
                print(f"  ✓ Placed {desc}")
                placed_count += 1
            else:
                print(f"  ✗ Failed to place {desc}")
        else:
            print(f"  ✗ No suitable EMS for {desc}")

    print(f"\nPlaced {placed_count}/{len(items_to_place)} items")

    # Print constraint info
    container.print_constraint_info()

    # Print color legend to console
    container.print_item_legend()

    # Show visual plots
    print("="*80)
    print("GENERATING PLOTS")
    print("="*80)
    print("\nGenerating plot3d (wireframe with legend)...")
    print("  - Legend shows: Item ID (n=count)")
    print("  - Colors use matplotlib's tab20 colormap")
    print("  - EMS shown as red dashed boxes")

    # Save wireframe plot
    container.plot3d(save_path="demo_wireframe_with_legend.png", show=False)
    print("  ✓ Saved: demo_wireframe_with_legend.png")

    print("\nGenerating plot3d_filled (filled boxes with legend)...")
    # Save filled plot
    container.plot3d_filled(save_path="demo_filled_with_legend.png", show=False)
    print("  ✓ Saved: demo_filled_with_legend.png")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("Both plots now include:")
    print("  1. Color-coded boxes by item ID")
    print("  2. Legend showing Item ID → Color mapping")
    print("  3. Count of boxes for each item (n=count)")
    print("  4. EMS visualization (red dashed boxes) in wireframe plot")
    print("\nYou can now easily verify:")
    print("  - Which colored boxes correspond to which item IDs")
    print("  - Whether relative positioning constraints are satisfied")
    print("  - Whether incompatibility constraints are satisfied")
    print("  - Whether affinity constraints are satisfied")
    print("="*80 + "\n")


if __name__ == "__main__":
    demo_plot_with_legend()
