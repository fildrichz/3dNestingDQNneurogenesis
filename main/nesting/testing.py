from packing_core import box3d, Container
# ============ sanity + plotting ============

def _mk(w, d, h, x=0, y=0, z=0):
    b = box3d(w, d, h); b.set_position(x, y, z); return b

def _SU(eps):  # sorted unique by (z,y,x)
    return sorted(set(eps), key=lambda p:(p[2], p[1], p[0]))

def run_visual_sanity(save_pngs: bool = False):
    """
    Places two 10x10x10 cubes:
      1) at origin
      2) adjacent in +X at (10,0,0)
    Checks EPs after each placement and plots the state.

    If save_pngs=True, saves 'sanity_after_1.png' and 'sanity_after_2.png'
    instead of opening windows.
    """
    W, D, H = 100, 60, 50

    # ---- Step 1: first cube at origin ----
    C = Container(W, D, H)
    C.eps = [(0, 0, 0)]
    C.placed = []

    k1 = _mk(10, 10, 10, 0, 0, 0)
    C.placed.append(k1)
    out1 = C.update_3depl(k1)

    expected1 = _SU([
        (10, 0, 0), (0, 10, 0), (0, 0, 10)
    ])
    assert _SU(out1) == expected1, f"EPs after cube #1 mismatch.\nGot {out1}\nExp {expected1}"
    assert (0,0,0) not in out1, "Seed EP (0,0,0) should be removed (swallowed)."

    if save_pngs:
        C.plot3d(title="After cube #1", save_path="sanity_after_1.png", show=False)
    else:
        C.plot3d(title="After cube #1")

    # ---- Step 2: second cube adjacent in +X at (10,0,0) ----
    k2 = _mk(10, 10, 10, 10, 0, 0)
    C.placed.append(k2)
    out2 = C.update_3depl(k2)

    # With only walls as blockers for k2, we expect the union of previous EPs
    # plus the four new wall-fallback EPs relative to k2's faces (the others duplicate):
    expected2 = _SU([(20, 0, 0), (0, 10, 0), (10, 10, 0), (0, 0, 10), (10, 0, 10)
    ])
    assert _SU(out2) == expected2, f"EPs after cube #2 mismatch.\nGot {out2}\nExp {expected2}"

    if save_pngs:
        C.plot3d(title="After cube #2", save_path="sanity_after_2.png", show=False)
        print("Saved: sanity_after_1.png, sanity_after_2.png")
    else:
        C.plot3d(title="After cube #2")

    #C._prune_colinear_min()
    #C.plot3d(title="After pruning colinear EPs")

    print("Visual sanity test passed.")

#run_visual_sanity(save_pngs=False)

def _mk(w,d,h,x=0,y=0,z=0):
    b = box3d(w,d,h); b.set_position(x,y,z); return b

def test_rs_caps():
    C = Container(100,60,50)
    C.eps=[(5,8,12)]
    n = _mk(20,20,10, 10,8,10)   # x:[10,30), y:[8,28), z:[10,20)
    C.placed.append(n)
    C.update_3depl(n)            # will call update_residual_space(n)
    #print("caps:", C.ep_rs[(5,8,12)])  # expect x_cap <= 5, y_cap unchanged, z_cap unchanged
    C.plot3d(title="After placing box for RS cap test")

test_rs_caps()

# =================== 4x4x4 cubes demo ===================

def _mk(w, d, h, x=0, y=0, z=0):
    b = box3d(w, d, h); b.set_position(x, y, z); return b

def run_4x4x4_demo(size: int = 10, n: int = 4, save_png: bool = True):
    """
    Builds an n×n×n grid of cubes (default 4x4x4) with edge length `size`
    using your inward-projection EP logic, then displays/saves the final plot.
    """
    W = n * size + 5
    D = n * size + 5
    H = n * size + 5

    C = Container(W, D, H)
    C.eps = [(0, 0, 0)]
    C.placed = []

    # If you added the top-index / smoothing helpers, they’ll just work.
    def _post_place(k):
        # optional: keep a fast index of tops by z (only if you added this field)
        if hasattr(C, "tops_by_z"):
            z_top = k.z + k.h
            C.tops_by_z.setdefault(z_top, []).append(k)
        # EP update
        C.update_3depl(k)
        # optional: residual caps + prunes if you added them
        if hasattr(C, "update_residual_space"):
            C.update_residual_space(k)
        if hasattr(C, "_prune_by_residual_caps"):
            C._prune_by_residual_caps()
        if hasattr(C, "_prune_smooth_ridges_x"):
            C._prune_smooth_ridges_x(k)
        if hasattr(C, "_prune_colinear_min"):
            C._prune_colinear_min()

    # place cubes in z-major order (bottom layer to top)
    for kz in range(n):
        for ky in range(n):
            for kx in range(n):
                x = kx * size
                y = ky * size
                z = kz * size
                k = _mk(size, size, size, x, y, z)
                C.placed.append(k)
                _post_place(k)

    title = f"{n}×{n}×{n} cubes (size={size}) — EPs: {len(C.eps)}"

    #print("number of extreme points:", len(C.eps))
    if save_png:
        C.plot3d(title=title, save_path=f"grid_{n}x{n}x{n}_size{size}.png", show=False)
        print(f"Saved: grid_{n}x{n}x{n}_size{size}.png  | EPs: {len(C.eps)}")
    else:
        C.plot3d(title=title)

    

# Uncomment to run:
#run_4x4x4_demo(size=10, n=8, save_png=False)

# =================== Greedy "first valid EP" fill demo ===================

# =================== many-boxes demo using Container helpers ===================

def run_many_boxes_demo(save_png: bool = True):
    """
    Greedy fill:
      - For each requested base size, try all 6 orientations
        at the first EP in (z,y,x) order using Container.place_first_fit().
      - Uses residual caps, bounds, and collision checks via Container methods.
      - Prints a summary and plots the final state.
    """
    # Tunables
    W, D, H = 140, 100, 90
    catalog = [
        (10,10,10), (12,8,10), (8,12,10), (6,6,6), (14,10,8),
        (10,14,8), (8,8,12), (10,6,10), (12,12,6), (16,8,6)
    ] * 15  # 50 items total

    # Container
    C = Container(W, D, H)
    C.eps = [(0,0,0)]
    C.placed = []

    placed = 0
    for idx, base in enumerate(catalog, 1):
        if C.place_first_fit(base):
            placed += 1
        else:
            print(f"[{idx:02d}] Could not place box {base} — skipping.")

    # Summary
    print("\n=== many-boxes demo ===")
    print(f"Container: {W}×{D}×{H}")
    print(f"Placed: {placed}/{len(catalog)} boxes")
    print(f"EPs left: {len(C.eps)}")
    # peek first few EPs to verify ordering and health
    head = sorted(set(C.eps), key=lambda p:(p[2],p[1],p[0]))[:12]
    print("First 12 EPs:", head)

    # Plot final state
    title = f"Placed {placed}/{len(catalog)} — EPs {len(C.eps)}"
    if save_png:
        C.plot3d(title=title, save_path="many_boxes_final.png", show=False)
        print("Saved: many_boxes_final.png")
    else:
        C.plot3d(title=title)

# Uncomment to run:
#run_many_boxes_demo(save_png=False)
