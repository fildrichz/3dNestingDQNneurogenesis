
#help me generate code for 3d cube nesting
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

#extreme point type
EP = Tuple[int, int, int]

#class
class box3d:
        def __init__(self, w, d, h):
            self.w, self.d, self.h = int(w), int(d), int(h)
            self.x, self.y, self.z = 0, 0, 0  # position (default origin)

        def orientations(self):
            """All 6 axis-aligned orientations of the box."""
            return [
                (self.w, self.d, self.h),
                (self.w, self.h, self.d),
                (self.d, self.w, self.h),
                (self.d, self.h, self.w),
                (self.h, self.w, self.d),
                (self.h, self.d, self.w),
            ]
        
        def set_position(self, x, y, z):
            self.x, self.y, self.z = int(x), int(y), int(z)

    
class Container(box3d):
    def __init__(self, W, D, H):
        super().__init__(W, D, H)
        self.placed: list[box3d] = []
        self.eps: list[EP] = [(0,0,0)]  # SDEPL
        self.ep_rs: Dict[EP, Tuple[int,int,int]] = {}  # EP -> (x_cap, y_cap, z_cap)

        #self.NOeps: set[EP] = set()

# ========== general placement and helpers ============

    @staticmethod
    def _orientations(w: int, d: int, h: int):
        return ((w,d,h),(w,h,d),(d,w,h),(d,h,w),(h,w,d),(h,d,w))
    
    def _fits_caps(self, ep: tuple[int,int,int], size: tuple[int,int,int]) -> bool:
        """True if size (w,d,h) fits within residual caps at EP. If EP has no caps yet, fall back to wall distances."""
        x, y, z = ep
        w, d, h = size
        caps = self.ep_rs.get(ep, (self.w - x, self.d - y, self.h - z))
        xcap, ycap, zcap = caps
        return (w <= xcap) and (d <= ycap) and (h <= zcap)
    
    def _fits_container(self, ep: tuple[int,int,int], size: tuple[int,int,int]) -> bool:
        x, y, z = ep; w, d, h = size
        return (0 <= x and 0 <= y and 0 <= z and
                x + w <= self.w and y + d <= self.d and z + h <= self.h)

    @staticmethod
    def colision_overlap(a1:int,a2:int,b1:int,b2:int) -> bool:
        return not (a2 <= b1 or b2 <= a1)

    def _fits_collision_free(self, ep: tuple[int,int,int], size: tuple[int,int,int]) -> bool:
        """Strict non-overlap against already placed boxes."""
        x, y, z = ep; w, d, h = size
        ax1, ay1, az1 = x, y, z
        ax2, ay2, az2 = x + w, y + d, z + h
        for b in self.placed:
            bx1, by1, bz1 = b.x, b.y, b.z
            bx2, by2, bz2 = b.x + b.w, b.y + b.d, b.z + b.h
            if (self.colision_overlap(ax1, ax2, bx1, bx2) and
                self.colision_overlap(ay1, ay2, by1, by2) and
                self.colision_overlap(az1, az2, bz1, bz2)):
                return False
        return True

    def place_at(self, ep: tuple[int,int,int], size: tuple[int,int,int]) -> bool:
        """
        Try to place a box of given size (w,d,h) at EP (x,y,z).
        Returns True on success (and updates self.placed/self.eps), else False.
        """
        if not (self._fits_caps(ep, size) and self._fits_container(ep, size) and self._fits_collision_free(ep, size)):
            return False

        x, y, z = ep; w, d, h = size
        k = box3d(w, d, h); k.set_position(x, y, z)
        self.placed.append(k)

        # if you keep a top index for ridge smoothing
        if hasattr(self, "tops_by_z"):
            self.tops_by_z.setdefault(k.z + k.h, []).append(k)

        # update EPs (your inward projection version), RS, and any prunes
        self.update_3depl(k)
        if hasattr(self, "update_residual_space"):
            self.update_residual_space(k)
        if hasattr(self, "_prune_by_residual_caps"):
            self._prune_by_residual_caps()
        if hasattr(self, "_prune_smooth_ridges_x"):
            self._prune_smooth_ridges_x(k)
        if hasattr(self, "_prune_colinear_min"):
            self._prune_colinear_min()

        return True
    
# ========== temporary placement demo ============

    def place_first_fit(self, base_size: tuple[int,int,int]) -> bool:
        """Try all 6 orientations at the first EP (z,y,x order) that fits; return True if placed."""
        for ep in sorted(set(self.eps), key=lambda p:(p[2], p[1], p[0])):
            for size in self._orientations(*base_size):
                if self.place_at(ep, size):
                    return True
        return False



# ========== 2nd algorithm helpers ============
    @staticmethod
    def _in_halfopen(v: int, a: int, b: int) -> bool:
        return a <= v < b

    def _is_on_side_x(self, ep: EP, it: box3d) -> bool:
        ex, _, _ = ep
        return self._in_halfopen(ex, it.x, it.x + it.w)

    def _is_on_side_y(self, ep: EP, it: box3d) -> bool:
        _, ey, _ = ep
        return self._in_halfopen(ey, it.y, it.y + it.d)

    def _is_on_side_xy(self, ep: EP, it: box3d) -> bool:
        ex, ey, _ = ep
        return (self._in_halfopen(ex, it.x, it.x + it.w) and
                self._in_halfopen(ey, it.y, it.y + it.d))
    
# ========== UpdateResidualSpace - 2nd algorithm  ============

    def update_residual_space(self, n: box3d) -> None:
        """
        Algorithm 2: tighten (x_cap, y_cap, z_cap) for each EP after placing item n.
        Caps are distances you can extend from the EP along +X, +Y, +Z.
        """
        nx1, ny1, nz1 = n.x, n.y, n.z
        nx2, ny2, nz2 = nx1 + n.w, ny1 + n.d, nz1 + n.h

        for ep in self.eps:
            ex, ey, ez = ep

            # init caps to walls if first time we see this EP
            if ep not in self.ep_rs:
                self.ep_rs[ep] = (self.w - ex, self.d - ey, self.h - ez)

            x_cap, y_cap, z_cap = self.ep_rs[ep]

            # --- first block: same vertical band as n (zn ≤ zEP < zn+hn)
            if nz1 <= ez < nz2:
                # clamp X if left of n and aligned in Y
                if ex <= nx1 and self._is_on_side_y(ep, n):
                    x_cap = min(x_cap, max(0, nx1 - ex))
                # clamp Y if behind n and aligned in X
                if ey <= ny1 and self._is_on_side_x(ep, n):
                    y_cap = min(y_cap, max(0, ny1 - ey))

            # --- second block: at/below n bottom and inside XY footprint
            if ez <= nz1 and self._is_on_side_xy(ep, n):
                z_cap = min(z_cap, max(0, nz1 - ez))

            self.ep_rs[ep] = (x_cap, y_cap, z_cap)

        # optional: drop caps for EPs that were removed this turn
        eps_set = set(self.eps)
        stale = [ep for ep in self.ep_rs.keys() if ep not in eps_set]
        for ep in stale:
            del self.ep_rs[ep]

    def _residual_ok(self, ep: EP, size: Tuple[int,int,int]) -> bool:
        caps = self.ep_rs.get(ep)
        if not caps:  # unseen EP → no extra restriction yet
            return True
        w, d, h = size
        x_cap, y_cap, z_cap = caps
        return (w <= x_cap) and (d <= y_cap) and (h <= z_cap)

# ======== 3D visualization (fix) ========

    @staticmethod
    def _cuboid_edges(x, y, z, w, d, h):
        X = [x, x+w, x+w, x,   x,   x+w, x+w, x]
        Y = [y, y,   y+d, y+d, y,   y,   y+d, y+d]
        Z = [z, z,   z,   z,   z+h, z+h, z+h, z+h]
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        return [[(X[i],Y[i],Z[i]), (X[j],Y[j],Z[j])] for i,j in edges]

    @staticmethod
    def _setup_axes(ax, W, D, H, title=""):
        ax.set_xlim(0, W); ax.set_ylim(0, D); ax.set_zlim(0, H)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_box_aspect((W, D, H))
        if title: ax.set_title(title)


    def plot3d(self, *, annotate_eps: bool = True,
               save_path: str | None = None,
               show: bool = True,
               title: str = "Packing state") -> None:
        """
        Visualize the container, placed cuboids, and EPs.
        - annotate_eps: show "(x,y,z)" next to each EP
        - save_path: if given, saves PNG there; otherwise uses plt.show()
        - show: call plt.show(); ignored if save_path is given
        """

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        self._setup_axes(ax, self.w, self.d, self.h, title)

        # container outline
        cont = Line3DCollection(self._cuboid_edges(0,0,0, self.w, self.d, self.h), linewidths=0.8)
        ax.add_collection3d(cont)

        # placed items (draw all; highlight last if any)
        if self.placed:
            *rest, last = self.placed
            for b in rest:
                lc = Line3DCollection(self._cuboid_edges(b.x,b.y,b.z, b.w,b.d,b.h), linewidths=1.4)
                ax.add_collection3d(lc)
            lc_last = Line3DCollection(self._cuboid_edges(last.x,last.y,last.z, last.w,last.d,last.h),
                                       linewidths=2.4)
            ax.add_collection3d(lc_last)

        # EPs
        if self.eps:
            xs, ys, zs = zip(*self.eps)
            ax.scatter(xs, ys, zs, s=28)
            if annotate_eps:
                for (x,y,z) in self.eps:
                    ax.text(x, y, z, f"({x},{y},{z})", fontsize=8)

        if save_path:
            fig.savefig(save_path, dpi=140, bbox_inches="tight")
            plt.close(fig)
        elif show:
            plt.show()
            plt.close(fig)
        else:
            # neither saving nor showing — still return the figure object
            return fig

    # ========== EP update (3DEPL) ============

    def _point_inside_box(self, p: EP, b: box3d) -> bool:
        """Closed-box test: True if p is inside OR on any face/edge/corner of b."""
        px, py, pz = p
        return (b.x <= px <= b.x + b.w and
                b.y <= py <= b.y + b.d and
                b.z <= pz <= b.z + b.h)


    def CanTakeProjection(self, k, i, axis: str, *, strict_overlap=True) -> bool:
        """
        INWARD version (project back toward 0):
        XY, XZ -> move along -X; blocker must be to the -X side (i.x2 <= k.x1)
        YX, YZ -> move along -Y; blocker must be to the -Y side (i.y2 <= k.y1)
        ZX, ZY -> move along -Z; blocker must be to the -Z side (i.z2 <= k.z1)
        And we still require overlap on the other two axes (strict by default).
        """
        kx1, ky1, kz1 = k.x, k.y, k.z
        kx2, ky2, kz2 = kx1 + k.w, ky1 + k.d, kz1 + k.h

        ix1, iy1, iz1 = i.x, i.y, i.z
        ix2, iy2, iz2 = ix1 + i.w, iy1 + i.d, iz1 + i.h

        def overlap_1d(a1, a2, b1, b2):
            return not (a2 <= b1 or b2 <= a1) if strict_overlap else not (a2 < b1 or b2 < a1)

        if axis in ("XY", "XZ"):  # move along -X
            ahead = ix2 <= kx1                     # blocker on the -X side
            oy = overlap_1d(iy1, iy2, ky1, ky2)    # Y overlap
            oz = overlap_1d(iz1, iz2, kz1, kz2)    # Z overlap
            return ahead and oy and oz

        if axis in ("YX", "YZ"):  # move along -Y
            ahead = iy2 <= ky1
            ox = overlap_1d(ix1, ix2, kx1, kx2)
            oz = overlap_1d(iz1, iz2, kz1, kz2)
            return ahead and ox and oz

        if axis in ("ZX", "ZY"):  # move along -Z
            ahead = iz2 <= kz1
            ox = overlap_1d(ix1, ix2, kx1, kx2)
            oy = overlap_1d(iy1, iy2, ky1, ky2)
            return ahead and ox and oy

        raise ValueError("Invalid axis label")

        

    def _prune_colinear_min(self):
        """
        Inward projection: on colinear rays, keep the EP closest to the origin.
        - same (y,z) → keep MIN x
        - same (x,z) → keep MIN y
        - same (x,y) → keep MIN z
        """
        if not self.eps:
            return

        by_yz = {}
        by_xz = {}
        by_xy = {}

        for (x, y, z) in self.eps:
            # track mins along each ray
            by_yz[(y, z)] = min(by_yz.get((y, z), x), x)
            by_xz[(x, z)] = min(by_xz.get((x, z), y), y)
            by_xy[(x, y)] = min(by_xy.get((x, y), z), z)

        kept = set()
        for (x, y, z) in self.eps:
            if x == by_yz[(y, z)]:
                kept.add((x, y, z))
            if y == by_xz[(x, z)]:
                kept.add((x, y, z))
            if z == by_xy[(x, y)]:
                kept.add((x, y, z))

        self.eps = sorted(kept, key=lambda p: (p[2], p[1], p[0]))


    @staticmethod
    def _in_bounds_static(p: EP, W: int, D: int, H: int) -> bool:
        x, y, z = p
        return (0 <= x <= W) and (0 <= y <= D) and (0 <= z <= H)

    def update_3depl(self, k) -> List[EP]:
        """
        Algorithm 1 + extensions:
        - Wall fallback if no blocker found for a label.
        - Remove EPs that fall inside the newly placed item k.
        """
        xk, yk, zk = k.x, k.y, k.z
        wk, dk, hk = k.w, k.d, k.h

        labels = ("YX", "YZ", "XY", "XZ", "ZX", "ZY")
        maxBound: Dict[str, int] = {lab: -1 for lab in labels}
        neweps: Dict[str, EP] = {}

        for i in self.placed:
            xi, yi, zi = i.x, i.y, i.z
            wi, di, hi = i.w, i.d, i.h

            # YX: project along -X; use i.x (lower X face)   — bound is xi
            if self.CanTakeProjection(k, i, "YX") and (xi > maxBound["YX"]):
                neweps["YX"] = (xi, yk + dk, zk)
                maxBound["YX"] = xi

            # YZ: project along -Y; use i.z (lower Z face)   — bound is zi
            if self.CanTakeProjection(k, i, "YZ") and (zi > maxBound["YZ"]):
                neweps["YZ"] = (xk, yk + dk, zi)
                maxBound["YZ"] = zi

            # XY: project along -X; use i.y (lower Y face)   — bound is yi
            if self.CanTakeProjection(k, i, "XY") and (yi > maxBound["XY"]):
                neweps["XY"] = (xk + wk, yi, zk)
                maxBound["XY"] = yi

            # XZ: project along -X; use i.z (lower Z face)   — bound is zi
            if self.CanTakeProjection(k, i, "XZ") and (zi > maxBound["XZ"]):
                neweps["XZ"] = (xk + wk, yk, zi)
                maxBound["XZ"] = zi

            # ZX: project along -Z; use i.x (lower X face)   — bound is xi
            if self.CanTakeProjection(k, i, "ZX") and (xi > maxBound["ZX"]):
                neweps["ZX"] = (xi, yk, zk + hk)
                maxBound["ZX"] = xi

            # ZY: project along -Z; use i.y (lower Y face)   — bound is yi
            if self.CanTakeProjection(k, i, "ZY") and (yi > maxBound["ZY"]):
                neweps["ZY"] = (xk, yi, zk + hk)
                maxBound["ZY"] = yi


        # --- ORIGIN FALLBACKS (if no blocker found for that label) ---
        if "YX" not in neweps: neweps["YX"] = (0,           yk + dk, zk)
        if "YZ" not in neweps: neweps["YZ"] = (xk,          yk + dk, 0)
        if "XY" not in neweps: neweps["XY"] = (xk + wk,     0,       zk)
        if "XZ" not in neweps: neweps["XZ"] = (xk + wk,     yk,      0)
        if "ZX" not in neweps: neweps["ZX"] = (0,           yk,      zk + hk)
        if "ZY" not in neweps: neweps["ZY"] = (xk,          0,       zk + hk)


        # Keep only EPs that are within container bounds
        W, D, H = self.w, self.d, self.h
        _in_bounds = lambda p: self._in_bounds_static(p, W, D, H)

        # survivors
        survivors = [ep for ep in self.eps if _in_bounds(ep) and not self._point_inside_box(ep, k)]

        # added
        added = [ep for ep in neweps.values() if _in_bounds(ep)]


        # Merge, dedup, and order by (z, y, x) non-decreasing
        merged = survivors + added
        self.eps = sorted(set(merged), key=lambda p: (p[2], p[1], p[0]))
        #self._prune_colinear_min()
        #self.update_residual_space(k)
        return self.eps



