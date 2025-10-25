# packing_core.py — Wall-fallback EP variant (inward projections)
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

EP = Tuple[int, int, int]

class box3d:
    def __init__(self, w, d, h):
        self.w, self.d, self.h = int(w), int(d), int(h)
        self.x, self.y, self.z = 0, 0, 0

    def orientations(self):
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
        self.eps: list[EP] = [(0, 0, 0)]
        self.ep_rs: Dict[EP, Tuple[int, int, int]] = {}

    # ========== general placement and helpers ============

    @staticmethod
    def _orientations(w: int, d: int, h: int):
        return ((w, d, h), (w, h, d), (d, w, h), (d, h, w), (h, w, d), (h, d, w))

    def _fits_caps(self, ep: tuple[int, int, int], size: tuple[int, int, int]) -> bool:
        x, y, z = ep
        w, d, h = size
        caps = self.ep_rs.get(ep, (self.w - x, self.d - y, self.h - z))
        xcap, ycap, zcap = caps
        return (w <= xcap) and (d <= ycap) and (h <= zcap)

    def _fits_container(self, ep: tuple[int, int, int], size: tuple[int, int, int]) -> bool:
        x, y, z = ep
        w, d, h = size
        return (0 <= x and 0 <= y and 0 <= z and
                x + w <= self.w and y + d <= self.d and z + h <= self.h)

    @staticmethod
    def colision_overlap(a1: int, a2: int, b1: int, b2: int) -> bool:
        return not (a2 <= b1 or b2 <= a1)

    def _fits_collision_free(self, ep: tuple[int, int, int], size: tuple[int, int, int]) -> bool:
        x, y, z = ep
        w, d, h = size
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

    def place_at(self, ep: tuple[int, int, int], size: tuple[int, int, int]) -> bool:
        if not (self._fits_caps(ep, size) and
                self._fits_container(ep, size) and
                self._fits_collision_free(ep, size)):
            return False

        x, y, z = ep
        w, d, h = size
        k = box3d(w, d, h)
        k.set_position(x, y, z)
        self.placed.append(k)

        if hasattr(self, "tops_by_z"):
            self.tops_by_z.setdefault(k.z + k.h, []).append(k)

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

    def place_first_fit(self, base_size: tuple[int, int, int]) -> bool:
        for ep in sorted(set(self.eps), key=lambda p: (p[2], p[1], p[0])):
            for size in self._orientations(*base_size):
                if self.place_at(ep, size):
                    return True
        return False

    # ========== residual-space helpers ============

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

    def update_residual_space(self, n: box3d) -> None:
        nx1, ny1, nz1 = n.x, n.y, n.z
        nx2, ny2, nz2 = nx1 + n.w, ny1 + n.d, nz1 + n.h

        for ep in self.eps:
            ex, ey, ez = ep
            if ep not in self.ep_rs:
                self.ep_rs[ep] = (self.w - ex, self.d - ey, self.h - ez)
            x_cap, y_cap, z_cap = self.ep_rs[ep]

            if nz1 <= ez < nz2:
                if ex <= nx1 and self._is_on_side_y(ep, n):
                    x_cap = min(x_cap, max(0, nx1 - ex))
                if ey <= ny1 and self._is_on_side_x(ep, n):
                    y_cap = min(y_cap, max(0, ny1 - ey))

            if ez <= nz1 and self._is_on_side_xy(ep, n):
                z_cap = min(z_cap, max(0, nz1 - ez))

            self.ep_rs[ep] = (x_cap, y_cap, z_cap)

        eps_set = set(self.eps)
        for ep in list(self.ep_rs.keys()):
            if ep not in eps_set:
                del self.ep_rs[ep]

    def _residual_ok(self, ep: EP, size: Tuple[int, int, int]) -> bool:
        caps = self.ep_rs.get(ep)
        if not caps:
            return True
        w, d, h = size
        x_cap, y_cap, z_cap = caps
        return (w <= x_cap) and (d <= y_cap) and (h <= z_cap)

    # ======== viz ========

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

    def plot3d(self, *, annotate_eps: bool = True, save_path: str | None = None,
               show: bool = True, title: str = "Packing state") -> None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        self._setup_axes(ax, self.w, self.d, self.h, title)
        cont = Line3DCollection(self._cuboid_edges(0,0,0, self.w, self.d, self.h), linewidths=0.8)
        ax.add_collection3d(cont)
        if self.placed:
            *rest, last = self.placed
            for b in rest:
                lc = Line3DCollection(self._cuboid_edges(b.x,b.y,b.z, b.w,b.d,b.h), linewidths=1.4)
                ax.add_collection3d(lc)
            lc_last = Line3DCollection(self._cuboid_edges(last.x,last.y,last.z, last.w,last.d,last.h), linewidths=2.4)
            ax.add_collection3d(lc_last)
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
            plt.show(); plt.close(fig)
        else:
            return fig

    # ========== EP update (3DEPL) — Wall-fallback ============

    def _point_inside_box(self, p: EP, b: box3d) -> bool:
        """Closed-box test (used elsewhere)."""
        px, py, pz = p
        return (b.x <= px <= b.x + b.w and
                b.y <= py <= b.y + b.d and
                b.z <= pz <= b.z + b.h)

    @staticmethod
    def _inside_strict(p: EP, b: box3d) -> bool:
        """Strict interior: EPs that lie on faces are NOT removed."""
        px, py, pz = p
        return (b.x < px < b.x + b.w and
                b.y < py < b.y + b.d and
                b.z < pz < b.z + b.h)

    def CanTakeProjection(self, k: box3d, i: box3d, axis: str, *, strict_overlap=True) -> bool:
        """
        INWARD projections (toward origin):
        XY, XZ -> moving along -X (blocker must be left of k): ix2 <= kx1
        YX, YZ -> moving along -Y (blocker must be behind k): iy2 <= ky1
        ZX, ZY -> moving along -Z (blocker must be below k):  iz2 <= kz1
        Require overlap on the other two axes.
        """
        kx1, ky1, kz1 = k.x, k.y, k.z
        kx2, ky2, kz2 = kx1 + k.w, ky1 + k.d, kz1 + k.h

        ix1, iy1, iz1 = i.x, i.y, i.z
        ix2, iy2, iz2 = ix1 + i.w, iy1 + i.d, iz1 + i.h

        def overlap_1d(a1, a2, b1, b2):
            return not (a2 <= b1 or b2 <= a1) if strict_overlap else not (a2 < b1 or b2 < a1)

        if axis in ("XY", "XZ"):  # move -X
            ahead = ix2 <= kx1
            oy = overlap_1d(iy1, iy2, ky1, ky2)
            oz = overlap_1d(iz1, iz2, kz1, kz2)
            return ahead and oy and oz

        if axis in ("YX", "YZ"):  # move -Y
            ahead = iy2 <= ky1
            ox = overlap_1d(ix1, ix2, kx1, kx2)
            oz = overlap_1d(iz1, iz2, kz1, kz2)
            return ahead and ox and oz

        if axis in ("ZX", "ZY"):  # move -Z
            ahead = iz2 <= kz1
            ox = overlap_1d(ix1, ix2, kx1, kx2)
            oy = overlap_1d(iy1, iy2, ky1, ky2)
            return ahead and ox and oy

        raise ValueError("Invalid axis label")

    @staticmethod
    def _in_bounds_static(p: EP, W: int, D: int, H: int) -> bool:
        x, y, z = p
        return (0 <= x <= W) and (0 <= y <= D) and (0 <= z <= H)

    def update_3depl(self, k) -> List[EP]:
        """
        Wall-fallback EP update (robust):
        - Always add the 3 face-corner EPs of the placed item k.
        - For each of the 3 inward rays (from right/front/top faces toward origin),
        add the nearest blocker EP; if no blocker, fallback to the wall.
        - Remove EPs strictly inside k (face/edge EPs remain), dedup, sort (z,y,x).
        """
        xk, yk, zk = k.x, k.y, k.z
        wk, dk, hk = k.w, k.d, k.h

        W, D, H = self.w, self.d, self.h
        def _in_bounds(p: EP) -> bool:
            return self._in_bounds_static(p, W, D, H)

        def _overlap_1d(a1, a2, b1, b2):
            # open-interval overlap on projections (faces touching is fine for EPs)
            return not (a2 <= b1 or b2 <= a1)

        # survivors: keep old EPs that are NOT strictly inside k
        survivors = [ep for ep in self.eps if not self._inside_strict(ep, k)]

        cand: List[EP] = []

        # (1) Always add the three face-corner EPs
        cand.append((xk + wk, yk,      zk     ))  # right face
        cand.append((xk,      yk + dk, zk     ))  # front face
        cand.append((xk,      yk,      zk + hk))  # top face

        # Helper: find nearest blocker face along a ray; otherwise wall fallback
        # RAY 1: from R=(xk+wk, yk, zk) along -X  -> EP at (xi+wi, yk, zk) or (0, yk, zk)
        nearest_x = None
        for i in self.placed:
            if i is k: 
                continue
            ix1, iy1, iz1 = i.x, i.y, i.z
            ix2, iy2, iz2 = ix1 + i.w, iy1 + i.d, iz1 + i.h
            # blocker must be to the left (ix2 <= xk) and overlap in Y and Z with the ray line (yk, zk)
            if ix2 <= xk and _overlap_1d(yk, yk+1, iy1, iy2) and _overlap_1d(zk, zk+1, iz1, iz2):
                # nearer to R means larger ix2 (closer to xk)
                if (nearest_x is None) or (ix2 > nearest_x):
                    nearest_x = ix2
        if nearest_x is not None:
            cand.append((nearest_x, yk, zk))
        else:
            cand.append((0, yk, zk))  # wall fallback on X

        # RAY 2: from F=(xk, yk+dk, zk) along -Y -> EP at (xk, yi+di, zk) or (xk, 0, zk)
        nearest_y = None
        for i in self.placed:
            if i is k:
                continue
            ix1, iy1, iz1 = i.x, i.y, i.z
            ix2, iy2, iz2 = ix1 + i.w, iy1 + i.d, iz1 + i.h
            # blocker must be behind (iy2 <= yk) and overlap in X and Z with the ray line (xk, zk)
            if iy2 <= yk and _overlap_1d(xk, xk+1, ix1, ix2) and _overlap_1d(zk, zk+1, iz1, iz2):
                if (nearest_y is None) or (iy2 > nearest_y):
                    nearest_y = iy2
        if nearest_y is not None:
            cand.append((xk, nearest_y, zk))
        else:
            cand.append((xk, 0, zk))  # wall fallback on Y

        # RAY 3: from T=(xk, yk, zk+hk) along -Z -> EP at (xk, yk, zi+hi) or (xk, yk, 0)
        nearest_z = None
        for i in self.placed:
            if i is k:
                continue
            ix1, iy1, iz1 = i.x, i.y, i.z
            ix2, iy2, iz2 = ix1 + i.w, iy1 + i.d, iz1 + i.h
            # blocker must be below (iz2 <= zk) and overlap in X and Y with the ray line (xk, yk)
            if iz2 <= zk and _overlap_1d(xk, xk+1, ix1, ix2) and _overlap_1d(yk, yk+1, iy1, iy2):
                if (nearest_z is None) or (iz2 > nearest_z):
                    nearest_z = iz2
        if nearest_z is not None:
            cand.append((xk, yk, nearest_z))
        else:
            cand.append((xk, yk, 0))  # wall fallback on Z

        # Filter: in-bounds and not strictly inside k
        cand = [ep for ep in cand if _in_bounds(ep) and not self._inside_strict(ep, k)]

        # Merge, dedup, order (z,y,x)
        merged = survivors + cand
        self.eps = sorted(set(merged), key=lambda p: (p[2], p[1], p[0]))
        return self.eps

