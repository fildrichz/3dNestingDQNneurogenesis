# packing_core.py — Enhanced with heightmap and constraints
from typing import List, Tuple, Dict, Optional, Set
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

EP = Tuple[int, int, int]

class box3d:
    def __init__(self, w, d, h, weight: int = 0, item_id: int = -1):
        self.w, self.d, self.h = int(w), int(d), int(h)
        self.x, self.y, self.z = 0, 0, 0
        self.weight = weight
        self.item_id = item_id  # Track which item type this box is

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
    
    def get_center_of_mass(self) -> Tuple[float, float, float]:
        """Returns the center of mass (center point) of this box."""
        return (
            self.x + self.w / 2.0,
            self.y + self.d / 2.0,
            self.z + self.h / 2.0
        )


class Container(box3d):
    def __init__(self, W, D, H, max_weight: Optional[int] = None, 
                 resolution: int = 10):
        """
        Initialize container with optional weight constraint and heightmap.
        
        Args:
            W, D, H: Container dimensions
            max_weight: Maximum weight capacity (optional)
            resolution: Grid resolution for heightmap (higher = finer grid but more memory)
        """
        super().__init__(W, D, H)
        self.placed: List[box3d] = []
        self.eps: List[EP] = [(0, 0, 0)]
        self.ep_rs: Dict[EP, Tuple[int, int, int]] = {}
        
        # Weight constraint
        self.max_weight = max_weight
        self.current_weight = 0
        
        # Heightmap: 2D grid tracking maximum height at each (x,y) position
        self.resolution = resolution
        self.heightmap = np.zeros((W // resolution + 1, D // resolution + 1), dtype=np.int32)
        
        # Constraint tracking
        self.item_ids_in_bin: Set[int] = set()  # Track which item types are in this bin
        self.incompatibilities: List[Tuple[int, int]] = []
        self.positive_affinities: List[Tuple[int, int]] = []
        self.center_of_mass_constraint: Optional[Tuple[int, int]] = None

    def set_constraints(self, 
                       incompatibilities: List[Tuple[int, int]] = None,
                       positive_affinities: List[Tuple[int, int]] = None,
                       center_of_mass: Optional[Tuple[int, int]] = None):
        """Set problem constraints."""
        if incompatibilities:
            self.incompatibilities = incompatibilities
        if positive_affinities:
            self.positive_affinities = positive_affinities
        if center_of_mass:
            self.center_of_mass_constraint = center_of_mass

    def get_heightmap_at(self, x: int, y: int) -> int:
        """Get the height at a specific (x, y) coordinate."""
        gx = min(x // self.resolution, self.heightmap.shape[0] - 1)
        gy = min(y // self.resolution, self.heightmap.shape[1] - 1)
        return int(self.heightmap[gx, gy])
    
    def update_heightmap(self, box: box3d) -> None:
        """Update heightmap after placing a box."""
        x1, y1 = box.x, box.y
        x2, y2 = box.x + box.w, box.y + box.d
        new_height = box.z + box.h
        
        # Update all grid cells covered by this box
        gx1 = x1 // self.resolution
        gy1 = y1 // self.resolution
        gx2 = min((x2 - 1) // self.resolution + 1, self.heightmap.shape[0])
        gy2 = min((y2 - 1) // self.resolution + 1, self.heightmap.shape[1])
        
        self.heightmap[gx1:gx2, gy1:gy2] = np.maximum(
            self.heightmap[gx1:gx2, gy1:gy2], 
            new_height
        )
    
    def get_heightmap_stats(self) -> Dict:
        """Get statistics about the current heightmap."""
        return {
            'max_height': int(np.max(self.heightmap)),
            'min_height': int(np.min(self.heightmap)),
            'avg_height': float(np.mean(self.heightmap)),
            'std_height': float(np.std(self.heightmap))
        }
    
    def visualize_heightmap(self, save_path: Optional[str] = None) -> None:
        """Visualize the heightmap as a 2D heatmap."""
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(self.heightmap.T, origin='lower', cmap='viridis', 
                       extent=[0, self.w, 0, self.d])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Container Heightmap')
        plt.colorbar(im, ax=ax, label='Height')
        
        if save_path:
            fig.savefig(save_path, dpi=140, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
            plt.close(fig)

    def check_weight_constraint(self, additional_weight: int) -> bool:
        """Check if adding a box would violate weight constraint."""
        if self.max_weight is None:
            return True
        return (self.current_weight + additional_weight) <= self.max_weight
    
    def check_incompatibility(self, item_id: int) -> bool:
        """Check if placing this item would violate incompatibility constraints."""
        for a, b in self.incompatibilities:
            if item_id == a and b in self.item_ids_in_bin:
                return False
            if item_id == b and a in self.item_ids_in_bin:
                return False
        return True
    
    def get_center_of_mass(self) -> Tuple[float, float]:
        """Calculate the current center of mass (x, y) of all placed items."""
        if not self.placed:
            return (0.0, 0.0)
        
        total_weight = 0
        weighted_x = 0.0
        weighted_y = 0.0
        
        for box in self.placed:
            cx, cy, _ = box.get_center_of_mass()
            weighted_x += cx * box.weight
            weighted_y += cy * box.weight
            total_weight += box.weight
        
        if total_weight == 0:
            return (0.0, 0.0)
        
        return (weighted_x / total_weight, weighted_y / total_weight)
    
    def check_center_of_mass_constraint(self) -> bool:
        """Check if current center of mass satisfies the constraint."""
        if self.center_of_mass_constraint is None:
            return True
        
        cx, cy = self.get_center_of_mass()
        target_x, target_y = self.center_of_mass_constraint
        
        # Allow some tolerance (e.g., within 10% of container size)
        tolerance = min(self.w, self.d) * 0.1
        
        return (abs(cx - target_x) <= tolerance and 
                abs(cy - target_y) <= tolerance)

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

    def place_at(self, ep: tuple[int, int, int], size: tuple[int, int, int], 
                 weight: int = 0, item_id: int = -1) -> bool:
        """
        Place a box at the given extreme point with constraint checking.
        
        Args:
            ep: Extreme point (x, y, z)
            size: Box dimensions (w, d, h)
            weight: Weight of the box
            item_id: Item type identifier
            
        Returns:
            True if placement successful, False otherwise
        """
        # Check all constraints
        if not (self._fits_caps(ep, size) and
                self._fits_container(ep, size) and
                self._fits_collision_free(ep, size)):
            return False
        
        # Check weight constraint
        if not self.check_weight_constraint(weight):
            return False
        
        # Check incompatibility constraint
        if not self.check_incompatibility(item_id):
            return False

        x, y, z = ep
        w, d, h = size
        k = box3d(w, d, h, weight, item_id)
        k.set_position(x, y, z)
        self.placed.append(k)
        
        # Update tracking
        self.current_weight += weight
        if item_id >= 0:
            self.item_ids_in_bin.add(item_id)
        
        # Update heightmap
        self.update_heightmap(k)

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

    def place_first_fit(self, base_size: tuple[int, int, int], 
                       weight: int = 0, item_id: int = -1) -> bool:
        """Try to place a box using first-fit strategy."""
        for ep in sorted(set(self.eps), key=lambda p: (p[2], p[1], p[0])):
            for size in self._orientations(*base_size):
                if self.place_at(ep, size, weight, item_id):
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
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        self._setup_axes(ax, self.w, self.d, self.h, title)
        cont = Line3DCollection(self._cuboid_edges(0,0,0, self.w, self.d, self.h), 
                               linewidths=0.8, colors='black')
        ax.add_collection3d(cont)
        
        if self.placed:
            # Color boxes by item_id if available
            colors = plt.cm.tab20(np.linspace(0, 1, 20))
            for b in self.placed:
                color = colors[b.item_id % 20] if b.item_id >= 0 else 'blue'
                lc = Line3DCollection(self._cuboid_edges(b.x,b.y,b.z, b.w,b.d,b.h), 
                                     linewidths=1.4, colors=color)
                ax.add_collection3d(lc)
        
        if self.eps and annotate_eps:
            xs, ys, zs = zip(*self.eps)
            ax.scatter(xs, ys, zs, s=28, c='red', marker='o')
            for (x,y,z) in self.eps[:10]:  # Limit annotations for clarity
                ax.text(x, y, z, f"({x},{y},{z})", fontsize=7)
        
        # Add weight and COM info
        info_text = f"Weight: {self.current_weight}"
        if self.max_weight:
            info_text += f"/{self.max_weight}"
        if self.placed:
            cx, cy = self.get_center_of_mass()
            info_text += f"\nCoM: ({cx:.1f}, {cy:.1f})"
        ax.text2D(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        if save_path:
            fig.savefig(save_path, dpi=140, bbox_inches="tight")
            plt.close(fig)
        elif show:
            plt.show()
            plt.close(fig)
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
        EP update (paper-accurate, wall-fallback, NO corner-EPs):
        From each face-corner of k, cast two inward rays.
        """
        xk, yk, zk = k.x, k.y, k.z
        wk, dk, hk = k.w, k.d, k.h

        W, D, H = self.w, self.d, self.h

        def in_bounds(p):
            x, y, z = p
            return 0 <= x <= W and 0 <= y <= D and 0 <= z <= H

        def inside_strict(p, b):
            px, py, pz = p
            return (b.x < px < b.x + b.w and
                    b.y < py < b.y + b.d and
                    b.z < pz < b.z + b.h)

        def line_overlaps_interval(v, a1, a2):
            return not ((v + 1) <= a1 or a2 <= v)

        survivors = [ep for ep in self.eps if not inside_strict(ep, k)]

        def nearest_blocker_on_ray(axis, x0, y0, z0):
            best = None
            for i in self.placed:
                if i is k: 
                    continue
                ix1, iy1, iz1 = i.x, i.y, i.z
                ix2, iy2, iz2 = ix1 + i.w, iy1 + i.d, iz1 + i.h

                if axis == 'negX':
                    if ix2 <= x0 and line_overlaps_interval(y0, iy1, iy2) and line_overlaps_interval(z0, iz1, iz2):
                        if best is None or ix2 > best: 
                            best = ix2
                elif axis == 'negY':
                    if iy2 <= y0 and line_overlaps_interval(x0, ix1, ix2) and line_overlaps_interval(z0, iz1, iz2):
                        if best is None or iy2 > best: 
                            best = iy2
                elif axis == 'negZ':
                    if iz2 <= z0 and line_overlaps_interval(x0, ix1, ix2) and line_overlaps_interval(y0, iy1, iy2):
                        if best is None or iz2 > best: 
                            best = iz2
            return best

        cand = []

        # R = (xk+wk, yk, zk): project along -Y and -Z
        Rx, Ry, Rz = xk + wk, yk, zk
        nbY = nearest_blocker_on_ray('negY', Rx, Ry, Rz)
        cand.append((Rx, nbY if nbY is not None else 0,  Rz))
        nbZ = nearest_blocker_on_ray('negZ', Rx, Ry, Rz)
        cand.append((Rx, Ry,  nbZ if nbZ is not None else 0))

        # F = (xk, yk+dk, zk): project along -X and -Z
        Fx, Fy, Fz = xk, yk + dk, zk
        nbX = nearest_blocker_on_ray('negX', Fx, Fy, Fz)
        cand.append((nbX if nbX is not None else 0,  Fy, Fz))
        nbZ = nearest_blocker_on_ray('negZ', Fx, Fy, Fz)
        cand.append((Fx, Fy,  nbZ if nbZ is not None else 0))

        # T = (xk, yk, zk+hk): project along -X and -Y
        Tx, Ty, Tz = xk, yk, zk + hk
        nbX = nearest_blocker_on_ray('negX', Tx, Ty, Tz)
        cand.append((nbX if nbX is not None else 0,  Ty, Tz))
        nbY = nearest_blocker_on_ray('negY', Tx, Ty, Tz)
        cand.append((Tx, nbY if nbY is not None else 0,  Tz))

        cand = [ep for ep in cand if in_bounds(ep) and not inside_strict(ep, k)]

        merged = survivors + cand
        self.eps = sorted(set(merged), key=lambda p: (p[2], p[1], p[0]))
        return self.eps

    def _pareto_prune_eps(self, points: list[tuple[int,int,int]]) -> list[tuple[int,int,int]]:
        """Keep only non-dominated EPs in the MIN sense over (x,y,z)."""
        pts = list(set(points))

        def dominates(a, b):
            return ((a[0] >= b[0]) and (a[1] >= b[1]) and (a[2] >= b[2]))

        front = []
        for i, p in enumerate(pts):
            dominated = False
            for j, q in enumerate(pts):
                if i != j and dominates(q, p):
                    dominated = True
                    break
            if not dominated:
                front.append(p)

        front.sort(key=lambda t: (t[2], t[1], t[0]))
        return front