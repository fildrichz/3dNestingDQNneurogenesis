from typing import List, Tuple, Dict, Optional, Set
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from .q4realbpp_loader import Item, Bin, Q4RealBPPInstance


EP = Tuple[int, int, int]


class Box3D:
    """Extended box with weight and category for Q4RealBPP"""
    def __init__(self, w, d, h, weight: float = 0.0, category: int = 0, item_id: int = -1):
        self.w, self.d, self.h = int(w), int(d), int(h)
        self.x, self.y, self.z = 0, 0, 0
        self.weight = weight
        self.category = category
        self.item_id = item_id

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


class Q4RealBPPContainer:
    """Container with real-world constraints from Q4RealBPP"""
    def __init__(self, W, D, H, max_weight: float = float('inf'), bin_id: int = 0):
        self.w, self.d, self.h = int(W), int(D), int(H)
        self.max_weight = max_weight
        self.bin_id = bin_id
        self.placed: List[Box3D] = []
        self.eps: List[EP] = [(0, 0, 0)]
        self.ep_rs: Dict[EP, Tuple[int, int, int]] = {}
        self.current_weight = 0.0
        
        # For constraint checking
        self.affinities: Dict[Tuple[int, int], int] = {}
        self.load_bearing_ratio: float = 1.5
        self.load_balancing: bool = False
        
        # Track items by category
        self.categories_in_bin: Set[int] = set()

    def set_constraints(self, affinities: Dict[Tuple[int, int], int], 
                       load_bearing_ratio: float, load_balancing: bool):
        """Set problem constraints"""
        self.affinities = affinities
        self.load_bearing_ratio = load_bearing_ratio
        self.load_balancing = load_balancing

    # ========== Constraint checking methods ==========

    def check_weight_constraint(self, item_weight: float) -> bool:
        """Check if adding this item would exceed max weight"""
        return (self.current_weight + item_weight) <= self.max_weight

    def check_affinity_constraint(self, category: int) -> bool:
        """
        Check affinity constraints:
        - If category has positive affinity with existing categories, OK
        - If category has negative affinity (incompatible), NOT OK
        """
        for existing_cat in self.categories_in_bin:
            key = tuple(sorted([category, existing_cat]))
            if key in self.affinities:
                aff_type = self.affinities[key]
                if aff_type == -1:  # Incompatible
                    return False
        return True

    def check_load_bearing_constraint(self, new_box: Box3D, ep: EP) -> bool:
        """
        Check load bearing: heavy items cannot be placed on lighter items.
        If weight_new / weight_below > load_bearing_ratio, placement is invalid.
        """
        nx1, ny1, nz1 = ep
        nx2, ny2, nz2 = nx1 + new_box.w, ny1 + new_box.d, nz1 + new_box.h
        
        # Check all items that would be below this item (same footprint, lower Z)
        for placed in self.placed:
            px1, py1, pz1 = placed.x, placed.y, placed.z
            px2, py2, pz2 = px1 + placed.w, py1 + placed.d, pz1 + placed.h
            
            # Check if placed item is directly below (overlaps in X-Y, and is lower in Z)
            if pz2 <= nz1:  # Below or touching from below
                # Check X-Y overlap
                x_overlap = not (nx2 <= px1 or px2 <= nx1)
                y_overlap = not (ny2 <= py1 or py2 <= ny1)
                
                if x_overlap and y_overlap and placed.weight > 0:
                    # Check load bearing ratio
                    weight_ratio = new_box.weight / placed.weight
                    if weight_ratio > self.load_bearing_ratio:
                        return False
        
        return True

    def get_load_balance_penalty(self, new_weight: float) -> float:
        """
        Calculate load balance penalty (for multi-bin scenarios).
        Returns a penalty score - lower is better.
        """
        new_total = self.current_weight + new_weight
        # Simple penalty: how far from ideal weight
        return abs(new_total)

    # ========== Base placement methods (from packing_core.py) ==========

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

    def can_place_at(self, ep: tuple[int, int, int], size: tuple[int, int, int],
                     weight: float = 0.0, category: int = 0) -> bool:
        """Check if placement is feasible with all constraints"""
        # Check geometric constraints
        if not (self._fits_caps(ep, size) and
                self._fits_container(ep, size) and
                self._fits_collision_free(ep, size)):
            return False
        
        # Check weight constraint
        if not self.check_weight_constraint(weight):
            return False
        
        # Check affinity constraint
        if not self.check_affinity_constraint(category):
            return False
        
        # Check load bearing constraint
        w, d, h = size
        temp_box = Box3D(w, d, h, weight, category)
        if not self.check_load_bearing_constraint(temp_box, ep):
            return False
        
        return True

    def place_at(self, ep: tuple[int, int, int], size: tuple[int, int, int],
                 weight: float = 0.0, category: int = 0, item_id: int = -1) -> bool:
        """Place item at EP with all constraint checking"""
        if not self.can_place_at(ep, size, weight, category):
            return False

        x, y, z = ep
        w, d, h = size
        k = Box3D(w, d, h, weight, category, item_id)
        k.set_position(x, y, z)
        self.placed.append(k)
        
        # Update weight and categories
        self.current_weight += weight
        self.categories_in_bin.add(category)

        self.update_3depl(k)
        self.update_residual_space(k)
        return True

    def place_item(self, item: Item) -> bool:
        """Try to place an item using first-fit with all orientations"""
        for ep in sorted(set(self.eps), key=lambda p: (p[2], p[1], p[0])):
            for w, d, h in self._orientations(int(item.width), int(item.depth), int(item.height)):
                if self.place_at(ep, (w, d, h), item.weight, item.category, item.id):
                    return True
        return False

    # ========== Residual space methods ==========

    @staticmethod
    def _in_halfopen(v: int, a: int, b: int) -> bool:
        return a <= v < b

    def _is_on_side_x(self, ep: EP, it: Box3D) -> bool:
        ex, _, _ = ep
        return self._in_halfopen(ex, it.x, it.x + it.w)

    def _is_on_side_y(self, ep: EP, it: Box3D) -> bool:
        _, ey, _ = ep
        return self._in_halfopen(ey, it.y, it.y + it.d)

    def _is_on_side_xy(self, ep: EP, it: Box3D) -> bool:
        ex, ey, _ = ep
        return (self._in_halfopen(ex, it.x, it.x + it.w) and
                self._in_halfopen(ey, it.y, it.y + it.d))

    def update_residual_space(self, n: Box3D) -> None:
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

    # ========== EP update (3DEPL) ==========

    @staticmethod
    def _inside_strict(p: EP, b: Box3D) -> bool:
        px, py, pz = p
        return (b.x < px < b.x + b.w and
                b.y < py < b.y + b.d and
                b.z < pz < b.z + b.h)

    def update_3depl(self, k) -> List[EP]:
        """EP update with wall-fallback (from packing_core.py)"""
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

        # R, F, T projections (from packing_core.py)
        Rx, Ry, Rz = xk + wk, yk, zk
        nbY = nearest_blocker_on_ray('negY', Rx, Ry, Rz)
        cand.append((Rx, nbY if nbY is not None else 0, Rz))
        nbZ = nearest_blocker_on_ray('negZ', Rx, Ry, Rz)
        cand.append((Rx, Ry, nbZ if nbZ is not None else 0))

        Fx, Fy, Fz = xk, yk + dk, zk
        nbX = nearest_blocker_on_ray('negX', Fx, Fy, Fz)
        cand.append((nbX if nbX is not None else 0, Fy, Fz))
        nbZ = nearest_blocker_on_ray('negZ', Fx, Fy, Fz)
        cand.append((Fx, Fy, nbZ if nbZ is not None else 0))

        Tx, Ty, Tz = xk, yk, zk + hk
        nbX = nearest_blocker_on_ray('negX', Tx, Ty, Tz)
        cand.append((nbX if nbX is not None else 0, Ty, Tz))
        nbY = nearest_blocker_on_ray('negY', Tx, Ty, Tz)
        cand.append((Tx, nbY if nbY is not None else 0, Tz))

        cand = [ep for ep in cand if in_bounds(ep) and not inside_strict(ep, k)]
        merged = survivors + cand
        self.eps = sorted(set(merged), key=lambda p: (p[2], p[1], p[0]))
        return self.eps

    # ========== Visualization ==========

    @staticmethod
    def _cuboid_edges(x, y, z, w, d, h):
        X = [x, x+w, x+w, x, x, x+w, x+w, x]
        Y = [y, y, y+d, y+d, y, y, y+d, y+d]
        Z = [z, z, z, z, z+h, z+h, z+h, z+h]
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        return [[(X[i],Y[i],Z[i]), (X[j],Y[j],Z[j])] for i,j in edges]

    @staticmethod
    def _setup_axes(ax, W, D, H, title=""):
        ax.set_xlim(0, W); ax.set_ylim(0, D); ax.set_zlim(0, H)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_box_aspect((W, D, H))
        if title: ax.set_title(title)

    def plot3d(self, *, annotate_eps: bool = True, save_path: str | None = None,
               show: bool = True, title: str | None = None) -> None:
        if title is None:
            title = f"Bin {self.bin_id} - Weight: {self.current_weight:.1f}/{self.max_weight:.1f}"
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        self._setup_axes(ax, self.w, self.d, self.h, title)
        
        # Draw container
        cont = Line3DCollection(self._cuboid_edges(0,0,0, self.w, self.d, self.h), 
                               linewidths=0.8, colors='black')
        ax.add_collection3d(cont)
        
        # Draw placed items with different colors per category
        import matplotlib.cm as cm
        if self.placed:
            categories = list(set(b.category for b in self.placed))
            colors = cm.rainbow([c / max(categories) if categories else 0 for c in categories])
            color_map = dict(zip(categories, colors))
            
            for b in self.placed:
                color = color_map[b.category]
                lc = Line3DCollection(self._cuboid_edges(b.x,b.y,b.z, b.w,b.d,b.h), 
                                     linewidths=1.4, colors=[color])
                ax.add_collection3d(lc)
                # Add item ID label
                cx, cy, cz = b.x + b.w/2, b.y + b.d/2, b.z + b.h/2
                ax.text(cx, cy, cz, f"{b.item_id}", fontsize=6, ha='center')
        
        # Draw EPs
        if self.eps:
            xs, ys, zs = zip(*self.eps)
            ax.scatter(xs, ys, zs, s=28, c='red', marker='o')
            if annotate_eps:
                for (x,y,z) in self.eps:
                    ax.text(x, y, z, f"({x},{y},{z})", fontsize=6)
        
        if save_path:
            fig.savefig(save_path, dpi=140, bbox_inches="tight")
            plt.close(fig)
        elif show:
            plt.show()
            plt.close(fig)
        else:
            return fig


def solve_q4realbpp_instance(instance: Q4RealBPPInstance, 
                             visualize: bool = True) -> Dict[int, Q4RealBPPContainer]:
    """
    Simple first-fit solver for Q4RealBPP instances.
    Returns dict of {bin_id: container} with items packed.
    """
    # Create containers from bins
    containers = {}
    for bin_spec in instance.bins:
        container = Q4RealBPPContainer(
            int(bin_spec.width), 
            int(bin_spec.depth), 
            int(bin_spec.height),
            bin_spec.max_weight,
            bin_spec.id
        )
        container.set_constraints(
            instance.affinities,
            instance.load_bearing_ratio,
            instance.load_balancing
        )
        containers[bin_spec.id] = container
    
    # Sort items by weight (heavy first - helps with load bearing)
    sorted_items = sorted(instance.items, key=lambda x: -x.weight)
    
    # First-fit packing
    unpacked = []
    for item in sorted_items:
        placed = False
        for bin_id in sorted(containers.keys()):
            if containers[bin_id].place_item(item):
                placed = True
                print(f"Item {item.id} (cat {item.category}, {item.weight:.1f}kg) -> Bin {bin_id}")
                break
        if not placed:
            unpacked.append(item)
            print(f"Item {item.id} could not be packed!")
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Packing Summary for '{instance.name}'")
    print(f"{'='*50}")
    print(f"Total items: {len(instance.items)}")
    print(f"Packed items: {len(instance.items) - len(unpacked)}")
    print(f"Unpacked items: {len(unpacked)}")
    
    for bin_id, container in containers.items():
        util = (sum(b.w * b.d * b.h for b in container.placed) / 
                (container.w * container.d * container.h) * 100)
        print(f"\nBin {bin_id}:")
        print(f"  Items: {len(container.placed)}")
        print(f"  Weight: {container.current_weight:.1f} / {container.max_weight:.1f} kg")
        print(f"  Volume utilization: {util:.1f}%")
        print(f"  Categories: {container.categories_in_bin}")
    
    # Visualize
    if visualize:
        for bin_id, container in containers.items():
            if container.placed:
                container.plot3d()
    
    return containers


def main():
    """Example usage"""
    from q4realbpp_loader import Q4RealBPPLoader
    
    # Generate a sample instance
    print("Generating sample Q4RealBPP instance...")
    instance = Q4RealBPPLoader.generate_sample_instance(num_items=25, num_bins=2)
    
    print(f"\nInstance details:")
    print(f"  Name: {instance.name}")
    print(f"  Items: {len(instance.items)}")
    print(f"  Bins: {len(instance.bins)}")
    print(f"  Affinities: {instance.affinities}")
    print(f"  Load bearing ratio: {instance.load_bearing_ratio}")
    
    # Solve
    print(f"\n{'='*50}")
    print("Starting packing...")
    print(f"{'='*50}\n")
    containers = solve_q4realbpp_instance(instance, visualize=False)
    
    print(f"\nDone!")


if __name__ == "__main__":
    main()
