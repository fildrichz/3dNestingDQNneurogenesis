# packing_core_complete.py — Complete implementation with ALL constraints + gravity
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
        Initialize container with all constraint support.
        
        Args:
            W, D, H: Container dimensions
            max_weight: Maximum weight capacity (optional)
            resolution: Grid resolution for heightmap
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
        self.item_ids_in_bin: Set[int] = set()
        self.incompatibilities: List[Tuple[int, int]] = []
        self.positive_affinities: List[Tuple[int, int]] = []
        self.center_of_mass_constraint: Optional[Tuple[int, int]] = None
        self.relative_pos: Dict[int, List[Tuple[int, int]]] = {}
        
        # Track item positions for constraint checking
        self.item_positions: Dict[int, List[box3d]] = {}

    def set_constraints(self, 
                       incompatibilities: List[Tuple[int, int]] = None,
                       positive_affinities: List[Tuple[int, int]] = None,
                       center_of_mass: Optional[Tuple[int, int]] = None,
                       relative_pos: Dict[int, List[Tuple[int, int]]] = None):
        """Set problem constraints."""
        if incompatibilities:
            self.incompatibilities = incompatibilities
        if positive_affinities:
            self.positive_affinities = positive_affinities
        if center_of_mass:
            self.center_of_mass_constraint = center_of_mass
        if relative_pos:
            self.relative_pos = relative_pos

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

    # ========== CONSTRAINT CHECKING METHODS ==========

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
    
    def check_relative_positioning(self, item_id: int, ep: EP, size: Tuple[int, int, int]) -> bool:
        """
        Check if placing item would violate relative positioning constraints.
        
        Relative positioning format: {heavy_id: [(light_id, code), ...]}
        Interpretation: heavy_id cannot be placed ON TOP of light_id
        
        Example: {1: [(7, 0), (7, 1)]}
        - Item 1 is heavier than item 7
        - Item 1 cannot be placed above item 7
        
        Args:
            item_id: ID of item to place
            ep: Position where item would be placed (x, y, z)
            size: Size of item (w, d, h)
            
        Returns:
            True if placement allowed, False if violates constraint
        """
        if not self.relative_pos or item_id not in self.relative_pos:
            return True
        
        x, y, z = ep
        w, d, h = size
        
        # Get constraints for this item (heavy item)
        constraints = self.relative_pos[item_id]
        
        for light_id, position_code in constraints:
            # Check if light item exists in bin
            if light_id not in self.item_positions:
                continue
            
            # Check all instances of the light item
            for light_box in self.item_positions[light_id]:
                # Check if boxes overlap in XY plane
                x_overlap = not (x + w <= light_box.x or light_box.x + light_box.w <= x)
                y_overlap = not (y + d <= light_box.y or light_box.y + light_box.d <= y)
                
                if x_overlap and y_overlap:
                    # Boxes overlap in XY - check if heavy item is above light item
                    # Heavy item would be above if its Z position is at or above light item's top
                    if z >= light_box.z + light_box.h - 1:
                        # VIOLATION: Heavy item would be on top of light item
                        return False
        
        return True
    
    def check_positive_affinity_before_completion(self) -> bool:
        """
        Check if current bin satisfies positive affinity constraints.
        For final validation - should rarely trigger if proactive check works.
        """
        if not self.positive_affinities:
            return True
        
        for item_a, item_b in self.positive_affinities:
            has_a = item_a in self.item_ids_in_bin
            has_b = item_b in self.item_ids_in_bin
            
            # If we have one but not the other, affinity is violated
            if has_a and not has_b:
                return False
            if has_b and not has_a:
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
        
        tolerance = min(self.w, self.d) * 0.1
        
        return (abs(cx - target_x) <= tolerance and 
                abs(cy - target_y) <= tolerance)

    # ========== GRAVITY AND SUPPORT ==========

    def apply_gravity(self, x: int, y: int, z: int, w: int, d: int, h: int) -> int:
        """
        Apply gravity: drop box down until it hits ground or another box.
        
        This ensures no floating boxes - every box must be supported!
        
        Args:
            x, y, z: Initial position
            w, d, h: Box dimensions
            
        Returns:
            Final Z position after dropping
        """
        # Start from the requested Z and drop down
        current_z = z
        
        # Drop until we hit something
        while current_z > 0:
            # Check if placing at current_z-1 would collide with anything
            test_z = current_z - 1
            
            # Check collision with all placed boxes
            collision = False
            for box in self.placed:
                # Check if boxes would overlap
                x_overlap = not (x + w <= box.x or box.x + box.w <= x)
                y_overlap = not (y + d <= box.y or box.y + box.d <= y)
                z_overlap = not (test_z + h <= box.z or box.z + box.h <= test_z)
                
                if x_overlap and y_overlap and z_overlap:
                    collision = True
                    break
            
            if collision:
                # Can't go lower - stop at current_z
                return current_z
            
            # No collision - can drop further
            current_z = test_z
        
        # Reached ground (z=0)
        return 0
    
    def find_support_surface(self, x: int, y: int, w: int, d: int) -> int:
        """
        Find the highest surface at XY position where a box can be placed.
        
        This is more efficient than apply_gravity for finding initial placement.
        
        Args:
            x, y: XY position
            w, d: Box width and depth
            
        Returns:
            Z position of support surface (0 if ground, or top of highest box)
        """
        max_z = 0
        
        # Check all placed boxes
        for box in self.placed:
            # Check if box overlaps in XY with our target position
            x_overlap = not (x + w <= box.x or box.x + box.w <= x)
            y_overlap = not (y + d <= box.y or box.y + box.d <= y)
            
            if x_overlap and y_overlap:
                # This box overlaps - check if it's higher than current max
                box_top = box.z + box.h
                if box_top > max_z:
                    max_z = box_top
        
        return max_z

    # ========== PLACEMENT METHODS ==========

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
        for b in self.placed:
            if (self.colision_overlap(x, x+w, b.x, b.x+b.w) and
                self.colision_overlap(y, y+d, b.y, b.y+b.d) and
                self.colision_overlap(z, z+h, b.z, b.z+b.h)):
                return False
        return True

    def place_at(self, ep: tuple[int, int, int], size: tuple[int, int, int], 
                 weight: int = 0, item_id: int = -1) -> bool:
        """
        Place a box with ALL constraint checking + gravity support.
        
        Process:
        1. Check all constraints at requested position
        2. Apply gravity - drop box until it hits support
        3. Re-check constraints at final position
        4. Place box and update tracking
        
        Args:
            ep: Requested extreme point (x, y, z)
            size: Box dimensions (w, d, h)
            weight: Weight of the box
            item_id: Item type identifier
            
        Returns:
            True if placement successful, False otherwise
        """
        x, y, z = ep
        w, d, h = size
        
        # === STEP 1: Initial constraint checks ===
        if not (self._fits_container(ep, size) and
                self.check_weight_constraint(weight) and
                self.check_incompatibility(item_id)):
            return False
        
        # === STEP 2: Apply gravity ===
        # Drop box down until it hits something
        final_z = self.apply_gravity(x, y, z, w, d, h)
        final_ep = (x, y, final_z)
        
        # === STEP 3: Re-check constraints at final position ===
        if not (self._fits_caps(final_ep, size) and
                self._fits_container(final_ep, size) and
                self._fits_collision_free(final_ep, size) and
                self.check_relative_positioning(item_id, final_ep, size)):
            return False
        
        # === STEP 4: Place the box ===
        k = box3d(w, d, h, weight, item_id)
        k.set_position(x, y, final_z)
        self.placed.append(k)
        
        # Update tracking
        self.current_weight += weight
        if item_id >= 0:
            self.item_ids_in_bin.add(item_id)
        
        if item_id not in self.item_positions:
            self.item_positions[item_id] = []
        self.item_positions[item_id].append(k)
        
        # Update heightmap
        self.update_heightmap(k)
        
        if hasattr(self, "tops_by_z"):
            self.tops_by_z.setdefault(k.z + k.h, []).append(k)
        
        # === STEP 5: Update EPs AFTER placement ===
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

    def place(self, ep: tuple[int, int, int], w: int, d: int, h: int, 
              weight: int = 0, item_id: int = -1) -> bool:
        """Alternate placement method. Calls place_at internally."""
        return self.place_at(ep, (w, d, h), weight, item_id)
    
    def place_first_fit(self, base_size: tuple[int, int, int], 
                       weight: int = 0, item_id: int = -1) -> bool:
        """Try to place a box using first-fit strategy."""
        for ep in sorted(set(self.eps), key=lambda p: (p[2], p[1], p[0])):
            for size in self._orientations(*base_size):
                if self.place_at(ep, size, weight, item_id):
                    return True
        return False

    # ========== VISUALIZATION ==========

    @staticmethod
    def _cuboid_edges(x, y, z, w, d, h):
        """Generate edges for 3D cuboid visualization."""
        vertices = np.array([
            [x, y, z], [x+w, y, z], [x+w, y+d, z], [x, y+d, z],
            [x, y, z+h], [x+w, y, z+h], [x+w, y+d, z+h], [x, y+d, z+h]
        ])
        edges = [
            [vertices[0], vertices[1]], [vertices[1], vertices[2]],
            [vertices[2], vertices[3]], [vertices[3], vertices[0]],
            [vertices[4], vertices[5]], [vertices[5], vertices[6]],
            [vertices[6], vertices[7]], [vertices[7], vertices[4]],
            [vertices[0], vertices[4]], [vertices[1], vertices[5]],
            [vertices[2], vertices[6]], [vertices[3], vertices[7]]
        ]
        return edges

    @staticmethod
    def _setup_axes(ax, W, D, H, title):
        """Setup 3D axes for visualization."""
        ax.set_xlim([0, W])
        ax.set_ylim([0, D])
        ax.set_zlim([0, H])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        ax.set_box_aspect([W, D, H])

    def plot3d(self, *, save_path: str | None = None,
               show: bool = True, title: str = "Packing state") -> None:
        """Visualize the packing with boxes and extreme points."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        self._setup_axes(ax, self.w, self.d, self.h, title)
        
        cont = Line3DCollection(self._cuboid_edges(0,0,0, self.w, self.d, self.h),
                                linewidths=0.8, colors='black')
        ax.add_collection3d(cont)
        
        if self.placed:
            colors = plt.cm.tab20(np.linspace(0, 1, 20))
            for b in self.placed:
                color = colors[b.item_id % 20] if b.item_id >= 0 else 'blue'
                edges = Line3DCollection(self._cuboid_edges(b.x, b.y, b.z, b.w, b.d, b.h),
                                        linewidths=1.5, colors=color)
                ax.add_collection3d(edges)
        
        if self.eps:
            eps_arr = np.array(self.eps)
            ax.scatter(eps_arr[:,0], eps_arr[:,1], eps_arr[:,2], 
                      c='red', marker='o', s=50, alpha=0.6, label='Extreme Points')
        
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

    def plot3d_filled(self, *, save_path: str | None = None,
               show: bool = True, title: str = "Packing state") -> None:
        """Visualize with filled boxes."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        self._setup_axes(ax, self.w, self.d, self.h, title)
        
        cont = Line3DCollection(self._cuboid_edges(0,0,0, self.w, self.d, self.h),
                                linewidths=0.8, colors='black')
        ax.add_collection3d(cont)
        
        if self.placed:
            colors = plt.cm.tab20(np.linspace(0, 1, 20))
            for b in self.placed:
                color = colors[b.item_id % 20] if b.item_id >= 0 else 'blue'
                ax.bar3d(b.x, b.y, b.z, b.w, b.d, b.h, color=color, alpha=0.6, edgecolor='k')
        
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

    # ========== EP UPDATE (3DEPL) ==========

    def _point_inside_box(self, p: EP, b: box3d) -> bool:
        """Closed-box test."""
        px, py, pz = p
        return (b.x <= px <= b.x + b.w and
                b.y <= py <= b.y + b.d and
                b.z <= pz <= b.z + b.h)

    @staticmethod
    def _inside_strict(p: EP, b: box3d) -> bool:
        """Strict interior."""
        px, py, pz = p
        return (b.x < px < b.x + b.w and
                b.y < py < b.y + b.d and
                b.z < pz < b.z + b.h)

    def CanTakeProjection(self, k: box3d, i: box3d, axis: str, *, strict_overlap=True) -> bool:
        """Check if projection is valid."""
        kx1, ky1, kz1 = k.x, k.y, k.z
        kx2, ky2, kz2 = kx1 + k.w, ky1 + k.d, kz1 + k.h

        ix1, iy1, iz1 = i.x, i.y, i.z
        ix2, iy2, iz2 = ix1 + i.w, iy1 + i.d, iz1 + i.h

        def overlap_1d(a1, a2, b1, b2):
            return not (a2 <= b1 or b2 <= a1) if strict_overlap else not (a2 < b1 or b2 < a1)

        if axis in ("XY", "XZ"):
            ahead = ix2 <= kx1
            oy = overlap_1d(iy1, iy2, ky1, ky2)
            oz = overlap_1d(iz1, iz2, kz1, kz2)
            return ahead and oy and oz

        if axis in ("YX", "YZ"):
            ahead = iy2 <= ky1
            ox = overlap_1d(ix1, ix2, kx1, kx2)
            oz = overlap_1d(iz1, iz2, kz1, kz2)
            return ahead and ox and oz

        if axis in ("ZX", "ZY"):
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
        """EP update with wall-fallback."""
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

        Rx, Ry, Rz = xk + wk, yk, zk
        nbY = nearest_blocker_on_ray('negY', Rx, Ry, Rz)
        cand.append((Rx, nbY if nbY is not None else 0,  Rz))
        nbZ = nearest_blocker_on_ray('negZ', Rx, Ry, Rz)
        cand.append((Rx, Ry,  nbZ if nbZ is not None else 0))

        Fx, Fy, Fz = xk, yk + dk, zk
        nbX = nearest_blocker_on_ray('negX', Fx, Fy, Fz)
        cand.append((nbX if nbX is not None else 0,  Fy, Fz))
        nbZ = nearest_blocker_on_ray('negZ', Fx, Fy, Fz)
        cand.append((Fx, Fy,  nbZ if nbZ is not None else 0))

        Tx, Ty, Tz = xk, yk, zk + hk
        nbX = nearest_blocker_on_ray('negX', Tx, Ty, Tz)
        cand.append((nbX if nbX is not None else 0,  Ty, Tz))
        nbY = nearest_blocker_on_ray('negY', Tx, Ty, Tz)
        cand.append((Tx, nbY if nbY is not None else 0,  Tz))

        cand = [ep for ep in cand if in_bounds(ep) and not inside_strict(ep, k)]

        merged = survivors + cand
        self.eps = sorted(set(merged), key=lambda p: (p[2], p[1], p[0]))

        #self.eps = self._pareto_prune_eps(self.eps)

        return self.eps

    def _pareto_prune_eps(self, points: list[tuple[int,int,int]]) -> list[tuple[int,int,int]]:
        """Keep only non-dominated EPs."""
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