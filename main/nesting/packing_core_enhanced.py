# packing_core_complete.py — Complete implementation with ALL constraints + gravity + EMS
from typing import List, Tuple, Dict, Optional, Set
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np


class EMS:
    """Empty Maximal Space - represents an available rectangular volume for placement."""
    def __init__(self, x: int, y: int, z: int, w: int, d: int, h: int):
        self.x, self.y, self.z = int(x), int(y), int(z)  # Corner position
        self.w, self.d, self.h = int(w), int(d), int(h)  # Dimensions

    def __repr__(self):
        return f"EMS(pos=({self.x},{self.y},{self.z}), size=({self.w},{self.d},{self.h}), vol={self.volume()})"

    def volume(self) -> int:
        """Calculate volume of this space."""
        return self.w * self.d * self.h

    def dominates(self, other: 'EMS') -> bool:
        """Check if this EMS completely contains another EMS."""
        return (self.x <= other.x and self.y <= other.y and self.z <= other.z and
                self.x + self.w >= other.x + other.w and
                self.y + self.d >= other.y + other.d and
                self.z + self.h >= other.z + other.h)

    def intersects(self, box: 'box3d') -> bool:
        """Check if this EMS intersects with a placed box."""
        return not (
            self.x + self.w <= box.x or box.x + box.w <= self.x or
            self.y + self.d <= box.y or box.y + box.d <= self.y or
            self.z + self.h <= box.z or box.z + box.h <= self.z
        )

    def as_tuple(self) -> Tuple[int, int, int, int, int, int]:
        """Return as tuple (x, y, z, w, d, h)."""
        return (self.x, self.y, self.z, self.w, self.d, self.h)

    def corner(self) -> Tuple[int, int, int]:
        """Get the corner position as tuple."""
        return (self.x, self.y, self.z)

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
                 resolution: int = 10, max_ems: int = 50):
        """
        Initialize container with all constraint support using Empty Maximal Spaces (EMS).

        Args:
            W, D, H: Container dimensions
            max_weight: Maximum weight capacity (optional)
            resolution: Grid resolution for heightmap
            max_ems: Maximum number of EMS to maintain (for performance)
        """
        super().__init__(W, D, H)
        self.placed: List[box3d] = []
        self.ems_list: List[EMS] = [EMS(0, 0, 0, W, D, H)]  # Start with one big space
        self.max_ems = max_ems
        
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
    
    def check_relative_positioning(self, item_id: int, ep: Tuple[int, int, int], size: Tuple[int, int, int]) -> bool:
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

    def _fits_ems(self, ems: EMS, size: tuple[int, int, int]) -> bool:
        """Check if item size fits within EMS dimensions."""
        w, d, h = size
        return (w <= ems.w) and (d <= ems.d) and (h <= ems.h)

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

    def place_at_ems(self, ems: EMS, size: tuple[int, int, int],
                     weight: int = 0, item_id: int = -1) -> bool:
        """
        Place a box using EMS with ALL constraint checking + gravity support.

        Process:
        1. Check all constraints at EMS corner
        2. Apply gravity - drop box until it hits support
        3. Re-check constraints at final position
        4. Place box and update EMS

        Args:
            ems: Empty Maximal Space to place item in
            size: Box dimensions (w, d, h)
            weight: Weight of the box
            item_id: Item type identifier

        Returns:
            True if placement successful, False otherwise
        """
        # Use EMS corner as initial position
        x, y, z = ems.x, ems.y, ems.z
        w, d, h = size
        ep = (x, y, z)

        # === STEP 1: Initial constraint checks ===
        if not (self._fits_ems(ems, size) and
                self._fits_container(ep, size) and
                self.check_weight_constraint(weight) and
                self.check_incompatibility(item_id)):
            return False

        # === STEP 2: Apply gravity ===
        # Drop box down until it hits something
        final_z = self.apply_gravity(x, y, z, w, d, h)
        final_ep = (x, y, final_z)

        # === STEP 3: Re-check constraints at final position ===
        if not (self._fits_container(final_ep, size) and
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

        # === STEP 5: Update EMS AFTER placement ===
        self.update_ems(k)

        return True

    def place_at(self, ep: tuple[int, int, int], size: tuple[int, int, int],
                 weight: int = 0, item_id: int = -1) -> bool:
        """
        LEGACY: Place at an explicit position (for backward compatibility).
        Creates a temporary EMS and calls place_at_ems.
        """
        x, y, z = ep
        # Create a large temporary EMS at this position
        temp_ems = EMS(x, y, z, self.w - x, self.d - y, self.h - z)
        return self.place_at_ems(temp_ems, size, weight, item_id)

    def place(self, ep: tuple[int, int, int], w: int, d: int, h: int, 
              weight: int = 0, item_id: int = -1) -> bool:
        """Alternate placement method. Calls place_at internally."""
        return self.place_at(ep, (w, d, h), weight, item_id)
    
    def place_first_fit(self, base_size: tuple[int, int, int],
                       weight: int = 0, item_id: int = -1) -> bool:
        """Try to place a box using first-fit strategy with EMS."""
        # Sort by volume (largest first) and then by z-coordinate
        sorted_ems = sorted(self.ems_list, key=lambda e: (-e.volume(), e.z, e.y, e.x))
        for ems in sorted_ems:
            for size in self._orientations(*base_size):
                if self.place_at_ems(ems, size, weight, item_id):
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
        
        if self.ems_list:
            # Visualize EMS as wireframe boxes
            for ems in self.ems_list[:10]:  # Show top 10 EMS to avoid clutter
                ems_edges = Line3DCollection(
                    self._cuboid_edges(ems.x, ems.y, ems.z, ems.w, ems.d, ems.h),
                    linewidths=0.5, colors='red', linestyles='dashed', alpha=0.3
                )
                ax.add_collection3d(ems_edges)
        
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

    # ========== EMS UPDATE ==========

    def update_ems(self, placed_box: box3d) -> None:
        """
        Update Empty Maximal Spaces after placing a box.

        For each EMS that intersects with the placed box:
        1. Compute 6-way difference (up to 6 new sub-spaces)
        2. Remove original EMS
        3. Add valid new sub-spaces

        Then prune:
        - Remove dominated spaces (contained within others)
        - Keep top-k by volume
        """
        new_ems_list = []

        for ems in self.ems_list:
            if ems.intersects(placed_box):
                # EMS intersects with placed box - split it
                sub_spaces = self._compute_ems_difference(ems, placed_box)
                new_ems_list.extend(sub_spaces)
            else:
                # EMS doesn't intersect - keep it
                new_ems_list.append(ems)

        # Prune dominated spaces
        new_ems_list = self._prune_dominated_ems(new_ems_list)

        # Sort by volume (descending) and keep top-k
        new_ems_list.sort(key=lambda e: e.volume(), reverse=True)
        self.ems_list = new_ems_list[:self.max_ems]

    def _compute_ems_difference(self, ems: EMS, box: box3d) -> List[EMS]:
        """
        Compute the difference between an EMS and a placed box.

        Returns up to 6 new sub-spaces (slices along each axis direction):
        - Left slice (x < box.x)
        - Right slice (x > box.x + box.w)
        - Front slice (y < box.y)
        - Back slice (y > box.y + box.d)
        - Bottom slice (z < box.z)
        - Top slice (z > box.z + box.h)
        """
        sub_spaces = []

        # Define EMS bounds
        ex1, ey1, ez1 = ems.x, ems.y, ems.z
        ex2, ey2, ez2 = ems.x + ems.w, ems.y + ems.d, ems.z + ems.h

        # Define box bounds
        bx1, by1, bz1 = box.x, box.y, box.z
        bx2, by2, bz2 = box.x + box.w, box.y + box.d, box.z + box.h

        # Left slice: EMS region to the left of the box
        if ex1 < bx1 < ex2:
            w = bx1 - ex1
            if w > 0:
                sub_spaces.append(EMS(ex1, ey1, ez1, w, ems.d, ems.h))

        # Right slice: EMS region to the right of the box
        if ex1 < bx2 < ex2:
            x = bx2
            w = ex2 - bx2
            if w > 0:
                sub_spaces.append(EMS(x, ey1, ez1, w, ems.d, ems.h))

        # Front slice: EMS region in front of the box
        if ey1 < by1 < ey2:
            d = by1 - ey1
            if d > 0:
                sub_spaces.append(EMS(ex1, ey1, ez1, ems.w, d, ems.h))

        # Back slice: EMS region behind the box
        if ey1 < by2 < ey2:
            y = by2
            d = ey2 - by2
            if d > 0:
                sub_spaces.append(EMS(ex1, y, ez1, ems.w, d, ems.h))

        # Bottom slice: EMS region below the box
        if ez1 < bz1 < ez2:
            h = bz1 - ez1
            if h > 0:
                sub_spaces.append(EMS(ex1, ey1, ez1, ems.w, ems.d, h))

        # Top slice: EMS region above the box
        if ez1 < bz2 < ez2:
            z = bz2
            h = ez2 - bz2
            if h > 0:
                sub_spaces.append(EMS(ex1, ey1, z, ems.w, ems.d, h))

        return sub_spaces

    def _prune_dominated_ems(self, ems_list: List[EMS]) -> List[EMS]:
        """
        Remove EMS that are completely contained within other EMS.

        An EMS A dominates EMS B if A completely contains B.
        """
        if len(ems_list) <= 1:
            return ems_list

        non_dominated = []

        for i, ems_a in enumerate(ems_list):
            dominated = False
            for j, ems_b in enumerate(ems_list):
                if i != j and ems_b.dominates(ems_a):
                    dominated = True
                    break
            if not dominated:
                non_dominated.append(ems_a)

        return non_dominated