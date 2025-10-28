# thpack_loader.py - Loader for OR-Library thpack9 format
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class BoxType:
    """Represents a box type with dimensions and quantity"""
    type_id: int
    w: int
    d: int
    h: int
    quantity: int
    orient_w: int = 1  # orientation flags (0/1) - usually all 1
    orient_d: int = 1
    orient_h: int = 1


@dataclass
class ThpackInstance:
    """Represents a single packing instance from thpack dataset"""
    problem_id: int
    container_w: int
    container_d: int
    container_h: int
    box_types: List[BoxType]
    
    def expand_to_items(self) -> List[Tuple[int, int, int]]:
        """Expand box types to individual items list"""
        items = []
        for bt in self.box_types:
            for _ in range(bt.quantity):
                items.append((bt.w, bt.d, bt.h))
        return items
    
    def total_boxes(self) -> int:
        return sum(bt.quantity for bt in self.box_types)
    
    def total_volume(self) -> int:
        return sum(bt.w * bt.d * bt.h * bt.quantity for bt in self.box_types)
    
    def container_volume(self) -> int:
        return self.container_w * self.container_d * self.container_h
    
    def __repr__(self):
        return (f"ThpackInstance(id={self.problem_id}, "
                f"container={self.container_w}x{self.container_d}x{self.container_h}, "
                f"types={len(self.box_types)}, boxes={self.total_boxes()})")


def load_thpack9(filepath: str) -> List[ThpackInstance]:
    """
    Load all instances from a thpack9 format file.
    
    Format:
        num_problems
        problem_id
        container_w container_d container_h
        num_box_types
        type_id w orient_w d orient_d h orient_h quantity
        ... (repeat for each box type)
        ... (repeat for each problem)
    
    Returns:
        List of ThpackInstance objects
    """
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    instances = []
    idx = 0
    
    # Read number of problems
    num_problems = int(lines[idx])
    idx += 1
    
    for _ in range(num_problems):
        # Problem ID
        problem_id = int(lines[idx])
        idx += 1
        
        # Container dimensions
        parts = list(map(int, lines[idx].split()))
        container_w, container_d, container_h = parts[0], parts[1], parts[2]
        idx += 1
        
        # Number of box types
        num_box_types = int(lines[idx])
        idx += 1
        
        # Read box types
        box_types = []
        for _ in range(num_box_types):
            parts = list(map(int, lines[idx].split()))
            
            # Handle malformed lines (some thpack9.txt files have missing orientation flags)
            if len(parts) == 7:
                # Missing one orientation flag (typically between d and h)
                # Format: type_id w orient_w d h orient_h quantity
                # Should be: type_id w orient_w d orient_d h orient_h quantity
                # Insert missing orient_d flag (always 1)
                parts = [parts[0], parts[1], parts[2], parts[3], 1, parts[4], parts[5], parts[6]]
                import warnings
                warnings.warn(f"Line {idx+1} in problem {problem_id}: Missing orientation flag, assuming 1")
            elif len(parts) != 8:
                raise ValueError(f"Line {idx+1}: Expected 8 values for box type, got {len(parts)}: {lines[idx]}")
            
            # Format: type_id w orient_w d orient_d h orient_h quantity
            type_id = parts[0]
            w = parts[1]
            orient_w = parts[2]
            d = parts[3]
            orient_d = parts[4]
            h = parts[5]
            orient_h = parts[6]
            quantity = parts[7]
            
            box_types.append(BoxType(
                type_id=type_id,
                w=w, d=d, h=h,
                quantity=quantity,
                orient_w=orient_w,
                orient_d=orient_d,
                orient_h=orient_h
            ))
            idx += 1
        
        instances.append(ThpackInstance(
            problem_id=problem_id,
            container_w=container_w,
            container_d=container_d,
            container_h=container_h,
            box_types=box_types
        ))
    
    return instances


def print_instance_info(instance: ThpackInstance) -> None:
    """Print summary of an instance"""
    print(f"\n{'='*60}")
    print(f"Problem {instance.problem_id}")
    print(f"{'='*60}")
    print(f"Container: {instance.container_w} x {instance.container_d} x {instance.container_h}")
    print(f"Container Volume: {instance.container_volume()}")
    print(f"Number of box types: {len(instance.box_types)}")
    print(f"Total boxes: {instance.total_boxes()}")
    print(f"Total box volume: {instance.total_volume()}")
    print(f"Density: {instance.total_volume() / instance.container_volume():.2f}x")
    print(f"\nBox types:")
    for bt in instance.box_types:
        volume = bt.w * bt.d * bt.h
        print(f"  Type {bt.type_id}: {bt.w}x{bt.d}x{bt.h} (vol={volume}) x{bt.quantity}")


if __name__ == "__main__":
    # Test loader
    import sys
    filepath = sys.argv[1] if len(sys.argv) > 1 else "thpack9.txt"
    
    instances = load_thpack9(filepath)
    print(f"Loaded {len(instances)} instances from {filepath}")
    
    # Show first 3 instances
    for inst in instances[:3]:
        print_instance_info(inst)
        items = inst.expand_to_items()
        print(f"Expanded to {len(items)} individual items")