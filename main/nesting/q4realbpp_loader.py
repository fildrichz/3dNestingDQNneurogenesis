"""
Q4RealBPP Dataset Loader
Loads instances from the Q4RealBPP benchmark for real-world 3D bin packing problems.
Dataset: https://data.mendeley.com/datasets/y258s6d939/3
"""
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Item:
    """Represents a package/item to be packed"""
    id: int
    width: float
    depth: float
    height: float
    weight: float
    category: int  # For affinity constraints
    
    def volume(self) -> float:
        return self.width * self.depth * self.height


@dataclass
class Bin:
    """Represents a container/bin"""
    id: int
    width: float
    depth: float
    height: float
    max_weight: float
    
    def volume(self) -> float:
        return self.width * self.depth * self.height


@dataclass
class Q4RealBPPInstance:
    """Complete instance of a Q4RealBPP problem"""
    name: str
    items: List[Item]
    bins: List[Bin]
    
    # Affinity constraints: {(cat1, cat2): affinity_type}
    # affinity_type: 1 = must be together, -1 = must NOT be together, 0 = neutral
    affinities: Dict[Tuple[int, int], int]
    
    # Load bearing constraint: ratio threshold
    # If weight_i / weight_j > load_bearing_ratio, item i cannot be placed above item j
    load_bearing_ratio: float
    
    # Load balancing: if True, try to balance weight across bins
    load_balancing: bool
    
    def __repr__(self):
        return (f"Q4RealBPPInstance(name={self.name}, "
                f"items={len(self.items)}, bins={len(self.bins)}, "
                f"categories={len(set(item.category for item in self.items))}")


class Q4RealBPPLoader:
    """Loads Q4RealBPP benchmark instances from JSON files"""
    
    @staticmethod
    def load_from_json(filepath: str) -> Q4RealBPPInstance:
        """
        Load a Q4RealBPP instance from a JSON file.
        
        Expected JSON format:
        {
            "name": "instance_name",
            "bins": [
                {"id": 0, "width": 100, "depth": 100, "height": 100, "max_weight": 1000},
                ...
            ],
            "items": [
                {"id": 0, "width": 10, "depth": 20, "height": 15, "weight": 5.0, "category": 0},
                ...
            ],
            "affinities": [
                {"categories": [0, 1], "type": 1},  # 1 = must be together
                {"categories": [2, 3], "type": -1}, # -1 = incompatible
                ...
            ],
            "load_bearing_ratio": 1.5,
            "load_balancing": true
        }
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Parse bins
        bins = [
            Bin(
                id=b['id'],
                width=float(b['width']),
                depth=float(b['depth']),
                height=float(b['height']),
                max_weight=float(b.get('max_weight', float('inf')))
            )
            for b in data['bins']
        ]
        
        # Parse items
        items = [
            Item(
                id=i['id'],
                width=float(i['width']),
                depth=float(i['depth']),
                height=float(i['height']),
                weight=float(i['weight']),
                category=int(i.get('category', 0))
            )
            for i in data['items']
        ]
        
        # Parse affinities
        affinities = {}
        if 'affinities' in data:
            for aff in data['affinities']:
                cats = tuple(sorted(aff['categories']))
                affinities[cats] = aff['type']
        
        # Parse constraints
        load_bearing_ratio = float(data.get('load_bearing_ratio', 1.5))
        load_balancing = bool(data.get('load_balancing', False))
        
        return Q4RealBPPInstance(
            name=data.get('name', 'unnamed'),
            items=items,
            bins=bins,
            affinities=affinities,
            load_bearing_ratio=load_bearing_ratio,
            load_balancing=load_balancing
        )
    
    @staticmethod
    def load_from_dict(data: Dict) -> Q4RealBPPInstance:
        """Load from dictionary (useful for programmatic generation)"""
        bins = [
            Bin(
                id=b['id'],
                width=float(b['width']),
                depth=float(b['depth']),
                height=float(b['height']),
                max_weight=float(b.get('max_weight', float('inf')))
            )
            for b in data['bins']
        ]
        
        items = [
            Item(
                id=i['id'],
                width=float(i['width']),
                depth=float(i['depth']),
                height=float(i['height']),
                weight=float(i['weight']),
                category=int(i.get('category', 0))
            )
            for i in data['items']
        ]
        
        affinities = {}
        if 'affinities' in data:
            for aff in data['affinities']:
                cats = tuple(sorted(aff['categories']))
                affinities[cats] = aff['type']
        
        load_bearing_ratio = float(data.get('load_bearing_ratio', 1.5))
        load_balancing = bool(data.get('load_balancing', False))
        
        return Q4RealBPPInstance(
            name=data.get('name', 'unnamed'),
            items=items,
            bins=bins,
            affinities=affinities,
            load_bearing_ratio=load_bearing_ratio,
            load_balancing=load_balancing
        )
    
    @staticmethod
    def generate_sample_instance(num_items: int = 20, num_bins: int = 2) -> Q4RealBPPInstance:
        """Generate a sample instance for testing"""
        np.random.seed(42)
        
        # Generate bins
        bins = [
            Bin(
                id=i,
                width=100.0,
                depth=100.0,
                height=100.0,
                max_weight=500.0
            )
            for i in range(num_bins)
        ]
        
        # Generate items with varying sizes and weights
        items = []
        num_categories = min(5, max(2, num_items // 5))
        
        for i in range(num_items):
            w = np.random.uniform(10, 40)
            d = np.random.uniform(10, 40)
            h = np.random.uniform(10, 40)
            weight = np.random.uniform(5, 50)
            category = i % num_categories
            
            items.append(Item(
                id=i,
                width=w,
                depth=d,
                height=h,
                weight=weight,
                category=category
            ))
        
        # Generate some affinities
        affinities = {}
        if num_categories >= 2:
            # Some categories must be together
            affinities[(0, 1)] = 1
            # Some categories are incompatible
            if num_categories >= 3:
                affinities[(1, 2)] = -1
        
        return Q4RealBPPInstance(
            name=f"sample_{num_items}_{num_bins}",
            items=items,
            bins=bins,
            affinities=affinities,
            load_bearing_ratio=1.5,
            load_balancing=True
        )


def main():
    """Example usage"""
    # Generate a sample instance
    instance = Q4RealBPPLoader.generate_sample_instance(num_items=30, num_bins=3)
    
    print(f"Instance: {instance}")
    print(f"\nBins:")
    for bin in instance.bins:
        print(f"  Bin {bin.id}: {bin.width}x{bin.depth}x{bin.height}, max_weight={bin.max_weight}")
    
    print(f"\nItems (showing first 5):")
    for item in instance.items[:5]:
        print(f"  Item {item.id}: {item.width:.1f}x{item.depth:.1f}x{item.height:.1f}, "
              f"weight={item.weight:.1f}, category={item.category}")
    
    print(f"\nAffinities: {instance.affinities}")
    print(f"Load bearing ratio: {instance.load_bearing_ratio}")
    print(f"Load balancing: {instance.load_balancing}")
    
    # Save to JSON
    data = {
        "name": instance.name,
        "bins": [
            {
                "id": b.id,
                "width": b.width,
                "depth": b.depth,
                "height": b.height,
                "max_weight": b.max_weight
            }
            for b in instance.bins
        ],
        "items": [
            {
                "id": i.id,
                "width": i.width,
                "depth": i.depth,
                "height": i.height,
                "weight": i.weight,
                "category": i.category
            }
            for i in instance.items
        ],
        "affinities": [
            {"categories": list(cats), "type": aff_type}
            for cats, aff_type in instance.affinities.items()
        ],
        "load_bearing_ratio": instance.load_bearing_ratio,
        "load_balancing": instance.load_balancing
    }
    
    with open('sample_instance.json', 'w') as f:
        json.dump(data, f, indent=2)
    print("\nSaved sample instance to 'sample_instance.json'")


if __name__ == "__main__":
    main()
