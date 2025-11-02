# dataset_loader.py - Loader for 3D Bin Packing Problem dataset
import re
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

@dataclass
class Item:
    """Represents a single item type in the problem."""
    id: int
    quantity: int
    length: int
    width: int
    height: int
    weight: int

@dataclass
class BinPackingProblem:
    """Represents a complete bin packing problem instance."""
    max_bins: int
    bin_dimensions: Tuple[int, int, int]  # (L, W, H)
    max_weight: Optional[int]
    relative_pos: Dict[int, List[Tuple[int, int]]]  # item_id -> list of (other_id, position_code)
    incompatibilities: List[Tuple[int, int]]  # pairs of items that can't be in same bin
    positive_affinities: List[Tuple[int, int]]  # pairs of items that should be together
    center_of_mass: Optional[Tuple[int, int]]  # (x, y) constraint
    items: List[Item]
    
    def get_all_items(self) -> List[Tuple[int, Item]]:
        """Returns a flat list of (item_id, item) for all items considering quantities."""
        all_items = []
        for item in self.items:
            for _ in range(item.quantity):
                all_items.append((item.id, item))
        return all_items


def parse_problem_file(filepath: str) -> BinPackingProblem:
    """
    Parse a 3D bin packing problem file from the dataset.
    
    Args:
        filepath: Path to the .txt problem file
        
    Returns:
        BinPackingProblem instance with all constraints and items
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Initialize variables
    max_bins = None
    bin_dims = None
    max_weight = None
    relative_pos = {}
    incompatibilities = []
    positive_affinities = []
    center_of_mass = None
    items = []
    
    # Parse header section
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Max num of bins
        if line.startswith('# Max num of bins:'):
            match = re.search(r':\s*(\d+)', line)
            if match:
                max_bins = int(match.group(1))
        
        # Bin dimensions
        elif line.startswith('# Bin dimensions'):
            match = re.search(r'\((\d+),\s*(\d+),\s*(\d+)\)', line)
            if match:
                bin_dims = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        
        # Max weight
        elif line.startswith('# Max weight:'):
            match = re.search(r':\s*(\d+)', line)
            if match:
                max_weight = int(match.group(1))
        
        # Relative positioning constraints
        elif line.startswith('# Relative pos:'):
            # Multi-line parsing for relative positions
            rel_text = line.split(':', 1)[1].strip()
            j = i + 1
            # Continue reading if the constraint spans multiple lines
            # Note: continuation lines may start with '#' followed by whitespace
            while j < len(lines):
                next_line = lines[j].strip()
                # Stop if we hit a line that starts with '# ' followed by a letter (new constraint)
                # But continue if it's '# \t' or similar (continuation with just whitespace)
                if next_line.startswith('#'):
                    # Check if it's a new constraint or a continuation
                    after_hash = next_line[1:].lstrip()
                    if after_hash and after_hash[0].isupper():
                        # New constraint (starts with capital letter like "Max", "Bin", etc)
                        break
                    # Otherwise it's a continuation line, remove the # and continue
                    # Add comma if the previous line doesn't end with comma/opening bracket
                    if rel_text and rel_text[-1] not in ',{[' and after_hash and after_hash[0] == '(':
                        rel_text += ', '
                    rel_text += ' ' + after_hash
                elif next_line:
                    # Add comma if needed
                    if rel_text and rel_text[-1] not in ',{[' and next_line[0] == '(':
                        rel_text += ', '
                    rel_text += ' ' + next_line
                elif not next_line:
                    # Empty line might end the constraint
                    break
                j += 1
            
            if rel_text and rel_text != '':
                # Parse dictionary-like format: {6: [(1, 0), (1, 7), ...]}
                try:
                    # Remove any remaining comments
                    rel_text = rel_text.split('#')[0].strip()
                    if rel_text:
                        relative_pos = eval(rel_text)
                except Exception as e:
                    # Silently fail and continue - relative_pos stays empty
                    pass
            i = j - 1
        
        # Incompatibilities
        elif line.startswith('# Incompatibilities:'):
            pairs_text = line.split(':', 1)[1].strip()
            if pairs_text:
                # Extract all pairs like (4, 7)
                pairs = re.findall(r'\((\d+),\s*(\d+)\)', pairs_text)
                incompatibilities = [(int(a), int(b)) for a, b in pairs]
        
        # Positive affinities
        elif line.startswith('# Positive affinities:'):
            pairs_text = line.split(':', 1)[1].strip()
            if pairs_text:
                pairs = re.findall(r'\((\d+),\s*(\d+)\)', pairs_text)
                positive_affinities = [(int(a), int(b)) for a, b in pairs]
        
        # Center of mass
        elif line.startswith('# Center of mass:'):
            match = re.search(r'\((\d+),\s*(\d+)\)', line)
            if match:
                center_of_mass = (int(match.group(1)), int(match.group(2)))
        
        # Start of items table
        elif line.startswith('id') and 'quantity' in line:
            # Skip the header line
            i += 1
            # Skip separator line (----)
            if i < len(lines) and '----' in lines[i]:
                i += 1
            
            # Parse item rows
            while i < len(lines):
                line = lines[i].strip()
                if not line:
                    break
                
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        item = Item(
                            id=int(parts[0]),
                            quantity=int(parts[1]),
                            length=int(parts[2]),
                            width=int(parts[3]),
                            height=int(parts[4]),
                            weight=int(parts[5])
                        )
                        items.append(item)
                    except ValueError:
                        pass
                i += 1
            break
        
        i += 1
    
    return BinPackingProblem(
        max_bins=max_bins or 1,
        bin_dimensions=bin_dims or (1000, 1000, 1000),
        max_weight=max_weight,
        relative_pos=relative_pos,
        incompatibilities=incompatibilities,
        positive_affinities=positive_affinities,
        center_of_mass=center_of_mass,
        items=items
    )


def load_problem(filepath: str) -> BinPackingProblem:
    """
    Convenience function to load a problem file.
    
    Args:
        filepath: Path to the .txt problem file
        
    Returns:
        BinPackingProblem instance
    """
    return parse_problem_file(filepath)


if __name__ == "__main__":
    # Example usage
    problem = load_problem("3dBPP_11.txt")
    print(f"Max bins: {problem.max_bins}")
    print(f"Bin dimensions: {problem.bin_dimensions}")
    print(f"Max weight: {problem.max_weight}")
    print(f"Number of item types: {len(problem.items)}")
    print(f"Total items to pack: {sum(item.quantity for item in problem.items)}")
    print(f"Incompatibilities: {problem.incompatibilities}")
    print(f"Affinities: {problem.positive_affinities}")