
"""
Q4RealBPP Dataset Loader
- JSON loader
- TXT loader for standart Q4RealBPP format
"""

import json
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class Item:
    id: int
    width: float
    depth: float
    height: float
    weight: float
    category: int  # original dataset item id


@dataclass
class Bin:
    id: int
    width: float
    depth: float
    height: float
    max_weight: float

    def volume(self) -> float:
        return self.width * self.depth * self.height


@dataclass
class Q4RealBPPInstance:
    name: str
    items: List[Item]
    bins: List[Bin]
    affinities: Dict[Tuple[int, int], int]
    load_bearing_ratio: float
    load_balancing: bool

    def __repr__(self):
        cats = len(set(i.category for i in self.items))
        return f"Q4RealBPPInstance(name={self.name}, items={len(self.items)}, bins={len(self.bins)}, categories={cats})"


class Q4RealBPPLoader:
    """Loads Q4RealBPP-style instances from JSON or TXT."""

    # -------------------------
    # Existing JSON interface
    # -------------------------
    @staticmethod
    def load_from_json(filepath: str) -> Q4RealBPPInstance:
        with open(filepath, "r") as f:
            data = json.load(f)
        return Q4RealBPPLoader.load_from_dict(data)

    @staticmethod
    def load_from_dict(data: Dict) -> Q4RealBPPInstance:
        bins = [
            Bin(
                id=int(b["id"]),
                width=float(b["width"]),
                depth=float(b["depth"]),
                height=float(b["height"]),
                max_weight=float(b.get("max_weight", float("inf"))),
            )
            for b in data["bins"]
        ]

        items = [
            Item(
                id=int(i["id"]),
                width=float(i["width"]),
                depth=float(i["depth"]),
                height=float(i["height"]),
                weight=float(i["weight"]),
                category=int(i.get("category", 0)),
            )
            for i in data["items"]
        ]

        affinities: Dict[Tuple[int, int], int] = {}
        for aff in data.get("affinities", []):
            cats = tuple(sorted(aff["categories"]))
            affinities[cats] = int(aff["type"])

        return Q4RealBPPInstance(
            name=data.get("name", "unnamed"),
            items=items,
            bins=bins,
            affinities=affinities,
            load_bearing_ratio=float(data.get("load_bearing_ratio", 1.5)),
            load_balancing=bool(data.get("load_balancing", False)),
        )

    # -------------------------
    # NEW: TXT loader
    # -------------------------
    @staticmethod
    def load_from_txt(filepath: str) -> Q4RealBPPInstance:
        """
        Parse TXT like the provided example (3dBPP_11.txt):
          # Max num of bins: 2
          # Bin dimensions (L * W * H): (900,900,900)
          # Max weight: 800
          # Relative pos: {...}              <-- ignored (not modeled)
          # Incompatibilities: (7,9)
          # Positive affinities: (0, 3) (0, 8)
          # Center of mass: (750, 750)      <-- ignored (not modeled)

            id  quantity  length  width  height  weight
            ... rows ...
        """
        with open(filepath, "r") as f:
            lines = [ln.rstrip("\n") for ln in f]

        # Header parsing
        max_bins = 1
        W = D = H = None
        max_weight = float("inf")
        pos_aff: List[Tuple[int, int]] = []
        neg_aff: List[Tuple[int, int]] = []

        def _pairs_from_line(s: str) -> List[Tuple[int, int]]:
            # extract "(a,b)" pairs
            return [tuple(map(int, m)) for m in re.findall(r"\((\d+)\s*,\s*(\d+)\)", s)]

        for ln in lines:
            if not ln.strip().startswith("#"):
                continue
            low = ln.lower()
            if "max num of bins" in low:
                m = re.search(r":\s*([0-9]+)", ln)
                if m:
                    max_bins = int(m.group(1))
            elif "bin dimensions" in low:
                m = re.search(r"\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)", ln)
                if m:
                    # dataset gives (L,W,H); map to (width=length, depth=width, height=height)
                    W = int(m.group(1))
                    D = int(m.group(2))
                    H = int(m.group(3))
            elif "max weight" in low:
                m = re.search(r":\s*([0-9]+(\.\d+)?)", ln)
                if m:
                    max_weight = float(m.group(1))
            elif "incompatibilities" in low:
                neg_aff.extend(_pairs_from_line(ln))
            elif "positive affinities" in low:
                pos_aff.extend(_pairs_from_line(ln))
            # "Relative pos" and "Center of mass" are intentionally ignored

        if W is None or D is None or H is None:
            raise ValueError("Could not parse bin dimensions from header.")

        # Build bins 0..max_bins-1 with identical spec
        bins = [
            Bin(id=i, width=W, depth=D, height=H, max_weight=max_weight) for i in range(max_bins)
        ]

        # Table parsing
        # Find the header line with the columns
        try:
            hdr_idx = next(
                i
                for i, ln in enumerate(lines)
                if re.search(r"\bid\b", ln) and re.search(r"\bquantity\b", ln)
            )
        except StopIteration:
            raise ValueError("Items table header not found (expected 'id  quantity  length  width  height  weight').")

        # Data starts after a possible separator line of dashes
        data_start = hdr_idx + 1
        # skip "----" line if present
        if data_start < len(lines) and set(lines[data_start].strip()) == {"-"}:
            data_start += 1

        items: List[Item] = []
        next_item_id = 0
        for ln in lines[data_start:]:
            if not ln.strip():
                continue
            if ln.strip().startswith("#"):
                continue
            # Expect 6 numeric columns
            cols = re.split(r"\s+", ln.strip())
            if len(cols) < 6:
                # allow for misaligned columns; skip if not data
                continue
            try:
                orig_id = int(cols[0])
                qty = int(cols[1])
                length = float(cols[2])
                width = float(cols[3])
                height = float(cols[4])
                weight = float(cols[5])
            except ValueError:
                # not a proper data row
                continue

            # Map dataset (length,width,height) -> (W,D,H) = (width,depth,height)
            item_w, item_d, item_h = length, width, height
            category = orig_id  # categories line up with original ids

            for _ in range(qty):
                items.append(
                    Item(
                        id=next_item_id,
                        width=item_w,
                        depth=item_d,
                        height=item_h,
                        weight=weight,
                        category=category,
                    )
                )
                next_item_id += 1

        # Affinity dictionary
        aff: Dict[Tuple[int, int], int] = {}
        for a, b in pos_aff:
            aff[tuple(sorted((a, b)))] = 1
        for a, b in neg_aff:
            aff[tuple(sorted((a, b)))] = -1

        return Q4RealBPPInstance(
            name=_basename_noext(filepath),
            items=items,
            bins=bins,
            affinities=aff,
            load_bearing_ratio=1.5,   # default; your code reads/uses this
            load_balancing=False,     # default; can be adjusted if needed
        )

    @staticmethod
    def generate_sample_instance(num_items: int = 20, num_bins: int = 2) -> Q4RealBPPInstance:
        np.random.seed(42)
        bins = [
            Bin(id=i, width=100.0, depth=100.0, height=100.0, max_weight=500.0)
            for i in range(num_bins)
        ]
        items = []
        num_categories = min(5, max(2, num_items // 5))
        for i in range(num_items):
            w = np.random.uniform(10, 40)
            d = np.random.uniform(10, 40)
            h = np.random.uniform(10, 40)
            weight = np.random.uniform(5, 50)
            category = i % num_categories
            items.append(Item(id=i, width=w, depth=d, height=h, weight=weight, category=category))
        affinities: Dict[Tuple[int, int], int] = {}
        if num_categories >= 2:
            affinities[(0, 1)] = 1
            if num_categories >= 3:
                affinities[(1, 2)] = -1
        return Q4RealBPPInstance(
            name=f"sample_{num_items}_{num_bins}",
            items=items,
            bins=bins,
            affinities=affinities,
            load_bearing_ratio=1.5,
            load_balancing=True,
        )


def _basename_noext(p: str) -> str:
    import os
    return os.path.splitext(os.path.basename(p))[0]


if __name__ == "__main__":
    # Quick manual test: switch to your own path if needed
    inst = Q4RealBPPLoader.load_from_txt("3dBPP_11.txt")
    print(inst)
