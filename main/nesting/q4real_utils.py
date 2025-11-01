# q4real_utils.py — REPLACEMENT

"""
Utilities for Q4RealBPP-style datasets.
- Convert .txt (3dBPP_* format) <-> project JSON schema
- Benchmark helpers
"""

import os
import glob
import json
import numpy as np
from typing import Dict

from q4realbpp_loader import Q4RealBPPLoader, Q4RealBPPInstance
from packing_core_q4real import solve_q4realbpp_instance
from packing_with_dqn_q4real import train_dqn_q4realbpp, evaluate_dqn_q4realbpp


def load_instance(path: str) -> Q4RealBPPInstance:
    """Load either .json (project schema) or .txt (3dBPP_* format)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        return Q4RealBPPLoader.load_from_json(path)
    elif ext == ".txt":
        return Q4RealBPPLoader.load_from_txt(path)
    else:
        raise ValueError(f"Unsupported instance extension: {ext}")


def convert_txt_to_json(txt_file: str, output_json: str = None) -> str:
    """Parse a 3dBPP_*.txt file and save as project JSON schema."""
    instance = Q4RealBPPLoader.load_from_txt(txt_file)

    data = {
        "name": instance.name,
        "bins": [
            {
                "id": b.id,
                "width": b.width,
                "depth": b.depth,
                "height": b.height,
                "max_weight": b.max_weight,
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
                "category": i.category,
            }
            for i in instance.items
        ],
        "affinities": [
            {"categories": list(k), "type": v} for k, v in instance.affinities.items()
        ],
        "load_bearing_ratio": instance.load_bearing_ratio,
        "load_balancing": instance.load_balancing,
    }

    if output_json is None:
        output_json = os.path.splitext(txt_file)[0] + ".json"

    with open(output_json, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved JSON: {output_json}")
    return output_json


def batch_convert_directory(input_dir: str, output_dir: str = None):
    if output_dir is None:
        output_dir = input_dir
    os.makedirs(output_dir, exist_ok=True)
    for txt in glob.glob(os.path.join(input_dir, "*.txt")):
        out = os.path.join(output_dir, os.path.basename(txt).replace(".txt", ".json"))
        try:
            convert_txt_to_json(txt, out)
        except Exception as e:
            print(f"[WARN] Failed to convert {txt}: {e}")


def benchmark_instance(instance_path: str,
                       method: str = "both",
                       dqn_episodes: int = 500,
                       dqn_eval_runs: int = 10) -> Dict:
    print(f"\n{'='*60}\nBenchmarking: {instance_path}\n{'='*60}")
    instance = load_instance(instance_path)
    print(f"Instance: {instance}")

    results: Dict = {"instance": instance.name}

    if method in ["heuristic", "both"]:
        containers = solve_q4realbpp_instance(instance, visualize=False)
        placed = sum(len(c.placed) for c in containers.values())
        total = len(instance.items)
        vol = np.mean([
            (sum(b.w * b.d * b.h for b in c.placed) / (c.w * c.d * c.h))
            if c.placed else 0.0
            for c in containers.values()
        ])
        results["heuristic"] = {
            "items_placed": placed,
            "packing_rate": placed / total if total else 0.0,
            "volume_utilization": float(vol),
        }

    if method in ["dqn", "both"]:
        agent = train_dqn_q4realbpp(instance, episodes=dqn_episodes, device="cpu", save_path=None)
        dqn_res = evaluate_dqn_q4realbpp(agent, instance, num_eval=dqn_eval_runs, visualize=False)
        results["dqn"] = dqn_res

    return results


def benchmark_suite(instance_dir: str,
                    output_file: str = "benchmark_results.json",
                    method: str = "both",
                    dqn_episodes: int = 500):
    files = glob.glob(os.path.join(instance_dir, "*.json")) + glob.glob(os.path.join(instance_dir, "*.txt"))
    print(f"Found {len(files)} instances in {instance_dir}")
    all_results = {}
    for p in files:
        try:
            res = benchmark_instance(p, method=method, dqn_episodes=dqn_episodes)
            all_results[os.path.basename(p)] = res
        except Exception as e:
            print(f"[WARN] Failed to benchmark {p}: {e}")

    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved all results to {output_file}")


if __name__ == "__main__":
    # tiny CLI
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--convert", help="Convert a .txt to JSON")
    ap.add_argument("--bench", help="Benchmark a single instance (.txt or .json)")
    ap.add_argument("--benchdir", help="Benchmark every .txt/.json in a directory")
    ap.add_argument("--episodes", type=int, default=200)
    args = ap.parse_args()

    if args.convert:
        convert_txt_to_json(args.convert)
    elif args.bench:
        print(json.dumps(benchmark_instance(args.bench, dqn_episodes=args.episodes), indent=2))
    elif args.benchdir:
        benchmark_suite(args.benchdir, dqn_episodes=args.episodes)
