"""
Utility script for Q4RealBPP dataset conversion and benchmarking.
Helps convert various formats to the expected JSON format and run benchmarks.
"""
import json
import os
import glob
from typing import Dict, List
import numpy as np

from q4realbpp_loader import Q4RealBPPLoader, Q4RealBPPInstance
from packing_core_q4real import solve_q4realbpp_instance
from packing_with_dqn_q4real import train_dqn_q4realbpp, evaluate_dqn_q4realbpp


def convert_txt_to_json(txt_file: str, output_json: str = None) -> str:
    """
    Convert Q4RealBPP text format to JSON.
    
    This is a template - adjust parsing logic based on actual file format.
    The Q4RealBPP dataset may use custom text formats.
    """
    print(f"Converting {txt_file} to JSON...")
    
    # Read file
    with open(txt_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # Parse (EXAMPLE - adjust based on actual format)
    data = {
        "name": os.path.basename(txt_file).replace('.txt', ''),
        "bins": [],
        "items": [],
        "affinities": [],
        "load_bearing_ratio": 1.5,
        "load_balancing": False
    }
    
    # Example parsing logic (MODIFY THIS):
    # Assuming format like:
    # BIN 0 120 100 100 500
    # ITEM 0 30 20 15 10.5 0
    # AFFINITY 0 1 1
    
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        
        if parts[0] == "BIN":
            data["bins"].append({
                "id": int(parts[1]),
                "width": float(parts[2]),
                "depth": float(parts[3]),
                "height": float(parts[4]),
                "max_weight": float(parts[5]) if len(parts) > 5 else float('inf')
            })
        
        elif parts[0] == "ITEM":
            data["items"].append({
                "id": int(parts[1]),
                "width": float(parts[2]),
                "depth": float(parts[3]),
                "height": float(parts[4]),
                "weight": float(parts[5]) if len(parts) > 5 else 1.0,
                "category": int(parts[6]) if len(parts) > 6 else 0
            })
        
        elif parts[0] == "AFFINITY":
            data["affinities"].append({
                "categories": [int(parts[1]), int(parts[2])],
                "type": int(parts[3])
            })
        
        elif parts[0] == "LOAD_BEARING":
            data["load_bearing_ratio"] = float(parts[1])
        
        elif parts[0] == "LOAD_BALANCING":
            data["load_balancing"] = bool(int(parts[1]))
    
    # Save
    if output_json is None:
        output_json = txt_file.replace('.txt', '.json')
    
    with open(output_json, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved to {output_json}")
    return output_json


def batch_convert_directory(input_dir: str, output_dir: str = None):
    """Convert all .txt files in a directory to JSON"""
    if output_dir is None:
        output_dir = input_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    txt_files = glob.glob(os.path.join(input_dir, "*.txt"))
    
    print(f"Found {len(txt_files)} text files in {input_dir}")
    
    for txt_file in txt_files:
        basename = os.path.basename(txt_file).replace('.txt', '.json')
        output_json = os.path.join(output_dir, basename)
        try:
            convert_txt_to_json(txt_file, output_json)
        except Exception as e:
            print(f"Error converting {txt_file}: {e}")


def benchmark_instance(instance_path: str, 
                      method: str = "both",
                      dqn_episodes: int = 500,
                      dqn_eval_runs: int = 10):
    """
    Benchmark an instance with heuristic and/or DQN.
    
    Args:
        instance_path: Path to JSON instance file
        method: "heuristic", "dqn", or "both"
        dqn_episodes: Number of training episodes for DQN
        dqn_eval_runs: Number of evaluation runs for DQN
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking: {instance_path}")
    print(f"{'='*60}")
    
    # Load instance
    instance = Q4RealBPPLoader.load_from_json(instance_path)
    print(f"\nInstance: {instance}")
    
    results = {}
    
    # Heuristic method
    if method in ["heuristic", "both"]:
        print(f"\n{'='*60}")
        print("Running First-Fit Heuristic")
        print(f"{'='*60}")
        
        containers = solve_q4realbpp_instance(instance, visualize=False)
        
        total_placed = sum(len(c.placed) for c in containers.values())
        total_weight = sum(c.current_weight for c in containers.values())
        total_volume_used = sum(
            sum(b.w * b.d * b.h for b in c.placed) 
            for c in containers.values()
        )
        total_volume_available = sum(
            c.w * c.d * c.h for c in containers.values()
        )
        
        results['heuristic'] = {
            'items_placed': total_placed,
            'total_items': len(instance.items),
            'packing_rate': total_placed / len(instance.items),
            'total_weight': total_weight,
            'volume_utilization': total_volume_used / total_volume_available,
        }
        
        print(f"\nHeuristic Results:")
        print(f"  Items packed: {total_placed}/{len(instance.items)} "
              f"({results['heuristic']['packing_rate']:.1%})")
        print(f"  Volume utilization: {results['heuristic']['volume_utilization']:.2%}")
    
    # DQN method
    if method in ["dqn", "both"]:
        print(f"\n{'='*60}")
        print("Training DQN Agent")
        print(f"{'='*60}")
        
        # Train
        agent = train_dqn_q4realbpp(
            instance,
            episodes=dqn_episodes,
            device="cpu",
            save_path=None  # Don't save by default in benchmark
        )
        
        # Evaluate
        print(f"\nEvaluating DQN Agent")
        eval_results = evaluate_dqn_q4realbpp(
            agent, 
            instance, 
            num_eval=dqn_eval_runs, 
            visualize=False
        )
        
        results['dqn'] = eval_results
        
        print(f"\nDQN Results:")
        print(f"  Avg items packed: {eval_results['avg_items_placed']:.1f}/{len(instance.items)}")
        print(f"  Avg volume utilization: {eval_results['avg_volume_utilization']:.2%}")
        print(f"  Avg reward: {eval_results['avg_reward']:.2f}")
    
    # Summary
    if method == "both":
        print(f"\n{'='*60}")
        print("Comparison Summary")
        print(f"{'='*60}")
        
        h_rate = results['heuristic']['packing_rate']
        d_rate = results['dqn']['avg_items_placed'] / len(instance.items)
        
        h_vol = results['heuristic']['volume_utilization']
        d_vol = results['dqn']['avg_volume_utilization']
        
        print(f"Packing Rate:")
        print(f"  Heuristic: {h_rate:.1%}")
        print(f"  DQN:       {d_rate:.1%}")
        print(f"  Winner:    {'DQN' if d_rate > h_rate else 'Heuristic'}")
        
        print(f"\nVolume Utilization:")
        print(f"  Heuristic: {h_vol:.2%}")
        print(f"  DQN:       {d_vol:.2%}")
        print(f"  Winner:    {'DQN' if d_vol > h_vol else 'Heuristic'}")
    
    return results


def benchmark_suite(instance_dir: str, 
                   output_file: str = "benchmark_results.json",
                   method: str = "both",
                   dqn_episodes: int = 500):
    """Run benchmarks on all instances in a directory"""
    json_files = glob.glob(os.path.join(instance_dir, "*.json"))
    
    print(f"Found {len(json_files)} instances in {instance_dir}")
    
    all_results = {}
    
    for json_file in json_files:
        instance_name = os.path.basename(json_file).replace('.json', '')
        
        try:
            results = benchmark_instance(
                json_file, 
                method=method,
                dqn_episodes=dqn_episodes
            )
            all_results[instance_name] = results
        except Exception as e:
            print(f"Error benchmarking {json_file}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"All results saved to {output_file}")
    print(f"{'='*60}")
    
    # Summary statistics
    print(f"\nSummary Statistics:")
    
    if method in ["heuristic", "both"]:
        h_rates = [r['heuristic']['packing_rate'] for r in all_results.values()]
        h_vols = [r['heuristic']['volume_utilization'] for r in all_results.values()]
        
        print(f"\nHeuristic:")
        print(f"  Avg packing rate: {np.mean(h_rates):.1%} ± {np.std(h_rates):.1%}")
        print(f"  Avg volume util: {np.mean(h_vols):.2%} ± {np.std(h_vols):.2%}")
    
    if method in ["dqn", "both"]:
        d_rates = [r['dqn']['avg_items_placed'] / len(all_results) for r in all_results.values()]
        d_vols = [r['dqn']['avg_volume_utilization'] for r in all_results.values()]
        
        print(f"\nDQN:")
        print(f"  Avg packing rate: {np.mean(d_rates):.1%} ± {np.std(d_rates):.1%}")
        print(f"  Avg volume util: {np.mean(d_vols):.2%} ± {np.std(d_vols):.2%}")


def main():
    """Interactive menu for dataset utilities"""
    print("="*60)
    print("Q4RealBPP Dataset Utilities")
    print("="*60)
    print("\nOptions:")
    print("1. Convert text file to JSON")
    print("2. Batch convert directory")
    print("3. Benchmark single instance")
    print("4. Benchmark suite")
    print("5. Generate sample instance")
    print("6. Exit")
    
    choice = input("\nSelect option (1-6): ").strip()
    
    if choice == "1":
        txt_file = input("Enter text file path: ").strip()
        output_json = input("Enter output JSON path (or press Enter for auto): ").strip()
        if not output_json:
            output_json = None
        convert_txt_to_json(txt_file, output_json)
    
    elif choice == "2":
        input_dir = input("Enter input directory: ").strip()
        output_dir = input("Enter output directory (or press Enter for same): ").strip()
        if not output_dir:
            output_dir = None
        batch_convert_directory(input_dir, output_dir)
    
    elif choice == "3":
        instance_path = input("Enter instance JSON path: ").strip()
        method = input("Method (heuristic/dqn/both) [both]: ").strip() or "both"
        benchmark_instance(instance_path, method=method)
    
    elif choice == "4":
        instance_dir = input("Enter instances directory: ").strip()
        method = input("Method (heuristic/dqn/both) [both]: ").strip() or "both"
        benchmark_suite(instance_dir, method=method)
    
    elif choice == "5":
        num_items = int(input("Number of items [30]: ").strip() or "30")
        num_bins = int(input("Number of bins [2]: ").strip() or "2")
        output_file = input("Output file [sample_instance.json]: ").strip() or "sample_instance.json"
        
        instance = Q4RealBPPLoader.generate_sample_instance(num_items, num_bins)
        
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
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Generated instance saved to {output_file}")
    
    elif choice == "6":
        print("Exiting...")
        return
    
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()
