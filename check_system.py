"""
Quick script to check what multiprocessing settings will be auto-detected.
"""
import torch
from multiprocessing import cpu_count

print("="*60)
print("SYSTEM DETECTION FOR MULTIPROCESSING")
print("="*60)

# Detect GPUs
num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"GPUs detected: {num_gpus}")
if num_gpus > 0:
    print(f"  GPU 0: {torch.cuda.get_device_name(0)}")

# Detect CPUs
num_cpus = cpu_count()
print(f"CPU cores detected: {num_cpus}")

# Simulate auto-detection for different population sizes
print("\nAuto-detected workers for different population sizes:")
print("-" * 60)

for pop_size in [2, 4, 8, 20]:
    if num_gpus > 1:
        device_mode = 'multi_gpu'
        workers = min(num_gpus, pop_size)
    elif num_gpus == 1:
        device_mode = 'cpu'
        workers = min(num_cpus, pop_size, 8)
    else:
        device_mode = 'cpu'
        workers = min(num_cpus, pop_size, 8)

    print(f"Population={pop_size:2d} → Workers={workers}, Mode={device_mode}")

print("=" * 60)
print(f"\nRecommended: Start with population_size=4 and num_workers=None")
print(f"This will use {min(num_cpus, 4, 8)} workers on your system.")
print("=" * 60)
