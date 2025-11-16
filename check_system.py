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
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(i)
            print(f"    CUDA cores: {props.multi_processor_count * 128}")  # Approximate
            print(f"    Memory: {props.total_memory / 1e9:.1f} GB")

# Detect CPUs
num_cpus = cpu_count()
print(f"CPU cores detected: {num_cpus}")

# Simulate auto-detection for different population sizes
print("\n" + "="*60)
print("AUTO-DETECTED CONFIGURATION")
print("="*60)

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

print("\n" + "="*60)
print("RECOMMENDATION FOR YOUR SYSTEM")
print("="*60)

if num_gpus == 1:
    print(f"You have 1 GPU ({torch.cuda.get_device_name(0) if num_gpus > 0 else 'none'})")
    print(f"Auto-detection will use: CPU mode with {min(num_cpus, 4, 8)} workers")
    print()
    print("This is CORRECT because:")
    print("  ✓ Your GPU's CUDA cores are already used when training each genome")
    print("  ✓ Running multiple models on 1 GPU causes OOM errors")
    print("  ✓ CPU parallelism gives 2-4x speedup without crashes")
    print()
    print("Alternative: Sequential GPU (faster per genome, no parallelism)")
    print("  Set: parallel=False, device_mode='single_gpu'")
elif num_gpus > 1:
    print(f"You have {num_gpus} GPUs - lucky you!")
    print(f"Auto-detection will use: multi_gpu mode")
    print("Each worker gets dedicated GPU - best performance!")
else:
    print("No GPU detected - using CPU mode")
    print(f"Will use {min(num_cpus, 8)} workers")

print("="*60)
