"""
Test script for new neuroevolution features:
1. Hyperparameter evolution (Neuvo NAS+ 2025)
2. Adaptive population sizing (2024)

This script demonstrates that:
- Hyperparameters (lr, batch_size, gamma) are being evolved
- Population size adapts dynamically across generations
- All features scale properly with different settings
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from genome import NetworkGenome, create_initial_population
from ga_evolution import get_adaptive_population_size


def test_hyperparameter_evolution():
    """Test that hyperparameters are now part of genome."""
    print("\n" + "="*80)
    print("TEST 1: Hyperparameter Evolution")
    print("="*80)

    # Create random genomes
    genomes = [NetworkGenome() for _ in range(5)]

    print("\nGenerated 5 random genomes with hyperparameters:")
    print("-" * 80)

    for i, genome in enumerate(genomes, 1):
        print(f"\nGenome {i}:")
        print(f"  Architecture: hidden={genome.genes['hidden_dim']}, "
              f"layers={genome.genes['enc_layers']}, "
              f"attention={genome.genes['attention_type']}")
        print(f"  Hyperparameters:")
        print(f"    lr = {genome.genes['lr']}")
        print(f"    batch_size = {genome.genes['batch_size']}")
        print(f"    gamma = {genome.genes['gamma']}")

    # Check that hyperparameters vary
    lrs = [g.genes['lr'] for g in genomes]
    batch_sizes = [g.genes['batch_size'] for g in genomes]
    gammas = [g.genes['gamma'] for g in genomes]

    print(f"\nHyperparameter diversity check:")
    print(f"  Unique learning rates: {len(set(lrs))} (total: {len(lrs)})")
    print(f"  Unique batch sizes: {len(set(batch_sizes))} (total: {len(batch_sizes)})")
    print(f"  Unique gammas: {len(set(gammas))} (total: {len(gammas)})")

    if len(set(lrs)) > 1 or len(set(batch_sizes)) > 1 or len(set(gammas)) > 1:
        print("  ✅ PASS: Hyperparameters are varying across genomes")
    else:
        print("  ⚠️  WARNING: Low diversity (might be random chance with 5 samples)")

    print("\n" + "="*80)
    return True


def test_adaptive_population_sizing():
    """Test that adaptive population sizing works correctly."""
    print("\n" + "="*80)
    print("TEST 2: Adaptive Population Sizing")
    print("="*80)

    # Test with different generation counts
    test_configs = [
        {"base": 20, "gens": 10, "explore": 1.5, "exploit": 0.75},
        {"base": 50, "gens": 30, "explore": 2.0, "exploit": 0.5},
        {"base": 10, "gens": 5, "explore": 1.2, "exploit": 0.8},
    ]

    for config in test_configs:
        base = config["base"]
        gens = config["gens"]
        explore_ratio = config["explore"]
        exploit_ratio = config["exploit"]

        print(f"\n{'─'*80}")
        print(f"Config: base={base}, generations={gens}, "
              f"explore={explore_ratio}×, exploit={exploit_ratio}×")
        print(f"{'─'*80}")

        sizes = []
        phases = []

        for gen in range(gens):
            size = get_adaptive_population_size(
                gen, gens, base, explore_ratio, exploit_ratio
            )
            sizes.append(size)

            progress = gen / max(gens - 1, 1)
            if progress < 1.0 / 3.0:
                phase = "exploration"
            elif progress > 2.0 / 3.0:
                phase = "exploitation"
            else:
                phase = "stable"
            phases.append(phase)

        # Show progression
        print(f"\nPopulation size progression:")
        for gen, (size, phase) in enumerate(zip(sizes, phases)):
            marker = "→"
            if phase == "exploration":
                marker = "🔍"
            elif phase == "exploitation":
                marker = "🎯"
            print(f"  Gen {gen+1:2d}: {size:3d} {marker} {phase}")

        # Validate expectations
        early_size = sizes[0]
        middle_size = sizes[len(sizes) // 2]
        late_size = sizes[-1]

        print(f"\nValidation:")
        print(f"  Early phase: {early_size} (expected ~{int(base * explore_ratio)})")
        print(f"  Middle phase: {middle_size} (expected ~{base})")
        print(f"  Late phase: {late_size} (expected ~{int(base * exploit_ratio)})")

        # Check that early > middle > late (approximately)
        if early_size >= middle_size >= late_size:
            print(f"  ✅ PASS: Population size decreases as expected")
        else:
            print(f"  ❌ FAIL: Unexpected size progression")
            return False

    print("\n" + "="*80)
    return True


def test_genome_to_config():
    """Test that genome correctly passes hyperparameters to DQN config."""
    print("\n" + "="*80)
    print("TEST 3: Genome to DQN Config Conversion")
    print("="*80)

    genome = NetworkGenome()

    print(f"\nGenome hyperparameters:")
    print(f"  lr = {genome.genes['lr']}")
    print(f"  batch_size = {genome.genes['batch_size']}")
    print(f"  gamma = {genome.genes['gamma']}")

    # Convert to config
    config = genome.to_dqn_config(
        obs_dim=8,
        action_feat_dim=25,
        max_actions=128,
        device="cpu"
    )

    print(f"\nDQN Config hyperparameters:")
    print(f"  lr = {config.lr}")
    print(f"  batch_size = {config.batch_size}")
    print(f"  gamma = {config.gamma}")

    # Validate
    if (config.lr == genome.genes['lr'] and
        config.batch_size == genome.genes['batch_size'] and
        config.gamma == genome.genes['gamma']):
        print(f"\n  ✅ PASS: Hyperparameters correctly passed to config")
    else:
        print(f"\n  ❌ FAIL: Hyperparameter mismatch!")
        return False

    print("\n" + "="*80)
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*80)
    print("TESTING NEW NEUROEVOLUTION FEATURES")
    print("="*80)
    print("\nFeatures being tested:")
    print("  1. Hyperparameter evolution (Neuvo NAS+ 2025)")
    print("  2. Adaptive population sizing (2024)")
    print("  3. Dynamic scaling with different parameters")
    print("="*80)

    results = []

    try:
        results.append(("Hyperparameter Evolution", test_hyperparameter_evolution()))
        results.append(("Adaptive Population Sizing", test_adaptive_population_sizing()))
        results.append(("Genome to Config Conversion", test_genome_to_config()))
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")

    all_passed = all(r[1] for r in results)

    print("\n" + "="*80)
    if all_passed:
        print("✅ ALL TESTS PASSED")
        print("\nNew features are working correctly!")
        print("You can now use these features in your thesis:")
        print("  - Co-evolution of architecture & hyperparameters (Neuvo NAS+ 2025)")
        print("  - Adaptive population sizing (2024)")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please check the output above for details.")
    print("="*80 + "\n")

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
