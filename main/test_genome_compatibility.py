"""
Quick test to validate genome compatibility with new genes.
Run this to verify the optimized neuroevolution is working.
"""

import sys
sys.path.insert(0, '.')

def test_genome_creation():
    """Test that genomes can be created with all new genes"""
    from genome import NetworkGenome

    print("=" * 80)
    print("TESTING GENOME COMPATIBILITY")
    print("=" * 80)

    # Test 1: Random genome creation
    print("\n[Test 1] Creating random genome...")
    genome = NetworkGenome()

    required_genes = [
        'hidden_dim', 'enc_layers', 'head_hidden',
        'attention_type', 'attention_heads', 'num_inducing_points',
        'patch_size', 'cnn_channels', 'dropout', 'activation'
    ]

    for gene in required_genes:
        assert gene in genome.genes, f"Missing gene: {gene}"
        print(f"  ✓ {gene}: {genome.genes[gene]}")

    print("\n✅ All genes present!")

    # Test 2: Manual genome creation (like baseline)
    print("\n[Test 2] Creating manual baseline genome...")
    baseline = NetworkGenome(genes={
        'hidden_dim': 256,
        'enc_layers': 2,
        'head_hidden': 256,
        'attention_type': 'standard',
        'attention_heads': 4,
        'num_inducing_points': 32,
        'patch_size': 7,
        'cnn_channels': [16, 32],
        'dropout': 0.0,
        'activation': 'relu',
    })

    print(f"  ✓ Baseline genome created")
    print(f"  Complexity: {baseline.get_network_complexity():.3f}M params")

    # Test 3: Convert to DQN config
    print("\n[Test 3] Converting to DQN config...")
    cfg = baseline.to_dqn_config(
        obs_dim=8,
        action_feat_dim=25,
        max_actions=128,
        device='cpu'
    )

    print(f"  ✓ Config created successfully")
    print(f"    - activation: {cfg.activation}")
    print(f"    - cnn_channels: {cfg.cnn_channels}")
    print(f"    - dropout: {cfg.dropout}")
    print(f"    - attention_type: {cfg.attention_type}")
    print(f"    - use_attention: {cfg.use_attention}")

    # Test 4: Mutation
    print("\n[Test 4] Testing mutation...")
    mutant = baseline.mutate(mutation_rate=0.5)

    different_genes = sum(1 for k in baseline.genes if baseline.genes[k] != mutant.genes[k])
    print(f"  ✓ Mutated {different_genes}/{len(baseline.genes)} genes")

    # Test 5: Crossover
    print("\n[Test 5] Testing crossover...")
    child = NetworkGenome.crossover(baseline, genome, method='uniform')
    print(f"  ✓ Child genome created")
    print(f"  Complexity: {child.get_network_complexity():.3f}M params")

    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED - Genome system is compatible!")
    print("=" * 80)
    return True


def test_different_attention_types():
    """Test all attention type configurations"""
    from genome import NetworkGenome

    print("\n" + "=" * 80)
    print("TESTING ATTENTION TYPES")
    print("=" * 80)

    attention_types = ['standard', 'set_transformer', 'none']

    for att_type in attention_types:
        print(f"\n[{att_type}]")
        genome = NetworkGenome(genes={
            'hidden_dim': 256,
            'enc_layers': 2,
            'head_hidden': 256,
            'attention_type': att_type,
            'attention_heads': 4,
            'num_inducing_points': 32,
            'patch_size': 7,
            'cnn_channels': [16, 32],
            'dropout': 0.1,
            'activation': 'gelu',
        })

        cfg = genome.to_dqn_config(8, 25, 128, 'cpu')

        print(f"  use_attention: {cfg.use_attention}")
        print(f"  attention_type: {cfg.attention_type}")
        print(f"  Complexity: {genome.get_network_complexity():.3f}M params")

    print("\n✅ All attention types work correctly!")


if __name__ == "__main__":
    try:
        test_genome_creation()
        test_different_attention_types()
        print("\n🎉 Neuroevolution system is fully compatible!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
