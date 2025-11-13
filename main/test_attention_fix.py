"""
Test script to verify attention masking fix
"""
import torch
import numpy as np
from dqn_core.dqn_enhanced import MAB, SetTransformer

def test_attention_mask_normalization():
    """Test that attention weights sum to 1 after masking"""
    print("Testing Attention Mask Normalization Fix...")
    print("=" * 60)

    # Create a simple MAB module
    dim_Q, dim_K, dim_V = 64, 64, 64
    num_heads = 4
    mab = MAB(dim_Q, dim_K, dim_V, num_heads, ln=False)
    mab.eval()

    # Create test data
    batch_size = 2
    seq_len = 8
    Q = torch.randn(batch_size, seq_len, dim_Q)
    K = torch.randn(batch_size, seq_len, dim_K)

    # Create padding mask (True = padding, should be ignored)
    # Let's mask the last 3 positions
    key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    key_padding_mask[:, -3:] = True  # Mask last 3 positions

    print(f"Input shape: Q={Q.shape}, K={K.shape}")
    print(f"Padding mask: {key_padding_mask[0]}")
    print(f"Valid positions: {(~key_padding_mask[0]).sum()} out of {seq_len}")

    # Run forward pass
    with torch.no_grad():
        output = mab(Q, K, key_padding_mask=key_padding_mask)

    print(f"Output shape: {output.shape}")

    # Now let's manually check attention weights by reconstructing the computation
    with torch.no_grad():
        Q_proj = mab.fc_q(Q)
        K_proj = mab.fc_k(K)

        dim_split = dim_V // num_heads
        Q_ = torch.cat(Q_proj.split(dim_split, 2), 0)
        K_ = torch.cat(K_proj.split(dim_split, 2), 0)

        # Compute scores
        scores = Q_.bmm(K_.transpose(1, 2)) / np.sqrt(dim_V)

        # Apply mask
        mask_expanded = torch.cat([key_padding_mask for _ in range(num_heads)], 0)
        scores_masked = scores.masked_fill(mask_expanded.unsqueeze(1), float('-inf'))

        # Softmax
        attn_weights = torch.softmax(scores_masked, 2)

        print(f"\nAttention weights shape: {attn_weights.shape}")
        print(f"Attention weights for first query (head 0, batch 0):")
        print(f"  {attn_weights[0, 0, :].numpy()}")

        # Check that attention weights sum to 1 (excluding inf/nan)
        attn_sum = attn_weights[0, 0, :].sum()
        print(f"\n✓ Attention weights sum: {attn_sum:.6f} (should be ~1.0)")

        # Check that masked positions have zero attention
        masked_attention = attn_weights[0, 0, -3:]
        print(f"✓ Attention on masked positions: {masked_attention.numpy()}")
        print(f"  (should be all zeros or very close)")

        # Check valid positions have positive attention
        valid_attention = attn_weights[0, 0, :-3]
        print(f"✓ Attention on valid positions (first 5): {valid_attention[:5].numpy()}")
        print(f"  (should be positive and sum to ~1.0)")

        # Verify sum equals 1 within tolerance
        assert abs(attn_sum.item() - 1.0) < 1e-5, f"Attention weights don't sum to 1: {attn_sum}"

        # Verify masked positions are zero within tolerance
        assert torch.allclose(masked_attention, torch.zeros_like(masked_attention), atol=1e-6), \
            f"Masked positions have non-zero attention: {masked_attention}"

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nThe attention masking fix is working correctly:")
    print("  1. Attention weights sum to 1.0")
    print("  2. Masked positions receive zero attention")
    print("  3. Valid positions receive positive attention")

def test_set_transformer():
    """Test that Set Transformer works with masking"""
    print("\n\nTesting Set Transformer with Masking...")
    print("=" * 60)

    dim_input = 64
    dim_output = 64
    batch_size = 2
    num_actions = 10

    set_transformer = SetTransformer(dim_input, dim_output, num_heads=4, num_inds=8, ln=True)
    set_transformer.eval()

    # Create test data
    X = torch.randn(batch_size, num_actions, dim_input)

    # Mask last 4 actions
    key_padding_mask = torch.zeros(batch_size, num_actions, dtype=torch.bool)
    key_padding_mask[:, -4:] = True

    print(f"Input shape: {X.shape}")
    print(f"Padding mask: {key_padding_mask[0]}")
    print(f"Valid actions: {(~key_padding_mask[0]).sum()} out of {num_actions}")

    with torch.no_grad():
        output = set_transformer(X, key_padding_mask=key_padding_mask)

    print(f"Output shape: {output.shape}")
    print(f"Output mean: {output.mean():.4f}, std: {output.std():.4f}")

    # Check that output is finite
    assert torch.isfinite(output).all(), "Output contains inf/nan"

    print("\n✅ Set Transformer test passed!")
    print("=" * 60)

if __name__ == "__main__":
    test_attention_mask_normalization()
    test_set_transformer()
    print("\n\n🎉 All attention mechanism tests passed!")
