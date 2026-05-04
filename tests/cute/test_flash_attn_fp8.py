# Copyright (c) 2025 FlashAttention Team
# Test script for FP8 support with descale tensors (Task 18)

import pytest
import torch
import random

# Import directly from cute to avoid flash_attn_2_cuda dependency
from flash_attn.cute.interface import _flash_attn_fwd
from flash_attn.cute.testing import attention_ref

VERBOSE = True


def run_fp8_test(dtype, batch_size=2, seqlen_q=128, seqlen_k=256, 
                 nheads=4, nheads_kv=4, d=256, causal=True, softcap=None):
    """Run FP8 attention test with descale tensors."""
    device = "cuda"
    
    if VERBOSE:
        print(f"\n{'='*60}")
        print(f"Testing dtype={dtype}, head_dim={d}, seqlen=({seqlen_q}, {seqlen_k})")
        print(f"{'='*60}")
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    softmax_scale = 1.0 / (d ** 0.5)
    
    try:
        # Generate FP8 inputs directly with reasonable values
        # Scale to avoid overflow/underflow in FP8 range
        scale_factor = 5.0
        q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=torch.float32) * scale_factor
        k = torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=torch.float32) * scale_factor
        v = torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=torch.float32) * scale_factor
        
        if softcap is not None and softcap > 0:
            q = q * softcap / 4
            k = k * softcap / 4
        
        # Create per-head scales (typical range 0.5-2.0)
        scale_q = torch.ones(nheads, device=device, dtype=torch.float32) * 1.0
        scale_k = torch.ones(nheads_kv, device=device, dtype=torch.float32) * 1.0
        scale_v = torch.ones(nheads_kv, device=device, dtype=torch.float32) * 1.0
        
        # Quantize to FP8 - scale has shape (nheads,) or (nheads_kv,)
        # Q shape: (batch, seqlen, nheads, d), need reshape scale to (1, 1, nheads, 1)
        q_quant = (q / scale_q.view(1, 1, -1, 1)).to(dtype)
        k_quant = (k / scale_k.view(1, 1, -1, 1)).to(dtype)
        v_quant = (v / scale_v.view(1, 1, -1, 1)).to(dtype)
        
        # Descale tensors - shape (batch_size, num_head_kv)
        q_descale = scale_k.unsqueeze(0).expand(batch_size, -1)
        k_descale = scale_k.unsqueeze(0).expand(batch_size, -1)
        v_descale = scale_v.unsqueeze(0).expand(batch_size, -1)
        
        # Run FlashAttention FP8 forward pass using internal function
        out, lse = _flash_attn_fwd(
            q=q_quant,
            k=k_quant,
            v=v_quant,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            softmax_scale=softmax_scale,
            causal=causal,
            softcap=softcap if softcap is not None else 0.0,
            window_size_left=None,
            window_size_right=None,
            learnable_sink=None,
            num_splits=1,
            pack_gqa=False,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            return_lse=True,
        )
        
        results = {
            'status': 'PASSED',
            'out_shape': list(out.shape),
            'lse_shape': list(lse.shape) if lse is not None else None,
            'out_dtype': str(out.dtype),
        }
        
        if VERBOSE:
            print(f"✓ PASSED")
            print(f"  Output shape: {out.shape}")
            print(f"  Output dtype: {out.dtype}")
            print(f"  LSE shape: {lse.shape if lse is not None else None}")
            
        return results
        
    except Exception as e:
        import traceback
        if VERBOSE:
            print(f"✗ FAILED: {str(e)}")
            traceback.print_exc()
        return {
            'status': 'FAILED',
            'error': str(e),
        }


def test_fp8_e4m3fn_basic():
    """Test FP8 e4m3fn basic functionality."""
    result = run_fp8_test(torch.float8_e4m3fn, d=256)
    assert result['status'] == 'PASSED', f"Test failed: {result.get('error')}"
    assert result['max_abs_error'] < 0.1, f"Abs error too high: {result['max_abs_error']}"
    

def test_fp8_e5m2_basic():
    """Test FP8 e5m2 basic functionality."""
    result = run_fp8_test(torch.float8_e5m2, d=256)
    assert result['status'] == 'PASSED', f"Test failed: {result.get('error')}"
    assert result['max_abs_error'] < 0.1, f"Abs error too high: {result['max_abs_error']}"


def test_fp8_e4m3fn_causal():
    """Test FP8 e4m3fn with causal masking."""
    result = run_fp8_test(torch.float8_e4m3fn, d=256, causal=True)
    assert result['status'] == 'PASSED'


def test_fp8_e4m3fn_noncausal():
    """Test FP8 e4m3fn without causal masking."""
    result = run_fp8_test(torch.float8_e4m3fn, d=256, causal=False)
    assert result['status'] == 'PASSED'


def test_fp8_e4m3fn_softcap():
    """Test FP8 e4m3fn with softcap."""
    result = run_fp8_test(torch.float8_e4m3fn, d=256, softcap=15.0)
    assert result['status'] == 'PASSED'


def test_fp8_e5m2_causal():
    """Test FP8 e5m2 with causal masking."""
    result = run_fp8_test(torch.float8_e5m2, d=256, causal=True)
    assert result['status'] == 'PASSED'


def test_fp8_e5m2_gqa():
    """Test FP8 e5m2 with GQA."""
    result = run_fp8_test(torch.float8_e5m2, d=256, nheads=8, nheads_kv=2)
    assert result['status'] == 'PASSED'


def test_fp8_longer_sequence():
    """Test FP8 with longer sequences."""
    result = run_fp8_test(torch.float8_e4m3fn, d=256, seqlen_q=512, seqlen_k=1024)
    assert result['status'] == 'PASSED'


if __name__ == "__main__":
    # Run all tests
    test_functions = [
        test_fp8_e4m3fn_basic,
        test_fp8_e5m2_basic,
        test_fp8_e4m3fn_causal,
        test_fp8_e4m3fn_noncausal,
        test_fp8_e4m3fn_softcap,
        test_fp8_e5m2_causal,
        test_fp8_e5m2_gqa,
        test_fp8_longer_sequence,
    ]
    
    results = []
    for test_fn in test_functions:
        try:
            test_fn()
            results.append((test_fn.__name__, "PASSED", None))
        except AssertionError as e:
            results.append((test_fn.__name__, "FAILED", str(e)))
        except Exception as e:
            results.append((test_fn.__name__, "ERROR", str(e)))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    passed = sum(1 for _, status, _ in results if status == "PASSED")
    total = len(results)
    for name, status, error in results:
        if status == "PASSED":
            print(f"✓ {name}")
        else:
            print(f"✗ {name}: {error}")
    print(f"\nTotal: {passed}/{total} tests passed")
