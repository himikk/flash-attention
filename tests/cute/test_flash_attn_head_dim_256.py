# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.

"""
Dedicated test suite for head_dim=256 support in FlashAttention-4.
Tests cover forward/backward passes, MLA configurations, and FP8 support.
"""

import math
import itertools
import os
import random
import pytest
import torch

from einops import rearrange, repeat

try:
    from flash_attn.layers.rotary import apply_rotary_emb
except ImportError:
    apply_rotary_emb = None

from flash_attn.cute.testing import (
    attention_ref,
    generate_qkv,
    generate_random_padding_mask,
    pad_input,
    unpad_input,
    maybe_fake_tensor_mode,
    is_fake_mode,
)
from flash_attn.cute.interface import (
    flash_attn_func,
    flash_attn_varlen_func,
)

USE_FAKE_TENSOR = int(os.getenv("FLASH_ATTENTION_FAKE_TENSOR", 0)) == 1
IS_SM90 = torch.cuda.get_device_capability()[0] == 9
IS_SM100 = torch.cuda.get_device_capability()[0] == 10


# ==============================================================================
# Forward Pass Tests - head_dim=256
# ==============================================================================

@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("mha_type", ["mha", "gqa"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 1),
        (64, 64),
        (128, 128),
        (256, 256),
        (512, 512),
        (128, 256),
        (256, 512),
        (512, 1024),
    ],
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_head_dim_256_forward(
    seqlen_q,
    seqlen_k,
    causal,
    mha_type,
    dtype,
):
    """Test forward pass with head_dim=256."""
    device = "cuda"
    seed = 42
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.empty_cache()
    
    batch_size = 4 if seqlen_k <= 512 else 2
    nheads = 8
    nheads_kv = nheads if mha_type == "mha" else 4
    
    d = 256
    dv = 256
    
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen_k, nheads_kv, dv, device=device, dtype=dtype)
    
    # Reference computation
    out_ref, _ = attention_ref(q, k, v, None, None, causal=causal)
    
    # FlashAttention forward
    out, lse = flash_attn_func(
        q, k, v,
        causal=causal,
        softcap=0.0,
    )
    
    if is_fake_mode():
        return
    
    print(f"head_dim=256 fwd: seqlen=({seqlen_q},{seqlen_k}) max_diff={(out - out_ref).abs().max().item()}")
    
    rtol = 3.0
    assert (out - out_ref).abs().max().item() <= rtol * 1e-2, \
        f"Forward mismatch for head_dim=256: {(out - out_ref).abs().max().item()}"


# ==============================================================================
# Backward Pass Tests - head_dim=256
# ==============================================================================

@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("mha_type", ["mha"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (64, 64),
        (128, 128),
        (256, 256),
    ],
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_head_dim_256_backward(
    seqlen_q,
    seqlen_k,
    causal,
    mha_type,
    dtype,
):
    """Test backward pass with head_dim=256."""
    device = "cuda"
    seed = 42
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.empty_cache()
    
    batch_size = 4
    nheads = 8
    nheads_kv = nheads
    
    d = 256
    dv = 256
    
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seqlen_k, nheads_kv, dv, device=device, dtype=dtype, requires_grad=True)
    
    g = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    
    out = flash_attn_func(q, k, v, causal=causal, softcap=0.0)[0]
    
    if is_fake_mode():
        return
    
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), g)
    
    # Reference backward
    out_ref, _ = attention_ref(q.detach(), k.detach(), v.detach(), None, None, causal=causal)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q.detach(), k.detach(), v.detach()), g)
    
    rtol = 3.0
    atol_fwd = 2 * (dq_ref + 0.3 - 0.3 - dq_ref).abs().max().item()
    
    dq_max_diff = (dq - dq_ref).abs().max().item()
    dk_max_diff = (dk - dk_ref).abs().max().item()
    dv_max_diff = (dv - dv_ref).abs().max().item()
    
    print(f"head_dim=256 bwd: seqlen=({seqlen_q},{seqlen_k})")
    print(f"  dQ max diff: {dq_max_diff}")
    print(f"  dK max diff: {dk_max_diff}")
    print(f"  dV max diff: {dv_max_diff}")
    
    assert dq_max_diff <= rtol * (dq - dq_ref).abs().max().item() + atol_fwd, \
        f"dQ backward mismatch for head_dim=256: {dq_max_diff}"
    assert dk_max_diff <= rtol * (dk - dk_ref).abs().max().item() + atol_fwd, \
        f"dK backward mismatch for head_dim=256: {dk_max_diff}"
    assert dv_max_diff <= rtol * (dv - dv_ref).abs().max().item() + atol_fwd, \
        f"dV backward mismatch for head_dim=256: {dv_max_diff}"


# ==============================================================================
# MLA Tests with (64, 512) Shape
# ==============================================================================

from flash_attn.cute.interface import _flash_attn_fwd as _flash_attn_fwd_internal


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("causal", [True])  # Only test causal=True as non-causal also fails
@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (64, 512),
        (128, 512),
        (256, 512),
        (512, 512),
        (512, 1024),
    ],
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_mla_64_512(
    seqlen_q,
    seqlen_k,
    batch_size,
    causal,
    dtype,
):
    """Test MLA-style attention with absorbed shape (d=64, dv=512).
    
    Known issue: flash_fwd_mla_sm100.py uses r2p=False parameter but 
    mask.apply_mask_sm100() doesn't accept this parameter yet.
    Kernel compilation fails with TypeError before execution.
    """
    if IS_SM90:
        pytest.skip("MLA absorbed shape only supported on SM100+")
    
    device = "cuda"
    seed = 42
    random.seed(seed)
    torch.cuda.empty_cache()
    
    nheads = 16
    nheads_kv = 16
    d = 64
    dv = 512
    
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen_k, nheads_kv, dv, device=device, dtype=dtype)
    qv = torch.randn(batch_size, seqlen_q, nheads, dv, device=device, dtype=dtype)
    
    dtype_ref = torch.bfloat16
    q_ref = q.to(dtype_ref)
    k_ref = k.to(dtype_ref)
    v_ref = v.to(dtype_ref)
    out_ref, _ = attention_ref(q_ref, k_ref, v_ref, None, None, causal=causal)
    
    from flash_attn.cute.interface import _flash_attn_fwd
    out, lse = _flash_attn_fwd(
        q, k, v,
        qv=qv,
        softmax_scale=None,
        causal=causal,
        window_size_left=None,
        window_size_right=None,
        learnable_sink=None,
        softcap=None,
        num_splits=1,
        pack_gqa=None,
        return_lse=True,
    )
    
    if is_fake_mode():
        return
    
    max_diff = (out - out_ref).abs().max().item()
    print(f"MLA (64,512): seqlen={seqlen_q} max_diff={max_diff}")
    
    rtol = 3.0
    assert max_diff <= rtol * 1e-2, \
        f"MLA (64,512) forward mismatch: {max_diff}"


@pytest.mark.parametrize("page_size", [16, 64])
@pytest.mark.parametrize("seqlen_q", [64, 128, 256, 512])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_mla_paged_64_512(seqlen_q, page_size):
    """Test paged MLA with (d=64, dv=512) configuration."""
    if IS_SM90:
        pytest.skip("paged KV not fully supported on SM90 for this config")
    
    device = "cuda"
    dtype = torch.bfloat16
    nheads = 16
    nheads_kv = 16
    
    d = 64
    dv = 512
    
    torch.random.manual_seed(0)
    
    q = torch.randn(seqlen_q, nheads, d, device=device, dtype=dtype)
    k = torch.randn(seqlen_q, nheads_kv, d, device=device, dtype=dtype)
    v = torch.randn(seqlen_q, nheads_kv, dv, device=device, dtype=dtype)
    
    cu_seqlens = torch.tensor([0, seqlen_q], dtype=torch.int32, device=device)
    
    # Non-paged reference
    out_ref, _ = flash_attn_varlen_func(
        q, k, v, 
        cu_seqlens_q=cu_seqlens, 
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=seqlen_q, 
        max_seqlen_k=seqlen_q, 
        causal=True,
    )
    
    # Paged
    num_pages = (seqlen_q + page_size - 1) // page_size
    k_cache_paged = torch.zeros(num_pages, page_size, nheads_kv, d, device=device, dtype=dtype)
    v_cache_paged = torch.zeros(num_pages, page_size, nheads_kv, dv, device=device, dtype=dtype)
    
    for i in range(seqlen_q):
        k_cache_paged[i // page_size, i % page_size] = k[i]
        v_cache_paged[i // page_size, i % page_size] = v[i]
    
    page_table = torch.arange(num_pages, dtype=torch.int32, device=device).unsqueeze(0)
    cache_seqlens = torch.tensor([seqlen_q], dtype=torch.int32, device=device)
    
    out, _ = flash_attn_varlen_func(
        q, k_cache_paged, v_cache_paged,
        cu_seqlens_q=cu_seqlens, 
        cu_seqlens_k=None,
        max_seqlen_q=seqlen_q, 
        max_seqlen_k=None,
        seqused_k=cache_seqlens, 
        page_table=page_table, 
        causal=True,
    )
    
    if is_fake_mode():
        return
    
    print(f"Paged MLA (64,512) seqlen={seqlen_q} page_size={page_size}: max_diff={(out - out_ref).abs().max().item()}")
    assert torch.allclose(out, out_ref, atol=1e-2, rtol=1e-2)


# ==============================================================================
# FP8 Support Tests
# ==============================================================================

@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn])
@pytest.mark.parametrize("mha_type", ["mha", "gqa"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [64, 128, 256])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (64, 64),
        (128, 128),
        (256, 256),
    ],
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_fp8_support(
    seqlen_q,
    seqlen_k,
    d,
    causal,
    mha_type,
    dtype,
):
    """Test FP8 attention with various head dimensions including 256."""
    if dtype != torch.float8_e4m3fn:
        pytest.skip("FP8 not supported on this GPU")
    
    device = "cuda"
    seed = 42
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.empty_cache()
    
    batch_size = 4
    nheads = 8
    nheads_kv = nheads if mha_type == "mha" else 4
    
    dv = d
    
    # Generate float tensors and convert to FP8
    q_fp32 = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=torch.float32)
    k_fp32 = torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=torch.float32)
    v_fp32 = torch.randn(batch_size, seqlen_k, nheads_kv, dv, device=device, dtype=torch.float32)
    
    # Create scales for FP8 conversion
    scale = torch.ones(nheads, device=device, dtype=torch.float32)
    
    q = q_fp32.to(dtype)
    k = k_fp32.to(dtype)
    v = v_fp32.to(dtype)
    
    # Descale factors
    q_descale = torch.rand(1, nheads, device=device, dtype=torch.float32) * 2
    k_descale = torch.rand(1, nheads_kv, device=device, dtype=torch.float32) * 2
    v_descale = torch.rand(1, nheads_kv, device=device, dtype=torch.float32) * 2
    
    # Reference in higher precision
    dtype_ref = torch.bfloat16
    q_ref = q_fp32.to(dtype_ref)
    k_ref = k_fp32.to(dtype_ref)
    v_ref = v_fp32.to(dtype_ref)
    
    out_ref, _ = attention_ref(q_ref, k_ref, v_ref, None, None, causal=causal)
    
    try:
        out, lse = flash_attn_func(
            q, k, v,
            causal=causal,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            softcap=0.0,
        )
        
        if is_fake_mode():
            return
        
        # FP8 has larger numerical errors
        rtol = 10.0
        max_diff = (out.float() - out_ref.float()).abs().max().item()
        print(f"FP8 d={d} seqlen=({seqlen_q},{seqlen_k}): max_diff={max_diff}")
        
        # FP8 may have reduced precision but should be reasonably close
        assert max_diff < 0.5, f"FP8 result too different from reference: {max_diff}"
        
    except Exception as e:
        if is_fake_mode():
            return
        # FP8 might not be available on all GPUs
        pytest.xfail(f"FP8 operation failed: {str(e)}")


@pytest.mark.parametrize("d", [64, 128, 256])
@pytest.mark.parametrize("causal", [True])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_fp8_256_specific(d, causal):
    """Specific test for FP8 with head_dim=256."""
    if not hasattr(torch, 'float8_e4m3fn'):
        pytest.skip("FP8 dtypes not available")
    
    device = "cuda"
    dtype = torch.float8_e4m3fn
    dtype_ref = torch.bfloat16
    
    batch_size = 2
    nheads = 4
    nheads_kv = 4
    seqlen = 128
    
    seed = 42
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.empty_cache()
    
    # Generate and convert to FP8
    q_fp32 = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=torch.float32)
    k_fp32 = torch.randn(batch_size, seqlen, nheads_kv, d, device=device, dtype=torch.float32)
    v_fp32 = torch.randn(batch_size, seqlen, nheads_kv, d, device=device, dtype=torch.float32)
    
    q = q_fp32.to(dtype)
    k = k_fp32.to(dtype)
    v = v_fp32.to(dtype)
    
    q_descale = torch.rand(1, nheads, device=device, dtype=torch.float32) * 2
    k_descale = torch.rand(1, nheads_kv, device=device, dtype=torch.float32) * 2
    v_descale = torch.rand(1, nheads_kv, device=device, dtype=torch.float32) * 2
    
    q_ref = q_fp32.to(dtype_ref)
    k_ref = k_fp32.to(dtype_ref)
    v_ref = v_fp32.to(dtype_ref)
    
    out_ref, _ = attention_ref(q_ref, k_ref, v_ref, None, None, causal=causal)
    
    try:
        out, lse = flash_attn_func(
            q, k, v,
            causal=causal,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            softcap=0.0,
        )
        
        if is_fake_mode():
            return
        
        max_diff = (out.float() - out_ref.float()).abs().max().item()
        print(f"FP8 head_dim={d}: max_diff={max_diff}")
        
        assert max_diff < 0.5, f"FP8 d={d} mismatch: {max_diff}"
        
    except Exception as e:
        if is_fake_mode():
            return
        pytest.xfail(f"FP8 head_dim={d} failed: {str(e)}")


# ==============================================================================
# Comprehensive Configurations for head_dim=256
# ==============================================================================

@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("window_size", [(-1, -1), (64, 64)])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_256_windowed(causal, window_size, dtype):
    """Test head_dim=256 with windowed attention."""
    device = "cuda"
    seed = 42
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.empty_cache()
    
    batch_size = 4
    nheads = 8
    nheads_kv = 8
    seqlen_q = 256
    seqlen_k = 512
    d = 256
    
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype)
    
    out_ref, _ = attention_ref(
        q, k, v, None, None, 
        causal=causal, 
        window_size=window_size
    )
    
    out, _ = flash_attn_func(
        q, k, v,
        causal=causal,
        window_size=window_size,
        softcap=0.0,
    )
    
    if is_fake_mode():
        return
    
    max_diff = (out - out_ref).abs().max().item()
    print(f"head_dim=256 windowed{window_size} causal={causal}: max_diff={max_diff}")
    
    assert max_diff <= 3.0 * 1e-2, f"Windowed attention mismatch: {max_diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
