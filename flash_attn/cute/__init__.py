"""Flash Attention CUTE (CUDA Template Engine) implementation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fa4")
except PackageNotFoundError:
    __version__ = "0.0.0"

import cutlass.cute as cute

from .interface import (
    flash_attn_func,
    flash_attn_varlen_func,
)

from flash_attn.cute.cute_dsl_utils import cute_compile_patched

cute.compile = cute_compile_patched

from flash_attn.cute.sm100_hd256_2cta_fmha_forward import (
    BlackwellFusedMultiHeadAttentionForward,
)
from flash_attn.cute.sm100_hd256_2cta_fmha_backward import (
    BlackwellFusedMultiHeadAttentionBackward,
)
from flash_attn.cute.sm100_hd256_2cta_fmha_backward_dqkernel import (
    BlackwellFusedMultiHeadAttentionBackwardDQKernel,
)
from flash_attn.cute.sm100_hd256_2cta_fmha_backward_dkdvkernel import (
    BlackwellFusedMultiHeadAttentionBackwardDKDVKernel,
)
from flash_attn.cute.flash_fwd_mla_sm100 import (
    FlashAttentionMLAForwardSm100,
)
from flash_attn.cute.topk_gather_kv import (
    CpasyncGatherKVManager,
)

sm100_hd256_2cta_fmha_forward = BlackwellFusedMultiHeadAttentionForward
sm100_hd256_2cta_fmha_backward = BlackwellFusedMultiHeadAttentionBackward
sm100_hd256_2cta_fmha_backward_dqkernel = BlackwellFusedMultiHeadAttentionBackwardDQKernel
sm100_hd256_2cta_fmha_backward_dkdvkernel = BlackwellFusedMultiHeadAttentionBackwardDKDVKernel
flash_fwd_mla_sm100 = FlashAttentionMLAForwardSm100
topk_gather_kv = CpasyncGatherKVManager

__all__ = [
    "flash_attn_func",
    "flash_attn_varlen_func",
    "sm100_hd256_2cta_fmha_forward",
    "BlackwellFusedMultiHeadAttentionForward",
    "sm100_hd256_2cta_fmha_backward",
    "BlackwellFusedMultiHeadAttentionBackward",
    "sm100_hd256_2cta_fmha_backward_dqkernel",
    "BlackwellFusedMultiHeadAttentionBackwardDQKernel",
    "sm100_hd256_2cta_fmha_backward_dkdvkernel",
    "BlackwellFusedMultiHeadAttentionBackwardDKDVKernel",
    "flash_fwd_mla_sm100",
    "FlashAttentionMLAForwardSm100",
    "topk_gather_kv",
    "CpasyncGatherKVManager",
]
