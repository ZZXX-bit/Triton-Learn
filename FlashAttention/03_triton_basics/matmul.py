"""
Triton 进阶：Tiled Matrix Multiplication

为什么要学 matmul tiling？
  FlashAttention 的核心操作就是 QK^T 和 (softmax) @ V 这两个 matmul
  理解 tiling 如何让数据在 SRAM 中复用，是理解 FlashAttention IO 分析的基础

内存层级回顾：
  HBM（显存主体）：~2TB/s 带宽，容量大（几十 GB）
  SRAM（片上缓存）：~19TB/s 带宽，容量小（A100 约 192KB/SM）

Tiling 的目标：把频繁访问的数据放进 SRAM，减少 HBM 读写次数
"""

import torch
import triton
import triton.language as tl
from triton.testing import do_bench, perf_report, Benchmark


# ─────────────────────────────────────────────
# Kernel：分块矩阵乘法 C = A @ B
# A: (M, K)  B: (K, N)  C: (M, N)
# ─────────────────────────────────────────────

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,   # A 的行步长、列步长
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,  # 每个 program 负责 C 的 BLOCK_M 行
    BLOCK_N: tl.constexpr,  # 每个 program 负责 C 的 BLOCK_N 列
    BLOCK_K: tl.constexpr,  # K 维度的分块大小
):
    """
    2D grid：每个 program 计算 C 的一个 [BLOCK_M, BLOCK_N] 子块
    通过沿 K 维度循环累加来完成点积
    """
    pid_m = tl.program_id(axis=0)   # 负责第几个行块
    pid_n = tl.program_id(axis=1)   # 负责第几个列块

    # 当前 program 负责的行/列下标范围
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # 累加器：存放当前 program 计算的 C 子块，初始化为 0
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 沿 K 维度分块循环
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # 加载 A 的子块 [BLOCK_M, BLOCK_K]
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=a_mask, other=0.0,
        )

        # 加载 B 的子块 [BLOCK_K, BLOCK_N]
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(
            b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=b_mask, other=0.0,
        )

        # 矩阵乘法累加（在 SRAM 中完成）
        acc += tl.dot(a, b)

    # 写回 C 子块到 HBM
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc.to(tl.float16),
        mask=c_mask,
    )


# ─────────────────────────────────────────────
# Python 封装
# ─────────────────────────────────────────────

def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """C = A @ B，A: (M, K)，B: (K, N)"""
    assert a.ndim == 2 and b.ndim == 2
    assert a.shape[1] == b.shape[0]
    assert a.is_cuda and b.is_cuda
    M, K = a.shape
    K, N = b.shape

    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c


# ─────────────────────────────────────────────
# 验证 & Benchmark
# ─────────────────────────────────────────────

def verify():
    torch.manual_seed(0)
    a = torch.randn(512, 256, device="cuda", dtype=torch.float16)
    b = torch.randn(256, 384, device="cuda", dtype=torch.float16)

    out_triton = matmul(a, b)
    out_torch  = torch.matmul(a, b)

    max_diff = (out_triton - out_torch).abs().max().item()
    print(f"最大误差: {max_diff:.2e}")
    # float16 精度有限，允许稍大误差
    assert max_diff < 1.0, f"误差过大: {max_diff}"
    print("✓ 正确性验证通过")


@perf_report(
    Benchmark(
        x_names=["M", "N", "K"],
        x_vals=[(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)],
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "PyTorch"],
        ylabel="TFLOPS",
        plot_name="matmul-performance",
        args={},
    )
)
def benchmark(M, N, K, provider):
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":
        ms, min_ms, max_ms = do_bench(lambda: matmul(a, b), quantiles=quantiles)
    else:
        ms, min_ms, max_ms = do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)

    tflops = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return tflops(ms), tflops(max_ms), tflops(min_ms)


if __name__ == "__main__":
    verify()
    benchmark.run(print_data=True)
