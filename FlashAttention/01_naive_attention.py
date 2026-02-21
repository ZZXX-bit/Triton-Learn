"""
阶段一：朴素 Scaled Dot-Product Attention 实现

目标：
1. 理解标准 Attention 的计算流程
2. 观察 O(N²) 的显存占用
3. 为理解 FlashAttention 的优化动机打基础

公式：Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) @ V
"""

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 1. 朴素实现：完整展开每一步，方便理解
# ─────────────────────────────────────────────

def naive_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Args:
        Q: (batch, heads, seq_len, d_k)
        K: (batch, heads, seq_len, d_k)
        V: (batch, heads, seq_len, d_v)
    Returns:
        out: (batch, heads, seq_len, d_v)

    内存分析：
        - scores = QK^T  →  shape (batch, heads, N, N)  ← 这里是 O(N²) 的大矩阵！
        - 当 N=4096, heads=32, batch=1, float16 时：
          4096 * 4096 * 32 * 2 bytes ≈ 1 GB 仅用于存 attention scores
    """
    d_k = Q.shape[-1]
    scale = d_k ** -0.5

    # Step 1: 计算 attention scores，产生 N×N 矩阵 —— 显存瓶颈所在
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (B, H, N, N)

    # Step 2: softmax 归一化（需要读写整个 N×N 矩阵）
    attn_weights = torch.softmax(scores, dim=-1)            # (B, H, N, N)

    # Step 3: 加权求和
    out = torch.matmul(attn_weights, V)                     # (B, H, N, d_v)

    return out


# ─────────────────────────────────────────────
# 2. 带 causal mask 的版本（用于自回归模型）
# ─────────────────────────────────────────────

def naive_causal_attention(Q, K, V):
    """
    Causal attention：每个 token 只能看到自己及之前的 token
    通过在 softmax 前将未来位置设为 -inf 实现
    """
    _, _, N, d_k = Q.shape
    scale = d_k ** -0.5

    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

    # 构造 causal mask：上三角（不含对角线）设为 -inf
    # triu表示取上三角，diagonal=1 表示上三角不含对角线
    causal_mask = torch.triu(torch.ones(N, N, device=Q.device), diagonal=1).bool()
    scores = scores.masked_fill(causal_mask, float('-inf'))

    attn_weights = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn_weights, V)
    return out



# ─────────────────────────────────────────────
# 3. 显存占用分析
# ─────────────────────────────────────────────

def memory_analysis():
    """
    分析不同序列长度下 attention score 矩阵的显存占用

    关键结论：
    - QKV 本身是 O(N·d) 的线性增长
    - attention score 矩阵是 O(N²) 的二次增长
    - 当 N 很大时，N² 项完全主导显存
    """
    print("=" * 55)
    print(f"{'seq_len':>10} {'QKV (MB)':>12} {'Score矩阵 (MB)':>16} {'比值':>8}")
    print("=" * 55)

    batch, heads, d_k = 1, 32, 64
    bytes_per_elem = 2  # float16

    for N in [512, 1024, 2048, 4096, 8192]:
        qkv_mem  = 3 * batch * heads * N * d_k * bytes_per_elem / 1e6
        score_mem = batch * heads * N * N * bytes_per_elem / 1e6
        ratio = score_mem / qkv_mem
        print(f"{N:>10,} {qkv_mem:>12.1f} {score_mem:>16.1f} {ratio:>8.1f}x")

    print("=" * 55)
    print("\n结论：序列长度翻倍，Score 矩阵显存增大 4 倍（二次方）")
    print("FlashAttention 的核心目标：避免将完整 N×N 矩阵写入 HBM\n")


# ─────────────────────────────────────────────
# 4. 正确性验证 & 性能对比
# ─────────────────────────────────────────────

def verify_correctness():
    """验证朴素实现与 PyTorch 内置实现结果一致"""
    torch.manual_seed(42)
    B, H, N, d = 2, 8, 128, 64
    Q = torch.randn(B, H, N, d, dtype=torch.float32)
    K = torch.randn(B, H, N, d, dtype=torch.float32)
    V = torch.randn(B, H, N, d, dtype=torch.float32)

    out_naive  = naive_attention(Q, K, V)
    # PyTorch 内置（F.scaled_dot_product_attention 在 2.0+ 可用）
    out_torch  = F.scaled_dot_product_attention(Q, K, V)

    max_diff = (out_naive - out_torch).abs().max().item()
    print(f"朴素实现 vs PyTorch 内置：最大误差 = {max_diff:.2e}")
    assert max_diff < 1e-5, "结果不一致！"
    print("✓ 正确性验证通过\n")


def benchmark(device="cuda"):
    """对比不同序列长度下的显存实际占用"""
    if not torch.cuda.is_available():
        print("无 GPU，跳过 benchmark")
        return

    print(f"{'seq_len':>10} {'显存峰值 (MB)':>16}")
    print("-" * 30)

    B, H, d = 1, 16, 64
    for N in [512, 1024, 2048, 4096]:
        Q = torch.randn(B, H, N, d, dtype=torch.float16, device=device)
        K = torch.randn(B, H, N, d, dtype=torch.float16, device=device)
        V = torch.randn(B, H, N, d, dtype=torch.float16, device=device)

        torch.cuda.reset_peak_memory_stats()
        _ = naive_attention(Q, K, V)
        torch.cuda.synchronize()

        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        print(f"{N:>10,} {peak_mb:>16.1f}")


if __name__ == "__main__":
    memory_analysis()
    verify_correctness()
    benchmark()
