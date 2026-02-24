"""
阶段四：FlashAttention v1 —— Triton 实现

论文：FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
      Dao et al., 2022  https://arxiv.org/abs/2205.14135

核心思想：
  标准 Attention 的瓶颈不是计算量，而是 HBM 读写次数（IO-bound）。
  N×N 的 attention score 矩阵反复在 HBM 和 SRAM 之间搬运，带宽成为瓶颈。

FlashAttention 的解法：
  1. 分块（Tiling）：Q、K、V 都切成小块，每次只把一小块载入 SRAM
  2. 融合（Fusion）：QK^T、softmax、@V 三步在 SRAM 内一次性完成，不写回 HBM
  3. Online Softmax：分块计算时用 (m, d) 统计量保证 softmax 数值正确

IO 复杂度对比：
  标准 Attention：O(N²)  次 HBM 读写
  FlashAttention：O(N² / M) 次 HBM 读写，M 为 SRAM 大小

本文件实现：
  - flash_attention_v1_kernel：前向 kernel（不含 causal mask）
  - flash_attention_v1：Python 封装
  - verify：与 PyTorch 标准实现对比正确性
  - benchmark：与 PyTorch 对比吞吐量和显存占用

算法流程（对应论文 Algorithm 1）：
  外层循环（沿 N 方向）：遍历 K、V 的块  →  对应 grid 的 j 维
  内层循环（沿 N 方向）：遍历 Q 的块    →  对应 kernel 内的 for 循环

  对每个 Q 块 q_i：
    初始化 o_i = 0，m_i = -inf，d_i = 0
    对每个 K/V 块 (k_j, v_j)：
      s_ij = q_i @ k_j^T / sqrt(d)          # [Br, Bc]，在 SRAM 中
      m_ij = rowmax(s_ij)
      p_ij = exp(s_ij - m_ij)               # 未归一化的 attention weight
      d_ij = rowsum(p_ij)
      # 合并旧统计量与新块的统计量
      m_new = max(m_i, m_ij)
      d_new = d_i * exp(m_i - m_new) + d_ij * exp(m_ij - m_new)
      # 更新输出（rescale 旧的 o_i，加上新块的贡献）
      o_i = o_i * (d_i * exp(m_i - m_new)) / d_new + p_ij * exp(m_ij - m_new) / d_new @ v_j
      m_i, d_i = m_new, d_new
    写回 o_i 到 HBM
"""

import torch
import triton
import triton.language as tl
from triton.testing import do_bench


# ─────────────────────────────────────────────
# FlashAttention v1 前向 Kernel
# ─────────────────────────────────────────────

@triton.jit
def flash_attention_v1_kernel(
    # 输入指针
    q_ptr, k_ptr, v_ptr,
    # 输出指针
    o_ptr,
    # 形状参数
    N, d,
    # 步长（stride）：用于在多维 tensor 中定位元素
    stride_qn, stride_qd,
    stride_kn, stride_kd,
    stride_vn, stride_vd,
    stride_on, stride_od,
    # scale = 1 / sqrt(d)，在 Python 侧计算好传入，避免 kernel 内做幂运算
    scale,
    # 编译期常量
    BLOCK_N: tl.constexpr,   # Q 方向的块大小 Br
    BLOCK_M: tl.constexpr,   # K/V 方向的块大小 Bc
    HEAD_DIM: tl.constexpr,  # head dimension d，必须是 2 的幂
):
    """
    每个 program 负责计算输出 O 的一个行块 [BLOCK_N, d]。

    Grid：(triton.cdiv(N, BLOCK_N),)  —— 每个 program 对应一个 Q 块

    内存访问模式：
      - Q 块：从 HBM 加载一次，在 SRAM 中复用
      - K/V 块：每次循环从 HBM 加载，用完即丢
      - O 块：最终写回 HBM 一次
      → 完整的 N×N score 矩阵从未出现在 HBM 中！
    """
    # 当前 program 负责第几个 Q 块
    pid = tl.program_id(axis=0)

    # 当前 Q 块的行偏移：[pid*BLOCK_N, (pid+1)*BLOCK_N)
    q_row_start = pid * BLOCK_N
    offs_n = q_row_start + tl.arange(0, BLOCK_N)   # Q 行下标
    offs_d = tl.arange(0, HEAD_DIM)                 # head dim 下标

    # ── 加载 Q 块到 SRAM ──────────────────────────────────────────
    # q: [BLOCK_N, HEAD_DIM]
    q_mask = (offs_n[:, None] < N) & (offs_d[None, :] < d)
    q = tl.load(
        q_ptr + offs_n[:, None] * stride_qn + offs_d[None, :] * stride_qd,
        mask=q_mask, other=0.0,
    )

    # ── 初始化 online softmax 统计量 ──────────────────────────────
    # m_i: 当前见过的 rowmax，shape [BLOCK_N]
    # d_i: 当前归一化分母（sum of exp），shape [BLOCK_N]
    # o_i: 累积输出，shape [BLOCK_N, HEAD_DIM]
    m_i = tl.full((BLOCK_N,), float("-inf"), dtype=tl.float32)
    d_i = tl.zeros((BLOCK_N,), dtype=tl.float32)
    o_i = tl.zeros((BLOCK_N, HEAD_DIM), dtype=tl.float32)

    # ── 遍历所有 K/V 块 ───────────────────────────────────────────
    for j in range(0, tl.cdiv(N, BLOCK_M)):
        kv_col_start = j * BLOCK_M
        offs_m = kv_col_start + tl.arange(0, BLOCK_M)   # K/V 列下标

        # 加载 K 块：[BLOCK_M, HEAD_DIM]
        k_mask = (offs_m[:, None] < N) & (offs_d[None, :] < d)
        k = tl.load(
            k_ptr + offs_m[:, None] * stride_kn + offs_d[None, :] * stride_kd,
            mask=k_mask, other=0.0,
        )

        # 加载 V 块：[BLOCK_M, HEAD_DIM]
        v = tl.load(
            v_ptr + offs_m[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            mask=k_mask, other=0.0,
        )

        # ── 计算 attention score：s = q @ k^T / sqrt(d) ──────────
        # s: [BLOCK_N, BLOCK_M]，完全在 SRAM 中，不写回 HBM
        s = tl.dot(q, tl.trans(k)) * scale

        # 对超出序列长度的位置 mask 掉（设为 -inf）
        valid_mask = offs_m[None, :] < N   # [1, BLOCK_M]
        s = tl.where(valid_mask, s, float("-inf"))

        # ── Online Softmax 更新 ───────────────────────────────────
        # 当前块的 rowmax：m_ij，shape [BLOCK_N]
        m_ij = tl.max(s, axis=1)

        # 当前块的未归一化 exp：p_ij = exp(s - m_ij)
        p_ij = tl.exp(s - m_ij[:, None])

        # 当前块的 rowsum：d_ij，shape [BLOCK_N]
        d_ij = tl.sum(p_ij, axis=1)

        # 合并旧统计量（m_i, d_i）与新块统计量（m_ij, d_ij）
        #   m_new = max(m_i, m_ij)
        #   d_new = d_i * exp(m_i - m_new) + d_ij * exp(m_ij - m_new)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i  - m_new)   # 旧统计量的缩放因子
        beta  = tl.exp(m_ij - m_new)   # 新块统计量的缩放因子
        d_new = d_i * alpha + d_ij * beta

        # 更新输出 o_i：
        #   o_new = (o_i * d_i * alpha + p_ij * beta @ v_j) / d_new
        # 等价写法（先不除 d_new，最后统一除）：
        #   o_new = o_i * (d_i * alpha / d_new) + (p_ij * beta / d_new) @ v_j
        #
        # 这里用"rescale 旧 o_i"的方式，避免除法精度问题：
        #   rescale = d_i * alpha / d_new
        rescale = (d_i * alpha) / d_new   # [BLOCK_N]

        # p_ij 归一化后乘以 v_j：[BLOCK_N, HEAD_DIM]
        p_norm = (p_ij * beta[:, None]) / d_new[:, None]

        o_i = o_i * rescale[:, None] + tl.dot(p_norm.to(tl.float16), v)

        # 更新统计量
        m_i = m_new
        d_i = d_new

    # ── 写回输出到 HBM ────────────────────────────────────────────
    o_mask = (offs_n[:, None] < N) & (offs_d[None, :] < d)
    tl.store(
        o_ptr + offs_n[:, None] * stride_on + offs_d[None, :] * stride_od,
        o_i.to(tl.float16),
        mask=o_mask,
    )


# ─────────────────────────────────────────────
# Python 封装：处理 batch 和 heads 维度
# ─────────────────────────────────────────────

def flash_attention_v1(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    BLOCK_N: int = 64,
    BLOCK_M: int = 64,
) -> torch.Tensor:
    """
    FlashAttention v1 前向传播。

    Args:
        Q: (batch, heads, N, d)
        K: (batch, heads, N, d)
        V: (batch, heads, N, d)
    Returns:
        O: (batch, heads, N, d)

    注意：
      - d 必须是 2 的幂（Triton tl.dot 的要求）
      - 当前实现不含 causal mask（非自回归场景）
    """
    assert Q.is_cuda, "需要 CUDA tensor"
    assert Q.shape == K.shape == V.shape
    B, H, N, d = Q.shape
    assert d in (16, 32, 64, 128), f"HEAD_DIM={d} 须为 16/32/64/128"

    O = torch.empty_like(Q)

    # 对每个 (batch, head) 独立调用 kernel
    for b in range(B):
        for h in range(H):
            q = Q[b, h].contiguous()   # (N, d)
            k = K[b, h].contiguous()
            v = V[b, h].contiguous()
            o = O[b, h]

            grid = (triton.cdiv(N, BLOCK_N),)

            flash_attention_v1_kernel[grid](
                q, k, v, o,
                N, d,
                q.stride(0), q.stride(1),
                k.stride(0), k.stride(1),
                v.stride(0), v.stride(1),
                o.stride(0), o.stride(1),
                scale=d ** -0.5,
                BLOCK_N=BLOCK_N,
                BLOCK_M=BLOCK_M,
                HEAD_DIM=d,
            )

    return O


# ─────────────────────────────────────────────
# 正确性验证
# ─────────────────────────────────────────────

def verify():
    """对比 FlashAttention 与 PyTorch 标准实现的输出误差"""
    import torch.nn.functional as F

    torch.manual_seed(42)
    B, H, N, d = 2, 4, 256, 64

    Q = torch.randn(B, H, N, d, device="cuda", dtype=torch.float16)
    K = torch.randn(B, H, N, d, device="cuda", dtype=torch.float16)
    V = torch.randn(B, H, N, d, device="cuda", dtype=torch.float16)

    # FlashAttention 输出
    out_flash = flash_attention_v1(Q, K, V)

    # PyTorch 标准实现（参考值）
    out_ref = F.scaled_dot_product_attention(Q, K, V)

    max_diff = (out_flash.float() - out_ref.float()).abs().max().item()
    mean_diff = (out_flash.float() - out_ref.float()).abs().mean().item()

    print(f"最大误差:  {max_diff:.4e}")
    print(f"平均误差:  {mean_diff:.4e}")

    # float16 精度下，误差在 1e-2 量级是正常的
    assert max_diff < 0.1, f"误差过大: {max_diff}"
    print("✓ 正确性验证通过\n")


# ─────────────────────────────────────────────
# 性能 & 显存对比
# ─────────────────────────────────────────────

def benchmark():
    """对比 FlashAttention 与标准 Attention 的吞吐量和显存占用"""
    import torch.nn.functional as F

    print(f"{'N':>8} {'标准Attn(ms)':>14} {'Flash(ms)':>12} {'加速比':>8} "
          f"{'标准显存(MB)':>14} {'Flash显存(MB)':>14}")
    print("-" * 80)

    B, H, d = 1, 8, 64

    for N in [256, 512, 1024, 2048, 4096]:
        Q = torch.randn(B, H, N, d, device="cuda", dtype=torch.float16)
        K = torch.randn(B, H, N, d, device="cuda", dtype=torch.float16)
        V = torch.randn(B, H, N, d, device="cuda", dtype=torch.float16)

        # 标准 Attention 耗时
        ms_ref = do_bench(lambda: F.scaled_dot_product_attention(Q, K, V))

        # FlashAttention 耗时
        ms_flash = do_bench(lambda: flash_attention_v1(Q, K, V))

        # 显存：标准 Attention（含 N×N score 矩阵）
        torch.cuda.reset_peak_memory_stats()
        _ = F.scaled_dot_product_attention(Q, K, V)
        torch.cuda.synchronize()
        mem_ref = torch.cuda.max_memory_allocated() / 1e6

        # 显存：FlashAttention
        torch.cuda.reset_peak_memory_stats()
        _ = flash_attention_v1(Q, K, V)
        torch.cuda.synchronize()
        mem_flash = torch.cuda.max_memory_allocated() / 1e6

        speedup = ms_ref / ms_flash
        print(f"{N:>8,} {ms_ref:>14.3f} {ms_flash:>12.3f} {speedup:>8.2f}x "
              f"{mem_ref:>14.1f} {mem_flash:>14.1f}")


if __name__ == "__main__":
    verify()
    benchmark()
