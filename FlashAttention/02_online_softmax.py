"""
阶段一：Online Softmax —— FlashAttention 的数学核心

问题：标准 softmax 需要两次遍历数据（先求 max，再求 exp/sum）
      FlashAttention 要分块计算，每次只看一小块，怎么保证 softmax 正确？

答案：Online Softmax —— 维护"运行时统计量"，一边读数据一边更新结果

本文件演示从朴素 softmax 到 online softmax 的推导过程。
"""

import torch
import math


# ─────────────────────────────────────────────
# 1. 朴素 softmax（数值不稳定）
# ─────────────────────────────────────────────

def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    直接按公式计算：softmax(x_i) = exp(x_i) / sum(exp(x_j))

    问题：当 x 中有较大值时，exp(x) 会溢出（overflow）
    例如：exp(1000) = inf，导致 nan
    """
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum()


# ─────────────────────────────────────────────
# 2. Safe softmax（数值稳定，需要 2 次遍历）
# ─────────────────────────────────────────────

def safe_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    减去最大值后再计算，数学上等价但数值稳定：
        softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))

    需要 2 次遍历：
        Pass 1: 求 max(x)
        Pass 2: 计算 exp 和 sum

    这是 PyTorch 内置 softmax 的做法。
    """
    m = x.max()                    # Pass 1: 求全局最大值
    exp_x = torch.exp(x - m)      # Pass 2: 减去 max 后求 exp
    return exp_x / exp_x.sum()


# ─────────────────────────────────────────────
# 3. Online Softmax（1 次遍历，流式处理）
# ─────────────────────────────────────────────

def online_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    核心思想：维护两个"运行时统计量"
        m_i = max(x_1, ..., x_i)   当前见过的最大值
        d_i = sum(exp(x_j - m_i))  用当前最大值归一化的 exp 之和

    当新元素 x_{i+1} 到来时，更新规则：
        m_{i+1} = max(m_i, x_{i+1})
        d_{i+1} = d_i * exp(m_i - m_{i+1}) + exp(x_{i+1} - m_{i+1})
                  ↑ 旧的 sum 需要用新 max 重新缩放

    最终：softmax(x_i) = exp(x_i - m_N) / d_N
    """
    N = x.shape[0]
    m = float('-inf')   # 运行时最大值，初始为 -∞
    d = 0.0             # 运行时归一化分母

    for i in range(N):
        x_i = x[i].item()
        m_new = max(m, x_i)
        # 关键：旧的 d 是基于 m 算的，现在 max 变成 m_new，需要乘以 exp(m - m_new) 来修正
        d = d * math.exp(m - m_new) + math.exp(x_i - m_new)
        m = m_new

    # 此时 m = max(x)，d = sum(exp(x_j - m))
    return torch.exp(x - m) / d



# ─────────────────────────────────────────────
# 4. Tiled Online Softmax（分块版，FlashAttention 的直接前身）
# ─────────────────────────────────────────────

def tiled_online_softmax(x: torch.Tensor, block_size: int = 4) -> torch.Tensor:
    """
    将序列分成若干块，每次只处理一块（模拟 GPU SRAM 只能装下一小块数据的场景）

    两个块 [block_0, block_1] 的合并规则：
        m_total = max(m_0, m_1)
        d_total = d_0 * exp(m_0 - m_total) + d_1 * exp(m_1 - m_total)

    这个合并规则可以推广到任意多块，且顺序无关（结合律）。
    这正是 FlashAttention 分块计算 attention 的数学基础。
    """
    N = x.shape[0]
    m = float('-inf')
    d = 0.0

    for start in range(0, N, block_size):
        block = x[start : start + block_size]

        # 处理当前块：求块内 max 和 sum(exp)
        m_block = block.max().item()
        d_block = torch.exp(block - m_block).sum().item()

        # 将当前块的统计量合并到全局统计量
        m_new = max(m, m_block)
        d = d * math.exp(m - m_new) + d_block * math.exp(m_block - m_new)
        m = m_new

    return torch.exp(x - m) / d


# ─────────────────────────────────────────────
# 5. 验证所有实现结果一致
# ─────────────────────────────────────────────

def verify():
    torch.manual_seed(0)
    x = torch.randn(16)

    ref   = safe_softmax(x)
    out1  = naive_softmax(x)
    out2  = online_softmax(x)
    out3  = tiled_online_softmax(x, block_size=4)

    print("各实现最大误差（相对 safe_softmax）：")
    print(f"  naive_softmax:        {(out1 - ref).abs().max():.2e}")
    print(f"  online_softmax:       {(out2 - ref).abs().max():.2e}")
    print(f"  tiled_online_softmax: {(out3 - ref).abs().max():.2e}")

    # 数值溢出演示
    print("\n数值溢出演示（x 中含大值 1000）：")
    x_large = torch.tensor([1000.0, 1001.0, 999.0])
    print(f"  naive_softmax:  {naive_softmax(x_large)}")   # 会出现 nan
    print(f"  safe_softmax:   {safe_softmax(x_large)}")    # 正常


# ─────────────────────────────────────────────
# 6. 图示：online softmax 的更新过程
# ─────────────────────────────────────────────

def visualize_update():
    """逐步打印 online softmax 的状态，帮助理解更新过程"""
    x = torch.tensor([2.0, 5.0, 1.0, 3.0])
    N = x.shape[0]
    m, d = float('-inf'), 0.0

    print("\nOnline Softmax 逐步更新过程：")
    print(f"{'step':>5} {'x_i':>6} {'m (max)':>10} {'d (sum_exp)':>14}")
    print("-" * 40)

    for i in range(N):
        x_i = x[i].item()
        m_new = max(m, x_i)
        d = d * math.exp(m - m_new) + math.exp(x_i - m_new)
        m = m_new
        print(f"{i+1:>5} {x_i:>6.1f} {m:>10.4f} {d:>14.6f}")

    result = torch.exp(x - m) / d
    print(f"\n最终结果: {result}")
    print(f"验证和为: {result.sum():.6f}（应为 1.0）")


if __name__ == "__main__":
    verify()
    visualize_update()
