"""
Triton 进阶：Row-wise Softmax Kernel

实现了两个版本：
  1. softmax_kernel        —— 单 block 版，要求整行能装进 BLOCK_SIZE
  2. online_softmax_kernel —— 分块版，支持任意长度的行（FlashAttention 的直接前身）

关键 Triton 特性：
  - tl.max / tl.sum：跨 block 的归约操作
  - tl.maximum：两个标量取最大值
  - for 循环：在 kernel 内部迭代多个 tile
"""

import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────
# Kernel：每个 program 处理矩阵的一行
# ─────────────────────────────────────────────

@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,   # 相邻两行之间的步长（元素数）
    output_row_stride,
    N_COLS,             # 列数（= 序列长度）
    BLOCK_SIZE: tl.constexpr,
):
    # 每个 program 负责一行
    row_idx = tl.program_id(axis=0)

    # 计算当前行的起始指针
    row_start_ptr = input_ptr + row_idx * input_row_stride

    # 列下标
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N_COLS

    # ── Pass 1：加载数据，求行最大值 ──
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=float('-inf'))
    row_max = tl.max(row, axis=0)   # 标量，当前行的最大值

    # ── Pass 2：减去 max，求 exp，求 sum ──
    row_minus_max = row - row_max
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)  # 标量

    # ── 归一化并写回 ──
    softmax_output = numerator / denominator
    out_row_start_ptr = output_ptr + row_idx * output_row_stride
    tl.store(out_row_start_ptr + col_offsets, softmax_output, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    """
    对 2D 矩阵 x 的每一行做 softmax

    注意：这个实现要求 N_COLS <= BLOCK_SIZE（一行能装进一个 block）
    FlashAttention 中用 tiling 解决超长序列的问题
    """
    assert x.ndim == 2 and x.is_cuda
    N_ROWS, N_COLS = x.shape

    # BLOCK_SIZE 必须是 2 的幂次，且 >= N_COLS
    BLOCK_SIZE = triton.next_power_of_2(N_COLS)

    out = torch.empty_like(x)
    grid = (N_ROWS,)   # 一个 program 处理一行

    softmax_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        N_COLS,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


# ─────────────────────────────────────────────
# Online Softmax Kernel（分块版，支持任意行长度）
# ─────────────────────────────────────────────
#
# 与上面版本的区别：
#   softmax_kernel       → 一次性加载整行，BLOCK_SIZE 必须 >= N_COLS
#   online_softmax_kernel → 分块迭代，每次只加载 BLOCK_SIZE 个元素
#                           维护运行时统计量 (m, d)，最后再做一遍归一化
#
# 两次 HBM 读取：Pass 1 求 (m, d)，Pass 2 归一化写回
# 这正是 FlashAttention 把 softmax 融合进 attention 计算的出发点

@triton.jit
def online_softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    N_COLS,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(axis=0)
    row_start_ptr = input_ptr + row_idx * input_row_stride

    # ── Pass 1：分块扫描，维护 online (m, d) ──
    m = float('-inf')   # 运行时最大值
    d = 0.0             # 运行时 sum(exp(x - m))

    for block_start in range(0, N_COLS, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N_COLS
        x = tl.load(row_start_ptr + col_offsets, mask=mask, other=float('-inf'))

        m_block = tl.max(x, axis=0)            # 当前块的局部最大值
        m_new   = tl.maximum(m, m_block)       # 更新全局最大值

        # 修正旧的 d（基于旧 m），再加入当前块的贡献
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m = m_new

    # ── Pass 2：再次分块读取，用最终 (m, d) 归一化写回 ──
    out_row_start_ptr = output_ptr + row_idx * output_row_stride
    for block_start in range(0, N_COLS, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N_COLS
        x = tl.load(row_start_ptr + col_offsets, mask=mask, other=0.0)
        out = tl.exp(x - m) / d
        tl.store(out_row_start_ptr + col_offsets, out, mask=mask)


def online_softmax(x: torch.Tensor, block_size: int = 256) -> torch.Tensor:
    """
    Online softmax：支持任意行长度，不受 BLOCK_SIZE 限制
    block_size 控制每次从 HBM 加载多少列，应为 2 的幂次
    """
    assert x.ndim == 2 and x.is_cuda
    N_ROWS, N_COLS = x.shape
    out = torch.empty_like(x)
    grid = (N_ROWS,)

    online_softmax_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        N_COLS,
        BLOCK_SIZE=block_size,
    )
    return out

def verify():
    torch.manual_seed(0)
    x = torch.randn(1823, 781, device="cuda")   # 故意用非 2 的幂次形状
    ref = torch.softmax(x, dim=1)

    out_softmax = softmax(x)
    # online_softmax 用小 block_size（64）测试分块逻辑
    out_online  = online_softmax(x, block_size=64)

    print(f"softmax        最大误差: {(out_softmax - ref).abs().max():.2e}")
    print(f"online_softmax 最大误差: {(out_online  - ref).abs().max():.2e}")
    assert torch.allclose(out_softmax, ref, atol=1e-5), "softmax 结果不一致！"
    assert torch.allclose(out_online,  ref, atol=1e-5), "online_softmax 结果不一致！"
    print("✓ 两个版本正确性验证通过")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[128, 256, 512, 1024, 2048, 4096],
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "PyTorch"],
        ylabel="GB/s",
        plot_name="softmax-bandwidth",
        args={"M": 4096},
    )
)
def benchmark(M, N, provider):
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: softmax(x), quantiles=quantiles)
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.softmax(x, dim=1), quantiles=quantiles)

    gbps = lambda ms: 2 * x.numel() * x.element_size() / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    verify()
    try:
        benchmark.run(print_data=True)
    except ModuleNotFoundError:
        print("（跳过 benchmark：缺少 matplotlib，可用 pip install matplotlib 安装）")
