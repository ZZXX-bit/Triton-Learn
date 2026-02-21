"""
Triton 入门：Vector Add

Triton 编程模型核心概念：
  - GPU 把工作分成若干 "program"（类似 CUDA 的 block）
  - 每个 program 用 tl.program_id(axis) 知道自己是第几个
  - 每个 program 处理 BLOCK_SIZE 个元素
  - tl.load / tl.store 负责从 HBM 读写数据，mask 处理边界

对比 CUDA：
  Triton program  ≈  CUDA thread block
  tl.program_id() ≈  blockIdx.x
  BLOCK_SIZE      ≈  blockDim.x（但 Triton 内部自动向量化）
"""

import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────
# Kernel 定义
# ─────────────────────────────────────────────

@triton.jit
def vector_add_kernel(
    x_ptr,          # 输入向量 x 的指针
    y_ptr,          # 输入向量 y 的指针
    out_ptr,        # 输出向量的指针
    N,              # 向量长度
    BLOCK_SIZE: tl.constexpr,   # 每个 program 处理的元素数，编译期常量
):
    # 1. 确定当前 program 负责哪段数据
    pid = tl.program_id(axis=0)                        # 我是第几个 program？
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # 我负责的元素下标

    # 2. 边界 mask：防止越界访问（当 N 不是 BLOCK_SIZE 整数倍时）
    mask = offsets < N

    # 3. 从 HBM 加载数据到寄存器
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    # 4. 计算（在寄存器中完成，不经过 HBM）
    out = x + y

    # 5. 写回 HBM
    tl.store(out_ptr + offsets, out, mask=mask)


# ─────────────────────────────────────────────
# Python 封装：负责分配内存、计算 grid、调用 kernel
# ─────────────────────────────────────────────

def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.shape == y.shape and x.is_cuda
    N = x.numel()
    out = torch.empty_like(x)

    BLOCK_SIZE = 1024
    # grid：需要多少个 program？向上取整
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    vector_add_kernel[grid](x, y, out, N, BLOCK_SIZE=BLOCK_SIZE)
    return out


# ─────────────────────────────────────────────
# 验证 & Benchmark
# ─────────────────────────────────────────────

def verify():
    torch.manual_seed(0)
    x = torch.randn(1 << 20, device="cuda", dtype=torch.float32)  # 1M 元素
    y = torch.randn_like(x)

    out_triton = vector_add(x, y)
    out_torch  = x + y

    print(f"最大误差: {(out_triton - out_torch).abs().max():.2e}")
    assert torch.allclose(out_triton, out_torch), "结果不一致！"
    print("✓ 正确性验证通过")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[2**i for i in range(12, 28, 2)],
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "PyTorch"],
        ylabel="GB/s",
        plot_name="vector-add-bandwidth",
        args={},
    )
)
def benchmark(N, provider):
    x = torch.randn(N, device="cuda", dtype=torch.float32)
    y = torch.randn_like(x)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: vector_add(x, y), quantiles=quantiles)
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)

    # 带宽 = 读 2 个向量 + 写 1 个向量，共 3N 个 float32
    gbps = lambda ms: 3 * N * x.element_size() / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    verify()
    try:
        benchmark.run(print_data=True)
    except ModuleNotFoundError:
        print("（跳过 benchmark：缺少 matplotlib，可用 pip install matplotlib 安装）")
