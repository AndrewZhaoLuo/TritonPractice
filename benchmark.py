import gc
from contextlib import ExitStack
from typing import Callable

import torch
from torch import nn

import triton
from torch_module import BaseTransformerGatedLinearLayer
from triton_module import OptimizedTransformerGatedLinearLayer

IMPLEMENTATIONS = ["torch", "triton"]
QUANTILES_TO_REPORT = [0.5, 0.2, 0.8]
BYTES_IN_MB = 1024 * 1024


def get_module(
    input_dim: int,
    implementation: str,
    projection_factor: int = 8,
    dtype: torch.dtype = torch.half,
) -> nn.Module:
    if implementation == "torch":
        module = BaseTransformerGatedLinearLayer(input_dim, projection_factor=projection_factor, dtype=dtype)
    elif implementation == "triton":
        # TODO: fixing this
        module = BaseTransformerGatedLinearLayer(input_dim, projection_factor=projection_factor, dtype=dtype)
        module = OptimizedTransformerGatedLinearLayer.from_torch(module)
    else:
        raise ValueError(f"Unknown implementation {implementation}")

    return module


def is_cuda_oom_err(err: RuntimeError) -> bool:
    return "out of memory" in str(err)


def _benchmark_runtime(func: Callable, no_grad: bool = False, warmup: int = 25, rep: int = 100):
    try:
        with ExitStack() as stack:
            if no_grad:
                stack.enter_context(torch.no_grad())
            return triton.testing.do_bench(func, percentiles=QUANTILES_TO_REPORT, warmup=warmup, rep=rep)
    except RuntimeError as e:
        if is_cuda_oom_err(e):
            return None
        raise e


def benchmark_forward_runtime(
    batch_size: int,
    input_dim: int,
    implementation: str,
    projection_factor: int = 8,
    dtype: torch.dtype = torch.half,
    warmup: int = 25,
    rep: int = 100,
):
    try:
        x = torch.randn(batch_size, input_dim, dtype=dtype, device="cuda")
        module = get_module(input_dim, implementation, projection_factor, dtype)
        module.to("cuda")

        def func():
            module(x)

        return _benchmark_runtime(func, no_grad=False, warmup=warmup, rep=rep)
    except RuntimeError as e:
        if is_cuda_oom_err(e):
            return None
        raise e


def benchmark_backward_runtime(
    batch_size: int,
    input_dim: int,
    implementation: str,
    projection_factor: int = 8,
    dtype: torch.dtype = torch.half,
    warmup: int = 25,
    rep: int = 100,
):
    try:
        x = torch.randn(batch_size, input_dim, dtype=dtype, device="cuda")
        module = get_module(input_dim, implementation, projection_factor, dtype)
        module.to("cuda")
        result = module(x).sum()

        def func():
            result.backward(retain_graph=True)

        return _benchmark_runtime(func, no_grad=False, warmup=warmup, rep=rep)
    except RuntimeError as e:
        if is_cuda_oom_err(e):
            return None
        raise e


def _benchmark_memory_usage(func: Callable, no_grad: bool = False):
    with ExitStack() as stack:
        if no_grad:
            stack.enter_context(torch.no_grad())

        # Warmup, used for populating autotuning cache
        func()

        # Force collect objects from warmup just in case
        gc.collect()

        # Note this may not pick up memory not allocated by torch.
        # Assume the triton kernel does not malloc cuda global memory
        baseline_memory_allocated = torch.cuda.memory_allocated()
        torch.cuda.reset_max_memory_allocated()
        _ = func()
        max_memory_allocated = torch.cuda.max_memory_allocated()
        return (max_memory_allocated - baseline_memory_allocated) / BYTES_IN_MB


def benchmark_forward_memory_usage(
    batch_size: int,
    input_dim: int,
    implementation: str,
    projection_factor: int = 8,
    dtype: torch.dtype = torch.half,
):
    try:
        x = torch.randn(batch_size, input_dim, dtype=dtype, device="cuda")
        module = get_module(input_dim, implementation, projection_factor, dtype)
        module.to("cuda")

        def func():
            return module(x)

        return _benchmark_memory_usage(func, no_grad=False)
    except RuntimeError as e:
        if is_cuda_oom_err(e):
            return None
        raise e


def benchmark_backward_memory_usage(
    batch_size: int,
    input_dim: int,
    implementation: str,
    projection_factor: int = 8,
    dtype: torch.dtype = torch.half,
):
    try:
        x = torch.randn(batch_size, input_dim, dtype=dtype, device="cuda")
        module = get_module(input_dim, implementation, projection_factor, dtype)
        module.to("cuda")
        result = module(x).sum()

        def func():
            result.backward(retain_graph=True)

        return _benchmark_memory_usage(func, no_grad=False)
    except RuntimeError as e:
        if is_cuda_oom_err(e):
            return None
        raise e


if __name__ == "__main__":
    """
    Good benchmarking juju:
        1. Set GPU settings to persist: `nvidia-smi -i 0 -pm ENABLED`
        2. Get supported clocks: `nvidia-smi -q -d SUPPORTED_CLOCKS`
        3. Set clock to max memory clock, 70% of gpu clock: sudo nvidia-smi -ac <memory_clock>,<gpu_clock>
            a. on my system -ac is not supported
            b. alternative run `sudo nvidia-smi --lock-gpu-clocks=1515` and `sudo nvidia-smi --lock-memory-clocks=7001`

    I did: sudo nvidia-smi -ac 7001,1515
    """
    # Sanity checks:
    # print(benchmark_backward_memory_usage(10240, 4096, "triton"))
    # print(benchmark_backward_memory_usage(10240, 4096, "torch"))
    # print(benchmark_forward_memory_usage(10240, 4096, "triton"))
    # print(benchmark_forward_memory_usage(10240, 4096, "torch"))
    # print(benchmark_backward_runtime(10240, 4096, "triton"))
    # print(benchmark_backward_runtime(10240, 4096, "torch"))
    # print(benchmark_forward_runtime(10240, 4096, "triton"))
    # print(benchmark_forward_runtime(10240, 4096, "torch"))
    # breakpoint()

    # Some popular hidden dimension sizes (d):
    # vicuna 13B - 5120
    # llama 7B - 4096
    # llama 64B - 8192
    input_dim_vals = [512 * i for i in range(1, 20)]  # [512, 1024, ... 9728]

    # Only large batch sizes are interesting, especially when N >> d
    batch_sizes = [1024, 1024 * 10, 1024 * 40, 1024 * 80]
    benchmark_reports_runtime_forward = [
        triton.testing.Benchmark(
            x_names=["input_dim"],  # Argument names to use as an x-axis for the plot
            x_vals=input_dim_vals,  # Different possible values for `x_name`
            line_arg="implementation",  # Argument name whose value corresponds to a different line in the plot
            # Possible values for `line_arg`
            line_vals=IMPLEMENTATIONS,
            # Label name for the lines
            line_names=IMPLEMENTATIONS,
            # Line styles
            styles=[("green", "-"), ("blue", "-")],
            ylabel="Runtime (ms)",  # Label name for the y-axis
            plot_name=f"runtime-gated-linear-attention-forward-bs={batch_size}",  # Name for the plot, used also as a file name for saving the plot.
            args={"batch_size": batch_size},
        )
        for batch_size in batch_sizes
    ]
    benchmark_reports_memory = [
        triton.testing.Benchmark(
            x_names=["input_dim"],  # Argument names to use as an x-axis for the plot
            x_vals=input_dim_vals,  # Different possible values for `x_name`
            line_arg="implementation",  # Argument name whose value corresponds to a different line in the plot
            # Possible values for `line_arg`
            line_vals=IMPLEMENTATIONS,
            # Label name for the lines
            line_names=IMPLEMENTATIONS,
            # Line styles
            styles=[("green", "-"), ("blue", "-")],
            ylabel="VRAM (MB) From Call",  # Label name for the y-axis
            plot_name=f"memory-gated-linear-attention-forward-bs={batch_size}",  # Name for the plot, used also as a file name for saving the plot.
            args={"batch_size": batch_size},
        )
        for batch_size in batch_sizes
    ]

    benchmark_runtime = triton.testing.perf_report(benchmark_reports_runtime_forward)(benchmark_forward_runtime)
    benchmark_memory = triton.testing.perf_report(benchmark_reports_memory)(benchmark_forward_memory_usage)

    # TODO: needs sudo
    # with triton.testing.set_gpu_clock():
    benchmark_memory.run(show_plots=True, print_data=True, save_path="outputs/")
    benchmark_runtime.run(show_plots=True, print_data=True, save_path="outputs/")
