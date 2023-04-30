from torch_module import BaseTransformerGatedLinearLayer
from triton_module import OptimizedTransformerGatedLinearLayer
import torch 
import triton
from contextlib import ExitStack
from torch import nn 
from typing import Tuple
import gc 
import itertools

IMPLEMENTATIONS = ['torch', 'triton']
QUANTILES_TO_REPORT = [0.5, 0.2, 0.8]
BYTES_IN_MB = 1024 * 1024

def get_module(input_dim: int, implementation: str, projection_factor: int = 8, dtype: torch.dtype=torch.half) -> nn.Module:
    if implementation == 'torch':
        module = BaseTransformerGatedLinearLayer(input_dim, projection_factor=projection_factor, dtype=dtype) 
    elif implementation == 'triton':
        # TODO: fixing this
        module = BaseTransformerGatedLinearLayer(input_dim, projection_factor=projection_factor, dtype=dtype) 
        module = OptimizedTransformerGatedLinearLayer.from_torch(module)
    else:
        raise ValueError(f"Unknown implementation {implementation}")

    return module

# TODO: adapt this benchmarking script
def benchmark_backward(batch_size: int, input_dim: int, implementation:str, projection_factor: int = 8, dtype: torch.dtype=torch.half) -> Tuple[Tuple[int, ...], int]:
    torch.cuda.reset_peak_memory_stats(device='cuda')
    x = torch.randn(batch_size, input_dim, dtype=dtype, device='cuda')
        
    module = get_module(input_dim, implementation, projection_factor, dtype)
    module.to('cuda')
        
    memory_baseline = torch.cuda.max_memory_allocated(device='cuda')    
    loss = torch.sum(torch.abs(module(x)))
    results = triton.testing.do_bench(lambda: loss.backward(retain_graph=True), percentiles=QUANTILES_TO_REPORT, grad_to_none=[x])
    peak_memory_backward = torch.cuda.max_memory_allocated(device='cuda')
    
    return results, peak_memory_backward - memory_baseline

def is_cuda_oom(err: RuntimeError) -> bool:
    return 'out of memory' in str(err)

def benchmark_forward_runtime(batch_size: int, input_dim: int, implementation: str, projection_factor: int = 8, dtype: torch.dtype=torch.half, no_grad: bool = False, warmup: int = 25, rep: int = 100):    
    print("Run", implementation, batch_size, input_dim, no_grad)
    try:
        x = torch.randn(batch_size, input_dim, dtype=dtype, device='cuda')
        module = get_module(input_dim, implementation, projection_factor, dtype)
        module.to('cuda')
        
        with ExitStack() as stack:
            if no_grad:
                stack.enter_context(torch.no_grad())
            results= triton.testing.do_bench(lambda: module(x), percentiles=QUANTILES_TO_REPORT, warmup=warmup, rep=rep)
        
        return results
    except RuntimeError as e:
        if is_cuda_oom(e):
            return 0, 0, 0
        raise e

def benchmark_forward_memory_usage(batch_size: int, input_dim: int, implementation: str, projection_factor: int = 8, dtype: torch.dtype=torch.half, no_grad: bool = False):
    print("Mem", implementation, batch_size, input_dim, no_grad)
    try:
        x = torch.randn(batch_size, input_dim, dtype=dtype, device='cuda')
        module = get_module(input_dim, implementation, projection_factor, dtype)
        module.to('cuda')
        
        with ExitStack() as stack:
            if no_grad:
                stack.enter_context(torch.no_grad())
                
            # Warmup, used for populating autotuning cache
            module(x)
            
            # Force collect objects from warmup just in case
            gc.collect()
            
            # Note this may not pick up memory not allocated by torch.
            # Assume the triton kernel does not malloc cuda global memory
            baseline_memory_allocated = torch.cuda.memory_allocated()
            torch.cuda.reset_max_memory_allocated()
            _ = module(x)
            max_memory_allocated = torch.cuda.max_memory_allocated()
            return (max_memory_allocated - baseline_memory_allocated) / BYTES_IN_MB
    except RuntimeError as e:
        if is_cuda_oom(e):
            return 0
        raise e

if __name__ == "__main__":
    """Benchmark implementations.
    
    Note before going run the following commands to fix clock speeds (suggest normal baselines):
        - TODO
    """
    
    # Some popular hidden dimension sizes (d):
    # vicuna 13B - 5120
    # llama 7B - 4096
    # llama 64B - 8192
    input_dim_vals = [512 * i for i in range(1, 20)] # [512, 1024, ... 9728]
    
    # Only large batch sizes are interesting, especially when N >> d
    batch_sizes = [1024, 1024 * 10, 1024 * 40, 1024 * 80]
    benchmark_reports_runtime = [
        triton.testing.Benchmark(
            x_names=['input_dim'],  # Argument names to use as an x-axis for the plot
            x_vals=input_dim_vals,  # Different possible values for `x_name`
            line_arg='implementation',  # Argument name whose value corresponds to a different line in the plot
            # Possible values for `line_arg`
            line_vals=IMPLEMENTATIONS,
            # Label name for the lines
            line_names=IMPLEMENTATIONS,
            # Line styles
            styles=[('green', '-'), ('blue', '-')],
            ylabel="Runtime (ms)",  # Label name for the y-axis
            plot_name=f"runtime-gated-linear-attention-forward-bs={batch_size}",  # Name for the plot, used also as a file name for saving the plot.
            args={'batch_size': batch_size},
        ) for batch_size in batch_sizes
    ]
    benchmark_reports_memory = [
        triton.testing.Benchmark(
            x_names=['input_dim'],  # Argument names to use as an x-axis for the plot
            x_vals=input_dim_vals,  # Different possible values for `x_name`
            line_arg='implementation',  # Argument name whose value corresponds to a different line in the plot
            # Possible values for `line_arg`
            line_vals=IMPLEMENTATIONS,
            # Label name for the lines
            line_names=IMPLEMENTATIONS,
            # Line styles
            styles=[('green', '-'), ('blue', '-')],
            ylabel="VRAM (MB) From Call",  # Label name for the y-axis
            plot_name=f"memory-gated-linear-attention-forward-bs={batch_size}",  # Name for the plot, used also as a file name for saving the plot.
            args={'batch_size': batch_size},
        ) for batch_size in batch_sizes
    ]
    
    benchmark_runtime = triton.testing.perf_report(benchmark_reports_runtime)(benchmark_forward_runtime)
    benchmark_memory = triton.testing.perf_report(benchmark_reports_memory)(benchmark_forward_memory_usage)
    
    # TODO: needs sudo
    # with triton.testing.set_gpu_clock():
    # run manually: sudo nvidia-smi -i 0 --lock-gpu-clocks=1350,1350
    # Note this also warmups the autotuning cache for runtime
    # benchmark_memory.run(show_plots=True, print_data=True, save_path='outputs/')
    benchmark_runtime.run(show_plots=True, print_data=True, save_path='outputs/')
