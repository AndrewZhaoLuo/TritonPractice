from torch_module import BaseTransformerGatedLinearLayer
import torch 
import triton
from contextlib import ExitStack

QUANTILES_TO_REPORT = [0.5, 0.2, 0.8]

def benchmark_backward(batch_size: int, input_dim: int, projection_factor: int = 8, dtype: torch.dtype=torch.half):
    torch.cuda.reset_peak_memory_stats(device='cuda')
    x = torch.randn(batch_size, input_dim, dtype=dtype, device='cuda')
        
    module = BaseTransformerGatedLinearLayer(input_dim, projection_factor=projection_factor, dtype=dtype)
    module.to('cuda')
        
    memory_baseline = torch.cuda.max_memory_allocated(device='cuda')    
    loss = torch.sum(torch.abs(module(x)))
    ms50, ms20, ms80 = triton.testing.do_bench(lambda: loss.backward(retain_graph=True), percentiles=QUANTILES_TO_REPORT, grad_to_none=[x])
    peak_memory_backward = torch.cuda.max_memory_allocated(device='cuda')
    print(f"Backward: {ms50}, {ms20}, {ms80}")
    print(f"Backward Mem: {peak_memory_backward - memory_baseline}")
    

def benchmark_forward(batch_size: int, input_dim: int, projection_factor: int = 8, dtype: torch.dtype=torch.half, no_grad: bool = False):
    torch.cuda.reset_peak_memory_stats(device='cuda')
    
    x = torch.randn(batch_size, input_dim, dtype=dtype, device='cuda')
    module = BaseTransformerGatedLinearLayer(input_dim, projection_factor=projection_factor, dtype=dtype)
    module.to('cuda')
    
    memory_baseline = torch.cuda.max_memory_allocated(device='cuda')
    
    with ExitStack() as stack:
        if no_grad:
            stack.enter_context(torch.no_grad())
        ms50, ms20, ms80 = triton.testing.do_bench(lambda: module(x), percentiles=QUANTILES_TO_REPORT)
    
    peak_memory_forward = torch.cuda.max_memory_allocated(device='cuda')
    print(f"Forward: {ms50}, {ms20}, {ms80}")
    print(f"Forward Mem: {peak_memory_forward - memory_baseline}")
        

if __name__ == "__main__":
    """Benchmark implementations.
    
    Note before going run the following commands to fix clock speeds (suggest normal baselines):
        - TODO
    """
    benchmark_forward(1024, 4096 * 4) 
    benchmark_forward(1024, 4096 * 4, no_grad=True) 
    benchmark_backward(1024, 4096 * 4)