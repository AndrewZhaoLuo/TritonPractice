import torch 
from torch import nn 
import triton_kernel
from torch_module import BaseTransformerGatedLinearLayer
from typing import *
import numpy as np 
import time 
import torch 

class TransformerGatedLinearLayerFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, 
        input_tensor, 
        weight_tensor, 
        bias_tensor
    ):
        ctx.save_for_backward(input_tensor, weight_tensor, bias_tensor)
        return triton_kernel.transformer_gated_linear_forward(
            input_tensor, 
            weight_tensor, 
            bias_tensor
        )
        
    @staticmethod
    def backward(ctx, grad_output) -> Any:
        input_tensor, weight_tensor, bias_tensor = ctx.saved_tensors
        
        # TODO: placeholder
        return input_tensor * 0, weight_tensor * 0, bias_tensor * 0
        
        
class OptimizedTransformerGatedLinearLayer(nn.Module):
    def __init__(self, dimension_in: int, projection_factor: int = 8, dtype: torch.dtype=torch.half) -> None:
        super().__init__()
        
        # TODO: proper initialization
        self.weight = nn.Parameter(torch.empty([dimension_in * projection_factor, dimension_in], dtype=dtype), requires_grad=True)
        self.bias = nn.Parameter(torch.empty([dimension_in * projection_factor], dtype=dtype, requires_grad=True))
        
    def forward(self, x: torch.Tensor):
        """Expect x to be shape [batch, dimension_in]."""
        return TransformerGatedLinearLayerFunction.apply(x, self.weight, self.bias)
    
    @staticmethod
    def from_torch(torch_module: BaseTransformerGatedLinearLayer):
        triton_module = OptimizedTransformerGatedLinearLayer(torch_module.dimension_in, torch_module.projection_factor, torch_module.linear.weight.dtype)
        
        # triton_module.weight = triton_module.weight.copy_(torch_module.linear.weight)
        # triton_module.bias = triton_module.bias.copy_(torch_module.linear.bias)
        return triton_module
    
def print_is_close(torch_tensor, triton_tensor, rtol=1e-2, atol=1e-1):
    if np.allclose(torch_tensor.detach().cpu(), triton_tensor.detach().cpu(), rtol=rtol, atol=atol):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
        print("Torch")
        print(forward_torch)
        print("Triton")
        print(forward_triton)
        print(forward_torch.shape)

if __name__ == "__main__":
    """Example of using module."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dim", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--seed", type=Optional[int], default=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
    
    input_tensor = torch.randn(args.batch_size, args.input_dim, dtype=torch.half)
    torch_layer = BaseTransformerGatedLinearLayer(args.input_dim)
    triton_layer = OptimizedTransformerGatedLinearLayer.from_torch(torch_layer)
    
    input_tensor = input_tensor.to('cuda')
    torch_layer = torch_layer.to('cuda')
    triton_layer = triton_layer.to('cuda')
        
    # Inference pass
    # TODO: grad support 
    repeats = args.repeats
    warmups = args.warmup
    forward_torch = torch_layer(input_tensor)
    forward_triton = triton_layer(input_tensor)
    
    loss_fn = lambda tensor: abs(tensor).sum()
    
    loss_torch = loss_fn(forward_torch)
    loss_triton = loss_fn(forward_triton)
    
    loss_triton.backward()
    loss_torch.backward()
    
    breakpoint()
    
    print("Forward output: ")
    print_is_close(forward_torch, forward_triton)
    print()
    
    print("Grad weight: ")
    print_is_close(torch_layer.linear.weight, triton_layer.weight)
    print()

    print("Grad bias: ")
    print_is_close(torch_layer.linear.bias, triton_layer.bias)
    print()
          
"""
transformer_gated_linear_forward_kernel_0d1d2d3d4d5d6d7d8c9d10c11c12d13c
Begins: 117.833s
Ends: 117.957s (+123.301 ms)
grid:  <<<40960, 1, 1>>>
block: <<<64, 1, 1>>>
Launch Type: Regular
Static Shared Memory: 0 bytes
Dynamic Shared Memory: 8,192 bytes
Registers Per Thread: 166
Local Memory Per Thread: 0 bytes
Local Memory Total: 64,815,104 bytes
Shared Memory executed: 65,536 bytes
Shared Memory Bank Size: 4 B
Theoretical occupancy: 25 %
Launched from thread: 1390772
Latency: ←13.597 s
Correlation ID: 15062
Stream: Default stream 7

void cutlass::Kernel<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x128_32x3_tn_align8>(T1::Params)
Begins: 31.886s
Ends: 31.9963s (+110.296 ms)
grid:  <<<128, 40, 1>>>
block: <<<256, 1, 1>>>
Launch Type: Regular
Static Shared Memory: 0 bytes
Dynamic Shared Memory: 73,728 bytes
Registers Per Thread: 232
Local Memory Per Thread: 0 bytes
Local Memory Total: 64,815,104 bytes
Shared Memory requested: 65,536 bytes
Shared Memory executed: 102,400 bytes
Shared Memory Bank Size: 4 B
Theoretical occupancy: 16.6667 %
Launched from thread: 1390772
Latency: ←10.458 s
Correlation ID: 10943
Stream: Default stream 7
"""