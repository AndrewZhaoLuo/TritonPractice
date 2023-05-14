import torch 
from torch import nn 
import triton_forward_kernel
import triton_backward_kernel
from torch_module import BaseTransformerGatedLinearLayer, gelu_fast, derivative_gelu_fast
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
        return triton_forward_kernel.transformer_gated_linear_forward(
            input_tensor, 
            weight_tensor, 
            bias_tensor
        )
        
    @staticmethod
    def backward(ctx, grad_output) -> Any:
        # TODO
        input_tensor, weight_tensor, bias_tensor = ctx.saved_tensors

        input_tensor = input_tensor.float()
        weight_tensor = weight_tensor.float()
        
        x = input_tensor @ weight_tensor.T
        x += bias_tensor
        x1, x2 = x.chunk(2, dim=(x.ndim - 1))
        two_N = weight_tensor.shape[0]
        N = two_N // 2
        w1 = weight_tensor[:N]
        w2 = weight_tensor[N:]
        
        # weight calculation        
        weight_grad = torch.cat(
            [
                (grad_output * gelu_fast(x2)).T @ input_tensor, 
                (grad_output * x1 * derivative_gelu_fast(x2)).T @ input_tensor,
            ], 
            dim=0
        )

        # Bug here? grad_output has .strides() of (0, 0) due to not being contiguous
        # Make contiguous and it's all ok.
        grad_output = grad_output.contiguous()
        weight_grad2 = triton_backward_kernel.transformer_gated_linear_backward_weight_grad(
            x, input_tensor, grad_output
        )
        print(weight_grad2[:16, :16])
        breakpoint()

        # bias calculation
        bias_grad = torch.cat([
                (gelu_fast(x2) * grad_output).sum(0).squeeze(), 
                (x1 * derivative_gelu_fast(x2) * grad_output).sum(0).squeeze()
            ], 
            dim=0
        )
        
        # input calculation
        input_grad = (grad_output * gelu_fast(x2)) @ w1 + (grad_output * derivative_gelu_fast(x2) * x1) @ w2 
        return input_grad.half(), weight_grad.half(), bias_grad.half()
        
        
class OptimizedTransformerGatedLinearLayer(nn.Module):
    def __init__(self, dimension_in: int, projection_factor: int = 8, dtype: torch.dtype=torch.half, weight_init: Optional[torch.Tensor] = None, bias_init: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        
        weight = torch.empty([dimension_in * projection_factor, dimension_in], dtype=dtype)
        bias = torch.empty([dimension_in * projection_factor], dtype=dtype)
        
        if weight_init is not None:
            weight = weight.copy_(weight_init)
        if bias_init is not None:
            bias = bias.copy_(bias_init)
            
        # TODO: proper initialization
        self.weight = nn.Parameter(weight, requires_grad=True)
        self.bias = nn.Parameter(bias, requires_grad=True)
        
    def forward(self, x: torch.Tensor):
        """Expect x to be shape [batch, dimension_in]."""
        return TransformerGatedLinearLayerFunction.apply(x, self.weight, self.bias)
    
    @staticmethod
    def from_torch(torch_module: BaseTransformerGatedLinearLayer):
        return OptimizedTransformerGatedLinearLayer(torch_module.dimension_in, torch_module.projection_factor, torch_module.linear.weight.dtype, weight_init=torch_module.linear.weight, bias_init = torch_module.linear.bias)    
    
def print_is_close(torch_tensor, triton_tensor, rtol=1e-2, atol=1e-1):
    if np.allclose(torch_tensor.detach().cpu(), triton_tensor.detach().cpu(), rtol=rtol, atol=atol):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
        print("Torch")
        print(torch_tensor)
        print("Triton")
        print(triton_tensor)
        print(triton_tensor.shape)

if __name__ == "__main__":
    """Example of using module."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dim", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    
    repeats = args.repeats
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
    
    for i in range(repeats):
        print(f"******Iteration {i}******")
        input_tensor = torch.randn(args.batch_size, args.input_dim, dtype=torch.half, requires_grad=True)
        torch_layer = BaseTransformerGatedLinearLayer(args.input_dim, init_non_zero_bias=True)
        triton_layer = OptimizedTransformerGatedLinearLayer.from_torch(torch_layer)
        
        input_tensor = input_tensor.to('cuda')
        torch_layer = torch_layer.to('cuda')
        triton_layer = triton_layer.to('cuda')
            
        input_tensor_torch = input_tensor.clone()
        input_tensor_torch.retain_grad()
        input_tensor_triton = input_tensor.clone()
        input_tensor_triton.retain_grad()
        forward_torch = torch_layer(input_tensor_torch)
        forward_triton = triton_layer(input_tensor_triton)
        
        print("****Forward****")
        print("Forward tensor: ")
        print_is_close(forward_torch, forward_triton)
        print()
        
        print("****Backward****")
        loss_fn = lambda tensor: torch.sum(tensor) # (tensor - torch.ones_like(tensor)).pow(2).sum()
        loss_torch = loss_fn(forward_torch)
        loss_triton = loss_fn(forward_triton)
        loss_triton.backward()
        loss_torch.backward()
        print("Grad weight: ")
        print_is_close(torch_layer.linear.weight.grad, triton_layer.weight.grad, atol=0.05, rtol=0.01)
        print()

        print("Grad bias: ")
        print_is_close(torch_layer.linear.bias.grad, triton_layer.bias.grad, atol=0.05, rtol=0.01)
        print()
        
        print("Grad input: ")
        print_is_close(input_tensor_torch.grad, input_tensor_triton.grad, atol=0.05, rtol=0.01)
        print()