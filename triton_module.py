import torch 
from torch import nn 
import triton_kernel
from torch_module import BaseTransformerGatedLinearLayer
from typing import *
import numpy as np 

class OptimizedTransformerGatedLinearLayer(nn.Module):
    def __init__(self, base_module: BaseTransformerGatedLinearLayer) -> None:
        super().__init__()
        
        # TODO: handle device movement better
        self.weight = base_module.linear.weight.detach().cuda()
        self.bias = base_module.linear.bias.detach().cuda()
        
    def forward(self, x: torch.Tensor):
        """Expect x to be shape [batch, dimension_in]."""
        return triton_kernel.transformer_gated_linear_forward(x, self.weight, self.bias)
        
if __name__ == "__main__":
    """Example of using module."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dim", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--seed", type=Optional[int], default=None)
    args = parser.parse_args()
    
    torch.manual_seed(0)
    
    input_tensor = torch.randn(args.batch_size, args.input_dim, dtype=torch.half)
    torch_layer = BaseTransformerGatedLinearLayer(args.input_dim)
    triton_layer = OptimizedTransformerGatedLinearLayer(torch_layer)
    
    input_tensor = input_tensor.to('cuda')
    torch_layer = torch_layer.to('cuda')
    triton_layer = triton_layer.to('cuda')
    
    # Inference pass
    # TODO: grad support 
    with torch.no_grad():
        output_torch = torch_layer(input_tensor)
        output_triton = triton_layer(input_tensor)
            
        # Print outputs
        print("Output: ")
        print(output_torch)
        print(output_triton)
        
        if np.allclose(output_torch.cpu(), output_triton.cpu(), rtol=1e-2, atol=1e-1):
            print("✅ Triton and Torch match")
        else:
            print("❌ Triton and Torch differ")
        print()