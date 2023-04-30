import torch 
from torch import nn 
import numpy as np 
from typing import Optional

SQRT_2_OVERPI = np.sqrt(2 / np.pi)
FAST_GELU_INNER_CONST = 0.044715

def gelu_fast(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(SQRT_2_OVERPI * x * (1.0 + FAST_GELU_INNER_CONST * x * x)))

class BaseTransformerGatedLinearLayer(nn.Module):
    def __init__(self, dimension_in: int, projection_factor: int = 8, dtype: torch.dtype=torch.half) -> None:
        super().__init__()
        
        if projection_factor % 2 != 0 or projection_factor < 2:
            raise ValueError("Projection factor must be even and >= 2.")
        
        self.dimension_in = dimension_in
        self.projection_factor = projection_factor
        self.linear = nn.Linear(self.dimension_in, self.dimension_in * self.projection_factor, bias=True, dtype=dtype)
        
    def forward(self, x: torch.Tensor):
        """Expect x to be shape [batch, dimension_in]."""
        x = self.linear(x)
        x1, x2 = x.chunk(2, dim=(x.ndim - 1))
        return x1 * gelu_fast(x2)

if __name__ == "__main__":
    """Example of using module."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dim", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--seed", type=Optional[int], default=None)
    args = parser.parse_args()
        
    input_tensor = torch.randn(args.batch_size, args.input_dim, dtype=torch.half)
    layer = BaseTransformerGatedLinearLayer(args.input_dim)
    
    input_tensor = input_tensor.to('cuda')
    layer = layer.to('cuda')
    
    # Forward pass 
    output_tensor = layer(input_tensor)
    
    # Backwards pass
    loss = torch.sum(torch.abs(output_tensor))
    loss.backward(retain_graph=True)
        
    # Print outputs
    print("Output: ")
    print(output_tensor)
    print()
    print("Grad: ")
    print(layer.linear.weight.grad)

    breakpoint()
    print()