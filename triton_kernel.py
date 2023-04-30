"""
Basic idea forward:
    - the hairy part of the gelu_fast is not decomposable (or at least I can't figure one out)
        - it is therefore relegated to being fused at the tail end of x2
    - we don't want to store intermediate activations
        - when we calculate `gelu_fast(x2)` we want to combine it with x1 
            without storing each tiled subsolution.
        - Luckily the multiplication in x1 * gelu_fast(x2) is decomposable
        - Consider a standard tiled matmul where C = \sum_i (A_i @ B_i)
            where C is an output tile and A_i and B_i are the corresponding tiles
            broken along the dimension of reduction
        - Let's say x1 was just a single tile, then x1 * gelu_fast(x2) -->
            [sum_i (A_i @ B_i)] * gelu_fast(x2)
        - To do this, just load the corresponding output tile in gelu_fast(x2) to shared memory
            then when you do your tiled matmul do as normal but do elemwise multiplication
            with the gelu_fast(x2) before accumulating. Otherwise do as normal
            
Basic idea backward:
    need to brush up on autograd but you want dL/dW where L is the loss, W is the weight matrix.
    if the input is x_in then Y can be written as this program:
        % x1 = x_in @ w1
        % x2 = x_in @ w2
        % Y = x1 * fast_gelu(x2)
    where @ is matmul and w1 and w2 are the weights of W split along the chunks.
    So given dL/dy we want dL/dw1 and dL/dw2
    
    dL/dw1 = dL/dy * dy/dw1 and dL/dw2 = dL/dy * dL/dw2       
    
    By wolfram alpha fast_gelu'(x) = 0.5 tanh(a(b * x^3 + x)) 
                                    + a * x (1.5 * b * x^2 + 0.5) * sech^2(a(b x^3 + x))
                                    + 0.5
    Where a = SQRT_2_OVERPI and b = FAST_GELU_INNER_CONST
                                    
    So dY/dw1 = fast_gelu(x2) * dx1/dw1    
       dY/dw2 = x1 * fast_gelu'(x2) * dx2/dw2 
       
    Where dx1/dw1 and dx2/dw2 we can just look up online for the lienar layers (pretty sure it's just x1).

    To calculate this efficiently you can just do. We don't need to store any intermediate tensors.
    TODO: deal with transposed weights
"""

import torch 
import triton 
import triton.language as tl 
from torch_module import SQRT_2_OVERPI, FAST_GELU_INNER_CONST, gelu_fast
import itertools
from typing import NamedTuple, Optional

class KernelLaunchParameters(NamedTuple):
    block_size_m: int 
    block_size_n: int 
    block_size_k: int 
    group_size_m: int

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'two_N', 'K'],
)
@triton.jit 
def transformer_gated_linear_forward_kernel(
    in_ptr, 
    weight_ptr,
    bias_ptr,
    out_ptr,
    # Matrix dimensions
    M, 
    two_N,
    K,
    stride_in_m, stride_in_k,
    stride_weight_n, stride_weight_k,
    stride_bias_n,
    stride_out_m, stride_out_n,
    # Meta-parameters
    ## Shared-Memory Blocking 
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    ## L2 Blocking
    GROUP_SIZE_M: tl.constexpr,
):
    """Calculates the equivalent of the following:
    
    T_in     : [M, K]
    T_weight : [2N, K]
    T_bias   : [2N]
    T_out    : [M, N]
    
    T_weight1 = T_weight[:N,:]
    T_bias1 = T_bias[:N]
    T_weight2 = T_weight[N:,:]
    T_bias2 = T_bias[N:]

    x1: [M, N] = T_in @ T_weight1.T + T_bias1
    x2: [M, N] = T_in @ T_weight2.T + T_bias2

    T_out = x1 * fast_gelu(x2)
    
    We do this by calculating buf = fast_gelu(x2) in tiles.
    We then calculate x1's corresponding tiles and then do 
    x1 * buf with these two pieces in shared memory. We note that 
    the corresponding tiles for x1 and x2 share accesses in the input 
    tensor so we fuse this to avoid an additional access to T_in.
    """
    
    pid = tl.program_id(axis=0)
    N = two_N // 2
    
    # Calculate tiling structures for subproblems calculating x1 and x2
    m_tiles = tl.cdiv(M, BLOCK_SIZE_M)
    n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
        
    # L2 blocking by group
    num_pid_in_group = GROUP_SIZE_M * n_tiles
    group_number = pid // num_pid_in_group 
    group_start_pid = group_number * GROUP_SIZE_M
    group_size_m = min(m_tiles - group_start_pid, GROUP_SIZE_M)
    
    # Map final output tile indices
    pid_tile_m = group_start_pid + (pid % group_size_m)
    pid_tile_n = (pid % num_pid_in_group) // group_size_m
    
    # # Part 1: fast_gelu(x2) calculation
    # accumulator_x2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)    
    # weight2_ptr = weight_ptr + N * stride_weight_n
    # bias2_ptr = bias_ptr + N * stride_bias_n
    # accumulator_x2 = calculate_linear_tile(
    #     accumulator_x2, 
    #     in_ptr, 
    #     weight2_ptr, 
    #     bias2_ptr,
    #     pid_tile_m, 
    #     pid_tile_n, 
    #     M, N, K, 
    #     stride_in_m, stride_in_k, 
    #     stride_weight_n, stride_weight_k, 
    #     stride_bias_n,
    #     BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    # )
    # accumulator_x2 = fast_gelu_kernel(accumulator_x2)
        
    # # Part 2: x1 calculation of corresponding tile
    # # TODO: parallelize this across SM + synchronization
    # accumulator_x1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)    
    # weight1_ptr = weight_ptr
    # bias1_ptr = bias_ptr
    # accumulator_x1 = calculate_linear_tile(
    #     accumulator_x1, 
    #     in_ptr, 
    #     weight1_ptr, 
    #     bias1_ptr,
    #     pid_tile_m, 
    #     pid_tile_n, 
    #     M, N, K, 
    #     stride_in_m, stride_in_k, 
    #     stride_weight_n, stride_weight_k, 
    #     stride_bias_n,
    #     BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    # )

    # Part 1 + 2: calculate x1 and x2 tile in parallel
    # TODO: bias leads to NaN, figure out why???
    accumulator_x1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)    
    weight1_ptr = weight_ptr
    bias1_ptr = bias_ptr
    accumulator_x2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)    
    weight2_ptr = weight_ptr + N * stride_weight_n
    bias2_ptr = bias_ptr + N * stride_bias_n
    accumulator_x1, accumulator_x2 = calculate_dual_linear_tile_fused(
        accumulator_x1,
        accumulator_x2,
        in_ptr,
        weight1_ptr,
        weight2_ptr,
        bias1_ptr,
        bias2_ptr, 
        pid_tile_m, 
        pid_tile_n, 
        M, N, K, 
        stride_in_m, stride_in_k, 
        stride_weight_n, stride_weight_k, 
        stride_bias_n,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )

    # Part 3: combine the results
    accumulator_x1 *= fast_gelu_kernel(accumulator_x2)

    # TODO: parametrize casting to dtype
    out = accumulator_x1.to(tl.float16)
    offs_out_m = pid_tile_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_out_n = pid_tile_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_ptrs = out_ptr + stride_out_m * offs_out_m[:, None] + stride_out_n * offs_out_n[None, :]
    mask = (offs_out_m[:, None] < M) & (offs_out_n[None, :] < N)
    tl.store(out_ptrs, out, mask=mask)

@triton.jit
def calculate_dual_linear_tile_fused(
    accumulator1,
    accumulator2, 
    in_ptr,
    weight1_ptr,
    weight2_ptr,
    bias1_ptr,
    bias2_ptr,
    pid_tile_m, 
    pid_tile_n, 
    # Dimensions for iterating
    M, N, K, 
    stride_in_m, stride_in_k,
    stride_weight_n, stride_weight_k,
    stride_bias_n,
    # Block size information
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
):
    """Calculates the tile at (pid_tile_m, pid_tile_n)"""
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    offs_in_m = (pid_tile_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_weight_n = (pid_tile_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)    
    in_load_ptrs = in_ptr + (offs_in_m[:, None] * stride_in_m + offs_k[None, :] * stride_in_k)
    weight1_load_ptrs = weight1_ptr + (offs_weight_n[:, None] * stride_weight_n + offs_k[None, :] * stride_weight_k)
    weight2_load_ptrs = weight2_ptr + (offs_weight_n[:, None] * stride_weight_n + offs_k[None, :] * stride_weight_k)
    for k in range(0, k_tiles):
        # Calculate x1 and x2 subanswers at the same time
        T_in = tl.load(in_load_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0) 
        
        # NOTE: if in the mask you do `< K - k * BLOCK_SIZE_K` instead, it will fail to type check in triton 
        # git+4b072516e7660d4e1cacea2b03371886ed88e81a and silently give you NaNs in 2.0.0
        # TODO: file a bug report after creating more reproducible example.
        T_weight1 = tl.load(weight1_load_ptrs, mask=offs_k[None, :] <= K - k * BLOCK_SIZE_K - 1, other=0.0)        
        T_weight2 = tl.load(weight2_load_ptrs, mask=offs_k[None, :] <= K - k * BLOCK_SIZE_K - 1, other=0.0)        

        # TODO: think about memory implications of transpose 
        # Pro: slightly different ordering of loops is faster
        # Con: using tl.dot probably allows Tensorcore out of box
        #      tl.dot might already support this
        result_out1 = tl.dot(T_in, T_weight1.T)
        result_out2 = tl.dot(T_in, T_weight2.T)
        accumulator1 += result_out1
        accumulator2 += result_out2

        in_load_ptrs += BLOCK_SIZE_K * stride_in_k
        weight1_load_ptrs += BLOCK_SIZE_K * stride_weight_k
        weight2_load_ptrs += BLOCK_SIZE_K * stride_weight_k
    
    # Handle bias
    offs_bias = pid_tile_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_bias = offs_bias[None, :]
    bias1_load_ptrs = bias1_ptr + (offs_bias * stride_bias_n)
    bias2_load_ptrs = bias2_ptr + (offs_bias * stride_bias_n)
    T_bias1 = tl.load(bias1_load_ptrs, mask=offs_bias < N, other=0.0)     
    T_bias2 = tl.load(bias2_load_ptrs, mask=offs_bias < N, other=0.0)     
    
    accumulator1 += T_bias1
    accumulator2 += T_bias2    

    return accumulator1, accumulator2

@triton.jit
def calculate_linear_tile(
    accumulator, 
    in_ptr,
    weight_ptr,
    bias_ptr,
    pid_tile_m, 
    pid_tile_n, 
    # Dimensions for iterating
    M, N, K, 
    stride_in_m, stride_in_k,
    stride_weight_n, stride_weight_k,
    stride_bias_n,
    # Block size information
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
):
    """Calculates the tile at (pid_tile_m, pid_tile_n)"""
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

    offs_in_m = (pid_tile_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_weight_n = (pid_tile_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    in_load_ptrs = in_ptr + (offs_in_m[:, None] * stride_in_m + offs_k[None, :] * stride_in_k)
    weight_load_ptrs = weight_ptr + (offs_weight_n[:, None] * stride_weight_n + offs_k[None, :] * stride_weight_k)
    for k in range(0, k_tiles):
        T_in = tl.load(in_load_ptrs, mask=offs_k[None, :] <= K - k * BLOCK_SIZE_K - 1, other=0.0) 
        T_weight = tl.load(weight_load_ptrs, mask=offs_k[None, :] <= K - k * BLOCK_SIZE_K - 1, other=0.0)        
        accumulator += tl.dot(T_in, tl.trans(T_weight))
        in_load_ptrs += BLOCK_SIZE_K * stride_in_k
        weight_load_ptrs += BLOCK_SIZE_K * stride_weight_k
        
    offs_bias = pid_tile_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_bias = offs_bias[None, :]
    bias_load_ptrs = bias_ptr + (offs_bias * stride_bias_n)
    T_bias = tl.load(bias_load_ptrs, mask=offs_bias < N, other=0.0)     
    accumulator += T_bias
    
    return accumulator
        
@triton.jit
def fast_gelu_kernel(buffer):
    return 0.5 * buffer * (1.0 + tl.math.tanh(SQRT_2_OVERPI * buffer * (1.0 + FAST_GELU_INNER_CONST * buffer * buffer)))

def transformer_gated_linear_forward(input_tensor: torch.Tensor, weight_tensor: torch.Tensor, bias_tensor: torch.Tensor, kernel_launch_parameters: Optional[KernelLaunchParameters] = None, synchronize: bool = False) -> torch.Tensor:
    # Check constraints.
    assert len(bias_tensor.shape) == 1, "Bias should have one dimension"
    assert bias_tensor.shape[0] == weight_tensor.shape[0], "Incompatible bias dimensions"
    assert input_tensor.shape[1] == weight_tensor.shape[1], "Incompatible reduction dimensions"
    assert input_tensor.is_contiguous(), "Matrix A must be contiguous"
    assert weight_tensor.is_contiguous(), "Matrix B must be contiguous"
    M, K = input_tensor.shape
    two_N, K = weight_tensor.shape
    
    assert two_N % 2 == 0, "Output dimension must be even."
    N = two_N // 2
    
    # Allocates output.
    output_tensor = torch.empty((M, N), device=input_tensor.device, dtype=input_tensor.dtype)        
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
      
    # TODO: figure out how autotuning works to make this cleaner
    if kernel_launch_parameters is not None:
        base_function = transformer_gated_linear_forward_kernel.fn
        base_function[grid](
            input_tensor, 
            weight_tensor, 
            bias_tensor, 
            output_tensor,
            M, two_N, K,
            input_tensor.stride(0), input_tensor.stride(1),
            weight_tensor.stride(0), weight_tensor.stride(1),
            bias_tensor.stride(0), 
            output_tensor.stride(0), output_tensor.stride(1),
            kernel_launch_parameters.block_size_m,
            kernel_launch_parameters.block_size_n,
            kernel_launch_parameters.block_size_k,
            kernel_launch_parameters.group_size_m
        )
    else:
        result = transformer_gated_linear_forward_kernel[grid](
            input_tensor, 
            weight_tensor, 
            bias_tensor, 
            output_tensor,
            M, two_N, K,
            input_tensor.stride(0), input_tensor.stride(1),
            weight_tensor.stride(0), weight_tensor.stride(1),
            bias_tensor.stride(0), 
            output_tensor.stride(0), output_tensor.stride(1),
        )
        # print("Autotuned best:", result.metadata)
    return output_tensor

if __name__ == "__main__":
    torch.manual_seed(0)

    M = [312, 512, 761, 1000]
    two_N = [312, 512]
    K = [i for i in range(761, 761 + 65)]
    kernel_launch_parameters = KernelLaunchParameters(block_size_m=16, block_size_n=64, block_size_k=16, group_size_m=1)
    for m, two_n, k in itertools.product(M, two_N, K):
        print(f"M: {m:<6} 2N: {two_n:<6} K: {k:<6}")
        T_in = torch.randn((m, k), device='cuda', dtype=torch.float16)
        T_weight = torch.randn((two_n, k), device='cuda', dtype=torch.float16)
        T_bias = torch.randn((two_n,), device='cuda', dtype=torch.float16) 
        
        triton_output = transformer_gated_linear_forward(T_in, T_weight, T_bias, kernel_launch_parameters=kernel_launch_parameters)
        torch_output = T_in.float() @ T_weight.T.float() + T_bias.float()
        x1, x2 = torch_output.chunk(2, dim=(torch_output.ndim - 1))
        expected_torch_output = (x1 * gelu_fast(x2)).half()
        
        # TODO: think about if these tolerances are too high
        if torch.allclose(triton_output, expected_torch_output, atol=1e-1, rtol=1e-1):
            print("✅ Triton and Torch match")
        else:
            print("❌ Triton and Torch differ")
            print("Triton")
            print(triton_output)
            print("Torch")
            print(expected_torch_output)
        print()