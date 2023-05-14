import torch 
import triton 
import triton.language as tl 
from torch_module import SQRT_2_OVERPI, FAST_GELU_INNER_CONST, gelu_fast, derivative_gelu_fast
import itertools
from typing import NamedTuple, Optional
from triton_forward_kernel import fast_gelu_kernel, derivate_fast_gelu_kernel

#### Backwards Kernel
def transformer_gated_linear_backward_weight_grad(
    x_tensor: torch.Tensor, 
    input_tensor: torch.Tensor, 
    grad_output_tensor: torch.Tensor, 
) -> torch.Tensor:
    '''
    T_x           : [M, 2N] = torch.cat[x1: [M, N], x2: [M, N], dim=1]
    T_input       : [M, n]
    T_grad_output : [M, N]
    T_weight_grad : [2N, n] = torch.cat[w1: [N, n], w2: [N, n], dim=0]
    '''
    assert len(x_tensor.shape) == 2, "Expect rank 2 for x_tensor"
    M, two_N = x_tensor.shape 
    assert two_N % 2 == 0, "Output dimension must be even."
    N = two_N // 2
    assert input_tensor.shape[0] == M, "Not compatible dimensions"
    n = input_tensor.shape[1]

    assert list(grad_output_tensor.shape) == [M, N], "Not compatible dimensions"
    
    # Allocates output.
    weight_grad_tensor = torch.zeros((two_N, n), device=input_tensor.device, dtype=input_tensor.dtype)
    grid = lambda META: (
        triton.cdiv(N, META['BLOCK_SIZE_N']) * triton.cdiv(n, META['BLOCK_SIZE_n']),
    )

    result = transformer_gated_linear_backward_kernel_weights[grid](
        x_tensor, 
        input_tensor, 
        grad_output_tensor, 
        weight_grad_tensor,
        two_N,
        n,
        M,
        x_tensor.stride(0), x_tensor.stride(1),
        input_tensor.stride(0), input_tensor.stride(1),
        grad_output_tensor.stride(0), grad_output_tensor.stride(1),
        weight_grad_tensor.stride(0), weight_grad_tensor.stride(1)
    )

    # print("Autotuned best:", result.metadata)
    return weight_grad_tensor

@triton.autotune(
    configs=[        
        #triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_n': 32, 'BLOCK_SIZE_M': 32, 'GROUP_SIZE_N': 8}, num_stages=5, num_warps=2),
        #triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_n': 32, 'BLOCK_SIZE_M': 32, 'GROUP_SIZE_N': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_n': 16, 'BLOCK_SIZE_M': 16, 'GROUP_SIZE_N': 8}, num_stages=4, num_warps=4),
        #triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_n': 32, 'BLOCK_SIZE_M': 32, 'GROUP_SIZE_N': 8}, num_stages=4, num_warps=4),
        #triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_n': 64, 'BLOCK_SIZE_M': 32, 'GROUP_SIZE_N': 8}, num_stages=4, num_warps=4),
    ],
    key=['two_N', 'n', 'M'],
)
@triton.jit 
def transformer_gated_linear_backward_kernel_weights(
    # Inputs
    x_ptr, 
    input_tensor_ptr,
    grad_output_ptr,
    # Outputs
    weight_grad_ptr,
    # Matrix dimensions
    two_N,
    n,
    M, 
    stride_x_M, stride_x_N,
    stride_input_M, stride_input_n,
    stride_grad_output_M, stride_grad_output_N,
    stride_weight_grad_N, stride_weight_grad_n,
    # Meta-parameters
    ## Shared-Memory Blocking 
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_n: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    ## L2 Blocking
    GROUP_SIZE_N: tl.constexpr,
):
    """Calculates the equivalent of the following:
    
    Let x = input_tensor @ weight_tensor.T + bias_tensor. Calculates the following:
    
    Let N = n * proj_factor:

    T_x           : [M, 2N] = torch.cat[x1: [M, N], x2: [M, N], dim=1]
    T_input       : [M, n]
    T_grad_output : [M, N]
    T_weight_grad : [2N, n] = torch.cat[w1: [N, n], w2: [N, n], dim=0]

    ```
        x1, x2 = x.chunk(2, dim=(x.ndim - 1))

        w1_grad = (grad_output * gelu_fast(x2)).T @ input_tensor,
        w2_grad = (grad_output * x1 * derivative_gelu_fast(x2)).T @ input_tensor
        weight_grad = torch.cat(
            [w1_grad, w2_grad], 
            dim=0
        )
    ```

    Each pid essentially calculates one block of w1 and w2 each. 
    Each pid reuses the corresponding tiles input_tensor, x2, and grad_output.
    """
    
    pid = tl.program_id(axis=0)
    N = two_N // 2

    # Calculate tiling structures for subproblems calculating x1 and x2
    # Each program calculates the corresponding block in w1_grad and w2_grad
    N_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    n_tiles = tl.cdiv(n, BLOCK_SIZE_n)
        
    # L2 blocking by group
    num_pid_in_group = GROUP_SIZE_N * n_tiles
    group_number = pid // num_pid_in_group 
    group_start_pid = group_number * GROUP_SIZE_N
    group_size_N = min(N_tiles - group_start_pid, GROUP_SIZE_N)
    
    # Map final output tile indices
    pid_tile_N = group_start_pid + (pid % group_size_N)
    pid_tile_n = (pid % num_pid_in_group) // group_size_N

    # DEBUG: remove this when done
    if pid_tile_N == 0 and pid_tile_n == 0:
        accumulator_w1_grad = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_n), dtype=tl.float32)    
        accumulator_w2_grad = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_n), dtype=tl.float32)    
        
        x1_ptr = x_ptr 
        x2_ptr = x_ptr + N * stride_x_N

        offsets_N = tl.minimum(pid_tile_N * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N), N - 1)
        offsets_n = tl.minimum(pid_tile_n * BLOCK_SIZE_n + tl.arange(0, BLOCK_SIZE_n), n - 1)
        offsets_M = tl.arange(0, BLOCK_SIZE_M)

        x1_load_ptrs = x1_ptr + (offsets_M[:, None] * stride_x_M + offsets_N[None, :] * stride_x_N)
        x2_load_ptrs = x2_ptr + (offsets_M[:, None] * stride_x_M + offsets_N[None, :] * stride_x_N)
        grad_output_load_ptrs = grad_output_ptr + (offsets_M[:, None] * stride_grad_output_M + offsets_N[None, :] * stride_grad_output_N)
        input_tensor_load_ptrs = input_tensor_ptr + (offsets_M[:, None] * stride_input_M + offsets_n[None, :] * stride_input_n)
        M_tiles = tl.cdiv(M, BLOCK_SIZE_M)

        for tile_M in range(0, M_tiles):
            bound_M = tl.maximum(0, M - tile_M * BLOCK_SIZE_M)
            T_x1 = tl.load(x1_load_ptrs, mask=offsets_M[:, None] < bound_M, other=0.0) 
            T_x2 = tl.load(x2_load_ptrs, mask=offsets_M[:, None] < bound_M, other=0.0) 
            T_grad_output = tl.load(grad_output_load_ptrs, mask=offsets_M[:, None] < bound_M, other=0.0) 
            T_input_tensor = tl.load(input_tensor_load_ptrs, mask=offsets_M[:, None] < bound_M, other=0.0) 
            
            left_side_w1 = T_grad_output * fast_gelu_kernel(T_x2) 
            left_side_w2 = T_grad_output * T_x1 * derivate_fast_gelu_kernel(T_x2)
            accumulator_w1_grad += tl.dot(left_side_w1.T, T_input_tensor)
            accumulator_w2_grad += tl.dot(left_side_w2.T, T_input_tensor)

            x1_load_ptrs += BLOCK_SIZE_M * stride_x_M
            x2_load_ptrs += BLOCK_SIZE_M * stride_x_M
            grad_output_load_ptrs += BLOCK_SIZE_M * stride_grad_output_M
            input_tensor_load_ptrs += BLOCK_SIZE_M * stride_input_M

        # TODO: parametrize casting to dtype
        accumulator_w1_grad = accumulator_w1_grad.to(tl.float16)
        accumulator_w2_grad = accumulator_w2_grad.to(tl.float16)
        
        offs_out_N = pid_tile_N * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_out_n = pid_tile_n * BLOCK_SIZE_n + tl.arange(0, BLOCK_SIZE_n)

        out_ptr_w1_grad = weight_grad_ptr
        out_ptr_w2_grad = weight_grad_ptr + N * stride_weight_grad_N

        out_ptrs_w1_grad = out_ptr_w1_grad + offs_out_N[:, None] * stride_weight_grad_N + offs_out_n[None, :] * stride_weight_grad_n
        out_ptrs_w2_grad = out_ptr_w2_grad + offs_out_N[:, None] * stride_weight_grad_N + offs_out_n[None, :] * stride_weight_grad_n
        mask = (offs_out_N[:, None] < N) & (offs_out_n[None, :] < n)

        tl.store(out_ptrs_w1_grad, accumulator_w1_grad, mask=mask)
        tl.store(out_ptrs_w2_grad, accumulator_w2_grad, mask=mask)
            