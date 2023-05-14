# Writeup

## Analysis of Memory Usage
### Forward Pass
Consider the number of elements in each part of our operation, where `projection_factor=8`. 
`d` is the embedding dimension and `N` is the batch size.

```
input: [N, d]
weights: [8d, d]
bias: [8d]
output: [N, 4d]
```

Consider the number of elements in parameters which must be placed on gpu:

`8 * d * d + 8 * d` (weight + bias elements)

Now consider the total activation elements stored in global memory in the idea scenario:

`5 * N * d` (input + output elements)

The issue with the naive torch implementation is it materializes all intermediate activations 
in global memory. In particular it stores the computation of `input @ weights.T : [N, 8d]`.
This means it stores at least the following number of activation elements:

`9 * N * d`

Which means in the ideal scenario we save:

`4 * N * d`

elements saved if we do not materialize the intermediate activations at all.
In practice it we save more, as torch operations are not done in place so additional buffers are 
allocated. Our planned optimization only reduces the activation elements, so it is helpful
to figure out the ratio to elements saved in activations vs elements in parameters.

`(4 * N * d) / (8 * d * d + 8 * d) = N / (2 * [d + 1])` 

So we can only really appreciably save memory when `N > d`. This is possible for some typical
workloads. For example llama-65B has `d=8192` and batch sizes are in the tens of thousands.
We showcase these ranges of `d` and `N` in `benchmark.py`.

### Backward Pass
I am not very good at this but here is raw torch pseudo-code:

```
def backward_reference(ctx, grad_output) -> Any:
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
        dim=0,
    )

    # bias calculation
    bias_grad = torch.cat(
        [
            (gelu_fast(x2) * grad_output).sum(0).squeeze(),
            (x1 * derivative_gelu_fast(x2) * grad_output).sum(0).squeeze(),
        ],
        dim=0,
    )

    # input calculation
    input_grad = (grad_output * gelu_fast(x2)) @ w1 + (
        grad_output * derivative_gelu_fast(x2) * x1
    ) @ w2
    return input_grad.half(), weight_grad.half(), bias_grad.half()
```

For weights, it is yet another fused matmul, where we calculate two tiles of output (along dim=0) at the same time.
For bias, the sum presents a difficulty due to efficient impl requiring reduction/scan ops. I'm going to assume
    triton's sum() method does stuff under the hood to make this good. Otherwise it is the same idea.
For inputs, it is similar to the forward pass strategy.

# Alternatives and Extensions

# Issues
Triton is kind of sus. Very much in development. Debugging triton is a journey and the documentation
isn't very good. For example, there are ways to dump out the IR at various steps, but this is not
super apparent to me how you do this. Reading the code reveals it is easy to get the SASS but this
is not feaseable to use on a day to day basis.

I could investigate more, but I am time-boxing this exercise.