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
                                    
    So dY/dw1 = fast_gelu(x2) * dx1/dw1    
       dY/dw2 = x1 * fast_gelu'(x2) * dx2/dw2 
       
    Where dx1/dw1 and dx2/dw2 we can just look up online for the lienar layers (pretty sure it's just x1).

    To calculate this efficiently you can just do. We don't need to store any intermediate tensors.
    TODO: deal with transposed weights
"""