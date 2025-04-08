# triton_scale.py
import torch
import triton
import triton.language as tl

@triton.jit
def scale_kernel(x_ptr, output_ptr, scale, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    out = x * scale
    tl.store(output_ptr + offsets, out, mask=mask)

def scale_vector_gpu(vec: torch.Tensor, scale: float):
    assert vec.is_cuda
    output = torch.empty_like(vec)
    n_elements = vec.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    scale_kernel[grid](vec, output, scale, n_elements, BLOCK_SIZE=1024)
    return output
