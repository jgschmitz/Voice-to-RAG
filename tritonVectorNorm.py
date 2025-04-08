@triton.jit
def normalize_kernel(x_ptr, output_ptr, n_elements, dim, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < dim

    x = tl.load(x_ptr + row_start * dim + offsets, mask=mask)
    norm = tl.sum(x * x, axis=0)
    norm = tl.sqrt(norm)
