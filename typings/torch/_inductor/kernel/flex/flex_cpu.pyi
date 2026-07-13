def check_cpu_supported() -> bool: ...
def lower_cpu(
    query,
    key,
    value,
    subgraph,
    block_mask,
    scale,
    kernel_options,
    score_mod_other_buffers,
    mask_mod_other_buffers,
) -> tuple[TensorBox | ShapeAsConstantBuffer | Any]: ...
