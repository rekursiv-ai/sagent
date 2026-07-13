__all__ = [
    "get_dynamic_sparse_quantized_mapping",
    "get_static_sparse_quantized_mapping",
]

def get_static_sparse_quantized_mapping() -> dict[
    type[torch.nn.modules.linear.Linear],
    type[torch.ao.nn.sparse.quantized.linear.Linear],
]: ...
def get_dynamic_sparse_quantized_mapping() -> dict[
    type[torch.nn.modules.linear.Linear],
    type[torch.ao.nn.sparse.quantized.dynamic.linear.Linear],
]: ...
