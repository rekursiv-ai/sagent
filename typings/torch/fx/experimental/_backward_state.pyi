import torch.fx

class BackwardState:
    proxy: torch.fx.Proxy
