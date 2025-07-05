import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_parent_module(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    """
    Given the root model and a module_name like 'a.b.c', returns (parent_module, 'c')
    so you can replace the child in the parent.
    """
    parts = module_name.split('.')
    attr = parts[-1]
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, attr


class LoRALinear(nn.Module):
    """
    A linear layer with a low-rank adaptation (LoRA) injection.
    Freezes the original weight and bias, and adds two small A/B matrices.
    """
    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int,
        alpha: int | None = None,
    ):
        super().__init__()
        # store original dimensions and device
        device = base_linear.weight.device
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        # freeze original weight and bias
        self.weight_orig = base_linear.weight.detach().clone().to(device)
        self.weight_orig.requires_grad = False
        if base_linear.bias is not None:
            self.bias_orig = base_linear.bias.detach().clone().to(device)
            self.bias_orig.requires_grad = False
        else:
            self.bias_orig = None
        # LoRA parameters settings
        self.rank = rank
        self.alpha = alpha if alpha is not None else rank
        self.scaling = self.alpha / self.rank
        # low-rank adaptation matrices on same device
        # A has shape [rank, in_features], B has [out_features, rank]
        self.A = nn.Parameter(torch.randn(self.rank, self.in_features, device=device) * 0.01)
        self.B = nn.Parameter(torch.randn(self.out_features, self.rank, device=device) * 0.01)
        # runtime scale (for inference strength)
        self.scale = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # original linear output
        y = F.linear(x, self.weight_orig, self.bias_orig)
        # compute LoRA update: (x @ A^T) @ B^T
        delta = (x @ self.A.T) @ self.B.T
        # add scaled LoRA update
        return y + delta * self.scaling * self.scale

    def set_scale(self, scale: float) -> None:
        """Set the runtime scaling factor for the LoRA update."""
        self.scale = scale


def inject_lora(
    model: nn.Module,
    target_modules: list[str],
    rank: int,
    alpha: int | None = None,
) -> nn.Module:
    """
    Traverse the model and replace any nn.Linear module whose full module
    name ends with one of target_modules with a LoRALinear wrapper.

    Args:
        model: The root nn.Module to patch.
        target_modules: List of suffixes for module names to wrap (e.g. ['to_q', 'to_k']).
        rank: The low-rank dimension for A and B.
        alpha: Scaling factor (defaults to rank if None).

    Returns:
        The model with LoRA adapters injected in place.
    """
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and any(name.endswith(tm) for tm in target_modules):
            parent, attr = _get_parent_module(model, name)
            lora_mod = LoRALinear(module, rank, alpha)
            # ensure adapter is on same device as original
            lora_mod.to(module.weight.device)
            setattr(parent, attr, lora_mod)
    return model


def extract_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """
    Collects and returns a state dict of only the LoRA parameters (A and B matrices).
    Keys are '<module_name>.A' and '<module_name>.B'.

    Args:
        model: The LoRA-injected model from which to extract parameters.

    Returns:
        A dict mapping names to CPU tensors for all LoRA modules.
    """
    sd: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            sd[f"{name}.A"] = module.A.detach().cpu()
            sd[f"{name}.B"] = module.B.detach().cpu()
    return sd
