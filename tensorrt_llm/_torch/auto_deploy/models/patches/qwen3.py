"""A patch for Qwen3 MoE to make it compatible with torch.export and reduce export time."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

from ...export.interface import BaseExportPatch, ExportPatchRegistry

# Import SiLUActivation for compatibility check
try:
    from transformers.activations import SiLUActivation

    _SILU_TYPES = (nn.SiLU, SiLUActivation)
except ImportError:
    _SILU_TYPES = (nn.SiLU,)


def _is_silu_activation(act_fn) -> bool:
    """Check if activation function is SiLU or equivalent."""
    return isinstance(act_fn, _SILU_TYPES)


def _forward_moe(self: Qwen3MoeSparseMoeBlock, hidden_states: torch.Tensor):
    # check if we can apply the patch
    unsupported_reasons = []
    # In transformers 5.x, self.experts may be a fused Qwen3MoeExperts object
    # that is not iterable (no individual expert modules). Skip the activation
    # check in that case — the fused implementation uses SiLU.
    if hasattr(self.experts, "__iter__"):
        if not all(_is_silu_activation(expert.act_fn) for expert in self.experts):
            unsupported_reasons.append("expert activation is not SiLU")

    if any(getattr(mod, "bias", None) is not None for mod in self.experts.modules()):
        unsupported_reasons.append("expert modules have bias")

    # Raise informative error for unsupported configurations
    # (fallback to original forward is not export-compatible with transformers >= 4.57.1)
    if unsupported_reasons:
        raise NotImplementedError(
            f"Qwen3MoeSparseMoeBlock forward patch does not support this model configuration: "
            f"{', '.join(unsupported_reasons)}. "
            f"The original transformers forward uses torch.nonzero() and tensor indexing "
            f"which are not compatible with torch.export on meta tensors."
        )

    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    # router_logits: (batch * sequence_length, n_experts)
    router_logits = self.gate(hidden_states)

    # In transformers 5.x the gate/router may return a tuple (logits, aux_loss).
    if isinstance(router_logits, tuple):
        router_logits = router_logits[0]

    top_k = (
        getattr(self, "top_k", None)
        or getattr(self, "num_experts_per_tok", None)
        or getattr(self.config, "num_experts_per_tok", 2)
    )
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    if getattr(self, "norm_topk_prob", getattr(self, "normalize_expert_weights", True)):
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    # we cast back to the input dtype
    routing_weights = routing_weights.to(hidden_states.dtype)

    # In transformers 5.x, self.experts may be a fused Qwen3MoeExperts object
    # with stacked weight tensors instead of individual expert modules.
    if hasattr(self.experts, "__iter__"):
        w1_weight = [expert.gate_proj.weight for expert in self.experts]
        w2_weight = [expert.down_proj.weight for expert in self.experts]
        w3_weight = [expert.up_proj.weight for expert in self.experts]
    else:
        # Fused experts: weights are [num_experts, ...] stacked tensors
        w1_weight = list(self.experts.gate_proj.weight.unbind(0))
        w2_weight = list(self.experts.down_proj.weight.unbind(0))
        w3_weight = list(self.experts.up_proj.weight.unbind(0))

    final_hidden_states = torch.ops.auto_deploy.torch_moe(
        hidden_states,
        selected_experts,
        routing_weights,
        w1_weight=w1_weight,
        w2_weight=w2_weight,
        w3_weight=w3_weight,
    )
    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits


@ExportPatchRegistry.register("hf_qwen3_moe")
class Qwen3MoePatch(BaseExportPatch):
    """Patch for Qwen3 MoE to make it compatible with torch.export and reduce export time.

    This patch replaces the forward method of Qwen3MoeSparseMoeBlock with
    a version that uses the torch_moe custom operator for better export compatibility.
    """

    def _apply_patch(self):
        """Apply the Qwen3 MoE patch."""
        # Store original forward method
        self.original_values["Qwen3MoeSparseMoeBlock.forward"] = Qwen3MoeSparseMoeBlock.forward

        # Apply patch by replacing the forward method
        Qwen3MoeSparseMoeBlock._original_forward = Qwen3MoeSparseMoeBlock.forward  # type: ignore
        Qwen3MoeSparseMoeBlock.forward = _forward_moe  # type: ignore

    def _revert_patch(self):
        """Revert the Qwen3 MoE patch."""
        # Restore original forward method
        Qwen3MoeSparseMoeBlock.forward = self.original_values["Qwen3MoeSparseMoeBlock.forward"]  # type: ignore

        # Clean up the temporary attribute
        if hasattr(Qwen3MoeSparseMoeBlock, "_original_forward"):
            delattr(Qwen3MoeSparseMoeBlock, "_original_forward")
