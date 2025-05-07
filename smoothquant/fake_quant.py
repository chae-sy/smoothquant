import torch
from torch import nn
from functools import partial


# ---------------------------------------------------------------------
# 1.  Pure‑tensor helpers – *stateless* functions that perform FakeQuant
# ---------------------------------------------------------------------

@torch.no_grad()
def quantize_weight_per_channel_absmax(w: torch.Tensor, n_bits: int = 8):
    """
    Fake‑quantise a *weight* matrix **per output channel** using abs‑max.

    Parameters
    ----------
    w       : Tensor of shape (out_features, in_features)
    n_bits  : Target bit‑width (default 8 → int8 simulation)

    Returns
    -------
    w       : Quantised‑and‑dequantised tensor (still fp16/fp32)
    """
    # Compute one scale per OUT channel (row) — keep dim for broadcasting
    scales = w.abs().max(dim=-1, keepdim=True)[0]  # (out, 1)

    # Compute max representable integer (for signed int)
    q_max = 2 ** (n_bits - 1) - 1

    # Turn full‑precision scale → integer range; clamp to avoid /0
    scales.clamp_(min=1e-5).div_(q_max)

    # FakeQuant:  w_int = round(w / s),    w_fp = w_int * s
    w.div_(scales).round_().mul_(scales)

    return w


@torch.no_grad()
def quantize_weight_per_tensor_absmax(w: torch.Tensor, n_bits: int = 8):
    """
    Same as above, but *one* scale for the whole tensor (per‑tensor).
    Useful when kernels are small or hardware does not support per‑channel.
    """
    scales = w.abs().max()                         # scalar
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_activation_per_token_absmax(t: torch.Tensor, n_bits: int = 8):
    """
    Per‑token activation FakeQuant (B, T, C … → flatten all but last dim).

    Each *token* (row in the flattened view) gets its own scale so the
    dynamic range adapts along the sequence dimension.
    """
    t_shape = t.shape
    t.view(-1, t_shape[-1])                        # flatten to (N, C)
    scales = t.abs().max(dim=-1, keepdim=True)[0]  # (N, 1)
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t


@torch.no_grad()
def quantize_activation_per_tensor_absmax(t: torch.Tensor, n_bits: int = 8):
    """
    Per‑*tensor* activation FakeQuant – one scale shared by all tokens.
    """
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max()                         # scalar
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t


# ---------------------------------------------------------------------
# 2.  A *drop‑in* nn.Linear replacement that performs W8A8 FakeQuant
# ---------------------------------------------------------------------

class W8A8Linear(nn.Module):
    """
    Weight‑8‑bit / Activation‑8‑bit linear layer.

    *   Weights are quantised once, at construction (`from_float`).
    *   Activations are quantised on‑the‑fly in `forward`.
    *   Optional output FakeQuant simulates int8 inputs to matmuls/BMMs.
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 act_quant: str = "per_token",
                 quantize_output: bool = False):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features

        # Pre‑allocate **quantised** weight / bias as buffers (non‑trainable)
        self.register_buffer(
            "weight",
            torch.randn(out_features, in_features,
                        dtype=torch.float16, requires_grad=False),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(1, out_features,
                            dtype=torch.float16, requires_grad=False),
            )
        else:
            self.register_buffer("bias", None)

        # Choose activation FakeQuant function
        if act_quant == "per_token":
            self.act_quant_name = "per_token"
            self.act_quant = partial(quantize_activation_per_token_absmax,
                                     n_bits=8)
        elif act_quant == "per_tensor":
            self.act_quant_name = "per_tensor"
            self.act_quant = partial(quantize_activation_per_tensor_absmax,
                                     n_bits=8)
        else:
            raise ValueError(f"Invalid act_quant: {act_quant}")

        # Optionally quantise the *output* (for chained matmuls in attention)
        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = "None"
            self.output_quant = lambda x: x  # identity

    # -----------------------------------------------------------------
    # Hugging Face modules often call .to(...) on sub‑modules after
    # conversion; we override to also move inner buffers.
    # -----------------------------------------------------------------
    def to(self, *args, **kwargs):
        super(W8A8Linear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    # -----------------------------------------------------------------
    # Forward pass:  Fake‑quantise input → linear → (optionally) quantise out
    # -----------------------------------------------------------------
    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        q_x = self.act_quant(x)
        y = torch.functional.F.linear(q_x, self.weight, self.bias)
        q_y = self.output_quant(y)
        return q_y

    # -----------------------------------------------------------------
    # Conversion helper: turns a *float* nn.Linear into W8A8Linear
    # -----------------------------------------------------------------
    @staticmethod
    def from_float(module: nn.Linear,
                   weight_quant: str = "per_channel",
                   act_quant: str = "per_token",
                   quantize_output: bool = False):
        assert isinstance(module, nn.Linear)
        new_module = W8A8Linear(module.in_features,
                                module.out_features,
                                module.bias is not None,
                                act_quant=act_quant,
                                quantize_output=quantize_output)

        # Weight FakeQuant (in place) ----------------------------------
        if weight_quant == "per_channel":
            new_module.weight = quantize_weight_per_channel_absmax(
                module.weight, n_bits=8)
        elif weight_quant == "per_tensor":
            new_module.weight = quantize_weight_per_tensor_absmax(
                module.weight, n_bits=8)
        else:
            raise ValueError(f"Invalid weight_quant: {weight_quant}")

        new_module.weight_quant_name = weight_quant

        # Bias is copied verbatim (bias FakeQuant seldom matters)
        if module.bias is not None:
            new_module.bias = module.bias

        return new_module

    def __repr__(self):
        return (f"W8A8Linear({self.in_features}, {self.out_features}, "
                f"bias={self.bias is not None}, "
                f"weight_quant={self.weight_quant_name}, "
                f"act_quant={self.act_quant_name}, "
                f"output_quant={self.output_quant_name})")


# ---------------------------------------------------------------------
# 3.  Architecture‑specific graph rewrites – swap fp32 linears for W8A8
# ---------------------------------------------------------------------

def quantize_opt(model,
                 weight_quant="per_tensor",
                 act_quant="per_tensor",
                 quantize_bmm_input=True):
    """
    Replace Linear layers inside an OPT model with W8A8Linear.

    * If `quantize_bmm_input` is True, outputs of q/k/v projections are
      FakeQuant‑ed so the subsequent matmul operates on int8‑like tensors.
    """
    from transformers.models.opt.modeling_opt import (
        OPTAttention,
        OPTDecoderLayer,
    )

    for name, m in model.model.named_modules():
        if isinstance(m, OPTDecoderLayer):            # FFN layers
            m.fc1 = W8A8Linear.from_float(m.fc1,
                                          weight_quant, act_quant)
            m.fc2 = W8A8Linear.from_float(m.fc2,
                                          weight_quant, act_quant)
        elif isinstance(m, OPTAttention):             # QKV & output proj
            m.q_proj = W8A8Linear.from_float(
                m.q_proj, weight_quant, act_quant,
                quantize_output=quantize_bmm_input)
            m.k_proj = W8A8Linear.from_float(
                m.k_proj, weight_quant, act_quant,
                quantize_output=quantize_bmm_input)
            m.v_proj = W8A8Linear.from_float(
                m.v_proj, weight_quant, act_quant,
                quantize_output=quantize_bmm_input)
            m.out_proj = W8A8Linear.from_float(
                m.out_proj, weight_quant, act_quant)
    return model


def quantize_llama_like(model,
                        weight_quant="per_channel",
                        act_quant="per_token",
                        quantize_bmm_input=False):
    """
    Same rewrite but for Llama / Mistral families (shares module names).
    """
    from transformers.models.llama.modeling_llama import (
        LlamaAttention,
        LlamaMLP,
    )
    from transformers.models.mistral.modeling_mistral import (
        MistralAttention,
        MistralMLP,
    )

    for name, m in model.model.named_modules():
        if isinstance(m, (LlamaMLP, MistralMLP)):
            m.gate_proj = W8A8Linear.from_float(m.gate_proj,
                                                weight_quant, act_quant)
            m.up_proj   = W8A8Linear.from_float(m.up_proj,
                                                weight_quant, act_quant)
            m.down_proj = W8A8Linear.from_float(m.down_proj,
                                                weight_quant, act_quant)

        elif isinstance(m, (LlamaAttention, MistralAttention)):
            m.q_proj = W8A8Linear.from_float(
                m.q_proj, weight_quant, act_quant,
                quantize_output=quantize_bmm_input)
            m.k_proj = W8A8Linear.from_float(
                m.k_proj, weight_quant, act_quant,
                quantize_output=quantize_bmm_input)
            m.v_proj = W8A8Linear.from_float(
                m.v_proj, weight_quant, act_quant,
                quantize_output=quantize_bmm_input)
            m.o_proj = W8A8Linear.from_float(m.o_proj,
                                             weight_quant, act_quant)
    return model


def quantize_mixtral(model,
                     weight_quant="per_channel",
                     act_quant="per_token",
                     quantize_bmm_input=False):
    """
    Mixtral MoE variant – handles extra experts and gate.
    """
    from transformers.models.mixtral.modeling_mixtral import (
        MixtralAttention,
        MixtralSparseMoeBlock,
        MixtralBLockSparseTop2MLP,
    )

    for name, m in model.model.named_modules():
        if isinstance(m, MixtralBLockSparseTop2MLP):  # expert MLP
            m.w1 = W8A8Linear.from_float(m.w1,
                                         weight_quant, act_quant)
            m.w2 = W8A8Linear.from_float(m.w2,
                                         weight_quant, act_quant)
            m.w3 = W8A8Linear.from_float(m.w3,
                                         weight_quant, act_quant)

        elif isinstance(m, MixtralAttention):        # shared attention
            m.q_proj = W8A8Linear.from_float(
                m.q_proj, weight_quant, act_quant,
                quantize_output=quantize_bmm_input)
            m.k_proj = W8A8Linear.from_float(
                m.k_proj, weight_quant, act_quant,
                quantize_output=quantize_bmm_input)
            m.v_proj = W8A8Linear.from_float(
                m.v_proj, weight_quant, act_quant,
                quantize_output=quantize_bmm_input)
            m.o_proj = W8A8Linear.from_float(m.o_proj,
                                             weight_quant, act_quant)

        elif isinstance(m, MixtralSparseMoeBlock):   # router
            m.gate = W8A8Linear.from_float(m.gate,
                                           weight_quant, act_quant)
    return model


def quantize_falcon(model,
                    weight_quant="per_channel",
                    act_quant="per_token",
                    quantize_bmm_input=True):
    """
    Falcon architecture (note: single query_key_value projection).
    """
    from transformers.models.falcon.modeling_falcon import (
        FalconAttention,
        FalconMLP,
    )

    for name, m in model.named_modules():
        if isinstance(m, FalconMLP):
            m.dense_h_to_4h = W8A8Linear.from_float(m.dense_h_to_4h,
                                                    weight_quant, act_quant)
            m.dense_4h_to_h = W8A8Linear.from_float(m.dense_4h_to_h,
                                                    weight_quant, act_quant)

        elif isinstance(m, FalconAttention):
            m.query_key_value = W8A8Linear.from_float(
                m.query_key_value, weight_quant, act_quant,
                quantize_output=quantize_bmm_input)
            m.dense = W8A8Linear.from_float(m.dense,
                                            weight_quant, act_quant)
    return model


# ---------------------------------------------------------------------
# 4.  Front‑door helper – pick rewrite based on model **class**
# ---------------------------------------------------------------------

def quantize_model(model,
                   weight_quant="per_channel",
                   act_quant="per_token",
                   quantize_bmm_input=False):
    """
    Entrypoint: detect family → call appropriate graph‑rewrite.

    Parameters
    ----------
    model               : Hugging Face *pre‑trained* model instance
    weight_quant        : "per_channel" | "per_tensor"
    act_quant           : "per_token"   | "per_tensor"
    quantize_bmm_input  : quantise outputs feeding matmuls (True/False)

    Returns
    -------
    model               : in‑place modified model with W8A8 linears
    """
    from transformers.models.opt.modeling_opt import OPTPreTrainedModel
    from transformers.models.llama.modeling_llama import LlamaPreTrainedModel
    from transformers.models.mistral.modeling_mistral import MistralPreTrainedModel
    from transformers.models.mixtral.modeling_mixtral import MixtralPreTrainedModel
    from transformers.models.falcon.modeling_falcon import FalconPreTrainedModel

    if isinstance(model, OPTPreTrainedModel):
        return quantize_opt(model, weight_quant, act_quant,
                            quantize_bmm_input)
    elif isinstance(model, (LlamaPreTrainedModel, MistralPreTrainedModel)):
        return quantize_llama_like(model, weight_quant, act_quant,
                                   quantize_bmm_input)
    elif isinstance(model, MixtralPreTrainedModel):
        return quantize_mixtral(model, weight_quant, act_quant,
                                quantize_bmm_input)
    elif isinstance(model, FalconPreTrainedModel):
        return quantize_falcon(model, weight_quant, act_quant,
                               quantize_bmm_input)
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

# ---------------------------------------------------------------------
# End of file
# ---------------------------------------------------------------------
