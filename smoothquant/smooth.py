import torch
import torch.nn as nn

# Decoder‑layer classes for all architectures we plan to support
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.bloom.modeling_bloom import BloomBlock
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.models.mistral.modeling_mistral import (
    MistralDecoderLayer,
    MistralRMSNorm,
)
from transformers.models.mixtral.modeling_mixtral import (
    MixtralDecoderLayer,
    MixtralRMSNorm,
)
from transformers.models.falcon.modeling_falcon import FalconDecoderLayer


# ---------------------------------------------------------------------
# 1. Low‑level helpers:  “smooth” a LayerNorm (or RMSNorm) together with
#    one or more incoming Linear layers.
# ---------------------------------------------------------------------

@torch.no_grad()
def smooth_ln_fcs(ln: nn.LayerNorm,
                  fcs,
                  act_scales: torch.Tensor,
                  alpha: float = 0.5):
    """
    Generic version for *LayerNorm*‑based models (OPT, BLOOM, Falcon, …).

    Args
    ----
    ln          : the LayerNorm whose parameters will be re‑scaled
    fcs         : a `nn.Linear` **or** list of linears that feed *into* `ln`
    act_scales  : per‑channel activation scales (obtained offline)
    alpha       : balancing factor between activation and weight scales
    """
    # Normalise the input to a list so later code can always iterate
    if not isinstance(fcs, list):
        fcs = [fcs]

    # --- sanity checks ------------------------------------------------
    assert isinstance(ln, nn.LayerNorm)
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        # shape agreement: LayerNorm and Linear must share feature dim
        assert ln.weight.numel() == fc.in_features == act_scales.numel()

    # Work in the same device/dtype as model weights
    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Find a *per‑input‑channel* max‑abs value across **all** connected
    # linear layers – this becomes the "weight scale" for each channel.
    # ------------------------------------------------------------------
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)  # avoid /0

    # Final smoothing factor for each channel
    #   scale = act_scale^α / weight_scale^(1‑α)
    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )

    # ------------------- apply the re‑scaling -------------------------
    # Divide LN parameters     (so subsequent normalisation is weaker)
    ln.weight.div_(scales)
    ln.bias.div_(scales)

    # Multiply incoming weights (so overall forward pass is unchanged)
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))


@torch.no_grad()
def smooth_ln_fcs_llama_like(ln,
                             fcs,
                             act_scales: torch.Tensor,
                             alpha: float = 0.5):
    """
    Version for *RMSNorm*‑based families (Llama, Mistral, Mixtral).

    RMSNorm has *weight* but no *bias*, so we skip the bias correction.
    Remaining logic is identical to `smooth_ln_fcs`.
    """
    if not isinstance(fcs, list):
        fcs = [fcs]

    # RMSNorm subclasses accepted
    assert isinstance(ln, (LlamaRMSNorm, MistralRMSNorm, MixtralRMSNorm))
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)

    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)

    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )

    # RMSNorm: only `weight`, no `bias`
    ln.weight.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))


# ---------------------------------------------------------------------
# 2. High‑level dispatcher: walk through *any* supported model and
#    smooth every (LayerNorm, Linear‑block) pair it finds.
# ---------------------------------------------------------------------

@torch.no_grad()
def smooth_lm(model: nn.Module,
              scales: dict[str, torch.Tensor],
              alpha: float = 0.5):
    """
    Apply SmoothQuant to an *entire* decoder‑only language model.

    Parameters
    ----------
    model   : HF model instance (OPT, BLOOM, Falcon, Llama, …)
    scales  : dict mapping **fully‑qualified layer names** to
              activation‑scale tensors (pre‑computed by RTN, GPTQ, etc.)
    alpha   : 0≤α≤1 trade‑off between activation and weight scaling
    """
    for name, module in model.named_modules():

        # -----------------------------------------------------------------
        # OPT
        # -----------------------------------------------------------------
        if isinstance(module, OPTDecoderLayer):
            # Attention block ------------------------------------------------
            attn_ln = module.self_attn_layer_norm
            qkv = [module.self_attn.q_proj,
                   module.self_attn.k_proj,
                   module.self_attn.v_proj]
            qkv_input_scales = scales[name + ".self_attn.q_proj"]
            smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)

            # FFN block ------------------------------------------------------
            ffn_ln = module.final_layer_norm
            fc1 = module.fc1
            fc1_input_scales = scales[name + ".fc1"]
            smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)

        # -----------------------------------------------------------------
        # BLOOM
        # -----------------------------------------------------------------
        elif isinstance(module, BloomBlock):
            attn_ln = module.input_layernorm
            qkv = module.self_attention.query_key_value
            qkv_input_scales = scales[name + ".self_attention.query_key_value"]
            smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)

            ffn_ln = module.post_attention_layernorm
            fc1 = module.mlp.dense_h_to_4h
            fc1_input_scales = scales[name + ".mlp.dense_h_to_4h"]
            smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)

        # -----------------------------------------------------------------
        # Falcon (two possible architectures)
        # -----------------------------------------------------------------
        elif isinstance(module, FalconDecoderLayer):
            qkv = module.self_attention.query_key_value
            qkv_input_scales = scales[name + ".self_attention.query_key_value"]

            fc1 = module.mlp.dense_h_to_4h
            fc1_input_scales = scales[name + ".mlp.dense_h_to_4h"]

            if (not module.config.new_decoder_architecture
                    and module.config.parallel_attn):
                # Old, *parallel* architecture: single LayerNorm feeds both
                attn_ln = module.input_layernorm
                smooth_ln_fcs(attn_ln, [qkv, fc1], qkv_input_scales, alpha)
            else:
                # New or *serial* architecture: separate norms
                attn_ln = (module.ln_attn if module.config.new_decoder_architecture
                           else module.input_layernorm)
                ffn_ln = (module.ln_mlp if module.config.new_decoder_architecture
                          else module.post_attention_layernorm)

                smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)
                smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)

        # -----------------------------------------------------------------
        # Llama & Mistral (RMSNorm)
        # -----------------------------------------------------------------
        elif isinstance(module, (LlamaDecoderLayer, MistralDecoderLayer)):
            # Attention ------------------------------------------------------
            attn_ln = module.input_layernorm
            qkv = [module.self_attn.q_proj,
                   module.self_attn.k_proj,
                   module.self_attn.v_proj]
            qkv_input_scales = scales[name + ".self_attn.q_proj"]
            smooth_ln_fcs_llama_like(attn_ln, qkv, qkv_input_scales, alpha)

            # Feed‑forward ---------------------------------------------------
            ffn_ln = module.post_attention_layernorm
            fcs = [module.mlp.gate_proj, module.mlp.up_proj]
            fcs_input_scales = scales[name + ".mlp.gate_proj"]
            smooth_ln_fcs_llama_like(ffn_ln, fcs, fcs_input_scales, alpha)

        # -----------------------------------------------------------------
        # Mixtral (MoE – multiple experts)
        # -----------------------------------------------------------------
        elif isinstance(module, MixtralDecoderLayer):
            # Attention ------------------------------------------------------
            attn_ln = module.input_layernorm
            qkv = [module.self_attn.q_proj,
                   module.self_attn.k_proj,
                   module.self_attn.v_proj]
            qkv_input_scales = scales[name + ".self_attn.q_proj"]
            smooth_ln_fcs_llama_like(attn_ln, qkv, qkv_input_scales, alpha)

            # MoE feed‑forward ----------------------------------------------
            ffn_ln = module.post_attention_layernorm
            fcs = [module.block_sparse_moe.gate]                      # router
            for expert in module.block_sparse_moe.experts:            # add w1/w3 for *every* expert
                fcs.append(expert.w1)
                fcs.append(expert.w3)
            fcs_input_scales = scales[name + ".block_sparse_moe.gate"]
            smooth_ln_fcs_llama_like(ffn_ln, fcs, fcs_input_scales, alpha)

# ---------------------------------------------------------------------
# End of file – after calling `smooth_lm(model, scales)`, every relevant
# LayerNorm (or RMSNorm) in `model` will be merged with its incoming
# Linear weights, improving post‑training quantisation accuracy.
# ---------------------------------------------------------------------
