"""Model/tokenizer construction: load base LM, attach LoRA, wrap with edge masks."""

from __future__ import annotations

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import TrainConfig
from ..masking.edge_masked_model import EdgeMaskedTransformer

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

# GPT-2's attention/MLP projections are Conv1D under different names than Llama's.
GPT2_LORA_TARGETS = ["c_attn", "c_proj", "c_fc"]


def _disable_dropout(module) -> None:
    """Zero every Dropout so the clean/full/masked forwards are deterministic
    (GPT-2 ships resid/attn/embd dropout; KL-to-original needs no sampling noise)."""
    for m in module.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0


def load_tokenizer(model_path: str):
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def build_masked_model(cfg: TrainConfig, device: str) -> EdgeMaskedTransformer:
    dtype = DTYPES[cfg.dtype] if device.startswith("cuda") else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        torch_dtype=dtype,
        attn_implementation="eager",  # ensures the additive 4D mask path is used
    )
    _disable_dropout(base)
    # GPT-2 (the IOI rig) needs its own LoRA target names; Llama/Qwen use the default.
    is_gpt2 = hasattr(base, "transformer") and hasattr(base.transformer, "h")
    target_modules = GPT2_LORA_TARGETS if is_gpt2 else cfg.lora.target_modules
    lora_cfg = LoraConfig(
        r=cfg.lora.rank,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base, lora_cfg)
    masked = EdgeMaskedTransformer(peft_model, init_open_prob=cfg.init_open_prob,
                                   granularity=cfg.graph_granularity)
    return masked.to(device)


def split_parameters(masked_model: EdgeMaskedTransformer):
    """Return (lora_params, gate_params) for separate optimizer groups."""
    gate_params = list(masked_model.gate.parameters())
    gate_ids = {id(p) for p in gate_params}
    lora_params = [
        p for p in masked_model.peft_model.parameters()
        if p.requires_grad and id(p) not in gate_ids
    ]
    return lora_params, gate_params
