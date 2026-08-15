"""Edge-masked transformer: the disentangled residual stream + edge gates.

Standard Edge Pruning (Bhaskar 2024) freezes the model and learns a mask that
*finds* a circuit. We invert it: the model's LoRA weights and the edge gates
train together, so the model reorganises itself until a sparse set of edges
reproduces its behaviour (joint co-training, doc 1 Option A).

Mechanism (the "disentangled residual stream"): instead of the native
``residual = residual + block(x)`` accumulation, every reader rebuilds its input
from the *individual contributions* of upstream writers, each blended by its
edge gate::

    reader_input_r = sum_{w upstream of r}  z_{w,r} * clean_w + (1 - z_{w,r}) * corrupted_w

An edge turned OFF (z=0) therefore feeds the writer's *corrupted-input*
contribution, not zero, keeping the model in-distribution.

This is implemented by driving each layer's attention / MLP submodules directly
(the native residual addition would otherwise re-mix the stream). The only
architecture-specific part is a small **adapter** (``_build_adapter``) that binds
the model's embed / attention / MLP / final-head calls; the disentangled-stream
logic below is identical for every architecture. Supported: Llama / Qwen2
(rotary, ``self_attn``/``mlp`` blocks) and GPT-2 (learned positional embeddings,
``attn``/``mlp`` blocks).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn

from .graph import CircuitGraph
from .hard_concrete import HardConcreteGate


@dataclass
class _Adapter:
    """Architecture-specific bindings; everything else in the wrapper is generic.

    layers         : the list of decoder blocks (one attn + one mlp each)
    embed          : input_ids -> the "embed" writer contribution (b, s, d)
    position       : hidden -> position info passed to run_attn (rotary tuple or None)
    run_attn       : (layer, x, position, mask) -> attention block output
    run_attn_heads : (layer, x, position, mask) -> list of per-head residual-space
                     contributions whose sum equals run_attn's output. ``None`` if
                     head-level granularity is unsupported for this architecture.
    run_mlp        : (layer, x) -> MLP block output
    final          : x -> logits (final norm + unembedding)
    n_heads        : number of attention heads per layer
    """

    layers: nn.ModuleList
    embed: Callable
    position: Callable
    run_attn: Callable
    run_mlp: Callable
    final: Callable
    n_heads: int
    run_attn_heads: Callable | None = None


def _rope_per_head_contribs(attn, x_normed, pos, mask, extra_scale=None):
    """Per-head residual-stream contributions for rotary GQA attention (Llama / Qwen3 / Gemma-3).

    Recomputes attention head-by-head so each head's write through its slice of ``o_proj`` is a
    separate writer; the sum of the returned contributions equals the block's attention output
    exactly. Handles the three family quirks: (1) GQA (fewer KV heads than Q heads, expanded via
    repeat_interleave); (2) per-head q/k RMSNorm (Qwen3, Gemma-3) applied BEFORE rotary, mirrored
    from the HF modules; (3) ``head_dim`` that need not equal hidden/n_heads (Gemma-3), so o_proj
    input slices use ``attn.head_dim``. All projections are called as modules, so LoRA is applied.
    ``extra_scale`` (b, s, d) multiplies every contribution elementwise; Gemma-3 uses it to fold
    its post-attention RMSNorm exactly (see the Gemma branch).
    """
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

    b, s, _ = x_normed.shape
    H = attn.config.num_attention_heads
    KV = attn.config.num_key_value_heads
    hd = attn.head_dim

    q = attn.q_proj(x_normed).view(b, s, H, hd)
    k = attn.k_proj(x_normed).view(b, s, KV, hd)
    v = attn.v_proj(x_normed).view(b, s, KV, hd)
    if hasattr(attn, "q_norm"):                      # Qwen3 / Gemma-3 per-head norms
        q = attn.q_norm(q)
    if hasattr(attn, "k_norm"):
        k = attn.k_norm(k)
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (b, h, s, hd)
    cos, sin = pos
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    k, v = repeat_kv(k, H // KV), repeat_kv(v, H // KV)

    scores = (q @ k.transpose(-1, -2)) * attn.scaling
    scores = scores + mask[:, :, :, : k.shape[-2]]
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    ctx = probs @ v                                   # (b, H, s, hd)
    concat = ctx.transpose(1, 2).reshape(b, s, H * hd)

    o_proj = attn.o_proj
    bias_mod = o_proj.base_layer if hasattr(o_proj, "base_layer") else o_proj
    bias = getattr(bias_mod, "bias", None)
    contribs = []
    for h in range(H):
        z = torch.zeros_like(concat)
        z[..., h * hd:(h + 1) * hd] = concat[..., h * hd:(h + 1) * hd]
        oh = o_proj(z)
        if bias is not None:
            oh = oh - bias
        contribs.append(oh)
    if bias is not None:
        contribs[0] = contribs[0] + bias
    if extra_scale is not None:
        contribs = [c * extra_scale for c in contribs]
    return contribs


def _build_adapter(base) -> _Adapter:
    """Bind the model-specific calls for the supported architectures."""
    # Gemma-3 (checked BEFORE the generic Llama branch: Gemma3ForCausalLM also has
    # base.model.layers, so it would silently match the Llama branch and get the WRONG
    # decomposition). Gemma's block differs from Llama in three ways the decomposition must
    # respect: (1) the attention output is wrapped in post_attention_layernorm and the MLP
    # is wrapped in pre_feedforward_layernorm ... post_feedforward_layernorm (so the ADDITIVE
    # writer contribution is the norm-wrapped output, not the raw attn/mlp output);
    # (2) sliding vs global layers use DIFFERENT rotary bases (rope_theta vs
    # rope_local_base_freq) -- routed per layer via layer.self_attn.layer_type; (3) the
    # embedding is scaled by sqrt(hidden_size), but that is baked into embed_tokens
    # (Gemma3TextScaledWordEmbedding), so embed_tokens(ids) is already correct. The sliding-
    # window mask == the full causal mask for our short prompts (seq_len << sliding_window
    # of 512), so a single causal mask serves both layer types. Validated by the
    # reconstruction parity check in reconstruction_max_error(); do not trust this adapter
    # for a new Gemma variant without re-running that check.
    _gemma_text = None
    if hasattr(base, "model"):
        if hasattr(base.model, "layers"):
            _gemma_text = base.model                     # Gemma3ForCausalLM (text-only, e.g. 1b)
        elif hasattr(base.model, "language_model") and hasattr(base.model.language_model, "layers"):
            _gemma_text = base.model.language_model      # Gemma3ForConditionalGeneration (multimodal, e.g. 4b)
    if _gemma_text is not None and hasattr(_gemma_text.layers[0], "pre_feedforward_layernorm"):
        inner, lm_head = _gemma_text, base.lm_head
        rotary = inner.rotary_emb
        layer_types = set(inner.config.layer_types)

        def position(hidden):
            b, s, _ = hidden.shape
            pos_ids = torch.arange(s, device=hidden.device).unsqueeze(0).expand(b, -1)
            return {lt: rotary(hidden, pos_ids, lt) for lt in layer_types}

        def run_attn(layer, x, pos, mask):
            out = layer.self_attn(hidden_states=layer.input_layernorm(x),
                                  position_embeddings=pos[layer.self_attn.layer_type],
                                  attention_mask=mask)
            a = out[0] if isinstance(out, tuple) else out
            return layer.post_attention_layernorm(a)

        def run_attn_heads(layer, x, pos, mask):
            """Per-head contributions whose sum equals run_attn's output EXACTLY, despite the
            post-attention RMSNorm. RMSNorm at the operating point is an elementwise rescale of
            its input, and the factor has a closed form that needs NO division by the input:
            norm(t) = t * rsqrt(mean(t^2) + eps) * (1 + gamma). We compute the raw per-head
            o_proj writes, form their total, compute that analytic factor from the total, and
            apply it to every head. The earlier implementation computed the factor as
            norm(total)/total elementwise; its backward contains 1/total^2, which explodes on
            near-zero coordinates and poisoned the controller's integrated-gradient ranking with
            NaNs (the head-level Gemma cliff). The analytic form is exact and gradient-safe."""
            attn = layer.self_attn
            xn = layer.input_layernorm(x)
            raw = _rope_per_head_contribs(attn, xn, pos[attn.layer_type], mask)
            total = raw[0]
            for c in raw[1:]:
                total = total + c
            ln = layer.post_attention_layernorm
            var = total.float().pow(2).mean(-1, keepdim=True)
            scale = (torch.rsqrt(var + ln.eps) * (1.0 + ln.weight.float())).to(total.dtype)
            return [c * scale for c in raw]

        return _Adapter(
            layers=inner.layers,
            embed=lambda ids: inner.embed_tokens(ids),  # sqrt(d) scale baked in (ScaledWordEmbedding)
            position=position,
            run_attn=run_attn,
            run_mlp=lambda layer, x: layer.post_feedforward_layernorm(
                layer.mlp(layer.pre_feedforward_layernorm(x))),
            final=lambda x: lm_head(inner.norm(x)),
            n_heads=inner.config.num_attention_heads,  # inner.config = text config (works for 1b & multimodal 4b)
            run_attn_heads=run_attn_heads,
        )

    # Llama / Qwen2: base.model.{embed_tokens, layers, norm, rotary_emb}, base.lm_head.
    if hasattr(base, "model") and hasattr(base.model, "layers"):
        inner, lm_head = base.model, base.lm_head
        rotary = getattr(inner, "rotary_emb", None)

        def position(hidden):
            if rotary is None:
                return None
            b, s, _ = hidden.shape
            pos_ids = torch.arange(s, device=hidden.device).unsqueeze(0).expand(b, -1)
            return rotary(hidden, pos_ids)

        def run_attn(layer, x, pos, mask):
            out = layer.self_attn(hidden_states=layer.input_layernorm(x),
                                  position_embeddings=pos, attention_mask=mask)
            return out[0] if isinstance(out, tuple) else out

        def run_attn_heads(layer, x, pos, mask):
            """Per-head o_proj writes (sum == run_attn output). Handles Llama GQA and Qwen3's
            per-head q/k RMSNorm inside _rope_per_head_contribs; the residual write here is the
            raw attention output (no post-norm wrap), so the sum is additive with no folding."""
            return _rope_per_head_contribs(layer.self_attn, layer.input_layernorm(x), pos, mask)

        return _Adapter(
            layers=inner.layers,
            embed=lambda ids: inner.embed_tokens(ids),
            position=position,
            run_attn=run_attn,
            run_mlp=lambda layer, x: layer.mlp(layer.post_attention_layernorm(x)),
            final=lambda x: lm_head(inner.norm(x)),
            n_heads=base.config.num_attention_heads,
            run_attn_heads=run_attn_heads,
        )

    # GPT-2: base.transformer.{wte, wpe, h, ln_f}, base.lm_head. Positions are
    # learned and folded into the embed contribution; no rotary.
    if hasattr(base, "transformer") and hasattr(base.transformer, "h"):
        inner, lm_head = base.transformer, base.lm_head

        def embed(ids):
            b, s = ids.shape
            pos_ids = torch.arange(s, device=ids.device).unsqueeze(0).expand(b, -1)
            return inner.wte(ids) + inner.wpe(pos_ids)

        n_heads = base.config.n_head

        def run_attn(layer, x, pos, mask):
            out = layer.attn(layer.ln_1(x), attention_mask=mask)
            return out[0] if isinstance(out, tuple) else out

        def run_attn_heads(layer, x, pos, mask):
            """Per-head contributions to the residual stream (sum == run_attn).

            We recompute GPT-2 attention head-by-head so each head's write through
            its slice of the output projection ``c_proj`` is a separate writer. The
            output projection is affine (``z @ W + b``); summing the per-head linear
            parts and adding the bias once reproduces the block output exactly.
            """
            attn = layer.attn
            qkv = attn.c_attn(layer.ln_1(x))                 # (b, s, 3d), LoRA-aware
            d = qkv.shape[-1] // 3
            q, k, v = qkv.split(d, dim=2)
            hd = d // n_heads
            b, s, _ = q.shape

            def heads(t):
                return t.view(b, s, n_heads, hd).permute(0, 2, 1, 3)  # (b, nh, s, hd)

            q, k, v = heads(q), heads(k), heads(v)
            scores = (q @ k.transpose(-1, -2)) / (hd ** 0.5)
            scores = scores + mask                            # causal + pad (additive)
            probs = torch.softmax(scores, dim=-1)
            ctx = probs @ v                                   # (b, nh, s, hd)
            concat = ctx.permute(0, 2, 1, 3).reshape(b, s, d)

            c_proj = attn.c_proj
            bias_mod = c_proj.base_layer if hasattr(c_proj, "base_layer") else c_proj
            bias = getattr(bias_mod, "bias", None)
            contribs = []
            for h in range(n_heads):
                z = torch.zeros_like(concat)
                z[..., h * hd:(h + 1) * hd] = concat[..., h * hd:(h + 1) * hd]
                oh = c_proj(z)
                contribs.append(oh - bias if bias is not None else oh)
            if bias is not None:
                contribs[0] = contribs[0] + bias              # add the single bias once
            return contribs

        return _Adapter(
            layers=inner.h,
            embed=embed,
            position=lambda hidden: None,
            run_attn=run_attn,
            run_mlp=lambda layer, x: layer.mlp(layer.ln_2(x)),
            final=lambda x: lm_head(inner.ln_f(x)),
            n_heads=n_heads,
            run_attn_heads=run_attn_heads,
        )

    raise ValueError(
        f"Unsupported architecture {type(base).__name__}: edge masking supports "
        "Llama/Qwen2 (model.layers) and GPT-2 (transformer.h)."
    )


def _build_causal_mask(attention_mask_2d: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """4D additive mask (b, 1, s, s): 0 where attendable, -inf for future/pad."""
    b, s = attention_mask_2d.shape
    min_val = torch.finfo(dtype).min
    causal = torch.triu(torch.full((s, s), min_val, device=attention_mask_2d.device), diagonal=1)
    mask = causal[None, None].expand(b, 1, s, s).clone()
    pad = (1.0 - attention_mask_2d[:, None, None, :].to(dtype)) * min_val
    return (mask + pad).to(dtype)


class EdgeMaskedTransformer(nn.Module):
    """Wraps a (LoRA) causal LM with an Edge-Pruning disentangled residual stream.

    Args:
        peft_model: a ``peft`` model wrapping a Llama/Qwen ``*ForCausalLM``.
        init_open_prob: starting open probability for every edge gate.
        granularity: "block" (one attn writer/layer) or "head" (one writer/head).
    """

    def __init__(self, peft_model, init_open_prob: float = 0.95, granularity: str = "block"):
        super().__init__()
        self.peft_model = peft_model
        base = peft_model.get_base_model() if hasattr(peft_model, "get_base_model") else peft_model
        self._base = base
        self._adapter = _build_adapter(base)
        self._layers = self._adapter.layers
        self.granularity = granularity
        if granularity == "head" and self._adapter.run_attn_heads is None:
            raise ValueError("head-level granularity is not supported for this architecture yet")

        self.graph = CircuitGraph(num_layers=len(self._layers),
                                  n_heads=self._adapter.n_heads, granularity=granularity)
        self.edges = self.graph.edges()
        self.edge_index = {e: i for i, e in enumerate(self.edges)}
        # Precompute, per reader, its incoming (writer, edge_index) list.
        self.reader_in: dict[str, list[tuple[str, int]]] = {
            r: [(w, self.edge_index[(w, r)]) for w in self.graph.upstream_writers(r)]
            for r in self.graph.readers
        }
        self.gate = HardConcreteGate(len(self.edges), init_open_prob)

    # -- helpers (delegate the architecture-specific calls to the adapter) ------

    def _position_embeddings(self, hidden: torch.Tensor):
        return self._adapter.position(hidden)

    def _run_attn(self, layer, x, pos_emb, mask):
        return self._adapter.run_attn(layer, x, pos_emb, mask)

    def _attn_writers(self, layer, i, x, pos_emb, mask) -> dict[str, torch.Tensor]:
        """Layer ``i``'s attention contribution(s), keyed by writer name. One entry
        ``attn_{i}`` in block mode; ``attn_{i}_h{j}`` per head in head mode."""
        if self.granularity == "head":
            heads = self._adapter.run_attn_heads(layer, x, pos_emb, mask)
            return {f"attn_{i}_h{j}": h for j, h in enumerate(heads)}
        return {f"attn_{i}": self._run_attn(layer, x, pos_emb, mask)}

    def _run_mlp(self, layer, x):
        return self._adapter.run_mlp(layer, x)

    # -- forward passes --------------------------------------------------------

    @torch.no_grad()
    def capture_contributions(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        """Plain forward; return each writer's additive contribution (detached).

        Used on the *corrupted* input to obtain the ablation values fed to OFF
        edges. Detached: the corrupted run is treated as a fixed baseline.
        """
        contrib: dict[str, torch.Tensor] = {}
        embed = self._adapter.embed(input_ids)
        contrib["embed"] = embed
        pos_emb = self._position_embeddings(embed)
        mask = _build_causal_mask(attention_mask, embed.dtype)

        residual = embed
        for i, layer in enumerate(self._layers):
            attn_writers = self._attn_writers(layer, i, residual, pos_emb, mask)
            contrib.update(attn_writers)
            residual = residual + sum(attn_writers.values())   # == block attn output
            mlp_out = self._run_mlp(layer, residual)
            contrib[f"mlp_{i}"] = mlp_out
            residual = residual + mlp_out
        return contrib

    def _reader_input(self, reader, clean, corrupted, gates):
        """Disentangled residual input for one reader."""
        acc = None
        for writer, eidx in self.reader_in[reader]:
            cw = clean[writer]
            if corrupted is None:
                term = cw
            else:
                z = gates[eidx]
                term = z * cw + (1.0 - z) * corrupted[writer]
            acc = term if acc is None else acc + term
        return acc

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        corrupted: dict | None = None,
        masked: bool = True,
    ) -> torch.Tensor:
        """Disentangled forward returning logits.

        masked=True with ``corrupted`` given: blend clean/corrupted per edge gate
        (the trimmed circuit). masked=False: plain full model (all edges ON).
        """
        gates = self.gate.sample(self.training) if masked else None
        clean: dict[str, torch.Tensor] = {}
        embed = self._adapter.embed(input_ids)
        clean["embed"] = embed
        pos_emb = self._position_embeddings(embed)
        mask = _build_causal_mask(attention_mask, embed.dtype)

        use_corr = corrupted if masked else None
        for i, layer in enumerate(self._layers):
            x_attn = self._reader_input(f"attn_{i}", clean, use_corr, gates)
            clean.update(self._attn_writers(layer, i, x_attn, pos_emb, mask))
            x_mlp = self._reader_input(f"mlp_{i}", clean, use_corr, gates)
            clean[f"mlp_{i}"] = self._run_mlp(layer, x_mlp)

        x_logits = self._reader_input("logits", clean, use_corr, gates)
        return self._adapter.final(x_logits)

    @torch.no_grad()
    def reconstruction_max_error(self, input_ids: torch.Tensor,
                                 attention_mask: torch.Tensor) -> float:
        """Parity check for the architecture adapter: the disentangled all-edges-ON forward
        must reproduce the underlying model's OWN logits (both include LoRA identically, so
        any gap is pure decomposition error -- e.g. a wrong norm placement or rotary). The
        additive residual-stream decomposition is only valid if this is ~0 (fp32 noise). Run
        this whenever adding a new architecture branch to _build_adapter."""
        got = self.forward(input_ids, attention_mask, corrupted=None, masked=False)
        ref = self._base(input_ids=input_ids, attention_mask=attention_mask).logits
        return float((got - ref).abs().max())

    # -- edge attribution patching (EAP), evaluation only ----------------------

    def edge_attributions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        corrupted: dict,
        metric_fn,
    ) -> torch.Tensor:
        """First-order EAP score per edge: (clean_w - corrupted_w) . d(metric)/d(reader_in_r).

        ``metric_fn(logits) -> scalar``. Returns a (num_edges,) tensor of absolute
        attributions over *all* edges. Evaluation only (it *measures* concentration;
        it is never the training signal).

        The forward runs through the **shaped circuit**: each reader input is the
        per-edge gate blend ``z*clean + (1-z)*corrupted`` that ``forward(masked=True)``
        uses, with the eval-time deterministic gates. This makes the attribution (and
        thus the concentration panel) reflect the trimmed circuit we trained, not the
        original full model — so the panel moves as gates close. Gates are detached
        (eval only) so this never writes into the gate parameters' gradients.
        """
        gates = self.gate.sample(self.training).detach()
        clean: dict[str, torch.Tensor] = {}
        reader_inputs: dict[str, torch.Tensor] = {}
        embed = self._adapter.embed(input_ids)
        # The embedding/base weights are frozen, so make the embedding a leaf that
        # requires grad; this lets every downstream reader input be differentiated
        # for the attribution dot-product.
        embed = embed.detach().requires_grad_(True)
        clean["embed"] = embed
        pos_emb = self._position_embeddings(embed)
        mask = _build_causal_mask(attention_mask, embed.dtype)

        for i, layer in enumerate(self._layers):
            x_attn = self._reader_input(f"attn_{i}", clean, corrupted, gates)
            x_attn.retain_grad()
            reader_inputs[f"attn_{i}"] = x_attn
            clean.update(self._attn_writers(layer, i, x_attn, pos_emb, mask))
            x_mlp = self._reader_input(f"mlp_{i}", clean, corrupted, gates)
            x_mlp.retain_grad()
            reader_inputs[f"mlp_{i}"] = x_mlp
            clean[f"mlp_{i}"] = self._run_mlp(layer, x_mlp)

        x_logits = self._reader_input("logits", clean, corrupted, gates)
        x_logits.retain_grad()
        reader_inputs["logits"] = x_logits
        logits = self._adapter.final(x_logits)

        metric = metric_fn(logits)
        self.zero_grad(set_to_none=True)
        metric.backward()

        scores = torch.zeros(len(self.edges))
        for (writer, reader), eidx in self.edge_index.items():
            grad = reader_inputs[reader].grad
            if grad is None:
                continue
            delta = (clean[writer] - corrupted[writer]).detach()
            scores[eidx] = (delta * grad).sum().abs().item()
        return scores

    def edge_attributions_ig(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        corrupted: dict,
        metric_fn,
        n_steps: int = 5,
        path_delta: bool = False,
    ) -> torch.Tensor:
        """EAP-IG: integrated-gradients version of ``edge_attributions`` (Hanna et al./
        Marks et al.-style; see EAP-IG / EAP-GP literature, e.g. arxiv 2502.06852).

        ``path_delta`` selects the integration variant. False (default, canonical EAP-IG):
        the (clean - corrupted) delta is FIXED at alpha=1 and only the gradient varies along
        the path. True (an EAP-GP-STYLE approximation, not a verbatim reimpl of 2502.06852):
        the delta is recomputed from the writer activations AT each alpha, so both the delta
        and the gradient follow the model-dependent path -- a path-integrated attribution that
        the discovery baseline exposes as a distinct method.

        Plain EAP takes ONE gradient sample -- at the current shaped-circuit point -- and
        multiplies by the full (clean - corrupted) delta. If the gradient happens to be
        near-zero at exactly that point (a saturated region), the edge is scored as
        unimportant even when it has a large true effect elsewhere along the clean/
        corrupted range -- a documented failure mode (EAP can even produce EMPTY circuits
        at high sparsity because of it). EAP-IG fixes this by averaging the gradient over
        ``n_steps`` points along the path from "fully corrupted" (every edge gate at 0) to
        the CURRENT shaped circuit (every edge gate at its real, trained value) -- i.e. we
        scale the real per-edge gate tensor by alpha in (0, 1] instead of using it directly,
        re-running the full forward+backward at each alpha, and average. delta is computed
        ONCE at alpha=1 (the real forward), matching the fixed-delta/varying-gradient form
        of the integrated-gradients estimator. Costs ~n_steps+1 forward/backward passes
        instead of 1 -- fine for a periodic ranking step, not meant for the training loop.
        """
        real_gates = self.gate.sample(self.training).detach()

        def forward_at(alpha: float, need_grad: bool):
            gates_a = real_gates * alpha
            clean: dict[str, torch.Tensor] = {}
            reader_inputs: dict[str, torch.Tensor] = {}
            embed = self._adapter.embed(input_ids)
            embed = embed.detach().requires_grad_(need_grad)
            clean["embed"] = embed
            pos_emb = self._position_embeddings(embed)
            mask = _build_causal_mask(attention_mask, embed.dtype)
            for i, layer in enumerate(self._layers):
                x_attn = self._reader_input(f"attn_{i}", clean, corrupted, gates_a)
                if need_grad:
                    x_attn.retain_grad()
                reader_inputs[f"attn_{i}"] = x_attn
                clean.update(self._attn_writers(layer, i, x_attn, pos_emb, mask))
                x_mlp = self._reader_input(f"mlp_{i}", clean, corrupted, gates_a)
                if need_grad:
                    x_mlp.retain_grad()
                reader_inputs[f"mlp_{i}"] = x_mlp
                clean[f"mlp_{i}"] = self._run_mlp(layer, x_mlp)
            x_logits = self._reader_input("logits", clean, corrupted, gates_a)
            if need_grad:
                x_logits.retain_grad()
            reader_inputs["logits"] = x_logits
            logits = self._adapter.final(x_logits)
            return clean, reader_inputs, logits

        # Fixed delta term, taken at alpha=1 (the exact current shaped circuit) -- no
        # grad needed here, we only read off the writer activations.
        with torch.no_grad():
            clean_full, _, _ = forward_at(1.0, need_grad=False)

        scores_sum = torch.zeros(len(self.edges))
        for j in range(1, n_steps + 1):
            alpha_j = j / n_steps
            clean_j, reader_inputs_j, logits_j = forward_at(alpha_j, need_grad=True)
            metric = metric_fn(logits_j)
            self.zero_grad(set_to_none=True)
            metric.backward()
            delta_src = clean_j if path_delta else clean_full
            for (writer, reader), eidx in self.edge_index.items():
                grad = reader_inputs_j[reader].grad
                if grad is None:
                    continue
                delta = (delta_src[writer] - corrupted[writer]).detach()
                scores_sum[eidx] += (delta * grad).sum().item()
        return (scores_sum / n_steps).abs()

    def edge_attributions_gp(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        corrupted: dict,
        metric_fn,
        n_steps: int = 5,
    ) -> torch.Tensor:
        """EAP-GP (Zhang et al. 2025, arXiv 2502.06852) -- FROM-EQUATIONS reimplementation.

        There is no official EAP-GP source release, so this follows the paper's equations
        (Eq. 6/9 + the GradPath update Eq. 7-8), not reference code. EAP-GP is EAP-IG with an
        ADAPTIVE integration path ("GradPath"): instead of the uniform corrupted->clean line,
        the path point is moved by gradient descent to minimise ||G(gamma) - G(corrupted)||^2
        (G = the model's output logits), taking a step of size 1/||grad|| each iteration so the
        path steers AROUND saturated regions where EAP's single gradient is uninformative.

        Framework adaptation (documented, our one deviation): the paper's path lives in raw
        activation space; we realise it in the per-EDGE interpolation-coefficient space -- a
        vector ``coef`` in [0,1]^E where the reader blend for edge e is
        ``coef[e]*gate[e]*clean + (1 - coef[e]*gate[e])*corrupted``. coef=1 is the clean
        circuit, coef=0 fully corrupted; each edge may interpolate independently, so GradPath
        can leave the uniform line -- the natural free path in the disentangled model. k=5
        matches the paper. The attribution (Eq. 9) is the fixed (clean-corrupted) delta times
        the metric gradient at each visited path point, averaged.
        """
        eps = 1e-8
        real_gates = self.gate.sample(self.training).detach()

        def forward_coef(coef, need_grad: bool):
            clean: dict[str, torch.Tensor] = {}
            reader_inputs: dict[str, torch.Tensor] = {}
            embed = self._adapter.embed(input_ids).detach().requires_grad_(need_grad)
            clean["embed"] = embed
            pos_emb = self._position_embeddings(embed)
            mask = _build_causal_mask(attention_mask, embed.dtype)

            def rinput(reader):
                acc = None
                for writer, eidx in self.reader_in[reader]:
                    c = coef[eidx] * real_gates[eidx]
                    term = c * clean[writer] + (1.0 - c) * corrupted[writer]
                    acc = term if acc is None else acc + term
                return acc

            for i, layer in enumerate(self._layers):
                xa = rinput(f"attn_{i}")
                reader_inputs[f"attn_{i}"] = xa
                clean.update(self._attn_writers(layer, i, xa, pos_emb, mask))
                xm = rinput(f"mlp_{i}")
                reader_inputs[f"mlp_{i}"] = xm
                clean[f"mlp_{i}"] = self._run_mlp(layer, xm)
            xl = rinput("logits")
            reader_inputs["logits"] = xl
            return clean, reader_inputs, self._adapter.final(xl)

        E = len(self.edges)
        with torch.no_grad():
            clean_full, _, _ = forward_coef(torch.ones(E, device=input_ids.device), need_grad=False)
            _, _, logits_corr = forward_coef(torch.zeros(E, device=input_ids.device), need_grad=False)
        logits_corr = logits_corr.detach()

        coef = torch.ones(E, device=input_ids.device, requires_grad=True)  # gamma(0) = clean
        scores_sum = torch.zeros(E)
        for _ in range(n_steps):
            clean_j, reader_inputs_j, logits_j = forward_coef(coef, need_grad=True)
            metric = metric_fn(logits_j)
            path_loss = ((logits_j - logits_corr) ** 2).sum()
            # GradPath step direction (Eq. 7-8): normalised gradient of the output-mismatch.
            (g_coef,) = torch.autograd.grad(path_loss, coef, retain_graph=True)
            # Metric attribution gradient at this path point (Eq. 9), wrt every reader input.
            readers = list(reader_inputs_j.keys())
            grads = torch.autograd.grad(metric, [reader_inputs_j[r] for r in readers],
                                        retain_graph=False, allow_unused=True)
            grad_by_reader = dict(zip(readers, grads))
            for (writer, reader), eidx in self.edge_index.items():
                g = grad_by_reader[reader]
                if g is None:
                    continue
                delta = (clean_full[writer] - corrupted[writer]).detach()
                scores_sum[eidx] += (delta * g).sum().item()
            with torch.no_grad():
                coef = (coef - g_coef / (g_coef.norm() + eps)).clamp(0.0, 1.0)
            coef = coef.detach().requires_grad_(True)
        return (scores_sum / n_steps).abs()
