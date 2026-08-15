"""The circuit graph: nodes and edges over a decoder transformer.

For the 1B pilot we mask at *node / block* granularity. The writers (things that
add a contribution to the residual stream) are:

    embed                  - token embeddings
    attn_0 .. attn_{L-1}   - each layer's attention block output
    mlp_0  .. mlp_{L-1}    - each layer's MLP block output

The readers (things that read the residual stream) are:

    attn_0 .. attn_{L-1}   - input to each layer's attention
    mlp_0  .. mlp_{L-1}    - input to each layer's MLP
    logits                 - input to the final norm / unembedding

An edge (writer -> reader) exists whenever the writer is causally upstream of
the reader in the computation. Each edge gets one hard-concrete gate. Turning an
edge OFF feeds the reader the writer's *corrupted-input* contribution instead of
its clean one (the disentangled residual stream, see ``edge_masked_model``).

This block-level graph is faithful to Edge Pruning's mechanism while staying
small and architecture-agnostic. Head-level / Q-K-V-split granularity (matching
the paper exactly) is a documented future refinement.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CircuitGraph:
    """Static description of writer/reader nodes and the legal edges between.

    ``granularity`` controls how attention is split on the *writer* side:
      "block" — one ``attn_{i}`` writer per layer (the whole attention block).
      "head"  — one ``attn_{i}_h{j}`` writer per head, so each head can be pruned
                independently (needed to match Wang 2023's per-head IOI circuit).
    The MLP stays one whole ``mlp_{i}`` writer in both modes (its finer parts are
    individual neurons, which Wang does not decompose either). Readers are block
    -level in both modes (the attention block reads one blended input); Q/K/V read
    splitting is a separate future step.
    """

    num_layers: int
    n_heads: int = 12
    granularity: str = "block"

    @property
    def writers(self) -> list[str]:
        names = ["embed"]
        for i in range(self.num_layers):
            if self.granularity == "head":
                names += [f"attn_{i}_h{j}" for j in range(self.n_heads)]
            else:
                names.append(f"attn_{i}")
            names.append(f"mlp_{i}")
        return names

    @property
    def readers(self) -> list[str]:
        names: list[str] = []
        for i in range(self.num_layers):
            names += [f"attn_{i}", f"mlp_{i}"]
        names.append("logits")
        return names

    def _writer_order(self, name: str) -> float:
        """Position of a writer in the forward pass (lower = earlier)."""
        if name == "embed":
            return -1.0
        parts = name.split("_")          # "attn_3" or "attn_3_h7" or "mlp_3"
        kind, idx = parts[0], int(parts[1])
        # attention writes before MLP within the same layer; all heads of a layer
        # share the attention step's position.
        return idx * 2 + (0 if kind == "attn" else 1)

    def _reader_order(self, name: str) -> float:
        """Position of a reader in the forward pass."""
        if name == "logits":
            return float("inf")
        kind, idx = name.split("_")
        return int(idx) * 2 + (0 if kind == "attn" else 1)

    def upstream_writers(self, reader: str) -> list[str]:
        """Writers that are strictly causally upstream of ``reader``."""
        r = self._reader_order(reader)
        return [w for w in self.writers if self._writer_order(w) < r]

    def edges(self) -> list[tuple[str, str]]:
        """All legal (writer, reader) edges, in a stable order."""
        out: list[tuple[str, str]] = []
        for reader in self.readers:
            for writer in self.upstream_writers(reader):
                out.append((writer, reader))
        return out

    @property
    def num_edges(self) -> int:
        return len(self.edges())
