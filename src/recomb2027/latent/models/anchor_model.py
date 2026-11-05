#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     anchor_model.py
#   Authors:  Raman Siarheyeu (raman.siarheyeu@neclab.eu)
#             Miguel Sánchez-Palomino (miguel.palomino@neclab.eu)
#             Nithya Bhasker (nithya.bhasker@neclab.eu)
#             Catherine Chaput (catherine.chaput@neclab.eu)
#
# NEC Laboratories Europe GmbH, Copyright (c) 2026, All rights reserved.
#
#        THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
#
#        PROPRIETARY INFORMATION ---
#
# SOFTWARE LICENSE AGREEMENT
#
# ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
#
# BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
# LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
# DOWNLOAD THE SOFTWARE.
#
# This is a license agreement ("Agreement") between your academic institution
# or non-profit organization or self (called "Licensee" or "You" in this
# Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
# Agreement).  All rights not specifically granted to you in this Agreement
# are reserved for Licensor.
#
# RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
# ownership of any copy of the Software (as defined below) licensed under this
# Agreement and hereby grants to Licensee a personal, non-exclusive,
# non-transferable license to use the Software for noncommercial research
# purposes, without the right to sublicense, pursuant to the terms and
# conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
# LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
# Agreement, the term "Software" means (i) the actual copy of all or any
# portion of code for program routines made accessible to Licensee by Licensor
# pursuant to this Agreement, inclusive of backups, updates, and/or merged
# copies permitted hereunder or subsequently supplied by Licensor,  including
# all or any file structures, programming instructions, user interfaces and
# screen formats and sequences as well as any and all documentation and
# instructions related to it, and (ii) all or any derivatives and/or
# modifications created or made by You to any of the items specified in (i).
#
# CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
# proprietary to Licensor, and as such, Licensee agrees to receive all such
# materials and to use the Software only in accordance with the terms of this
# Agreement.  Licensee agrees to use reasonable effort to protect the Software
# from unauthorized use, reproduction, distribution, or publication. All
# publication materials mentioning features or use of this software must
# explicitly include an acknowledgement the software was developed by NEC
# Laboratories Europe GmbH.
#
# COPYRIGHT: The Software is owned by Licensor.
#
# PERMITTED USES:  The Software may be used for your own noncommercial
# internal research purposes. You understand and agree that Licensor is not
# obligated to implement any suggestions and/or feedback you might provide
# regarding the Software, but to the extent Licensor does so, you are not
# entitled to any compensation related thereto.
#
# DERIVATIVES: You may create derivatives of or make modifications to the
# Software, however, You agree that all and any such derivatives and
# modifications will be owned by Licensor and become a part of the Software
# licensed to You under this Agreement.  You may only use such derivatives and
# modifications for your own noncommercial internal research purposes, and you
# may not otherwise use, distribute or copy such derivatives and modifications
# in violation of this Agreement.
#
# BACKUPS:  If Licensee is an organization, it may make that number of copies
# of the Software necessary for internal noncommercial use at a single site
# within its organization provided that all information appearing in or on
# the original labels, including the copyright and trademark notices are copied
# onto the labels of the copies.
#
# USES NOT PERMITTED:  You may not distribute, copy or use the Software except
# as explicitly permitted herein. Licensee has not been granted any trademark
# license as part of this Agreement.  Neither the name of NEC Laboratories
# Europe GmbH nor the names of its contributors may be used to endorse or
# promote products derived from this Software without specific prior written
# permission.
#
# You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
# whole or in part, or provide third parties access to prior or present
# versions (or any parts thereof) of the Software.
#
# ASSIGNMENT: You may not assign this Agreement or your rights hereunder
# without the prior written consent of Licensor. Any attempted assignment
# without such consent shall be null and void.
#
# TERM: The term of the license granted to you under this Agreement is from Licensee's
# acceptance of this Agreement by downloading the Software or by using the
# Software until terminated as provided below.
#
# The Agreement automatically terminates without notice if you fail to comply
# with any provision of this Agreement.  Licensee may terminate this Agreement
# by ceasing using the Software.  Upon any termination of this Agreement,
# Licensee will delete any and all copies of the Software. You agree that all
# provisions which operate to protect the proprietary rights of Licensor shall
# remain in force should breach occur and that the obligation of
# confidentiality described in this Agreement is binding in perpetuity and, as
# such, survives the term of the Agreement.
#
# FEE: Provided Licensee abides completely by the terms and conditions of this
# Agreement, there is no fee due to Licensor for Licensee's use of the
# Software in accordance with this Agreement.
#
# DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
# OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
# FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
# BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
# RELATED MATERIALS.
#
# SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
# provided as part of this Agreement.
#
# EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
# permitted under applicable law, Licensor shall not be liable for direct,
# indirect, special, incidental, or consequential damages or lost profits
# related to Licensee's use of and/or inability to use the Software, even if
# Licensor is advised of the possibility of such damage.
#
# EXPORT REGULATION: Licensee agrees to comply with any and all applicable
# export control laws, regulations, and/or other laws related to embargoes and
# sanction programs administered by law.
#
# SEVERABILITY: If any provision(s) of this Agreement shall be held to be
# invalid, illegal, or unenforceable by a court or other tribunal of competent
# jurisdiction, the validity, legality and enforceability of the remaining
# provisions shall not in any way be affected or impaired thereby.
#
# NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
# or remedy under this Agreement shall be construed as a waiver of any future
# or other exercise of such right or remedy by Licensor.
#
# GOVERNING LAW: This Agreement shall be construed and enforced in accordance
# with the laws of Germany without reference to conflict of laws principles.
# You consent to the personal jurisdiction of the courts of this country and
# waive their rights to venue outside of Germany.
#
# ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
# entire agreement between Licensee and Licensor as to the matter set forth
# herein and supersedes any previous agreements, understandings, and
# arrangements between the parties relating hereto.
#
#        THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
#
"""Anchor (A4) model: site-aware GPT over mean + PCA-2D latents.

Contents (in definition order): the PCA-2D bottleneck, the GPT backbone (:class:`GPTModel`), the site
bottleneck (site-resolved gate / mean-conditioned attention), and :class:`GPTModelPca2dSite`, the model trained
by ``recomb2027.latent.predictors.train_anchor`` and frozen as the trunk of the residual head."""
from __future__ import annotations

import torch
import torch.nn as nn

from contextlib import nullcontext
from typing import Optional, Dict, Sequence, Tuple, Union
from recomb2027.core.constants import PRIMARY_TARGET
from recomb2027.latent.data_series.pca2d_features import PCA2D_S1_TARGET, is_pca2d_target
from recomb2027.data.extra_encoding_features import (
    normalise_dual_branch_layout,
    normalise_position_stat_slices,
    normalise_triple_branch_layout,
)


# ============================================================================
# (merged from pca2d_bottleneck.py)
# ============================================================================

def _autocast_off(device: torch.device):
    """Disable CUDA/CPU autocast so large pca2d matmuls stay in fp32."""
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", enabled=False)
    if device.type == "cpu":
        return torch.autocast(device_type="cpu", enabled=False)
    return nullcontext()


class Pca2dBottleneckMLP(nn.Module):
    """Encode ``[..., flat_dim]`` → bottleneck and decode back to flat.

    Used so the GPT trunk/head operate in a small latent (e.g. 256–512) while
    predictions can be expanded to ``L * k`` for ``pca2d_euclidean`` / GA.

    Encode/decode always run in **float32** (autocast disabled) to avoid AMP
    fp16 overflows on the large ``L*k`` linears.
    """

    def __init__(
        self,
        flat_dim: int,
        bottleneck_dim: int,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        flat_dim = int(flat_dim)
        bottleneck_dim = int(bottleneck_dim)
        if flat_dim <= 0 or bottleneck_dim <= 0:
            raise ValueError(
                f"flat_dim and bottleneck_dim must be positive; got {flat_dim}, {bottleneck_dim}"
            )
        if hidden_dim is None:
            hidden_dim = int(max(bottleneck_dim * 2, min(1024, max(bottleneck_dim, flat_dim // 8))))
        else:
            hidden_dim = int(hidden_dim)
        self.flat_dim = flat_dim
        self.bottleneck_dim = bottleneck_dim
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, flat_dim),
        )

    def encode(
        self, x: torch.Tensor, condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # ``condition`` ignored (API parity with Pca2dSiteBottleneck).
        del condition
        if x.size(-1) != self.flat_dim:
            raise ValueError(
                f"encode expected last dim {self.flat_dim}; got {tuple(x.shape)}"
            )
        with _autocast_off(x.device):
            return self.encoder(x.float())

    def decode(
        self, z: torch.Tensor, gate_cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        del gate_cond
        if z.size(-1) != self.bottleneck_dim:
            raise ValueError(
                f"decode expected last dim {self.bottleneck_dim}; got {tuple(z.shape)}"
            )
        with _autocast_off(z.device):
            return self.decoder(z.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Autoencoder pass ``x → encode → decode`` (for diagnostics)."""
        return self.decode(self.encode(x))


# ============================================================================
# (merged from gpt.py)
# ============================================================================

DEFAULT_PREDICTION_TARGETS: Tuple[str, ...] = (PRIMARY_TARGET,)


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding supporting variable-length sequences
    up to max_seq_len."""

    def __init__(self, max_seq_len, embed_dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, embed_dim) * 0.02)

    def forward(self, x):
        return x + self.pos_embedding[:, :x.size(1)]


class GPTBlock(nn.Module):
    """Pre-norm transformer decoder block with causal self-attention."""

    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffwd = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.drop_resid = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        x_norm = self.ln1(x)
        attn_output, _ = self.attn(
            x_norm, x_norm, x_norm, attn_mask=attn_mask)
        x = x + self.drop_resid(attn_output)

        x_norm = self.ln2(x)
        x = x + self.ffwd(x_norm)
        return x


# ---------------------------------------------------------------------------
# Optional projection for position-index extra stats (argmax_pos / argmin_pos)
# ---------------------------------------------------------------------------

class PositionStatProjection(nn.Module):
    """Per-stat ``Linear(D, D)`` (+ optional ``LayerNorm``) on input channel blocks."""

    def __init__(
        self,
        slices: dict[str, tuple[int, int]],
        *,
        use_layernorm: bool = False,
    ):
        super().__init__()
        self.slices = dict(slices)
        self.projections = nn.ModuleDict()
        for stat, (start, end) in self.slices.items():
            dim = int(end - start)
            if use_layernorm:
                self.projections[stat] = nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.LayerNorm(dim),
                )
            else:
                self.projections[stat] = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.slices:
            return x
        out = x.clone()
        for stat, (start, end) in self.slices.items():
            out[..., start:end] = self.projections[stat](x[..., start:end])
        return out


# ---------------------------------------------------------------------------
# Dual-branch input encoder (primary mean+vel vs position stats, then fuse)
# ---------------------------------------------------------------------------

class DualBranchInputEncoder(nn.Module):
    """Project primary and position-stat blocks separately, then fuse to ``embed_dim``."""

    def __init__(
        self,
        primary_slice: tuple[int, int],
        position_slice: tuple[int, int],
        embed_dim: int,
        *,
        edge_phylo_slice: Optional[Tuple[int, int]] = None,
        use_layernorm: bool = False,
    ):
        super().__init__()
        p0, p1 = primary_slice
        s0, s1 = position_slice
        primary_dim = int(p1 - p0)
        position_dim = int(s1 - s0)
        if position_dim <= 0:
            raise ValueError(
                f"dual branch requires a non-empty position_slice; got {position_slice!r}"
            )

        edge_dim = 0
        if edge_phylo_slice is not None:
            edge_dim = int(edge_phylo_slice[1] - edge_phylo_slice[0])

        self.primary_slice = primary_slice
        self.position_slice = position_slice
        self.edge_phylo_slice = edge_phylo_slice

        self.head_primary = nn.Linear(primary_dim + edge_dim, embed_dim)
        self.head_position = nn.Linear(position_dim, embed_dim)
        self.fuse = nn.Linear(2 * embed_dim, embed_dim)
        self.ln_primary = nn.LayerNorm(embed_dim) if use_layernorm else nn.Identity()
        self.ln_position = nn.LayerNorm(embed_dim) if use_layernorm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xp = x[..., self.primary_slice[0] : self.primary_slice[1]]
        if self.edge_phylo_slice is not None:
            ep = x[..., self.edge_phylo_slice[0] : self.edge_phylo_slice[1]]
            xp = torch.cat([xp, ep], dim=-1)
        xpos = x[..., self.position_slice[0] : self.position_slice[1]]
        hp = self.ln_primary(self.head_primary(xp))
        hpos = self.ln_position(self.head_position(xpos))
        return self.fuse(torch.cat([hp, hpos], dim=-1))


class TripleBranchInputEncoder(nn.Module):
    """Primary + position-stat + block-mean branches fused to ``embed_dim``."""

    def __init__(
        self,
        primary_slice: tuple[int, int],
        position_slice: tuple[int, int],
        block_slice: tuple[int, int],
        embed_dim: int,
        *,
        edge_phylo_slice: Optional[Tuple[int, int]] = None,
        use_layernorm: bool = False,
    ):
        super().__init__()
        p0, p1 = primary_slice
        s0, s1 = position_slice
        b0, b1 = block_slice
        primary_dim = int(p1 - p0)
        position_dim = int(s1 - s0)
        block_dim = int(b1 - b0)
        if block_dim <= 0:
            raise ValueError(
                f"triple branch requires a non-empty block_slice; got {block_slice!r}"
            )
        if position_dim <= 0:
            raise ValueError(
                "TripleBranchInputEncoder requires a non-empty position_slice "
                f"(argmax_pos/argmin_pos extra stats); got {position_slice!r}"
            )

        edge_dim = 0
        if edge_phylo_slice is not None:
            edge_dim = int(edge_phylo_slice[1] - edge_phylo_slice[0])

        self.primary_slice = primary_slice
        self.position_slice = position_slice
        self.block_slice = block_slice
        self.edge_phylo_slice = edge_phylo_slice

        self.head_primary = nn.Linear(primary_dim + edge_dim, embed_dim)
        self.head_position = nn.Linear(position_dim, embed_dim)
        self.head_block = nn.Linear(block_dim, embed_dim)
        self.fuse = nn.Linear(3 * embed_dim, embed_dim)
        self.ln_primary = nn.LayerNorm(embed_dim) if use_layernorm else nn.Identity()
        self.ln_position = nn.LayerNorm(embed_dim) if use_layernorm else nn.Identity()
        self.ln_block = nn.LayerNorm(embed_dim) if use_layernorm else nn.Identity()
        self._position_dim = position_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xp = x[..., self.primary_slice[0] : self.primary_slice[1]]
        if self.edge_phylo_slice is not None:
            ep = x[..., self.edge_phylo_slice[0] : self.edge_phylo_slice[1]]
            xp = torch.cat([xp, ep], dim=-1)
        if self._position_dim <= 0:
            raise ValueError(
                "TripleBranchInputEncoder requires a non-empty position_slice "
                f"(argmax_pos/argmin_pos extra stats); got {self.position_slice!r}"
            )
        xpos = x[..., self.position_slice[0] : self.position_slice[1]]
        xblk = x[..., self.block_slice[0] : self.block_slice[1]]
        hp = self.ln_primary(self.head_primary(xp))
        hpos = self.ln_position(self.head_position(xpos))
        hblk = self.ln_block(self.head_block(xblk))
        return self.fuse(torch.cat([hp, hpos, hblk], dim=-1))


class MultiTargetDecoder(nn.Module):
    """One linear head per stat target.

    ``output_dim`` may be an int (shared width) or a ``{target: dim}`` map for
    heterogeneous targets (e.g. mean=1280, pca2d_s1=5712).
    """

    def __init__(
        self,
        embed_dim: int,
        output_dim: Union[int, Dict[str, int]],
        targets: Sequence[str],
    ):
        super().__init__()
        self.targets = tuple(targets)
        if not self.targets:
            raise ValueError("MultiTargetDecoder requires at least one target name")
        if isinstance(output_dim, dict):
            dims = {str(k): int(v) for k, v in output_dim.items()}
            missing = [t for t in self.targets if t not in dims]
            if missing:
                raise ValueError(
                    f"output_dim dict missing targets {missing}; got {list(dims)}"
                )
            self.output_dims = {t: dims[t] for t in self.targets}
        else:
            d = int(output_dim)
            self.output_dims = {t: d for t in self.targets}
        self.heads = nn.ModuleDict(
            {name: nn.Linear(embed_dim, self.output_dims[name]) for name in self.targets}
        )

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {name: self.heads[name](h) for name in self.targets}


class MeanPca2dBranchEncoder(nn.Module):
    """Dual-branch input: mean (gatt) channels + pca2d flat channels → embed.

    When ``bottleneck`` is provided, each flat half of the pca2d slice
    (absolute, and velocity if gatt) is encoded to the bottleneck, fused,
    then projected to ``embed_dim``. Otherwise a single linear maps the
    full pca2d slice to ``embed_dim``.
    """

    def __init__(
        self,
        primary_slice: tuple[int, int],
        pca2d_slice: tuple[int, int],
        embed_dim: int,
        *,
        use_layernorm: bool = True,
        pca2d_flat_dim: Optional[int] = None,
    ):
        super().__init__()
        primary_dim = int(primary_slice[1] - primary_slice[0])
        pca2d_dim = int(pca2d_slice[1] - pca2d_slice[0])
        if primary_dim <= 0 or pca2d_dim <= 0:
            raise ValueError(
                f"MeanPca2dBranchEncoder requires positive slices; "
                f"got primary={primary_slice}, pca2d={pca2d_slice}"
            )
        self.primary_slice = primary_slice
        self.pca2d_slice = pca2d_slice
        self.pca2d_in_dim = pca2d_dim
        self.pca2d_flat_dim = int(pca2d_flat_dim) if pca2d_flat_dim else None
        self.head_primary = nn.Linear(primary_dim, embed_dim)
        # Direct linear only when not using a shared bottleneck encode path.
        self.head_pca2d: Optional[nn.Linear]
        self.bn_to_embed: Optional[nn.Linear]
        if self.pca2d_flat_dim is not None:
            self.head_pca2d = None
            self.bn_to_embed = nn.Linear(embed_dim, embed_dim)  # resized in configure_*
        else:
            self.head_pca2d = nn.Linear(pca2d_dim, embed_dim)
            self.bn_to_embed = None
        self.fuse = nn.Linear(2 * embed_dim, embed_dim)
        self.ln_primary = nn.LayerNorm(embed_dim) if use_layernorm else nn.Identity()
        self.ln_pca2d = nn.LayerNorm(embed_dim) if use_layernorm else nn.Identity()

    def configure_bottleneck_proj(self, bottleneck_dim: int, gatt_plus: bool) -> None:
        """Size ``bn_to_embed`` for 1× or 2× bottleneck channels."""
        in_dim = int(bottleneck_dim) * (2 if gatt_plus else 1)
        embed_dim = self.head_primary.out_features
        self.bn_to_embed = nn.Linear(in_dim, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        *,
        bottleneck: Optional[Pca2dBottleneckMLP] = None,
    ) -> torch.Tensor:
        xp = x[..., self.primary_slice[0] : self.primary_slice[1]]
        xq = x[..., self.pca2d_slice[0] : self.pca2d_slice[1]]
        hp = self.ln_primary(self.head_primary(xp))
        if bottleneck is not None and self.bn_to_embed is not None:
            # Keep pca2d encode + project in fp32 (same reason as bottleneck MLP).
            flat = int(bottleneck.flat_dim)
            with _autocast_off(xq.device):
                xq32 = xq.float()
                # Mean-branch token for optional site mean-cond attention (option C).
                # Pass fp32 hp so site encode stays in float32 under autocast-off.
                cond = hp.float()
                if xq32.size(-1) == flat:
                    z = bottleneck.encode(xq32, condition=cond)
                elif xq32.size(-1) == 2 * flat:
                    z = torch.cat(
                        [
                            bottleneck.encode(xq32[..., :flat], condition=cond),
                            bottleneck.encode(xq32[..., flat:], condition=cond),
                        ],
                        dim=-1,
                    )
                else:
                    raise ValueError(
                        f"pca2d slice width {xq.size(-1)} incompatible with "
                        f"bottleneck flat_dim={flat} (expected {flat} or {2 * flat})"
                    )
                hq = self.ln_pca2d(self.bn_to_embed(z.float()))
        else:
            if self.head_pca2d is None:
                raise RuntimeError(
                    "MeanPca2dBranchEncoder requires head_pca2d or a bottleneck"
                )
            hq = self.ln_pca2d(self.head_pca2d(xq))
        # Cast hq to hp dtype so fuse stays in the ambient autocast dtype.
        if hq.dtype != hp.dtype:
            hq = hq.to(dtype=hp.dtype)
        return self.fuse(torch.cat([hp, hq], dim=-1))


def normalise_mean_pca2d_layout(
    raw: Optional[dict[str, Sequence[int]]],
) -> dict[str, tuple[int, int]]:
    if not raw:
        return {"primary_slice": (0, 0), "pca2d_slice": (0, 0)}
    primary = tuple(int(x) for x in raw["primary_slice"])
    pca2d = tuple(int(x) for x in raw["pca2d_slice"])
    return {
        "primary_slice": (primary[0], primary[1]),
        "pca2d_slice": (pca2d[0], pca2d[1]),
    }


def _normalise_prediction_targets(
    raw: Optional[Sequence[str]],
) -> Tuple[str, ...]:
    if raw is None:
        return DEFAULT_PREDICTION_TARGETS
    targets = tuple(str(t) for t in raw)
    if not targets:
        return DEFAULT_PREDICTION_TARGETS
    if PRIMARY_TARGET not in targets:
        raise ValueError(
            f"prediction_targets must include {PRIMARY_TARGET!r}; got {list(targets)}"
        )
    return targets


def _dual_branch_layout_hparams(
    layout: dict[str, Optional[Tuple[int, int]]],
) -> dict[str, Optional[list[int]]]:
    return {
        "primary_slice": list(layout["primary_slice"]),
        "position_slice": list(layout["position_slice"]),
        "edge_phylo_slice": (
            None
            if layout.get("edge_phylo_slice") is None
            else list(layout["edge_phylo_slice"])
        ),
    }


def _triple_branch_layout_hparams(
    layout: dict[str, Optional[Tuple[int, int]]],
) -> dict[str, Optional[list[int]]]:
    return {
        "primary_slice": list(layout["primary_slice"]),
        "position_slice": list(layout["position_slice"]),
        "block_slice": list(layout["block_slice"]),
        "edge_phylo_slice": (
            None
            if layout.get("edge_phylo_slice") is None
            else list(layout["edge_phylo_slice"])
        ),
    }


# ---------------------------------------------------------------------------
# Deterministic GPT
# ---------------------------------------------------------------------------

class GPTModel(nn.Module):
    """Causal GPT for next-step evolutionary prediction in latent space.

    Projects input embeddings into the transformer's internal dimension,
    applies causal self-attention blocks, and projects back to the
    output embedding space.
    """

    def __init__(self, input_dim, embed_dim, ff_dim, output_dim=None,
                 num_heads=8, num_gpt_blocks=4, dropout=0.1,
                 max_seq_len=32,
                 extra_stat_projection: bool = False,
                 extra_stat_projection_layernorm: bool = False,
                 position_stat_slices: Optional[
                     Union[dict[str, Sequence[int]], Sequence[Sequence[Union[str, int]]]]
                 ] = None,
                 extra_stat_dual_branch: bool = False,
                 extra_stat_dual_branch_layernorm: bool = False,
                 dual_branch_layout: Optional[dict[str, Sequence[int]]] = None,
                 block_branch: bool = False,
                 triple_branch_layout: Optional[dict[str, Sequence[int]]] = None,
                 pca2d_branch: bool = False,
                 mean_pca2d_layout: Optional[dict[str, Sequence[int]]] = None,
                 mean_pca2d_layernorm: bool = True,
                 pca2d_bottleneck_dim: Optional[int] = None,
                 pca2d_flat_dim: Optional[int] = None,
                 pca2d_bottleneck_hidden: Optional[int] = None,
                 prediction_targets: Optional[Sequence[str]] = None,
                 target_output_dims: Optional[dict[str, int]] = None,
                 jepa_mask_token: bool = False,
                 jepa_predict_targets: Optional[Sequence[str]] = None):
        super().__init__()
        output_dim = output_dim or input_dim
        pos_slices = normalise_position_stat_slices(position_stat_slices)
        branch_layout = normalise_dual_branch_layout(dual_branch_layout)
        triple_layout = normalise_triple_branch_layout(triple_branch_layout)
        pca_layout = normalise_mean_pca2d_layout(mean_pca2d_layout)
        use_pca2d = bool(
            pca2d_branch
            and pca_layout["pca2d_slice"][1] > pca_layout["pca2d_slice"][0]
        )
        use_triple = bool(
            not use_pca2d
            and block_branch
            and triple_layout["block_slice"][1] > triple_layout["block_slice"][0]
        )
        use_dual = bool(
            not use_pca2d
            and not use_triple
            and extra_stat_dual_branch
            and branch_layout["position_slice"][1] > branch_layout["position_slice"][0]
        )
        use_pos_proj = bool(
            extra_stat_projection and pos_slices and not use_dual and not use_triple and not use_pca2d
        )

        self.prediction_targets = _normalise_prediction_targets(prediction_targets)
        self.primary_target = PRIMARY_TARGET
        jepa_targets = _normalise_prediction_targets(
            jepa_predict_targets or (PRIMARY_TARGET,)
        )
        self.jepa_predict_targets = jepa_targets

        pca2d_target_name = next(
            (t for t in self.prediction_targets if is_pca2d_target(t)),
            PCA2D_S1_TARGET,
        )
        self.pca2d_target_name = pca2d_target_name

        if target_output_dims:
            self.target_output_dims = {
                str(k): int(v) for k, v in target_output_dims.items()
            }
        else:
            self.target_output_dims = {
                t: int(output_dim) for t in self.prediction_targets
            }
        for t in self.prediction_targets:
            self.target_output_dims.setdefault(t, int(output_dim))
        for t in self.jepa_predict_targets:
            self.target_output_dims.setdefault(t, int(output_dim))

        # Bottleneck: head predicts bn dims; decode expands to flat for GA/loss.
        bn_dim = int(pca2d_bottleneck_dim) if pca2d_bottleneck_dim else 0
        flat_for_bn = int(pca2d_flat_dim) if pca2d_flat_dim else 0
        if flat_for_bn <= 0 and use_pca2d and pca2d_target_name in self.target_output_dims:
            flat_for_bn = int(self.target_output_dims[pca2d_target_name])
        self.pca2d_bottleneck: Optional[Pca2dBottleneckMLP] = None
        self.pca2d_bottleneck_dim: Optional[int] = None
        self.pca2d_flat_dim: Optional[int] = None
        if use_pca2d and bn_dim > 0:
            if flat_for_bn <= 0:
                raise ValueError(
                    "pca2d_bottleneck_dim requires pca2d_flat_dim or "
                    f"target_output_dims[{pca2d_target_name!r}]"
                )
            self.pca2d_bottleneck = Pca2dBottleneckMLP(
                flat_dim=flat_for_bn,
                bottleneck_dim=bn_dim,
                hidden_dim=pca2d_bottleneck_hidden,
            )
            self.pca2d_bottleneck_dim = bn_dim
            self.pca2d_flat_dim = flat_for_bn
            # Keep public/API dims as flat; internal head uses bottleneck.
            self.target_output_dims[pca2d_target_name] = flat_for_bn

        head_dims = {t: self.target_output_dims[t] for t in self.prediction_targets}
        if self.pca2d_bottleneck is not None:
            head_dims[pca2d_target_name] = bn_dim
        jepa_dims = {t: self.target_output_dims[t] for t in self.jepa_predict_targets}

        pca2d_slice_w = int(
            pca_layout["pca2d_slice"][1] - pca_layout["pca2d_slice"][0]
        )
        pca2d_gatt = bool(
            self.pca2d_flat_dim is not None and pca2d_slice_w == 2 * self.pca2d_flat_dim
        )

        self.hparams = {
            "input_dim": input_dim,
            "embed_dim": embed_dim,
            "ff_dim": ff_dim,
            "output_dim": output_dim,
            "num_heads": num_heads,
            "num_gpt_blocks": num_gpt_blocks,
            "dropout": dropout,
            "max_seq_len": max_seq_len,
            "extra_stat_projection": use_pos_proj,
            "extra_stat_projection_layernorm": bool(extra_stat_projection_layernorm),
            "position_stat_slices": [
                [stat, int(s), int(e)] for stat, (s, e) in pos_slices.items()
            ],
            "extra_stat_dual_branch": use_dual,
            "extra_stat_dual_branch_layernorm": bool(extra_stat_dual_branch_layernorm),
            "dual_branch_layout": _dual_branch_layout_hparams(branch_layout),
            "block_branch": use_triple,
            "triple_branch_layout": _triple_branch_layout_hparams(triple_layout),
            "pca2d_branch": use_pca2d,
            "mean_pca2d_layout": {
                "primary_slice": list(pca_layout["primary_slice"]),
                "pca2d_slice": list(pca_layout["pca2d_slice"]),
            },
            "mean_pca2d_layernorm": bool(mean_pca2d_layernorm),
            "pca2d_bottleneck_dim": self.pca2d_bottleneck_dim,
            "pca2d_flat_dim": self.pca2d_flat_dim,
            "pca2d_bottleneck_hidden": (
                int(pca2d_bottleneck_hidden) if pca2d_bottleneck_hidden else None
            ),
            "prediction_targets": list(self.prediction_targets),
            "target_output_dims": dict(self.target_output_dims),
            "jepa_mask_token": bool(jepa_mask_token),
            "jepa_predict_targets": list(self.jepa_predict_targets),
        }

        self.dual_branch_encoder: Optional[DualBranchInputEncoder] = None
        self.triple_branch_encoder: Optional[TripleBranchInputEncoder] = None
        self.mean_pca2d_encoder: Optional[MeanPca2dBranchEncoder] = None
        self.position_stat_proj: Optional[PositionStatProjection] = None
        self.input_proj: Optional[nn.Linear] = None

        if use_pca2d:
            self.mean_pca2d_encoder = MeanPca2dBranchEncoder(
                pca_layout["primary_slice"],
                pca_layout["pca2d_slice"],
                embed_dim,
                use_layernorm=mean_pca2d_layernorm,
                pca2d_flat_dim=self.pca2d_flat_dim,
            )
            if self.pca2d_bottleneck is not None:
                self.mean_pca2d_encoder.configure_bottleneck_proj(
                    bn_dim, gatt_plus=pca2d_gatt or pca2d_slice_w == 2 * flat_for_bn
                )
        elif use_triple:
            self.triple_branch_encoder = TripleBranchInputEncoder(
                triple_layout["primary_slice"],
                triple_layout["position_slice"],
                triple_layout["block_slice"],
                embed_dim,
                edge_phylo_slice=triple_layout.get("edge_phylo_slice"),
                use_layernorm=extra_stat_dual_branch_layernorm,
            )
        elif use_dual:
            self.dual_branch_encoder = DualBranchInputEncoder(
                branch_layout["primary_slice"],
                branch_layout["position_slice"],
                embed_dim,
                edge_phylo_slice=branch_layout.get("edge_phylo_slice"),
                use_layernorm=extra_stat_dual_branch_layernorm,
            )
        else:
            self.position_stat_proj = (
                PositionStatProjection(
                    pos_slices,
                    use_layernorm=extra_stat_projection_layernorm,
                )
                if use_pos_proj
                else None
            )
            self.input_proj = nn.Linear(input_dim, embed_dim)

        self.pos_encoding = LearnablePositionalEncoding(max_seq_len, embed_dim)
        self.blocks = nn.ModuleList([
            GPTBlock(embed_dim, num_heads, ff_dim, dropout=dropout)
            for _ in range(num_gpt_blocks)
        ])
        self.ln_out = nn.LayerNorm(embed_dim)
        if len(self.prediction_targets) == 1 and len(set(head_dims.values())) == 1:
            only_dim = next(iter(head_dims.values()))
            self.lm_head: Optional[nn.Linear] = nn.Linear(embed_dim, only_dim)
            self.lm_heads: Optional[MultiTargetDecoder] = None
        else:
            self.lm_head = None
            self.lm_heads = MultiTargetDecoder(
                embed_dim, head_dims, self.prediction_targets
            )

        self.mask_token: Optional[nn.Parameter]
        self.jepa_heads: Optional[MultiTargetDecoder] = None
        if jepa_mask_token:
            self.mask_token = nn.Parameter(torch.randn(input_dim) * 0.02)
            self.jepa_heads = MultiTargetDecoder(
                embed_dim, jepa_dims, self.jepa_predict_targets
            )
        else:
            self.mask_token = None

    def _causal_mask(self, seq_len, device):
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float('-inf'),
            diagonal=1)

    def _preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.position_stat_proj is not None:
            x = self.position_stat_proj(x)
        return x

    def _project_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean_pca2d_encoder is not None:
            return self.mean_pca2d_encoder(x, bottleneck=self.pca2d_bottleneck)
        if self.triple_branch_encoder is not None:
            return self.triple_branch_encoder(x)
        if self.dual_branch_encoder is not None:
            return self.dual_branch_encoder(x)
        assert self.input_proj is not None
        return self.input_proj(self._preprocess_input(x))

    def encode_trunk(self, x: torch.Tensor) -> torch.Tensor:
        """Causal transformer trunk ending in ``ln_out`` (before decoders)."""
        h = self._project_input(x)
        h = self.pos_encoding(h)
        mask = self._causal_mask(h.size(1), h.device)
        for block in self.blocks:
            h = block(h, attn_mask=mask)
        return self.ln_out(h)

    def _decode_diff(
        self,
        h: torch.Tensor,
        *,
        expand_pca2d: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if self.lm_heads is not None:
            out = self.lm_heads(h)
        else:
            assert self.lm_head is not None
            out = {self.prediction_targets[0]: self.lm_head(h)}
        if (
            expand_pca2d
            and self.pca2d_bottleneck is not None
            and self.pca2d_target_name in out
        ):
            out[self.pca2d_target_name] = self.pca2d_bottleneck.decode(
                out[self.pca2d_target_name]
            )
        return out

    def decode_pca2d_bottleneck(self, z: torch.Tensor) -> torch.Tensor:
        """Expand bottleneck pca2d predictions to flat ``L*k`` for GA metrics."""
        if self.pca2d_bottleneck is None:
            raise RuntimeError("decode_pca2d_bottleneck requires pca2d_bottleneck_dim")
        return self.pca2d_bottleneck.decode(z)

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_dict: bool = False,
        expand_pca2d: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Parameters
        ----------
        x : Tensor [B, T, input_dim]
        return_dict : bool
            If True, return ``{target_name: [B,T,D]}`` for all diff heads.
        expand_pca2d : bool
            When a pca2d bottleneck is enabled, decode head outputs to flat
            ``L*k`` (default True so val/GA see the k=8 space).

        Returns
        -------
        Tensor [B, T, output_dim] for the primary (mean) diff head, or a dict
        when ``return_dict=True``.
        """
        out = self._decode_diff(self.encode_trunk(x), expand_pca2d=expand_pca2d)
        if return_dict:
            return out
        return out[self.primary_target]

    def forward_jepa_aux(
        self,
        x: torch.Tensor,
        mask_indices: torch.Tensor,
        *,
        predict_targets: Optional[Sequence[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Predict absolute encodings at masked prefix steps (JEPA auxiliary).

        Parameters
        ----------
        x : Tensor [B, T, input_dim]
        mask_indices : Tensor [B, M] int64; ``-1`` pads unused slots.
        predict_targets : optional subset of ``jepa_predict_targets``.
        """
        if self.mask_token is None or self.jepa_heads is None:
            raise RuntimeError("forward_jepa_aux requires jepa_mask_token=True")

        targets = tuple(predict_targets or self.jepa_predict_targets)
        x_masked = x.clone()
        for b in range(x.size(0)):
            for t in mask_indices[b]:
                ti = int(t.item())
                if ti >= 0:
                    x_masked[b, ti] = self.mask_token

        trunk = self.encode_trunk(x_masked)
        decoded = self.jepa_heads(trunk)

        out: Dict[str, torch.Tensor] = {}
        max_m = int(mask_indices.size(1))
        for tname in targets:
            if tname not in decoded:
                continue
            rows = []
            for b in range(x.size(0)):
                picks = []
                for t in mask_indices[b]:
                    ti = int(t.item())
                    if ti >= 0:
                        picks.append(decoded[tname][b, ti])
                if not picks:
                    picks = [decoded[tname].new_zeros(decoded[tname].size(-1))]
                row = torch.stack(picks, dim=0)
                if row.size(0) < max_m:
                    pad = row.new_zeros(max_m - row.size(0), row.size(-1))
                    row = torch.cat([row, pad], dim=0)
                rows.append(row[:max_m])
            out[tname] = torch.stack(rows, dim=0)
        return out

    def encode_prefix(self, x):
        """Last-token trunk representation (for diffusion cond)."""
        return self.encode_trunk(x)[:, -1, :]


# ---------------------------------------------------------------------------
# Variational GPT  (V-GPT)
# ---------------------------------------------------------------------------


# ============================================================================
# (merged from pca2d_site_bottleneck.py)
# ============================================================================

class _SiteBasis(nn.Module):
    """Learned site×rank basis ``U ∈ R^{L×r}`` for low-rank decode expand."""

    def __init__(self, seq_len: int, rank: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(int(seq_len), int(rank)))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, factors: torch.Tensor) -> torch.Tensor:
        # factors: [..., r, d] → H: [..., L, d]
        return torch.einsum("lr,...rd->...ld", self.weight, factors)


class Pca2dSiteBottleneck(nn.Module):
    """Encode ``[..., L*k]`` → bottleneck and decode back to flat via ``[L, k]``.

    Encode/decode run in **float32** (autocast off) like :class:`Pca2dBottleneckMLP`.

    ``site_pool``:
      - ``"mean"`` — uniform average over sites (baseline)
      - ``"attention"`` — learned softmax attention over sites, then pool

    ``decode_rank``:
      - ``0`` — dense ``bn → L·d`` expand (legacy)
      - ``>0`` — low-rank ``H = U · reshape(V z)`` with ``U∈R^{L×r}``

    ``mean_cond_attn`` (option C): when True and ``encode(..., condition=m)`` is
    given, scores are ``uᵀ tanh(W_h h_ℓ + W_m m)``. Without condition (AE),
    falls back to unconditional ``attn_score``.

    ``gated_residual`` (option C): per-PC gate
    ``g = σ(MLP([z; gate_cond])) ∈ R^k`` broadcast over sites.
    """

    def __init__(
        self,
        seq_len: int,
        k: int,
        bottleneck_dim: int,
        *,
        site_hidden: int = 64,
        site_dim: int = 32,
        mixer_layers: int = 1,
        dropout: float = 0.1,
        site_pool: str = "mean",
        decode_rank: int = 0,
        mean_cond_attn: bool = False,
        gated_residual: bool = False,
        site_resolved_gate: bool = False,
        mean_cond_dim: int = 0,
        gate_cond_dim: int = 0,
    ):
        super().__init__()
        seq_len = int(seq_len)
        k = int(k)
        bottleneck_dim = int(bottleneck_dim)
        if seq_len <= 0 or k <= 0 or bottleneck_dim <= 0:
            raise ValueError(
                f"seq_len, k, bottleneck_dim must be positive; got "
                f"{seq_len}, {k}, {bottleneck_dim}"
            )
        pool = str(site_pool or "mean").lower()
        if pool not in ("mean", "attention"):
            raise ValueError(
                f"site_pool must be 'mean' or 'attention'; got {site_pool!r}"
            )
        rank = int(decode_rank or 0)
        if rank < 0:
            raise ValueError(f"decode_rank must be >= 0; got {decode_rank}")
        mean_cond = bool(mean_cond_attn)
        gated = bool(gated_residual)
        site_gate = bool(site_resolved_gate)
        cond_dim = int(mean_cond_dim or 0)
        g_cond_dim = int(gate_cond_dim or 0)
        if mean_cond:
            if pool != "attention":
                raise ValueError("mean_cond_attn requires site_pool='attention'")
            if cond_dim <= 0:
                raise ValueError(
                    "mean_cond_attn requires mean_cond_dim > 0 (embed dim)"
                )
        if gated and g_cond_dim <= 0 and not site_gate:
            raise ValueError(
                "gated_residual requires gate_cond_dim > 0 "
                "(trunk hidden / embed dim) unless site_resolved_gate=true"
            )
        if site_gate and not gated:
            raise ValueError("site_resolved_gate requires gated_residual=true")

        self.seq_len = seq_len
        self.k = k
        self.flat_dim = seq_len * k
        self.bottleneck_dim = bottleneck_dim
        self.site_dim = int(site_dim)
        self.site_hidden = int(site_hidden)
        self.site_pool = pool
        self.decode_rank = rank
        self.mean_cond_attn = mean_cond
        self.gated_residual = gated
        self.site_resolved_gate = site_gate
        self.mean_cond_dim = cond_dim
        self.gate_cond_dim = g_cond_dim

        self.site_encode = nn.Sequential(
            nn.Linear(k, self.site_hidden),
            nn.GELU(),
            nn.Linear(self.site_hidden, self.site_dim),
        )
        mixers: list[nn.Module] = []
        for _ in range(max(0, int(mixer_layers))):
            mixers.append(_SiteConvMixer(self.site_dim, dropout=float(dropout)))
        self.site_mixer = nn.Sequential(*mixers) if mixers else nn.Identity()
        self.attn_score: nn.Module
        if pool == "attention":
            self.attn_score = nn.Linear(self.site_dim, 1)
        else:
            self.attn_score = nn.Identity()
        self.pool_to_bn = nn.Linear(self.site_dim, bottleneck_dim)

        # Mean-conditioned attention (option C encode).
        self.cond_Wh: Optional[nn.Module]
        self.cond_Wm: Optional[nn.Module]
        self.cond_u: Optional[nn.Module]
        if mean_cond:
            self.cond_Wh = nn.Linear(self.site_dim, self.site_dim, bias=False)
            self.cond_Wm = nn.Linear(cond_dim, self.site_dim, bias=False)
            self.cond_u = nn.Linear(self.site_dim, 1, bias=False)
        else:
            self.cond_Wh = None
            self.cond_Wm = None
            self.cond_u = None

        # Named ``decoder`` so pca2d curriculum freeze matches flat MLP API.
        site_decode = nn.Sequential(
            nn.Linear(self.site_dim, self.site_hidden),
            nn.GELU(),
            nn.Linear(self.site_hidden, k),
        )
        decoder_mods: dict[str, nn.Module] = {}
        if rank > 0:
            decoder_mods["bn_to_rank"] = nn.Linear(
                bottleneck_dim, rank * self.site_dim
            )
            decoder_mods["site_basis"] = _SiteBasis(seq_len, rank)
        else:
            decoder_mods["bn_to_sites"] = nn.Linear(
                bottleneck_dim, seq_len * self.site_dim
            )
        decoder_mods["site_decode"] = site_decode
        if gated:
            if site_gate:
                gate = nn.Linear(self.site_dim, k)
                nn.init.zeros_(gate.weight)
                nn.init.constant_(gate.bias, -4.0)
                decoder_mods["site_gate"] = gate
            else:
                gate_hidden = max(self.site_hidden, bottleneck_dim // 2)
                gate = nn.Sequential(
                    nn.Linear(bottleneck_dim + g_cond_dim, gate_hidden),
                    nn.GELU(),
                    nn.Linear(gate_hidden, k),
                )
                last_gate = gate[-1]
                assert isinstance(last_gate, nn.Linear)
                nn.init.zeros_(last_gate.weight)
                nn.init.constant_(last_gate.bias, -4.0)
                decoder_mods["residual_gate"] = gate
        self.decoder = nn.ModuleDict(decoder_mods)

    def _to_sites(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.flat_dim:
            raise ValueError(
                f"expected last dim {self.flat_dim}; got {tuple(x.shape)}"
            )
        return x.reshape(*x.shape[:-1], self.seq_len, self.k)

    def _pool_sites(
        self, h: torch.Tensor, condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Pool ``[..., L, C]`` → ``[..., C]``."""
        if self.site_pool != "attention":
            return h.mean(dim=-2)
        use_cond = (
            self.mean_cond_attn
            and condition is not None
            and self.cond_Wh is not None
            and self.cond_Wm is not None
            and self.cond_u is not None
        )
        if use_cond:
            # condition: [..., mean_cond_dim] → broadcast over L
            m = condition.float()
            if m.shape[:-1] != h.shape[:-2]:
                raise ValueError(
                    f"condition leading shape {tuple(m.shape[:-1])} != "
                    f"site leading {tuple(h.shape[:-2])}"
                )
            if m.size(-1) != self.mean_cond_dim:
                raise ValueError(
                    f"condition last dim {m.size(-1)} != mean_cond_dim="
                    f"{self.mean_cond_dim}"
                )
            scores = self.cond_u(
                torch.tanh(self.cond_Wh(h) + self.cond_Wm(m).unsqueeze(-2))
            )
        else:
            scores = self.attn_score(h)
        weights = torch.softmax(scores, dim=-2)
        return (weights * h).sum(dim=-2)

    def _bn_to_site_codes(self, z: torch.Tensor) -> torch.Tensor:
        """Expand bottleneck ``z`` to site codes ``[..., L, site_dim]``."""
        if self.decode_rank > 0:
            factors = self.decoder["bn_to_rank"](z)
            factors = factors.reshape(*z.shape[:-1], self.decode_rank, self.site_dim)
            return self.decoder["site_basis"](factors)
        h = self.decoder["bn_to_sites"](z)
        return h.reshape(*z.shape[:-1], self.seq_len, self.site_dim)

    def encode(
        self, x: torch.Tensor, condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        with _autocast_off(x.device):
            sites = self._to_sites(x.float())
            h = self.site_encode(sites)
            h = self.site_mixer(h)
            pooled = self._pool_sites(h, condition=condition)
            return self.pool_to_bn(pooled)

    def decode(
        self, z: torch.Tensor, gate_cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if z.size(-1) != self.bottleneck_dim:
            raise ValueError(
                f"decode expected last dim {self.bottleneck_dim}; got {tuple(z.shape)}"
            )
        with _autocast_off(z.device):
            z32 = z.float()
            h = self._bn_to_site_codes(z32)
            sites = self.decoder["site_decode"](h)
            if self.gated_residual and self.site_resolved_gate and "site_gate" in self.decoder:
                g = torch.sigmoid(self.decoder["site_gate"](h))
                sites = sites * g
            elif (
                self.gated_residual
                and gate_cond is not None
                and "residual_gate" in self.decoder
            ):
                gc = gate_cond.float()
                if gc.shape[:-1] != z32.shape[:-1]:
                    raise ValueError(
                        f"gate_cond leading shape {tuple(gc.shape[:-1])} != "
                        f"z leading {tuple(z32.shape[:-1])}"
                    )
                if gc.size(-1) != self.gate_cond_dim:
                    raise ValueError(
                        f"gate_cond last dim {gc.size(-1)} != gate_cond_dim="
                        f"{self.gate_cond_dim}"
                    )
                g = torch.sigmoid(
                    self.decoder["residual_gate"](torch.cat([z32, gc], dim=-1))
                )
                sites = sites * g.unsqueeze(-2)
            return sites.reshape(*z.shape[:-1], self.flat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def encoder_parameters(self):
        """Parameters to freeze after AE warmup (encode path only)."""
        yield from self.site_encode.parameters()
        yield from self.site_mixer.parameters()
        if self.site_pool == "attention":
            yield from self.attn_score.parameters()
        if self.mean_cond_attn:
            if self.cond_Wh is not None:
                yield from self.cond_Wh.parameters()
            if self.cond_Wm is not None:
                yield from self.cond_Wm.parameters()
            if self.cond_u is not None:
                yield from self.cond_u.parameters()
        yield from self.pool_to_bn.parameters()


class _SiteConvMixer(nn.Module):
    """Depthwise-ish 1D conv along the site axis (channels = site_dim)."""

    def __init__(self, site_dim: int, dropout: float = 0.1, kernel_size: int = 5):
        super().__init__()
        pad = kernel_size // 2
        self.norm = nn.LayerNorm(site_dim)
        self.conv = nn.Conv1d(
            site_dim, site_dim, kernel_size=kernel_size, padding=pad, groups=1
        )
        self.drop = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(site_dim, site_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(site_dim * 2, site_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [..., L, C]
        lead = h.shape[:-2]
        L, C = h.shape[-2], h.shape[-1]
        x = h.reshape(-1, L, C)
        y = self.norm(x)
        y = self.conv(y.transpose(1, 2)).transpose(1, 2)
        x = x + self.drop(y)
        x = x + self.ff(self.norm(x))
        return x.reshape(*lead, L, C)


# ============================================================================
# (merged from gpt_pca2d_site.py)
# ============================================================================

class GPTModelPca2dSite(GPTModel):
    """GPTModel that swaps the flat pca2d MLP for :class:`Pca2dSiteBottleneck`."""

    def __init__(
        self,
        *args,
        pca2d_seq_len: int = 712,
        pca2d_k: int = 4,
        pca2d_site_hidden: int = 64,
        pca2d_site_dim: int = 32,
        pca2d_site_mixer_layers: int = 1,
        pca2d_site_pool: str = "mean",
        pca2d_decode_rank: int = 0,
        pca2d_mean_cond_attn: bool = False,
        pca2d_gated_residual: bool = False,
        pca2d_site_resolved_gate: bool = False,
        pca2d_residual: bool = True,
        **kwargs,
    ):
        # Parent builds a flat Pca2dBottleneckMLP; we replace it immediately after.
        super().__init__(*args, **kwargs)
        if self.pca2d_bottleneck is None:
            raise ValueError(
                "GPTModelPca2dSite requires pca2d_bottleneck_dim / pca2d_flat_dim"
            )
        flat = int(self.pca2d_flat_dim or 0)
        bn = int(self.pca2d_bottleneck_dim or 0)
        seq_len = int(pca2d_seq_len)
        k = int(pca2d_k)
        if seq_len * k != flat:
            raise ValueError(
                f"pca2d_seq_len*k={seq_len * k} != pca2d_flat_dim={flat}"
            )
        dropout = float(self.hparams.get("dropout", 0.1))
        pool = str(pca2d_site_pool or "mean")
        decode_rank = int(pca2d_decode_rank or 0)
        mean_cond = bool(pca2d_mean_cond_attn)
        gated = bool(pca2d_gated_residual)
        site_gate = bool(pca2d_site_resolved_gate)
        embed_dim = int(self.hparams.get("embed_dim", 0) or 0)
        self.pca2d_bottleneck = Pca2dSiteBottleneck(
            seq_len=seq_len,
            k=k,
            bottleneck_dim=bn,
            site_hidden=int(pca2d_site_hidden),
            site_dim=int(pca2d_site_dim),
            mixer_layers=int(pca2d_site_mixer_layers),
            dropout=dropout,
            site_pool=pool,
            decode_rank=decode_rank,
            mean_cond_attn=mean_cond,
            gated_residual=gated,
            site_resolved_gate=site_gate,
            mean_cond_dim=embed_dim if mean_cond else 0,
            gate_cond_dim=embed_dim if gated and not site_gate else 0,
        )
        self.pca2d_residual = bool(pca2d_residual)
        self.pca2d_seq_len = seq_len
        self.pca2d_k = k
        self.hparams["name"] = "gpt_pca2d_site"
        self.hparams["pca2d_bottleneck_type"] = "site"
        self.hparams["pca2d_seq_len"] = seq_len
        self.hparams["pca2d_k"] = k
        self.hparams["pca2d_site_hidden"] = int(pca2d_site_hidden)
        self.hparams["pca2d_site_dim"] = int(pca2d_site_dim)
        self.hparams["pca2d_site_mixer_layers"] = int(pca2d_site_mixer_layers)
        self.hparams["pca2d_site_pool"] = pool
        self.hparams["pca2d_decode_rank"] = decode_rank
        self.hparams["pca2d_mean_cond_attn"] = mean_cond
        self.hparams["pca2d_gated_residual"] = gated
        self.hparams["pca2d_site_resolved_gate"] = site_gate
        self.hparams["pca2d_residual"] = self.pca2d_residual
        # Zero-init last site decode layer so residuals start near 0.
        last = self.pca2d_bottleneck.decoder["site_decode"][-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def extract_abs_pca2d(self, x: torch.Tensor) -> torch.Tensor:
        """Absolute pca2d flat from dual-branch input ``x`` ``[B,T,Din]``."""
        layout = self.hparams.get("mean_pca2d_layout") or {}
        sl = layout.get("pca2d_slice")
        if not sl:
            raise RuntimeError("mean_pca2d_layout.pca2d_slice missing")
        s, e = int(sl[0]), int(sl[1])
        xq = x[..., s:e]
        flat = int(self.pca2d_flat_dim)
        if xq.size(-1) == flat:
            return xq
        if xq.size(-1) == 2 * flat:
            return xq[..., :flat]
        raise ValueError(
            f"pca2d slice width {xq.size(-1)} incompatible with flat_dim={flat}"
        )

    def _decode_diff(
        self,
        h: torch.Tensor,
        *,
        expand_pca2d: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if self.lm_heads is not None:
            out = self.lm_heads(h)
        else:
            assert self.lm_head is not None
            out = {self.prediction_targets[0]: self.lm_head(h)}
        if (
            expand_pca2d
            and self.pca2d_bottleneck is not None
            and self.pca2d_target_name in out
        ):
            z = out[self.pca2d_target_name]
            gate_cond = h if bool(self.hparams.get("pca2d_gated_residual")) else None
            out[self.pca2d_target_name] = self.pca2d_bottleneck.decode(
                z, gate_cond=gate_cond
            )
        return out

    def decode_pca2d_residual(
        self, z: torch.Tensor, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(delta_flat, abs_next)`` with ``abs_next = pca_t + delta``."""
        assert self.pca2d_bottleneck is not None
        # GA path without trunk h: ungated decode (gate_cond=None).
        delta = self.pca2d_bottleneck.decode(z)
        base = self.extract_abs_pca2d(x)
        if base.shape[:-1] != delta.shape[:-1]:
            raise ValueError(
                f"base/delta shape mismatch: {tuple(base.shape)} vs {tuple(delta.shape)}"
            )
        return delta, base + delta

    def decode_pca2d_bottleneck(self, z: torch.Tensor) -> torch.Tensor:
        """Expand bottleneck pca2d predictions to flat ``L*k`` for GA metrics."""
        if self.pca2d_bottleneck is None:
            raise RuntimeError("decode_pca2d_bottleneck requires pca2d_bottleneck_dim")
        return self.pca2d_bottleneck.decode(z)

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_dict: bool = False,
        expand_pca2d: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        # Diff-target training: decode(Δz) ≈ Δpca; GA absolute =
        # pca_t + decode(Δz) via decode_pca2d_residual.
        out = self._decode_diff(self.encode_trunk(x), expand_pca2d=expand_pca2d)
        if return_dict:
            return out
        return out[self.primary_target]

    def set_pca2d_encoder_trainable(self, trainable: bool) -> int:
        """Freeze/unfreeze site encode path (after AE warmup)."""
        assert self.pca2d_bottleneck is not None
        n = 0
        for p in self.pca2d_bottleneck.encoder_parameters():
            p.requires_grad = bool(trainable)
            n += p.numel()
        return n


def extract_pca2d_flats_from_batch(
    model: GPTModelPca2dSite, xb: torch.Tensor
) -> torch.Tensor:
    """``[B,T,flat]`` absolute pca2d for AE warmup."""
    return model.extract_abs_pca2d(xb)
