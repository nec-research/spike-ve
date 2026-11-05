#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     residual_model.py
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
"""Residual head (B4) on top of the frozen anchor.

Contents (in definition order): zero-inflated / low-rank Gaussian site utilities, the PCA-2D context memory,
site-gate losses, the AdaLN site blocks and DiT heads, the residual regressor, and the unified models
:class:`Pca2dSiteUnifiedV3` and :class:`Pca2dSiteUnifiedV4` (the paper's B4: mean read out from the gated
residual through the frozen ridge). Trained by ``recomb2027.latent.predictors.train_residual``."""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
import logging

from typing import Optional, Union, Any, Tuple
from pathlib import Path
from recomb2027.latent.models.anchor_model import GPTModelPca2dSite
from recomb2027.latent.models.model_utils import filter_model_ctor_kwargs
from recomb2027.latent.data_series.pca2d_features import PCA2D_S1_TARGET
from recomb2027.core.constants import PRIMARY_TARGET
from recomb2027.latent.models.consistency_loss import (
    FrozenPca2dDeltaReadout,
    attach_consistency_to_loss,
    delta_consistency_mse,
)


# ============================================================================
# (merged from pca2d_zi_gaussian.py)
# ============================================================================

def _site_gaussian_log_prob(
    diff: torch.Tensor,
    log_sigma: torch.Tensor,
    U: torch.Tensor,
    *,
    min_var: float = 1e-4,
) -> torch.Tensor:
    """Log N(diff; 0, diag(sigma^2)+UU^T) per site, shape ``[B,L]``."""
    k = diff.shape[-1]
    sigma2 = torch.exp(2.0 * log_sigma)
    low = torch.einsum("blkr,blks->blrs", U, U)
    eye = torch.eye(k, device=U.device, dtype=U.dtype).view(1, 1, k, k)
    cov = sigma2.unsqueeze(-1) * eye + low + min_var * eye
    B, L = diff.shape[:2]
    diff_f = diff.reshape(B * L, k, 1)
    cov_f = cov.reshape(B * L, k, k)
    try:
        chol = torch.linalg.cholesky(cov_f)
        solved = torch.cholesky_solve(diff_f, chol)
        log_det = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
    except RuntimeError:
        solved = torch.linalg.solve(cov_f, diff_f)
        log_det = torch.logdet(cov_f)
    maha = (diff_f * solved).sum(dim=(-2, -1))
    log_2pi = diff.new_tensor(2.0 * 3.141592653589793).log()
    return (-0.5 * (maha + log_det + k * log_2pi)).reshape(B, L)


def point_pred_from_residual(mu_flat: torch.Tensor, mu_r: torch.Tensor) -> torch.Tensor:
    """Point prediction ``mu + mu_r`` flattened."""
    return mu_flat + mu_r.reshape(mu_flat.shape[0], -1)


def low_rank_site_nll(
    target_flat: torch.Tensor,
    mu_flat: torch.Tensor,
    mu_r: torch.Tensor,
    log_sigma: torch.Tensor,
    U: torch.Tensor,
    *,
    seq_len: int,
    k: int,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Low-rank Gaussian NLL on residual ``target - mu`` around ``mu_r``."""
    target_s = target_flat.reshape(-1, seq_len, k)
    mu_s = mu_flat.reshape(-1, seq_len, k)
    residual = target_s - mu_s
    log_prob = _site_gaussian_log_prob(residual - mu_r, log_sigma, U)
    per_sample = (-log_prob).mean(dim=-1)
    if sample_weight is not None:
        w = sample_weight.reshape(-1).to(dtype=per_sample.dtype)
        return (per_sample * w).sum() / w.sum().clamp(min=1e-12)
    return per_sample.mean()


def site_total_variance(log_sigma: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """Per-site total variance ``diag(sigma^2) + ||U_row||^2``, shape ``[B,L,k]``."""
    sigma2 = torch.exp(2.0 * log_sigma)
    low_var = U.pow(2).sum(dim=-1)
    return sigma2 + low_var


# ============================================================================
# (merged from pca2d_context_memory.py)
# ============================================================================

class Pca2dContextMemory(nn.Module):
    """Map trunk hidden states ``[B,T,D]`` → memory ``[B,T,mem_dim]``."""

    def __init__(self, embed_dim: int, mem_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(int(embed_dim), int(mem_dim)),
            nn.LayerNorm(int(mem_dim)),
        )

    def forward(self, h_seq: torch.Tensor) -> torch.Tensor:
        return self.proj(h_seq)


# ============================================================================
# (merged from site_gate_losses.py)
# ============================================================================

def gate_ranking_metrics(
    scores: np.ndarray,
    changed: np.ndarray,
    prefix_count: Optional[np.ndarray] = None,
    *,
    topk: int = 20,
) -> dict[str, float]:
    """AUROC / AP / recall@k of the gate on ``[N, L]`` scores.

    ``all``: changed vs unchanged sites.  ``novel``: restricted to sites that never changed
    in the prefix (``prefix_count == 0``) -- removes the "mutated before" shortcut that makes
    reversions trivial, so it is a proxy for genuine forward-site ranking.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    out: dict[str, float] = {}
    y = changed.astype(bool)

    def _pair(tag: str, s: np.ndarray, lab: np.ndarray, valid: np.ndarray) -> None:
        sv, lv = s[valid], lab[valid]
        if lv.sum() == 0 or lv.sum() == lv.size:
            out[f"gate_auroc_{tag}"] = out[f"gate_ap_{tag}"] = float("nan")
        else:
            out[f"gate_auroc_{tag}"] = float(roc_auc_score(lv, sv))
            out[f"gate_ap_{tag}"] = float(average_precision_score(lv, sv))
        # recall@k per series over valid sites
        sc = np.where(valid, s, -np.inf)
        k = min(int(topk), sc.shape[1])
        top = np.argpartition(-sc, k - 1, axis=1)[:, :k]
        hit = np.take_along_axis(lab & valid, top, axis=1).sum()
        npos = (lab & valid).sum()
        out[f"gate_recall_{tag}"] = float(hit / npos) if npos > 0 else float("nan")

    _pair("all", scores, y, np.ones_like(y))
    if prefix_count is not None:
        valid = prefix_count <= 0
        _pair("novel", scores, y, valid)
    return out


# ============================================================================
# (merged from pca2d_site_residual_diffusion.py)
# ============================================================================

def load_frozen_site_gpt_anchor(
    path: Union[str, Path],
    device: torch.device,
    *,
    strict: bool = True,
) -> GPTModelPca2dSite:
    """Load ``GPTModelPca2dSite_best_model.pt`` (A4) as a frozen anchor."""
    path = Path(path).expanduser()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cls_name = ckpt.get("model_cls_name")
    if cls_name != "GPTModelPca2dSite":
        raise ValueError(
            f"Anchor checkpoint must be GPTModelPca2dSite; got {cls_name!r}"
        )
    hp = ckpt.get("hyperparameters")
    if hp is None:
        raise ValueError(f"Checkpoint missing hyperparameters: {path}")
    ctor = filter_model_ctor_kwargs(GPTModelPca2dSite, dict(hp))
    model = GPTModelPca2dSite(**ctor)
    model.load_state_dict(ckpt["model_state_dict"], strict=strict)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


class SiteAdaLNConvBlock(nn.Module):
    """AdaLN + Conv1d along L + pointwise FF (channels = C)."""

    def __init__(
        self,
        channels: int,
        cond_dim: int,
        *,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        pad = int(kernel_size) // 2
        self.norm1 = nn.LayerNorm(channels)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, channels * 2),
        )
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=int(kernel_size),
            padding=pad,
        )
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
        )

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """``h``: ``[B, L, C]``, ``cond``: ``[B, cond_dim]``."""
        scale, shift = self.adaLN(cond).chunk(2, dim=-1)
        # Broadcast AdaLN over sites.
        y = self.norm1(h) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        y = self.conv(y.transpose(1, 2)).transpose(1, 2)
        h = h + self.drop(y)
        h = h + self.ff(self.norm2(h))
        return h


# ============================================================================
# (merged from pca2d_site_residual_regressor.py)
# ============================================================================

class MlpBnResidualHead(nn.Module):
    """MLP ``cond → residual_z`` (zero-init last layer)."""

    def __init__(
        self,
        bottleneck_dim: int,
        cond_dim: int,
        *,
        hidden_dim: int = 512,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        dims = [cond_dim] + [int(hidden_dim)] * max(1, int(n_layers) - 1) + [
            int(bottleneck_dim)
        ]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.extend(
                [
                    nn.Linear(dims[i], dims[i + 1]),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
        self.trunk = nn.Sequential(*layers) if layers else nn.Identity()
        self.out = nn.Linear(dims[-2], dims[-1])
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.out(self.trunk(cond))


# ============================================================================
# (merged from site_resolved_dit_head.py)
# ============================================================================

class SiteAdaLNZeroBlock(nn.Module):
    """Transformer block with AdaLN-Zero (DiT-style) conditioning."""

    def __init__(
        self,
        channels: int,
        cond_dim: int,
        *,
        n_heads: int = 4,
        dropout: float = 0.1,
        attn_bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.channels = int(channels)
        self.n_heads = int(n_heads)
        self.head_dim = self.channels // self.n_heads
        if self.channels % self.n_heads != 0:
            raise ValueError(f"channels {channels} not divisible by n_heads {n_heads}")
        self.norm1 = nn.LayerNorm(self.channels)
        self.norm2 = nn.LayerNorm(self.channels)
        self.qkv = nn.Linear(self.channels, self.channels * 3)
        self.out_proj = nn.Linear(self.channels, self.channels)
        self.ff = nn.Sequential(
            nn.Linear(self.channels, self.channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.channels * 2, self.channels),
        )
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, self.channels * 6),
        )
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)
        self.register_buffer(
            "attn_bias",
            attn_bias if attn_bias is not None else torch.zeros(1),
            persistent=False,
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B, L, C = h.shape
        params = self.adaLN(cond)
        shift_msa, scale_msa, gate_msa, shift_ff, scale_ff, gate_ff = params.chunk(
            6, dim=-1
        )
        y = self.norm1(h) * (1.0 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        qkv = self.qkv(y).reshape(B, L, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.attn_bias.numel() == L * L:
            attn = attn + self.attn_bias.view(1, 1, L, L)
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)
        y = (attn @ v).transpose(1, 2).reshape(B, L, C)
        y = self.out_proj(y)
        h = h + gate_msa.unsqueeze(1) * self.drop(y)
        y = self.norm2(h) * (1.0 + scale_ff.unsqueeze(1)) + shift_ff.unsqueeze(1)
        h = h + gate_ff.unsqueeze(1) * self.drop(self.ff(y))
        return h


def sequence_distance_bias(seq_len: int, *, n_buckets: int = 32) -> torch.Tensor:
    """Fallback attention bias from sequence distance (monotonic decay)."""
    idx = torch.arange(seq_len, dtype=torch.float32)
    dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    # Map distance to negative bias (closer = less penalty)
    max_d = max(float(dist.max().item()), 1.0)
    norm = dist / max_d
    return (-norm * 2.0).view(1, 1, seq_len, seq_len)


class SiteResolvedDiTHead(nn.Module):
    """Site-resolved residual head with self-attention along L."""

    def __init__(
        self,
        seq_len: int,
        k: int,
        site_in_dim: int,
        cond_dim: int,
        *,
        channels: int = 32,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        attn_bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.k = int(k)
        self.channels = int(channels)
        self.lift = nn.Linear(int(site_in_dim), self.channels)
        if attn_bias is None:
            attn_bias = sequence_distance_bias(self.seq_len)
        self.blocks = nn.ModuleList(
            [
                SiteAdaLNZeroBlock(
                    self.channels,
                    int(cond_dim),
                    n_heads=n_heads,
                    dropout=dropout,
                    attn_bias=attn_bias,
                )
                for _ in range(max(1, int(n_layers)))
            ]
        )
        self.out_norm = nn.LayerNorm(self.channels)
        self.out_proj = nn.Linear(self.channels, self.k)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self, site_feats: torch.Tensor, global_cond: torch.Tensor
    ) -> torch.Tensor:
        h = self.lift(site_feats)
        for block in self.blocks:
            h = block(h, global_cond)
        return self.out_proj(self.out_norm(h))


# ============================================================================
# (merged from pca2d_site_residual_regressor_v2.py)
# ============================================================================

class SiteResolvedConvHead(nn.Module):
    """Lift per-site features → AdaLN–Conv1d → ``L×k`` residual."""

    def __init__(
        self,
        seq_len: int,
        k: int,
        site_in_dim: int,
        cond_dim: int,
        *,
        channels: int = 32,
        n_layers: int = 2,
        kernel_size: int = 5,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.k = int(k)
        self.channels = int(channels)
        self.lift = nn.Linear(int(site_in_dim), self.channels)
        self.blocks = nn.ModuleList(
            [
                SiteAdaLNConvBlock(
                    self.channels,
                    int(cond_dim),
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
                for _ in range(max(1, int(n_layers)))
            ]
        )
        self.out_norm = nn.LayerNorm(self.channels)
        self.out_proj = nn.Linear(self.channels, self.k)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self, site_feats: torch.Tensor, global_cond: torch.Tensor
    ) -> torch.Tensor:
        """``site_feats``: ``[B,L,in]``, ``global_cond``: ``[B,cond]`` → ``[B,L,k]``."""
        h = self.lift(site_feats)
        for block in self.blocks:
            h = block(h, global_cond)
        return self.out_proj(self.out_norm(h))


class Pca2dSiteResidualRegressorV2(nn.Module):
    """Frozen site GPT + site-resolved deterministic residual head."""

    def __init__(
        self,
        site_gpt: GPTModelPca2dSite,
        *,
        denoiser_kind: str = "site_conv",
        cond_dim: int = 256,
        denoiser_hidden: int = 512,
        denoiser_layers: int = 2,
        denoiser_dropout: float = 0.25,
        site_channels: int = 32,
        site_kernel: int = 5,
        site_embed_dim: int = 16,
        residual_scale: float = 1.0,
        horizon_dim: int = 0,
        head_kind: str = "conv",
        use_temporal_pca2d: bool = False,
        temporal_proj_dim: int = 16,
        edge_phylo_dim: int = 0,
        conservation_dim: int = 0,
        substitution_dim: int = 0,
        dit_n_heads: int = 4,
        pca2d_name: str = PCA2D_S1_TARGET,
    ):
        super().__init__()
        if site_gpt.pca2d_bottleneck is None:
            raise ValueError("site_gpt requires pca2d_bottleneck")
        kind = str(denoiser_kind or "site_conv").lower()
        if kind not in ("site_conv", "mlp_bn"):
            raise ValueError(f"denoiser_kind must be site_conv or mlp_bn; got {kind!r}")

        bn = int(site_gpt.pca2d_bottleneck_dim or 0)
        embed = int(site_gpt.hparams.get("embed_dim", 0) or 0)
        self.site_gpt = site_gpt
        self.denoiser_kind = kind
        self.pca2d_name = str(
            getattr(site_gpt, "pca2d_target_name", "") or pca2d_name
        )
        self.bottleneck_dim = bn
        self.embed_dim = embed
        self.flat_dim = int(site_gpt.pca2d_flat_dim or 0)
        self.seq_len = int(getattr(site_gpt, "pca2d_seq_len", 0) or 0)
        self.k = int(getattr(site_gpt, "pca2d_k", 0) or 0)
        self.cond_dim = int(cond_dim)
        self.residual_scale = float(residual_scale)
        self.site_channels = int(site_channels)
        self.site_kernel = int(site_kernel)
        self.site_embed_dim = int(site_embed_dim)
        self.horizon_dim = int(horizon_dim)
        self.head_kind = str(head_kind or "conv").lower()
        self.use_temporal_pca2d = bool(use_temporal_pca2d)
        self.temporal_proj_dim = int(temporal_proj_dim)
        self.edge_phylo_dim = int(edge_phylo_dim)
        self.conservation_dim = int(conservation_dim)
        self.substitution_dim = int(substitution_dim)
        self.dit_n_heads = int(dit_n_heads)

        layout = site_gpt.hparams.get("mean_pca2d_layout") or {}
        pca2d_slice = layout.get("pca2d_slice", [0, self.flat_dim])
        self._pca2d_slice = (int(pca2d_slice[0]), int(pca2d_slice[1]))
        pca2d_gatt = bool(site_gpt.hparams.get("pca2d_gatt_plus", True))
        self._pca2d_gatt_plus = pca2d_gatt

        global_in = embed + bn + max(0, self.horizon_dim) + max(0, self.edge_phylo_dim)
        self.global_cond_mlp = nn.Sequential(
            nn.Linear(global_in, self.cond_dim),
            nn.SiLU(),
            nn.Linear(self.cond_dim, self.cond_dim),
        )
        self.h_proj = nn.Linear(embed, self.cond_dim)

        site_in = self.k * (3 if pca2d_gatt else 2) + self.site_embed_dim + self.cond_dim
        if self.use_temporal_pca2d:
            self.temporal_proj = nn.Linear(2 * self.k, self.temporal_proj_dim)
            site_in += self.temporal_proj_dim
        else:
            self.temporal_proj = None
        if self.conservation_dim > 0:
            self.register_buffer(
                "conservation_stats",
                torch.zeros(self.seq_len, self.conservation_dim),
                persistent=True,
            )
            site_in += self.conservation_dim
        else:
            self.conservation_stats = None
        if self.substitution_dim > 0:
            site_in += self.substitution_dim
        self.site_embed = nn.Embedding(self.seq_len, self.site_embed_dim)

        if kind == "site_conv":
            head_cls = (
                SiteResolvedDiTHead
                if self.head_kind == "dit"
                else SiteResolvedConvHead
            )
            head_kw = dict(
                seq_len=self.seq_len,
                k=self.k,
                site_in_dim=site_in,
                cond_dim=self.cond_dim,
                channels=self.site_channels,
                n_layers=int(denoiser_layers),
                dropout=float(denoiser_dropout),
            )
            if self.head_kind == "dit":
                head_kw["n_heads"] = self.dit_n_heads
            else:
                head_kw["kernel_size"] = self.site_kernel
            self.head = head_cls(**head_kw)
        else:
            self.head = MlpBnResidualHead(
                bn,
                self.cond_dim,
                hidden_dim=int(denoiser_hidden),
                n_layers=int(denoiser_layers),
                dropout=float(denoiser_dropout),
            )

        self.hparams: dict[str, Any] = {
            "pca2d_site_residual_regressor_v2": True,
            "denoiser_kind": kind,
            "flat_dim": self.flat_dim,
            "seq_len": self.seq_len,
            "k": self.k,
            "cond_dim": self.cond_dim,
            "site_channels": self.site_channels,
            "denoiser_layers": int(denoiser_layers),
            "residual_scale": self.residual_scale,
            "pca2d_name": self.pca2d_name,
            "head_kind": self.head_kind,
            "use_temporal_pca2d": self.use_temporal_pca2d,
            "edge_phylo_dim": self.edge_phylo_dim,
            "conservation_dim": self.conservation_dim,
            "substitution_dim": self.substitution_dim,
            "site_gpt_hparams": dict(site_gpt.hparams),
        }

    def set_conservation_stats(self, stats: torch.Tensor) -> None:
        if self.conservation_dim <= 0:
            return
        if stats.shape != (self.seq_len, self.conservation_dim):
            raise ValueError(
                f"conservation stats shape {tuple(stats.shape)} != "
                f"({self.seq_len}, {self.conservation_dim})"
            )
        self.conservation_stats.copy_(stats.to(self.conservation_stats.device))

    def trainable_parameters(self):
        yield from self.global_cond_mlp.parameters()
        yield from self.h_proj.parameters()
        yield from self.site_embed.parameters()
        if self.temporal_proj is not None:
            yield from self.temporal_proj.parameters()
        yield from self.head.parameters()

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def _to_sites(self, flat: torch.Tensor) -> torch.Tensor:
        return flat.reshape(flat.shape[0], self.seq_len, self.k)

    def _from_sites(self, sites: torch.Tensor) -> torch.Tensor:
        return sites.reshape(sites.shape[0], self.flat_dim)

    def _parse_pca2d_slice(self, xq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat = int(self.flat_dim)
        if xq.size(-1) == flat:
            abs_pca = xq
            vel = torch.zeros_like(abs_pca)
        elif xq.size(-1) == 2 * flat:
            abs_pca = xq[..., :flat]
            vel = xq[..., flat:] if self._pca2d_gatt_plus else torch.zeros_like(abs_pca)
        else:
            raise ValueError(f"pca2d slice width {xq.size(-1)} != flat or 2*flat")
        return abs_pca, vel

    def _extract_pca2d_input(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Last-step pca2d abs (+ velocity) from model input ``[B,T,D]``."""
        s, e = self._pca2d_slice
        return self._parse_pca2d_slice(x[:, -1, s:e])

    def _extract_pca2d_temporal(self, x: torch.Tensor) -> torch.Tensor:
        """Prefix mean abs pca2d + mean velocity per site ``[B,L,2k]``."""
        s, e = self._pca2d_slice
        xq = x[:, :, s:e]
        abs_all = []
        for t in range(xq.shape[1]):
            abs_t, _ = self._parse_pca2d_slice(xq[:, t])
            abs_all.append(self._to_sites(abs_t))
        stack = torch.stack(abs_all, dim=1)
        t_mean = stack.mean(dim=1)
        if stack.shape[1] >= 2:
            t_vel = (stack[:, 1:] - stack[:, :-1]).mean(dim=1)
        else:
            t_vel = torch.zeros_like(t_mean)
        return torch.cat([t_mean, t_vel], dim=-1)

    def _build_global_cond(
        self,
        h_last: torch.Tensor,
        mu_z: torch.Tensor,
        *,
        horizon: Optional[torch.Tensor] = None,
        edge_phylo: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        parts = [h_last, mu_z]
        if horizon is not None:
            parts.append(horizon)
        elif self.horizon_dim > 0:
            parts.append(
                torch.zeros(
                    h_last.shape[0],
                    self.horizon_dim,
                    device=h_last.device,
                    dtype=h_last.dtype,
                )
            )
        if self.edge_phylo_dim > 0:
            if edge_phylo is not None:
                parts.append(edge_phylo)
            else:
                parts.append(
                    torch.zeros(
                        h_last.shape[0],
                        self.edge_phylo_dim,
                        device=h_last.device,
                        dtype=h_last.dtype,
                    )
                )
        return self.global_cond_mlp(torch.cat(parts, dim=-1))

    def build_site_features(
        self,
        x: torch.Tensor,
        mu_flat: torch.Tensor,
        h_last: torch.Tensor,
        *,
        mu_z: Optional[torch.Tensor] = None,
        horizon: Optional[torch.Tensor] = None,
        edge_phylo: Optional[torch.Tensor] = None,
        substitution_feats: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        abs_pca, vel = self._extract_pca2d_input(x)
        mu_s = self._to_sites(mu_flat)
        abs_s = self._to_sites(abs_pca)
        vel_s = self._to_sites(vel)
        B = x.shape[0]
        idx = torch.arange(self.seq_len, device=x.device).unsqueeze(0).expand(B, -1)
        emb = self.site_embed(idx)
        h_b = self.h_proj(h_last).unsqueeze(1).expand(B, self.seq_len, -1)
        parts = [abs_s, vel_s, mu_s, emb, h_b]
        if self.use_temporal_pca2d and self.temporal_proj is not None:
            t_mean = self._extract_pca2d_temporal(x)
            parts.append(self.temporal_proj(t_mean))
        if self.conservation_dim > 0 and self.conservation_stats is not None:
            cons = self.conservation_stats.unsqueeze(0).expand(B, -1, -1)
            parts.append(cons)
        if self.substitution_dim > 0:
            if substitution_feats is None:
                substitution_feats = torch.zeros(
                    B,
                    self.seq_len,
                    self.substitution_dim,
                    device=x.device,
                    dtype=x.dtype,
                )
            parts.append(substitution_feats)
        site_feats = torch.cat(parts, dim=-1)
        if mu_z is None:
            mu_z, _, _, _ = self.predict_anchor(x)
        global_cond = self._build_global_cond(
            h_last, mu_z, horizon=horizon, edge_phylo=edge_phylo
        )
        return site_feats, global_cond

    @torch.no_grad()
    def predict_anchor(
        self, x: torch.Tensor, *, return_h: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """Return ``(mu_z, h_last, mu_flat, mean_anchor)`` from frozen site-GPT."""
        from recomb2027.core.constants import PRIMARY_TARGET

        was = self.site_gpt.training
        self.site_gpt.eval()
        h = self.site_gpt.encode_trunk(x)
        out_z = self.site_gpt._decode_diff(h, expand_pca2d=False)
        mu_z = out_z[self.pca2d_name][:, -1, :]
        h_last = h[:, -1, :]
        mu_flat = self.site_gpt.pca2d_bottleneck.decode(mu_z, gate_cond=h_last)
        if PRIMARY_TARGET in out_z:
            mean_anchor = out_z[PRIMARY_TARGET][:, -1, :]
        else:
            # Fallback: first LM head / primary slice if mean head missing
            mean_anchor = next(iter(out_z.values()))[:, -1, :]
        self.site_gpt.train(was)
        if return_h:
            return mu_z, h_last, mu_flat, mean_anchor, h
        return mu_z, h_last, mu_flat, mean_anchor

    def predict_flat(
        self,
        x: torch.Tensor,
        *,
        horizon: Optional[torch.Tensor] = None,
        edge_phylo: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu_z, h_last, mu_flat, _ = self.predict_anchor(x)
        if self.denoiser_kind == "site_conv":
            site_feats, global_cond = self.build_site_features(
                x,
                mu_flat,
                h_last,
                mu_z=mu_z,
                horizon=horizon,
                edge_phylo=edge_phylo,
            )
            pred_r = self.head(site_feats, global_cond) * self.residual_scale
            pred_flat = mu_flat + self._from_sites(pred_r)
            return pred_flat, mu_flat, pred_r
        global_cond = self._build_global_cond(
            h_last, mu_z, horizon=horizon, edge_phylo=edge_phylo
        )
        pred_r = self.head(global_cond) * self.residual_scale
        pred_z = mu_z + pred_r
        pred_flat = self.site_gpt.pca2d_bottleneck.decode(pred_z, gate_cond=h_last)
        return pred_flat, mu_flat, pred_r

    def forward(
        self,
        x: torch.Tensor,
        y_pca_delta: torch.Tensor,
        *,
        sample_weight: Optional[torch.Tensor] = None,
        horizon: Optional[torch.Tensor] = None,
        edge_phylo: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if y_pca_delta.dim() == 3:
            target_flat = y_pca_delta[:, -1, :]
        elif y_pca_delta.dim() == 2:
            target_flat = y_pca_delta
        else:
            raise ValueError(f"bad y shape {tuple(y_pca_delta.shape)}")

        with torch.no_grad():
            mu_z, h_last, mu_flat, _ = self.predict_anchor(x)
            if self.denoiser_kind == "site_conv":
                residual = self._to_sites(target_flat - mu_flat)
            else:
                residual = (
                    self.site_gpt.pca2d_bottleneck.encode(target_flat, condition=None)
                    - mu_z
                )

        if self.denoiser_kind == "site_conv":
            site_feats, global_cond = self.build_site_features(
                x,
                mu_flat,
                h_last,
                mu_z=mu_z,
                horizon=horizon,
                edge_phylo=edge_phylo,
            )
            pred_r = self.head(site_feats, global_cond) * self.residual_scale
            pred_flat = mu_flat + self._from_sites(pred_r)
        else:
            global_cond = self._build_global_cond(
                h_last, mu_z, horizon=horizon, edge_phylo=edge_phylo
            )
            pred_r = self.head(global_cond) * self.residual_scale
            pred_z = mu_z + pred_r
            pred_flat = self.site_gpt.pca2d_bottleneck.decode(
                pred_z, gate_cond=h_last
            )

        if sample_weight is not None:
            w = sample_weight.reshape(-1, *([1] * (pred_r.dim() - 1)))
            residual_mse = ((pred_r - residual).pow(2) * w).mean()
        else:
            residual_mse = F.mse_loss(pred_r, residual)

        flat_mse = F.mse_loss(pred_flat, target_flat)
        mu_mse = F.mse_loss(mu_flat, target_flat)
        loss_info = {
            "residual_mse": residual_mse.detach(),
            "full_x0_mse": flat_mse.detach(),
            "mu_mse": mu_mse.detach(),
        }
        return residual_mse, pred_flat, target_flat, loss_info


# ============================================================================
# (merged from site_cross_attn_dit_head.py)
# ============================================================================

class SiteCrossAttnDiTBlock(nn.Module):
    """Self-attn along L + cross-attn to prefix context memory."""

    def __init__(
        self,
        channels: int,
        cond_dim: int,
        mem_dim: int,
        *,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_block = SiteAdaLNZeroBlock(
            channels, cond_dim, n_heads=n_heads, dropout=dropout
        )
        self.channels = int(channels)
        self.n_heads = int(n_heads)
        self.head_dim = self.channels // self.n_heads
        self.norm_cross = nn.LayerNorm(self.channels)
        self.q_proj = nn.Linear(self.channels, self.channels)
        self.kv_proj = nn.Linear(int(mem_dim), self.channels * 2)
        self.out_cross = nn.Linear(self.channels, self.channels)
        self.cross_ada = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, self.channels * 3),
        )
        nn.init.zeros_(self.cross_ada[-1].weight)
        nn.init.zeros_(self.cross_ada[-1].bias)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        cond: torch.Tensor,
        memory: torch.Tensor,
        *,
        mem_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.self_block(h, cond)
        B, L, C = h.shape
        _, T, _ = memory.shape
        cparams = self.cross_ada(cond)
        shift, scale, gate = cparams.chunk(3, dim=-1)
        y = self.norm_cross(h) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        q = self.q_proj(y).reshape(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(memory).reshape(B, T, 2, self.n_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mem_mask is not None:
            attn = attn.masked_fill(~mem_mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)
        y = (attn @ v).transpose(1, 2).reshape(B, L, C)
        y = self.out_cross(y)
        return h + gate.unsqueeze(1) * self.drop(y)


class SiteCrossAttnDiTHead(nn.Module):
    """Site-resolved head with probabilistic outputs per site."""

    def __init__(
        self,
        seq_len: int,
        k: int,
        site_in_dim: int,
        cond_dim: int,
        mem_dim: int,
        *,
        channels: int = 96,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
        cov_rank: int = 4,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.k = int(k)
        self.channels = int(channels)
        self.cov_rank = int(cov_rank)
        self.lift = nn.Linear(int(site_in_dim), self.channels)
        self.blocks = nn.ModuleList(
            [
                SiteCrossAttnDiTBlock(
                    self.channels,
                    int(cond_dim),
                    int(mem_dim),
                    n_heads=n_heads,
                    dropout=dropout,
                )
                for _ in range(max(1, int(n_layers)))
            ]
        )
        self.out_norm = nn.LayerNorm(self.channels)
        self.mu_proj = nn.Linear(self.channels, self.k)
        self.log_sigma_proj = nn.Linear(self.channels, self.k)
        self.U_proj = nn.Linear(self.channels, self.k * self.cov_rank)
        self.vel_proj = nn.Linear(self.channels, self.k)
        self.flow_state_proj = nn.Linear(self.k, self.channels)
        self.flow_time_mlp = nn.Sequential(
            nn.Linear(1, int(cond_dim)),
            nn.SiLU(),
            nn.Linear(int(cond_dim), self.channels),
        )
        nn.init.zeros_(self.mu_proj.weight)
        nn.init.zeros_(self.mu_proj.bias)
        nn.init.zeros_(self.vel_proj.weight)
        nn.init.zeros_(self.vel_proj.bias)
        nn.init.zeros_(self.flow_state_proj.weight)
        nn.init.zeros_(self.flow_state_proj.bias)
        nn.init.zeros_(self.flow_time_mlp[-1].weight)
        nn.init.zeros_(self.flow_time_mlp[-1].bias)

    def forward(
        self,
        site_feats: torch.Tensor,
        global_cond: torch.Tensor,
        memory: torch.Tensor,
        *,
        mem_mask: Optional[torch.Tensor] = None,
        return_velocity: bool = False,
        flow_state: Optional[torch.Tensor] = None,
        flow_t: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> dict[str, torch.Tensor]:
        h = self.lift(site_feats)
        if flow_state is not None:
            h = h + self.flow_state_proj(flow_state)
        if flow_t is not None:
            t_emb = self.flow_time_mlp(flow_t.reshape(-1, 1).to(h.dtype))
            h = h + t_emb.unsqueeze(1)
        for block in self.blocks:
            h = block(h, global_cond, memory, mem_mask=mem_mask)
        h = self.out_norm(h)
        out = {
            "mu_r": self.mu_proj(h),
            "log_sigma": self.log_sigma_proj(h),
            "U": self.U_proj(h).reshape(h.shape[0], self.seq_len, self.k, self.cov_rank),
        }
        if return_velocity:
            out["velocity"] = self.vel_proj(h)
        if return_hidden:
            # Post-DiT site features ``[B, L, channels]``: already conditioned on
            # ``global_cond`` (AdaLN-Zero: horizon/edge/trunk) and on the prefix
            # context memory (cross-attention). Used by the activation gate.
            out["hidden"] = h
        return out


# ============================================================================
# (merged from pca2d_site_unified_v3.py)
# ============================================================================

def _build_mean_residual_mlp(cond_dim: int, mean_dim: int, hidden: int) -> nn.Sequential:
    """Pooled mean residual: last Linear zero-init so Δmean≈0 at init."""
    last = nn.Linear(hidden, mean_dim)
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)
    return nn.Sequential(
        nn.Linear(cond_dim, hidden),
        nn.GELU(),
        last,
    )


class Pca2dSiteUnifiedV3(nn.Module):
    """Frozen ABC anchor + cross-attention site head with low-rank Gaussian / bridge flow."""

    def __init__(
        self,
        site_gpt: GPTModelPca2dSite,
        *,
        cond_dim: int = 256,
        mem_dim: int = 128,
        site_channels: int = 96,
        dit_layers: int = 4,
        dit_heads: int = 4,
        dropout: float = 0.1,
        cov_rank: int = 4,
        spike_sigma: float = 1e-3,
        horizon_dim: int = 4,
        edge_phylo_dim: int = 1,
        use_temporal_pca2d: bool = True,
        temporal_proj_dim: int = 16,
        conservation_dim: int = 0,
        substitution_dim: int = 0,
        use_activation_supervision: bool = False,
        activation_aux_weight: float = 0.5,
        use_mean_residual: bool = False,
        mean_aux_weight: float = 1.0,
        mean_dim: Optional[int] = None,
        bridge_sigma: float = 0.1,
        pca2d_name: str = PCA2D_S1_TARGET,
        activation_head_input: str = "site_feats",
        activation_pos_weight: float = 1.0,
    ):
        """``activation_head_input``:

        - ``"site_feats"`` (legacy, checkpoint-compatible): gate sees only the
          pre-DiT per-site token, so it can never depend on ``horizon`` /
          ``edge_phylo`` (those only enter ``global_cond``).
        - ``"dit"``: gate sees the post-DiT per-site features (AdaLN-conditioned
          on ``global_cond`` + cross-attention to the prefix memory), so the
          per-site mutation probability can respond to the horizon condition.

        ``activation_pos_weight`` up-weights mutated sites in the BCE (only about
        0.5% of sites mutate per hop); 1.0 = plain BCE.
        """
        super().__init__()
        self.feature_builder = Pca2dSiteResidualRegressorV2(
            site_gpt,
            denoiser_kind="site_conv",
            cond_dim=cond_dim,
            site_channels=site_channels,
            denoiser_layers=1,
            horizon_dim=horizon_dim,
            edge_phylo_dim=edge_phylo_dim,
            use_temporal_pca2d=use_temporal_pca2d,
            temporal_proj_dim=temporal_proj_dim,
            conservation_dim=conservation_dim,
            substitution_dim=substitution_dim,
            head_kind="conv",
            pca2d_name=pca2d_name,
        )
        self.substitution_dim = int(substitution_dim)
        self.use_activation_supervision = bool(use_activation_supervision)
        self.activation_aux_weight = float(activation_aux_weight)
        self.use_mean_residual = bool(use_mean_residual)
        self.mean_aux_weight = float(mean_aux_weight)
        self.site_gpt = site_gpt
        self.pca2d_name = str(pca2d_name)
        self.flat_dim = int(site_gpt.pca2d_flat_dim or 0)
        self.seq_len = int(getattr(site_gpt, "pca2d_seq_len", 0) or 0)
        self.k = int(getattr(site_gpt, "pca2d_k", 0) or 0)
        self.cond_dim = int(cond_dim)
        self.bridge_sigma = float(bridge_sigma)
        out_dims = getattr(site_gpt, "output_dims", None) or {}
        if mean_dim is not None:
            self.mean_dim = int(mean_dim)
        else:
            self.mean_dim = int(out_dims.get(PRIMARY_TARGET, 1280) or 1280)
        embed = int(site_gpt.hparams.get("embed_dim", 384) or 384)
        self.context_memory = Pca2dContextMemory(embed, mem_dim)
        pca2d_gatt = bool(site_gpt.hparams.get("pca2d_gatt_plus", True))
        site_in = self.k * (3 if pca2d_gatt else 2) + self.feature_builder.site_embed_dim
        site_in += self.cond_dim
        if use_temporal_pca2d:
            site_in += temporal_proj_dim
        if conservation_dim > 0:
            site_in += conservation_dim
        if substitution_dim > 0:
            site_in += substitution_dim
        self.head = SiteCrossAttnDiTHead(
            self.seq_len,
            self.k,
            site_in,
            cond_dim,
            mem_dim,
            channels=site_channels,
            n_layers=dit_layers,
            n_heads=dit_heads,
            dropout=dropout,
            cov_rank=cov_rank,
        )
        if activation_head_input not in ("site_feats", "dit"):
            raise ValueError(
                "activation_head_input must be 'site_feats' or 'dit', "
                f"got {activation_head_input!r}"
            )
        self.activation_head_input = str(activation_head_input)
        self.activation_pos_weight = float(activation_pos_weight)
        self.activation_head: Optional[nn.Linear]
        if self.use_activation_supervision:
            gate_in = site_in if self.activation_head_input == "site_feats" else int(site_channels)
            self.activation_head = nn.Linear(gate_in, 1)
        else:
            self.activation_head = None
        self.mean_residual_mlp: Optional[nn.Sequential]
        if self.use_mean_residual:
            hid = max(int(cond_dim), 128)
            self.mean_residual_mlp = _build_mean_residual_mlp(
                int(cond_dim), self.mean_dim, hid
            )
        else:
            self.mean_residual_mlp = None
        self.hparams: dict[str, Any] = {
            "pca2d_site_unified_v3": True,
            "flat_dim": self.flat_dim,
            "seq_len": self.seq_len,
            "k": self.k,
            "bridge_sigma": self.bridge_sigma,
            "pca2d_name": self.pca2d_name,
            "use_mean_residual": self.use_mean_residual,
            "mean_dim": self.mean_dim,
            "activation_head_input": self.activation_head_input,
            "activation_pos_weight": self.activation_pos_weight,
        }

    def _run_head(
        self,
        site_feats: torch.Tensor,
        global_cond: torch.Tensor,
        memory: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Run the DiT head, also returning post-DiT features when the gate needs them."""
        need_hidden = (
            self.activation_head is not None and self.activation_head_input == "dit"
        )
        if need_hidden:
            kwargs["return_hidden"] = True
        return self.head(site_feats, global_cond, memory, **kwargs)

    def set_conservation_stats(self, stats: torch.Tensor) -> None:
        self.feature_builder.set_conservation_stats(stats)

    def trainable_parameters(self):
        yield from self.context_memory.parameters()
        yield from self.head.parameters()
        yield from self.feature_builder.global_cond_mlp.parameters()
        yield from self.feature_builder.h_proj.parameters()
        yield from self.feature_builder.site_embed.parameters()
        if self.feature_builder.temporal_proj is not None:
            yield from self.feature_builder.temporal_proj.parameters()
        if self.activation_head is not None:
            yield from self.activation_head.parameters()
        if self.mean_residual_mlp is not None:
            yield from self.mean_residual_mlp.parameters()

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    @torch.no_grad()
    def predict_anchor(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu_z, h_last, mu_flat, mean_anchor = self.feature_builder.predict_anchor(x)
        h_seq = self.site_gpt.encode_trunk(x)
        return mu_z, h_last, mu_flat, h_seq, mean_anchor

    def _encode(
        self,
        x: torch.Tensor,
        *,
        horizon: Optional[torch.Tensor] = None,
        edge_phylo: Optional[torch.Tensor] = None,
        substitution_feats: Optional[torch.Tensor] = None,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        mu_z, h_last, mu_flat, h_seq, mean_anchor = self.predict_anchor(x)
        memory = self.context_memory(h_seq)
        site_feats, global_cond = self.feature_builder.build_site_features(
            x,
            mu_flat,
            h_last,
            mu_z=mu_z,
            horizon=horizon,
            edge_phylo=edge_phylo,
            substitution_feats=substitution_feats,
        )
        return mu_flat, site_feats, global_cond, memory, h_seq, mean_anchor

    def _apply_mean_residual(
        self,
        global_cond: torch.Tensor,
        mean_anchor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(pred_mean, mean_anchor, delta_mean)``."""
        if self.mean_residual_mlp is None:
            zeros = mean_anchor.new_zeros(mean_anchor.shape)
            return mean_anchor, mean_anchor, zeros
        delta = self.mean_residual_mlp(global_cond)
        return mean_anchor + delta, mean_anchor, delta

    def _apply_activation_gate(
        self,
        head_out: dict[str, torch.Tensor],
        site_feats: torch.Tensor,
        *,
        activation_target: Optional[torch.Tensor] = None,
    ) -> tuple[dict[str, torch.Tensor], Optional[torch.Tensor]]:
        if self.activation_head is None:
            return head_out, None
        head_out = dict(head_out)
        hidden = head_out.pop("hidden", None)
        if self.activation_head_input == "dit":
            if hidden is None:
                raise RuntimeError(
                    "activation_head_input='dit' needs head_out['hidden']; "
                    "call the head via _run_head()"
                )
            gate_in = hidden
        else:
            gate_in = site_feats
        logits = self.activation_head(gate_in).squeeze(-1)
        gate = torch.sigmoid(logits).unsqueeze(-1)
        head_out["mu_r"] = head_out["mu_r"] * gate
        head_out["activation_logits"] = logits
        head_out["activation_probs"] = gate.squeeze(-1)
        act_loss = None
        if activation_target is not None:
            # Training may pass [B, L]; callers that forgot step-select may pass
            # [B, T, L] — resolve to last step (matches last-step eval / infer).
            target = activation_target
            if target.dim() == 3:
                target = target[:, -1]
            if target.shape != logits.shape:
                raise ValueError(
                    "activation_target shape "
                    f"{tuple(activation_target.shape)} incompatible with logits "
                    f"{tuple(logits.shape)} (expected [B, L] or [B, T, L])"
                )
            pos_weight = None
            if self.activation_pos_weight != 1.0:
                pos_weight = logits.new_tensor(self.activation_pos_weight)
            act_loss = F.binary_cross_entropy_with_logits(
                logits, target.to(logits.dtype), pos_weight=pos_weight
            )
        return head_out, act_loss

    def _mean_aux_loss(
        self,
        pred_mean: torch.Tensor,
        y_mean: Optional[torch.Tensor],
        *,
        step_index: Optional[int] = None,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if (
            not self.use_mean_residual
            or y_mean is None
            or self.mean_aux_weight <= 0.0
        ):
            return None
        target = self._resolve_target(y_mean, step_index=step_index)
        per = (pred_mean - target).pow(2).mean(dim=-1)
        if sample_weight is not None:
            w = sample_weight.reshape(-1).to(dtype=per.dtype)
            return (per * w).sum() / w.sum().clamp(min=1e-12)
        return per.mean()

    def predict_distribution(
        self,
        x: torch.Tensor,
        *,
        horizon: Optional[torch.Tensor] = None,
        edge_phylo: Optional[torch.Tensor] = None,
        substitution_feats: Optional[torch.Tensor] = None,
        activation_target: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        mu_flat, site_feats, global_cond, memory, _, mean_anchor = self._encode(
            x,
            horizon=horizon,
            edge_phylo=edge_phylo,
            substitution_feats=substitution_feats,
        )
        out = self._run_head(site_feats, global_cond, memory)
        # Aux BCE is computed only in training mode (``mode="mse"`` trains through
        # this method, so dropping the target here silently disabled the gate
        # supervision). Eval / inference only applies the gate.
        out, act_loss = self._apply_activation_gate(
            out,
            site_feats,
            activation_target=activation_target if self.training else None,
        )
        if act_loss is not None:
            out["activation_loss"] = act_loss
        out["mu_flat"] = mu_flat
        out["pred_flat"] = point_pred_from_residual(mu_flat, out["mu_r"])
        pred_mean, mean_anchor, delta_mean = self._apply_mean_residual(
            global_cond, mean_anchor
        )
        out["pred_mean"] = pred_mean
        out["mean_anchor"] = mean_anchor
        out["delta_mean"] = delta_mean
        return out

    def _resolve_target(
        self, y_pca_delta: torch.Tensor, *, step_index: Optional[int] = None
    ) -> torch.Tensor:
        if y_pca_delta.dim() == 2:
            return y_pca_delta
        if step_index is not None:
            return y_pca_delta[:, int(step_index), :]
        return y_pca_delta[:, -1, :]

    def forward_nll(
        self,
        x: torch.Tensor,
        y_pca_delta: torch.Tensor,
        *,
        horizon: Optional[torch.Tensor] = None,
        edge_phylo: Optional[torch.Tensor] = None,
        mse_aux_weight: float = 0.0,
        sample_weight: Optional[torch.Tensor] = None,
        step_index: Optional[int] = None,
        substitution_feats: Optional[torch.Tensor] = None,
        activation_target: Optional[torch.Tensor] = None,
        y_mean: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
        target_flat = self._resolve_target(y_pca_delta, step_index=step_index)
        mu_flat, site_feats, global_cond, memory, _, mean_anchor = self._encode(
            x,
            horizon=horizon,
            edge_phylo=edge_phylo,
            substitution_feats=substitution_feats,
        )
        out = self._run_head(site_feats, global_cond, memory)
        out, act_loss = self._apply_activation_gate(
            out, site_feats, activation_target=activation_target
        )
        nll = low_rank_site_nll(
            target_flat,
            mu_flat,
            out["mu_r"],
            out["log_sigma"],
            out["U"],
            seq_len=self.seq_len,
            k=self.k,
            sample_weight=sample_weight,
        )
        pred_flat = point_pred_from_residual(mu_flat, out["mu_r"])
        pred_mean, mean_anchor, delta_mean = self._apply_mean_residual(
            global_cond, mean_anchor
        )
        mu_mse = F.mse_loss(mu_flat, target_flat)
        full_mse = F.mse_loss(pred_flat, target_flat)
        loss = nll
        if mse_aux_weight > 0.0:
            loss = loss + float(mse_aux_weight) * full_mse
        if act_loss is not None and self.activation_aux_weight > 0.0:
            loss = loss + float(self.activation_aux_weight) * act_loss
        mean_loss = self._mean_aux_loss(
            pred_mean, y_mean, step_index=step_index, sample_weight=sample_weight
        )
        if mean_loss is not None:
            loss = loss + float(self.mean_aux_weight) * mean_loss
        info = {
            "nll": nll.detach(),
            "full_x0_mse": full_mse.detach(),
            "mu_mse": mu_mse.detach(),
            "pred_flat": pred_flat,
            "pred_mean": pred_mean,
            "mean_anchor": mean_anchor,
            "delta_mean": delta_mean,
        }
        if act_loss is not None:
            info["activation_loss"] = act_loss.detach()
        if mean_loss is not None:
            info["mean_mse"] = mean_loss.detach()
        return loss, info

    def forward_flow(
        self,
        x: torch.Tensor,
        y_pca_delta: torch.Tensor,
        *,
        horizon: Optional[torch.Tensor] = None,
        edge_phylo: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        sample_weight: Optional[torch.Tensor] = None,
        step_index: Optional[int] = None,
        substitution_feats: Optional[torch.Tensor] = None,
        activation_target: Optional[torch.Tensor] = None,
        y_mean: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Bridge flow: x0=0 residual, x1=true residual around mu."""
        target_flat = self._resolve_target(y_pca_delta, step_index=step_index)
        mu_flat, site_feats, global_cond, memory, _, mean_anchor = self._encode(
            x,
            horizon=horizon,
            edge_phylo=edge_phylo,
            substitution_feats=substitution_feats,
        )
        x1 = (target_flat - mu_flat).reshape(-1, self.seq_len, self.k)
        x0 = torch.zeros_like(x1)
        B = x1.shape[0]
        if t is None:
            t = torch.rand(B, device=x1.device)
        t_view = t.view(B, 1, 1)
        eps = torch.randn_like(x1)
        bridge = self.bridge_sigma * torch.sqrt(t_view * (1.0 - t_view).clamp(min=1e-8))
        xt = (1.0 - t_view) * x0 + t_view * x1 + bridge * eps
        target_u = x1 - x0
        out = self.head(
            site_feats,
            global_cond,
            memory,
            return_velocity=True,
            flow_state=xt,
            flow_t=t,
        )
        pred_u = out["velocity"]
        per_sample = (pred_u - target_u).pow(2).mean(dim=(1, 2))
        if sample_weight is not None:
            w = sample_weight.reshape(-1).to(dtype=per_sample.dtype)
            fm_loss = (per_sample * w).sum() / w.sum().clamp(min=1e-12)
        else:
            fm_loss = per_sample.mean()
        pred_flat = mu_flat + (x0 + pred_u).reshape(B, -1)
        pred_mean, mean_anchor, delta_mean = self._apply_mean_residual(
            global_cond, mean_anchor
        )
        loss = fm_loss
        mean_loss = self._mean_aux_loss(
            pred_mean, y_mean, step_index=step_index, sample_weight=sample_weight
        )
        if mean_loss is not None:
            loss = loss + float(self.mean_aux_weight) * mean_loss
        info = {
            "flow_loss": fm_loss.detach(),
            "full_x0_mse": F.mse_loss(pred_flat, target_flat).detach(),
            "mu_mse": F.mse_loss(mu_flat, target_flat).detach(),
            "pred_flat": pred_flat,
            "pred_mean": pred_mean,
            "mean_anchor": mean_anchor,
            "delta_mean": delta_mean,
        }
        if mean_loss is not None:
            info["mean_mse"] = mean_loss.detach()
        return loss, info

    def forward(
        self,
        x: torch.Tensor,
        y_pca_delta: torch.Tensor,
        *,
        mode: str = "nll",
        horizon: Optional[torch.Tensor] = None,
        edge_phylo: Optional[torch.Tensor] = None,
        mse_aux_weight: float = 0.0,
        sample_weight: Optional[torch.Tensor] = None,
        step_index: Optional[int] = None,
        substitution_feats: Optional[torch.Tensor] = None,
        activation_target: Optional[torch.Tensor] = None,
        y_mean: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if mode == "flow":
            loss, info = self.forward_flow(
                x,
                y_pca_delta,
                horizon=horizon,
                edge_phylo=edge_phylo,
                sample_weight=sample_weight,
                step_index=step_index,
                substitution_feats=substitution_feats,
                activation_target=activation_target,
                y_mean=y_mean,
            )
            pred_flat = info["pred_flat"]
        elif mode == "mse":
            dist = self.predict_distribution(
                x,
                horizon=horizon,
                edge_phylo=edge_phylo,
                substitution_feats=substitution_feats,
                activation_target=activation_target,
            )
            target_flat = self._resolve_target(y_pca_delta, step_index=step_index)
            pred_flat = dist["pred_flat"]
            per_sample = (pred_flat - target_flat).pow(2).mean(dim=-1)
            if sample_weight is not None:
                w = sample_weight.reshape(-1).to(dtype=per_sample.dtype)
                loss = (per_sample * w).sum() / w.sum().clamp(min=1e-12)
            else:
                loss = per_sample.mean()
            mean_loss = self._mean_aux_loss(
                dist["pred_mean"],
                y_mean,
                step_index=step_index,
                sample_weight=sample_weight,
            )
            if mean_loss is not None:
                loss = loss + float(self.mean_aux_weight) * mean_loss
            act_loss = dist.get("activation_loss")
            if act_loss is not None and self.activation_aux_weight > 0.0:
                loss = loss + float(self.activation_aux_weight) * act_loss
            info = {
                "full_x0_mse": per_sample.mean().detach(),
                "mu_mse": F.mse_loss(dist["mu_flat"], target_flat).detach(),
                "pred_flat": pred_flat,
                "pred_mean": dist["pred_mean"],
                "mean_anchor": dist["mean_anchor"],
                "delta_mean": dist["delta_mean"],
            }
            if act_loss is not None:
                info["activation_loss"] = act_loss.detach()
            if mean_loss is not None:
                info["mean_mse"] = mean_loss.detach()
        else:
            loss, info = self.forward_nll(
                x,
                y_pca_delta,
                horizon=horizon,
                edge_phylo=edge_phylo,
                mse_aux_weight=mse_aux_weight,
                sample_weight=sample_weight,
                step_index=step_index,
                substitution_feats=substitution_feats,
                activation_target=activation_target,
                y_mean=y_mean,
            )
            pred_flat = info["pred_flat"]
        return loss, pred_flat, info

    @torch.no_grad()
    def sample_bridge_flow(
        self,
        x: torch.Tensor,
        *,
        steps: int = 20,
        horizon: Optional[torch.Tensor] = None,
        edge_phylo: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mu_flat, site_feats, global_cond, memory, _, _ = self._encode(
            x, horizon=horizon, edge_phylo=edge_phylo
        )
        B = x.shape[0]
        xt = torch.zeros(B, self.seq_len, self.k, device=x.device)
        dt = 1.0 / max(int(steps), 1)
        for i in range(max(int(steps), 1)):
            t_val = i * dt
            t = torch.full((B,), t_val, device=x.device)
            out = self.head(
                site_feats,
                global_cond,
                memory,
                return_velocity=True,
                flow_state=xt,
                flow_t=t,
            )
            xt = xt + dt * out["velocity"]
        return mu_flat + xt.reshape(B, -1)


# ============================================================================
# (merged from pca2d_site_unified_v4.py)
# ============================================================================

logger = logging.getLogger(__name__)


class Pca2dSiteUnifiedV4(Pca2dSiteUnifiedV3):
    """B4: mean is a readout of gated ``μ_r``; optional weak independent MLP."""

    def __init__(
        self,
        site_gpt,
        *,
        mean_readout: Optional[FrozenPca2dDeltaReadout],
        consistency_weight: float = 0.2,
        use_independent_mean_mlp: bool = False,
        stop_grad_pca2d_for_consistency: bool = True,
        use_ridge_readout: bool = True,
        **kwargs,
    ):
        if not use_ridge_readout:
            # B3-equivalent mean path: delta_mean = MLP(global_cond) only. No gradient from the
            # mean loss reaches the DiT residual head and no cross-view consistency exists.
            if not use_independent_mean_mlp:
                raise ValueError(
                    "use_ridge_readout=False requires use_independent_mean_mlp=True "
                    "(otherwise the model would have no mean head at all)."
                )
            if float(consistency_weight) > 0.0:
                logger.warning(
                    "Pca2dSiteUnifiedV4: consistency_weight=%.3g ignored because "
                    "use_ridge_readout=False (B3-equivalent mean path, no coupling).",
                    float(consistency_weight),
                )
            consistency_weight = 0.0
            mean_readout = None
        elif mean_readout is None:
            raise ValueError("use_ridge_readout=True requires a mean_readout.")
        if not use_independent_mean_mlp:
            if kwargs.get("use_mean_residual"):
                logger.warning(
                    "Pca2dSiteUnifiedV4: use_mean_residual=True ignored because "
                    "use_independent_mean_mlp=False (mean = readout of gated mu_r only)."
                )
            kwargs["use_mean_residual"] = False
        self.mean_product_weight = float(kwargs.get("mean_aux_weight", 0.3))
        super().__init__(site_gpt, **kwargs)
        self.mean_readout = mean_readout
        self.consistency_weight_requested = float(consistency_weight)
        if not use_independent_mean_mlp and float(consistency_weight) > 0.0:
            # pred_delta_mean == readout(gated mu_r) and the target is
            # readout(sg(gated mu_r)): the term is identically zero with zero
            # gradient, so it is disabled rather than silently kept as a no-op.
            logger.warning(
                "Pca2dSiteUnifiedV4: consistency_weight=%.3g is a zero-gradient no-op "
                "without an independent mean MLP; setting effective weight to 0. "
                "Set use_independent_mean_mlp=true to get a real cross-view consistency term.",
                float(consistency_weight),
            )
            consistency_weight = 0.0
        self.consistency_weight = float(consistency_weight)
        self.use_independent_mean_mlp = bool(use_independent_mean_mlp)
        self.use_ridge_readout = bool(use_ridge_readout)
        self.stop_grad_pca2d_for_consistency = bool(stop_grad_pca2d_for_consistency)
        self.hparams = dict(self.hparams)
        self.hparams.update(
            {
                "pca2d_site_unified_v4": True,
                "consistency_weight": self.consistency_weight,
                "consistency_weight_requested": self.consistency_weight_requested,
                "ridge_readout_sha256": getattr(mean_readout, "sha256", ""),
                "ridge_readout_source": str(getattr(mean_readout, "source", "") or ""),
                "use_independent_mean_mlp": self.use_independent_mean_mlp,
                "use_ridge_readout": self.use_ridge_readout,
            }
        )

    def trainable_parameters(self):
        yield from super().trainable_parameters()
        if self.mean_readout is not None and self.mean_readout.linear.weight.requires_grad:
            yield from self.mean_readout.parameters()

    def _predict_mean_from_residual(
        self,
        gated_mu_r: torch.Tensor,
        global_cond: torch.Tensor,
        mean_anchor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.mean_readout is None:
            delta_readout = mean_anchor.new_zeros(mean_anchor.shape)
        else:
            delta_readout = self.mean_readout(gated_mu_r.reshape(gated_mu_r.shape[0], -1))
        pred_mean = mean_anchor + delta_readout
        delta_mlp = mean_anchor.new_zeros(mean_anchor.shape)
        if self.mean_residual_mlp is not None:
            delta_mlp = self.mean_residual_mlp(global_cond)
            pred_mean = pred_mean + delta_mlp
        delta_total = pred_mean - mean_anchor
        return pred_mean, delta_total, delta_readout

    def _consistency_loss(
        self,
        pred_delta_mean: torch.Tensor,
        gated_mu_r: torch.Tensor,
        *,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if self.consistency_weight <= 0.0 or self.mean_readout is None:
            return None
        return delta_consistency_mse(
            pred_delta_mean,
            gated_mu_r.reshape(gated_mu_r.shape[0], -1),
            self.mean_readout,
            stop_grad_pca2d=self.stop_grad_pca2d_for_consistency,
            sample_weight=sample_weight,
        )

    def _mean_aux_loss(
        self,
        pred_mean: torch.Tensor,
        y_mean: Optional[torch.Tensor],
        *,
        step_index: Optional[int] = None,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        weight = float(self.mean_aux_weight if self.use_independent_mean_mlp else self.mean_product_weight)
        if y_mean is None or weight <= 0.0:
            return None
        target = self._resolve_target(y_mean, step_index=step_index)
        per = (pred_mean - target).pow(2).mean(dim=-1)
        if sample_weight is not None:
            w = sample_weight.reshape(-1).to(dtype=per.dtype)
            return (per * w).sum() / w.sum().clamp(min=1e-12)
        return per.mean()

    def predict_distribution(
        self,
        x: torch.Tensor,
        *,
        horizon: Optional[torch.Tensor] = None,
        edge_phylo: Optional[torch.Tensor] = None,
        substitution_feats: Optional[torch.Tensor] = None,
        activation_target: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        mu_flat, site_feats, global_cond, memory, _, mean_anchor = self._encode(
            x,
            horizon=horizon,
            edge_phylo=edge_phylo,
            substitution_feats=substitution_feats,
        )
        out = self._run_head(site_feats, global_cond, memory)
        # Aux BCE only in training mode (``mode="mse"`` trains through here).
        out, act_loss = self._apply_activation_gate(
            out,
            site_feats,
            activation_target=activation_target if self.training else None,
        )
        if act_loss is not None:
            out["activation_loss"] = act_loss
        out["mu_flat"] = mu_flat
        out["pred_flat"] = point_pred_from_residual(mu_flat, out["mu_r"])
        pred_mean, delta_mean, delta_readout = self._predict_mean_from_residual(
            out["mu_r"], global_cond, mean_anchor
        )
        out["pred_mean"] = pred_mean
        out["mean_anchor"] = mean_anchor
        out["delta_mean"] = delta_mean
        out["delta_mean_readout"] = delta_readout
        return out

    def forward_nll(
        self,
        x: torch.Tensor,
        y_pca_delta: torch.Tensor,
        *,
        horizon: Optional[torch.Tensor] = None,
        edge_phylo: Optional[torch.Tensor] = None,
        mse_aux_weight: float = 0.0,
        sample_weight: Optional[torch.Tensor] = None,
        step_index: Optional[int] = None,
        substitution_feats: Optional[torch.Tensor] = None,
        activation_target: Optional[torch.Tensor] = None,
        y_mean: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
        target_flat = self._resolve_target(y_pca_delta, step_index=step_index)
        mu_flat, site_feats, global_cond, memory, _, mean_anchor = self._encode(
            x,
            horizon=horizon,
            edge_phylo=edge_phylo,
            substitution_feats=substitution_feats,
        )
        out = self._run_head(site_feats, global_cond, memory)
        out, act_loss = self._apply_activation_gate(
            out, site_feats, activation_target=activation_target
        )
        nll = low_rank_site_nll(
            target_flat,
            mu_flat,
            out["mu_r"],
            out["log_sigma"],
            out["U"],
            seq_len=self.seq_len,
            k=self.k,
            sample_weight=sample_weight,
        )
        pred_flat = point_pred_from_residual(mu_flat, out["mu_r"])
        pred_mean, delta_mean, delta_readout = self._predict_mean_from_residual(
            out["mu_r"], global_cond, mean_anchor
        )
        loss = nll
        if mse_aux_weight > 0.0:
            loss = loss + float(mse_aux_weight) * F.mse_loss(pred_flat, target_flat)
        if act_loss is not None and self.activation_aux_weight > 0.0:
            loss = loss + float(self.activation_aux_weight) * act_loss
        mean_loss = self._mean_aux_loss(
            pred_mean, y_mean, step_index=step_index, sample_weight=sample_weight
        )
        if mean_loss is not None:
            loss = loss + float(
                self.mean_aux_weight if self.use_independent_mean_mlp else self.mean_product_weight
            ) * mean_loss
        cons = self._consistency_loss(
            pred_mean - mean_anchor, out["mu_r"], sample_weight=sample_weight
        )
        info: dict[str, Any] = {
            "nll": nll.detach(),
            "full_x0_mse": F.mse_loss(pred_flat, target_flat).detach(),
            "mu_mse": F.mse_loss(mu_flat, target_flat).detach(),
            "pred_flat": pred_flat,
            "pred_mean": pred_mean,
            "mean_anchor": mean_anchor,
            "delta_mean": delta_mean,
            "delta_mean_readout": delta_readout,
        }
        if act_loss is not None:
            info["activation_loss"] = act_loss.detach()
        if mean_loss is not None:
            info["mean_mse"] = mean_loss.detach()
        loss = attach_consistency_to_loss(loss, cons, self.consistency_weight, info)
        return loss, info

    def forward_flow(
        self,
        x: torch.Tensor,
        y_pca_delta: torch.Tensor,
        *,
        horizon: Optional[torch.Tensor] = None,
        edge_phylo: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        sample_weight: Optional[torch.Tensor] = None,
        step_index: Optional[int] = None,
        substitution_feats: Optional[torch.Tensor] = None,
        activation_target: Optional[torch.Tensor] = None,
        y_mean: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss, info = super().forward_flow(
            x,
            y_pca_delta,
            horizon=horizon,
            edge_phylo=edge_phylo,
            t=t,
            sample_weight=sample_weight,
            step_index=step_index,
            substitution_feats=substitution_feats,
            activation_target=activation_target,
            y_mean=y_mean,
        )
        # Recompute mean via readout for consistency (flow path uses parent mean MLP).
        mu_flat, site_feats, global_cond, memory, _, mean_anchor = self._encode(
            x,
            horizon=horizon,
            edge_phylo=edge_phylo,
            substitution_feats=substitution_feats,
        )
        out = self._run_head(
            site_feats,
            global_cond,
            memory,
            return_velocity=True,
            flow_state=torch.zeros(
                x.shape[0], self.seq_len, self.k, device=x.device, dtype=x.dtype
            ),
            flow_t=torch.zeros(x.shape[0], device=x.device, dtype=x.dtype),
        )
        out, _ = self._apply_activation_gate(out, site_feats, activation_target=None)
        pred_mean, delta_mean, delta_readout = self._predict_mean_from_residual(
            out["mu_r"], global_cond, mean_anchor
        )
        info["pred_mean"] = pred_mean
        info["delta_mean"] = delta_mean
        info["delta_mean_readout"] = delta_readout
        cons = self._consistency_loss(
            pred_mean - mean_anchor, out["mu_r"], sample_weight=sample_weight
        )
        loss = attach_consistency_to_loss(loss, cons, self.consistency_weight, info)
        return loss, info

    def forward(
        self,
        x: torch.Tensor,
        y_pca_delta: torch.Tensor,
        *,
        mode: str = "nll",
        horizon: Optional[torch.Tensor] = None,
        edge_phylo: Optional[torch.Tensor] = None,
        mse_aux_weight: float = 0.0,
        sample_weight: Optional[torch.Tensor] = None,
        step_index: Optional[int] = None,
        substitution_feats: Optional[torch.Tensor] = None,
        activation_target: Optional[torch.Tensor] = None,
        y_mean: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if mode == "mse":
            dist = self.predict_distribution(
                x,
                horizon=horizon,
                edge_phylo=edge_phylo,
                substitution_feats=substitution_feats,
                activation_target=activation_target,
            )
            target_flat = self._resolve_target(y_pca_delta, step_index=step_index)
            pred_flat = dist["pred_flat"]
            per_sample = (pred_flat - target_flat).pow(2).mean(dim=-1)
            if sample_weight is not None:
                w = sample_weight.reshape(-1).to(dtype=per_sample.dtype)
                loss = (per_sample * w).sum() / w.sum().clamp(min=1e-12)
            else:
                loss = per_sample.mean()
            mean_loss = self._mean_aux_loss(
                dist["pred_mean"],
                y_mean,
                step_index=step_index,
                sample_weight=sample_weight,
            )
            if mean_loss is not None:
                loss = loss + float(
                    self.mean_aux_weight
                    if self.use_independent_mean_mlp
                    else self.mean_product_weight
                ) * mean_loss
            act_loss = dist.get("activation_loss")
            if act_loss is not None and self.activation_aux_weight > 0.0:
                loss = loss + float(self.activation_aux_weight) * act_loss
            cons = self._consistency_loss(
                dist["pred_mean"] - dist["mean_anchor"],
                dist["pred_flat"] - dist["mu_flat"],
                sample_weight=sample_weight,
            )
            info = {
                "full_x0_mse": per_sample.mean().detach(),
                "mu_mse": F.mse_loss(dist["mu_flat"], target_flat).detach(),
                "pred_flat": pred_flat,
                "pred_mean": dist["pred_mean"],
                "mean_anchor": dist["mean_anchor"],
                "delta_mean": dist["delta_mean"],
            }
            if act_loss is not None:
                info["activation_loss"] = act_loss.detach()
            if mean_loss is not None:
                info["mean_mse"] = mean_loss.detach()
            loss = attach_consistency_to_loss(loss, cons, self.consistency_weight, info)
            return loss, pred_flat, info
        return super().forward(
            x,
            y_pca_delta,
            mode=mode,
            horizon=horizon,
            edge_phylo=edge_phylo,
            mse_aux_weight=mse_aux_weight,
            sample_weight=sample_weight,
            step_index=step_index,
            substitution_feats=substitution_feats,
            activation_target=activation_target,
            y_mean=y_mean,
        )
