#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     nar_fused_decoder.py
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
"""Non-autoregressive decoder fusing per-site PCA2D latents with the mean-pooled ESM vector.

Why these two inputs are complementary
--------------------------------------
* PCA2D sites ``z_i = (e_i - mu) W^T`` (k=4 per site) keep *position* but only ~18% of the feature variance.
* The mean ``m = mean_i e_i`` keeps all 1280 feature dims but *no position*.
* They overlap exactly in one subspace: ``mean_i z_i = (m - mu) W^T`` (verified numerically), so the 4-d
  W-subspace of the mean is redundant with the sites (and a consistency check for noisy predictions),
  while the other 1276 dims of the mean (~73% of its between-sequence variance) are new information.

Fusion (loses nothing, no hand-made projection of either input)
---------------------------------------------------------------
Both inputs are normalised with train statistics so that noise/dropout are defined in units of real
between-sequence variation::

    sites = (z - site_mean[L,k]) / latent_scale[k]          # as in PCA2DNARDecoder
    mean  = (m - mean_mu[D])    / mean_scale[D]             # per-dim: std varies ~30x across dims

The full standardised mean is injected through two paths:
  1. ``n_global`` global tokens (MLP of the mean) prepended to the site tokens, so every site can attend to
     the whole-sequence context and the global tokens can read the sites (cross-checking the two inputs);
  2. an additive per-site bias ``Linear(D -> d_model)(mean)`` (== concatenating the mean to each site feature).

Modality dropout (learned null embeddings) makes the decoder usable with either input alone and robust to
the disagreement between independently predicted heads. Inputs are passed *packed* as one flat vector
``[sites (L*k) | mean (D)]`` so the existing decode pipeline (``model.generate(z)``) works unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

INFERENCE_MODES = ("both", "pca2d_only", "mean_only")


class PCA2DMeanFusedDecoder(nn.Module):
    def __init__(
        self,
        seq_len: int,
        k: int,
        mean_dim: int = 1280,
        vocab_size: int = 22,
        d_model: int = 256,
        n_layers: int = 3,
        n_heads: int = 4,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        bos_id: int = 21,
        n_global: int = 4,
        latent_scale=1.0,
        site_mean: "torch.Tensor | None" = None,
        mean_mu: "torch.Tensor | None" = None,
        mean_scale=1.0,
    ):
        super().__init__()
        self.seq_len, self.k, self.mean_dim = int(seq_len), int(k), int(mean_dim)
        self.vocab_size, self.d_model = int(vocab_size), int(d_model)
        self.n_layers, self.n_heads, self.ff_dim = int(n_layers), int(n_heads), int(ff_dim)
        self.dropout, self.bos_id, self.n_global = float(dropout), int(bos_id), int(n_global)
        self.site_cond_dim = self.k
        self.latent_dim = self.seq_len * self.k + self.mean_dim  # packed input width
        self.inference_mode = "both"

        def _vec(x, n):
            t = torch.as_tensor(x, dtype=torch.float32).reshape(-1)
            if t.numel() == 1:
                t = t.expand(n).clone()
            if t.numel() != n:
                raise ValueError(f"expected scalar or length {n}, got {t.numel()}")
            return t

        self.register_buffer("latent_scale", _vec(latent_scale, k))
        self.register_buffer("site_mean", torch.zeros(seq_len, k) if site_mean is None else torch.as_tensor(site_mean, dtype=torch.float32).clone())
        self.register_buffer("mean_mu", torch.zeros(mean_dim) if mean_mu is None else _vec(mean_mu, mean_dim))
        self.register_buffer("mean_scale", _vec(mean_scale, mean_dim))

        d = d_model
        self.input_proj = nn.Linear(k, d)
        self.pos_emb = nn.Embedding(seq_len, d)  # torch default N(0,1), as in DiMA
        self.mean_tokens = nn.Sequential(
            nn.Linear(mean_dim, 2 * d), nn.GELU(), nn.Linear(2 * d, n_global * d)
        )
        self.mean_bias = nn.Linear(mean_dim, d)
        self.global_pos = nn.Parameter(torch.randn(n_global, d))
        self.null_mean = nn.Parameter(torch.zeros(mean_dim))  # learned "mean missing" (normalised space)
        self.null_sites = nn.Parameter(torch.zeros(seq_len, k))  # learned "sites missing"
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=ff_dim, dropout=dropout,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)
        self.fc = nn.Linear(d, vocab_size)

    # ---- input handling -------------------------------------------------------------------------
    def set_inference_mode(self, mode: str) -> None:
        if mode not in INFERENCE_MODES:
            raise ValueError(f"mode must be one of {INFERENCE_MODES}")
        self.inference_mode = mode

    def split(self, z_packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, L*k + D) -> sites (B, L, k), mean (B, D)."""
        if z_packed.dim() != 2 or z_packed.shape[1] != self.latent_dim:
            raise ValueError(f"expected packed (B, {self.latent_dim}) = [L*k={self.seq_len * self.k} | D={self.mean_dim}], got {tuple(z_packed.shape)}")
        n = self.seq_len * self.k
        return z_packed[:, :n].reshape(-1, self.seq_len, self.k), z_packed[:, n:]

    @staticmethod
    def pack(sites: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
        return torch.cat([sites.reshape(sites.shape[0], -1), mean], dim=1)

    def forward(
        self,
        z_packed: torch.Tensor,
        tokens: "torch.Tensor | None" = None,  # ignored (API parity)
        drop_mean: "torch.Tensor | None" = None,  # (B,) bool
        drop_sites: "torch.Tensor | None" = None,  # (B,) bool
    ) -> torch.Tensor:
        z, m = self.split(z_packed.to(self.input_proj.weight.dtype))
        b = z.shape[0]
        if drop_mean is None and drop_sites is None:
            if self.inference_mode == "pca2d_only":
                drop_mean = torch.ones(b, dtype=torch.bool, device=z.device)
            elif self.inference_mode == "mean_only":
                drop_sites = torch.ones(b, dtype=torch.bool, device=z.device)
        sites = (z - self.site_mean) / self.latent_scale
        mean = (m - self.mean_mu) / self.mean_scale
        if drop_sites is not None:
            sites = torch.where(drop_sites.view(b, 1, 1), self.null_sites.expand(b, -1, -1), sites)
        if drop_mean is not None:
            mean = torch.where(drop_mean.view(b, 1), self.null_mean.expand(b, -1), mean)

        pos = torch.arange(self.seq_len, device=z.device)
        h_sites = self.input_proj(sites) + self.pos_emb(pos).unsqueeze(0) + self.mean_bias(mean).unsqueeze(1)
        g = self.mean_tokens(mean).view(b, self.n_global, self.d_model) + self.global_pos
        h = self.transformer(torch.cat([g, h_sites], dim=1))
        logits = self.fc(self.ln(h[:, self.n_global:]))
        logits[:, :, self.bos_id] = -1e9
        return logits

    @torch.no_grad()
    def generate(self, z_packed: torch.Tensor, greedy: bool = True, temperature: float = 1.0) -> torch.Tensor:
        self.eval()
        logits = self.forward(z_packed)
        if greedy or temperature <= 0.0:
            return logits.argmax(dim=-1)
        probs = F.softmax(logits / temperature, dim=-1)
        b, l, v = probs.shape
        return torch.multinomial(probs.view(-1, v), 1).view(b, l)
