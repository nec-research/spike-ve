#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     nar_decoder.py
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
"""Non-autoregressive DiMA-style decoder for per-site PCA2D latents.

Mirrors ``DiMA/src/encoders/transformer_decoder.py``::

    x = x + position_embeddings(pos)            # latent already has one vector per position
    logits = fc(transformer_encoder(x))         # bidirectional, no causal mask, no cross-attention

Adaptation: the original latent already has the transformer hidden size, so here a linear map
``k -> d_model`` is added. The model consumes *raw* PCA2D projections (same interface as
``PCA2DEncoderDecoder``) and normalises them internally::

    z_norm = (z - site_mean[L, k]) / latent_scale[k]

``site_mean`` is the per-site training mean (site identity is carried by the position embedding)
and ``latent_scale`` the per-component std of the *residual* ``z - site_mean`` (the informative
between-sequence variation). One unit of noise therefore means "one std of real sequence
variation", the regime in which DiMA's noise levels ``min_T``/``max_T`` make sense. (A global std
is ~6x larger because it is dominated by the site offsets, which silently inflated the noise.)
Aligned sequences are 1:1 with sites, so all positions are decoded in one forward pass.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PCA2DNARDecoder(nn.Module):
    def __init__(
        self,
        seq_len: int,
        k: int,
        vocab_size: int = 22,
        d_model: int = 256,
        n_layers: int = 3,  # DiMA transformer.yaml: num_hidden_layers: 3
        n_heads: int = 4,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        bos_id: int = 21,
        latent_scale: "float | torch.Tensor | list[float]" = 1.0,
        site_mean: "torch.Tensor | None" = None,
        pos_init_std: "float | None" = None,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.k = int(k)
        self.latent_dim = int(k)
        self.site_cond_dim = int(k)
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.ff_dim = int(ff_dim)
        self.dropout = float(dropout)
        self.bos_id = int(bos_id)
        scale = torch.as_tensor(latent_scale, dtype=torch.float32).reshape(-1)
        if scale.numel() == 1:
            scale = scale.expand(k).clone()
        if scale.numel() != k:
            raise ValueError(f"latent_scale must be scalar or length {k}, got {scale.numel()}")
        mean = torch.zeros(seq_len, k) if site_mean is None else torch.as_tensor(site_mean, dtype=torch.float32)
        if tuple(mean.shape) != (seq_len, k):
            raise ValueError(f"site_mean must be ({seq_len}, {k}), got {tuple(mean.shape)}")
        self.register_buffer("latent_scale", scale)  # (k,)
        self.register_buffer("site_mean", mean.clone())  # (L, k)

        self.input_proj = nn.Linear(k, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)
        # DiMA uses torch's default nn.Embedding init, N(0, 1): strong position identity from step 0.
        if pos_init_std is not None:
            nn.init.normal_(self.pos_emb.weight, std=float(pos_init_std))

    def _ensure_sites_3d(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 2:
            if z.shape[1] != self.seq_len * self.k:
                raise ValueError(f"Expected {self.seq_len * self.k} features, got {z.shape[1]}")
            return z.reshape(z.shape[0], self.seq_len, self.k)
        if z.dim() == 3 and tuple(z.shape[1:]) == (self.seq_len, self.k):
            return z
        raise ValueError(f"Expected (B, {self.seq_len}, {self.k}) or (B, L*k), got {tuple(z.shape)}")

    def forward(self, z_sites: torch.Tensor, tokens: torch.Tensor | None = None) -> torch.Tensor:
        """``tokens`` is accepted (and ignored) for API parity with the autoregressive model."""
        z = self._ensure_sites_3d(z_sites).to(self.input_proj.weight.dtype)
        z = (z - self.site_mean) / self.latent_scale
        pos = torch.arange(self.seq_len, device=z.device)
        h = self.input_proj(z) + self.pos_emb(pos).unsqueeze(0)
        logits = self.fc(self.ln(self.transformer(h)))
        logits[:, :, self.bos_id] = -1e9  # BOS is never a valid output
        return logits

    @torch.no_grad()
    def generate(self, z_sites: torch.Tensor, greedy: bool = True, temperature: float = 1.0) -> torch.Tensor:
        self.eval()
        logits = self.forward(z_sites)
        if greedy or temperature <= 0.0:
            return logits.argmax(dim=-1)
        probs = F.softmax(logits / temperature, dim=-1)
        b, l, v = probs.shape
        return torch.multinomial(probs.view(-1, v), 1).view(b, l)


def add_latent_noise(
    z: torch.Tensor,
    latent_scale,
    noise_min: float,
    noise_max: float,
) -> torch.Tensor:
    """DiMA-style decoder noise augmentation, in RAW latent space.

    A per-sample noise level ``sigma ~ U(noise_min, noise_max)`` (DiMA ``min_T``/``max_T``) is applied in
    units of ``latent_scale`` (scalar or per-component ``(k,)``): ``z + sigma * latent_scale * eps``.
    With the model's per-component residual-std scale, ``sigma = 1`` equals one std of real
    between-sequence variation.
    """
    if noise_max <= 0.0:
        return z
    if noise_min > noise_max:
        raise ValueError(f"noise_min ({noise_min}) > noise_max ({noise_max})")
    scale = torch.as_tensor(latent_scale, dtype=z.dtype, device=z.device)
    sigma = torch.empty(z.shape[0], *([1] * (z.dim() - 1)), device=z.device, dtype=z.dtype).uniform_(
        noise_min, noise_max
    )
    return z + sigma * scale * torch.randn_like(z)
