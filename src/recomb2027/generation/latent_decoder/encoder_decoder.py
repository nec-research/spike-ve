#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     encoder_decoder.py
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
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PCA2DEncoder(nn.Module):
    """Transformer encoder over per-site PCA2D latent features (B, L, k)."""

    def __init__(
        self,
        k: int,
        seq_len: int,
        d_model: int = 256,
        n_layers: int = 3,
        n_heads: int = 4,
        ff_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.k = int(k)
        self.seq_len = int(seq_len)
        self.d_model = int(d_model)
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
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)

        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, k)
        b, l, _ = x.shape
        pos = torch.arange(l, device=x.device)
        h = self.input_proj(x) + self.pos_emb(pos).unsqueeze(0)
        h = self.encoder(h)
        return self.ln(h)  # (B, L, d_model)


class ARDecoderWithCrossAttn(nn.Module):
    """Autoregressive transformer decoder with causal self-attention and cross-attention to encoder memory."""

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        bos_id: int | None = None,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.seq_len = int(seq_len)
        self.d_model = int(d_model)
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.bos_id = int(bos_id if bos_id is not None else vocab_size - 1)

        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.lm_head.weight)

    def forward(self, tgt_tokens: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # tgt_tokens: (B, l), memory: (B, L_mem, d_model)
        b, l = tgt_tokens.shape
        pos = torch.arange(l, device=tgt_tokens.device)
        tgt = self.token_emb(tgt_tokens) + self.pos_emb(pos).unsqueeze(0)
        # causal mask
        causal = nn.Transformer.generate_square_subsequent_mask(l).to(device=tgt_tokens.device, dtype=tgt.dtype)
        out = self.decoder(tgt, memory, tgt_mask=causal)
        out = self.ln(out)
        logits = self.lm_head(out)
        logits[:, :, self.bos_id] = -1e9
        return logits


class PCA2DEncoderDecoder(nn.Module):
    """Full DiMA encoder-decoder model linking PCA2DEncoder and ARDecoderWithCrossAttn."""

    def __init__(
        self,
        seq_len: int,
        k: int,
        vocab_size: int = 22,
        d_model: int = 256,
        enc_layers: int = 3,
        dec_layers: int = 4,
        n_heads: int = 4,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        bos_id: int = 21,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.k = int(k)
        self.latent_dim = int(k)
        self.site_cond_dim = int(k)
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.enc_layers = int(enc_layers)
        self.dec_layers = int(dec_layers)
        self.n_heads = int(n_heads)
        self.ff_dim = int(ff_dim)
        self.dropout = float(dropout)
        self.bos_id = int(bos_id)

        self.encoder = PCA2DEncoder(
            k=k,
            seq_len=seq_len,
            d_model=d_model,
            n_layers=enc_layers,
            n_heads=n_heads,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.decoder = ARDecoderWithCrossAttn(
            vocab_size=vocab_size,
            seq_len=seq_len,
            d_model=d_model,
            n_layers=dec_layers,
            n_heads=n_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            bos_id=bos_id,
        )

    def _ensure_sites_3d(self, z_sites: torch.Tensor) -> torch.Tensor:
        if z_sites.dim() == 2:
            expected = self.seq_len * self.k
            if z_sites.shape[1] == expected:
                return z_sites.view(z_sites.shape[0], self.seq_len, self.k)
            raise ValueError(
                f"Expected 2D input with {expected} features (seq_len={self.seq_len} * k={self.k}), "
                f"got {z_sites.shape[1]}"
            )
        elif z_sites.dim() == 3:
            if z_sites.shape[1] != self.seq_len or z_sites.shape[2] != self.k:
                raise ValueError(
                    f"Expected 3D input of shape (B, {self.seq_len}, {self.k}), got {tuple(z_sites.shape)}"
                )
            return z_sites
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got dim {z_sites.dim()}")

    def forward(self, z_sites: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Forward pass with teacher-forcing token shifting.

        Args:
            z_sites: (B, L, k) or (B, L*k) per-site latent conditioning
            tokens: (B, L) target token ids (ground-truth sequence)

        Returns:
            logits: (B, L, vocab_size) predictions for each target position tokens[:, i]
        """
        z_sites = self._ensure_sites_3d(z_sites)
        b, l = tokens.shape
        if l != self.seq_len:
            raise ValueError(f"Expected tokens of length {self.seq_len}, got {l}")
        memory = self.encoder(z_sites)
        # Shift target tokens right by 1, prepending BOS: [BOS, t_0, t_1, ..., t_{L-2}]
        # Position i receives tokens up to i-1 and predicts tokens[:, i]
        bos = torch.full((b, 1), fill_value=self.bos_id, dtype=tokens.dtype, device=tokens.device)
        shifted_tokens = torch.cat([bos, tokens[:, :-1]], dim=1)
        logits = self.decoder(shifted_tokens, memory)
        return logits

    @torch.no_grad()
    def generate(
        self,
        z_sites: torch.Tensor,
        greedy: bool = True,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Autoregressively generate full sequences from per-site latent embeddings.

        Args:
            z_sites: (B, L, k) or (B, L*k) per-site latent embeddings
            greedy: if True, take argmax; if False, sample with temperature
            temperature: sampling temperature (when greedy is False)

        Returns:
            generated: (B, seq_len) decoded token ids
        """
        z_sites = self._ensure_sites_3d(z_sites)
        self.eval()
        b = z_sites.shape[0]
        device = z_sites.device
        length = self.seq_len
        memory = self.encoder(z_sites)
        generated = torch.zeros((b, length), dtype=torch.long, device=device)
        bos = torch.full((b, 1), fill_value=self.bos_id, dtype=torch.long, device=device)

        for step in range(length):
            if step == 0:
                prefix = bos
            else:
                prefix = torch.cat([bos, generated[:, :step]], dim=1)
            logits = self.decoder(prefix, memory)
            last = logits[:, -1, :]
            if greedy or temperature <= 0.0:
                next_id = last.argmax(dim=-1)
            else:
                probs = F.softmax(last / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
            generated[:, step] = next_id
        return generated
