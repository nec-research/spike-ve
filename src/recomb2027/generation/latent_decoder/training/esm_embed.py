#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     esm_embed.py
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
"""ESM-1 mean-pooled embeddings with optional on-disk cache."""

from __future__ import annotations

import hashlib
import os

import torch

ESM_MODEL_TAG = "esm1_t34_670M_UR100"
ESM_T34_NUM_LAYERS = 34


def resolve_esm_repr_layer(
    esm_repr_layer: int | None,
    num_layers: int = ESM_T34_NUM_LAYERS,
) -> int:
    """Resolve ESM representation layer index (1-indexed through ``num_layers``)."""
    layer = num_layers if esm_repr_layer is None else int(esm_repr_layer)
    if not 1 <= layer <= num_layers:
        raise ValueError(
            f"esm_repr_layer must be in [1, {num_layers}], got {layer}."
        )
    return layer


def sequences_digest(seqs: list[str]) -> str:
    """SHA-256 digest (truncated) of an ordered sequence list."""
    digest = hashlib.sha256()
    for s in seqs:
        digest.update(s.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()[:16]


def _esm_dtype_tag(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return "fp32"
    if dtype == torch.float16:
        return "fp16"
    if dtype == torch.bfloat16:
        return "bf16"
    return str(dtype)


def _parse_esm_dtype(esm_dtype: str) -> torch.dtype:
    if esm_dtype == "fp32":
        return torch.float32
    if esm_dtype == "fp16":
        return torch.float16
    if esm_dtype == "bf16":
        return torch.bfloat16
    raise ValueError(
        f"Unknown esm_dtype: {esm_dtype!r}. Expected one of: fp32, fp16, bf16."
    )


def embedding_cache_path(
    cache_dir: str,
    cache_basename: str,
    n: int,
    seq_hash: str,
    esm_dtype_tag: str,
    repr_layer: int,
    model_tag: str = ESM_MODEL_TAG,
) -> str:
    return os.path.join(
        cache_dir,
        f"{cache_basename}.esm.{model_tag}.L{repr_layer}.N{n}."
        f"{seq_hash}.{esm_dtype_tag}.pt",
    )


class ESMEmbedder:
    """Mean-pool ESM-1 (t34) embeddings with batched forwards and disk cache."""

    def __init__(
        self,
        device: torch.device | str | None = None,
        *,
        esm_batch_size: int = 8,
        esm_dtype: str = "fp32",
        esm_repr_layer: int | None = None,
    ):
        self.device = (
            torch.device(device)
            if device is not None
            else (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("mps")
                if torch.backends.mps.is_available()
                else torch.device("cpu")
            )
        )
        self.esm_batch_size = max(1, int(esm_batch_size))
        self.esm_dtype = _parse_esm_dtype(esm_dtype)
        self._load_model(esm_repr_layer)

    def _load_model(self, esm_repr_layer: int | None) -> None:
        try:
            import esm
        except ImportError as e:
            raise ImportError(
                "fair-esm is required for decoder training. "
                "Install with `pip install fair-esm`."
            ) from e

        model, alphabet = esm.pretrained.esm1_t34_670M_UR100()
        model = model.to(self.device).eval()
        if self.esm_dtype != torch.float32:
            if self.device.type == "cpu" and self.esm_dtype == torch.float16:
                raise ValueError(
                    "esm_dtype='fp16' is not supported on CPU; use 'bf16' or 'fp32'."
                )
            model = model.to(self.esm_dtype)
        self.esm_model = model
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter()
        self.repr_layer = resolve_esm_repr_layer(
            esm_repr_layer, self.esm_model.num_layers,
        )

    @torch.no_grad()
    def _embed_chunk(self, seqs: list[str]) -> torch.Tensor:
        batch = [(f"s{i}", s) for i, s in enumerate(seqs)]
        _, _, tokens = self.batch_converter(batch)
        tokens = tokens.to(self.device)
        out = self.esm_model(tokens, repr_layers=[self.repr_layer])
        rep = out["representations"][self.repr_layer]

        mask = (
            (tokens != self.alphabet.padding_idx)
            & (tokens != self.alphabet.cls_idx)
            & (tokens != self.alphabet.eos_idx)
        )
        mask_f = mask.unsqueeze(-1).to(rep.dtype)
        summed = (rep * mask_f).sum(dim=1)
        counts = mask.sum(dim=1, keepdim=True).clamp(min=1).to(rep.dtype)
        return (summed / counts).float().cpu()

    @torch.no_grad()
    def embed_batch(self, seqs: list[str], chunk_size: int | None = None) -> torch.Tensor:
        if not seqs:
            return torch.empty(0)
        chunk_size = self.esm_batch_size if chunk_size is None else chunk_size
        chunks = [
            self._embed_chunk(seqs[i : i + chunk_size])
            for i in range(0, len(seqs), chunk_size)
        ]
        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def embed_sequences(
        self,
        sequences: list[str],
        *,
        embedding_cache_dir: str | None = None,
        embedding_cache_basename: str,
    ) -> torch.Tensor:
        """Embed all sequences, loading from disk cache when available."""
        n = len(sequences)
        if n == 0:
            return torch.empty(0)

        seq_hash = sequences_digest(sequences)
        dtype_tag = _esm_dtype_tag(self.esm_dtype)
        cache_path = None
        if embedding_cache_dir:
            try:
                os.makedirs(embedding_cache_dir, exist_ok=True)
                cache_path = embedding_cache_path(
                    embedding_cache_dir,
                    embedding_cache_basename,
                    n,
                    seq_hash,
                    dtype_tag,
                    self.repr_layer,
                )
            except OSError:
                cache_path = None

        if cache_path and os.path.exists(cache_path):
            try:
                emb = torch.load(cache_path, map_location="cpu", weights_only=False)
                if isinstance(emb, torch.Tensor) and emb.shape[0] == n:
                    print(
                        f"Loaded cached ESM embeddings: {cache_path} "
                        f"(N={n}, D={emb.shape[1]})"
                    )
                    return emb.float().contiguous()
            except (OSError, RuntimeError):
                pass

        print(f"Computing ESM embeddings for N={n} sequences...")
        emb = self.embed_batch(sequences)
        if cache_path is not None:
            try:
                torch.save(emb, cache_path)
                print(f"Cached ESM embeddings to {cache_path}")
            except OSError as e:
                print(f"Warning: failed to write embedding cache: {e}")
        return emb
