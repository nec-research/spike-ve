#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     decoder_dataset.py
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
"""Holdout train/val data with ESM embeddings for decoder training."""

from __future__ import annotations

import os
from pathlib import Path
import random

import numpy as np
import torch

from recomb2027.generation.latent_decoder.training.esm_embed import ESMEmbedder
from recomb2027.generation.latent_decoder.training.splits import (
    parse_fasta_with_metadata,
    select_holdout_split,
)
from recomb2027.latent.data_series.pca2d_features import (
    load_pca2d_basis,
    project_gid_to_pca2d_flat,
)
from recomb2027.latent.data_series.esm_2d_io import build_gid_file_index


class DecoderSplitDataset(torch.utils.data.Dataset):
    """``(latent, token_ids)`` pairs for one split."""

    def __init__(self, embeddings: torch.Tensor, tokens: torch.Tensor):
        if embeddings.shape[0] != tokens.shape[0]:
            raise ValueError(
                f"embeddings rows ({embeddings.shape[0]}) != "
                f"tokens rows ({tokens.shape[0]})."
            )
        self.embeddings = embeddings.float()
        self.tokens = tokens.long()

    def __len__(self) -> int:
        return int(self.embeddings.shape[0])

    def __getitem__(self, idx: int):
        return self.embeddings[idx], self.tokens[idx]

    @staticmethod
    def collate(batch):
        z = torch.stack([b[0] for b in batch])
        tokens = torch.stack([b[1] for b in batch])
        return z, tokens


class DecoderHoldoutData:
    """Holdout split with cached ESM embeddings or PCA2D features for train and validation."""

    def __init__(
        self,
        in_fasta: str,
        device=None,
        *,
        limit: int | None = None,
        test_fraction: float = 0.15,
        seed: int = 42,
        esm_batch_size: int = 8,
        max_len: int | None = None,
        esm_dtype: str = "fp32",
        embedding_cache_dir: str | None = None,
        embedding_cache_basename: str | None = None,
        esm_repr_layer: int | None = None,
        train_global_idxs: list[int] | None = None,
        test_global_idxs: list[int] | None = None,
        # PCA2D mode: path to fitted PCA basis (.pt) and directory with 2D .pt encodings
        pca2d_basis: str | None = None,
        enc_2d_dir: str | None = None,
        # Optional date cutoff (YYYY-MM-DD) — use only sequences with collection
        # date on or before this date when selecting sequences for holdout mode.
        before_date: str | None = None,
    ):
        all_seqs, all_metas = parse_fasta_with_metadata(in_fasta)
        # If before_date given and no explicit global idxs are provided, filter
        # FASTA rows to only those with collection date <= cutoff.
        if before_date is not None and train_global_idxs is None and test_global_idxs is None:
            from datetime import date

            try:
                cutoff = date.fromisoformat(before_date)
            except Exception:
                raise ValueError(f"--before-date must be YYYY-MM-DD, got {before_date!r}")

            kept_seqs: list[str] = []
            kept_metas: list[dict] = []
            for seq, meta in zip(all_seqs, all_metas):
                # metadata keys are lowercased by parser
                dstr = meta.get("collection_date") or meta.get("date")
                if dstr is None:
                    continue
                try:
                    d = date.fromisoformat(str(dstr))
                except Exception:
                    continue
                if d <= cutoff:
                    kept_seqs.append(seq)
                    kept_metas.append(meta)
            removed = len(all_seqs) - len(kept_seqs)
            if removed:
                print(f"Filtered out {removed} FASTA rows after --before-date={before_date}", flush=True)
            all_seqs, all_metas = kept_seqs, kept_metas
        if max_len is not None and max_len > 0:
            all_seqs = [s[:max_len] for s in all_seqs]
        n_total = len(all_seqs)
        if n_total < 2:
            raise ValueError(f"Need at least 2 sequences, got {n_total}.")

        self.in_fasta = in_fasta
        self.seed = seed

        if train_global_idxs is not None and test_global_idxs is not None:
            self.train_global_idxs = list(train_global_idxs)
            self.test_global_idxs = list(test_global_idxs)
        elif limit is not None and limit > 0:
            if limit >= n_total:
                print(
                    f"Requested limit={limit} >= total available sequences ({n_total}). "
                    f"Partitioning full dataset with test_fraction={test_fraction}.",
                    flush=True,
                )
                rng = random.Random(seed)
                n_test = max(1, int(round(n_total * test_fraction)))
                self.test_global_idxs = sorted(rng.sample(range(n_total), n_test))
                test_set = set(self.test_global_idxs)
                self.train_global_idxs = sorted([i for i in range(n_total) if i not in test_set])
            else:
                self.train_global_idxs, self.test_global_idxs, _ = select_holdout_split(
                    n_total, limit, test_fraction, seed,
                )
        else:
            # Full dataset split when limit is None or <= 0
            rng = random.Random(seed)
            n_test = max(1, int(round(n_total * test_fraction)))
            self.test_global_idxs = sorted(rng.sample(range(n_total), n_test))
            test_set = set(self.test_global_idxs)
            self.train_global_idxs = sorted([i for i in range(n_total) if i not in test_set])
            print(
                f"No --limit or --splits-pkl provided: using all {n_total} sequences with "
                f"test_fraction={test_fraction} (train={len(self.train_global_idxs)}, test={len(self.test_global_idxs)}).",
                flush=True,
            )

        needed = sorted(set(self.train_global_idxs) | set(self.test_global_idxs))
        needed_seqs = [all_seqs[i] for i in needed]
        self.global_to_local = {g: loc for loc, g in enumerate(needed)}

        fasta_base = os.path.splitext(os.path.basename(in_fasta))[0]
        if embedding_cache_dir == "":
            emb_cache = None
        elif embedding_cache_dir is not None:
            emb_cache = embedding_cache_dir
        else:
            emb_cache = os.path.dirname(os.path.abspath(in_fasta))

        emb_name = (
            embedding_cache_basename
            or f"{fasta_base}.decoder_holdout.N{len(needed)}"
        )

        # PCA2D mode: project precomputed 2D encodings through fitted PCA basis
        if pca2d_basis is not None:
            basis = load_pca2d_basis(pca2d_basis)
            enc_path = Path(enc_2d_dir) if enc_2d_dir else Path(".")
            layer = basis.get("layer")
            # Auto-resolve layer subdirectory if enc_path does not contain .pt files directly
            if layer is not None and (enc_path / str(layer)).is_dir():
                direct_pt = next(enc_path.glob("*.pt"), None)
                if direct_pt is None:
                    print(f"Auto-resolving enc-2d-dir to layer subdirectory: {enc_path / str(layer)}", flush=True)
                    enc_path = enc_path / str(layer)
            gid_index = build_gid_file_index(enc_path, layer=layer)
            if not gid_index:
                raise FileNotFoundError(f"No .pt 2D encodings found under {enc_path}")

            # Build FASTA-index -> genome_id mapping from metadata
            gids: list[str] = []
            for m in all_metas:
                gid = m.get("genome_id") or m.get("id") or None
                if gid is None:
                    raise KeyError("FASTA metadata missing 'genome_id' required for PCA2D mode")
                gids.append(str(gid))

            # Cache projected vectors for needed unique gids
            unique_needed = sorted({gids[i] for i in needed})
            seq_len = int(basis["seq_len"])
            k = int(basis["k"])
            zero_mat = np.zeros((seq_len, k), dtype=np.float32)

            cache_file = None
            if emb_cache:
                os.makedirs(emb_cache, exist_ok=True)
                basis_stem = Path(pca2d_basis).stem
                cache_file = Path(emb_cache) / f"{fasta_base}.pca2d_{basis_stem}.N{len(needed)}.pt"

            if cache_file and cache_file.exists():
                print(f"Loading cached PCA2D projections from {cache_file}...", flush=True)
                cached_data = torch.load(cache_file, map_location="cpu")
                self.train_embeddings = cached_data["train_embeddings"]
                self.test_embeddings = cached_data["test_embeddings"]
            else:
                import time
                from concurrent.futures import ThreadPoolExecutor

                print(f"Projecting {len(unique_needed):,} unique 2D encodings through PCA2D basis...", flush=True)
                cache: dict[str, np.ndarray] = {}
                t0 = time.time()

                def _project_one(gid: str) -> tuple[str, np.ndarray]:
                    if gid in gid_index:
                        flat = project_gid_to_pca2d_flat(gid, gid_index, basis)
                        try:
                            return gid, flat.reshape(seq_len, k).astype(np.float32)
                        except Exception:
                            return gid, np.asarray(flat, dtype=np.float32)
                    return gid, zero_mat

                max_workers = min(32, max(4, os.cpu_count() or 8))
                completed = 0
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    for gid, mat in executor.map(_project_one, unique_needed):
                        cache[gid] = mat
                        completed += 1
                        if completed % 10000 == 0 or completed == len(unique_needed):
                            elapsed = time.time() - t0
                            rate = completed / max(0.1, elapsed)
                            print(
                                f"  projected {completed:,}/{len(unique_needed):,} "
                                f"({rate:.1f} seq/s, elapsed {elapsed:.1f}s)",
                                flush=True,
                            )

                train_gids = [gids[g] for g in self.train_global_idxs]
                test_gids = [gids[g] for g in self.test_global_idxs]
                train_mats = [cache.get(gid, zero_mat) for gid in train_gids]
                test_mats = [cache.get(gid, zero_mat) for gid in test_gids]
                self.train_embeddings = torch.from_numpy(np.stack(train_mats, axis=0))
                self.test_embeddings = torch.from_numpy(np.stack(test_mats, axis=0))

                if cache_file:
                    try:
                        torch.save({
                            "train_embeddings": self.train_embeddings,
                            "test_embeddings": self.test_embeddings,
                            "seq_len": seq_len,
                            "k": k,
                        }, cache_file)
                        print(f"Saved PCA2D projections cache -> {cache_file}", flush=True)
                    except Exception as e:
                        print(f"Warning: could not save cache to {cache_file}: {e}", flush=True)

            train_local = [self.global_to_local[g] for g in self.train_global_idxs]
            test_local = [self.global_to_local[g] for g in self.test_global_idxs]
            self.train_sequences = [needed_seqs[i] for i in train_local]
            self.test_sequences = [needed_seqs[i] for i in test_local]
            self.esm_repr_layer = int(basis.get("layer", esm_repr_layer or 34))
            print(
                f"Decoder holdout (pca2d): train={len(self.train_sequences)}, "
                f"test={len(self.test_sequences)}  dim={self.train_embeddings.shape[1]}"
            )
        else:
            embedder = ESMEmbedder(
                device=device,
                esm_batch_size=esm_batch_size,
                esm_dtype=esm_dtype,
                esm_repr_layer=esm_repr_layer,
            )
            self.esm_repr_layer = embedder.repr_layer
            all_emb = embedder.embed_sequences(
                needed_seqs,
                embedding_cache_dir=emb_cache,
                embedding_cache_basename=emb_name,
            )

            train_local = [self.global_to_local[g] for g in self.train_global_idxs]
            test_local = [self.global_to_local[g] for g in self.test_global_idxs]
            self.train_sequences = [needed_seqs[i] for i in train_local]
            self.test_sequences = [needed_seqs[i] for i in test_local]
            self.train_embeddings = all_emb[train_local]
            self.test_embeddings = all_emb[test_local]
            print(
                f"Decoder holdout: train={len(self.train_sequences)}, "
                f"test={len(self.test_sequences)}"
            )
