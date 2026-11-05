#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     edge_phylo_features.py
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
"""Per-edge evolutionary (phylo) distances for GPT / predictor inputs.

Distances use ``calculate_true_phylo_distance`` (BLOSUM-based dissimilarity) on MSA
sequences matched by genome id. Stored as ``(N, K-1)`` consecutive edges on the full
series. At input step ``t`` we append ``log1p(d_t)`` where ``d_t`` is the phylo
distance on the **same edge as** ``v_t = x_t − x_{t−1}`` (i.e. ``(t−1) → t`` on the
active timeline; ``d_0 = 0``). When epoch subsample drops interior timepoints,
distances over dropped gaps are **summed** (same convention as
``train_gated_lstm.AugmentedDataset`` hammings).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from recomb2027.data.series import SeriesEncodingsDataset
from recomb2027.data.extra_encoding_features import (
    expected_model_input_dim,
    parse_block_names,
    parse_extra_encoding_stats,
)
from recomb2027.data.seq_pairwise_distance import (
    compute_consecutive_edge_distances,
)

logger = logging.getLogger(__name__)


def _series_k_from_dataset(dataset: SeriesEncodingsDataset) -> int:
    """Infer series length ``K`` for empty or ragged ``sample_gids`` layouts."""
    k = int(dataset.get_block_size())
    if k <= 0 and getattr(dataset, "sample_encodings", None) is not None:
        enc = np.asarray(dataset.sample_encodings)
        if enc.ndim >= 2:
            k = int(enc.shape[1])
    return k


def _empty_edge_distances(dataset: SeriesEncodingsDataset) -> np.ndarray:
    k = _series_k_from_dataset(dataset)
    return np.zeros((0, max(k - 1, 0)), dtype=np.float32)


def expected_predictor_input_dim(
    *,
    output_dim: int,
    gatt_plus: bool,
    append_edge_phylo: bool,
    fused_input_width: Optional[int] = None,
    extra_encoding_stats: Optional[Sequence[str]] = None,
    block_names: Optional[Sequence[str]] = None,
    feature_mode: str = "absolute",
) -> int:
    """Feature width for ``input_proj`` (before optional cross-stream doubling)."""
    primary = int(output_dim)
    fused = int(fused_input_width if fused_input_width is not None else output_dim)
    if fused != primary:
        raise ValueError(
            "extra_encoding_stats with fused_input_width != output_dim is not supported"
        )
    stats = parse_extra_encoding_stats(extra_encoding_stats or [])
    blocks = list(block_names or [])
    fm = "gatt_plus" if gatt_plus else feature_mode
    return expected_model_input_dim(
        primary_dim=primary,
        extra_stats=stats,
        block_names=blocks,
        gatt_plus=gatt_plus,
        append_edge_phylo=append_edge_phylo,
        feature_mode=fm,
    )


def resolve_predictor_input_layout_from_checkpoint(
    hp: dict,
    *,
    fused_input_width: int,
) -> tuple[bool, bool]:
    """Infer ``(gatt_plus, append_edge_phylo)`` from checkpoint ``input_dim``."""
    got_in = int(hp["input_dim"])
    output_dim = int(hp.get("output_dim", fused_input_width))
    extra_stats = parse_extra_encoding_stats(hp.get("extra_encoding_stats", []))
    block_names = list(parse_block_names(hp.get("block_names", [])))
    matches: list[tuple[bool, bool]] = []
    for gatt_plus in (False, True):
        for append_phylo in (False, True):
            exp = expected_predictor_input_dim(
                output_dim=output_dim,
                gatt_plus=gatt_plus,
                append_edge_phylo=append_phylo,
                fused_input_width=fused_input_width,
                extra_encoding_stats=extra_stats,
                block_names=block_names,
            )
            if exp == got_in:
                matches.append((gatt_plus, append_phylo))
    if not matches:
        raise ValueError(
            f"Checkpoint input_dim={got_in} does not match any layout for "
            f"output_dim={output_dim}, fused_width={fused_input_width} "
            "(expected D, D+1, 2D, or 2D+1 on fused width)."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous input layout for input_dim={got_in}: {matches}"
        )
    return matches[0]


def attach_edge_phylo_distances(
    dataset: SeriesEncodingsDataset,
    gid_to_seq: dict,
    *,
    cache_path: Optional[Union[str, Path]] = None,
    use_log1p: bool = True,
    log_prefix: str = "",
) -> np.ndarray:
    """Precompute ``(N, K-1)`` edge distances and enable append on ``dataset``."""
    if len(dataset) == 0:
        edge = _empty_edge_distances(dataset)
        dataset.edge_distances = edge
        dataset.append_edge_phylo_distance = True
        dataset.edge_phylo_use_log1p = bool(use_log1p)
        return dataset.edge_distances

    gids = np.stack(dataset.sample_gids)
    edge = compute_consecutive_edge_distances(
        gids,
        gid_to_seq,
        cache_path=cache_path,
        log_prefix=log_prefix,
        metric="blosum",
    )
    dataset.edge_distances = edge.astype(np.float32)
    dataset.append_edge_phylo_distance = True
    dataset.edge_phylo_use_log1p = bool(use_log1p)
    return dataset.edge_distances


def attach_edge_phylo_from_msa(
    dataset: SeriesEncodingsDataset,
    msa_path: Union[str, Path],
    *,
    cache_dir: Optional[Union[str, Path]] = None,
    split: Optional[str] = None,
    use_log1p: bool = True,
    log_prefix: str = "",
) -> np.ndarray:
    """Parse *msa_path*, compute/cache edge distances, attach to *dataset*."""
    msa_path = Path(msa_path)
    cache_path = None
    if cache_dir is not None:
        split_tag = split or getattr(dataset, "split", "data")
        cache_path = (
            Path(cache_dir)
            / f"{split_tag}_consec_phylo_{msa_path.stem}.npy"
        )

    if cache_path is not None and cache_path.exists():
        if len(dataset) == 0:
            edge = _empty_edge_distances(dataset)
            dataset.edge_distances = edge
            dataset.append_edge_phylo_distance = True
            dataset.edge_phylo_use_log1p = bool(use_log1p)
            return dataset.edge_distances

        gids = np.stack(dataset.sample_gids)
        edge = np.load(cache_path)
        if edge.shape[0] != len(gids):
            raise ValueError(
                f"Cached edge distances rows {edge.shape[0]} != dataset N={len(gids)} "
                f"({cache_path}). Delete cache and recompute."
            )
        print(
            f"  {log_prefix}Loading cached blosum edge distances from {cache_path}",
            flush=True,
        )
        dataset.edge_distances = edge.astype(np.float32)
        dataset.append_edge_phylo_distance = True
        dataset.edge_phylo_use_log1p = bool(use_log1p)
        return dataset.edge_distances

    from recomb2027.latent.predictors.predictor_utils import parse_msa

    gid_to_seq = parse_msa(str(msa_path))

    return attach_edge_phylo_distances(
        dataset,
        gid_to_seq,
        cache_path=cache_path,
        use_log1p=use_log1p,
        log_prefix=log_prefix,
    )


def prefix_timeline_indices(
    last_idx: int,
    prefix_len: int,
) -> list[int]:
    """Series indices for each slot in a ``ContinuousDtDataset`` prefix window."""
    n_obs = last_idx + 1
    if n_obs >= prefix_len:
        return list(range(last_idx - prefix_len + 1, last_idx + 1))
    pad = prefix_len - n_obs
    return [0] * pad + list(range(n_obs))


def prefix_step_edge_distances_from_timeline(
    edge_row: np.ndarray,
    timeline_indices: Sequence[int],
) -> np.ndarray:
    """Raw per-slot phylo distances for an explicit prefix timeline.

    ``timeline_indices`` has length ``prefix_len`` (one series index per slot).
    Slot ``t`` uses the summed consecutive edge distance from ``timeline[t-1]``
    to ``timeline[t]``; slot ``0`` is ``0``.
    """
    edge_row = np.asarray(edge_row, dtype=np.float32).reshape(-1)
    orig = [int(x) for x in timeline_indices]
    out = np.zeros(len(orig), dtype=np.float32)
    for t in range(1, len(orig)):
        lo, hi = orig[t - 1], orig[t]
        if hi <= lo:
            out[t] = 0.0
        else:
            out[t] = float(edge_row[lo:hi].sum())
    return out


def prefix_step_edge_distances(
    edge_row: np.ndarray,
    last_idx: int,
    prefix_len: int,
) -> np.ndarray:
    """Raw per-slot phylo distances aligned with ``ContinuousDtDataset`` prefix slots.

    Slot ``t`` uses distance on ``(t−1) → t`` (same edge as ``v_t``); ``d_0 = 0``.
    ``edge_row[k]`` is the consecutive distance ``k → k+1`` on the full series.
    """
    edge_row = np.asarray(edge_row, dtype=np.float32).reshape(-1)
    orig = prefix_timeline_indices(last_idx, prefix_len)
    return prefix_step_edge_distances_from_timeline(edge_row, orig)


def batch_prefix_edge_distances(
    edge_distances: np.ndarray,
    last_idx: int | np.ndarray,
    prefix_len: int,
) -> np.ndarray:
    """Stack ``prefix_step_edge_distances`` for many series → ``(N, L)``."""
    edge = np.asarray(edge_distances, dtype=np.float32)
    if np.ndim(last_idx) == 0:
        last_idx = np.full(edge.shape[0], int(last_idx), dtype=np.int64)
    else:
        last_idx = np.asarray(last_idx, dtype=np.int64).reshape(-1)
    if last_idx.shape[0] != edge.shape[0]:
        raise ValueError(
            f"last_idx length {last_idx.shape[0]} != N={edge.shape[0]}"
        )
    rows = [
        prefix_step_edge_distances(edge[i], int(last_idx[i]), prefix_len)
        for i in range(edge.shape[0])
    ]
    return np.stack(rows, axis=0).astype(np.float32)
