#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     seq_pairwise_distance.py
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
"""Pairwise sequence distances for aligned protein sequences.

``fast_pairwise_blosum_distance`` matches the definition in
``recomb2027.generation/.../genetic_algorithm_phylo_dist_model_optimized.py``:
normalised BLOSUM62 dissimilarity with optional global alignment when lengths differ.

For MSA-fixed–length Spike sequences this reduces to a fast matrix lookup sum.

Aligned with ``Bio.Align.substitution_matrices`` (BLOSUM62).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    from Bio import pairwise2
    from Bio.Align import substitution_matrices
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "seq_pairwise_distance requires biopython (Bio.pairwise2, substitution_matrices)."
    ) from e

_MATRIX_CACHE: Tuple[np.ndarray, dict] | None = None


def _build_matrix_with_gap(gap_penalty=-5.0):
    blosum = substitution_matrices.load("BLOSUM62")
    alphabet = list(blosum.alphabet) + ["-"]
    aa_to_idx = {a: i for i, a in enumerate(alphabet)}
    size = len(alphabet)
    matrix = np.zeros((size, size), dtype=np.float64)
    for a in blosum.alphabet:
        for b in blosum.alphabet:
            matrix[aa_to_idx[a], aa_to_idx[b]] = blosum[a, b]
    gap_idx = aa_to_idx["-"]
    for a in blosum.alphabet:
        i = aa_to_idx[a]
        matrix[i, gap_idx] = gap_penalty
        matrix[gap_idx, i] = gap_penalty
    matrix[gap_idx, gap_idx] = 0.0
    return matrix, aa_to_idx


def _align_sequences(seq1: str, seq2: str, gap_penalty: float):
    blosum = substitution_matrices.load("BLOSUM62")
    alignments = pairwise2.align.globalds(
        seq1, seq2, blosum, gap_penalty, gap_penalty
    )
    aligned_seq1, aligned_seq2, _, _, _ = alignments[0]
    return aligned_seq1, aligned_seq2


def calculate_true_phylo_distance(query_seq: str, target_seq: str) -> float:
    """Evolutionary distance between two aligned sequences (BLOSUM dissimilarity).

    Matches ``recomb2027.generation.genetic_algorithm.
    genetic_algorithm_phylo_dist_model_optimized.calculate_true_phylo_distance``.
    """
    return fast_pairwise_blosum_distance(query_seq, target_seq)


def fast_pairwise_blosum_distance(
    seq1: str, seq2: str, gap_penalty=-5.0
) -> float:
    """Return ``1 - cross_score / max(self1, self2)`` in ``[0, ~2]``.

    Same semantics as
    ``genetic_algorithm_phylo_dist_model_optimized.fast_pairwise_blosum_distance``.
    """
    global _MATRIX_CACHE
    if _MATRIX_CACHE is None:
        _MATRIX_CACHE = _build_matrix_with_gap(gap_penalty)
    matrix, aa_to_idx = _MATRIX_CACHE

    if len(seq1) != len(seq2):
        seq1, seq2 = _align_sequences(seq1, seq2, gap_penalty)

    unk = aa_to_idx["-"]

    def _idx(seq: str) -> np.ndarray:
        return np.fromiter((aa_to_idx.get(a, unk) for a in seq), dtype=np.int32)

    i1 = _idx(seq1)
    i2 = _idx(seq2)
    observed = float(matrix[i1, i2].sum())
    self1 = float(matrix[i1, i1].sum())
    self2 = float(matrix[i2, i2].sum())
    max_score = max(self1, self2)
    if max_score <= 0:
        return 0.0
    return 1.0 - observed / max_score


def hamming_mismatch_count(seq1: str, seq2: str) -> int:
    """Count positions where residues differ (ignores columns where both are gap)."""
    return sum(1 for a, b in zip(seq1, seq2)
               if a != b and not (a == "-" and b == "-"))


def pairwise_jump_distance(
    gids_src, gids_tgt, gid_to_seq: dict, metric: str = "blosum",
    gap_penalty=-5.0,
) -> np.ndarray:
    """Per-example distance for the last predictive jump (src → tgt genomes).

    Returns ``(N,)`` array — ``int32`` for ``metric=="hamming"``, ``float32``
    for ``metric=="blosum"``.
    """
    N = len(gids_src)
    if metric == "hamming":
        out = np.zeros(N, dtype=np.int32)
        for i in range(N):
            g1, g2 = str(gids_src[i]), str(gids_tgt[i])
            if g1 in gid_to_seq and g2 in gid_to_seq:
                out[i] = hamming_mismatch_count(
                    gid_to_seq[g1], gid_to_seq[g2])
        return out
    if metric != "blosum":
        raise ValueError(f"unknown metric={metric!r}")
    out = np.zeros(N, dtype=np.float32)
    for i in range(N):
        g1, g2 = str(gids_src[i]), str(gids_tgt[i])
        if g1 in gid_to_seq and g2 in gid_to_seq:
            out[i] = float(fast_pairwise_blosum_distance(
                gid_to_seq[g1], gid_to_seq[g2], gap_penalty=gap_penalty))
    return out


def compute_consecutive_edge_distances(
    gids_array: np.ndarray,
    gid_to_seq: dict,
    cache_path=None,
    log_prefix: str = "",
    metric: str = "blosum",
    gap_penalty: float = -5.0,
) -> np.ndarray:
    """Per-series distances along consecutive phylogeny steps: ``(N, K-1)``.

    Each entry is the pairwise distance between genome at step ``k`` and ``k+1``.
    Cached to ``cache_path`` when provided (filename should encode ``metric``).

    ``metric`` is ``"hamming"`` (int counts) or ``"blosum"`` (float dissimilarity).
    """
    if cache_path is not None and Path(cache_path).exists():
        print(f"  {log_prefix}Loading cached {metric} edge distances from {cache_path}",
              flush=True)
        return np.load(cache_path)

    N, K = gids_array.shape
    t0 = time.time()

    unique: dict = {}
    for row in gids_array:
        for gid in row:
            gid = str(gid)
            if gid not in unique and gid in gid_to_seq:
                unique[gid] = gid_to_seq[gid]

    print(f"  {log_prefix}Unique genomes with sequences: {len(unique)}",
          flush=True)

    if metric == "hamming":
        out = np.zeros((N, K - 1), dtype=np.int32)

        def _pair(a: str, b: str) -> int:
            return hamming_mismatch_count(a, b)
    elif metric == "blosum":
        out = np.zeros((N, K - 1), dtype=np.float32)

        def _pair(a: str, b: str) -> float:
            return float(fast_pairwise_blosum_distance(a, b, gap_penalty))
    else:
        raise ValueError(f"unknown metric={metric!r}")

    log_every = max(1, N // 10)
    for i in range(N):
        if i % log_every == 0 and i > 0:
            elapsed = time.time() - t0
            print(f"  {log_prefix}  {i}/{N}  ({elapsed:.1f}s)", flush=True)
        for k in range(K - 1):
            ga, gb = str(gids_array[i, k]), str(gids_array[i, k + 1])
            sa, sb = unique.get(ga), unique.get(gb)
            if sa is None or sb is None:
                continue
            out[i, k] = _pair(sa, sb)

    print(f"  {log_prefix}Done in {time.time() - t0:.1f}s", flush=True)

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, out)
        print(f"  {log_prefix}Cached → {cache_path}", flush=True)
    return out


# ---------------------------------------------------------------------------
# Evaluation buckets (BLOSUM distances on full aligned Spike ~1273 AA)
# ---------------------------------------------------------------------------
# Heuristic float cutoffs: loosely mirror the old Hamming strata.  Calibrate
# from your cohort if bucket counts become imbalanced.


def iter_eval_bucket_masks(dist: np.ndarray, metric: str):
    """Yield ``(label, boolean_mask)`` for each evaluation stratum.

    Hamming reproduces the original integer buckets.  BLOSUM uses float strata;
    ``"1"`` is ``(0 < d <= 0.0025]`` so exact identity stays in ``"0"``.
    """
    if metric == "hamming":
        d = np.asarray(dist)
        specs = [
            (0, 0, "0"), (1, 1, "1"), (2, 2, "2"), (3, 5, "3-4"),
            (5, 10, "5-9"), (10, 50, "10-49"), (50, 500, "50+"),
        ]
        for lo, hi, lab in specs:
            if lo == hi:
                yield lab, d == lo
            else:
                yield lab, (d >= lo) & (d < hi)
        return

    d = np.asarray(dist, dtype=np.float64)
    yield "0", d == 0.0
    # Cumulative upper edges for (0, .0025], (.0025, .0045], …
    edges = [0.0025, 0.0045, 0.0080, 0.0180, 0.048]
    labels = ["1", "2", "3-4", "5-9", "10-49"]
    lo = 0.0
    for hi, lab in zip(edges, labels):
        if lo == 0.0:
            yield lab, (d > 0.0) & (d <= hi)
        else:
            yield lab, (d > lo) & (d <= hi)
        lo = hi
    yield "50+", d > lo
