#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     esm_pooling.py
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
"""Pool per-residue ESM token embeddings into compact feature vectors.

The default pipeline uses ``average_token`` (mean over amino-acid positions) which
collapses ``[L, D]`` to ``[D]``.  This module adds ``stats_summary`` pooling that
retains per-dimension statistics across sequence positions: mean, std, min, max,
median, plus the **first** AA index (0-based) where each dimension attains its
max/min → shape ``[D, 7]``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

POSITION_STAT_NAMES: tuple[str, ...] = ("argmax_pos", "argmin_pos")
VALUE_STAT_NAMES: tuple[str, ...] = ("mean", "std", "min", "max", "median")
# Stats selectable via ``encode_proteins_from_fasta.py --extra-pooling-stats``
# (mean is the primary pooling, e.g. ``average_token``).
EXTRA_STAT_NAMES: tuple[str, ...] = ("std", "min", "max", "median", "argmax_pos", "argmin_pos")
STAT_NAMES: tuple[str, ...] = VALUE_STAT_NAMES + POSITION_STAT_NAMES
N_STATS = len(STAT_NAMES)

# Alignment gap columns (``.`` is normalised to ``-`` before tokenisation in the encoder).
MSA_GAP_CHARS = frozenset("-")


def prepare_msa_seq_for_esm(seq: str, *, strip_gaps: bool = False) -> str:
    """Normalize an aligned MSA string for ESM tokenisation.

    Uppercase residues, map ``.`` → ``-`` (hmmalign insertion columns), and map
    ``*`` → ``X`` (stop/invalid codon markers in alignments — not in ESM alphabet).
    Lowercase columns (e.g. from soft-masked alignments) must be uppercased or ESM
    groups them into invalid multi-character tokens such as ``'mplf'``.
    """
    out = str(seq).upper().replace(".", "-").replace("*", "X")
    if strip_gaps:
        out = out.replace("-", "")
    return out


def is_msa_gap_char(ch: str) -> bool:
    return ch in MSA_GAP_CHARS


def msa_non_gap_mask(aligned_seq: str, truncate_len: int) -> torch.Tensor:
    """Per-column mask ``[truncate_len]``: True where the MSA character is not a gap."""
    n = min(int(truncate_len), len(aligned_seq))
    mask = [not is_msa_gap_char(aligned_seq[i]) for i in range(n)]
    if n < int(truncate_len):
        mask.extend([False] * (int(truncate_len) - n))
    return torch.tensor(mask, dtype=torch.bool)


def mean_pool_sequence_dim(aa_tokens: torch.Tensor) -> torch.Tensor:
    """``average_sequence`` pooling: mean over hidden dim per row → ``[L]``.

    Gap columns are included (no ``non_gap_mask``); matches post-hoc aggregation.
    """
    if aa_tokens.ndim != 2:
        raise ValueError(f"aa_tokens must be [L, D]; got {tuple(aa_tokens.shape)}")
    if aa_tokens.numel() == 0:
        raise ValueError("Empty token matrix for average_sequence pooling")
    return aa_tokens.mean(dim=1).float()


def mean_pool_aa_tokens(
    aa_tokens: torch.Tensor,
    non_gap_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean-pool ``[L, D]`` rows, optionally keeping only non-gap columns."""
    if aa_tokens.ndim != 2:
        raise ValueError(f"aa_tokens must be [L, D]; got {tuple(aa_tokens.shape)}")
    if non_gap_mask is None:
        return aa_tokens.mean(dim=0).float()
    if non_gap_mask.shape[0] != aa_tokens.shape[0]:
        raise ValueError(
            f"non_gap_mask length {non_gap_mask.shape[0]} != aa rows {aa_tokens.shape[0]}"
        )
    valid = aa_tokens[non_gap_mask]
    if valid.numel() == 0:
        # Degenerate decode (e.g. all-gap early in training); fall back to full-sequence mean.
        return aa_tokens.mean(dim=0).float()
    return valid.mean(dim=0).float()


def first_non_gap_row(aa_tokens: torch.Tensor, non_gap_mask: torch.Tensor | None) -> torch.Tensor:
    """Return the first non-gap row, or row 0 when ``non_gap_mask`` is None."""
    if non_gap_mask is None:
        return aa_tokens[0, :]
    idx = torch.where(non_gap_mask)[0]
    if idx.numel() == 0:
        raise ValueError("No non-gap residues for first_token pooling")
    return aa_tokens[int(idx[0].item()), :]


def first_argext_position_msa(
    aa_tokens: torch.Tensor,
    non_gap_mask: torch.Tensor,
    *,
    which: str,
) -> torch.Tensor:
    """Like ``first_argext_position`` but indices are MSA column positions."""
    if which not in ("max", "min"):
        raise ValueError(f"which must be 'max' or 'min'; got {which!r}")
    valid_idx = torch.where(non_gap_mask)[0]
    if valid_idx.numel() == 0:
        raise ValueError("No non-gap residues for position stat pooling")
    sub = aa_tokens[non_gap_mask]
    if which == "max":
        ext = sub.max(dim=0, keepdim=True).values
    else:
        ext = sub.min(dim=0, keepdim=True).values
    is_ext = sub.eq(ext)
    pos_in_sub = is_ext.float().argmax(dim=0).long()
    return valid_idx[pos_in_sub].float()


def _as_aa_token_matrix(token_repr, truncate_len: int | None = None) -> torch.Tensor:
    """Return AA token rows ``[L, D]`` (exclude CLS at row 0).

    ``token_repr`` is ``[seq+1, D]`` or ``[seq+2, D]`` with CLS at 0.  When
    ``truncate_len`` is set, keep rows ``1 : truncate_len + 1`` (matching
    ``encode_proteins_from_fasta.ESMProteinEncoder._pool``).
    """
    if token_repr.ndim != 2:
        raise ValueError(f"Expected 2D token_repr [L, D]; got shape {tuple(token_repr.shape)}")
    if truncate_len is None:
        return token_repr[1:, :]
    return token_repr[1 : truncate_len + 1, :]


def first_argext_position(aa_tokens: torch.Tensor, *, which: str) -> torch.Tensor:
    """First AA index (0-based) where each dimension reaches its max or min.

    On ties, the **earliest** sequence position is returned (``argmax`` over the
    equality mask along ``dim=0``).
    """
    if which not in ("max", "min"):
        raise ValueError(f"which must be 'max' or 'min'; got {which!r}")
    if aa_tokens.ndim != 2:
        raise ValueError(f"aa_tokens must be [L, D]; got {tuple(aa_tokens.shape)}")
    if which == "max":
        ext = aa_tokens.max(dim=0, keepdim=True).values
    else:
        ext = aa_tokens.min(dim=0, keepdim=True).values
    is_ext = aa_tokens.eq(ext)
    return is_ext.float().argmax(dim=0).float()


def pool_aa_tokens_stats(
    aa_tokens: torch.Tensor | np.ndarray,
    *,
    stat_names: Sequence[str] = STAT_NAMES,
    non_gap_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute per-dimension statistics across AA positions.

    Args:
        aa_tokens: ``[L, D]`` token embeddings (amino-acid rows only).
        non_gap_mask: optional ``[L]`` bool; True = include column in pooling.

    Returns:
        ``[D, n_stats]`` float32 tensor; column order follows ``stat_names``.
    """
    if isinstance(aa_tokens, np.ndarray):
        t = torch.from_numpy(aa_tokens)
    else:
        t = aa_tokens
    if t.ndim != 2:
        raise ValueError(f"aa_tokens must be [L, D]; got {tuple(t.shape)}")

    t = t.float()
    if non_gap_mask is not None:
        if non_gap_mask.shape[0] != t.shape[0]:
            raise ValueError(
                f"non_gap_mask length {non_gap_mask.shape[0]} != aa rows {t.shape[0]}"
            )
        if non_gap_mask.sum().item() == 0:
            raise ValueError("No non-gap residues for stats pooling")
        t = t[non_gap_mask]
        msa_idx = torch.where(non_gap_mask)[0]
    else:
        msa_idx = None

    d = t.shape[1]
    cols: list[torch.Tensor] = []
    for name in stat_names:
        if name == "mean":
            cols.append(t.mean(dim=0))
        elif name == "std":
            cols.append(t.std(dim=0, unbiased=False))
        elif name == "min":
            cols.append(t.min(dim=0).values)
        elif name == "max":
            cols.append(t.max(dim=0).values)
        elif name == "median":
            cols.append(t.median(dim=0).values)
        elif name == "argmax_pos":
            if msa_idx is not None:
                full = aa_tokens if isinstance(aa_tokens, torch.Tensor) else torch.from_numpy(aa_tokens)
                cols.append(
                    first_argext_position_msa(full.float(), non_gap_mask, which="max")
                )
            else:
                cols.append(first_argext_position(t, which="max"))
        elif name == "argmin_pos":
            if msa_idx is not None:
                full = aa_tokens if isinstance(aa_tokens, torch.Tensor) else torch.from_numpy(aa_tokens)
                cols.append(
                    first_argext_position_msa(full.float(), non_gap_mask, which="min")
                )
            else:
                cols.append(first_argext_position(t, which="min"))
        else:
            raise ValueError(f"Unknown stat {name!r}; choose from {STAT_NAMES}")
    return torch.stack(cols, dim=1).reshape(d, len(stat_names)).float()


def pool_token_repr_stats(
    token_repr: torch.Tensor | np.ndarray,
    truncate_len: int | None = None,
    *,
    non_gap_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pool full token matrix (with CLS) to ``[D, n_stats]``."""
    if isinstance(token_repr, np.ndarray):
        token_repr = torch.from_numpy(token_repr)
    aa = _as_aa_token_matrix(token_repr, truncate_len)
    mask = non_gap_mask
    if mask is not None and truncate_len is not None:
        mask = mask[:truncate_len]
    return pool_aa_tokens_stats(aa, non_gap_mask=mask)


def pool_single_stat(
    aa_tokens: torch.Tensor | np.ndarray,
    stat_name: str,
    *,
    seq_len: int | float | None = None,
    normalize_positions: bool = True,
    non_gap_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pool one stat column → ``[D]`` vector.

    Position stats are optionally divided by ``seq_len`` → ``[0, 1]``.
    When ``non_gap_mask`` is set, ``seq_len`` defaults to the non-gap count.
    """
    if stat_name not in STAT_NAMES:
        raise ValueError(f"stat_name must be one of {STAT_NAMES}; got {stat_name!r}")
    col = pool_aa_tokens_stats(
        aa_tokens, stat_names=[stat_name], non_gap_mask=non_gap_mask
    )[:, 0]
    if (
        normalize_positions
        and stat_name in POSITION_STAT_NAMES
        and seq_len is not None
    ):
        col = col / max(float(seq_len), 1.0)
    return col.float()


def pool_token_repr_single_stat(
    token_repr: torch.Tensor | np.ndarray,
    stat_name: str,
    truncate_len: int | None = None,
    *,
    normalize_positions: bool = True,
    non_gap_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pool one stat from a token matrix that includes CLS at row 0."""
    if isinstance(token_repr, np.ndarray):
        token_repr = torch.from_numpy(token_repr)
    aa = _as_aa_token_matrix(token_repr, truncate_len)
    mask = non_gap_mask
    if mask is not None and truncate_len is not None:
        mask = mask[:truncate_len]
    if mask is not None:
        seq_len = int(mask.sum().item())
    else:
        seq_len = aa.shape[0] if truncate_len is None else truncate_len
    return pool_single_stat(
        aa,
        stat_name,
        seq_len=seq_len,
        normalize_positions=normalize_positions,
        non_gap_mask=mask,
    )


def pool_token_repr_slice(
    token_repr: torch.Tensor | np.ndarray,
    start: int,
    end: int,
    pooling_strategy: str,
    *,
    normalize_positions: bool = True,
    non_gap_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pool one MSA column slice ``[start, end)`` from a CLS-prefixed token matrix.

    Row ``1 + col`` corresponds to MSA column ``col`` (half-open slice on columns).
    Gap columns (``-``) are excluded from mean/stat pooling when ``non_gap_mask`` is set.
    """
    if isinstance(token_repr, np.ndarray):
        token_repr = torch.from_numpy(token_repr)
    if token_repr.ndim != 2:
        raise ValueError(
            f"token_repr must be [L+1, D] with CLS at row 0; got {tuple(token_repr.shape)}"
        )
    start_i = int(start)
    end_i = int(end)
    if start_i < 0 or end_i <= start_i:
        raise ValueError(f"Invalid block slice [{start_i}, {end_i})")
    aa = token_repr[1 + start_i : 1 + end_i, :]
    if aa.numel() == 0:
        raise ValueError(f"Empty block slice [{start_i}, {end_i}) on {tuple(token_repr.shape)}")
    slice_mask = None
    if non_gap_mask is not None:
        slice_mask = non_gap_mask[start_i:end_i]

    if pooling_strategy == "average_token":
        return mean_pool_aa_tokens(aa, slice_mask)
    if pooling_strategy == "average_sequence":
        return mean_pool_sequence_dim(aa)
    if pooling_strategy == "cls_token":
        return token_repr[0, :].clone().float()
    if pooling_strategy == "first_token":
        return first_non_gap_row(aa, slice_mask).clone().float()
    if pooling_strategy == "stats_summary":
        return pool_aa_tokens_stats(aa, non_gap_mask=slice_mask).clone()
    if pooling_strategy in EXTRA_STAT_NAMES:
        if slice_mask is not None:
            block_len = int(slice_mask.sum().item())
        else:
            block_len = end_i - start_i
        return pool_single_stat(
            aa,
            pooling_strategy,
            seq_len=block_len,
            normalize_positions=normalize_positions,
            non_gap_mask=slice_mask,
        ).float()
    raise ValueError(
        f"Unknown pooling strategy for block slice: {pooling_strategy!r}. "
        f"Choose from average_token, stats_summary, or {EXTRA_STAT_NAMES}."
    )


def normalize_position_stats(
    stats: torch.Tensor | np.ndarray,
    seq_len: float,
) -> np.ndarray:
    """Scale ``argmax_pos`` / ``argmin_pos`` columns to ``[0, 1]`` by ``seq_len``."""
    arr = stats.detach().cpu().numpy().copy() if isinstance(stats, torch.Tensor) else np.asarray(stats, dtype=np.float32).copy()
    if arr.ndim != 2 or arr.shape[1] != N_STATS:
        raise ValueError(f"Expected [D, {N_STATS}]; got {arr.shape}")
    denom = max(float(seq_len), 1.0)
    for name in POSITION_STAT_NAMES:
        arr[:, STAT_NAMES.index(name)] /= denom
    return arr.astype(np.float32)


def flatten_stats(
    stats: torch.Tensor | np.ndarray,
    *,
    stat_major: bool = False,
) -> np.ndarray:
    """``[D, n_stats]`` → ``[D * n_stats]``.

    * ``stat_major=False`` (legacy): dim-major — ``[d0s0, d0s1, …, d1s0, …]``
    * ``stat_major=True``: stat blocks — ``[mean_0…mean_D-1, std_0…, …]``
    """
    arr = stats.detach().cpu().numpy() if isinstance(stats, torch.Tensor) else np.asarray(stats)
    if arr.ndim != 2:
        raise ValueError(f"Expected [D, n_stats]; got {arr.shape}")
    if stat_major:
        return np.concatenate([arr[:, j] for j in range(arr.shape[1])], axis=0).astype(np.float32)
    return arr.reshape(-1).astype(np.float32)


def stats_slice(stats: torch.Tensor | np.ndarray, stat_name: str) -> np.ndarray:
    """Extract one stat column as ``[D]`` vector."""
    if stat_name not in STAT_NAMES:
        raise ValueError(f"stat_name must be one of {STAT_NAMES}; got {stat_name!r}")
    idx = STAT_NAMES.index(stat_name)
    arr = stats.detach().cpu().numpy() if isinstance(stats, torch.Tensor) else np.asarray(stats)
    return arr[:, idx].astype(np.float32)
