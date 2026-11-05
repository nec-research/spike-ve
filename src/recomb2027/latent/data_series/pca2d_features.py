#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     pca2d_features.py
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
"""PCA2D feature helpers for S1 (and region) JEPA-GPT training.

Projects per-residue ESM AA matrices through a feature-PCA basis
(``W`` [k, D], ``mu`` [D]) into flattened site features ``[L * k]``,
matching the GA metric ``pca2d_euclidean``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import torch

from recomb2027.latent.data_series.esm_2d_io import (
    build_gid_file_index,
    load_2d_encoding,
    pad_or_truncate_aa_matrix,
)

logger = logging.getLogger(__name__)

PCA2D_S1_TARGET = "pca2d_s1"


def load_pca2d_basis(path: Union[str, Path]) -> dict[str, Any]:
    """Load a ``fit_pca2d_*`` artifact and normalise fields to numpy/float."""
    path = Path(path)
    raw = torch.load(path, map_location="cpu", weights_only=False)
    W = raw["W"]
    mu = raw["mu"]
    if hasattr(W, "detach"):
        W = W.detach().cpu().numpy()
    if hasattr(mu, "detach"):
        mu = mu.detach().cpu().numpy()
    W = np.asarray(W, dtype=np.float32)
    mu = np.asarray(mu, dtype=np.float32)
    if W.ndim != 2:
        raise ValueError(f"basis W must be [k, D]; got {W.shape}")
    k = int(raw.get("k", W.shape[0]))
    seq_len = int(raw["seq_len"])
    layer = int(raw.get("layer", 33))
    return {
        "W": W,
        "mu": mu,
        "k": k,
        "seq_len": seq_len,
        "layer": layer,
        "feature_dim": int(raw.get("feature_dim", W.shape[1])),
        "variance_explained": float(raw.get("variance_explained", float("nan"))),
        "region": raw.get("region"),
        "path": str(path),
        "flat_dim": int(seq_len * k),
    }


def load_aa_matrix_for_basis(
    fpath: Path,
    *,
    layer: int,
    seq_len: int,
) -> np.ndarray:
    """Load AA tokens aligned to ``seq_len`` (CLS auto-detected)."""
    arr = load_2d_encoding(fpath, layer, drop_cls=False)
    if arr.shape[0] == seq_len + 1:
        arr = arr[1:, :]
    return pad_or_truncate_aa_matrix(arr, seq_len)


def project_aa_to_pca2d_flat(
    aa: np.ndarray,
    *,
    W: np.ndarray,
    mu: np.ndarray,
) -> np.ndarray:
    """``[L, D]`` → flattened ``[L * k]`` site PCA features."""
    aa = np.asarray(aa, dtype=np.float32)
    sites = (aa - mu.reshape(1, -1)) @ W.T
    return sites.reshape(-1).astype(np.float32)


def project_gid_to_pca2d_flat(
    gid: str,
    gid_index: dict[str, Path],
    basis: dict[str, Any],
) -> np.ndarray:
    if gid not in gid_index:
        raise KeyError(f"genome_id {gid!r} missing from 2D encoding index")
    aa = load_aa_matrix_for_basis(
        gid_index[gid],
        layer=int(basis["layer"]),
        seq_len=int(basis["seq_len"]),
    )
    return project_aa_to_pca2d_flat(aa, W=basis["W"], mu=basis["mu"])


def _infer_series_k(
    sample_gids: np.ndarray,
    *,
    k_series: Optional[int] = None,
) -> int:
    """Infer series length ``K`` from stacked or ragged ``sample_gids``."""
    if k_series is not None and int(k_series) > 0:
        return int(k_series)
    gids = np.asarray(sample_gids, dtype=object)
    if gids.size == 0:
        return 0
    if gids.ndim >= 2:
        return int(gids.shape[1])
    return len(gids[0])


def build_pca2d_cube_for_series(
    sample_gids: np.ndarray,
    enc_2d_dir: Union[str, Path],
    basis: dict[str, Any],
    *,
    progress_every: int = 2000,
    k_series: Optional[int] = None,
) -> np.ndarray:
    """Stack pca2d flats for every series row → ``[N, K, L*k]``."""
    enc_2d_dir = Path(enc_2d_dir)
    gid_index = build_gid_file_index(enc_2d_dir)
    if not gid_index:
        raise FileNotFoundError(f"No .pt files under {enc_2d_dir}")

    gids = np.asarray(sample_gids, dtype=object)
    flat_dim = int(basis["flat_dim"])
    n = int(gids.shape[0]) if gids.ndim >= 1 else 0
    k_series = _infer_series_k(gids, k_series=k_series)
    if n == 0:
        return np.zeros((0, k_series, flat_dim), dtype=np.float32)

    ragged = gids.ndim == 1
    out = np.zeros((n, k_series, flat_dim), dtype=np.float32)
    cache: dict[str, np.ndarray] = {}
    missing: list[str] = []

    total = n * k_series
    done = 0
    for i in range(n):
        row = gids[i]
        row_len = len(row) if ragged else k_series
        for j in range(min(row_len, k_series)):
            gid = str(row[j] if ragged else gids[i, j])
            if gid not in cache:
                if gid not in gid_index:
                    missing.append(gid)
                    cache[gid] = np.zeros(flat_dim, dtype=np.float32)
                else:
                    cache[gid] = project_gid_to_pca2d_flat(gid, gid_index, basis)
            out[i, j] = cache[gid]
            done += 1
            if progress_every and done % progress_every == 0:
                logger.info("pca2d project %d / %d (unique=%d)", done, total, len(cache))

    if missing:
        uniq = sorted(set(missing))
        logger.warning(
            "pca2d: %d missing genome ids (zero-filled); showing up to 5: %s",
            len(uniq),
            uniq[:5],
        )
    return out


def mean_pca2d_input_layout(
    *,
    primary_dim: int,
    pca2d_dim: int,
    gatt_plus: bool = True,
    pca2d_gatt_plus: bool = True,
) -> dict[str, tuple[int, int]]:
    """Channel slices for mean | pca2d dual-branch input."""
    d = int(primary_dim)
    p = int(pca2d_dim)
    mean_w = 2 * d if gatt_plus else d
    pca_w = 2 * p if pca2d_gatt_plus else p
    return {
        "primary_slice": (0, mean_w),
        "pca2d_slice": (mean_w, mean_w + pca_w),
    }


def expected_mean_pca2d_input_dim(
    *,
    primary_dim: int,
    pca2d_dim: int,
    gatt_plus: bool = True,
    pca2d_gatt_plus: bool = True,
) -> int:
    layout = mean_pca2d_input_layout(
        primary_dim=primary_dim,
        pca2d_dim=pca2d_dim,
        gatt_plus=gatt_plus,
        pca2d_gatt_plus=pca2d_gatt_plus,
    )
    return int(layout["pca2d_slice"][1])


def concat_mean_pca2d_prefix_features(
    x_abs_mean: np.ndarray,
    x_abs_pca2d: np.ndarray,
    *,
    gatt_plus: bool = True,
    pca2d_gatt_plus: bool = True,
) -> np.ndarray:
    """Build ``[T, Din]`` from absolute mean + pca2d prefix steps."""
    x_mean = np.asarray(x_abs_mean, dtype=np.float32)
    x_pca = np.asarray(x_abs_pca2d, dtype=np.float32)
    if x_mean.shape[0] != x_pca.shape[0]:
        raise ValueError(
            f"mean/pca2d prefix length mismatch: {x_mean.shape[0]} vs {x_pca.shape[0]}"
        )

    def _maybe_gatt(x: np.ndarray, enabled: bool) -> np.ndarray:
        if not enabled:
            return x
        v = np.zeros_like(x)
        if x.shape[0] > 1:
            v[1:] = x[1:] - x[:-1]
        return np.concatenate([x, v], axis=-1)

    return np.concatenate(
        [_maybe_gatt(x_mean, gatt_plus), _maybe_gatt(x_pca, pca2d_gatt_plus)],
        axis=-1,
    ).astype(np.float32)


def is_pca2d_target(name: str) -> bool:
    return str(name).startswith("pca2d_")
