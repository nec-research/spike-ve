#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     cycle_eval.py
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
"""Helpers for fair decoder evaluation.

* ``load_source_latents``: *clean* (ground-truth) latents of source genomes, built from the
  per-site ESM 2D encodings. Gives both the PCA2D flattened projection (what PCA2D decoders were
  trained on) and the mean-pooled raw ESM vector (what mean decoders were trained on).
* ``load_source_sequences`` / ``sequence_accuracy``: compare decoded sequences with ground truth.
* ``derive_split_prefix`` / ``default_*_key``: NPZ key conventions (``test_*`` / ``cal_*``).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

GAP = "-"


def derive_split_prefix(latent_key: Optional[str]) -> str:
    """``test_pred_pca2d_flat_raw`` -> ``test_``; ``cal_x`` -> ``cal_``; default ``test_``."""
    if latent_key:
        for p in ("test_", "cal_", "train_", "val_"):
            if str(latent_key).startswith(p):
                return p
    return "test_"


def default_source_gids_key(latent_key: Optional[str]) -> str:
    return f"{derive_split_prefix(latent_key)}source_gids"


def default_mean_reference_key(latent_key: Optional[str]) -> str:
    return f"{derive_split_prefix(latent_key)}pred_mean_raw"


def load_source_latents(
    gids: Sequence[str],
    enc_2d_dir: Path,
    *,
    layer: int,
    seq_len: int,
    basis: Optional[dict[str, Any]] = None,
    num_workers: int = 16,
    log: Any = None,
) -> dict[str, Any]:
    """Clean latents for ``gids`` from 2D ESM encodings (duplicates are computed once).

    Returns ``{"pca2d": (N, L*k) | None, "mean": (N, D), "found": bool mask (N,)}``.
    Rows of missing genomes are zero and flagged ``found=False``.
    """
    from recomb2027.latent.data_series.esm_2d_io import build_gid_file_index
    from recomb2027.latent.data_series.pca2d_features import (
        load_aa_matrix_for_basis,
        project_aa_to_pca2d_flat,
    )

    index = build_gid_file_index(Path(enc_2d_dir), layer=layer)
    gids = [str(g) for g in gids]
    unique = list(dict.fromkeys(gids))
    if log is not None:
        log.info(f"clean baseline: {len(unique):,} unique source gids ({len(gids):,} rows)")

    def one(gid: str):
        if gid not in index:
            return None
        aa = load_aa_matrix_for_basis(index[gid], layer=layer, seq_len=seq_len)
        mean = aa.mean(axis=0).astype(np.float32)
        pca = None
        if basis is not None:
            pca = project_aa_to_pca2d_flat(aa, W=basis["W"], mu=basis["mu"])
        return mean, pca

    with ThreadPoolExecutor(max_workers=max(1, int(num_workers))) as ex:
        results = list(ex.map(one, unique))

    ok = [r for r in results if r is not None]
    if not ok:
        raise ValueError(f"None of the {len(unique)} source gids found in {enc_2d_dir}")
    d_mean = ok[0][0].shape[0]
    d_pca = ok[0][1].shape[0] if ok[0][1] is not None else None
    u_mean = np.zeros((len(unique), d_mean), dtype=np.float32)
    u_pca = np.zeros((len(unique), d_pca), dtype=np.float32) if d_pca is not None else None
    u_found = np.zeros(len(unique), dtype=bool)
    for i, r in enumerate(results):
        if r is None:
            continue
        u_found[i] = True
        u_mean[i] = r[0]
        if u_pca is not None:
            u_pca[i] = r[1]

    pos = {g: i for i, g in enumerate(unique)}
    rows = np.array([pos[g] for g in gids], dtype=np.int64)
    out = {
        "mean": u_mean[rows],
        "pca2d": u_pca[rows] if u_pca is not None else None,
        "found": u_found[rows],
    }
    n_missing = int((~out["found"]).sum())
    if n_missing and log is not None:
        log.info(f"clean baseline: {n_missing:,} rows have no 2D encoding (will be dropped)")
    return out


def load_source_sequences(
    fasta_path: Path,
    gids: Sequence[str],
) -> list[Optional[str]]:
    """Aligned ground-truth sequences by ``genome_id`` (``None`` when missing)."""
    from recomb2027.generation.latent_decoder.training.splits import (
        parse_fasta_with_metadata,
    )

    seqs, metas = parse_fasta_with_metadata(str(fasta_path))
    by_gid: dict[str, str] = {}
    for s, m in zip(seqs, metas):
        gid = m.get("genome_id") or m.get("id")
        if gid is not None and str(gid) not in by_gid:
            by_gid[str(gid)] = s
    return [by_gid.get(str(g)) for g in gids]


def sequence_accuracy(
    decoded: Sequence[str],
    targets: Sequence[Optional[str]],
) -> dict[str, Any]:
    """Per-row identity vs ground truth (all sites and non-gap target sites)."""
    ident, nongap, exact = [], [], []
    for d, t in zip(decoded, targets):
        if t is None:
            ident.append(np.nan)
            nongap.append(np.nan)
            exact.append(np.nan)
            continue
        n = min(len(d), len(t))
        a = np.frombuffer(d[:n].encode(), dtype=np.uint8)
        b = np.frombuffer(t[:n].encode(), dtype=np.uint8)
        eq = a == b
        ident.append(float(eq.sum()) / max(len(t), 1))
        m = b != ord(GAP)
        nongap.append(float(eq[m].sum()) / max(int(m.sum()), 1))
        exact.append(float(len(d) == len(t) and bool(eq.all())))
    ident_a = np.asarray(ident)
    nongap_a = np.asarray(nongap)
    exact_a = np.asarray(exact)
    valid = ~np.isnan(ident_a)
    return {
        "identity": ident_a,
        "nongap_acc": nongap_a,
        "n_with_truth": int(valid.sum()),
        "identity_mean": float(np.nanmean(ident_a)) if valid.any() else float("nan"),
        "nongap_acc_mean": float(np.nanmean(nongap_a)) if valid.any() else float("nan"),
        "exact_match_frac": float(np.nanmean(exact_a)) if valid.any() else float("nan"),
    }


def input_scale_warnings(
    model: Any,
    z_np: np.ndarray,
    *,
    n: int = 256,
    lo: float = 0.25,
    hi: float = 3.0,
) -> list[str]:
    """Detect inputs in the wrong units (e.g. ``*_pred_pca2d`` instead of ``*_pred_pca2d_flat_raw``).

    Decoders with stored normalisation (NAR / fused) expect raw latents; after their own normalisation
    healthy inputs have std ~0.6-1.2 (predictions are shrunk), whereas scaled/delta keys land at 6-20.
    ``z_np`` is exactly what is fed to the decoder (packed ``[sites | mean]`` for fused models).
    """
    import torch

    if not hasattr(model, "site_mean"):
        return []
    z = torch.as_tensor(np.asarray(z_np[:n]), dtype=torch.float32)
    if hasattr(model, "mean_mu"):  # fused: packed input
        sites, mean = model.split(z)
        parts = {
            "PCA2D sites": (sites, model.site_mean, model.latent_scale),
            "mean vector": (mean, model.mean_mu, model.mean_scale),
        }
    else:
        parts = {
            "PCA2D sites": (z.reshape(-1, model.seq_len, model.k), model.site_mean, model.latent_scale)
        }
    msgs = []
    for name, (x, mu, sc) in parts.items():
        std = float(((x - mu.cpu()) / sc.cpu()).std())
        if not (lo <= std <= hi):
            msgs.append(
                f"{name} look to be in the wrong units: std after the decoder's own normalisation "
                f"is {std:.2f} (expected {lo}-{hi}). Use the *_raw NPZ keys."
            )
    return msgs
