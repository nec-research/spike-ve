#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     analyse_baseline_knn.py
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
""" 
Search kNN real sequences close to predicted embeddings, create consensus sequence out of kNNs.
Compare GT sequences to kNN when provided.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from pathlib import Path
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from recomb2027.config_io import load_config

logger = logging.getLogger(__name__)

KNN_SPACES = ("mean", "pca2d", "combined")
KNN_METRICS = ("euclidean", "cosine")
_DIST_CHUNK_ROWS = 2048
DEFAULT_KNN_BEFORE_DATE = "2022-12-01"


def gid_str(value: Any) -> str:
    """Stable genome-id string (avoid float scientific notation)."""
    if value is None:
        return ""
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace").strip()
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return ""
        return np.format_float_positional(float(value), trim="-")
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "<na>"}:
        return ""
    return text


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )


def hamming(a: str, b: str) -> int:
    n = min(len(a), len(b))
    return sum(1 for i in range(n) if a[i] != b[i]) + abs(len(a) - len(b))


def identity(a: str, b: str) -> float:
    n = max(len(a), len(b), 1)
    return 1.0 - hamming(a, b) / float(n)


def _gid_from_label_or_name(label: Any, stem: str) -> str:
    text = str(label) if label is not None else ""
    m = re.search(r"genome_id:([^|]+)", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"genome_id_([\d]+\.[\d]+)", stem)
    if m:
        return m.group(1)
    m = re.search(r"(\d+\.\d+)", stem)
    if m:
        return m.group(1)
    return stem


def _as_mean_vector(enc: Any) -> Optional[np.ndarray]:
    if isinstance(enc, torch.Tensor):
        enc = enc.detach().cpu().numpy()
    arr = np.asarray(enc, dtype=np.float32)
    if arr.ndim == 0:
        return None
    return arr.reshape(-1)


def load_encoding_index(
    encodings_dir: Path,
    *,
    max_files: Optional[int] = None,
    allowed_gids: Optional[set[str]] = None,
) -> tuple[np.ndarray, list[str]]:
    """Load mean embeddings from `.pt` files under encodings_dir (gid → vector).

    ``max_files`` caps the number of *kept* vectors (after ``allowed_gids``).
    Values ``<= 0`` are treated as unlimited (same as ``None``).
    """
    if max_files is not None and int(max_files) <= 0:
        max_files = None

    files = sorted(encodings_dir.glob("*.pt"))

    vecs: list[np.ndarray] = []
    gids: list[str] = []

    for fp in files:
        if allowed_gids is not None:
            m_stem = re.search(r"genome_id_([\d]+\.[\d]+)", fp.stem)
            if m_stem and gid_str(m_stem.group(1)) not in allowed_gids:
                continue
        try:
            data = torch.load(fp, map_location="cpu", weights_only=False)
        except Exception as e:
            logger.warning("skip %s: %s", fp, e)
            continue

        if not isinstance(data, dict):
            logger.warning("skip %s: expected dict, got %s", fp, type(data))
            continue

        label = None
        for k in ("label", "labels"):
            if k in data:
                label = data[k]
                break
        gid = gid_str(_gid_from_label_or_name(label, fp.stem))
        if not gid:
            continue
        if allowed_gids is not None and gid not in allowed_gids:
            continue

        enc = None
        for k in ("encodings", "mean", "embedding"):
            if k in data:
                enc = _as_mean_vector(data[k])
                break

        if enc is None:
            continue

        gids.append(gid)
        vecs.append(enc)
        if max_files is not None and len(vecs) >= int(max_files):
            break

    if not vecs:
        raise FileNotFoundError(
            f"no encodings in {encodings_dir}"
            + (" after date/gid filter" if allowed_gids is not None else "")
        )

    return np.stack(vecs, axis=0), gids


def _pca2d_basis_mtime(basis_path: Path) -> float:
    try:
        return float(Path(basis_path).stat().st_mtime)
    except OSError:
        return float("nan")


def default_pca2d_index_cache_path(
    enc_2d_dir: Path,
    basis_path: Path,
    *,
    knn_before_date: Optional[str] = None,
    max_files: Optional[int] = None,
) -> Path:
    """Stable cache path under ``enc_2d_dir/_knn_pca2d_cache/``."""
    date_tag = "none" if knn_before_date is None else str(knn_before_date)
    max_tag = "all" if max_files is None else str(int(max_files))
    safe_basis = re.sub(r"[^\w.\-]+", "_", Path(basis_path).stem)
    name = f"{safe_basis}__before{date_tag}__max{max_tag}.npz"
    return Path(enc_2d_dir) / "_knn_pca2d_cache" / name


def try_load_pca2d_index_cache(
    cache_path: Path,
    *,
    requested_gids: list[str],
    basis_path: Path,
    enc_2d_dir: Path,
) -> Optional[tuple[np.ndarray, list[str]]]:
    """Load cached pca2d flats when genome ids (and basis path) match."""
    if not cache_path.is_file():
        return None
    try:
        z = np.load(cache_path, allow_pickle=True)
    except Exception as e:
        logger.warning("pca2d cache unreadable %s: %s", cache_path, e)
        return None
    try:
        if "pca2d" not in z.files or "gids" not in z.files:
            logger.warning("pca2d cache missing keys in %s", cache_path)
            return None
        cached_basis = (
            str(np.asarray(z["basis_path"]).item()) if "basis_path" in z.files else ""
        )
        if cached_basis and Path(cached_basis).resolve() != Path(basis_path).resolve():
            logger.info(
                "pca2d cache miss: basis_path mismatch (%s vs %s)",
                cached_basis,
                basis_path,
            )
            return None
        if "basis_mtime" in z.files:
            cached_mtime = float(np.asarray(z["basis_mtime"]).item())
            cur_mtime = _pca2d_basis_mtime(basis_path)
            if (
                math.isfinite(cached_mtime)
                and math.isfinite(cur_mtime)
                and abs(cached_mtime - cur_mtime) > 1e-6
            ):
                logger.info("pca2d cache miss: basis file mtime changed")
                return None
        cached_gids = [str(g) for g in np.asarray(z["gids"]).tolist()]
        arr = np.asarray(z["pca2d"], dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] != len(cached_gids):
            logger.warning(
                "pca2d cache shape mismatch in %s: %s vs n_gids=%d",
                cache_path,
                arr.shape,
                len(cached_gids),
            )
            return None
        if cached_gids == list(requested_gids):
            logger.info(
                "pca2d cache hit %s (n=%d dim=%d, exact gid list)",
                cache_path,
                arr.shape[0],
                arr.shape[1],
            )
            return arr, cached_gids
        idx_map = {g: i for i, g in enumerate(cached_gids)}
        if all(g in idx_map for g in requested_gids):
            order = [idx_map[g] for g in requested_gids]
            logger.info(
                "pca2d cache hit %s (subset/reorder %d → %d)",
                cache_path,
                len(cached_gids),
                len(requested_gids),
            )
            return arr[np.asarray(order, dtype=np.int64)], list(requested_gids)
        logger.info(
            "pca2d cache miss %s: requested %d gids, cache has %d, overlap=%d",
            cache_path,
            len(requested_gids),
            len(cached_gids),
            sum(1 for g in requested_gids if g in idx_map),
        )
        return None
    finally:
        z.close()


def save_pca2d_index_cache(
    cache_path: Path,
    *,
    pca2d: np.ndarray,
    gids: list[str],
    basis_path: Path,
    enc_2d_dir: Path,
    knn_before_date: Optional[str] = None,
    max_files: Optional[int] = None,
) -> None:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pca2d": np.asarray(pca2d, dtype=np.float32),
        "gids": np.asarray(gids, dtype=object),
        "basis_path": str(Path(basis_path)),
        "basis_mtime": np.float64(_pca2d_basis_mtime(basis_path)),
        "enc_2d_dir": str(Path(enc_2d_dir)),
        "knn_before_date": "" if knn_before_date is None else str(knn_before_date),
        "max_files": -1 if max_files is None else int(max_files),
    }
    # np.savez_compressed appends .npz if missing; keep a real .npz tmp name.
    tmp = cache_path.with_name(cache_path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **payload)
    tmp.replace(cache_path)
    logger.info("wrote pca2d cache %s (n=%d dim=%d)", cache_path, pca2d.shape[0], pca2d.shape[1])


def load_pca2d_encoding_index(
    enc_2d_dir: Path,
    basis_path: Path,
    *,
    gids: Optional[list[str]] = None,
    max_files: Optional[int] = None,
    allowed_gids: Optional[set[str]] = None,
    progress_every: int = 500,
    cache_path: Optional[Path] = None,
    knn_before_date: Optional[str] = None,
) -> tuple[np.ndarray, list[str]]:
    """Project 2D AA encodings to flattened pca2d features for an MSA index.

    When ``cache_path`` is set, reuse a previous NPZ if its genome-id list matches
    (exact or reorderable subset); otherwise recompute and overwrite the cache.
    """
    from recomb2027.latent.data_series.esm_2d_io import build_gid_file_index
    from recomb2027.latent.data_series.pca2d_features import (
        load_pca2d_basis,
        project_gid_to_pca2d_flat,
    )

    enc_2d_dir = Path(enc_2d_dir)
    basis_path = Path(basis_path)
    if max_files is not None and int(max_files) <= 0:
        max_files = None
    basis = load_pca2d_basis(basis_path)
    gid_index = build_gid_file_index(enc_2d_dir)
    if not gid_index:
        raise FileNotFoundError(f"no 2D encodings in {enc_2d_dir}")

    if gids is None:
        ordered = sorted(gid_index.keys())
    else:
        ordered = list(gids)

    if allowed_gids is not None:
        ordered = [g for g in ordered if g in allowed_gids]

    # Resolve candidates that can actually be projected (present on disk).
    candidate_gids: list[str] = []
    for gid in ordered:
        if max_files is not None and len(candidate_gids) >= int(max_files):
            break
        if gid in gid_index:
            candidate_gids.append(gid)

    if not candidate_gids:
        raise FileNotFoundError(
            f"no pca2d projections from {enc_2d_dir} (no overlapping 2D files)"
        )

    if cache_path is not None:
        cached = try_load_pca2d_index_cache(
            Path(cache_path),
            requested_gids=candidate_gids,
            basis_path=basis_path,
            enc_2d_dir=enc_2d_dir,
        )
        if cached is not None:
            return cached

    vecs: list[np.ndarray] = []
    out_gids: list[str] = []
    missing = 0
    for i, gid in enumerate(candidate_gids):
        try:
            vec = project_gid_to_pca2d_flat(gid, gid_index, basis)
        except Exception as e:
            logger.warning("skip pca2d %s: %s", gid, e)
            missing += 1
            continue
        out_gids.append(gid)
        vecs.append(vec)
        if progress_every and (i + 1) % int(progress_every) == 0:
            logger.info(
                "pca2d index %d / %d (kept=%d)",
                i + 1,
                len(candidate_gids),
                len(out_gids),
            )

    if not vecs:
        raise FileNotFoundError(
            f"no pca2d projections from {enc_2d_dir} (missing={missing})"
        )
    if missing:
        logger.warning("pca2d index skipped %d gids without usable 2D encodings", missing)

    arr = np.stack(vecs, axis=0)
    if cache_path is not None:
        # Store successfully projected rows (genome ids that match this matrix).
        save_pca2d_index_cache(
            Path(cache_path),
            pca2d=arr,
            gids=out_gids,
            basis_path=basis_path,
            enc_2d_dir=enc_2d_dir,
            knn_before_date=knn_before_date,
            max_files=max_files,
        )
    return arr, out_gids


def split_threshold_from_dataset_config(path: Path) -> str:
    """Read ``split_threshold`` from a latent dataset YAML."""
    from omegaconf import OmegaConf

    cfg = load_config(str(path))
    thr = cfg.get("split_threshold")
    if thr is None:
        raise ValueError(f"no split_threshold in {path}")
    return str(thr)


def resolve_knn_before_date(
    *,
    knn_before_date: Optional[str] = None,
    dataset_config: Optional[Path] = None,
    no_knn_date_filter: bool = False,
) -> Optional[str]:
    """Resolve inclusive collection-date cutoff for kNN index leakage control.

    Returns ``None`` to disable filtering. Default cutoff is ``split_threshold``
    from ``dataset_config`` when provided, else ``DEFAULT_KNN_BEFORE_DATE``.
    """
    if no_knn_date_filter:
        return None
    if knn_before_date is not None:
        text = str(knn_before_date).strip()
        if text.lower() in {"", "none", "off", "false"}:
            return None
        return text
    if dataset_config is not None:
        return split_threshold_from_dataset_config(Path(dataset_config))
    return DEFAULT_KNN_BEFORE_DATE


def gids_on_or_before_date(
    gid_to_date: dict[str, str],
    before_date: str,
    *,
    inclusive: bool = True,
) -> set[str]:
    """Return genome ids with ``collection_date`` <= (or <) ``before_date``."""
    cutoff = pd.Timestamp(before_date)
    keep: set[str] = set()
    n_bad = 0
    for gid, raw in gid_to_date.items():
        ts = pd.to_datetime(raw, errors="coerce")
        if pd.isna(ts):
            n_bad += 1
            continue
        if inclusive:
            if ts <= cutoff:
                keep.add(gid)
        elif ts < cutoff:
            keep.add(gid)
    if n_bad:
        logger.warning(
            "ignored %d genome ids with unparseable collection_date", n_bad
        )
    return keep


def load_collection_dates_from_metadata(
    metadata_csv: Path | str,
) -> dict[str, str]:
    """Load ``{genome_id: collection_date}`` from a labels CSV (string ids)."""
    path = Path(metadata_csv)
    meta = pd.read_csv(
        path,
        dtype={"genome_id": "string"},
        usecols=lambda c: c in {"genome_id", "collection_date"},
    )
    if "genome_id" not in meta.columns or "collection_date" not in meta.columns:
        raise ValueError(
            f"{path} must contain genome_id and collection_date columns"
        )
    out: dict[str, str] = {}
    gids = meta["genome_id"].astype(str)
    dates = meta["collection_date"].astype(str)
    for gid_raw, date_raw in zip(gids.tolist(), dates.tolist()):
        gid = str(gid_raw).strip()
        if not gid or gid.lower() in {"nan", "<na>", "none"}:
            continue
        date = str(date_raw).strip()
        if not date or date.lower() in {"nan", "none", "nat", "<na>"}:
            continue
        out[gid] = date
    return out


def load_lineages_from_metadata(
    metadata_csv: Path | str,
    *,
    lineage_col: str = "lineage",
) -> dict[str, str]:
    """Load ``{genome_id: lineage}`` from a labels CSV (string ids).

    Prefers ``lineage_col`` (default ``lineage``); falls back to ``who_lineage``
    when the preferred column is absent.
    """
    path = Path(metadata_csv)
    meta = pd.read_csv(path, dtype={"genome_id": "string"}, low_memory=False)
    if "genome_id" not in meta.columns:
        raise ValueError(f"{path} must contain a genome_id column")
    col = lineage_col if lineage_col in meta.columns else None
    if col is None and "who_lineage" in meta.columns:
        col = "who_lineage"
    if col is None and "lineage" in meta.columns:
        col = "lineage"
    if col is None:
        logger.warning(
            "metadata %s has no lineage/who_lineage column; lineage headers skipped",
            path,
        )
        return {}
    out: dict[str, str] = {}
    gids = meta["genome_id"].astype(str)
    lins = meta[col].astype(str)
    for gid_raw, lin_raw in zip(gids.tolist(), lins.tolist()):
        gid = str(gid_raw).strip()
        if not gid or gid.lower() in {"nan", "<na>", "none"}:
            continue
        lin = str(lin_raw).strip()
        if not lin or lin.lower() in {"nan", "none", "<na>"}:
            continue
        out[gid] = lin
    logger.info("loaded %d lineages from %s (column=%s)", len(out), path, col)
    return out


def load_dates_from_metadata(
    metadata_csv: Path | str,
    *,
    date_col: str = "collection_date",
) -> dict[str, str]:
    """Load ``{genome_id: lineage}`` from a labels CSV (string ids).

    Prefers ``lineage_col`` (default ``lineage``); falls back to ``who_lineage``
    when the preferred column is absent.
    """
    path = Path(metadata_csv)
    meta = pd.read_csv(path, dtype={"genome_id": "string"}, low_memory=False)
    if "genome_id" not in meta.columns:
        raise ValueError(f"{path} must contain a genome_id column")
    col = date_col if date_col in meta.columns else None
    if col is None:
        logger.warning(
            "metadata %s has no lineage/who_lineage column; lineage headers skipped",
            path,
        )
        return {}
    out: dict[str, str] = {}
    gids = meta["genome_id"].astype(str)
    dates = meta[col].astype(str)
    for gid_raw, date_raw in zip(gids.tolist(), dates.tolist()):
        gid = str(gid_raw).strip()
        if not gid or gid.lower() in {"nan", "<na>", "none"}:
            continue
        date = str(date_raw).strip()
        if not date or date.lower() in {"nan", "none", "<na>"}:
            continue
        out[gid] = date
    logger.info("loaded %d collection dates from %s (column=%s)", len(out), path, col)
    return out


def resolve_lineage(
    gid: str,
    gid_to_lineage: Optional[dict[str, str]],
    *,
    fallback: str = "",
) -> str:
    if not gid:
        return fallback
    if gid_to_lineage:
        if gid in gid_to_lineage:
            return gid_to_lineage[gid]
        matches = [
            v
            for k, v in gid_to_lineage.items()
            if k.startswith(gid) or gid.startswith(k)
        ]
        if len(matches) == 1:
            return matches[0]
    return fallback


def resolve_collection_date(
    gid: str,
    gid_to_date: Optional[dict[str, str]],
    *,
    fallback: str = "",
    date_format: str = "%Y-%m-%d",
) -> str:
    if not gid:
        return fallback
    if gid_to_date:
        if gid in gid_to_date:
            return gid_to_date[gid]
        matches = [
            v
            for k, v in gid_to_date.items()
            if k.startswith(gid)
        ]
        if not matches:
            return fallback
        try:
            earliest_date = min(
                matches, key=lambda d: datetime.strptime(d, date_format)
            )
            return str(earliest_date)
        except ValueError:
            # Fallback if any date string fails to parse
            return fallback           

            # if len(matches) >= 1:
            #     return matches[0]
    return fallback


def fasta_header(
    kind: str,
    gid: str,
    *,
    lineage: str = "",
    extras: Optional[list[str]] = None,
) -> str:
    """Build a FASTA header with optional lineage and extra ``key=value`` fields."""
    parts = [str(kind)]
    if gid:
        parts.append(str(gid))
    if lineage:
        parts.append(f"lineage={lineage}")
    if extras:
        parts.extend(extras)
    return ">" + "|".join(parts)


def build_knn_allowed_gids(
    msa_path: Path | str,
    *,
    before_date: Optional[str],
    metadata_csv: Optional[Path | str] = None,
    msa_gids: Optional[set[str]] = None,
) -> Optional[set[str]]:
    """Build allowed kNN genome ids from collection dates, restricted to ``msa_path``.

    When ``metadata_csv`` is provided it has **priority** over MSA FASTA header
    dates; MSA headers only fill genome ids missing from metadata. Sequences
    without a usable date, or absent from the target MSA (e.g. removed during
    deduplication), are excluded when filtering is enabled.
    """
    if before_date is None:
        return None
    from recomb2027.latent.data_series.latent_decoder_io import (
        parse_msa_collection_dates,
        parse_msa_gids,
    )

    gid_to_date: dict[str, str] = {}
    meta_n = 0
    if metadata_csv is not None:
        meta_path = Path(metadata_csv)
        if not meta_path.is_file():
            raise FileNotFoundError(f"metadata CSV not found: {meta_path}")
        gid_to_date = load_collection_dates_from_metadata(meta_path)
        meta_n = len(gid_to_date)
        logger.info(
            "kNN dates: loaded %d collection_date rows from metadata %s",
            meta_n,
            meta_path,
        )

    msa_dates = parse_msa_collection_dates(msa_path)
    filled = 0
    for gid, date in msa_dates.items():
        if gid not in gid_to_date:
            gid_to_date[gid] = date
            filled += 1
    if metadata_csv is None:
        logger.info(
            "kNN dates: using %d collection_date values from MSA headers only",
            len(msa_dates),
        )
    elif filled:
        logger.info(
            "kNN dates: filled %d genome ids from MSA headers "
            "(metadata has priority for overlapping ids)",
            filled,
        )

    allowed = gids_on_or_before_date(gid_to_date, before_date, inclusive=True)
    valid_msa_gids = (
        {gid_str(g) for g in msa_gids if gid_str(g)}
        if msa_gids is not None
        else {gid_str(g) for g in parse_msa_gids(msa_path).keys() if gid_str(g)}
    )
    pre_msa_n = len(allowed)
    allowed &= valid_msa_gids
    logger.info(
        "kNN date filter: collection_date <= %s → %d / %d dated genome ids (%d in MSA %s)",
        before_date,
        pre_msa_n,
        len(gid_to_date),
        len(allowed),
        Path(msa_path).name,
    )
    if not allowed:
        raise ValueError(
            f"no genome ids with collection_date <= {before_date}; "
            "check --metadata-csv / --msa headers / --knn-before-date"
        )
    return allowed


def align_indices_by_gid(
    mean_index: Optional[np.ndarray],
    mean_gids: Optional[list[str]],
    pca_index: Optional[np.ndarray],
    pca_gids: Optional[list[str]],
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], list[str]]:
    """Intersect mean/pca2d rows on shared genome ids (stable mean order)."""
    if mean_index is not None and mean_gids is not None and pca_index is None:
        return mean_index, None, list(mean_gids)
    if pca_index is not None and pca_gids is not None and mean_index is None:
        return None, pca_index, list(pca_gids)
    if (
        mean_index is None
        or mean_gids is None
        or pca_index is None
        or pca_gids is None
    ):
        return mean_index, pca_index, list(mean_gids or pca_gids or [])

    pca_map = {g: i for i, g in enumerate(pca_gids)}
    keep_mean: list[int] = []
    keep_pca: list[int] = []
    shared: list[str] = []
    for i, gid in enumerate(mean_gids):
        j = pca_map.get(gid)
        if j is None:
            continue
        keep_mean.append(i)
        keep_pca.append(j)
        shared.append(gid)
    if not shared:
        raise ValueError("no shared genome ids between mean and pca2d encoding indexes")
    return mean_index[np.asarray(keep_mean)], pca_index[np.asarray(keep_pca)], shared


_SQ_NORM_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _index_sq_norms(x: np.ndarray) -> np.ndarray:
    """Float64 squared row norms of ``x`` (cached per index array object)."""
    hit = _SQ_NORM_CACHE.get(id(x))
    if hit is not None and hit[0] is x:
        return hit[1]
    n = x.shape[0]
    out = np.empty(n, dtype=np.float64)
    step = max(1, int(_DIST_CHUNK_ROWS))
    for start in range(0, n, step):
        xb = np.asarray(x[start : start + step], dtype=np.float64)
        out[start : start + step] = np.einsum("ij,ij->i", xb, xb)
    if len(_SQ_NORM_CACHE) > 8:
        _SQ_NORM_CACHE.clear()
    _SQ_NORM_CACHE[id(x)] = (x, out)
    return out


def pairwise_knn_distances(
    query: np.ndarray,
    index: np.ndarray,
    *,
    metric: str = "euclidean",
) -> np.ndarray:
    """Pairwise distances from one query vector to every index row.

    ``euclidean``: L2. ``cosine``: ``1 - cos`` after L2-normalizing query and
    each index row (equivalent ranking to unit-sphere L2). Near-zero vectors
    get cosine similarity 0 (distance 1).
    """
    if metric not in KNN_METRICS:
        raise ValueError(f"metric must be one of {KNN_METRICS}; got {metric!r}")
    q = np.asarray(query, dtype=np.float64).reshape(-1)
    x = np.asarray(index)
    if x.ndim != 2:
        raise ValueError(f"index must be 2D, got shape {x.shape}")
    if q.shape[0] != x.shape[1]:
        raise ValueError(
            f"query dim {q.shape[0]} != index dim {x.shape[1]}"
        )
    eps = 1e-12
    q_norm = float(np.linalg.norm(q))
    q_u = np.zeros_like(q) if q_norm < eps else q / q_norm
    q_sq = float(q @ q)

    # Process the index in float64 chunks: identical values to a full float64
    # copy, but bounded memory (matters when many worker processes run at once).
    n = x.shape[0]
    out = np.empty(n, dtype=np.float64)
    step = max(1, int(_DIST_CHUNK_ROWS))
    buf = np.empty((min(step, n), x.shape[1]), dtype=np.float64)
    sq_norms = _index_sq_norms(x) if metric == "euclidean" else None
    for start in range(0, n, step):
        xs = x[start : start + step]
        xb = buf[: xs.shape[0]]
        np.copyto(xb, xs)  # reuse one float64 buffer (no per-chunk allocation)
        if metric == "euclidean":
            # ||x-q||^2 = ||x||^2 - 2 x.q + ||q||^2 with cached float64 row norms
            # (one gemv pass instead of subtract+norm passes). Rows where the
            # expansion is cancellation-prone (near-identical vectors) are
            # recomputed exactly.
            xn2 = sq_norms[start : start + step]
            d2 = xn2 - 2.0 * (xb @ q) + q_sq
            tol = 1e-9 * (xn2 + q_sq)
            bad = d2 <= tol
            d2 = np.maximum(d2, 0.0)
            res = np.sqrt(d2)
            if bad.any():
                res[bad] = np.linalg.norm(xb[bad] - q.reshape(1, -1), axis=1)
            out[start : start + step] = res
            continue
        x_norm = np.linalg.norm(xb, axis=1)
        x_u = np.zeros_like(xb)
        ok = x_norm >= eps
        x_u[ok] = xb[ok] / x_norm[ok, None]
        cos = np.clip(x_u @ q_u, -1.0, 1.0)
        out[start : start + step] = 1.0 - cos
    return out


def pairwise_knn_distances_multi(
    queries: np.ndarray,
    index: np.ndarray,
    *,
    metric: str = "euclidean",
) -> np.ndarray:
    """Distances from ``Q`` query vectors to every index row → ``[Q, N]`` float64.

    Same definition as :func:`pairwise_knn_distances` applied per query, but the
    (expensive) float32→float64 pass over the index is shared by all queries and
    the dot products run as one GEMM per chunk.
    """
    if metric not in KNN_METRICS:
        raise ValueError(f"metric must be one of {KNN_METRICS}; got {metric!r}")
    qs = np.asarray(queries, dtype=np.float64)
    if qs.ndim == 1:
        qs = qs.reshape(1, -1)
    x = np.asarray(index)
    if x.ndim != 2 or qs.ndim != 2:
        raise ValueError(f"expected 2D queries/index, got {qs.shape} / {x.shape}")
    if qs.shape[1] != x.shape[1]:
        raise ValueError(f"query dim {qs.shape[1]} != index dim {x.shape[1]}")
    eps = 1e-12
    n, nq = x.shape[0], qs.shape[0]
    out = np.empty((nq, n), dtype=np.float64)
    step = max(1, int(_DIST_CHUNK_ROWS))
    buf = np.empty((min(step, n), x.shape[1]), dtype=np.float64)
    if metric == "euclidean":
        sq_norms = _index_sq_norms(x)
        q_sq = np.einsum("ij,ij->i", qs, qs)
    else:
        q_norm = np.linalg.norm(qs, axis=1)
        q_u = np.zeros_like(qs)
        okq = q_norm >= eps
        q_u[okq] = qs[okq] / q_norm[okq, None]
    for start in range(0, n, step):
        xs = x[start : start + step]
        xb = buf[: xs.shape[0]]
        np.copyto(xb, xs)
        sl = slice(start, start + xs.shape[0])
        if metric == "euclidean":
            xn2 = sq_norms[sl]
            d2 = xn2[:, None] - 2.0 * (xb @ qs.T) + q_sq[None, :]
            tol = 1e-9 * (xn2[:, None] + q_sq[None, :])
            bad = d2 <= tol
            res = np.sqrt(np.maximum(d2, 0.0))
            if bad.any():
                bi, bj = np.nonzero(bad)
                res[bi, bj] = np.linalg.norm(xb[bi] - qs[bj], axis=1)
            out[:, sl] = res.T
        else:
            x_norm = np.linalg.norm(xb, axis=1)
            okx = x_norm >= eps
            cos = np.zeros((xb.shape[0], nq), dtype=np.float64)
            cos[okx] = (xb[okx] / x_norm[okx, None]) @ q_u.T
            out[:, sl] = 1.0 - np.clip(cos, -1.0, 1.0).T
    return out


def knn_indices(query: np.ndarray, index: np.ndarray, k: int) -> np.ndarray:
    d = pairwise_knn_distances(query, index, metric="euclidean")
    k = min(int(k), len(d))
    return np.argpartition(d, k - 1)[:k]


def knn_indices_weighted(
    *,
    knn_space: str,
    knn_k: int,
    pred_mean: Optional[np.ndarray],
    enc_mean: Optional[np.ndarray],
    pred_pca2d: Optional[np.ndarray],
    enc_pca2d: Optional[np.ndarray],
    mean_weight: float = 1.0,
    pca2d_weight: float = 0.75,
    knn_metric: str = "euclidean",
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (idxs sorted by distance, combined_dists, mean_dists, pca_dists)."""
    if knn_space not in KNN_SPACES:
        raise ValueError(f"knn_space must be one of {KNN_SPACES}; got {knn_space!r}")
    if knn_metric not in KNN_METRICS:
        raise ValueError(f"knn_metric must be one of {KNN_METRICS}; got {knn_metric!r}")

    n = None
    mean_d = None
    pca_d = None

    if knn_space in ("mean", "combined"):
        if pred_mean is None or enc_mean is None:
            raise ValueError(f"knn_space={knn_space} requires pred_mean and enc_mean")
        mean_d = pairwise_knn_distances(pred_mean, enc_mean, metric=knn_metric)
        n = len(mean_d)

    if knn_space in ("pca2d", "combined"):
        if pred_pca2d is None or enc_pca2d is None:
            raise ValueError(f"knn_space={knn_space} requires pred_pca2d and enc_pca2d")
        pca_d = pairwise_knn_distances(pred_pca2d, enc_pca2d, metric=knn_metric)
        if n is None:
            n = len(pca_d)
        elif len(pca_d) != n:
            raise ValueError(
                f"mean/pca2d index length mismatch: {n} vs {len(pca_d)}"
            )

    assert n is not None
    if knn_space == "mean":
        assert mean_d is not None
        combined = mean_d
    elif knn_space == "pca2d":
        assert pca_d is not None
        combined = pca_d
    else:
        assert mean_d is not None and pca_d is not None
        combined = float(mean_weight) * mean_d + float(pca2d_weight) * pca_d

    k = min(max(int(knn_k), 0), n)
    if k == 0:
        empty = np.asarray([], dtype=np.int64)
        return empty, empty.astype(np.float64), None, None
    idxs = np.argpartition(combined, k - 1)[:k]
    order = np.argsort(combined[idxs])
    idxs = idxs[order]
    return idxs, combined[idxs], None if mean_d is None else mean_d[idxs], (
        None if pca_d is None else pca_d[idxs]
    )


_MEAN_SUFFIXES = ("pred_mean_raw", "pred_mean_abs", "pred_mean")
_PCA_SUFFIXES = (
    "pred_pca2d_flat_raw",
    "pred_pca2d_sites_raw",
    "pred_pca2d_abs",
    "pred_pca2d",
)


def _pick_pred_row(
    z: Any,
    row: int,
    keys: list[str],
) -> tuple[Optional[np.ndarray], Optional[str]]:
    for k in keys:
        if k not in z.files:
            continue
        arr = np.asarray(z[k])
        if arr.ndim == 0:
            continue
        if arr.ndim == 1:
            return arr.astype(np.float32), k
        if int(row) < 0 or int(row) >= int(arr.shape[0]):
            raise IndexError(
                f"row {row} out of range for {k} with shape {arr.shape}"
            )
        # sites raw: [N, L, k] → flatten last dims
        row_arr = arr[int(row)]
        return np.asarray(row_arr, dtype=np.float32).reshape(-1), k
    return None, None


def _pred_key(split: Optional[str], suffix: str) -> str:
    if split:
        return f"{split}_{suffix}"
    return suffix


def _discover_pred_splits(z: Any) -> list[str]:
    """Return split prefixes present in NPZ ('' means unprefixed keys)."""
    found: set[str] = set()
    for k in z.files:
        for suffix in _MEAN_SUFFIXES + _PCA_SUFFIXES:
            if k == suffix:
                found.add("")
            elif k.endswith("_" + suffix):
                found.add(k[: -(len(suffix) + 1)])
    # Prefer common eval splits first when auto-picking.
    order = ["test", "val", "cal", "train", ""]
    ranked = [s for s in order if s in found]
    ranked += sorted(s for s in found if s not in order)
    return ranked


def _resolve_row_via_source_gid(
    z: Any,
    split: str,
    source_gid: Optional[str],
    row: int,
) -> int:
    """If NPZ has ``{split}_source_gids``, map pairs source id → row when possible."""
    if not source_gid:
        return int(row)
    key = _pred_key(split or None, "source_gids")
    if key not in z.files:
        return int(row)
    gids = [gid_str(g) for g in np.asarray(z[key]).tolist()]
    gid = gid_str(source_gid)
    if not gid:
        return int(row)
    matches = [i for i, g in enumerate(gids) if g == gid or g.startswith(gid) or gid.startswith(g)]
    if len(matches) == 1:
        resolved = int(matches[0])
        if resolved != int(row):
            logger.info(
                "resolved pred row via %s: pairs row=%d → npz row=%d (gid=%s)",
                key,
                int(row),
                resolved,
                gid,
            )
        return resolved
    return int(row)


class PreloadedPredictions:
    """In-memory subset of a prediction NPZ (npz-like: ``.files`` and ``[key]``).

    Avoids re-reading/decompressing the NPZ arrays for every row when many rows
    are analysed in one process. Only keys of ``pred_split`` are kept when that
    split exists in the file.
    """

    def __init__(self, path: Path, pred_split: Optional[str] = None) -> None:
        z = np.load(path, allow_pickle=True)
        try:
            available = _discover_pred_splits(z)
            only = None
            if pred_split is not None and str(pred_split) in available:
                only = str(pred_split)
            data: dict[str, np.ndarray] = {}
            for k in z.files:
                for suffix in _MEAN_SUFFIXES + _PCA_SUFFIXES + ("source_gids",):
                    if k == suffix:
                        split = ""
                    elif k.endswith("_" + suffix):
                        split = k[: -(len(suffix) + 1)]
                    else:
                        continue
                    if only is None or split == only:
                        data[k] = np.asarray(z[k])
                    break
        finally:
            z.close()
        self._data = data

    @property
    def files(self) -> list[str]:
        return list(self._data.keys())

    def __getitem__(self, key: str) -> np.ndarray:
        return self._data[key]


def load_predictions(
    pred_npz: Path,
    *,
    row: int,
    pred_split: Optional[str] = None,
    source_gid: Optional[str] = None,
    npz: Any = None,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[int]]:
    """Load absolute/raw mean and pca2d prediction vectors for one pair row.

    Returns ``(pred_mean, pred_pca2d, resolved_row)``. If ``pred_split`` is
    missing from the NPZ (e.g. ``val`` requested but only ``test_`` / ``cal_``
    keys exist), auto-selects an available split and logs a warning. Optionally
    remaps ``row`` via ``{split}_source_gids``.
    """
    z = npz if npz is not None else np.load(pred_npz, allow_pickle=True)
    available = _discover_pred_splits(z)
    if not available:
        logger.warning(
            "no pred_mean*/pred_pca2d* keys in %s; keys=%s",
            pred_npz,
            sorted(z.files),
        )
        return None, None, None

    splits_to_try: list[str] = []
    if pred_split is not None:
        splits_to_try.append(str(pred_split))
    for s in available:
        if s not in splits_to_try:
            splits_to_try.append(s)

    last_err: Optional[Exception] = None
    for split in splits_to_try:
        mean_keys = [_pred_key(split or None, s) for s in _MEAN_SUFFIXES]
        pca_keys = [_pred_key(split or None, s) for s in _PCA_SUFFIXES]
        try:
            resolved_row = _resolve_row_via_source_gid(z, split, source_gid, row)
            mean, mean_key = _pick_pred_row(z, resolved_row, mean_keys)
            pca, pca_key = _pick_pred_row(z, resolved_row, pca_keys)
        except IndexError as e:
            last_err = e
            continue
        if mean is None and pca is None:
            continue
        if pred_split is not None and split != str(pred_split):
            logger.warning(
                "pred-split %r not in %s (available=%s); using split %r instead",
                pred_split,
                pred_npz,
                [s or "<unprefixed>" for s in available],
                split or "<unprefixed>",
            )
        else:
            logger.info(
                "using pred split %r keys mean=%s pca2d=%s row=%d",
                split or "<unprefixed>",
                mean_key,
                pca_key,
                resolved_row,
            )
        return mean, pca, int(resolved_row)

    logger.warning(
        "failed to load predictions from %s (tried splits=%s); available=%s%s",
        pred_npz,
        [s or "<unprefixed>" for s in splits_to_try],
        [s or "<unprefixed>" for s in available],
        f"; last_error={last_err}" if last_err else "",
    )
    return None, None, None


def _resolve_pca2d_cache_path(
    *,
    enc_2d_dir: Path,
    basis_path: Path,
    knn_before_date: Optional[str],
    max_files: Optional[int],
    pca2d_index_cache: Optional[Path],
    no_pca2d_index_cache: bool,
) -> Optional[Path]:
    """Resolve pca2d cache path; ``none``/``off`` disables caching."""
    if no_pca2d_index_cache:
        return None
    if pca2d_index_cache is not None:
        text = str(pca2d_index_cache).strip().lower()
        if text in {"none", "off", ""}:
            return None
        return Path(pca2d_index_cache)
    return default_pca2d_index_cache_path(
        Path(enc_2d_dir),
        Path(basis_path),
        knn_before_date=knn_before_date,
        max_files=max_files,
    )


def find_ga_elite_fasta(run_dir: Path) -> Optional[Path]:
    cands = list(run_dir.rglob("*elite*.fasta")) + list(run_dir.rglob("*best*.fasta"))
    if cands:
        return sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    finals = list(run_dir.rglob("*.fasta"))
    return sorted(finals, key=lambda p: p.stat().st_mtime, reverse=True)[0] if finals else None


def read_first_fasta_seq(path: Path) -> str:
    seq_parts: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_parts:
                    break
                continue
            seq_parts.append(line)
    return "".join(seq_parts)


def write_knn_fasta(
    path: Path,
    *,
    source_id: str,
    source_seq: str,
    target_id: str,
    target_seq: str,
    neighbors: list[dict[str, Any]],
    gid_to_seq: dict[str, str],
    gid_to_lineage: Optional[dict[str, str]] = None,
    gid_to_date: Optional[dict[str, str]] = None,
    source_lineage: str = "",
    target_lineage: str = "",
    gt_blosum_neighbors: Optional[list[dict[str, Any]]] = None,
    out_consensus_fasta: Path = None,
) -> None:
    src_lin = resolve_lineage(source_id, gid_to_lineage, fallback=source_lineage)
    tgt_lin = resolve_lineage(target_id, gid_to_lineage, fallback=target_lineage)
    src_date = resolve_collection_date(source_id, gid_to_date, fallback="")
    tgt_date = resolve_collection_date(target_id, gid_to_date, fallback="")   

    lines: list[str] = [
        fasta_header("source", source_id, lineage=src_lin, extras=[f"collection_date={src_date}",]),
        source_seq,
        fasta_header("target", target_id, lineage=tgt_lin, extras=[f"collection_date={tgt_date}",]),
        target_seq,
    ]
    

    knn_seqs = []
    for i, nb in enumerate(neighbors):
        gid = str(nb["gid"])
        seq = gid_to_seq.get(gid)
        if seq is None:
            continue
        lin = resolve_lineage(gid, gid_to_lineage, fallback=str(nb.get("lineage") or ""))
        collection_date = resolve_collection_date(gid, gid_to_date, fallback=str(nb.get("collection_date") or ""))
        meta = [
            f"rank={i}",
            f"distance={nb.get('distance', '')}",
            f"hamming_to_target={nb.get('hamming_to_gt', '')}",
            f"hamming_to_source={nb.get('hamming_to_source', '')}",
            f"identity_to_target={nb.get('identity_to_gt', '')}",
            f"identity_to_source={nb.get('identity_to_source', '')}",
            f"collection_date={collection_date}"
        ]
        lines.append(fasta_header(f"knn_{i}", gid, lineage=lin, extras=meta))
        lines.append(seq)
        knn_seqs.append(seq)
    
    knn_consensus_seq = build_knn_consensus(knn_seqs)
    consensus_stats = compute_mutation_site_stats(
        source_seq,
        target_seq,
        [knn_consensus_seq] if knn_consensus_seq else [],
        label="knn_consensus",
    )
    consensus_confusion = consensus_stats.get("confusion_change_from_source", {})
    consensus_extras = [
        f"tp={consensus_confusion.get('tp', 0)}",
        f"fn={consensus_confusion.get('fn', 0)}",
        f"tn={consensus_confusion.get('tn', 0)}",
        f"fp={consensus_confusion.get('fp', 0)}",
    ]
    lines.append(
        fasta_header("knn_consensus", "", extras=consensus_extras)
        + "\n"
        + f"{knn_consensus_seq}\n"
    )
    # Consensus-only FASTA output
    if out_consensus_fasta:
        out_consensus_fasta.parent.mkdir(parents=True, exist_ok=True)
        with open(out_consensus_fasta, "w", encoding="utf-8") as f:
            f.write(
                fasta_header("knn_consensus", "", extras=consensus_extras)
                + "\n"
                + f"{knn_consensus_seq}\n"
            )
        logger.info("Wrote kNN consensus sequence to %s", out_consensus_fasta)


    for nb in gt_blosum_neighbors or []:
        gid = str(nb["gid"])
        seq = gid_to_seq.get(gid) or nb.get("seq")
        if seq is None:
            continue
        lin = resolve_lineage(gid, gid_to_lineage, fallback=str(nb.get("lineage") or ""))
        meta = [
            f"rank={nb.get('rank', '')}",
            f"blosum_to_target={nb.get('blosum_to_gt', '')}",
            f"hamming_to_target={nb.get('hamming_to_gt', '')}",
            f"hamming_to_source={nb.get('hamming_to_source', '')}",
        ]
        lines.append(fasta_header("gt_blosum", gid, lineage=lin, extras=meta))
        lines.append(str(seq))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


_BLOSUM_BATCH_CACHE: Optional[tuple[float, np.ndarray, dict[str, int]]] = None


def _blosum_matrix_and_map(gap_penalty: float = -5.0) -> tuple[np.ndarray, dict[str, int]]:
    global _BLOSUM_BATCH_CACHE
    if _BLOSUM_BATCH_CACHE is None or float(_BLOSUM_BATCH_CACHE[0]) != float(gap_penalty):
        from recomb2027.data.seq_pairwise_distance import _build_matrix_with_gap

        matrix, aa_to_idx = _build_matrix_with_gap(gap_penalty)
        _BLOSUM_BATCH_CACHE = (float(gap_penalty), matrix, aa_to_idx)
    return _BLOSUM_BATCH_CACHE[1], _BLOSUM_BATCH_CACHE[2]


def blosum_distances_to_query(
    query: str,
    sequences: list[str],
    *,
    gap_penalty: float = -5.0,
    chunk_size: int = 2048,
) -> np.ndarray:
    """Vectorised BLOSUM dissimilarity from ``query`` to each sequence.

    Uses the same ``1 - cross / max(self_q, self_s)`` definition as
    ``fast_pairwise_blosum_distance``. Equal-length (MSA) sequences are
    scored without pairwise alignment; unequal lengths fall back to the
    slow pairwise path.
    """
    from recomb2027.data.seq_pairwise_distance import fast_pairwise_blosum_distance

    n = len(sequences)
    out = np.full(n, np.inf, dtype=np.float64)
    if n == 0:
        return out
    matrix, aa_to_idx = _blosum_matrix_and_map(gap_penalty)
    unk = int(aa_to_idx["-"])
    q_len = len(query)
    equal = [i for i, s in enumerate(sequences) if len(s) == q_len]
    unequal = [i for i, s in enumerate(sequences) if len(s) != q_len]
    if equal:
        q_idx = np.fromiter((aa_to_idx.get(a, unk) for a in query), dtype=np.int32)
        self_q = float(matrix[q_idx, q_idx].sum())
        cs = max(1, int(chunk_size))
        for start in range(0, len(equal), cs):
            block = equal[start : start + cs]
            rows = np.empty((len(block), q_len), dtype=np.int32)
            for row_i, seq_i in enumerate(block):
                rows[row_i] = np.fromiter(
                    (aa_to_idx.get(a, unk) for a in sequences[seq_i]), dtype=np.int32
                )
            observed = matrix[q_idx[None, :], rows].sum(axis=1)
            self_s = matrix[rows, rows].sum(axis=1)
            max_score = np.maximum(self_q, self_s)
            dists = np.where(max_score > 0, 1.0 - observed / max_score, 0.0)
            for row_i, seq_i in enumerate(block):
                out[seq_i] = float(dists[row_i])
    for seq_i in unequal:
        out[seq_i] = float(
            fast_pairwise_blosum_distance(query, sequences[seq_i], gap_penalty=gap_penalty)
        )
    return out


def find_blosum_nearest_to_target(
    *,
    target_seq: str,
    source_seq: str = "",
    gid_to_seq: dict[str, str],
    allowed_gids: Optional[set[str]] = None,
    k: int = 5,
    exclude_gids: Optional[set[str]] = None,
    gid_to_lineage: Optional[dict[str, str]] = None,
) -> list[dict[str, Any]]:
    """Return ``k`` MSA sequences closest to GT target by BLOSUM distance.

    Candidates are date-filtered via ``allowed_gids`` (same as latent kNN).
    Target / excluded genome ids are dropped from the candidate pool.
    """
    if int(k) <= 0:
        return []
    exclude = {gid_str(g) for g in (exclude_gids or set()) if gid_str(g)}
    cand_gids: list[str] = []
    cand_seqs: list[str] = []
    for gid, seq in gid_to_seq.items():
        g = gid_str(gid)
        if not g or g in exclude:
            continue
        if allowed_gids is not None and g not in allowed_gids:
            continue
        if not seq:
            continue
        cand_gids.append(g)
        cand_seqs.append(str(seq))
    if not cand_gids:
        logger.warning("gt-blosum search: no candidates after date/exclude filter")
        return []
    dists = blosum_distances_to_query(target_seq, cand_seqs)
    order = np.argsort(dists, kind="mergesort")
    take = min(int(k), len(order))
    neighbors: list[dict[str, Any]] = []
    for rank, j in enumerate(order[:take]):
        gid = cand_gids[int(j)]
        seq = cand_seqs[int(j)]
        neighbors.append(
            {
                "gid": gid,
                "lineage": resolve_lineage(gid, gid_to_lineage) or None,
                "rank": int(rank),
                "blosum_to_gt": float(dists[int(j)]),
                "hamming_to_gt": hamming(seq, target_seq) if target_seq else None,
                "hamming_to_source": hamming(seq, source_seq) if source_seq else None,
                "identity_to_gt": identity(seq, target_seq) if target_seq else None,
                "identity_to_source": identity(seq, source_seq) if source_seq else None,
                "seq": seq,
            }
        )
    return neighbors


def _safe_div(num: float, den: float) -> Optional[float]:
    if den <= 0:
        return None
    return float(num) / float(den)


def compute_mutation_site_stats(
    source_seq: str,
    target_seq: str,
    knn_seqs: list[str],
    *,
    label: str = "knn",
) -> dict[str, Any]:
    """Site-level coverage of source→target mutations by a neighbor pool.

    Definitions (aligned MSA columns, 0-based)::

      R  = sites where source ≠ target          (required transitions)
      V  = sites variable in {source} ∪ knn     (≥2 distinct residues)
      C  = sites where ≥1 knn differs from source
      A  = sites in R where ≥1 knn has target allele

    Confusion (change-from-source)::

      TP = |R ∩ C|   needed site and knn varies from source
      FN = |R − C|   needed site but all knn match source
      FP = |C − R|   knn varies where source==target
      TN = rest      source==target and all knn match source

    Allele-aware confusion uses A instead of C ∩ R for positives.
    """
    n = min(len(source_seq), len(target_seq))
    if len(source_seq) != len(target_seq):
        logger.warning(
            "mutation stats (%s): source/target length mismatch %d vs %d; "
            "using first %d columns",
            label,
            len(source_seq),
            len(target_seq),
            n,
        )
    if n == 0:
        return {
            "label": label,
            "n_sites": 0,
            "n_knn_seqs": len(knn_seqs),
            "n_required_source_to_target": 0,
            "n_variable_knn_plus_source": 0,
            "n_variable_overlap_required": 0,
            "n_required_invariant_in_knn_pool": 0,
            "n_required_recovered_target_allele": 0,
            "n_required_missing_target_allele": 0,
            "n_required_wrong_allele_only": 0,
            "confusion_change_from_source": {},
            "confusion_target_allele": {},
        }

    src = source_seq[:n]
    tgt = target_seq[:n]
    pool: list[str] = []
    n_skipped_short = 0
    for s in knn_seqs:
        if not s:
            continue
        if len(s) < n:
            n_skipped_short += 1
            continue
        pool.append(s[:n])
    if n_skipped_short:
        logger.warning(
            "mutation stats (%s): skipped %d / %d knn sequences shorter than "
            "source/target length %d",
            label,
            n_skipped_short,
            len(knn_seqs),
            n,
        )

    required: list[int] = []
    for i in range(n):
        if src[i] != tgt[i]:
            # use indexing from 1
            required.append(i+1)
    required_set = set(required)

    variable: list[int] = []
    change_from_source: list[int] = []
    recovered: list[int] = []
    wrong_allele_only: list[int] = []

    for i in range(n):
        alleles = {src[i]}
        differs = False
        has_target = False
        for seq in pool:
            aa = seq[i]
            alleles.add(aa)
            if aa != src[i]:
                differs = True
            if aa == tgt[i]:
                has_target = True
        if len(alleles) >= 2:
            # reindex to start from 1 in the report
            variable.append(i+1)
        if differs:
            # reindex to start from 1 in the report
            change_from_source.append(i+1)
        if i+1 in required_set:
            if has_target:
                recovered.append(i+1)
            elif differs:
                wrong_allele_only.append(i+1)

    variable_set = set(variable)
    change_set = set(change_from_source)
    recovered_set = set(recovered)

    tp = len(required_set & change_set)
    fn = len(required_set - change_set)
    fp = len(change_set - required_set)
    tn = n - tp - fn - fp

    tp_a = len(recovered_set)
    fn_a = len(required_set - recovered_set)
    # FP_a: sites where knn introduces non-source allele but source==target
    # (any change is spurious since target allele == source allele)
    fp_a = fp
    tn_a = n - tp_a - fn_a - fp_a

    def _confusion(tp_: int, fp_: int, tn_: int, fn_: int) -> dict[str, Any]:
        prec = _safe_div(tp_, tp_ + fp_)
        rec = _safe_div(tp_, tp_ + fn_)
        f1 = None
        if prec is not None and rec is not None and (prec + rec) > 0:
            f1 = 2.0 * prec * rec / (prec + rec)
        return {
            "tp": int(tp_),
            "fp": int(fp_),
            "tn": int(tn_),
            "fn": int(fn_),
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "specificity": _safe_div(tn_, tn_ + fp_),
        }

    return {
        "label": label,
        "n_sites": int(n),
        "n_knn_seqs": int(len(pool)),
        "n_required_source_to_target": int(len(required_set)),
        "n_variable_knn_plus_source": int(len(variable_set)),
        "n_variable_overlap_required": int(len(variable_set & required_set)),
        "n_required_invariant_in_knn_pool": int(len(required_set - variable_set)),
        "n_required_recovered_target_allele": int(len(recovered_set)),
        "n_required_missing_target_allele": int(len(required_set - recovered_set)),
        "n_required_wrong_allele_only": int(len(wrong_allele_only)),
        "fraction_required_recovered": _safe_div(len(recovered_set), len(required_set)),
        "fraction_required_variable_in_pool": _safe_div(
            len(variable_set & required_set), len(required_set)
        ),
        "confusion_change_from_source": _confusion(tp, fp, tn, fn),
        "confusion_target_allele": _confusion(tp_a, fp_a, tn_a, fn_a),
        # Compact site lists for debugging (1-based MSA columns).
        "required_sites": required,
        "variable_sites_knn_plus_source": variable,
        "recovered_required_sites": recovered,
        "missing_required_sites": sorted(required_set - recovered_set),
    }


def mutation_stats_for_neighbor_records(
    source_seq: str,
    target_seq: str,
    neighbors: list[dict[str, Any]],
    gid_to_seq: dict[str, str],
    *,
    label: str = "knn",
    include_site_lists: bool = True,
) -> dict[str, Any]:
    """Resolve neighbor sequences and compute :func:`compute_mutation_site_stats`."""
    seqs: list[str] = []
    gids: list[str] = []
    seen: set[str] = set()
    for nb in neighbors:
        gid = gid_str(nb.get("gid"))
        if not gid or gid in seen:
            continue
        seq = gid_to_seq.get(gid) or nb.get("seq")
        if not seq:
            continue
        seen.add(gid)
        gids.append(gid)
        seqs.append(str(seq))
    stats = compute_mutation_site_stats(
        source_seq, target_seq, seqs, label=label
    )
    stats["neighbor_gids"] = gids
    stats["n_neighbor_records"] = int(len(neighbors))
    if neighbors and not seqs:
        logger.warning(
            "mutation stats (%s): %d neighbor records but no resolvable sequences",
            label,
            len(neighbors),
        )
    if not include_site_lists:
        for key in (
            "required_sites",
            "variable_sites_knn_plus_source",
            "recovered_required_sites",
            "missing_required_sites",
        ):
            stats.pop(key, None)
    return stats


def build_knn_consensus(knn_seqs: list[str]) -> str:
    """
    Creates a consensus sequence from kNN sequences.
    Variable positions (where not all kNNs share the same amino acid/nucleotide) 
    are substituted with the 'X' symbol.
    """
    if not knn_seqs:
        return ""
    
    seq_len = len(knn_seqs[0])
    consensus = []
    
    for i in range(seq_len):
        # Extract unique characters at position i across all kNN sequences
        unique_chars = {seq[i] for seq in knn_seqs}
        if len(unique_chars) == 1:
            consensus.append(unique_chars.pop())
        else:
            consensus.append('X')
            
    return "".join(consensus)

def compute_mutation_stats(
    source_seq: str, 
    target_seq: str, 
    neighbors: list[str], 
    gid_to_seq: dict[str, str]
) -> dict:
    """
    Computes true positive / false negative positions and alleles between 
    the source sequence, target sequence, and the retrieved kNN sequences.
    """
    tp_pos = []
    fn_pos = []
    tp_alleles = []
    fn_alleles = []

    knn_seqs = []
    for nb in neighbors:
        gid = gid_str(nb.get("gid"))
        seq = gid_to_seq.get(gid) or nb.get("seq")
        if not seq:
            continue
        knn_seqs.append(str(seq))

    # Iterate through the sequence (assuming pre-aligned sequences of equal length)
    for i in range(min(len(source_seq), len(target_seq))):
        source_char = source_seq[i]
        target_char = target_seq[i]
        
        # Check if mutation is needed to transition from source to target
        if source_char != target_char:
            pos_1_based = i + 1 
            allele_str = f"{source_char}{pos_1_based}{target_char}"
            
            # --- POSITIONS ---
            # TP Pos: Needs mutation AND is mutated (away from source) in at least one kNN
            if any(knn[i] != source_char for knn in knn_seqs):
                tp_pos.append(pos_1_based)
            else:
                # FN Pos: Needs mutation BUT is never mutated in any kNN
                fn_pos.append(pos_1_based)
                
            # --- ALLELES (Concrete Mutations) ---
            # TP Allele: Mutated to the EXACT target allele in at least one kNN
            if any(knn[i] == target_char for knn in knn_seqs):
                tp_alleles.append(allele_str)
            else:
                # FN Allele: Never mutated to the exact target allele in any kNN
                fn_alleles.append(allele_str)

    return {
        "true_positive_positions": tp_pos,
        "false_negative_positions": fn_pos,
        "true_positive_alleles": tp_alleles,
        "false_negative_alleles": fn_alleles
    }


def analyse_row(
    *,
    source_seq: str,
    target_seq: str,
    ga_seq: Optional[str],
    pred_mean: Optional[np.ndarray] = None,
    pred_pca2d: Optional[np.ndarray] = None,
    enc_index: Optional[np.ndarray] = None,
    enc_pca2d: Optional[np.ndarray] = None,
    enc_gids: Optional[list[str]] = None,
    gid_to_seq: Optional[dict[str, str]] = None,
    gid_to_lineage: Optional[dict[str, str]] = None,
    gid_to_date: Optional[dict[str, str]] = None,
    knn_k: int = 5,
    knn_space: str = "combined",
    knn_metric: str = "euclidean",
    knn_mean_weight: float = 1.0,
    knn_pca2d_weight: float = 0.75,
) -> dict[str, Any]:
    """Score GA / kNN neighbors vs target and source sequences.

    ``enc_index`` is the mean-embedding index (kept for backward compatibility).
    ``knn_metric`` is ``euclidean`` (L2) or ``cosine`` (``1 - cos`` after
    unit-norm; same ranking as unit-sphere L2).
    """
    out: dict[str, Any] = {
        "gt_hamming_source_target": hamming(source_seq, target_seq),
        "gt_identity_source_target": identity(source_seq, target_seq),
        "knn_target_sequence_found": False,
        "knn_target_sequence_gids": [],
        "knn_space": knn_space,
        "knn_metric": knn_metric,
        "knn_mean_weight": float(knn_mean_weight),
        "knn_pca2d_weight": float(knn_pca2d_weight),
    }
    if ga_seq:
        out["ga_hamming_to_gt"] = hamming(ga_seq, target_seq)
        out["ga_identity_to_gt"] = identity(ga_seq, target_seq)
        out["ga_hamming_to_source"] = hamming(ga_seq, source_seq)
        out["ga_identity_to_source"] = identity(ga_seq, source_seq)
        out["ga_edits_vs_gt_edits"] = float(
            out["ga_hamming_to_gt"] / max(out["gt_hamming_source_target"], 1)
        )

    need_mean = knn_space in ("mean", "combined")
    need_pca = knn_space in ("pca2d", "combined")
    have_index = enc_gids is not None and gid_to_seq is not None
    have_mean = (not need_mean) or (pred_mean is not None and enc_index is not None)
    have_pca = (not need_pca) or (pred_pca2d is not None and enc_pca2d is not None)

    if have_index and have_mean and have_pca:
        idxs, dists, mean_dists, pca_dists = knn_indices_weighted(
            knn_space=knn_space,
            knn_k=knn_k,
            pred_mean=pred_mean,
            enc_mean=enc_index,
            pred_pca2d=pred_pca2d,
            enc_pca2d=enc_pca2d,
            mean_weight=knn_mean_weight,
            pca2d_weight=knn_pca2d_weight,
            knn_metric=knn_metric,
        )
        neighbors: list[dict[str, Any]] = []
        best_ham_gt = None
        best_ham_src = None
        best_seq = None
        for rank, j in enumerate(idxs):
            gid = enc_gids[int(j)]
            seq = gid_to_seq.get(gid)
            if seq is None:
                continue
            h_gt = hamming(seq, target_seq) if target_seq else None
            h_src = hamming(seq, source_seq) if source_seq else None
            nb: dict[str, Any] = {
                "gid": gid,
                "lineage": resolve_lineage(gid, gid_to_lineage) or None,
                "collection_date": resolve_collection_date(gid, gid_to_date) or None,
                "rank": int(rank),
                "distance": float(dists[rank]),
                "hamming_to_gt": h_gt,
                "identity_to_gt": (
                    identity(seq, target_seq) if target_seq else None
                ),
                "hamming_to_source": h_src,
                "identity_to_source": (
                    identity(seq, source_seq) if source_seq else None
                ),
            }
            if mean_dists is not None:
                nb["distance_mean"] = float(mean_dists[rank])
            if pca_dists is not None:
                nb["distance_pca2d"] = float(pca_dists[rank])
            neighbors.append(nb)
            if h_gt is not None and (best_ham_gt is None or h_gt < best_ham_gt):
                best_ham_gt = h_gt
                best_seq = seq
            if h_src is not None and (best_ham_src is None or h_src < best_ham_src):
                best_ham_src = h_src

        out["knn_neighbors"] = neighbors
        target_sequence_gids = [
            str(nb["gid"])
            for nb in neighbors
            if gid_to_seq.get(str(nb["gid"])) == target_seq
        ]
        out["knn_target_sequence_found"] = bool(target_sequence_gids)
        out["knn_target_sequence_gids"] = target_sequence_gids
        out["knn_best_hamming_to_gt"] = best_ham_gt
        out["knn_best_hamming_to_source"] = best_ham_src
        if neighbors:
            # best identity among neighbors (max)
            id_gt = [n["identity_to_gt"] for n in neighbors if n["identity_to_gt"] is not None]
            id_src = [
                n["identity_to_source"]
                for n in neighbors
                if n["identity_to_source"] is not None
            ]
            if id_gt:
                out["knn_best_identity_to_gt"] = float(max(id_gt))
            if id_src:
                out["knn_best_identity_to_source"] = float(max(id_src))
        if ga_seq and best_ham_gt is not None:
            out["ga_better_than_knn"] = bool(
                out.get("ga_hamming_to_gt", 10**9) < best_ham_gt
            )
            out["knn_better_than_ga"] = bool(
                best_ham_gt < out.get("ga_hamming_to_gt", 10**9)
            )
        if ga_seq and best_seq is not None:
            out["ga_hamming_to_best_knn"] = hamming(ga_seq, best_seq)
            out["ga_identity_to_best_knn"] = identity(ga_seq, best_seq)
    return out


# ---------------------------------------------------------------------------
# CLI: shared (once-per-process) context + per-row analysis
# ---------------------------------------------------------------------------

_ROW_PLACEHOLDERS = ("{row}", "{}")
_LOG_ROW: Optional[int] = None  # row currently processed (log prefix in batch mode)


class _RowLogFilter(logging.Filter):
    """Prefix log lines with ``[row N]`` while a row is being processed."""

    def filter(self, record: logging.LogRecord) -> bool:
        if _LOG_ROW is not None and not getattr(record, "_row_prefixed", False):
            record.msg = f"[row {_LOG_ROW}] {record.getMessage()}"
            record.args = None
            record._row_prefixed = True
        return True


def _install_row_log_filter() -> None:
    for h in logging.getLogger().handlers:
        h.addFilter(_RowLogFilter())


def _has_row_placeholder(path: Optional[Path]) -> bool:
    return path is not None and any(ph in str(path) for ph in _ROW_PLACEHOLDERS)


def _fmt_row_path(path: Optional[Path], row: int) -> Optional[Path]:
    """Substitute ``{row}`` / ``{}`` in an output path with the pairs row index."""
    if path is None:
        return None
    text = str(path)
    for ph in _ROW_PLACEHOLDERS:
        text = text.replace(ph, str(int(row)))
    return Path(text)


def parse_row_spec(spec: str, n_rows: int) -> list[int]:
    """Parse ``"0-10,15,20-"`` into sorted unique row indices (inclusive ranges)."""
    rows: set[int] = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo = int(lo_s) if lo_s.strip() else 0
            hi = int(hi_s) if hi_s.strip() else n_rows - 1
            rows.update(range(lo, hi + 1))
        else:
            rows.add(int(part))
    bad = [r for r in rows if r < 0 or r >= n_rows]
    if bad:
        raise SystemExit(
            f"--rows out of range for pairs CSV with {n_rows} rows: {sorted(bad)[:5]}"
        )
    return sorted(rows)


class _SharedContext:
    """Heavy, read-only data loaded once and shared by all rows (and workers)."""

    df: pd.DataFrame
    metadata_csv: Optional[Path]
    gid_to_lineage: dict[str, str]
    gid_to_date: dict[str, str]
    ga_seq: Optional[str]
    max_index_files: Optional[int]
    knn_before: Optional[str]
    allowed_gids: Optional[set[str]]
    gid_to_seq: Optional[dict[str, str]]
    enc_mean: Optional[np.ndarray]
    enc_pca2d: Optional[np.ndarray]
    enc_gids: Optional[list[str]]
    pred_store: Optional[PreloadedPredictions]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Analyse GT and kNN-to-B3-prediction sequences."
    )
    p.add_argument("--pairs-csv", type=Path, required=True)
    p.add_argument("--row", type=int, default=0)
    p.add_argument(
        "--all-rows",
        action="store_true",
        help="Analyse every row of --pairs-csv in one process: the MSA, encoding "
        "indexes, metadata and predictions are loaded once and shared by all "
        "workers. --out-* paths must contain '{row}' (or '{}'), replaced by the "
        "pairs row index (same naming as one run per --row).",
    )
    p.add_argument(
        "--rows",
        default=None,
        help="Batch mode on a subset of pairs rows, e.g. '0-99,150,200-' "
        "(inclusive ranges). Implies the batch behaviour of --all-rows.",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Worker processes for --all-rows/--rows (fork; index data shared "
        "copy-on-write). Default 1 (sequential).",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Batch mode: skip rows whose .md report already exists.",
    )
    p.add_argument("--ga-run-dir", type=Path, default=None)
    p.add_argument("--ga-fasta", type=Path, default=None)
    p.add_argument("--pred-npz", type=Path, default=None, help="B3 infer or packed NPZ")
    p.add_argument("--pred-split", default=None)
    p.add_argument(
        "--encodings-dir",
        type=Path,
        default=None,
        help="Mean (average-token) encoding .pt directory",
    )
    p.add_argument(
        "--enc-2d-dir",
        type=Path,
        default=None,
        help="Per-residue 2D encoding .pt directory (for pca2d knn)",
    )
    p.add_argument(
        "--pca2d-basis-path",
        type=Path,
        default=None,
        help="Feature-PCA basis .pt used to project 2D encodings to pca2d flats",
    )
    p.add_argument("--msa", default=None)
    p.add_argument("--knn-k", type=int, default=5)
    p.add_argument(
        "--knn-space",
        choices=KNN_SPACES,
        default="combined",
        help="Latent space for kNN: mean only, pca2d only, or weighted combination",
    )
    p.add_argument(
        "--knn-metric",
        choices=KNN_METRICS,
        default="euclidean",
        help="Neighbor distance: euclidean (L2, default; MSE-aligned) or "
        "cosine (1-cos after unit-norm; better for ESM mean direction). "
        "Applied to each active space before --knn-*-weight combination.",
    )
    p.add_argument(
        "--knn-mean-weight",
        type=float,
        default=1.0,
        help="Weight on mean distance when --knn-space=combined",
    )
    p.add_argument(
        "--knn-pca2d-weight",
        type=float,
        default=0.75,
        help="Weight on pca2d distance when --knn-space=combined",
    )
    p.add_argument(
        "--max-index-files",
        type=int,
        default=5000,
        help="Cap encoding index size after date filter (sorted filenames). "
        "Use 0 for all files.",
    )
    p.add_argument(
        "--dataset-config",
        type=Path,
        default=None,
        help="Latent dataset YAML; used for default --knn-before-date "
        f"(split_threshold) and optional metadata CSV. Default cutoff "
        f"without this flag: {DEFAULT_KNN_BEFORE_DATE}.",
    )
    p.add_argument(
        "--metadata-csv",
        type=Path,
        default=None,
        help="Labels CSV (e.g. df_labels.csv) with genome_id, collection_date, "
        "lineage. Dates take priority over MSA headers for the kNN date filter; "
        "lineage is written into FASTA headers.",
    )
    p.add_argument(
        "--knn-before-date",
        default=None,
        help="Only index MSA sequences with collection_date <= DATE "
        "(inclusive; matches train split). Default: split_threshold from "
        f"--dataset-config, else {DEFAULT_KNN_BEFORE_DATE}. Pass 'none' to disable.",
    )
    p.add_argument(
        "--no-knn-date-filter",
        action="store_true",
        help="Disable collection_date filtering of the kNN index.",
    )
    p.add_argument(
        "--pca2d-index-cache",
        type=Path,
        default=None,
        help="NPZ cache for projected pca2d kNN index. Default: auto path under "
        "enc-2d-dir/_knn_pca2d_cache/. Reused across rows when genome ids match. "
        "Pass 'none' to disable.",
    )
    p.add_argument(
        "--no-pca2d-index-cache",
        action="store_true",
        help="Disable pca2d index caching.",
    )
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument(
        "--out-fasta",
        type=Path,
        default=None,
        help="FASTA with source, target, and kNN neighbor sequences "
        "(default: same stem as --out-json)",
    )
    p.add_argument(
        "--out-consensus-fasta",
        type=Path,
        default=None,
        help="Output path for consensus-only FASTA"
    )
    p.add_argument(
        "--gt-blosum-k",
        type=int,
        default=0,
        help="Also append K MSA sequences closest to the ground-truth target "
        "by BLOSUM distance (date-filtered like latent kNN). 0 disables.",
    )
    p.add_argument(
        "--out-mutation-stats",
        type=Path,
        default=None,
        help="Write source→target mutation coverage stats (vs latent kNN, "
        "and vs gt-BLOSUM neighbors when --gt-blosum-k>0) to this JSON file.",
    )
    p.add_argument("--log-level", default="INFO")
    return p


def _prepare_shared(args: argparse.Namespace, *, batch: bool) -> _SharedContext:
    """Load everything that does not depend on the row (done once per process)."""
    ctx = _SharedContext()
    ctx.df = pd.read_csv(
        args.pairs_csv,
        dtype={"genome_id_source": "string", "genome_id_target": "string"},
    )
    for col in ("genome_id_source", "genome_id_target"):
        if col in ctx.df.columns:
            ctx.df[col] = ctx.df[col].map(gid_str)

    metadata_csv = args.metadata_csv
    if metadata_csv is None and args.dataset_config is not None and Path(args.dataset_config).is_file():
        from omegaconf import OmegaConf

        dcfg = load_config(str(args.dataset_config))
        meta = dcfg.get("metadata_msa_path")
        if meta:
            metadata_csv = Path(str(meta))
    ctx.metadata_csv = metadata_csv
    ctx.gid_to_lineage = {}
    if metadata_csv is not None and Path(metadata_csv).is_file():
        ctx.gid_to_lineage = load_lineages_from_metadata(metadata_csv)
    ctx.gid_to_date = {}
    if metadata_csv is not None and Path(metadata_csv).is_file():
        ctx.gid_to_date = load_dates_from_metadata(metadata_csv)

    ctx.ga_seq = None
    fasta = args.ga_fasta
    if fasta is None and args.ga_run_dir is not None:
        fasta = find_ga_elite_fasta(args.ga_run_dir)
    if fasta is not None and Path(fasta).is_file():
        ctx.ga_seq = read_first_fasta_seq(Path(fasta))
        logger.info("GA sequence from %s (len=%d)", fasta, len(ctx.ga_seq))

    if int(args.max_index_files) <= 0:
        ctx.max_index_files = None
        logger.info("encoding index: using all files under encodings dirs")
    else:
        ctx.max_index_files = int(args.max_index_files)

    ctx.pred_store = None
    if batch and args.pred_npz is not None:
        ctx.pred_store = PreloadedPredictions(
            Path(args.pred_npz), pred_split=args.pred_split
        )
        logger.info(
            "preloaded predictions from %s (keys=%s)",
            args.pred_npz,
            sorted(ctx.pred_store.files),
        )

    ctx.enc_mean = ctx.enc_pca2d = ctx.enc_gids = ctx.gid_to_seq = None
    ctx.allowed_gids = None
    need_mean = args.knn_space in ("mean", "combined")
    need_pca = args.knn_space in ("pca2d", "combined")
    need_latent_knn = (
        (need_mean and args.encodings_dir is not None)
        or (need_pca and args.enc_2d_dir is not None and args.pca2d_basis_path is not None)
    )
    need_msa = bool(
        args.msa is not None
        and (
            need_latent_knn
            or int(args.gt_blosum_k) > 0
            or args.out_mutation_stats is not None
        )
    )
    ctx.knn_before = resolve_knn_before_date(
        knn_before_date=args.knn_before_date,
        dataset_config=args.dataset_config,
        no_knn_date_filter=bool(args.no_knn_date_filter),
    )

    if need_msa:
        from recomb2027.latent.data_series.latent_decoder_io import parse_msa_gids

        ctx.gid_to_seq = parse_msa_gids(args.msa)
        msa_gid_set = set(ctx.gid_to_seq.keys())
        ctx.allowed_gids = build_knn_allowed_gids(
            args.msa,
            before_date=ctx.knn_before,
            metadata_csv=metadata_csv,
            msa_gids=msa_gid_set,
        )
        if ctx.knn_before is None:
            logger.info(
                "kNN date filter: disabled (restricting index to %d MSA genome ids)",
                len(msa_gid_set),
            )
        else:
            logger.info("kNN date filter: collection_date <= %s", ctx.knn_before)

        index_allowed_gids = (
            ctx.allowed_gids if ctx.allowed_gids is not None else msa_gid_set
        )
        mean_gids = None
        pca_gids = None
        enc_mean = enc_pca2d = None

        if need_latent_knn and need_mean:
            if args.encodings_dir is None:
                raise SystemExit("--encodings-dir required for knn-space mean/combined")
            enc_mean, mean_gids = load_encoding_index(
                Path(args.encodings_dir),
                max_files=ctx.max_index_files,
                allowed_gids=index_allowed_gids,
            )
            logger.info("mean index: n=%d dim=%d", enc_mean.shape[0], enc_mean.shape[1])

        if need_latent_knn and need_pca:
            if args.enc_2d_dir is None or args.pca2d_basis_path is None:
                raise SystemExit(
                    "--enc-2d-dir and --pca2d-basis-path required for knn-space pca2d/combined"
                )
            pca_cache = _resolve_pca2d_cache_path(
                enc_2d_dir=Path(args.enc_2d_dir),
                basis_path=Path(args.pca2d_basis_path),
                knn_before_date=ctx.knn_before,
                max_files=ctx.max_index_files,
                pca2d_index_cache=args.pca2d_index_cache,
                no_pca2d_index_cache=bool(args.no_pca2d_index_cache),
            )
            # Prefer projecting the same gids as the mean index when available.
            enc_pca2d, pca_gids = load_pca2d_encoding_index(
                Path(args.enc_2d_dir),
                Path(args.pca2d_basis_path),
                gids=mean_gids,
                max_files=ctx.max_index_files if mean_gids is None else None,
                allowed_gids=index_allowed_gids,
                cache_path=pca_cache,
                knn_before_date=ctx.knn_before,
            )
            logger.info(
                "pca2d index: n=%d dim=%d", enc_pca2d.shape[0], enc_pca2d.shape[1]
            )

        if need_latent_knn:
            enc_mean, enc_pca2d, ctx.enc_gids = align_indices_by_gid(
                enc_mean, mean_gids, enc_pca2d, pca_gids
            )
            logger.info(
                "aligned encoding index n=%d (space=%s)",
                len(ctx.enc_gids),
                args.knn_space,
            )
        ctx.enc_mean, ctx.enc_pca2d = enc_mean, enc_pca2d
    elif int(args.gt_blosum_k) > 0:
        raise SystemExit("--gt-blosum-k requires --msa")
    return ctx


def _process_row(args: argparse.Namespace, ctx: _SharedContext, row_idx: int) -> None:
    """Analyse one pairs row and write its JSON / FASTA / mutation-stats / MD outputs."""
    out_json = _fmt_row_path(args.out_json, row_idx)
    out_fasta = _fmt_row_path(args.out_fasta, row_idx)
    out_consensus_fasta = _fmt_row_path(args.out_consensus_fasta, row_idx)
    out_mutation_stats = _fmt_row_path(args.out_mutation_stats, row_idx)
    assert out_json is not None

    row = ctx.df.iloc[int(row_idx)]
    gid_to_lineage, gid_to_date = ctx.gid_to_lineage, ctx.gid_to_date
    gid_to_seq, allowed_gids, knn_before = ctx.gid_to_seq, ctx.allowed_gids, ctx.knn_before

    source_seq = str(row["seq_source"])
    target_seq = str(row["seq_target"])
    source_id = gid_str(row.get("genome_id_source", ""))
    target_id = gid_str(row.get("genome_id_target", ""))
    source_lineage_fb = str(row.get("lineage_source", "") or "")
    target_lineage_fb = str(row.get("lineage_target", "") or "")
    if source_lineage_fb.lower() in {"nan", "<na>", "none", "unknown"}:
        source_lineage_fb = ""
    if target_lineage_fb.lower() in {"nan", "<na>", "none", "unknown"}:
        target_lineage_fb = ""
    # Prefer series_row when present so skipped MSA rows do not desync NPZ index.
    pairs_iloc = int(row_idx)
    pred_row = pairs_iloc
    if "series_row" in row.index and pd.notna(row["series_row"]):
        pred_row = int(row["series_row"])
    elif "index" in row.index and pd.notna(row["index"]):
        pred_row = int(row["index"])

    source_lineage = resolve_lineage(
        source_id, gid_to_lineage, fallback=source_lineage_fb
    )
    target_lineage = resolve_lineage(
        target_id, gid_to_lineage, fallback=target_lineage_fb
    )
    source_date = resolve_collection_date(source_id, gid_to_date, fallback="")
    target_date = resolve_collection_date(target_id, gid_to_date, fallback="")

    pred_mean = pred_pca2d = None
    if args.pred_npz is not None:
        pred_mean, pred_pca2d, resolved_pred_row = load_predictions(
            args.pred_npz,
            row=int(pred_row),
            pred_split=args.pred_split,
            source_gid=source_id or None,
            npz=ctx.pred_store,
        )
        if resolved_pred_row is not None:
            pred_row = int(resolved_pred_row)
        logger.info(
            "predictions: mean=%s pca2d=%s (pairs_iloc=%d pred_row=%d)",
            None if pred_mean is None else tuple(pred_mean.shape),
            None if pred_pca2d is None else tuple(pred_pca2d.shape),
            pairs_iloc,
            int(pred_row),
        )

    report = {
        "row": int(pred_row),
        "pairs_iloc": pairs_iloc,
        "pred_row": int(pred_row),
        "genome_id_source": source_id,
        "genome_id_target": target_id,
        "lineage_source": source_lineage,
        "lineage_target": target_lineage,
        "date_source": source_date,
        "date_target": target_date,
        "knn_before_date": knn_before,
        "max_index_files": ctx.max_index_files,
        "metrics": analyse_row(
            source_seq=source_seq,
            target_seq=target_seq,
            ga_seq=ctx.ga_seq,
            pred_mean=pred_mean,
            pred_pca2d=pred_pca2d,
            enc_index=ctx.enc_mean,
            enc_pca2d=ctx.enc_pca2d,
            enc_gids=ctx.enc_gids,
            gid_to_seq=gid_to_seq,
            gid_to_lineage=gid_to_lineage or None,
            gid_to_date=gid_to_date or None,
            knn_k=args.knn_k,
            knn_space=args.knn_space,
            knn_metric=args.knn_metric,
            knn_mean_weight=args.knn_mean_weight,
            knn_pca2d_weight=args.knn_pca2d_weight,
        ),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", out_json)

    m = report["metrics"]
    neighbors = list(m.get("knn_neighbors") or [])
    if args.pred_npz is not None and not neighbors:
        logger.warning(
            "kNN neighbors empty for knn-space=%s "
            "(check pred keys / encodings / enc-2d-dir / pca2d-basis-path)",
            args.knn_space,
        )

    gt_blosum_neighbors: list[dict[str, Any]] = []
    if int(args.gt_blosum_k) > 0:
        if gid_to_seq is None:
            raise SystemExit("--gt-blosum-k requires --msa")
        logger.info(
            "searching %d BLOSUM-nearest sequences to GT target (date filter=%s)",
            int(args.gt_blosum_k),
            knn_before,
        )
        gt_blosum_neighbors = find_blosum_nearest_to_target(
            target_seq=target_seq,
            source_seq=source_seq,
            gid_to_seq=gid_to_seq,
            allowed_gids=allowed_gids,
            k=int(args.gt_blosum_k),
            exclude_gids={target_id} if target_id else set(),
            gid_to_lineage=gid_to_lineage or None,
        )
        # Drop bulky seq payloads from the JSON report.
        report["gt_blosum_neighbors"] = [
            {kk: vv for kk, vv in nb.items() if kk != "seq"}
            for nb in gt_blosum_neighbors
        ]
        report["gt_blosum_k"] = int(args.gt_blosum_k)
        logger.info("gt-blosum neighbors: %d", len(gt_blosum_neighbors))

    if out_mutation_stats is not None:
        mut_report: dict[str, Any] = {
            "genome_id_source": source_id,
            "genome_id_target": target_id,
            "lineage_source": source_lineage,
            "lineage_target": target_lineage,
            "date_source": source_date,
            "date_target": target_date,
            "gt_hamming_source_target": hamming(source_seq, target_seq),
            "knn_space": m.get("knn_space"),
            "knn_k": int(args.knn_k),
            "gt_blosum_k": int(args.gt_blosum_k),
            "latent_knn": mutation_stats_for_neighbor_records(
                source_seq,
                target_seq,
                neighbors,
                gid_to_seq or {},
                label="latent_knn",
            ),
        }
        if gt_blosum_neighbors:
            mut_report["gt_blosum"] = mutation_stats_for_neighbor_records(
                source_seq,
                target_seq,
                gt_blosum_neighbors,
                gid_to_seq or {},
                label="gt_blosum",
            )
        # positions and alleles
        mut_report["position_stat"] = compute_mutation_stats(
            source_seq,
            target_seq,
            neighbors,
            gid_to_seq or {},
        )

        out_mut = Path(out_mutation_stats)
        out_mut.parent.mkdir(parents=True, exist_ok=True)
        out_mut.write_text(json.dumps(mut_report, indent=2) + "\n", encoding="utf-8")
        report["out_mutation_stats"] = str(out_mut)
        logger.info("Wrote mutation stats %s", out_mut)

    if out_fasta is None:
        out_fasta = out_json.with_suffix(".fasta")
    # Always write source/target; neighbors included when available.
    if gid_to_seq is not None or source_seq:
        write_knn_fasta(
            out_fasta,
            source_id=source_id,
            source_seq=source_seq,
            target_id=target_id,
            target_seq=target_seq,
            neighbors=neighbors,
            gid_to_seq=gid_to_seq or {},
            gid_to_lineage=gid_to_lineage or None,
            gid_to_date=gid_to_date or None,
            source_lineage=source_lineage,
            target_lineage=target_lineage,
            gt_blosum_neighbors=gt_blosum_neighbors or None,
            out_consensus_fasta=out_consensus_fasta,
        )
        logger.info("Wrote %s", out_fasta)
        report["out_fasta"] = str(out_fasta)
        out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    md_path = out_json.with_suffix(".md")
    lines = [
        f"# GA vs GT / kNN (row {report['row']})",
        "",
        f"- source: `{report['genome_id_source']}` lineage=`{report.get('lineage_source','')}` collection_date=`{report.get('date_source','')}`",
        f"- target: `{report['genome_id_target']}` lineage=`{report.get('lineage_target','')}` collection_date=`{report.get('date_target','')}`",
        f"- knn_space: `{m.get('knn_space')}` metric=`{m.get('knn_metric')}` "
        f"(mean_w={m.get('knn_mean_weight')}, pca2d_w={m.get('knn_pca2d_weight')})",
        f"- knn_before_date: `{report.get('knn_before_date')}`",
        f"- gt_blosum_k: `{int(args.gt_blosum_k)}`",
        "",
        "| metric | value |",
        "|--------|-------|",
    ]
    for k in (
        "gt_hamming_source_target",
        "ga_hamming_to_gt",
        "ga_identity_to_gt",
        "ga_hamming_to_source",
        "ga_identity_to_source",
        "knn_best_hamming_to_gt",
        "knn_best_identity_to_gt",
        "knn_best_hamming_to_source",
        "knn_best_identity_to_source",
        "knn_target_sequence_found",
        "knn_target_sequence_gids",
        "ga_better_than_knn",
        "knn_better_than_ga",
        "ga_hamming_to_best_knn",
    ):
        if k in m:
            lines.append(f"| {k} | {m[k]} |")
            logger.info("  %s=%s", k, m[k])
    if neighbors:
        lines += [
            "",
            "## kNN neighbors",
            "",
            "| rank | gid | lineage | dist | ham_gt | ham_src | collection_date |",
            "|------|-----|---------|------|--------|---------|-----------------|",
        ]
        for nb in neighbors:
            lines.append(
                f"| {nb.get('rank')} | {nb.get('gid')} | {nb.get('lineage', '')} | "
                f"{nb.get('distance')} | {nb.get('hamming_to_gt')} | "
                f"{nb.get('hamming_to_source')} |  {nb.get('collection_date', '')} |"
            )
    if gt_blosum_neighbors:
        lines += [
            "",
            "## GT BLOSUM neighbors",
            "",
            "| rank | gid | lineage | blosum_gt | ham_gt | ham_src |",
            "|------|-----|---------|-----------|--------|---------|",
        ]
        for nb in gt_blosum_neighbors:
            lines.append(
                f"| {nb.get('rank')} | {nb.get('gid')} | {nb.get('lineage', '')} | "
                f"{nb.get('blosum_to_gt')} | {nb.get('hamming_to_gt')} | "
                f"{nb.get('hamming_to_source')} |"
            )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote %s", md_path)


# --- batch (multi-row, multi-process) execution -----------------------------

_BATCH_STATE: Optional[tuple[argparse.Namespace, _SharedContext]] = None


def _run_row_logged(
    args: argparse.Namespace, ctx: _SharedContext, row_idx: int
) -> Optional[str]:
    """Process one row with ``[row N]`` log prefix; return error text or None."""
    global _LOG_ROW
    _LOG_ROW = int(row_idx)
    try:
        _process_row(args, ctx, row_idx)
        return None
    except Exception as e:  # keep the batch going; summarised by caller
        logger.exception("row %d failed: %s", row_idx, e)
        return f"{type(e).__name__}: {e}"
    finally:
        _LOG_ROW = None


def _batch_worker_init() -> None:
    # Avoid BLAS/torch thread oversubscription when many workers run at once.
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=1)
    except Exception:
        pass


def _batch_worker(row_idx: int) -> tuple[int, Optional[str]]:
    assert _BATCH_STATE is not None
    args, ctx = _BATCH_STATE
    return row_idx, _run_row_logged(args, ctx, row_idx)


def _run_batch(
    args: argparse.Namespace, ctx: _SharedContext, rows: list[int], num_workers: int
) -> list[tuple[int, str]]:
    global _BATCH_STATE
    n = len(rows)
    failures: list[tuple[int, str]] = []
    done = 0
    t0 = datetime.now()

    def _progress(row_idx: int, err: Optional[str]) -> None:
        nonlocal done
        done += 1
        if err is not None:
            failures.append((row_idx, err))
        logger.info(
            "batch progress: %d / %d rows done (row %d %s, failures=%d, elapsed=%s)",
            done,
            n,
            row_idx,
            "FAILED" if err else "ok",
            len(failures),
            str(datetime.now() - t0).split(".")[0],
        )

    if num_workers <= 1 or n <= 1:
        for r in rows:
            _progress(r, _run_row_logged(args, ctx, r))
        return failures

    import gc
    import multiprocessing as mp

    try:
        mp_ctx = mp.get_context("fork")
    except ValueError as e:
        raise SystemExit(
            "--num-workers > 1 requires the 'fork' start method (Linux/macOS)"
        ) from e
    # Workers inherit the loaded index/MSA via fork (copy-on-write).
    _BATCH_STATE = (args, ctx)
    gc.collect()
    gc.freeze()  # keep GC from touching (and copying) shared pages
    nw = min(int(num_workers), n)
    logger.info("batch: %d rows, %d worker processes", n, nw)
    try:
        with mp_ctx.Pool(processes=nw, initializer=_batch_worker_init) as pool:
            for row_idx, err in pool.imap_unordered(_batch_worker, rows, chunksize=1):
                _progress(row_idx, err)
    finally:
        _BATCH_STATE = None
    return failures


def main() -> None:
    args = _build_parser().parse_args()
    _setup_logging(args.log_level)
    _install_row_log_filter()
    if int(args.gt_blosum_k) < 0:
        raise SystemExit("--gt-blosum-k must be >= 0")
    if int(args.num_workers) < 1:
        raise SystemExit("--num-workers must be >= 1")

    if args.knn_space in ("mean", "combined") and args.knn_mean_weight < 0:
        raise SystemExit("--knn-mean-weight must be >= 0")
    if args.knn_space in ("pca2d", "combined") and args.knn_pca2d_weight < 0:
        raise SystemExit("--knn-pca2d-weight must be >= 0")

    batch = bool(args.all_rows or args.rows is not None)
    if batch:
        for name in ("out_json", "out_fasta", "out_consensus_fasta", "out_mutation_stats"):
            val = getattr(args, name)
            if val is not None and not _has_row_placeholder(val):
                raise SystemExit(
                    f"--{name.replace('_', '-')} must contain '{{row}}' (or '{{}}') "
                    "in --all-rows/--rows mode so rows do not overwrite each other"
                )
    elif int(args.num_workers) > 1:
        raise SystemExit("--num-workers > 1 requires --all-rows or --rows")

    ctx = _prepare_shared(args, batch=batch)

    if not batch:
        _process_row(args, ctx, int(args.row))
        return

    n_rows = len(ctx.df)
    rows = (
        parse_row_spec(args.rows, n_rows)
        if args.rows is not None
        else list(range(n_rows))
    )
    if args.skip_existing:
        keep = [
            r
            for r in rows
            if not _fmt_row_path(args.out_json, r).with_suffix(".md").is_file()
        ]
        logger.info("skip-existing: %d / %d rows already done", len(rows) - len(keep), len(rows))
        rows = keep
    logger.info(
        "batch mode: %d rows of %d (num_workers=%d)", len(rows), n_rows, args.num_workers
    )
    failures = _run_batch(args, ctx, rows, int(args.num_workers))
    if failures:
        logger.error(
            "%d / %d rows failed: %s",
            len(failures),
            len(rows),
            ", ".join(str(r) for r, _ in sorted(failures)[:50]),
        )
        raise SystemExit(1)
    logger.info("batch complete: %d rows", len(rows))


if __name__ == "__main__":
    main()
