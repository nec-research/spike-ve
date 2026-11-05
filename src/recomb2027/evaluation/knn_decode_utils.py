#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     knn_decode_utils.py
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


"""Shared helpers for kNN-to-prediction decode baselines and GA-vs-latent comparisons."""
from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from recomb2027.config_io import load_config

logger = logging.getLogger(__name__)

KNN_SPACES = ("mean", "pca2d", "combined")
KNN_METRICS = ("euclidean", "cosine")
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


def fasta_header(
    kind: str,
    gid: str,
    *,
    lineage: str = "",
    extras: Optional[list[str]] = None,
) -> str:
    """Build a FASTA header with optional lineage and extra ``key=value`` fields."""
    parts = [str(kind), str(gid)]
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
    x = np.asarray(index, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"index must be 2D, got shape {x.shape}")
    if q.shape[0] != x.shape[1]:
        raise ValueError(
            f"query dim {q.shape[0]} != index dim {x.shape[1]}"
        )
    if metric == "euclidean":
        return np.linalg.norm(x - q.reshape(1, -1), axis=1)

    eps = 1e-12
    q_norm = float(np.linalg.norm(q))
    if q_norm < eps:
        q_u = np.zeros_like(q)
    else:
        q_u = q / q_norm
    x_norm = np.linalg.norm(x, axis=1)
    x_u = np.zeros_like(x)
    ok = x_norm >= eps
    x_u[ok] = x[ok] / x_norm[ok, None]
    cos = np.clip(x_u @ q_u, -1.0, 1.0)
    return 1.0 - cos


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


def load_predictions(
    pred_npz: Path,
    *,
    row: int,
    pred_split: Optional[str] = None,
    source_gid: Optional[str] = None,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[int]]:
    """Load absolute/raw mean and pca2d prediction vectors for one pair row.

    Returns ``(pred_mean, pred_pca2d, resolved_row)``. If ``pred_split`` is
    missing from the NPZ (e.g. ``val`` requested but only ``test_`` / ``cal_``
    keys exist), auto-selects an available split and logs a warning. Optionally
    remaps ``row`` via ``{split}_source_gids``.
    """
    z = np.load(pred_npz, allow_pickle=True)
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
    source_lineage: str = "",
    target_lineage: str = "",
    gt_blosum_neighbors: Optional[list[dict[str, Any]]] = None,
) -> None:
    src_lin = resolve_lineage(source_id, gid_to_lineage, fallback=source_lineage)
    tgt_lin = resolve_lineage(target_id, gid_to_lineage, fallback=target_lineage)
    lines: list[str] = [
        fasta_header("source", source_id, lineage=src_lin),
        source_seq,
        fasta_header("target", target_id, lineage=tgt_lin),
        target_seq,
    ]
    for i, nb in enumerate(neighbors):
        gid = str(nb["gid"])
        seq = gid_to_seq.get(gid)
        if seq is None:
            continue
        lin = resolve_lineage(gid, gid_to_lineage, fallback=str(nb.get("lineage") or ""))
        meta = [
            f"rank={i}",
            f"distance={nb.get('distance', '')}",
            f"hamming_to_target={nb.get('hamming_to_gt', '')}",
            f"hamming_to_source={nb.get('hamming_to_source', '')}",
            f"identity_to_target={nb.get('identity_to_gt', '')}",
            f"identity_to_source={nb.get('identity_to_source', '')}",
        ]
        lines.append(fasta_header(f"knn_{i}", gid, lineage=lin, extras=meta))
        lines.append(seq)
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
            required.append(i)
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
            variable.append(i)
        if differs:
            change_from_source.append(i)
        if i in required_set:
            if has_target:
                recovered.append(i)
            elif differs:
                wrong_allele_only.append(i)

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
        # Compact site lists for debugging (0-based MSA columns).
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
