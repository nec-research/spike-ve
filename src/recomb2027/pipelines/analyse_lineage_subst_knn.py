#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     analyse_lineage_subst_knn.py
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
"""kNN decode for lineage-subst clouds: baseline + all slots → one package.

Same source/target pair; queries = unmodified B3 baseline plus each finite
prefix-swap / last_step draw. Neighbors are real MSA sequences (for consensus).

Optional ``--gt-blosum-k`` appends date-filtered MSA sequences nearest the GT
target by BLOSUM distance. ``--out-mutation-stats`` writes source→target
mutation coverage (TP/FP/TN/FN) vs latent kNN / GT-BLOSUM pools.

Example::

    .venv/bin/python3 -m recomb2027.pipelines.analyse_lineage_subst_knn \\
        --pairs-csv .../validation_pairs_test.csv --row 0 \\
        --pred-npz "$OUT/b3_lineage_subst_prefix_swap_test.npz" \\
        --encodings-dir .../average_token/33 \\
        --enc-2d-dir .../2d/33 \\
        --pca2d-basis-path .../pca2d_k5_layer33_S1_train.pt \\
        --msa .../filtered_msa_S1.fasta \\
        --knn-space combined --knn-k 5 \\
        --gt-blosum-k 5 \\
        --out-json "$OUT/knn_row0_cloud.json" \\
        --out-fasta "$OUT/knn_row0_cloud.fasta" \\
        --out-consensus-fasta "$OUT/knn_row0_consensus_pool.fasta" \\
        --out-mutation-stats "$OUT/knn_row0_mutation_stats.json"
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from recomb2027.core.paths import repo_root
from recomb2027.config_io import load_config
from recomb2027.evaluation.knn_decode_utils import (
    DEFAULT_KNN_BEFORE_DATE,
    KNN_METRICS,
    KNN_SPACES,
    _resolve_pca2d_cache_path,
    align_indices_by_gid,
    analyse_row,
    build_knn_allowed_gids,
    compute_mutation_site_stats,
    fasta_header,
    find_blosum_nearest_to_target,
    gid_str,
    hamming,
    identity,
    load_encoding_index,
    load_lineages_from_metadata,
    load_pca2d_encoding_index,
    mutation_stats_for_neighbor_records,
    resolve_knn_before_date,
    resolve_lineage,
)
from recomb2027.pipelines.analyse_baseline_knn import build_knn_consensus

REPO_ROOT = repo_root(__file__)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )


def _finite_vec(v: Optional[np.ndarray]) -> bool:
    if v is None:
        return False
    return bool(np.isfinite(v).all())


def read_pairs_csv(path: Path) -> pd.DataFrame:
    """Load pairs CSV with genome ids kept as strings."""
    dtype = {
        "genome_id_source": "string",
        "genome_id_target": "string",
    }
    df = pd.read_csv(path, dtype=dtype)
    for col in ("genome_id_source", "genome_id_target"):
        if col in df.columns:
            df[col] = df[col].map(gid_str)
    return df


def _source_gids_from_npz(z: Any, split: Optional[str] = None) -> Optional[np.ndarray]:
    keys: list[str] = []
    if split:
        keys.append(f"{split}_source_gids")
    keys.extend(["source_gids", "test_source_gids", "val_source_gids", "cal_source_gids"])
    for k in keys:
        if k in z.files:
            return np.asarray(z[k])
    return None


def resolve_pred_row(
    z: Any,
    *,
    pairs_iloc: int,
    pairs_row: pd.Series,
    source_id: str,
    split: Optional[str] = None,
) -> int:
    """Map a pairs CSV row onto the NPZ series index.

    Preference order:
    1. ``series_row`` / ``index`` column when present and in-range
    2. unique match of ``source_id`` against NPZ ``source_gids``
    3. ``pairs_iloc`` fallback
    """
    n_pred: Optional[int] = None
    pred_keys = (
        "pred_baseline_mean_raw",
        "pred_baseline_mean",
        "pred_samples_mean_raw",
        "pred_samples_mean",
        "pred_mean_raw",
        "pred_mean",
        "source_gids",
    )
    if split:
        pred_keys = (
            f"{split}_pred_mean_raw",
            f"{split}_pred_mean",
            f"{split}_source_gids",
            *pred_keys,
        )
    for key in pred_keys:
        if key in z.files:
            n_pred = int(np.asarray(z[key]).shape[0])
            break

    candidates: list[tuple[str, int]] = []
    for col in ("series_row", "index"):
        if col not in pairs_row.index:
            continue
        raw = pairs_row[col]
        if pd.isna(raw):
            continue
        try:
            idx = int(raw)
        except (TypeError, ValueError):
            continue
        if n_pred is not None and not (0 <= idx < n_pred):
            logger.warning(
                "pairs %s=%d out of range for pred n=%d; ignoring",
                col,
                idx,
                n_pred,
            )
            continue
        candidates.append((col, idx))

    gids = _source_gids_from_npz(z, split=split)
    gid_match: Optional[int] = None
    if gids is not None and source_id:
        gid_list = [gid_str(g) for g in gids.tolist()]
        matches = [
            i
            for i, g in enumerate(gid_list)
            if g == source_id or g.startswith(source_id) or source_id.startswith(g)
        ]
        if len(matches) == 1:
            gid_match = int(matches[0])
        elif len(matches) > 1:
            logger.warning(
                "source_id %r matches %d NPZ rows; not using gid remap",
                source_id,
                len(matches),
            )

    if candidates:
        col, idx = candidates[0]
        if gid_match is not None and gid_match != idx:
            logger.warning(
                "pairs %s=%d disagrees with source_gids match %d for %r; "
                "using source_gids",
                col,
                idx,
                gid_match,
                source_id,
            )
            return gid_match
        if idx != int(pairs_iloc):
            logger.info(
                "resolved pred row via pairs.%s: iloc=%d → npz=%d",
                col,
                int(pairs_iloc),
                idx,
            )
        return idx

    if gid_match is not None:
        if gid_match != int(pairs_iloc):
            logger.info(
                "resolved pred row via source_gids: iloc=%d → npz=%d (gid=%s)",
                int(pairs_iloc),
                gid_match,
                source_id,
            )
        return gid_match

    return int(pairs_iloc)


def assert_source_gid_aligned(
    z: Any,
    *,
    pred_row: int,
    source_id: str,
    split: Optional[str] = None,
) -> None:
    gids = _source_gids_from_npz(z, split=split)
    if gids is None or not source_id:
        return
    if not (0 <= int(pred_row) < len(gids)):
        raise IndexError(
            f"pred row {pred_row} out of range for source_gids (n={len(gids)})"
        )
    npz_gid = gid_str(gids[int(pred_row)])
    if npz_gid != source_id and not (
        npz_gid.startswith(source_id) or source_id.startswith(npz_gid)
    ):
        raise SystemExit(
            f"pairs genome_id_source={source_id!r} does not match "
            f"NPZ source_gids[{pred_row}]={npz_gid!r}; "
            "rebuild pairs from the same series NPZ as infer, or pass the correct --row"
        )


class InMemoryNpz:
    """npz-like (``.files``, ``[key]``, ``close()``) holding arrays in memory.

    ``np.load(...)[key]`` re-reads (and decompresses) the whole array on each
    access; keeping the arrays in memory makes per-row slicing cheap and lets
    forked workers share them. Arrays never used by the kNN analysis
    (activation probabilities, targets, ``*_sites_raw`` duplicates of the
    flat arrays) are skipped.
    """

    _SKIP_SUBSTR = ("activation_probs",)
    _SKIP_PREFIX = ("target_", "test_target_", "val_target_", "cal_target_")

    def __init__(self, path: Path) -> None:
        z = np.load(path, allow_pickle=True)
        try:
            files = list(z.files)
            data: dict[str, np.ndarray] = {}
            for k in files:
                if any(t in k for t in self._SKIP_SUBSTR) or k.startswith(self._SKIP_PREFIX):
                    continue
                if k.endswith("pca2d_sites_raw") and k.replace("sites_raw", "flat_raw") in files:
                    continue
                data[k] = np.asarray(z[k])
        finally:
            z.close()
        self._data = data

    @property
    def files(self) -> list[str]:
        return list(self._data.keys())

    def __getitem__(self, key: str) -> np.ndarray:
        return self._data[key]

    def close(self) -> None:  # shared; nothing to release per row
        pass


def load_cloud_queries(
    pred_npz: Path,
    *,
    row: int,
    split: Optional[str] = None,
    prefer_raw: bool = True,
    knn_space: str = "combined",
    npz: Any = None,
) -> list[dict[str, Any]]:
    """Build query list: baseline + finite sample slots for one series row.

    ``npz`` may be an already-open npz-like object (see :class:`InMemoryNpz`);
    it is then used as is (and not closed).
    """
    if knn_space not in KNN_SPACES:
        raise ValueError(f"knn_space must be one of {KNN_SPACES}; got {knn_space!r}")
    need_mean = knn_space in ("mean", "combined")
    need_pca = knn_space in ("pca2d", "combined")

    z = npz if npz is not None else np.load(pred_npz, allow_pickle=True)
    try:
        sp = (
            str(split)
            if split
            else str(
                np.asarray(z["split"]).item() if "split" in z.files else "test"
            )
        )

        def _base_mean() -> Optional[np.ndarray]:
            if prefer_raw:
                for key in (f"{sp}_pred_mean_raw", "pred_mean_raw"):
                    if key in z.files:
                        return np.asarray(z[key][row], dtype=np.float32)
            if prefer_raw and "pred_baseline_mean_raw" in z.files:
                return np.asarray(z["pred_baseline_mean_raw"][row], dtype=np.float32)
            for key in (
                "pred_baseline_mean",
                f"{sp}_pred_baseline_mean",
                f"{sp}_pred_mean",
                "pred_mean",
            ):
                if key in z.files:
                    return np.asarray(z[key][row], dtype=np.float32)
            return None

        def _base_pca() -> Optional[np.ndarray]:
            if prefer_raw:
                for key in (
                    f"{sp}_pred_pca2d_flat_raw",
                    "pred_pca2d_flat_raw",
                ):
                    if key in z.files:
                        return np.asarray(z[key][row], dtype=np.float32).reshape(-1)
                for key in (
                    f"{sp}_pred_pca2d_sites_raw",
                    "pred_pca2d_sites_raw",
                ):
                    if key in z.files:
                        return np.asarray(z[key][row], dtype=np.float32).reshape(-1)
            if prefer_raw and "pred_baseline_pca2d_flat_raw" in z.files:
                return np.asarray(
                    z["pred_baseline_pca2d_flat_raw"][row], dtype=np.float32
                ).reshape(-1)
            if prefer_raw and "pred_baseline_pca2d_sites_raw" in z.files:
                return np.asarray(
                    z["pred_baseline_pca2d_sites_raw"][row], dtype=np.float32
                ).reshape(-1)
            for key in (
                "pred_baseline_pca2d",
                f"{sp}_pred_baseline_pca2d",
                f"{sp}_pred_pca2d",
                "pred_pca2d",
            ):
                if key in z.files:
                    return np.asarray(z[key][row], dtype=np.float32).reshape(-1)
            return None

        def _samp_mean() -> Optional[np.ndarray]:
            if prefer_raw and "pred_samples_mean_raw" in z.files:
                return np.asarray(z["pred_samples_mean_raw"][row], dtype=np.float32)
            for key in ("pred_samples_mean", f"{sp}_pred_samples_mean"):
                if key in z.files:
                    return np.asarray(z[key][row], dtype=np.float32)
            return None

        def _samp_pca() -> Optional[np.ndarray]:
            if prefer_raw and "pred_samples_pca2d_flat_raw" in z.files:
                return np.asarray(
                    z["pred_samples_pca2d_flat_raw"][row], dtype=np.float32
                )
            if prefer_raw and "pred_samples_pca2d_sites_raw" in z.files:
                a = np.asarray(z["pred_samples_pca2d_sites_raw"][row], dtype=np.float32)
                return a.reshape(a.shape[0], -1)
            for key in ("pred_samples_pca2d", f"{sp}_pred_samples_pca2d"):
                if key in z.files:
                    return np.asarray(z[key][row], dtype=np.float32)
            return None

        def _slot_ok(pm: Optional[np.ndarray], pp: Optional[np.ndarray]) -> bool:
            if need_mean and not _finite_vec(pm):
                return False
            if need_pca and not _finite_vec(pp):
                return False
            return True

        queries: list[dict[str, Any]] = []
        bm = _base_mean()
        bp = _base_pca()
        if _slot_ok(bm, bp):
            queries.append(
                {
                    "query_id": "baseline",
                    "slot": -1,
                    "pred_mean": None if bm is None else bm.reshape(-1),
                    "pred_pca2d": None if bp is None else bp.reshape(-1),
                }
            )
        else:
            logger.warning(
                "baseline skipped for knn_space=%s (finite mean=%s pca=%s)",
                knn_space,
                _finite_vec(bm),
                _finite_vec(bp),
            )

        sm = _samp_mean()
        spca = _samp_pca()
        if sm is None and spca is None:
            return queries

        n_mean = int(sm.shape[0]) if sm is not None else None
        n_pca = int(spca.shape[0]) if spca is not None else None
        if n_mean is not None and n_pca is not None and n_mean != n_pca:
            logger.warning(
                "sample slot count mismatch mean=%d pca2d=%d; using min=%d",
                n_mean,
                n_pca,
                min(n_mean, n_pca),
            )
            n_slots = min(n_mean, n_pca)
        else:
            n_slots = int(n_mean if n_mean is not None else n_pca)
        skipped = 0
        for j in range(n_slots):
            pm = None if sm is None else sm[j].reshape(-1)
            pp = None if spca is None else spca[j].reshape(-1)
            if not _slot_ok(pm, pp):
                skipped += 1
                continue
            queries.append(
                {
                    "query_id": f"slot_{j}",
                    "slot": int(j),
                    "pred_mean": pm,
                    "pred_pca2d": pp,
                }
            )
        if skipped:
            logger.info(
                "skipped %d / %d sample slots (non-finite for knn_space=%s)",
                skipped,
                n_slots,
                knn_space,
            )
        return queries
    finally:
        if npz is None:
            z.close()


def write_multi_knn_fasta(
    path: Path,
    *,
    source_id: str,
    source_seq: str,
    target_id: str,
    target_seq: str,
    queries: list[dict[str, Any]],
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
    for q in queries:
        qid = str(q["query_id"])
        for nb in q.get("knn_neighbors") or []:
            gid = str(nb["gid"])
            seq = gid_to_seq.get(gid)
            if seq is None:
                continue
            lin = resolve_lineage(
                gid, gid_to_lineage, fallback=str(nb.get("lineage") or "")
            )
            meta = [
                f"query={qid}",
                f"slot={q.get('slot', '')}",
                f"rank={nb.get('rank', '')}",
                f"distance={nb.get('distance', '')}",
                f"hamming_to_target={nb.get('hamming_to_gt', '')}",
                f"hamming_to_source={nb.get('hamming_to_source', '')}",
            ]
            if "distance_mean" in nb:
                meta.append(f"distance_mean={nb['distance_mean']}")
            if "distance_pca2d" in nb:
                meta.append(f"distance_pca2d={nb['distance_pca2d']}")
            lines.append(fasta_header("knn", gid, lineage=lin, extras=meta))
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


def write_consensus_pool_fasta(
    path: Path,
    *,
    source_id: str,
    source_seq: str,
    target_id: str,
    target_seq: str,
    queries: list[dict[str, Any]],
    gid_to_seq: dict[str, str],
    gid_to_lineage: Optional[dict[str, str]] = None,
    source_lineage: str = "",
    target_lineage: str = "",
) -> dict[str, Any]:
    """Unique neighbor sequences across all queries, with vote counts."""
    votes: dict[str, dict[str, Any]] = {}
    for q in queries:
        qid = str(q["query_id"])
        for nb in q.get("knn_neighbors") or []:
            gid = str(nb["gid"])
            if gid not in votes:
                votes[gid] = {
                    "gid": gid,
                    "lineage": resolve_lineage(
                        gid, gid_to_lineage, fallback=str(nb.get("lineage") or "")
                    ),
                    "n_queries": 0,
                    "queries": [],
                    "best_rank": int(nb.get("rank", 10**9)),
                    "best_distance": float(nb.get("distance", np.inf)),
                    "hamming_to_gt": nb.get("hamming_to_gt"),
                    "hamming_to_source": nb.get("hamming_to_source"),
                }
            entry = votes[gid]
            if qid not in entry["queries"]:
                entry["queries"].append(qid)
                entry["n_queries"] = len(entry["queries"])
            entry["best_rank"] = min(int(entry["best_rank"]), int(nb.get("rank", 10**9)))
            entry["best_distance"] = min(
                float(entry["best_distance"]), float(nb.get("distance", np.inf))
            )

    ranked = sorted(
        votes.values(),
        key=lambda e: (-int(e["n_queries"]), int(e["best_rank"]), float(e["best_distance"])),
    )
    src_lin = resolve_lineage(source_id, gid_to_lineage, fallback=source_lineage)
    tgt_lin = resolve_lineage(target_id, gid_to_lineage, fallback=target_lineage)
    lines: list[str] = [
        fasta_header("source", source_id, lineage=src_lin),
        source_seq,
        fasta_header("target", target_id, lineage=tgt_lin),
        target_seq,
    ]
    for e in ranked:
        gid = str(e["gid"])
        seq = gid_to_seq.get(gid)
        if seq is None:
            continue
        meta = [
            f"votes={e['n_queries']}",
            f"best_rank={e['best_rank']}",
            f"best_distance={e['best_distance']}",
            f"queries={','.join(e['queries'])}",
            f"hamming_to_target={e.get('hamming_to_gt', '')}",
            f"hamming_to_source={e.get('hamming_to_source', '')}",
        ]
        lines.append(
            fasta_header(
                "consensus_cand",
                gid,
                lineage=str(e.get("lineage") or ""),
                extras=meta,
            )
        )
        lines.append(seq)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"n_unique_neighbors": len(ranked), "ranked": ranked}


def build_lineage_subst_mutation_stats_report(
    *,
    source_id: str,
    target_id: str,
    source_seq: str,
    target_seq: str,
    source_lineage: str,
    target_lineage: str,
    queries: list[dict[str, Any]],
    gid_to_seq: dict[str, str],
    knn_space: str,
    knn_k: int,
    gt_blosum_k: int,
    gt_blosum_neighbors: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Assemble ``--out-mutation-stats`` JSON payload.

    Sections:
    - ``per_query``: metrics for each baseline/slot neighbor list
    - ``union_all_slots``: unique latent kNN over ``slot_*`` queries only
      (excludes baseline and gt_blosum)
    - ``union_all_queries``: unique latent kNN over baseline + all slots
      (excludes gt_blosum)
    - ``gt_blosum``: optional oracle BLOSUM-to-GT pool
    """
    per_query_stats: list[dict[str, Any]] = []
    union_all_queries: list[dict[str, Any]] = []
    union_slots: list[dict[str, Any]] = []
    seen_all: set[str] = set()
    seen_slots: set[str] = set()
    slot_query_ids: list[str] = []
    all_query_ids: list[str] = []

    for q in queries:
        qid = str(q.get("query_id", ""))
        slot = q.get("slot", None)
        try:
            slot_i = int(slot) if slot is not None else -1
        except (TypeError, ValueError):
            slot_i = -1
        is_slot = slot_i >= 0 or qid.startswith("slot_")
        nbs = list(q.get("knn_neighbors") or [])
        all_query_ids.append(qid)
        per_query_stats.append(
            mutation_stats_for_neighbor_records(
                source_seq,
                target_seq,
                nbs,
                gid_to_seq,
                label=qid,
                include_site_lists=False,
            )
        )
        for nb in nbs:
            gid = gid_str(nb.get("gid"))
            if not gid:
                continue
            if gid not in seen_all:
                seen_all.add(gid)
                union_all_queries.append(nb)
            if is_slot:
                if qid not in slot_query_ids:
                    slot_query_ids.append(qid)
                if gid not in seen_slots:
                    seen_slots.add(gid)
                    union_slots.append(nb)

    union_slots_stats = mutation_stats_for_neighbor_records(
        source_seq,
        target_seq,
        union_slots,
        gid_to_seq,
        label="union_all_slots",
        include_site_lists=True,
    )
    union_slots_stats["query_ids"] = list(slot_query_ids)
    union_slots_stats["n_queries"] = len(slot_query_ids)

    union_all_stats = mutation_stats_for_neighbor_records(
        source_seq,
        target_seq,
        union_all_queries,
        gid_to_seq,
        label="union_all_queries",
        include_site_lists=True,
    )
    union_all_stats["query_ids"] = list(all_query_ids)
    union_all_stats["n_queries"] = len(all_query_ids)

    mut_report: dict[str, Any] = {
        "genome_id_source": source_id,
        "genome_id_target": target_id,
        "lineage_source": source_lineage,
        "lineage_target": target_lineage,
        "gt_hamming_source_target": hamming(source_seq, target_seq),
        "knn_space": knn_space,
        "knn_k": int(knn_k),
        "gt_blosum_k": int(gt_blosum_k),
        "per_query": per_query_stats,
        # Unique latent-kNN over sample slots only (excludes baseline + gt_blosum).
        "union_all_slots": union_slots_stats,
        # Unique latent-kNN over baseline + all slots (excludes gt_blosum).
        "union_all_queries": union_all_stats,
        # Backward-compatible alias of union_all_queries.
        "pooled_unique_latent_knn": union_all_stats,
    }

    if gt_blosum_neighbors:
        mut_report["gt_blosum"] = mutation_stats_for_neighbor_records(
            source_seq,
            target_seq,
            gt_blosum_neighbors,
            gid_to_seq,
            label="gt_blosum",
            include_site_lists=True,
        )
    return mut_report


def _neighbor_sequences(
    neighbors: list[dict[str, Any]],
    gid_to_seq: dict[str, str],
) -> list[str]:
    seqs: list[str] = []
    for nb in neighbors:
        gid = gid_str(nb.get("gid"))
        seq = gid_to_seq.get(gid) or nb.get("seq")
        if seq:
            seqs.append(str(seq))
    return seqs


def _knn_aa_consensus_block(
    source_seq: str,
    target_seq: str,
    neighbors: list[dict[str, Any]],
    gid_to_seq: dict[str, str],
    *,
    label: str,
) -> dict[str, Any]:
    """AA majority consensus over neighbor sequences + mutation-site stats."""
    seqs = _neighbor_sequences(neighbors, gid_to_seq)
    if seqs:
        n = min(len(s) for s in seqs)
        seqs = [s[:n] for s in seqs]
    consensus = build_knn_consensus(seqs)
    stats = compute_mutation_site_stats(
        source_seq,
        target_seq,
        [consensus] if consensus else [],
        label=label,
    )
    change = stats.get("confusion_change_from_source") or {}
    return {
        "label": label,
        "sequence": consensus,
        "n_neighbor_seqs": len(seqs),
        "confusion_change_from_source": change,
        "confusion_target_allele": stats.get("confusion_target_allele") or {},
        "true_positive_positions": stats.get("true_positive_positions", []),
        "false_negative_positions": stats.get("false_negative_positions", []),
    }


def enrich_mutation_report_with_aa_consensus(
    mut_report: dict[str, Any],
    *,
    source_seq: str,
    target_seq: str,
    queries: list[dict[str, Any]],
    gid_to_seq: dict[str, str],
) -> dict[str, Any]:
    """Add AA consensus blocks for baseline, slots-only, and pooled kNN neighbors."""
    baseline_nbs: list[dict[str, Any]] = []
    slot_nbs: list[dict[str, Any]] = []
    all_nbs: list[dict[str, Any]] = []
    seen_all: set[str] = set()
    seen_slots: set[str] = set()

    for q in queries:
        qid = str(q.get("query_id", ""))
        slot = q.get("slot", None)
        try:
            slot_i = int(slot) if slot is not None else -1
        except (TypeError, ValueError):
            slot_i = -1
        is_slot = slot_i >= 0 or qid.startswith("slot_")
        nbs = list(q.get("knn_neighbors") or [])
        if qid == "baseline" or slot_i < 0:
            baseline_nbs.extend(nbs)
        for nb in nbs:
            gid = gid_str(nb.get("gid"))
            if not gid:
                continue
            if gid not in seen_all:
                seen_all.add(gid)
                all_nbs.append(nb)
            if is_slot and gid not in seen_slots:
                seen_slots.add(gid)
                slot_nbs.append(nb)

    mut_report["knn_consensus_baseline"] = _knn_aa_consensus_block(
        source_seq, target_seq, baseline_nbs, gid_to_seq, label="knn_consensus_baseline"
    )
    mut_report["knn_consensus_slots"] = _knn_aa_consensus_block(
        source_seq, target_seq, slot_nbs, gid_to_seq, label="knn_consensus_slots"
    )
    mut_report["knn_consensus_pooled"] = _knn_aa_consensus_block(
        source_seq, target_seq, all_nbs, gid_to_seq, label="knn_consensus_pooled"
    )
    return mut_report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "kNN neighbors for lineage-subst baseline + all cloud slots "
            "(one JSON/FASTA package for consensus)."
        )
    )
    p.add_argument("--pairs-csv", type=Path, required=True)
    p.add_argument(
        "--row",
        type=int,
        default=0,
        help="Row iloc in --pairs-csv (NPZ index resolved via series_row/source_gids)",
    )
    p.add_argument("--pred-npz", type=Path, required=True)
    p.add_argument("--pred-split", default=None)
    p.add_argument("--encodings-dir", type=Path, default=None)
    p.add_argument("--enc-2d-dir", type=Path, default=None)
    p.add_argument("--pca2d-basis-path", type=Path, default=None)
    p.add_argument("--msa", required=True)
    p.add_argument("--knn-k", type=int, default=5)
    p.add_argument("--knn-space", choices=KNN_SPACES, default="combined")
    p.add_argument(
        "--knn-metric",
        choices=KNN_METRICS,
        default="euclidean",
        help="euclidean (L2, default) or cosine (1-cos after unit-norm).",
    )
    p.add_argument("--knn-mean-weight", type=float, default=1.0)
    p.add_argument("--knn-pca2d-weight", type=float, default=0.75)
    p.add_argument(
        "--max-index-files",
        type=int,
        default=5000,
        help="Cap encoding index size (sorted filenames). Use 0 for all files.",
    )
    p.add_argument(
        "--dataset-config",
        type=Path,
        default=None,
        help="Latent dataset YAML; default --knn-before-date from split_threshold "
        f"and optional metadata CSV. Fallback cutoff: {DEFAULT_KNN_BEFORE_DATE}.",
    )
    p.add_argument(
        "--metadata-csv",
        type=Path,
        default=None,
        help="Labels CSV (e.g. df_labels.csv) with genome_id, collection_date, "
        "lineage. Dates take priority over MSA headers; lineage is written into "
        "FASTA headers.",
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
        "enc-2d-dir/_knn_pca2d_cache/. Reused across rows when genome ids match.",
    )
    p.add_argument(
        "--no-pca2d-index-cache",
        action="store_true",
        help="Disable pca2d index caching.",
    )
    p.add_argument(
        "--prefer-raw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use *_raw prediction arrays when present (default: true).",
    )
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--out-fasta", type=Path, default=None)
    p.add_argument(
        "--out-consensus-fasta",
        type=Path,
        default=None,
        help="Unique neighbor sequences across queries, ranked by vote count.",
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
        help="Write source→target mutation coverage JSON: per_query, "
        "union_all_slots (unique latent kNN over slot_* only), "
        "union_all_queries (baseline+slots), and optional gt_blosum. "
        "Never mixes gt_blosum sequences into the latent-union sections.",
    )
    p.add_argument("--log-level", default="INFO")
    return p


class SharedContext:
    """Row-independent heavy data (MSA, kNN index, metadata, predictions)."""

    df: pd.DataFrame
    knn_before: Optional[str]
    gid_to_lineage: dict
    allowed_gids: Optional[set]
    gid_to_seq: dict
    need_mean: bool
    need_pca: bool
    enc_mean: Optional[np.ndarray]
    enc_pca2d: Optional[np.ndarray]
    enc_gids: list
    max_index_files: Optional[int]
    pred_npz: Optional[InMemoryNpz]


def prepare_shared(
    args: argparse.Namespace, *, preload_pred: bool = False, check_row: bool = True
) -> SharedContext:
    """Load everything that does not depend on the pairs row."""
    ctx = SharedContext()
    if int(args.max_index_files) <= 0:
        ctx.max_index_files = None
        logger.info("encoding index: using all files under encodings dirs")
    else:
        ctx.max_index_files = int(args.max_index_files)

    df = read_pairs_csv(args.pairs_csv)
    ctx.df = df
    if check_row:
        pairs_iloc = int(args.row)
        if not (0 <= pairs_iloc < len(df)):
            raise SystemExit(f"--row {pairs_iloc} out of range for pairs csv (n={len(df)})")

    from recomb2027.latent.data_series.latent_decoder_io import parse_msa_gids

    knn_before = resolve_knn_before_date(
        knn_before_date=args.knn_before_date,
        dataset_config=args.dataset_config,
        no_knn_date_filter=bool(args.no_knn_date_filter),
    )
    metadata_csv = args.metadata_csv
    if metadata_csv is None and args.dataset_config is not None and Path(args.dataset_config).is_file():
        from omegaconf import OmegaConf

        dcfg = load_config(str(args.dataset_config))
        meta = dcfg.get("metadata_msa_path")
        if meta:
            metadata_csv = Path(str(meta))
    gid_to_lineage: dict[str, str] = {}
    if metadata_csv is not None and Path(metadata_csv).is_file():
        gid_to_lineage = load_lineages_from_metadata(metadata_csv)
    ctx.knn_before = knn_before
    ctx.gid_to_lineage = gid_to_lineage
    gid_to_seq = parse_msa_gids(args.msa)
    msa_gid_set = set(gid_to_seq.keys())
    allowed_gids = build_knn_allowed_gids(
        args.msa,
        before_date=knn_before,
        metadata_csv=metadata_csv,
        msa_gids=msa_gid_set,
    )
    if knn_before is None:
        logger.info(
            "kNN date filter: disabled (restricting index to %d MSA genome ids)",
            len(msa_gid_set),
        )
    else:
        logger.info("kNN date filter: collection_date <= %s", knn_before)

    index_allowed_gids = (
        allowed_gids if allowed_gids is not None else msa_gid_set
    )
    need_mean = args.knn_space in ("mean", "combined")
    need_pca = args.knn_space in ("pca2d", "combined")
    enc_mean = enc_pca2d = None
    mean_gids = pca_gids = None

    if need_mean:
        if args.encodings_dir is None:
            raise SystemExit("--encodings-dir required for mean/combined")
        enc_mean, mean_gids = load_encoding_index(
            Path(args.encodings_dir),
            max_files=ctx.max_index_files,
            allowed_gids=index_allowed_gids,
        )
        logger.info("mean index n=%d dim=%d", *enc_mean.shape)
    if need_pca:
        if args.enc_2d_dir is None or args.pca2d_basis_path is None:
            raise SystemExit(
                "--enc-2d-dir and --pca2d-basis-path required for pca2d/combined"
            )
        pca_cache = _resolve_pca2d_cache_path(
            enc_2d_dir=Path(args.enc_2d_dir),
            basis_path=Path(args.pca2d_basis_path),
            knn_before_date=knn_before,
            max_files=ctx.max_index_files,
            pca2d_index_cache=args.pca2d_index_cache,
            no_pca2d_index_cache=bool(args.no_pca2d_index_cache),
        )
        enc_pca2d, pca_gids = load_pca2d_encoding_index(
            Path(args.enc_2d_dir),
            Path(args.pca2d_basis_path),
            gids=mean_gids,
            max_files=ctx.max_index_files if mean_gids is None else None,
            allowed_gids=index_allowed_gids,
            cache_path=pca_cache,
            knn_before_date=knn_before,
        )
        logger.info("pca2d index n=%d dim=%d", *enc_pca2d.shape)

    enc_mean, enc_pca2d, enc_gids = align_indices_by_gid(
        enc_mean, mean_gids, enc_pca2d, pca_gids
    )
    logger.info("aligned index n=%d", len(enc_gids))

    ctx.allowed_gids = allowed_gids
    ctx.gid_to_seq = gid_to_seq
    ctx.need_mean, ctx.need_pca = need_mean, need_pca
    ctx.enc_mean, ctx.enc_pca2d, ctx.enc_gids = enc_mean, enc_pca2d, enc_gids
    ctx.pred_npz = None
    if preload_pred:
        ctx.pred_npz = InMemoryNpz(Path(args.pred_npz))
        logger.info(
            "preloaded predictions from %s (keys=%s)", args.pred_npz, sorted(ctx.pred_npz.files)
        )
    return ctx


def process_row(
    args: argparse.Namespace,
    ctx: SharedContext,
    *,
    pairs_iloc: Optional[int] = None,
    on_queries: Optional[Callable[[list[dict[str, Any]], SharedContext, argparse.Namespace], None]] = None,
) -> dict[str, Any]:
    """Run the kNN analysis for one pairs row using preloaded shared data.

    ``on_queries(queries, ctx, args)`` is called once the row's query vectors are
    built (used by the batch runner to precompute all distances in one pass).
    """
    df = ctx.df
    if pairs_iloc is None:
        pairs_iloc = int(args.row)
    pairs_iloc = int(pairs_iloc)
    gid_to_lineage = ctx.gid_to_lineage
    gid_to_seq = ctx.gid_to_seq
    allowed_gids = ctx.allowed_gids
    knn_before = ctx.knn_before
    need_mean, need_pca = ctx.need_mean, ctx.need_pca
    enc_mean, enc_pca2d, enc_gids = ctx.enc_mean, ctx.enc_pca2d, ctx.enc_gids
    max_index_files = ctx.max_index_files

    pairs_row = df.iloc[pairs_iloc]
    source_seq = str(pairs_row["seq_source"])
    target_seq = str(pairs_row["seq_target"])
    source_id = gid_str(pairs_row.get("genome_id_source", ""))
    target_id = gid_str(pairs_row.get("genome_id_target", ""))
    source_lineage_fb = gid_str(pairs_row.get("lineage_source", ""))
    target_lineage_fb = gid_str(pairs_row.get("lineage_target", ""))
    if source_lineage_fb.lower() in {"unknown", "nan", "none"}:
        source_lineage_fb = ""
    if target_lineage_fb.lower() in {"unknown", "nan", "none"}:
        target_lineage_fb = ""

    source_lineage = resolve_lineage(
        source_id, gid_to_lineage, fallback=source_lineage_fb
    )
    target_lineage = resolve_lineage(
        target_id, gid_to_lineage, fallback=target_lineage_fb
    )
    z_meta = ctx.pred_npz if ctx.pred_npz is not None else np.load(
        args.pred_npz, allow_pickle=True
    )
    try:
        pred_split = args.pred_split
        if pred_split is None and "split" in z_meta.files:
            pred_split = str(np.asarray(z_meta["split"]).item())
        pred_row = resolve_pred_row(
            z_meta,
            pairs_iloc=pairs_iloc,
            pairs_row=pairs_row,
            source_id=source_id,
            split=pred_split,
        )
        assert_source_gid_aligned(
            z_meta,
            pred_row=pred_row,
            source_id=source_id,
            split=pred_split,
        )
    finally:
        if ctx.pred_npz is None:
            z_meta.close()

    queries = load_cloud_queries(
        Path(args.pred_npz),
        row=int(pred_row),
        split=pred_split,
        prefer_raw=bool(args.prefer_raw),
        knn_space=str(args.knn_space),
        npz=ctx.pred_npz,
    )
    if on_queries is not None:
        on_queries(queries, ctx, args)
    logger.info(
        "queries: %d (baseline + finite slots) pairs_iloc=%d pred_row=%d",
        len(queries),
        pairs_iloc,
        pred_row,
    )

    enriched: list[dict[str, Any]] = []
    for q in queries:
        metrics = analyse_row(
            source_seq=source_seq,
            target_seq=target_seq,
            ga_seq=None,
            pred_mean=q["pred_mean"] if need_mean else None,
            pred_pca2d=q["pred_pca2d"] if need_pca else None,
            enc_index=enc_mean,
            enc_pca2d=enc_pca2d,
            enc_gids=enc_gids,
            gid_to_seq=gid_to_seq,
            gid_to_lineage=gid_to_lineage or None,
            knn_k=int(args.knn_k),
            knn_space=str(args.knn_space),
            knn_metric=str(args.knn_metric),
            knn_mean_weight=float(args.knn_mean_weight),
            knn_pca2d_weight=float(args.knn_pca2d_weight),
        )
        entry = {
            "query_id": q["query_id"],
            "slot": q["slot"],
            "knn_neighbors": metrics.get("knn_neighbors") or [],
            "knn_best_hamming_to_gt": metrics.get("knn_best_hamming_to_gt"),
            "knn_best_identity_to_gt": metrics.get("knn_best_identity_to_gt"),
            "knn_best_hamming_to_source": metrics.get("knn_best_hamming_to_source"),
        }
        enriched.append(entry)
        logger.info(
            "  %s: n_neighbors=%d best_ham_gt=%s",
            entry["query_id"],
            len(entry["knn_neighbors"]),
            entry["knn_best_hamming_to_gt"],
        )

    report: dict[str, Any] = {
        "pairs_iloc": pairs_iloc,
        "pred_row": int(pred_row),
        "row": int(pred_row),  # backward-compatible alias for NPZ index
        "genome_id_source": source_id,
        "genome_id_target": target_id,
        "lineage_source": source_lineage,
        "lineage_target": target_lineage,
        "pred_npz": str(args.pred_npz),
        "pred_split": pred_split,
        "knn_space": args.knn_space,
        "knn_metric": args.knn_metric,
        "knn_k": int(args.knn_k),
        "knn_mean_weight": float(args.knn_mean_weight),
        "knn_pca2d_weight": float(args.knn_pca2d_weight),
        "knn_before_date": knn_before,
        "max_index_files": max_index_files,
        "gt_hamming_source_target": hamming(source_seq, target_seq),
        "gt_identity_source_target": identity(source_seq, target_seq),
        "n_queries": len(enriched),
        "queries": enriched,
        "gt_blosum_k": int(args.gt_blosum_k),
    }

    gt_blosum_neighbors: list[dict[str, Any]] = []
    if int(args.gt_blosum_k) > 0:
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
        report["gt_blosum_neighbors"] = [
            {kk: vv for kk, vv in nb.items() if kk != "seq"}
            for nb in gt_blosum_neighbors
        ]
        logger.info("gt-blosum neighbors: %d", len(gt_blosum_neighbors))

    if args.out_mutation_stats is not None:
        mut_report = build_lineage_subst_mutation_stats_report(
            source_id=source_id,
            target_id=target_id,
            source_seq=source_seq,
            target_seq=target_seq,
            source_lineage=source_lineage,
            target_lineage=target_lineage,
            queries=enriched,
            gid_to_seq=gid_to_seq,
            knn_space=str(args.knn_space),
            knn_k=int(args.knn_k),
            gt_blosum_k=int(args.gt_blosum_k),
            gt_blosum_neighbors=gt_blosum_neighbors or None,
        )
        mut_report = enrich_mutation_report_with_aa_consensus(
            mut_report,
            source_seq=source_seq,
            target_seq=target_seq,
            queries=enriched,
            gid_to_seq=gid_to_seq,
        )
        out_mut = Path(args.out_mutation_stats)
        out_mut.parent.mkdir(parents=True, exist_ok=True)
        out_mut.write_text(json.dumps(mut_report, indent=2) + "\n", encoding="utf-8")
        report["out_mutation_stats"] = str(out_mut)
        logger.info(
            "Wrote mutation stats %s "
            "(per_query=%d, union_all_slots n_knn=%s, union_all_queries n_knn=%s)",
            out_mut,
            len(mut_report.get("per_query") or []),
            (mut_report.get("union_all_slots") or {}).get("n_knn_seqs"),
            (mut_report.get("union_all_queries") or {}).get("n_knn_seqs"),
        )

    out_fasta = args.out_fasta or args.out_json.with_suffix(".fasta")
    write_multi_knn_fasta(
        out_fasta,
        source_id=source_id,
        source_seq=source_seq,
        target_id=target_id,
        target_seq=target_seq,
        queries=enriched,
        gid_to_seq=gid_to_seq,
        gid_to_lineage=gid_to_lineage or None,
        source_lineage=source_lineage,
        target_lineage=target_lineage,
        gt_blosum_neighbors=gt_blosum_neighbors or None,
    )
    report["out_fasta"] = str(out_fasta)
    logger.info("Wrote %s", out_fasta)

    if args.out_consensus_fasta is not None:
        cons = write_consensus_pool_fasta(
            Path(args.out_consensus_fasta),
            source_id=source_id,
            source_seq=source_seq,
            target_id=target_id,
            target_seq=target_seq,
            queries=enriched,
            gid_to_seq=gid_to_seq,
            gid_to_lineage=gid_to_lineage or None,
            source_lineage=source_lineage,
            target_lineage=target_lineage,
        )
        report["consensus"] = {
            "n_unique_neighbors": cons["n_unique_neighbors"],
            "top": cons["ranked"][:20],
        }
        report["out_consensus_fasta"] = str(args.out_consensus_fasta)
        logger.info(
            "Wrote consensus pool (%d unique) → %s",
            cons["n_unique_neighbors"],
            args.out_consensus_fasta,
        )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", args.out_json)
    return report


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.gt_blosum_k) < 0:
        raise SystemExit("--gt-blosum-k must be >= 0")
    if int(args.knn_k) < 0:
        raise SystemExit("--knn-k must be >= 0")
    if args.knn_space in ("mean", "combined") and args.knn_mean_weight < 0:
        raise SystemExit("--knn-mean-weight must be >= 0")
    if args.knn_space in ("pca2d", "combined") and args.knn_pca2d_weight < 0:
        raise SystemExit("--knn-pca2d-weight must be >= 0")


def main(argv: Optional[list[str]] = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    _setup_logging(args.log_level)
    _validate_args(args)
    ctx = prepare_shared(args)
    return process_row(args, ctx)


if __name__ == "__main__":
    main()
