#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     prefix_swap_esm_cloud.py
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
"""ESM embedding-space analysis for prefix-swap prediction clouds vs GT targets.

Encodes decoded baseline + slot predictions and labeled ground-truth targets with
ESM1B layer-33 mean pooling, compares clouds in embedding space, and builds a
Plotly UMAP with a sampled MSA background.
"""
from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import umap

from recomb2027.data.encode_proteins_from_fasta import (
    DEFAULT_SEQ_LENGTH,
    DEFAULT_TOKEN_BUDGET,
    ESMProteinEncoder,
    ModelFactory,
)
from recomb2027.latent.data_series.latent_decoder_io import (
    build_gid_pt_index,
    load_raw_pooled_pt,
    resolve_genome_id,
)
from recomb2027.latent.plotting import (
    _lineage_label_positions,
    _umap_series_figure_layout,
    _umap_transform_rows,
)
from recomb2027.latent.predictors.visualize_infer_umap import (
    load_background_from_pt_dir,
    load_gid_lineage_map,
)
from recomb2027.pipelines.analyse_baseline_knn import fasta_header
from recomb2027.pipelines.analyse_lineage_subst_knn import gid_str, read_pairs_csv
from recomb2027.pipelines.generate_evaluation_report import _row_id_from_pairs_row
from recomb2027.pipelines.prefix_swap_metrics import (
    build_pred_cloud_consensus,
    parse_prefix_swap_decoded_fasta,
)

logger = logging.getLogger(__name__)

ESM_LAYER = 33
EMBED_DIM = 1280

METADATA_CSV_DEFAULT = (
    ""
)

_PLACEHOLDER_LINEAGES = frozenset(
    {"", "unknown", "n/a", "na", "nan", "none", ".", "null"}
)


def _is_placeholder_lineage(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in _PLACEHOLDER_LINEAGES


def resolve_lineage_for_gid(
    gid: Any,
    column_value: Any,
    lineage_map: Optional[dict[str, str]],
) -> str:
    """Resolve lineage from metadata CSV, falling back to pairs-column value."""
    meta = ""
    if lineage_map is not None:
        meta = str(lineage_map.get(gid_str(gid), "") or "").strip()
    if not _is_placeholder_lineage(column_value):
        return str(column_value).strip()
    return meta


def enrich_pairs_lineages(
    df: pd.DataFrame,
    lineage_map: Optional[dict[str, str]],
) -> pd.DataFrame:
    """Add resolved lineage columns (metadata overrides placeholder CSV values)."""
    out = df.copy()
    if "genome_id_source" in out.columns:
        out["lineage_source_resolved"] = out.apply(
            lambda r: resolve_lineage_for_gid(
                r.get("genome_id_source"),
                r.get("lineage_source") if "lineage_source" in r.index else None,
                lineage_map,
            ),
            axis=1,
        )
    if "genome_id_target" in out.columns:
        out["lineage_target_resolved"] = out.apply(
            lambda r: resolve_lineage_for_gid(
                r.get("genome_id_target"),
                r.get("lineage_target") if "lineage_target" in r.index else None,
                lineage_map,
            ),
            axis=1,
        )
    return out


def _count_lineage_matches(
    df: pd.DataFrame,
    *,
    lineage_col: str,
    pattern: str,
    how: str,
) -> int:
    if lineage_col not in df.columns:
        return 0
    return int(
        df[lineage_col]
        .map(lambda x: _str_match(str(x), pattern, how))
        .sum()
    )


def _format_filter_empty_error(
    df: pd.DataFrame,
    filt: SeriesFilter,
    lineage_map: Optional[dict[str, str]],
) -> str:
    """Build a helpful message when lineage filters exclude every row."""
    parts = ["No series left after filters."]
    enriched = enrich_pairs_lineages(df, lineage_map)

    if filt.source_lineage is not None:
        raw_n = _count_lineage_matches(
            df, lineage_col="lineage_source", pattern=filt.source_lineage, how=filt.source_lineage_match
        )
        res_n = _count_lineage_matches(
            enriched,
            lineage_col="lineage_source_resolved",
            pattern=filt.source_lineage,
            how=filt.source_lineage_match,
        )
        parts.append(
            f"source lineage {filt.source_lineage_match}={filt.source_lineage!r}: "
            f"{raw_n} matches in pairs CSV column lineage_source"
        )
        if lineage_map is not None:
            parts.append(
                f"{res_n} matches after metadata lookup (lineage_source_resolved)"
            )
        else:
            if "lineage_source" in df.columns and df["lineage_source"].astype(str).str.lower().eq("unknown").all():
                parts.append(
                    "pairs CSV lineage_source is all 'unknown'; pass --metadata-csv "
                    f"(default: {METADATA_CSV_DEFAULT}) for lineage filters"
                )
        if res_n == 0 and filt.source_lineage_match == "exact":
            pref_n = _count_lineage_matches(
                enriched,
                lineage_col="lineage_source_resolved",
                pattern=filt.source_lineage,
                how="prefix",
            )
            if pref_n > 0:
                parts.append(
                    f"hint: {pref_n} series match with --filter-source-lineage-match prefix"
                )

    if filt.target_lineage is not None and "lineage_target_resolved" in enriched.columns:
        res_n = _count_lineage_matches(
            enriched,
            lineage_col="lineage_target_resolved",
            pattern=filt.target_lineage,
            how=filt.target_lineage_match,
        )
        parts.append(
            f"target lineage {filt.target_lineage_match}={filt.target_lineage!r}: "
            f"{res_n} matches after metadata lookup"
        )

    if "lineage_source_resolved" in enriched.columns:
        top = (
            enriched["lineage_source_resolved"]
            .replace("", np.nan)
            .dropna()
            .value_counts()
            .head(8)
        )
        if not top.empty:
            parts.append(
                "top source lineages (resolved): "
                + ", ".join(f"{k} ({v})" for k, v in top.items())
            )
    return " ".join(parts)


@dataclass
class SeriesFilter:
    """Criteria to subset validation-pairs rows / decoded series."""

    rows: Optional[list[int]] = None
    source_lineage: Optional[str] = None
    source_lineage_match: str = "prefix"
    target_lineage: Optional[str] = None
    target_lineage_match: str = "prefix"
    min_gt_hamming: Optional[int] = None
    max_gt_hamming: Optional[int] = None
    max_series: Optional[int] = None
    random_state: int = 42


def _str_match(hay: str, needle: str, how: str) -> bool:
    h = str(hay).strip()
    n = str(needle).strip()
    if how == "exact":
        return h == n
    if how == "prefix":
        return h.startswith(n) or h == n
    if how == "contains":
        return n in h
    raise ValueError(f"Unknown match mode: {how!r}")


def filter_pairs_dataframe(
    df: pd.DataFrame,
    *,
    filt: SeriesFilter,
    lineage_map: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """Return a filtered copy of the pairs table."""
    out = df.copy()
    if filt.rows is not None:
        row_set = {int(r) for r in filt.rows}
        keep: list[bool] = []
        for idx, row in out.iterrows():
            rid = _row_id_from_pairs_row(row, idx)
            keep.append(rid in row_set)
        out = out.loc[keep]

    if filt.source_lineage is not None:
        enriched = enrich_pairs_lineages(out, lineage_map)
        out = enriched[
            enriched["lineage_source_resolved"].map(
                lambda x: _str_match(str(x), filt.source_lineage, filt.source_lineage_match)
            )
        ]

    if filt.target_lineage is not None:
        if "lineage_target_resolved" not in out.columns:
            out = enrich_pairs_lineages(out, lineage_map)
        out = out[
            out["lineage_target_resolved"].map(
                lambda x: _str_match(str(x), filt.target_lineage, filt.target_lineage_match)
            )
        ]

    if filt.min_gt_hamming is not None or filt.max_gt_hamming is not None:
        if "seq_source" not in out.columns or "seq_target" not in out.columns:
            raise ValueError("Hamming filters require seq_source and seq_target columns")
        hams = out.apply(
            lambda r: sum(
                1
                for a, b in zip(str(r["seq_source"]), str(r["seq_target"]))
                if a != b
            ),
            axis=1,
        )
        if filt.min_gt_hamming is not None:
            out = out[hams >= int(filt.min_gt_hamming)]
        if filt.max_gt_hamming is not None:
            out = out[hams <= int(filt.max_gt_hamming)]

    if filt.max_series is not None and len(out) > int(filt.max_series):
        rng = random.Random(filt.random_state)
        pick = sorted(rng.sample(range(len(out)), k=int(filt.max_series)))
        out = out.iloc[pick].copy()

    return out.reset_index(drop=True)


@dataclass
class CloudPoint:
    """One embedding-space point for plotting / metrics."""

    row_id: int
    point_type: str  # gt | baseline | slot | source
    query_id: str
    slot: Optional[int]
    genome_id_source: str
    genome_id_target: str
    lineage_source: str
    lineage_target: str
    sequence: str
    label: str


def collect_cloud_points(
    df_pairs: pd.DataFrame,
    decoded_by_row: dict[int, dict[str, dict[str, Any]]],
    *,
    lineage_map: Optional[dict[str, str]] = None,
    include_source: bool = False,
) -> list[CloudPoint]:
    """Gather GT, optional source, baseline, and slot sequences for selected rows."""
    points: list[CloudPoint] = []
    for idx, row in df_pairs.iterrows():
        row_id = _row_id_from_pairs_row(row, idx)
        queries = decoded_by_row.get(row_id)
        if not queries:
            logger.warning("row %d: no decoded FASTA entries; skipping", row_id)
            continue

        src_gid = gid_str(row.get("genome_id_source", ""))
        tgt_gid = gid_str(row.get("genome_id_target", ""))
        src_seq = str(row.get("seq_source", ""))
        tgt_seq = str(row.get("seq_target", ""))
        lin_src = resolve_lineage_for_gid(
            src_gid,
            row.get("lineage_source") if "lineage_source" in row.index else None,
            lineage_map,
        )
        lin_tgt = resolve_lineage_for_gid(
            tgt_gid,
            row.get("lineage_target") if "lineage_target" in row.index else None,
            lineage_map,
        )
        if "lineage_source_resolved" in row.index and str(row.get("lineage_source_resolved", "")):
            lin_src = str(row.get("lineage_source_resolved", "")) or lin_src
        if "lineage_target_resolved" in row.index and str(row.get("lineage_target_resolved", "")):
            lin_tgt = str(row.get("lineage_target_resolved", "")) or lin_tgt

        if include_source and src_seq:
            points.append(
                CloudPoint(
                    row_id=row_id,
                    point_type="source",
                    query_id="source",
                    slot=None,
                    genome_id_source=src_gid,
                    genome_id_target=tgt_gid,
                    lineage_source=lin_src,
                    lineage_target=lin_tgt,
                    sequence=src_seq,
                    label=f"row{row_id}_source",
                )
            )

        if tgt_seq:
            points.append(
                CloudPoint(
                    row_id=row_id,
                    point_type="gt",
                    query_id="gt",
                    slot=None,
                    genome_id_source=src_gid,
                    genome_id_target=tgt_gid,
                    lineage_source=lin_src,
                    lineage_target=lin_tgt,
                    sequence=tgt_seq,
                    label=f"row{row_id}_gt_{tgt_gid}",
                )
            )

        for qid, meta in sorted(queries.items()):
            seq = str(meta.get("sequence", ""))
            if not seq:
                continue
            slot = meta.get("slot")
            try:
                slot_i = int(slot) if slot is not None else None
            except (TypeError, ValueError):
                slot_i = None
            if qid == "baseline" or (slot_i is not None and slot_i < 0):
                ptype = "baseline"
            else:
                ptype = "slot"
            points.append(
                CloudPoint(
                    row_id=row_id,
                    point_type=ptype,
                    query_id=qid,
                    slot=slot_i,
                    genome_id_source=src_gid,
                    genome_id_target=tgt_gid,
                    lineage_source=lin_src,
                    lineage_target=lin_tgt,
                    sequence=seq,
                    label=f"row{row_id}_{qid}",
                )
            )
    return points


def _l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na <= 0 or nb <= 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _pairwise_dists(emb: np.ndarray) -> np.ndarray:
    if emb.shape[0] < 2:
        return np.asarray([], dtype=np.float64)
    diff = emb[:, None, :] - emb[None, :, :]
    return np.linalg.norm(diff, axis=2)


def compute_series_overlap_metrics(
    emb_by_label: dict[str, np.ndarray],
    points: list[CloudPoint],
    *,
    coverage_radius: Optional[float] = None,
) -> list[dict[str, Any]]:
    """Per-series overlap metrics between pred clouds and GT in ESM space."""
    by_row: dict[int, list[CloudPoint]] = {}
    for p in points:
        by_row.setdefault(p.row_id, []).append(p)

    records: list[dict[str, Any]] = []
    for row_id, row_pts in sorted(by_row.items()):
        gt = next((p for p in row_pts if p.point_type == "gt"), None)
        baseline = next((p for p in row_pts if p.point_type == "baseline"), None)
        slots = [p for p in row_pts if p.point_type == "slot"]
        if gt is None:
            continue
        gt_emb = emb_by_label.get(gt.label)
        if gt_emb is None:
            continue

        slot_embs = np.stack(
            [emb_by_label[p.label] for p in slots if p.label in emb_by_label],
            axis=0,
        ) if slots else np.zeros((0, EMBED_DIM), dtype=np.float32)

        all_pred_pts = [p for p in row_pts if p.point_type in {"baseline", "slot"}]
        all_pred_embs = np.stack(
            [emb_by_label[p.label] for p in all_pred_pts if p.label in emb_by_label],
            axis=0,
        ) if all_pred_pts else np.zeros((0, EMBED_DIM), dtype=np.float32)

        rec: dict[str, Any] = {
            "row": row_id,
            "genome_id_source": gt.genome_id_source,
            "genome_id_target": gt.genome_id_target,
            "lineage_source": gt.lineage_source,
            "lineage_target": gt.lineage_target,
            "n_slots": len(slots),
            "n_pred_points": len(all_pred_pts),
        }

        if baseline is not None and baseline.label in emb_by_label:
            b_emb = emb_by_label[baseline.label]
            rec["baseline_to_gt_l2"] = _l2(b_emb, gt_emb)
            rec["baseline_to_gt_cosine"] = _cosine(b_emb, gt_emb)

        if slot_embs.shape[0] > 0:
            slot_dists = np.linalg.norm(slot_embs - gt_emb.reshape(1, -1), axis=1)
            rec["slot_min_to_gt_l2"] = float(np.min(slot_dists))
            rec["slot_mean_to_gt_l2"] = float(np.mean(slot_dists))
            rec["slot_median_to_gt_l2"] = float(np.median(slot_dists))
            rec["slot_max_to_gt_l2"] = float(np.max(slot_dists))
            if coverage_radius is not None and coverage_radius > 0:
                rec["slot_coverage_frac"] = float(
                    np.mean(slot_dists <= coverage_radius)
                )
                rec["n_slots_within_radius"] = int(np.sum(slot_dists <= coverage_radius))
            if slot_embs.shape[0] > 1:
                pw = _pairwise_dists(slot_embs)
                tri = pw[np.triu_indices(slot_embs.shape[0], k=1)]
                rec["slot_cloud_spread_mean"] = float(np.mean(tri)) if tri.size else None
        else:
            rec["slot_min_to_gt_l2"] = None

        if all_pred_embs.shape[0] > 0:
            pred_dists = np.linalg.norm(all_pred_embs - gt_emb.reshape(1, -1), axis=1)
            rec["pred_min_to_gt_l2"] = float(np.min(pred_dists))
            rec["pred_mean_to_gt_l2"] = float(np.mean(pred_dists))
            centroid = np.mean(all_pred_embs, axis=0)
            rec["pred_centroid_to_gt_l2"] = _l2(centroid, gt_emb)
            if all_pred_embs.shape[0] > 1:
                pw = _pairwise_dists(all_pred_embs)
                tri = pw[np.triu_indices(all_pred_embs.shape[0], k=1)]
                rec["pred_cloud_spread_mean"] = float(np.mean(tri)) if tri.size else None

        records.append(rec)
    return records


def summarize_overlap_metrics(per_series: pd.DataFrame) -> pd.DataFrame:
    """Cohort-level summary of numeric overlap metrics."""
    if per_series.empty:
        return pd.DataFrame()
    numeric = per_series.select_dtypes(include=[np.number])
    rows: list[dict[str, Any]] = []
    for col in numeric.columns:
        if col in {"row", "n_slots", "n_pred_points", "n_slots_within_radius"}:
            continue
        s = numeric[col].dropna()
        if s.empty:
            continue
        rows.append(
            {
                "metric": col,
                "n": int(s.shape[0]),
                "mean": float(s.mean()),
                "median": float(s.median()),
                "p25": float(s.quantile(0.25)),
                "p75": float(s.quantile(0.75)),
            }
        )
    return pd.DataFrame(rows)


def build_cohort_consensus_fasta_lines(
    points: list[CloudPoint],
    *,
    n_series: int,
) -> list[str]:
    """FASTA records: targets consensus, baseline consensus, full pred-cloud consensus."""
    target_seqs = [p.sequence for p in points if p.point_type == "gt" and p.sequence]
    baseline_seqs = [p.sequence for p in points if p.point_type == "baseline" and p.sequence]
    pred_seqs = [
        p.sequence
        for p in points
        if p.point_type in {"baseline", "slot"} and p.sequence
    ]

    lines: list[str] = []
    targets_consensus = build_pred_cloud_consensus(target_seqs)
    if targets_consensus:
        lines.append(
            fasta_header(
                "targets_consensus",
                "",
                extras=[f"n_series={n_series}", f"n_seqs={len(target_seqs)}"],
            )
        )
        lines.append(targets_consensus)

    baseline_consensus = build_pred_cloud_consensus(baseline_seqs)
    if baseline_consensus:
        lines.append(
            fasta_header(
                "baseline_preds_consensus",
                "",
                extras=[f"n_series={n_series}", f"n_seqs={len(baseline_seqs)}"],
            )
        )
        lines.append(baseline_consensus)

    cloud_consensus = build_pred_cloud_consensus(pred_seqs)
    if cloud_consensus:
        lines.append(
            fasta_header(
                "pred_cloud_consensus",
                "",
                extras=[
                    f"n_series={n_series}",
                    f"n_preds={len(pred_seqs)}",
                    f"n_baselines={len(baseline_seqs)}",
                    f"n_slots={len(pred_seqs) - len(baseline_seqs)}",
                ],
            )
        )
        lines.append(cloud_consensus)
    return lines


def compute_optimistic_cross_series_metrics(
    emb_by_label: dict[str, np.ndarray],
    points: list[CloudPoint],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-target min L2 to the best-matching prediction from the full cohort cloud.

    Optimistic assignment: any baseline or slot from any series may cover any target.
    """
    gt_pts = [p for p in points if p.point_type == "gt" and p.label in emb_by_label]
    pred_pts = [
        p
        for p in points
        if p.point_type in {"baseline", "slot"} and p.label in emb_by_label
    ]
    if not gt_pts or not pred_pts:
        return pd.DataFrame(), pd.DataFrame()

    pred_stack = np.stack([emb_by_label[p.label] for p in pred_pts], axis=0)
    records: list[dict[str, Any]] = []
    for gt in gt_pts:
        gt_emb = emb_by_label[gt.label]
        dists = np.linalg.norm(pred_stack - gt_emb.reshape(1, -1), axis=1)
        best_i = int(np.argmin(dists))
        best_p = pred_pts[best_i]
        records.append(
            {
                "row": gt.row_id,
                "genome_id_source": gt.genome_id_source,
                "genome_id_target": gt.genome_id_target,
                "lineage_source": gt.lineage_source,
                "lineage_target": gt.lineage_target,
                "best_pred_label": best_p.label,
                "best_pred_row": best_p.row_id,
                "best_pred_query_id": best_p.query_id,
                "best_pred_type": best_p.point_type,
                "best_pred_slot": best_p.slot,
                "cross_series_match": bool(best_p.row_id != gt.row_id),
                "optimistic_min_l2": float(dists[best_i]),
                "optimistic_best_cosine": _cosine(pred_stack[best_i], gt_emb),
            }
        )

    per_target = pd.DataFrame(records)
    cohort = summarize_optimistic_metrics(per_target, n_pred_pool=len(pred_pts))
    return per_target, cohort


def summarize_optimistic_metrics(
    per_target: pd.DataFrame,
    *,
    n_pred_pool: int,
) -> pd.DataFrame:
    """Cohort summary of optimistic cross-series nearest-prediction distances."""
    if per_target.empty or "optimistic_min_l2" not in per_target.columns:
        return pd.DataFrame()
    s = per_target["optimistic_min_l2"].dropna()
    if s.empty:
        return pd.DataFrame()
    cross = per_target["cross_series_match"].astype(bool)
    rows = [
        {
            "metric": "optimistic_min_l2",
            "n_targets": int(s.shape[0]),
            "n_pred_pool": int(n_pred_pool),
            "mean": float(s.mean()),
            "median": float(s.median()),
            "p05": float(s.quantile(0.05)),
            "p25": float(s.quantile(0.25)),
            "p75": float(s.quantile(0.75)),
            "p95": float(s.quantile(0.95)),
            "min": float(s.min()),
            "max": float(s.max()),
            "n_cross_series_best": int(cross.sum()),
            "frac_cross_series_best": float(cross.mean()),
        }
    ]
    if "optimistic_best_cosine" in per_target.columns:
        cos = per_target["optimistic_best_cosine"].dropna()
        if not cos.empty:
            rows.append(
                {
                    "metric": "optimistic_best_cosine",
                    "n_targets": int(cos.shape[0]),
                    "n_pred_pool": int(n_pred_pool),
                    "mean": float(cos.mean()),
                    "median": float(cos.median()),
                    "p05": float(cos.quantile(0.05)),
                    "p25": float(cos.quantile(0.25)),
                    "p75": float(cos.quantile(0.75)),
                    "p95": float(cos.quantile(0.95)),
                    "min": float(cos.min()),
                    "max": float(cos.max()),
                    "n_cross_series_best": int(cross.sum()),
                    "frac_cross_series_best": float(cross.mean()),
                }
            )
    return pd.DataFrame(rows)


def _estimate_coverage_radius(bg_emb: np.ndarray, quantile: float = 0.05) -> float:
    """Heuristic neighborhood radius from background pairwise distances."""
    if bg_emb.shape[0] < 20:
        return 5.0
    rng = np.random.default_rng(42)
    idx = rng.choice(bg_emb.shape[0], size=min(500, bg_emb.shape[0]), replace=False)
    sub = bg_emb[idx]
    pw = _pairwise_dists(sub)
    tri = pw[np.triu_indices(sub.shape[0], k=1)]
    if tri.size == 0:
        return 5.0
    return float(np.quantile(tri, quantile))


def encode_sequences_esm(
    items: Sequence[tuple[str, str]],
    *,
    gpu: int = 0,
    layer: int = ESM_LAYER,
    seq_length: int = DEFAULT_SEQ_LENGTH,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> dict[str, np.ndarray]:
    """Encode (label, sequence) pairs with ESM1B mean pooling at ``layer``."""
    if not items:
        return {}

    import esm

    labels = [lbl for lbl, _ in items]
    seqs = [seq.upper().replace(".", "-") for _, seq in items]

    model, alphabet = ModelFactory.load_model("ESM1B")
    model.eval()
    use_cuda = gpu >= 0 and torch.cuda.is_available()
    device = f"cuda:{gpu}" if use_cuda else "cpu"
    if use_cuda:
        model = model.to(device)

    dataset = esm.data.FastaBatchedDataset(labels, seqs)
    encoder = ESMProteinEncoder(model, truncate_seq_length=seq_length)
    encoder.load_data(dataset, alphabet, token_budget=token_budget)

    out: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for batch_idx, (batch_labels, strs, toks) in enumerate(encoder.data_loader):
            if use_cuda:
                toks = toks.to(device=device, non_blocking=True)
            repr_out = model(toks, repr_layers=[layer], return_contacts=False)
            repr_cpu = repr_out["representations"][layer].detach().cpu()
            for i, label in enumerate(batch_labels):
                truncate_len = min(seq_length, len(strs[i]))
                pooled = encoder._pool(repr_cpu[i], "average_token", truncate_len)
                out[str(label)] = pooled.numpy().astype(np.float32, copy=False)
            del repr_out, repr_cpu, toks
            if use_cuda and batch_idx % 5 == 0:
                torch.cuda.empty_cache()
    return out


def resolve_embeddings(
    points: list[CloudPoint],
    *,
    encodings_dir: Optional[Path],
    layer: int = ESM_LAYER,
    gpu: int = 0,
    encode_missing: bool = True,
) -> dict[str, np.ndarray]:
    """Load GT/source from precomputed .pt when possible; encode the rest."""
    emb: dict[str, np.ndarray] = {}
    gid_index: dict[str, Path] = {}
    if encodings_dir is not None:
        pt_dir = Path(encodings_dir)
        if pt_dir.is_dir():
            gid_index = build_gid_pt_index(pt_dir)

    to_encode: list[tuple[str, str]] = []
    for p in points:
        loaded = False
        if p.point_type in {"gt", "source"} and gid_index:
            gid = p.genome_id_target if p.point_type == "gt" else p.genome_id_source
            key = resolve_genome_id(gid, gid_index) or gid
            fpath = gid_index.get(key)
            if fpath is not None:
                try:
                    emb[p.label] = load_raw_pooled_pt(fpath, layer=layer)
                    loaded = True
                except Exception as exc:
                    logger.warning("Failed loading %s: %s", fpath, exc)
        if not loaded:
            to_encode.append((p.label, p.sequence))

    if to_encode and encode_missing:
        logger.info("Encoding %d sequences with ESM1B layer %d", len(to_encode), layer)
        emb.update(encode_sequences_esm(to_encode, gpu=gpu, layer=layer))
    elif to_encode:
        raise ValueError(f"{len(to_encode)} sequences lack encodings and encode_missing=False")

    return emb


def _row_indices_by_type(
    fg_labels: list[str],
    label_to_point: dict[str, CloudPoint],
) -> dict[int, dict[str, Any]]:
    """Map series row → point indices in ``fg_xy`` by point type."""
    by_row: dict[int, dict[str, Any]] = {}
    for i, lbl in enumerate(fg_labels):
        p = label_to_point[lbl]
        bucket = by_row.setdefault(p.row_id, {})
        if p.point_type == "slot":
            bucket.setdefault("slots", []).append(i)
        else:
            bucket[p.point_type] = i
    return by_row


def _add_connector_trace(
    fig: go.Figure,
    fg_xy: np.ndarray,
    segments: list[tuple[int, int]],
    *,
    name: str,
    color: str,
    width: int = 1,
    visible: str = "legendonly",
) -> None:
    """Add one legend-toggleable polyline trace (hidden by default)."""
    if not segments:
        return
    xs: list[Optional[float]] = []
    ys: list[Optional[float]] = []
    for i0, i1 in segments:
        xs.extend([float(fg_xy[i0, 0]), float(fg_xy[i1, 0]), None])
        ys.extend([float(fg_xy[i0, 1]), float(fg_xy[i1, 1]), None])
    fig.add_trace(
        go.Scattergl(
            x=xs,
            y=ys,
            mode="lines",
            name=name,
            line=dict(color=color, width=width),
            visible=visible,
            hoverinfo="skip",
        )
    )


def _add_series_connectors(
    fig: go.Figure,
    fg_xy: np.ndarray,
    fg_labels: list[str],
    label_to_point: dict[str, CloudPoint],
    *,
    show_on_load: bool = False,
) -> None:
    """Legend-toggleable connector traces (off by default unless show_on_load)."""
    visible = True if show_on_load else "legendonly"
    by_row = _row_indices_by_type(fg_labels, label_to_point)

    src_tgt: list[tuple[int, int]] = []
    tgt_baseline: list[tuple[int, int]] = []
    gt_slot: list[tuple[int, int]] = []

    for _row_id, idxs in by_row.items():
        gt_i = idxs.get("gt")
        src_i = idxs.get("source")
        base_i = idxs.get("baseline")
        slot_is: list[int] = idxs.get("slots") or []

        if gt_i is not None and src_i is not None:
            src_tgt.append((src_i, gt_i))
        if gt_i is not None and base_i is not None:
            tgt_baseline.append((gt_i, base_i))
        if gt_i is not None and slot_is:
            gt_xy = fg_xy[gt_i]
            slot_xy = fg_xy[slot_is]
            dists = np.linalg.norm(slot_xy - gt_xy.reshape(1, -1), axis=1)
            gt_slot.append((gt_i, slot_is[int(np.argmin(dists))]))

    _add_connector_trace(
        fig,
        fg_xy,
        src_tgt,
        name="Source → target",
        color="rgba(148,103,189,0.45)",
        visible=visible,
    )
    _add_connector_trace(
        fig,
        fg_xy,
        tgt_baseline,
        name="Target → baseline pred",
        color="rgba(31,119,180,0.45)",
        visible=visible,
    )
    _add_connector_trace(
        fig,
        fg_xy,
        gt_slot,
        name="Target → nearest slot",
        color="rgba(214,39,40,0.35)",
        visible=visible,
    )


def build_esm_cloud_umap_figure(
    *,
    bg_emb: np.ndarray,
    bg_lineages: list[str],
    points: list[CloudPoint],
    emb_by_label: dict[str, np.ndarray],
    umap_metric: str = "cosine",
    random_state: int = 42,
    title: str = "ESM UMAP: prefix-swap pred clouds vs GT",
    fig_width: int = 1500,
    fig_height: int = 1000,
    show_lineage_labels: bool = True,
    min_cluster_size: int = 20,
    show_connectors_on_load: bool = False,
) -> go.Figure:
    """Plotly UMAP: background MSA + GT + baseline + slot prediction clouds."""
    fg_labels = [p.label for p in points if p.label in emb_by_label]
    fg_emb = np.stack([emb_by_label[lbl] for lbl in fg_labels], axis=0).astype(np.float64)
    bg = np.asarray(bg_emb, dtype=np.float64)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(30, max(2, bg.shape[0] - 1)),
        min_dist=0.1,
        metric=umap_metric,
        random_state=random_state,
    )
    reducer.fit(bg)
    bg_xy = _umap_transform_rows(reducer, bg)
    fg_xy = _umap_transform_rows(reducer, fg_emb)

    label_to_point = {p.label: p for p in points}
    fig = go.Figure()

    fig.add_trace(
        go.Scattergl(
            x=bg_xy[:, 0],
            y=bg_xy[:, 1],
            mode="markers",
            name=f"MSA background (n={bg.shape[0]})",
            marker=dict(size=3, color="lightgray", opacity=0.35),
            text=bg_lineages,
            hovertemplate="lineage=%{text}<extra>background</extra>",
        )
    )

    style = {
        "gt": dict(color="#d62728", symbol="diamond", size=11, name="Ground truth"),
        "source": dict(color="#9467bd", symbol="square", size=8, name="Source"),
        "baseline": dict(color="#1f77b4", symbol="circle", size=7, name="Pred baseline"),
        "slot": dict(color="#ff7f0e", symbol="circle-open", size=6, name="Pred slots"),
    }

    for ptype, st in style.items():
        idxs = [
            i
            for i, lbl in enumerate(fg_labels)
            if label_to_point[lbl].point_type == ptype
        ]
        if not idxs:
            continue
        sub_pts = [label_to_point[fg_labels[i]] for i in idxs]
        hover = [
            (
                f"row={p.row_id}<br>type={p.point_type}<br>query={p.query_id}"
                f"<br>src={p.genome_id_source}<br>tgt={p.genome_id_target}"
                f"<br>lin_src={p.lineage_source}<br>lin_tgt={p.lineage_target}"
            )
            for p in sub_pts
        ]
        fig.add_trace(
            go.Scattergl(
                x=fg_xy[idxs, 0],
                y=fg_xy[idxs, 1],
                mode="markers",
                name=st["name"],
                marker=dict(
                    size=st["size"],
                    color=st["color"],
                    symbol=st["symbol"],
                    line=dict(width=1, color=st["color"]),
                ),
                text=hover,
                hovertemplate="%{text}<extra></extra>",
            )
        )

    _add_series_connectors(
        fig,
        fg_xy,
        fg_labels,
        label_to_point,
        show_on_load=show_connectors_on_load,
    )

    if show_lineage_labels and bg_lineages and len(bg_lineages) == bg_xy.shape[0]:
        cent = _lineage_label_positions(bg_xy, bg_lineages, min_cluster_size)
        if not cent.empty:
            cent_text = [
                f"{row.lineage}<br>n={int(row.size)}"
                for row in cent.itertuples(index=False)
            ]
            fig.add_trace(
                go.Scatter(
                    x=cent["U1"],
                    y=cent["U2"],
                    mode="text",
                    name="Lineage labels",
                    text=cent_text,
                    textposition="middle center",
                    textfont=dict(size=11, color="black"),
                    showlegend=False,
                    hovertemplate="<b>%{text}</b><extra></extra>",
                )
            )

    fig.update_layout(
        **_umap_series_figure_layout(
            title=title,
            fig_width=fig_width,
            fig_height=fig_height,
            n_legend_entries=9,
        )
    )
    return fig


def run_prefix_swap_esm_cloud_analysis(
    *,
    pairs_csv: Path,
    decoded_fasta: Path,
    output_dir: Path,
    encodings_dir: Optional[Path] = None,
    metadata_csv: Optional[Path] = None,
    filt: Optional[SeriesFilter] = None,
    n_background: int = 5000,
    gpu: int = 0,
    umap_metric: str = "cosine",
    include_source: bool = False,
    encode_missing: bool = True,
    coverage_quantile: float = 0.05,
    show_lineage_labels: bool = True,
    min_cluster_size: int = 20,
    show_connectors_on_load: bool = False,
) -> dict[str, Any]:
    """Full pipeline: filter → encode → metrics → UMAP → CSV/JSON/HTML."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filt = filt or SeriesFilter()

    df = read_pairs_csv(pairs_csv)
    lineage_map: Optional[dict[str, str]] = None
    meta_path = metadata_csv
    if meta_path is None and Path(METADATA_CSV_DEFAULT).is_file():
        meta_path = Path(METADATA_CSV_DEFAULT)
        logger.info("Using default metadata CSV: %s", meta_path)
    if meta_path is not None and meta_path.is_file():
        lineage_map = load_gid_lineage_map(meta_path)
        logger.info("Loaded %d genome→lineage entries from %s", len(lineage_map), meta_path)
    elif filt.source_lineage is not None or filt.target_lineage is not None:
        logger.warning(
            "Lineage filters requested but no metadata CSV found; "
            "pairs-table lineage columns may be placeholders (e.g. 'unknown')."
        )

    df_sel = filter_pairs_dataframe(df, filt=filt, lineage_map=lineage_map)
    if df_sel.empty:
        raise ValueError(_format_filter_empty_error(df, filt, lineage_map))

    decoded_by_row = parse_prefix_swap_decoded_fasta(str(decoded_fasta))
    points = collect_cloud_points(
        df_sel,
        decoded_by_row,
        lineage_map=lineage_map,
        include_source=include_source,
    )
    if not points:
        raise ValueError("No cloud points collected (decoded FASTA mismatch?)")

    emb_by_label = resolve_embeddings(
        points,
        encodings_dir=encodings_dir,
        gpu=gpu,
        encode_missing=encode_missing,
    )
    missing = [p.label for p in points if p.label not in emb_by_label]
    if missing:
        raise ValueError(f"Missing embeddings for {len(missing)} points (first: {missing[0]})")

    # Background
    if encodings_dir is None:
        raise ValueError("encodings_dir is required for MSA background sampling")
    bg_emb, _bg_gids, bg_lins = load_background_from_pt_dir(
        Path(encodings_dir),
        n_background,
        lineage_map,
        random_state=filt.random_state,
    )

    coverage_r = _estimate_coverage_radius(bg_emb, quantile=coverage_quantile)
    per_series = pd.DataFrame(
        compute_series_overlap_metrics(
            emb_by_label,
            points,
            coverage_radius=coverage_r,
        )
    )
    per_series["coverage_radius"] = coverage_r
    cohort = summarize_overlap_metrics(per_series)
    optimistic_per_target, optimistic_cohort = compute_optimistic_cross_series_metrics(
        emb_by_label,
        points,
    )

    consensus_fasta_path = output_dir / "esm_cloud_cohort_consensus.fasta"
    consensus_lines = build_cohort_consensus_fasta_lines(
        points,
        n_series=int(per_series.shape[0]),
    )
    if consensus_lines:
        consensus_fasta_path.write_text(
            "\n".join(consensus_lines) + "\n",
            encoding="utf-8",
        )
        logger.info("Wrote cohort consensus FASTA → %s", consensus_fasta_path)

    points_csv = output_dir / "esm_cloud_points.csv"
    pd.DataFrame(
        [
            {
                "label": p.label,
                "row": p.row_id,
                "point_type": p.point_type,
                "query_id": p.query_id,
                "slot": p.slot,
                "genome_id_source": p.genome_id_source,
                "genome_id_target": p.genome_id_target,
                "lineage_source": p.lineage_source,
                "lineage_target": p.lineage_target,
            }
            for p in points
        ]
    ).to_csv(points_csv, index=False)

    metrics_csv = output_dir / "esm_cloud_overlap_per_series.csv"
    per_series.to_csv(metrics_csv, index=False)
    cohort_csv = output_dir / "esm_cloud_overlap_cohort.csv"
    cohort.to_csv(cohort_csv, index=False)
    optimistic_per_target_csv = output_dir / "esm_cloud_optimistic_per_target.csv"
    optimistic_cohort_csv = output_dir / "esm_cloud_optimistic_cohort.csv"
    if not optimistic_per_target.empty:
        optimistic_per_target.to_csv(optimistic_per_target_csv, index=False)
    if not optimistic_cohort.empty:
        optimistic_cohort.to_csv(optimistic_cohort_csv, index=False)

    emb_path = output_dir / "esm_cloud_embeddings.npz"
    np.savez_compressed(
        emb_path,
        labels=np.asarray(list(emb_by_label.keys()), dtype=object),
        embeddings=np.stack(list(emb_by_label.values()), axis=0),
    )

    title_bits = [f"n_series={per_series.shape[0]}"]
    if filt.source_lineage:
        title_bits.append(f"src={filt.source_lineage}")
    if filt.target_lineage:
        title_bits.append(f"tgt={filt.target_lineage}")
    fig = build_esm_cloud_umap_figure(
        bg_emb=bg_emb,
        bg_lineages=bg_lins,
        points=points,
        emb_by_label=emb_by_label,
        umap_metric=umap_metric,
        random_state=filt.random_state,
        title="ESM cloud overlap: prefix-swap preds vs GT — " + ", ".join(title_bits),
        show_lineage_labels=show_lineage_labels,
        min_cluster_size=min_cluster_size,
        show_connectors_on_load=show_connectors_on_load,
    )
    html_path = output_dir / "esm_cloud_umap.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")

    summary = {
        "n_series": int(per_series.shape[0]),
        "n_points": len(points),
        "n_background": int(bg_emb.shape[0]),
        "coverage_radius": coverage_r,
        "output_dir": str(output_dir),
        "points_csv": str(points_csv),
        "per_series_csv": str(metrics_csv),
        "cohort_csv": str(cohort_csv),
        "cohort_consensus_fasta": str(consensus_fasta_path) if consensus_lines else None,
        "optimistic_per_target_csv": (
            str(optimistic_per_target_csv) if not optimistic_per_target.empty else None
        ),
        "optimistic_cohort_csv": (
            str(optimistic_cohort_csv) if not optimistic_cohort.empty else None
        ),
        "embeddings_npz": str(emb_path),
        "umap_html": str(html_path),
        "cohort_summary": cohort.to_dict(orient="records"),
        "optimistic_cohort_summary": optimistic_cohort.to_dict(orient="records"),
    }
    summary_path = output_dir / "esm_cloud_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
