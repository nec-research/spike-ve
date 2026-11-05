#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     plot_position_heatmaps.py
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
"""Position-level heatmaps for evaluation_report.csv.

Example MSA-style tracks for sampled rows and cohort FN spike summaries.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from recomb2027.pipelines.generate_evaluation_report import (
    extract_mutations,
    parse_fasta,
    parse_knn_analysis_fasta,
)

logger = logging.getLogger(__name__)

try:
    import plotly.graph_objects as go
except ImportError as e:  # pragma: no cover
    raise ImportError("plotly is required: pip install plotly") from e

STATUS_COLORS = {
    "TP": "#2ca02c",
    "FN": "#d62728",
    "FP": "#ff7f0e",
    "TN": "#e8e8e8",
    ".": "#ffffff",
}
# z codes for heatmap cells
Z_MATCH_SOURCE = 0
Z_TP = 1
Z_FN = 2
Z_FP = 3
Z_MISSING = 4
Z_REQUIRED_SITE = 5  # target row: source→target mutation required

STATUS_CODE = {"TN": Z_MATCH_SOURCE, "TP": Z_TP, "FN": Z_FN, "FP": Z_FP, ".": Z_MISSING}

TRACK_COLORSCALE = [
    [0.0, STATUS_COLORS["TN"]],
    [0.19, STATUS_COLORS["TN"]],
    [0.20, STATUS_COLORS["TP"]],
    [0.39, STATUS_COLORS["TP"]],
    [0.40, STATUS_COLORS["FN"]],
    [0.59, STATUS_COLORS["FN"]],
    [0.60, STATUS_COLORS["FP"]],
    [0.79, STATUS_COLORS["FP"]],
    [0.80, "#aec7e8"],  # required site on labeled-target row
    [0.99, "#aec7e8"],
    [1.0, STATUS_COLORS["."]],
]

_LAYOUT = dict(
    template="plotly_white",
    font=dict(size=11),
    margin=dict(l=70, r=30, t=70, b=60),
)


def position_labels(source: str, target: str, query: str) -> list[str]:
    """Per-column TP/FN/FP/TN labels (change-from-source vs labeled target)."""
    n = min(len(source), len(target), len(query))
    labels: list[str] = []
    for i in range(n):
        required = source[i] != target[i]
        changed = query[i] != source[i]
        if required and changed:
            labels.append("TP")
        elif required and not changed:
            labels.append("FN")
        elif (not required) and changed:
            labels.append("FP")
        else:
            labels.append("TN")
    return labels


def _parse_json_list(value: Any) -> list[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return [int(x) for x in value]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [int(x) for x in parsed]
    except json.JSONDecodeError:
        pass
    return []


def _row_id(row: pd.Series) -> int:
    if "row" in row.index and pd.notna(row.get("row")):
        return int(row["row"])
    return int(row.name)


def load_decoded_fasta(fasta_path: Optional[Path]) -> dict[int, str]:
    if not fasta_path or not Path(fasta_path).is_file():
        return {}
    return parse_fasta(str(fasta_path))


def load_row_sequences(
    row: pd.Series,
    *,
    decoded_fasta: dict[int, str],
    json_dir: Optional[Path],
) -> dict[str, str]:
    """Resolve sequences for one evaluation row."""
    rid = _row_id(row)
    decoded_col = (
        "decoded_prediction_sequence"
        if "decoded_prediction_sequence" in row.index
        else "decoded_prediction"
    )
    decoded = row.get(decoded_col, "")
    decoded = "" if pd.isna(decoded) else str(decoded)
    if not decoded and rid in decoded_fasta:
        decoded = decoded_fasta[rid]

    knn_closest = row.get("knn_closest_sequence", "")
    knn_closest = "" if pd.isna(knn_closest) else str(knn_closest)
    knn_consensus = row.get("knn_consensus_sequence", "")
    knn_consensus = "" if pd.isna(knn_consensus) else str(knn_consensus)

    if json_dir is not None:
        fasta_path = Path(json_dir) / f"knn_baseline_row{rid}.fasta"
        knn_fasta = parse_knn_analysis_fasta(fasta_path)
        if not knn_closest:
            knn_closest = knn_fasta.get("knn_0", "")
        if not knn_consensus:
            knn_consensus = knn_fasta.get("knn_consensus", "")

    consensus_col = (
        "seq_target_consensus_branches"
        if "seq_target_consensus_branches" in row.index
        else "target_consensus_sequence"
    )
    branch_consensus = row.get(consensus_col, "")
    branch_consensus = "" if pd.isna(branch_consensus) else str(branch_consensus)

    src = str(row.get("seq_source", ""))
    tgt = str(row.get("seq_target", ""))
    return {
        "source": src,
        "target": tgt,
        "decoder": decoded,
        "closest_knn": knn_closest,
        "knn_pool_consensus": knn_consensus,
        "branch_targets_consensus": branch_consensus,
    }


def select_positions(
    source: str,
    target: str,
    decoder: str,
    *,
    required_sites: list[int],
    max_positions: int,
    knn: str = "",
) -> list[int]:
    """Positions to display: required sites plus predictor FP sites."""
    positions: set[int] = set(required_sites)
    for query in (decoder, knn):
        if not query:
            continue
        labels = position_labels(source, target, query)
        for i, lab in enumerate(labels, start=1):
            if lab == "FP":
                positions.add(i)
    ordered = sorted(positions)
    if len(ordered) > max_positions:
        ordered = ordered[:max_positions]
    return ordered


def _predictor_track_defs(sequences: dict[str, str]) -> list[tuple[str, str, str, list[str]]]:
    """Return (y_label, seq, track_kind, labels) for each sequence row.

    track_kind is ``status``, ``reference``, ``reference_target``, or ``predictor``.
    """
    source = sequences["source"]
    target = sequences["target"]
    decoder = sequences.get("decoder", "")
    knn = sequences.get("closest_knn", "")
    knn_cons = sequences.get("knn_pool_consensus", "")
    branch_cons = sequences.get("branch_targets_consensus", "")
    cloud_cons = sequences.get("pred_cloud_consensus", "")

    dec_labels = position_labels(source, target, decoder) if decoder else []
    knn_labels = position_labels(source, target, knn) if knn else []
    knn_cons_labels = position_labels(source, target, knn_cons) if knn_cons else []
    branch_cons_labels = (
        position_labels(source, target, branch_cons) if branch_cons else []
    )
    cloud_cons_labels = (
        position_labels(source, target, cloud_cons) if cloud_cons else []
    )
    labels_map = sequences.get("_track_labels") or {}

    def _lbl(key: str, default: str) -> str:
        return str(labels_map.get(key, default))

    tracks: list[tuple[str, str, str, list[str]]] = [
        (_lbl("status", "decoder status (labeled target)"), "", "status", dec_labels),
        (_lbl("source", "source"), source, "reference", []),
        (_lbl("target", "target (labeled)"), target, "reference_target", []),
        (_lbl("decoder", "decoder"), decoder, "predictor", dec_labels),
        (_lbl("closest_knn", "closest kNN"), knn, "predictor", knn_labels),
    ]
    if cloud_cons:
        tracks.append(
            (
                _lbl("pred_cloud_consensus", "pred cloud consensus"),
                cloud_cons,
                "predictor",
                cloud_cons_labels,
            )
        )
    if knn_cons:
        tracks.append(
            (
                _lbl("knn_pool_consensus", "kNN pool consensus"),
                knn_cons,
                "predictor",
                knn_cons_labels,
            )
        )
    if branch_cons and branch_cons != knn_cons:
        tracks.append(
            (
                _lbl("branch_targets_consensus", "branch targets consensus"),
                branch_cons,
                "predictor",
                branch_cons_labels,
            )
        )
    elif branch_cons and not knn_cons:
        tracks.append(
            (
                _lbl("branch_targets_consensus", "branch targets consensus"),
                branch_cons,
                "predictor",
                branch_cons_labels,
            )
        )
    return tracks


def _cell_z_code(
    *,
    track_kind: str,
    labels: list[str],
    idx: int,
    source: str,
    target: str,
) -> int:
    if track_kind == "status":
        if labels and 0 <= idx < len(labels):
            return STATUS_CODE.get(labels[idx], Z_MISSING)
        return Z_MISSING
    if track_kind == "reference":
        return Z_MATCH_SOURCE
    if track_kind == "reference_target":
        if idx < len(source) and idx < len(target) and source[idx] != target[idx]:
            return Z_REQUIRED_SITE
        return Z_MATCH_SOURCE
    if track_kind == "predictor":
        if labels and 0 <= idx < len(labels):
            return STATUS_CODE.get(labels[idx], Z_MISSING)
        return Z_MISSING
    return Z_MISSING


def select_example_rows(
    df: pd.DataFrame,
    *,
    examples_per_bucket: int = 3,
) -> list[tuple[str, int]]:
    """Return (bucket_name, row_id) examples for track plots."""
    if "row" not in df.columns:
        df = df.copy()
        df["row"] = df.index

    chosen: list[tuple[str, int]] = []
    seen: set[int] = set()

    def _add(bucket: str, subset: pd.DataFrame, n: int) -> None:
        for _, r in subset.head(n).iterrows():
            rid = _row_id(r)
            if rid in seen:
                continue
            seen.add(rid)
            chosen.append((bucket, rid))

    if "decoder_fn" in df.columns:
        _add("worst_fn", df.sort_values(["decoder_fn", "decoder_tp"], ascending=[False, True]), examples_per_bucket)
    if {"decoder_tp", "decoder_fn"}.issubset(df.columns):
        best = df[(df["decoder_fn"] == 0) & (df["decoder_tp"] > 0)].sort_values(
            "decoder_tp", ascending=False
        )
        _add("best_tp", best, examples_per_bucket)
    if "recall_delta_decoder_minus_knn" in df.columns:
        _add(
            "decoder_gt_knn",
            df.sort_values("recall_delta_decoder_minus_knn", ascending=False),
            examples_per_bucket,
        )
    return chosen


def _write_fig(fig: go.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn")
    logger.info("Wrote %s", path)
    return path


def plot_example_position_track(
    row: pd.Series,
    sequences: dict[str, str],
    *,
    positions: list[int],
    bucket: str,
    out_path: Path,
) -> Path:
    """MSA-style position track for one evaluation row."""
    source = sequences["source"]
    target = sequences["target"]
    track_defs = _predictor_track_defs(sequences)

    z_rows: list[list[int]] = []
    text_rows: list[list[str]] = []
    customdata_rows: list[list[str]] = []
    y_labels: list[str] = []

    for y_label, seq, track_kind, labels in track_defs:
        y_labels.append(y_label)
        z_row: list[int] = []
        text_row: list[str] = []
        hover_row: list[str] = []
        for pos in positions:
            idx = pos - 1
            if track_kind == "status":
                if labels and 0 <= idx < len(labels):
                    lab = labels[idx]
                    z_row.append(STATUS_CODE.get(lab, Z_MISSING))
                    text_row.append(f"^{lab}" if lab in STATUS_CODE else "")
                    hover_row.append(lab)
                else:
                    z_row.append(Z_MISSING)
                    text_row.append("")
                    hover_row.append("")
            elif seq and idx < len(seq):
                aa = seq[idx]
                text_row.append(aa)
                z_row.append(
                    _cell_z_code(
                        track_kind=track_kind,
                        labels=labels,
                        idx=idx,
                        source=source,
                        target=target,
                    )
                )
                if track_kind == "predictor" and labels and 0 <= idx < len(labels):
                    hover_row.append(labels[idx])
                elif track_kind == "reference_target":
                    hover_row.append(
                        "required" if source[idx] != target[idx] else "unchanged"
                    )
                else:
                    hover_row.append("")
            else:
                z_row.append(Z_MISSING)
                text_row.append(".")
                hover_row.append("")
        z_rows.append(z_row)
        text_rows.append(text_row)
        customdata_rows.append(hover_row)

    fig = go.Figure(
        data=go.Heatmap(
            z=z_rows,
            x=[str(p) for p in positions],
            y=y_labels,
            text=text_rows,
            customdata=customdata_rows,
            texttemplate="%{text}",
            hovertemplate=(
                "Track=%{y}<br>Pos=%{x}<br>Residue=%{text}"
                "<br>Status=%{customdata}<extra></extra>"
            ),
            colorscale=TRACK_COLORSCALE,
            zmin=0,
            zmax=5,
            showscale=False,
        )
    )
    rid = _row_id(row)
    fig.update_layout(
        **_LAYOUT,
        title=(
            f"Position track — row {rid} ({bucket})<br>"
            "<sup>Colors on predictor rows: green=TP, red=FN, orange=FP, grey=TN. "
            "kNN pool consensus = majority vote across retrieved kNN neighbors; "
            "branch targets consensus = agreement across labeled branch targets "
            "sharing the same prefix (X where they disagree).</sup>"
        ),
        xaxis_title="MSA position (1-based)",
        yaxis=dict(autorange="reversed"),
        height=max(360, 55 * len(y_labels) + 140),
        width=max(700, 28 * len(positions) + 200),
    )
    return _write_fig(fig, out_path)


def build_cohort_position_table(
    df: pd.DataFrame,
    *,
    decoded_fasta: dict[int, str],
    json_dir: Optional[Path],
) -> pd.DataFrame:
    """Long table: one record per (row, position, method, label)."""
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        seqs = load_row_sequences(row, decoded_fasta=decoded_fasta, json_dir=json_dir)
        source, target = seqs["source"], seqs["target"]
        if not source or not target:
            continue
        required = _parse_json_list(row.get("required_mutation_sites"))
        if not required:
            required, _ = extract_mutations(source, target)

        for method, query_key in [("decoder", "decoder"), ("closest_knn", "closest_knn")]:
            query = seqs.get(query_key, "")
            if not query:
                continue
            labels = position_labels(source, target, query)
            for pos in required:
                idx = pos - 1
                if idx < 0 or idx >= len(labels):
                    continue
                records.append(
                    {
                        "row": _row_id(row),
                        "pos": int(pos),
                        "method": method,
                        "label": labels[idx],
                    }
                )
    return pd.DataFrame(records)


def plot_cohort_fn_spikes(
    long_df: pd.DataFrame,
    out_dir: Path,
    *,
    top_k: int = 40,
) -> Optional[Path]:
    """Top positions by decoder FN count across the cohort."""
    if long_df.empty:
        return None

    fn = long_df[long_df["label"] == "FN"]
    if fn.empty:
        return None

    counts = (
        fn.groupby(["pos", "method"])
        .size()
        .reset_index(name="count")
    )
    top_positions = (
        counts[counts["method"] == "decoder"]
        .sort_values("count", ascending=False)
        .head(top_k)["pos"]
        .tolist()
    )
    if not top_positions:
        return None

    sub = counts[counts["pos"].isin(top_positions)]
    pivot = sub.pivot(index="pos", columns="method", values="count").fillna(0)
    pivot = pivot.reindex(top_positions)

    fig = go.Figure()
    for method, color in [("decoder", "#1f77b4"), ("closest_knn", "#ff7f0e")]:
        if method not in pivot.columns:
            continue
        fig.add_trace(
            go.Bar(
                x=[str(int(p)) for p in pivot.index],
                y=pivot[method].values,
                name=method,
                marker_color=color,
            )
        )

    fig.update_layout(
        **_LAYOUT,
        title=f"Top {len(top_positions)} positions by decoder FN count",
        xaxis_title="MSA position (1-based)",
        yaxis_title="FN count (required sites only)",
        barmode="group",
        height=480,
        width=max(800, 20 * len(top_positions) + 200),
    )
    return _write_fig(fig, out_dir / "10_cohort_position_fn_spikes.html")


def plot_cohort_position_rates(
    long_df: pd.DataFrame,
    out_dir: Path,
    *,
    top_k: int = 40,
) -> Optional[Path]:
    """FN rate per position (FN / series where position is required)."""
    if long_df.empty:
        return None

    req = long_df[long_df["label"].isin(["TP", "FN"])]
    if req.empty:
        return None

    totals = req.groupby(["pos", "method"]).size().reset_index(name="n_required")
    fn = (
        req[req["label"] == "FN"]
        .groupby(["pos", "method"])
        .size()
        .reset_index(name="fn")
    )
    merged = totals.merge(fn, on=["pos", "method"], how="left").fillna({"fn": 0})
    merged["fn_rate"] = merged["fn"] / merged["n_required"]

    top_positions = (
        merged[merged["method"] == "decoder"]
        .sort_values("fn", ascending=False)
        .head(top_k)["pos"]
        .tolist()
    )
    if not top_positions:
        return None

    sub = merged[merged["pos"].isin(top_positions)]
    fig = go.Figure()
    for method, color in [("decoder", "#1f77b4"), ("closest_knn", "#ff7f0e")]:
        msub = sub[sub["method"] == method].sort_values("pos")
        if msub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=[str(int(p)) for p in msub["pos"]],
                y=msub["fn_rate"],
                mode="lines+markers",
                name=f"{method} FN rate",
                line=dict(color=color),
            )
        )

    fig.update_layout(
        **_LAYOUT,
        title=f"FN rate at top decoder-FN positions (top {len(top_positions)})",
        xaxis_title="MSA position (1-based)",
        yaxis=dict(title="FN rate", range=[0, 1]),
        height=460,
        width=max(800, 20 * len(top_positions) + 200),
    )
    return _write_fig(fig, out_dir / "10b_cohort_position_fn_rates.html")


def plot_position_heatmaps(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    decoded_fasta_path: Optional[Path] = None,
    json_dir: Optional[Path] = None,
    examples_per_bucket: int = 3,
    max_positions: int = 60,
    top_k_positions: int = 40,
) -> list[Path]:
    """Generate example tracks and cohort position summaries."""
    out_dir = Path(out_dir)
    decoded_fasta = load_decoded_fasta(decoded_fasta_path)
    json_dir = Path(json_dir) if json_dir else None

    paths: list[Path] = []
    examples = select_example_rows(df, examples_per_bucket=examples_per_bucket)
    row_by_id = {_row_id(r): r for _, r in df.iterrows()}

    for bucket, rid in examples:
        row = row_by_id.get(rid)
        if row is None:
            continue
        seqs = load_row_sequences(row, decoded_fasta=decoded_fasta, json_dir=json_dir)
        if not seqs["source"] or not seqs["target"]:
            continue
        required = _parse_json_list(row.get("required_mutation_sites"))
        positions = select_positions(
            seqs["source"],
            seqs["target"],
            seqs.get("decoder", ""),
            required_sites=required,
            max_positions=max_positions,
            knn=seqs.get("closest_knn", ""),
        )
        if not positions:
            continue
        out_path = out_dir / f"09_position_track_row{rid}_{bucket}.html"
        paths.append(
            plot_example_position_track(
                row,
                seqs,
                positions=positions,
                bucket=bucket,
                out_path=out_path,
            )
        )

    long_df = build_cohort_position_table(
        df, decoded_fasta=decoded_fasta, json_dir=json_dir
    )
    if not long_df.empty:
        csv_path = out_dir / "position_level_long.csv"
        long_df.to_csv(csv_path, index=False)
        logger.info("Wrote %s", csv_path)

    spike = plot_cohort_fn_spikes(long_df, out_dir, top_k=top_k_positions)
    if spike is not None:
        paths.append(spike)
    rates = plot_cohort_position_rates(long_df, out_dir, top_k=top_k_positions)
    if rates is not None:
        paths.append(rates)

    return paths
