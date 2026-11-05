#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     plot_prefix_swap_evaluation_report.py
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
"""Visual analytics for prefix-swap evaluation CSVs.

Produces interactive Plotly HTML summaries and an index page with cohort statistics
focused on baseline vs prefix-swap cloud metrics and optional kNN / ESM overlap data.

Example::

    PYTHONPATH=src python -m recomb2027.pipelines.plot_prefix_swap_evaluation_report \\
        --eval-dir .../evaluation_prefix_swap \\
        --decoded-fasta .../b3_lineage_subst_prefix_swap_test_decoded.fasta
"""
from __future__ import annotations

import argparse
import html as html_module
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from recomb2027.pipelines.plot_evaluation_report import (
    GT_EDIT_BIN_EDGES,
    GT_EDIT_BIN_LABELS,
    GT_EDIT_BIN_LEGEND,
    _axis_range,
    _recall,
    _safe_div,
    _setup_logging,
    _write_fig,
    plot_distance_vs_recall,
    plot_stratified_recall,
)

from recomb2027.pipelines.plot_prefix_swap_position_heatmaps import (
    plot_prefix_swap_position_heatmaps,
)

logger = logging.getLogger(__name__)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as e:  # pragma: no cover
    raise ImportError("plotly is required: pip install plotly") from e

_LAYOUT_BASE = dict(
    template="plotly_white",
    font=dict(size=12),
    margin=dict(l=60, r=40, t=80, b=60),
)
_LAYOUT = dict(
    _LAYOUT_BASE,
    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
)

_LEGEND_OUTSIDE_RIGHT = dict(
    orientation="v",
    yanchor="top",
    y=1.0,
    xanchor="left",
    x=1.02,
)
_MARGIN_LEGEND_OUTSIDE = dict(l=60, r=150, t=80, b=60)
_LAYOUT_LEGEND_OUTSIDE = dict(
    _LAYOUT_BASE,
    legend=_LEGEND_OUTSIDE_RIGHT,
    margin=_MARGIN_LEGEND_OUTSIDE,
)

PRED_METHOD_SPECS: list[tuple[str, str, str, str, str, str]] = [
    (
        "pred_baseline",
        "Baseline prediction",
        "pred_baseline_tp",
        "pred_baseline_fn",
        "pred_baseline_recall",
        "pred_baseline_fp",
    ),
    (
        "pred_pooled_all",
        "Pooled cloud (baseline+slots)",
        "pred_pooled_all_tp",
        "pred_pooled_all_fn",
        "pred_pooled_all_recall",
        "pred_pooled_all_fp",
    ),
    (
        "pred_pooled_slots",
        "Pooled slots only",
        "pred_pooled_slots_tp",
        "pred_pooled_slots_fn",
        "pred_pooled_slots_recall",
        "pred_pooled_slots_fp",
    ),
    (
        "pred_best_slot",
        "Best slot (position recall)",
        "pred_best_slot_tp",
        "pred_best_slot_fn",
        "pred_best_slot_recall",
        "pred_best_slot_fp",
    ),
    (
        "pred_consensus_all",
        "Cloud consensus (X = any disagreement; equals pooled cloud)",
        "pred_consensus_all_tp",
        "pred_consensus_all_fn",
        "pred_consensus_all_recall",
        "pred_consensus_all_fp",
    ),
]

KNN_METHOD_SPECS: list[tuple[str, str, str, str, str, str]] = [
    (
        "knn_baseline",
        "kNN neighbors (baseline)",
        "knn_baseline_change_tp",
        "knn_baseline_change_fn",
        "knn_baseline_change_recall",
        "knn_baseline_change_fp",
    ),
    (
        "knn_slots",
        "kNN neighbors (slots)",
        "knn_slots_change_tp",
        "knn_slots_change_fn",
        "knn_slots_change_recall",
        "knn_slots_change_fp",
    ),
    (
        "knn_pooled",
        "kNN neighbors (pooled)",
        "knn_pooled_change_tp",
        "knn_pooled_change_fn",
        "knn_pooled_change_recall",
        "knn_pooled_change_fp",
    ),
]


def _to_float0(value: Any) -> float:
    """Coerce to float; NaN/None/missing → 0."""
    v = pd.to_numeric(value, errors="coerce")
    if pd.isna(v):
        return 0.0
    return float(v)


def _methods_from_specs(
    df: pd.DataFrame,
    specs: list[tuple[str, str, str, str, str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method_id, label, tp_c, fn_c, recall_c, fp_c in specs:
        if not {tp_c, fn_c}.issubset(df.columns):
            continue
        tp = float(pd.to_numeric(df[tp_c], errors="coerce").fillna(0).sum())
        fn = float(pd.to_numeric(df[fn_c], errors="coerce").fillna(0).sum())
        fp = (
            float(pd.to_numeric(df[fp_c], errors="coerce").fillna(0).sum())
            if fp_c in df.columns
            else 0.0
        )
        recall = (
            float(df[recall_c].mean())
            if recall_c in df.columns
            else _safe_div(tp, tp + fn)
        )
        rows.append(
            {
                "method": method_id,
                "label": label,
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "recall": recall,
                "n_rows": int(df[[tp_c, fn_c]].notna().all(axis=1).sum()),
            }
        )
    return rows


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived recall and difficulty columns used across plots."""
    out = df.copy()
    numeric_cols = [
        "gt_hamming_source_target",
        "blosum_distance_source_to_target",
        "blast_identity_source_to_target",
        "blast_similarity_source_to_target",
        "n_branch_targets_same_prefix",
        "n_branch_variable_sites_between_targets",
        "n_pred_queries",
        "n_pred_unique_sequences",
        "n_pred_variable_sites",
        "pred_pairwise_hamming_mean",
        "pred_per_site_entropy_mean",
        "pred_pooled_all_vs_baseline_delta_recall",
        "pred_pooled_slots_vs_baseline_delta_recall",
        "knn_pooled_vs_baseline_delta_recall",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for _mid, _label, tp_c, fn_c, recall_c, _fp_c in PRED_METHOD_SPECS + KNN_METHOD_SPECS:
        if {tp_c, fn_c}.issubset(out.columns) and recall_c not in out.columns:
            out[recall_c] = _recall(
                pd.to_numeric(out[tp_c], errors="coerce"),
                pd.to_numeric(out[fn_c], errors="coerce"),
            )

    if {"pred_best_slot_recall", "pred_baseline_recall"}.issubset(out.columns):
        out["pred_best_slot_vs_baseline_delta_recall"] = (
            out["pred_best_slot_recall"] - out["pred_baseline_recall"]
        )
    if {"pred_pooled_all_recall", "knn_pooled_change_recall"}.issubset(out.columns):
        out["pred_pooled_all_vs_knn_pooled_delta_recall"] = (
            out["pred_pooled_all_recall"] - out["knn_pooled_change_recall"]
        )

    if "gt_hamming_source_target" in out.columns:
        out["gt_edit_bin"] = pd.cut(
            out["gt_hamming_source_target"],
            bins=GT_EDIT_BIN_EDGES,
            labels=GT_EDIT_BIN_LABELS,
            right=True,
        )
    return out


def _common_cohort(
    df: pd.DataFrame, specs: list[tuple[str, str, str, str, str, str]]
) -> pd.DataFrame:
    """Rows where TP and FN exist for every method in ``specs`` (present columns).

    Methods are not defined for every series (e.g. slot-based methods need at
    least one prefix-swap slot, kNN needs a mutation-stats JSON). Summing or
    averaging each method over its own rows compares different cohorts and
    turns missing values into zeros in the per-series violins.
    """
    cols = [c for _m, _l, tp, fn, _r, _f in specs for c in (tp, fn) if c in df.columns]
    return df.dropna(subset=cols) if cols else df


def cohort_summary(df: pd.DataFrame) -> dict[str, Any]:
    pred_df = _common_cohort(df, PRED_METHOD_SPECS)
    knn_df = _common_cohort(df, KNN_METHOD_SPECS)
    pred_methods = _methods_from_specs(pred_df, PRED_METHOD_SPECS)
    knn_methods = _methods_from_specs(knn_df, KNN_METHOD_SPECS)
    summary: dict[str, Any] = {
        "n_series": int(len(df)),
        "n_series_pred_common": int(len(pred_df)),
        "n_series_knn_common": int(len(knn_df)) if knn_methods else 0,
        "pred_methods": pred_methods,
        "knn_methods": knn_methods,
    }
    if {"pred_baseline_recall", "pred_best_slot_recall"}.issubset(df.columns):
        summary["best_slot_wins_recall"] = int(
            (df["pred_best_slot_recall"] > df["pred_baseline_recall"]).sum()
        )
        summary["baseline_wins_recall"] = int(
            (df["pred_baseline_recall"] > df["pred_best_slot_recall"]).sum()
        )
        summary["ties_best_slot_baseline"] = int(
            (df["pred_best_slot_recall"] == df["pred_baseline_recall"]).sum()
        )
    if {"pred_pooled_all_recall", "pred_baseline_recall"}.issubset(df.columns):
        summary["pooled_all_wins_recall"] = int(
            (df["pred_pooled_all_recall"] > df["pred_baseline_recall"]).sum()
        )
    if {"pred_pooled_all_recall", "knn_pooled_change_recall"}.issubset(df.columns):
        summary["pred_cloud_wins_vs_knn"] = int(
            (df["pred_pooled_all_recall"] > df["knn_pooled_change_recall"]).sum()
        )
        summary["knn_wins_vs_pred_cloud"] = int(
            (df["knn_pooled_change_recall"] > df["pred_pooled_all_recall"]).sum()
        )
    return summary


def _plot_cohort_tp_fn_recall(
    methods: list[dict[str, Any]],
    *,
    title: str,
    path: Path,
) -> Path:
    if not methods:
        raise ValueError("No TP/FN columns found for cohort summary plot.")

    labels = [m["label"] for m in methods]
    tp = [m["tp"] for m in methods]
    fn = [m["fn"] for m in methods]
    fp = [m.get("fp", 0) for m in methods]
    recalls = [m["recall"] for m in methods]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(name="TP", x=labels, y=tp, marker_color="#2ca02c"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(name="FN", x=labels, y=fn, marker_color="#d62728"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(name="FP", x=labels, y=fp, marker_color="#ff7f0e"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            name="Mean recall",
            x=labels,
            y=recalls,
            mode="lines+markers",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=8),
        ),
        secondary_y=True,
    )
    fig.update_layout(
        **_LAYOUT_LEGEND_OUTSIDE,
        title=title,
        barmode="group",
        height=520,
        width=max(900, 150 * len(labels)),
    )
    fig.update_yaxes(title_text="Count (sum over series)", secondary_y=False)
    fig.update_yaxes(title_text="Mean recall", range=[0, 1], secondary_y=True)
    return _write_fig(fig, path)


def _tn_column_for_fp(fp_c: str) -> str:
    if fp_c.endswith("_change_fp"):
        return fp_c.replace("_change_fp", "_change_tn")
    if fp_c.endswith("_fp"):
        return fp_c[:-3] + "_tn"
    return fp_c.replace("fp", "tn")


_METRIC_COLORS = {"TP": "#2ca02c", "FN": "#d62728", "FP": "#ff7f0e"}
_VIOLIN_GROUP_OFFSETS = {"TP": -0.2, "FN": 0.0, "FP": 0.2}


def _grouped_categorical_x(
    methods: pd.Series,
    method_order: list[str],
    metric: str,
) -> np.ndarray:
    """Numeric x positions aligned with grouped violins for one metric."""
    offset = _VIOLIN_GROUP_OFFSETS[metric]
    positions: list[float] = []
    for method in methods:
        try:
            base = float(method_order.index(method))
        except ValueError:
            base = 0.0
        positions.append(base + offset)
    return np.asarray(positions, dtype=float)


def _categorical_axis_ticks(method_order: list[str]) -> dict[str, Any]:
    return dict(
        tickmode="array",
        tickvals=list(range(len(method_order))),
        ticktext=method_order,
        tickangle=-30,
    )


def _per_series_confusion_long(
    df: pd.DataFrame,
    specs: list[tuple[str, str, str, str, str, str]],
    *,
    normalized: bool = False,
) -> pd.DataFrame:
    """Long table: one row per (series, method, metric) for violin plots."""
    records: list[dict[str, Any]] = []
    has_edit_bin = "gt_edit_bin" in df.columns
    for row_idx, row in df.iterrows():
        series_id = row.get("row", row_idx)
        edit_bin = None
        if has_edit_bin:
            raw_bin = row.get("gt_edit_bin")
            if pd.notna(raw_bin):
                edit_bin = str(raw_bin)
        for _method_id, label, tp_c, fn_c, _recall_c, fp_c in specs:
            if not {tp_c, fn_c}.issubset(df.columns):
                continue
            tp = _to_float0(row.get(tp_c))
            fn = _to_float0(row.get(fn_c))
            fp = _to_float0(row.get(fp_c)) if fp_c in df.columns else 0.0
            tn_c = _tn_column_for_fp(fp_c)
            tn = _to_float0(row.get(tn_c)) if tn_c in df.columns else 0.0
            total = tp + fn + fp + tn
            for metric, val in (("TP", tp), ("FN", fn), ("FP", fp)):
                if normalized:
                    if total <= 0:
                        continue
                    val = val / total
                records.append(
                    {
                        "row": series_id,
                        "method": label,
                        "metric": metric,
                        "value": val,
                        "gt_edit_bin": edit_bin,
                    }
                )
    return pd.DataFrame(records)


def _per_method_metric_confusion_stats(
    count_long: pd.DataFrame,
    method_order: list[str],
) -> pd.DataFrame:
    """Cohort summary stats per (method, metric) for violin overlays."""
    records: list[dict[str, Any]] = []
    for method in method_order:
        for metric in ("TP", "FN", "FP"):
            vals = count_long[
                (count_long["method"] == method) & (count_long["metric"] == metric)
            ]["value"]
            if vals.empty:
                continue
            arr = vals.to_numpy(dtype=float)
            records.append(
                {
                    "method": method,
                    "metric": metric,
                    "mean": float(np.mean(arr)),
                    "q25": float(np.quantile(arr, 0.25)),
                    "median": float(np.median(arr)),
                    "q75": float(np.quantile(arr, 0.75)),
                }
            )
    return pd.DataFrame(records)


_STAT_MARKER_STYLES: tuple[tuple[str, str, str, str, int, str, bool], ...] = (
    ("mean", "Mean", "mean", "diamond", 9, "#1a1a1a", True),
    ("median", "Median", "median", "square", 8, "#1a1a1a", True),
    ("q25", "Quantiles", "quantiles", "line-ew", 8, "#4d4d4d", True),
    ("q75", "Quantiles", "quantiles", "line-ew", 8, "#4d4d4d", False),
)


def _method_order_from_specs(
    specs: list[tuple[str, str, str, str, str, str]],
    count_long: pd.DataFrame,
) -> list[str]:
    labels = [label for _mid, label, *_ in specs]
    present = set(count_long["method"])
    return [label for label in labels if label in present]


def _add_grouped_tp_fn_fp_violin_traces(
    fig: go.Figure,
    count_long: pd.DataFrame,
    method_order: list[str],
    *,
    row: Optional[int] = None,
    col: Optional[int] = None,
    panel_id: str = "all",
    show_metric_legend: bool = True,
    show_stat_legend: bool = True,
) -> None:
    """Add grouped TP/FN/FP violins plus mean/median/quantile overlays to a panel."""
    subplot_kw: dict[str, int] = {}
    if row is not None and col is not None:
        subplot_kw = {"row": row, "col": col}

    stats = _per_method_metric_confusion_stats(count_long, method_order)
    for metric in ("TP", "FN", "FP"):
        sub = count_long[count_long["metric"] == metric]
        if not sub.empty:
            fig.add_trace(
                go.Violin(
                    x=_grouped_categorical_x(sub["method"], method_order, metric),
                    y=sub["value"],
                    name=metric,
                    legendgroup=metric,
                    scalegroup=f"{metric}_{panel_id}",
                    line_color=_METRIC_COLORS[metric],
                    fillcolor=_METRIC_COLORS[metric],
                    opacity=0.55,
                    box_visible=True,
                    meanline_visible=False,
                    points="outliers",
                    showlegend=show_metric_legend,
                    width=0.55,
                ),
                **subplot_kw,
            )
        mstats = stats[stats["metric"] == metric]
        for stat_key, stat_label, legendgroup, symbol, size, color, show_in_legend in (
            _STAT_MARKER_STYLES
        ):
            if mstats.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=_grouped_categorical_x(mstats["method"], method_order, metric),
                    y=mstats[stat_key],
                    mode="markers",
                    name=stat_label,
                    legendgroup=legendgroup,
                    marker=dict(
                        symbol=symbol,
                        size=size,
                        color=_METRIC_COLORS[metric],
                        line=dict(width=1, color=color),
                    ),
                    showlegend=show_stat_legend and show_in_legend and metric == "TP",
                ),
                **subplot_kw,
            )

    xaxis_kwargs = _categorical_axis_ticks(method_order)
    if row is not None and col is not None:
        fig.update_xaxes(**xaxis_kwargs, row=row, col=col)
        fig.update_yaxes(title_text="Site count", row=row, col=col)
    else:
        fig.update_xaxes(**xaxis_kwargs)
        fig.update_yaxes(title_text="Site count")


def _plot_per_series_tp_fn_fp_violins(
    df: pd.DataFrame,
    specs: list[tuple[str, str, str, str, str, str]],
    *,
    title: str,
    path: Path,
) -> Optional[Path]:
    """Grouped TP/FN/FP violin distributions per method on a single panel."""
    count_long = _per_series_confusion_long(df, specs, normalized=False)
    if count_long.empty:
        return None

    method_order = _method_order_from_specs(specs, count_long)
    fig = go.Figure()
    _add_grouped_tp_fn_fp_violin_traces(
        fig,
        count_long,
        method_order,
        panel_id="all",
    )

    fig.update_layout(
        **_LAYOUT_LEGEND_OUTSIDE,
        title=title,
        height=460,
        width=max(1000, 180 * max(len(method_order), 1)),
        violinmode="group",
    )
    return _write_fig(fig, path)


def _plot_per_series_tp_fn_fp_violins_by_edit_bin(
    df: pd.DataFrame,
    specs: list[tuple[str, str, str, str, str, str]],
    *,
    title: str,
    path: Path,
) -> Optional[Path]:
    """Grouped TP/FN/FP violins, one panel per GT source→target edit-distance bin."""
    if "gt_edit_bin" not in df.columns:
        return None

    count_long = _per_series_confusion_long(df, specs, normalized=False)
    if count_long.empty:
        return None

    count_long = count_long.dropna(subset=["gt_edit_bin"])
    if count_long.empty:
        return None

    method_order = _method_order_from_specs(specs, count_long)
    panel_bins: list[tuple[str, int]] = []
    for bin_label in GT_EDIT_BIN_LABELS:
        sub = count_long[count_long["gt_edit_bin"] == bin_label]
        if sub.empty:
            continue
        n_series = int(sub["row"].nunique())
        panel_bins.append((bin_label, n_series))

    if not panel_bins:
        return None

    subplot_titles = [
        f"GT source→target edits: {bin_label} (n={n_series} series)"
        for bin_label, n_series in panel_bins
    ]
    fig = make_subplots(
        rows=len(panel_bins),
        cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08,
    )
    for row_idx, (bin_label, _n_series) in enumerate(panel_bins, start=1):
        sub = count_long[count_long["gt_edit_bin"] == bin_label]
        _add_grouped_tp_fn_fp_violin_traces(
            fig,
            sub,
            method_order,
            row=row_idx,
            col=1,
            panel_id=str(bin_label),
            show_metric_legend=row_idx == 1,
            show_stat_legend=row_idx == 1,
        )

    fig.update_layout(
        **_LAYOUT_LEGEND_OUTSIDE,
        title=title,
        height=max(360, 320 * len(panel_bins)),
        width=max(1000, 180 * max(len(method_order), 1)),
        violinmode="group",
    )
    return _write_fig(fig, path)


def plot_cohort_tp_fn(
    df: pd.DataFrame,
    summary: dict[str, Any],
    out_dir: Path,
) -> list[Path]:
    paths: list[Path] = []
    pred_methods = summary.get("pred_methods", [])
    knn_methods = summary.get("knn_methods", [])
    # Same series for every method in a plot (see _common_cohort).
    pred_df = _common_cohort(df, PRED_METHOD_SPECS)
    knn_df = _common_cohort(df, KNN_METHOD_SPECS)
    if pred_methods:
        paths.append(
            _plot_cohort_tp_fn_recall(
                pred_methods,
                title="Cohort TP / FN / FP / Recall — prefix-swap predictions",
                path=out_dir / "01_cohort_tp_fn_predictions.html",
            )
        )
        v = _plot_per_series_tp_fn_fp_violins(
            pred_df,
            PRED_METHOD_SPECS,
            title=(
                "Per-series TP / FN / FP distributions — prefix-swap predictions "
                "(grouped by method; markers: mean, median, quantiles)"
            ),
            path=out_dir / "01a_per_series_tp_fn_fp_violin_predictions.html",
        )
        if v is not None:
            paths.append(v)
        v_bin = _plot_per_series_tp_fn_fp_violins_by_edit_bin(
            pred_df,
            PRED_METHOD_SPECS,
            title=(
                "Per-series TP / FN / FP by GT edit distance — prefix-swap predictions "
                "(grouped by method; markers: mean, median, quantiles)"
            ),
            path=out_dir / "01a_by_bin_per_series_tp_fn_fp_violin_predictions.html",
        )
        if v_bin is not None:
            paths.append(v_bin)
    if knn_methods:
        paths.append(
            _plot_cohort_tp_fn_recall(
                knn_methods,
                title="Cohort TP / FN / FP / Recall — kNN neighbor pools",
                path=out_dir / "01b_cohort_tp_fn_knn.html",
            )
        )
        v = _plot_per_series_tp_fn_fp_violins(
            knn_df,
            KNN_METHOD_SPECS,
            title=(
                "Per-series TP / FN / FP distributions — kNN neighbor pools "
                "(grouped by method; markers: mean, median, quantiles)"
            ),
            path=out_dir / "01b2_per_series_tp_fn_fp_violin_knn.html",
        )
        if v is not None:
            paths.append(v)
        v_bin = _plot_per_series_tp_fn_fp_violins_by_edit_bin(
            knn_df,
            KNN_METHOD_SPECS,
            title=(
                "Per-series TP / FN / FP by GT edit distance — kNN neighbor pools "
                "(grouped by method; markers: mean, median, quantiles)"
            ),
            path=out_dir / "01b2_by_bin_per_series_tp_fn_fp_violin_knn.html",
        )
        if v_bin is not None:
            paths.append(v_bin)
    return paths


def _recall_scatter(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    x_title: str,
    y_title: str,
    title: str,
    path: Path,
    color_col: Optional[str] = "gt_hamming_source_target",
) -> Optional[Path]:
    if not {x_col, y_col}.issubset(df.columns):
        return None
    plot_df = df.dropna(subset=[x_col, y_col])
    if plot_df.empty:
        return None

    color = plot_df[color_col] if color_col and color_col in plot_df.columns else None
    colorbar = None
    if color is not None:
        colorbar = dict(
            title="GT edit distance",
            x=1.12,
            xanchor="left",
        )
    fig = go.Figure(
        go.Scatter(
            x=plot_df[x_col],
            y=plot_df[y_col],
            mode="markers",
            marker=dict(
                size=7,
                opacity=0.55,
                color=color,
                colorscale="Viridis",
                showscale=color is not None,
                colorbar=colorbar,
            ),
            text=plot_df.get("row", plot_df.index),
            hovertemplate=(
                f"{x_title}=%{{x:.3f}}<br>"
                f"{y_title}=%{{y:.3f}}<br>"
                "row=%{text}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="gray"),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    layout = dict(_LAYOUT_BASE, margin=_MARGIN_LEGEND_OUTSIDE)
    fig.update_layout(
        **layout,
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        height=560,
        width=620,
    )
    return _write_fig(fig, path)


def plot_delta_recall_histogram(df: pd.DataFrame, out_dir: Path) -> Optional[Path]:
    cols = [
        ("pred_pooled_all_vs_baseline_delta_recall", "Pooled cloud − baseline"),
        ("pred_pooled_slots_vs_baseline_delta_recall", "Pooled slots − baseline"),
        ("pred_best_slot_vs_baseline_delta_recall", "Best slot − baseline"),
    ]
    present = [(c, name) for c, name in cols if c in df.columns]
    if not present:
        return None

    fig = go.Figure()
    for col, name in present:
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if vals.empty:
            continue
        fig.add_trace(go.Histogram(x=vals, name=name, opacity=0.65))
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        **_LAYOUT,
        title="Per-series recall improvement over baseline",
        xaxis_title="Δ recall vs baseline",
        yaxis_title="Series count",
        barmode="overlay",
        height=480,
        width=760,
    )
    return _write_fig(fig, out_dir / "03_delta_recall_vs_baseline.html")


def plot_cloud_diversity_vs_recall(df: pd.DataFrame, out_dir: Path) -> Optional[Path]:
    if not {"n_pred_variable_sites", "pred_best_slot_recall"}.issubset(df.columns):
        return None
    plot_df = df.dropna(subset=["n_pred_variable_sites", "pred_best_slot_recall"])
    if plot_df.empty:
        return None

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Cloud variable sites vs best-slot recall",
            "Pairwise cloud hamming vs best-slot recall",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["n_pred_variable_sites"],
            y=plot_df["pred_best_slot_recall"],
            mode="markers",
            marker=dict(size=6, opacity=0.5, color="#1f77b4"),
            name="variable sites",
        ),
        row=1,
        col=1,
    )
    if "pred_pairwise_hamming_mean" in plot_df.columns:
        sub = plot_df.dropna(subset=["pred_pairwise_hamming_mean"])
        fig.add_trace(
            go.Scatter(
                x=sub["pred_pairwise_hamming_mean"],
                y=sub["pred_best_slot_recall"],
                mode="markers",
                marker=dict(size=6, opacity=0.5, color="#ff7f0e"),
                name="pairwise hamming",
            ),
            row=1,
            col=2,
        )
    x1 = _axis_range(plot_df["n_pred_variable_sites"], clamp_lo=0)
    if "pred_pairwise_hamming_mean" in plot_df.columns:
        x2 = _axis_range(plot_df["pred_pairwise_hamming_mean"], clamp_lo=0)
        fig.update_xaxes(title_text="Mean pairwise hamming (cloud)", range=x2, row=1, col=2)
    fig.update_xaxes(title_text="Variable sites in pred cloud", range=x1, row=1, col=1)
    fig.update_yaxes(title_text="Best-slot recall", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="Best-slot recall", range=[0, 1], row=1, col=2)
    fig.update_layout(**_LAYOUT, height=500, width=1000, showlegend=False)
    return _write_fig(fig, out_dir / "05_cloud_diversity_vs_recall.html")


def plot_branch_size_vs_recall(df: pd.DataFrame, out_dir: Path) -> Optional[Path]:
    if not {"n_branch_targets_same_prefix", "pred_best_slot_recall"}.issubset(df.columns):
        return None
    plot_df = df.dropna(subset=["n_branch_targets_same_prefix", "pred_best_slot_recall"])
    if plot_df.empty:
        return None

    fig = go.Figure(
        go.Scatter(
            x=plot_df["n_branch_targets_same_prefix"],
            y=plot_df["pred_best_slot_recall"],
            mode="markers",
            marker=dict(
                size=7,
                opacity=0.55,
                color=plot_df.get("pred_baseline_recall"),
                colorscale="RdYlGn",
                showscale=True,
                colorbar=dict(title="Baseline recall"),
            ),
            text=plot_df.get("row", plot_df.index),
            hovertemplate=(
                "branch targets=%{x}<br>best-slot recall=%{y:.3f}<br>"
                "row=%{text}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        **_LAYOUT,
        title="Branch ambiguity vs best-slot recall",
        xaxis_title="Branch targets with same prefix",
        yaxis_title="Best-slot recall",
        yaxis=dict(range=[0, 1]),
        height=520,
        width=700,
    )
    return _write_fig(fig, out_dir / "06_branch_size_vs_recall.html")


def plot_slot_recall_distribution(
    slots_long: pd.DataFrame,
    out_dir: Path,
) -> Optional[Path]:
    if slots_long.empty or "recall" not in slots_long.columns:
        return None
    plot_df = slots_long.copy()
    plot_df["recall"] = pd.to_numeric(plot_df["recall"], errors="coerce")
    plot_df = plot_df.dropna(subset=["recall"])
    if plot_df.empty:
        return None

    plot_df["track"] = np.where(
        plot_df.get("is_baseline", False).astype(bool),
        "baseline",
        "slot",
    )
    fig = go.Figure()
    for track, color in [("baseline", "#1f77b4"), ("slot", "#ff7f0e")]:
        sub = plot_df[plot_df["track"] == track]
        if sub.empty:
            continue
        fig.add_trace(
            go.Violin(
                y=sub["recall"],
                name=track,
                box_visible=True,
                meanline_visible=True,
                marker_color=color,
                opacity=0.7,
            )
        )
    fig.update_layout(
        **_LAYOUT,
        title="Per-query recall distribution (long slots table)",
        yaxis_title="Recall",
        yaxis=dict(range=[0, 1]),
        height=480,
        width=520,
    )
    return _write_fig(fig, out_dir / "07_slot_recall_distribution.html")


def plot_esm_overlap_vs_recall(
    df: pd.DataFrame,
    esm_df: pd.DataFrame,
    out_dir: Path,
) -> Optional[Path]:
    if esm_df.empty or "row" not in esm_df.columns:
        return None
    merged = df.merge(esm_df, on="row", how="inner", suffixes=("", "_esm"))
    if merged.empty:
        return None
    if not {"slot_min_to_gt_l2", "pred_best_slot_recall"}.issubset(merged.columns):
        return None

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "ESM slot-min L2 vs best-slot recall",
            "ESM pred-min L2 vs pooled-cloud recall",
        ),
    )
    sub1 = merged.dropna(subset=["slot_min_to_gt_l2", "pred_best_slot_recall"])
    if not sub1.empty:
        fig.add_trace(
            go.Scatter(
                x=sub1["slot_min_to_gt_l2"],
                y=sub1["pred_best_slot_recall"],
                mode="markers",
                marker=dict(size=6, opacity=0.5, color="#9467bd"),
            ),
            row=1,
            col=1,
        )
    if {"pred_min_to_gt_l2", "pred_pooled_all_recall"}.issubset(merged.columns):
        sub2 = merged.dropna(subset=["pred_min_to_gt_l2", "pred_pooled_all_recall"])
        if not sub2.empty:
            fig.add_trace(
                go.Scatter(
                    x=sub2["pred_min_to_gt_l2"],
                    y=sub2["pred_pooled_all_recall"],
                    mode="markers",
                    marker=dict(size=6, opacity=0.5, color="#2ca02c"),
                ),
                row=1,
                col=2,
            )
    fig.update_xaxes(title_text="Min slot L2 to GT (ESM)", row=1, col=1)
    fig.update_xaxes(title_text="Min pred L2 to GT (ESM)", row=1, col=2)
    fig.update_yaxes(title_text="Best-slot recall", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="Pooled-cloud recall", range=[0, 1], row=1, col=2)
    fig.update_layout(**_LAYOUT, height=500, width=1000, showlegend=False)
    return _write_fig(fig, out_dir / "08_esm_overlap_vs_recall.html")


def plot_optimistic_esm_histogram(
    optimistic_df: pd.DataFrame,
    out_dir: Path,
) -> Optional[Path]:
    if optimistic_df.empty or "optimistic_min_l2" not in optimistic_df.columns:
        return None
    vals = pd.to_numeric(optimistic_df["optimistic_min_l2"], errors="coerce").dropna()
    if vals.empty:
        return None
    fig = go.Figure(go.Histogram(x=vals, marker_color="#17becf"))
    fig.update_layout(
        **_LAYOUT,
        title="Optimistic cross-series min L2 (ESM) per target",
        xaxis_title="Min L2 to best prediction in full cloud",
        yaxis_title="Target count",
        height=460,
        width=700,
    )
    return _write_fig(fig, out_dir / "09_optimistic_esm_min_l2.html")


def plot_blast_similarity_calibration(df: pd.DataFrame, out_dir: Path) -> Optional[Path]:
    needed = {
        "blast_similarity_source_to_target",
        "pred_best_slot_recall",
        "pred_best_slot_fn",
    }
    if not needed.issubset(df.columns):
        return None
    plot_df = df.dropna(subset=["blast_similarity_source_to_target", "pred_best_slot_recall"])
    if plot_df.empty:
        return None

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "BLAST similarity vs best-slot recall",
            "Best-slot FN vs BLAST similarity",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["blast_similarity_source_to_target"],
            y=plot_df["pred_best_slot_recall"],
            mode="markers",
            marker=dict(size=6, opacity=0.5, color="#1f77b4"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["blast_similarity_source_to_target"],
            y=plot_df["pred_best_slot_fn"],
            mode="markers",
            marker=dict(size=6, opacity=0.5, color="#d62728"),
        ),
        row=1,
        col=2,
    )
    x_range = _axis_range(plot_df["blast_similarity_source_to_target"], clamp_lo=0, clamp_hi=1)
    fig.update_xaxes(title_text="BLAST similarity (source → target)", range=x_range, row=1, col=1)
    fig.update_xaxes(title_text="BLAST similarity (source → target)", range=x_range, row=1, col=2)
    fig.update_yaxes(title_text="Best-slot recall", range=[0, 1], row=1, col=1)
    y2 = _axis_range(plot_df["pred_best_slot_fn"], clamp_lo=0)
    fig.update_yaxes(title_text="Best-slot FN", range=y2, row=1, col=2)
    fig.update_layout(**_LAYOUT, height=500, width=1000, showlegend=False)
    return _write_fig(fig, out_dir / "11_blast_similarity_calibration.html")


def plot_recall_by_lineage(df: pd.DataFrame, out_dir: Path) -> Optional[Path]:
    if "lineage_source" not in df.columns:
        return None
    plot_df = df.copy()
    plot_df["lineage_source"] = plot_df["lineage_source"].fillna("").astype(str)
    plot_df = plot_df[plot_df["lineage_source"].str.len() > 0]
    if plot_df.empty:
        return None
    plot_df["lineage_prefix"] = plot_df["lineage_source"].str.split(".").str[:2].str.join(".")

    metrics = [
        ("baseline", "pred_baseline_recall", "#1f77b4"),
        ("best slot", "pred_best_slot_recall", "#ff7f0e"),
        ("pooled cloud", "pred_pooled_all_recall", "#2ca02c"),
    ]
    present = [(n, c, color) for n, c, color in metrics if c in plot_df.columns]
    if not present:
        return None

    grouped = plot_df.groupby("lineage_prefix", observed=False)
    prefixes = [str(p) for p in grouped.size().sort_values(ascending=False).head(20).index]
    if not prefixes:
        return None

    fig = go.Figure()
    for name, col, color in present:
        means = grouped[col].mean().reindex(prefixes)
        fig.add_trace(go.Bar(name=name, x=prefixes, y=means, marker_color=color))
    fig.update_layout(
        **_LAYOUT,
        title="Mean recall by source lineage prefix (top 20)",
        xaxis_title="Source lineage prefix",
        yaxis=dict(title="Mean recall", range=[0, 1]),
        barmode="group",
        height=520,
        width=max(900, 40 * len(prefixes)),
    )
    return _write_fig(fig, out_dir / "12_recall_by_lineage_prefix.html")


def plot_optimistic_cross_series_scatter(
    optimistic_df: pd.DataFrame,
    out_dir: Path,
) -> Optional[Path]:
    if optimistic_df.empty or "cross_series_match" not in optimistic_df.columns:
        return None
    if "optimistic_min_l2" not in optimistic_df.columns:
        return None
    plot_df = optimistic_df.copy()
    plot_df["cross_series_match"] = plot_df["cross_series_match"].astype(bool)
    fig = go.Figure()
    for label, sub, color in [
        ("same series", plot_df[~plot_df["cross_series_match"]], "#1f77b4"),
        ("cross series", plot_df[plot_df["cross_series_match"]], "#d62728"),
    ]:
        if sub.empty:
            continue
        fig.add_trace(
            go.Histogram(
                x=pd.to_numeric(sub["optimistic_min_l2"], errors="coerce"),
                name=label,
                opacity=0.65,
                marker_color=color,
            )
        )
    fig.update_layout(
        **_LAYOUT,
        title="Optimistic min L2: same-series vs cross-series best match",
        xaxis_title="Min L2 to best prediction in full cloud",
        yaxis_title="Target count",
        barmode="overlay",
        height=460,
        width=760,
    )
    return _write_fig(fig, out_dir / "13_optimistic_cross_series_l2.html")


def _rename_for_distance_plot(df: pd.DataFrame) -> pd.DataFrame:
    """Adapt prefix-swap recall column for shared distance-vs-recall plot."""
    out = df.copy()
    if "pred_best_slot_recall" in out.columns:
        out["decoder_recall"] = out["pred_best_slot_recall"]
    elif "pred_pooled_all_recall" in out.columns:
        out["decoder_recall"] = out["pred_pooled_all_recall"]
    return out


def write_html_index(
    out_dir: Path,
    summary: dict[str, Any],
    plot_paths: list[Path],
    *,
    external_links: Optional[list[tuple[str, str]]] = None,
) -> Path:
    lines = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'>"
        "<title>Prefix-swap evaluation analytics</title>",
        "<style>",
        "body{font-family:sans-serif;max-width:1200px;margin:24px auto;padding:0 16px;}",
        "iframe{width:100%;height:560px;border:1px solid #ddd;margin:12px 0;}",
        "table{border-collapse:collapse;margin:12px 0;}td,th{border:1px solid #ccc;padding:6px 10px;}",
        ".note{color:#444;font-size:0.95em;}",
        "a.plot-link{display:block;margin:4px 0;}",
        "</style></head><body>",
        "<h1>Prefix-swap evaluation analytics</h1>",
        f"<p>Series rows: <b>{summary.get('n_series', 0)}</b></p>",
        "<h2>Prediction cloud recall summary</h2>",
        "<p class='note'>All methods below use the same "
        f"<b>{summary.get('n_series_pred_common', summary.get('n_series', 0))}</b> "
        "series (those with a baseline and at least one slot prediction). "
        "Position-level, scored against the labeled target only. "
        "<i>Best slot</i> is the slot with the highest recall vs the target "
        "(oracle choice, optimistic). <i>Pooled cloud</i> counts a site as changed "
        "if any member changed it, which raises both recall and FP; the "
        "<i>cloud consensus</i> row is identical to it by construction.</p>",
        "<table><tr><th>Method</th><th>TP</th><th>FN</th><th>FP</th><th>Mean recall</th></tr>",
    ]
    for m in summary.get("pred_methods", []):
        recall = m["recall"]
        recall_s = f"{recall:.3f}" if recall == recall else "n/a"
        fp = int(m.get("fp", 0))
        lines.append(
            f"<tr><td>{html_module.escape(m['label'])}</td>"
            f"<td>{int(m['tp'])}</td><td>{int(m['fn'])}</td>"
            f"<td>{fp}</td><td>{recall_s}</td></tr>"
        )
    lines.append("</table>")
    if summary.get("knn_methods"):
        lines.append("<h2>kNN neighbor recall summary</h2>")
        lines.append(
            "<table><tr><th>Method</th><th>TP</th><th>FN</th><th>FP</th><th>Mean recall</th></tr>"
        )
        for m in summary["knn_methods"]:
            recall = m["recall"]
            recall_s = f"{recall:.3f}" if recall == recall else "n/a"
            fp = int(m.get("fp", 0))
            lines.append(
                f"<tr><td>{html_module.escape(m['label'])}</td>"
                f"<td>{int(m['tp'])}</td><td>{int(m['fn'])}</td>"
                f"<td>{fp}</td><td>{recall_s}</td></tr>"
            )
        lines.append("</table>")
    lines.append(f"<p class='note'>{html_module.escape(GT_EDIT_BIN_LEGEND)}</p>")
    if "best_slot_wins_recall" in summary:
        lines.append(
            "<p>"
            f"Best slot higher recall than baseline: <b>{summary['best_slot_wins_recall']}</b> | "
            f"Baseline higher: <b>{summary['baseline_wins_recall']}</b> | "
            f"Ties: <b>{summary['ties_best_slot_baseline']}</b>"
            "</p>"
        )
    if "pred_cloud_wins_vs_knn" in summary:
        lines.append(
            "<p>"
            f"Pooled cloud beats kNN pooled: <b>{summary['pred_cloud_wins_vs_knn']}</b> | "
            f"kNN pooled beats cloud: <b>{summary['knn_wins_vs_pred_cloud']}</b>"
            "</p>"
        )
    lines.append("<h2>Plots</h2>")
    if external_links:
        lines.append("<h3>Related reports</h3><ul>")
        for label, href in external_links:
            lines.append(
                f'<li><a href="{html_module.escape(href)}" target="_blank">'
                f"{html_module.escape(label)}</a></li>"
            )
        lines.append("</ul>")
    for p in plot_paths:
        if p is not None and p.is_file():
            name = html_module.escape(p.name)
            stem = html_module.escape(p.stem)
            lines.append(f"<h3>{stem}</h3>")
            lines.append(
                f'<iframe src="{name}" title="{stem}" loading="lazy"></iframe>'
            )
            lines.append(
                f'<a class="plot-link" href="{name}" target="_blank">Open {name}</a>'
            )
    lines.append("</body></html>")
    index_path = out_dir / "index.html"
    index_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", index_path)
    return index_path


def _default_sibling(path: Path, name: str) -> Optional[Path]:
    candidate = path.parent / name
    return candidate if candidate.is_file() else None


def _resolve_eval_paths(eval_dir: Path) -> dict[str, Optional[Path]]:
    """Default file layout under evaluation_prefix_swap/."""
    return {
        "report_csv": eval_dir / "prefix_swap_evaluation_report.csv",
        "slots_long_csv": eval_dir / "prefix_swap_slots_long.csv",
        "esm_per_series_csv": eval_dir / "esm_cloud_umap" / "esm_cloud_overlap_per_series.csv",
        "esm_optimistic_csv": eval_dir / "esm_cloud_umap" / "esm_cloud_optimistic_per_target.csv",
        "esm_umap_html": eval_dir / "esm_cloud_umap" / "esm_cloud_umap.html",
        "decoded_fasta": _default_sibling(eval_dir.parent, "b3_lineage_subst_prefix_swap_test_decoded.fasta"),
    }


def plot_prefix_swap_evaluation_report(
    report_csv: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    slots_long_csv: Optional[Union[str, Path]] = None,
    esm_per_series_csv: Optional[Union[str, Path]] = None,
    esm_optimistic_csv: Optional[Union[str, Path]] = None,
    decoded_fasta: Optional[Union[str, Path]] = None,
    position_heatmaps: bool = True,
    examples_per_bucket: int = 3,
    max_positions: int = 60,
    top_k_positions: int = 40,
) -> dict[str, Any]:
    report_csv = Path(report_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(report_csv, low_memory=False)
    df = prepare_dataframe(df)
    summary = cohort_summary(df)
    (out_dir / "cohort_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    slots_path = Path(slots_long_csv) if slots_long_csv else None
    if slots_path is None:
        slots_path = _default_sibling(report_csv, "prefix_swap_slots_long.csv")
    slots_long = (
        pd.read_csv(slots_path, low_memory=False) if slots_path and slots_path.is_file() else pd.DataFrame()
    )

    esm_path = Path(esm_per_series_csv) if esm_per_series_csv else None
    if esm_path is None:
        esm_path = _default_sibling(
            report_csv, "esm_cloud_umap/esm_cloud_overlap_per_series.csv"
        )
    esm_df = (
        pd.read_csv(esm_path, low_memory=False) if esm_path and esm_path.is_file() else pd.DataFrame()
    )

    opt_path = Path(esm_optimistic_csv) if esm_optimistic_csv else None
    if opt_path is None:
        opt_path = _default_sibling(
            report_csv, "esm_cloud_umap/esm_cloud_optimistic_per_target.csv"
        )
    optimistic_df = (
        pd.read_csv(opt_path, low_memory=False) if opt_path and opt_path.is_file() else pd.DataFrame()
    )

    fasta_path = Path(decoded_fasta) if decoded_fasta else None
    if fasta_path is None:
        fasta_path = _default_sibling(
            report_csv.parent.parent,
            "b3_lineage_subst_prefix_swap_test_decoded.fasta",
        )
        if fasta_path is None:
            fasta_path = _default_sibling(
                report_csv.parent,
                "b3_lineage_subst_prefix_swap_test_decoded.fasta",
            )

    esm_umap_html = _default_sibling(
        report_csv.parent / "esm_cloud_umap",
        "esm_cloud_umap.html",
    )
    if esm_umap_html is None:
        esm_umap_html = report_csv.parent / "esm_cloud_umap" / "esm_cloud_umap.html"
        if not esm_umap_html.is_file():
            esm_umap_html = None

    plot_paths: list[Path] = []
    plot_paths.extend(plot_cohort_tp_fn(df, summary, out_dir))

    optional_plots = [
        _recall_scatter(
            df,
            x_col="pred_baseline_recall",
            y_col="pred_best_slot_recall",
            x_title="Baseline recall",
            y_title="Best-slot recall",
            title="Per-series recall: baseline vs best slot",
            path=out_dir / "02_baseline_vs_best_slot_recall.html",
        ),
        _recall_scatter(
            df,
            x_col="pred_baseline_recall",
            y_col="pred_pooled_all_recall",
            x_title="Baseline recall",
            y_title="Pooled-cloud recall",
            title="Per-series recall: baseline vs pooled cloud",
            path=out_dir / "02b_baseline_vs_pooled_cloud_recall.html",
        ),
        _recall_scatter(
            df,
            x_col="knn_pooled_change_recall",
            y_col="pred_pooled_all_recall",
            x_title="kNN pooled recall",
            y_title="Prefix-swap pooled-cloud recall",
            title="Per-series recall: kNN pooled vs prefix-swap cloud",
            path=out_dir / "02c_knn_vs_prefix_swap_cloud.html",
        ),
        plot_delta_recall_histogram(df, out_dir),
        plot_stratified_recall(
            df,
            out_dir,
            bin_col="gt_edit_bin",
            title="Mean recall by GT edit distance",
            filename="04_recall_by_gt_edits.html",
            metrics=[
                ("baseline", "pred_baseline_recall", "#1f77b4"),
                ("best slot", "pred_best_slot_recall", "#ff7f0e"),
                ("pooled cloud", "pred_pooled_all_recall", "#2ca02c"),
                ("kNN pooled", "knn_pooled_change_recall", "#9467bd"),
            ],
        ),
        plot_cloud_diversity_vs_recall(df, out_dir),
        plot_branch_size_vs_recall(df, out_dir),
        plot_slot_recall_distribution(slots_long, out_dir),
        plot_esm_overlap_vs_recall(df, esm_df, out_dir),
        plot_optimistic_esm_histogram(optimistic_df, out_dir),
        plot_optimistic_cross_series_scatter(optimistic_df, out_dir),
        plot_blast_similarity_calibration(df, out_dir),
        plot_recall_by_lineage(df, out_dir),
        plot_distance_vs_recall(
            _rename_for_distance_plot(df),
            out_dir,
            recall_label="Best-slot recall (labeled target; oracle slot choice)",
        ),
    ]
    plot_paths.extend(p for p in optional_plots if p is not None)

    if position_heatmaps:
        try:
            pos_paths = plot_prefix_swap_position_heatmaps(
                df,
                out_dir,
                decoded_fasta_path=fasta_path,
                examples_per_bucket=examples_per_bucket,
                max_positions=max_positions,
                top_k_positions=top_k_positions,
            )
            plot_paths.extend(pos_paths)
        except Exception:
            logger.exception("Prefix-swap position heatmap generation failed")

    external_links: list[tuple[str, str]] = []
    if esm_umap_html is not None and esm_umap_html.is_file():
        rel = Path(
            os.path.relpath(esm_umap_html.resolve(), out_dir.resolve())
        ).as_posix()
        external_links.append(("ESM cloud UMAP", rel))

    index_path = write_html_index(
        out_dir, summary, plot_paths, external_links=external_links or None
    )
    return {
        "report_csv": str(report_csv),
        "slots_long_csv": str(slots_path) if slots_path else None,
        "esm_per_series_csv": str(esm_path) if esm_path and esm_path.is_file() else None,
        "decoded_fasta": str(fasta_path) if fasta_path and fasta_path.is_file() else None,
        "esm_umap_html": str(esm_umap_html) if esm_umap_html and esm_umap_html.is_file() else None,
        "out_dir": str(out_dir),
        "index_html": str(index_path),
        "summary": summary,
        "plots": [str(p) for p in plot_paths if p is not None],
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Plot visual analytics for prefix_swap_evaluation_report.csv."
    )
    p.add_argument(
        "--eval-dir",
        type=Path,
        default=None,
        help=(
            "Evaluation output directory containing prefix_swap_evaluation_report.csv "
            "and optional esm_cloud_umap/ subfolder. Sets defaults for other paths."
        ),
    )
    p.add_argument(
        "--report-csv",
        type=Path,
        default=None,
        help="Path to prefix_swap_evaluation_report.csv.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for HTML plots, cohort_summary.json, and index.html.",
    )
    p.add_argument(
        "--slots-long-csv",
        type=Path,
        default=None,
        help="Optional prefix_swap_slots_long.csv (default: sibling of report CSV).",
    )
    p.add_argument(
        "--esm-per-series-csv",
        type=Path,
        default=None,
        help="Optional esm_cloud_overlap_per_series.csv for ESM scatter plots.",
    )
    p.add_argument(
        "--esm-optimistic-csv",
        type=Path,
        default=None,
        help="Optional esm_cloud_optimistic_per_target.csv.",
    )
    p.add_argument(
        "--decoded-fasta",
        type=Path,
        default=None,
        help="Decoded prefix-swap FASTA for position heatmaps.",
    )
    p.add_argument(
        "--no-position-heatmaps",
        action="store_true",
        help="Skip per-row position track heatmaps.",
    )
    p.add_argument("--examples-per-bucket", type=int, default=3)
    p.add_argument("--max-positions", type=int, default=60)
    p.add_argument("--top-k-positions", type=int, default=40)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    _setup_logging(args.log_level)

    report_csv = args.report_csv
    slots_long = args.slots_long_csv
    esm_per_series = args.esm_per_series_csv
    esm_optimistic = args.esm_optimistic_csv
    decoded_fasta = args.decoded_fasta
    out_dir = args.out_dir

    if args.eval_dir is not None:
        paths = _resolve_eval_paths(args.eval_dir)
        report_csv = report_csv or paths["report_csv"]
        out_dir = out_dir or (args.eval_dir / "plots")
        slots_long = slots_long or paths["slots_long_csv"]
        esm_per_series = esm_per_series or paths["esm_per_series_csv"]
        esm_optimistic = esm_optimistic or paths["esm_optimistic_csv"]
        decoded_fasta = decoded_fasta or paths["decoded_fasta"]

    if report_csv is None:
        p.error("Provide --report-csv or --eval-dir")
    if out_dir is None:
        out_dir = report_csv.parent / "plots"

    result = plot_prefix_swap_evaluation_report(
        report_csv,
        out_dir,
        slots_long_csv=slots_long,
        esm_per_series_csv=esm_per_series,
        esm_optimistic_csv=esm_optimistic,
        decoded_fasta=decoded_fasta,
        position_heatmaps=not args.no_position_heatmaps,
        examples_per_bucket=args.examples_per_bucket,
        max_positions=args.max_positions,
        top_k_positions=args.top_k_positions,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
