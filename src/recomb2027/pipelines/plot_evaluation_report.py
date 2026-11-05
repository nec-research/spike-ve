#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     plot_evaluation_report.py
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
"""Visual analytics for ``evaluation_report.csv`` from generate_prediction_report.

Produces interactive Plotly HTML summaries and an index page with cohort statistics
focused on TP/FN/recall for decoder, closest kNN, and ambiguity-aware metrics.

Example::

    PYTHONPATH=src python -m recomb2027.pipelines.plot_evaluation_report \\
        --report-csv /path/to/evaluation_report.csv \\
        --out-dir /path/to/evaluation_plots \\
        --fasta /path/to/decoded.fasta \\
        --json-dir /path/to/knn_analysis
"""
from __future__ import annotations

import argparse
import html as html_module
import json
import logging
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from recomb2027.pipelines.generate_evaluation_report import (
    change_from_source_confusion_stats,
    closest_knn_confusion_stats,
    extract_mutations,
)
from recomb2027.pipelines.plot_position_heatmaps import plot_position_heatmaps

logger = logging.getLogger(__name__)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as e:  # pragma: no cover
    raise ImportError("plotly is required: pip install plotly") from e


GT_EDIT_BIN_EDGES = [0, 1, 2, 3, 5, 10_000]
GT_EDIT_BIN_LABELS = ["1", "2", "3", "4-5", "6+"]
GT_EDIT_BIN_LEGEND = (
    "GT edit distance bins: "
    "1 = 1 edit, 2 = 2 edits, 3 = 3 edits, 4-5 = 4–5 edits, 6+ = 6 or more edits"
)

METHOD_LABELS = {
    "decoder": "Decoder (labeled target)",
    "closest_knn": "Closest kNN (labeled target)",
    "knn_pool": "kNN pool consensus",
    "decoder_any_branch": "Decoder (any branch)",
    "closest_knn_any_branch": "Closest kNN (any branch)",
    "decoder_all_branches": "Decoder (all branches)",
    "closest_knn_all_branches": "Closest kNN (all branches)",
    "decoder_best_branch": "Decoder (best branch)",
    "closest_knn_best_branch": "Closest kNN (best branch)",
    "layer_consensus": "Layer33+34 consensus",
}

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

LABELED_METHOD_SPECS: list[tuple[str, str, str, str, str, str]] = [
    (
        "decoder",
        METHOD_LABELS["decoder"],
        "decoder_tp",
        "decoder_fn",
        "decoder_recall",
        "decoder_fp_alleles_count",  # allele-exact FP (decoder_fp is position-level)
    ),
    (
        "closest_knn",
        METHOD_LABELS["closest_knn"],
        "closest_knn_tp",
        "closest_knn_fn",
        "closest_knn_recall",
        "closest_knn_fp",
    ),
    (
        "knn_pool",
        METHOD_LABELS["knn_pool"],
        "knn_tp",
        "knn_fn",
        "knn_pool_recall",
        "knn_fp",
    ),
    (
        "layer_consensus",
        METHOD_LABELS["layer_consensus"],
        "layer_consensus_tp",
        "layer_consensus_fn",
        "layer_consensus_recall",
        "layer_consensus_fp_alleles_count",
    ),
]
LABELED_POSITION_METHOD_SPECS: list[tuple[str, str, str, str, str, str]] = [
    (
        "decoder_position",
        "Decoder (position-level)",
        "decoder_position_tp",
        "decoder_position_fn",
        "decoder_position_recall",
        "decoder_position_fp",
    ),
    (
        "closest_knn",
        "Closest kNN (position-level)",
        "closest_knn_tp",
        "closest_knn_fn",
        "closest_knn_recall",
        "closest_knn_fp",
    ),
    (
        "knn_pool",
        "kNN pool consensus (position-level)",
        "knn_tp",
        "knn_fn",
        "knn_pool_recall",
        "knn_fp",
    ),
    (
        "layer_consensus_position",
        "Layer33+34 consensus (position-level)",
        "layer_consensus_position_tp",
        "layer_consensus_position_fn",
        "layer_consensus_position_recall",
        "layer_consensus_position_fp",
    ),
]
LAYER_CONSENSUS_BRANCH_METHOD_SPECS: list[tuple[str, str, str, str, str, str]] = [
    (
        "layer_consensus_any_branch",
        "Layer33+34 consensus (any branch)",
        "layer_consensus_tp_any_branch",
        "layer_consensus_fn_any_branch",
        "layer_consensus_recall_any_branch",
        "layer_consensus_fp_alleles_count",
    ),
    (
        "layer_consensus_all_branches",
        "Layer33+34 consensus (all branches)",
        "layer_consensus_tp_alleles_all_branches",
        "layer_consensus_fn_alleles_all_branches",
        "layer_consensus_recall_alleles_all_branches",
        "layer_consensus_fp_alleles_count",
    ),
]
BRANCH_METHOD_SPECS: list[tuple[str, str, str, str, str, str]] = [
    (
        "decoder_any_branch",
        METHOD_LABELS["decoder_any_branch"],
        "decoder_tp_any_branch",
        "decoder_fn_any_branch",
        "decoder_recall_any_branch",
        "decoder_fp_alleles_count",
    ),
    (
        "closest_knn_any_branch",
        METHOD_LABELS["closest_knn_any_branch"],
        "closest_knn_tp_any_branch",
        "closest_knn_fn_any_branch",
        "closest_knn_recall_any_branch",
        "closest_knn_fp",
    ),
    (
        # Allele-exact: alleles (e.g. A123B) required in EVERY branch target.
        # (The position-level variant lives in BRANCH_POSITION_METHOD_SPECS.)
        "decoder_all_branches",
        METHOD_LABELS["decoder_all_branches"],
        "decoder_tp_alleles_all_branches",
        "decoder_fn_alleles_all_branches",
        "decoder_recall_alleles_all_branches",
        "decoder_fp_alleles_count",
    ),
    (
        "closest_knn_all_branches",
        METHOD_LABELS["closest_knn_all_branches"],
        "closest_knn_tp_all_branches",
        "closest_knn_fn_all_branches",
        "closest_knn_recall_all_branches",
        "closest_knn_fp",
    ),
]
BRANCH_POSITION_METHOD_SPECS: list[tuple[str, str, str, str, str, str]] = [
    (
        "closest_knn_any_branch",
        METHOD_LABELS["closest_knn_any_branch"] + " [position]",
        "closest_knn_tp_any_branch",
        "closest_knn_fn_any_branch",
        "closest_knn_recall_any_branch",
        "closest_knn_fp",
    ),
    (
        "decoder_all_branches",
        METHOD_LABELS["decoder_all_branches"] + " [position]",
        "decoder_tp_all_branches",
        "decoder_fn_all_branches",
        "decoder_recall_all_branches",
        "decoder_fp",
    ),
    (
        "closest_knn_all_branches",
        METHOD_LABELS["closest_knn_all_branches"] + " [position]",
        "closest_knn_tp_all_branches",
        "closest_knn_fn_all_branches",
        "closest_knn_recall_all_branches",
        "closest_knn_fp",
    ),
]
LAYER_CONSENSUS_BRANCH_POSITION_METHOD_SPECS: list[tuple[str, str, str, str, str, str]] = [
    (
        "layer_consensus_all_branches",
        "Layer33+34 consensus (all branches, position)",
        "layer_consensus_tp_all_branches",
        "layer_consensus_fn_all_branches",
        "layer_consensus_recall_all_branches",
        "layer_consensus_fp",
    ),
]
_METRIC_COLORS = {"TP": "#2ca02c", "FN": "#d62728", "FP": "#ff7f0e"}
_VIOLIN_GROUP_OFFSETS = {"TP": -0.2, "FN": 0.0, "FP": 0.2}
_STAT_MARKER_STYLES: tuple[tuple[str, str, str, str, int, str, bool], ...] = (
    ("mean", "Mean", "mean", "diamond", 9, "#1a1a1a", True),
    ("median", "Median", "median", "square", 8, "#1a1a1a", True),
    ("q25", "Quantiles", "quantiles", "line-ew", 8, "#4d4d4d", True),
    ("q75", "Quantiles", "quantiles", "line-ew", 8, "#4d4d4d", False),
)


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )


def _to_float0(value: Any) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if np.isnan(out) else out


def _methods_from_specs(
    df: pd.DataFrame,
    specs: list[tuple[str, str, str, str, str, str]],
) -> list[dict[str, Any]]:
    methods: list[dict[str, Any]] = []
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
        macro_recall = (
            float(df[recall_c].mean())
            if recall_c in df.columns
            else _safe_div(tp, tp + fn)
        )
        micro_recall = _safe_div(tp, tp + fn)
        methods.append(
            {
                "method": method_id,
                "label": label,
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "recall": macro_recall,
                "macro_recall": macro_recall,
                "micro_recall": micro_recall,
                "n_rows": int(df[[tp_c, fn_c]].notna().all(axis=1).sum()),
            }
        )
    return methods


def _grouped_categorical_x(
    methods: pd.Series,
    method_order: list[str],
    metric: str,
) -> np.ndarray:
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
) -> pd.DataFrame:
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
            for metric, val in (("TP", tp), ("FN", fn), ("FP", fp)):
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
    count_long = _per_series_confusion_long(df, specs)
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
    if "gt_edit_bin" not in df.columns:
        return None

    count_long = _per_series_confusion_long(df, specs)
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
        panel_bins.append((bin_label, int(sub["row"].nunique())))

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
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den > 0 else float("nan")


def _recall(tp: pd.Series, fn: pd.Series) -> pd.Series:
    return tp / (tp + fn).replace(0, np.nan)


def _write_fig(fig: go.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn")
    logger.info("Wrote %s", path)
    return path


def _axis_range(
    *series: pd.Series | np.ndarray | list[float],
    pad_fraction: float = 0.05,
    clamp_lo: Optional[float] = None,
    clamp_hi: Optional[float] = None,
) -> Optional[list[float]]:
    """Data-driven axis range with padding."""
    parts: list[pd.Series] = []
    for s in series:
        if isinstance(s, pd.Series):
            parts.append(pd.to_numeric(s, errors="coerce"))
        else:
            parts.append(pd.Series(s, dtype="float64"))
    if not parts:
        return None
    vals = pd.concat(parts).replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return None
    lo, hi = float(vals.min()), float(vals.max())
    if lo == hi:
        delta = max(abs(lo) * pad_fraction, 0.01) if lo != 0 else 0.01
        lo -= delta
        hi += delta
    else:
        span = hi - lo
        pad = span * pad_fraction
        lo -= pad
        hi += pad
    if clamp_lo is not None:
        lo = max(lo, clamp_lo)
    if clamp_hi is not None:
        hi = min(hi, clamp_hi)
    return [lo, hi]


def _allele_tp_fn(source_seq: str, pred_seq: str, target_seq: str) -> tuple[int, int]:
    _, gt_alleles = extract_mutations(source_seq, target_seq)
    _, pred_alleles = extract_mutations(source_seq, pred_seq)
    gt_set = set(gt_alleles)
    pred_set = set(pred_alleles)
    tp = len(gt_set & pred_set)
    fn = len(gt_set - pred_set)
    return tp, fn


def _best_branch_tp_fn(
    source_seq: str,
    pred_seq: str,
    branch_targets: list[str],
    *,
    metric: str,
) -> tuple[Optional[int], Optional[int]]:
    """Pick the branch target that maximizes recall for the given prediction."""
    if not source_seq or not branch_targets:
        return None, None
    if not pred_seq:
        if metric != "decoder":
            return None, None
        # Same convention as the report's labeled decoder metrics: a missing
        # decoded sequence is a complete miss (TP=0, FN=#GT alleles), so the
        # best-branch bars use the same cohort as the labeled bars.
        fns = [
            len(extract_mutations(source_seq, t)[1]) for t in branch_targets if t
        ]
        return (0, min(fns)) if fns else (None, None)

    best_recall = -1.0
    best_tp: Optional[int] = None
    best_fn: Optional[int] = None
    for tgt in branch_targets:
        if not tgt:
            continue
        if metric == "decoder":
            tp, fn = _allele_tp_fn(source_seq, pred_seq, tgt)
        else:
            stats = closest_knn_confusion_stats(source_seq, tgt, pred_seq)
            tp = stats["closest_knn_tp"]
            fn = stats["closest_knn_fn"]
            if tp is None or fn is None:
                continue
            tp = int(tp)
            fn = int(fn)

        denom = tp + fn
        recall = float(tp) / float(denom) if denom > 0 else float("nan")
        if recall > best_recall or (
            recall == best_recall and best_tp is not None and tp > best_tp
        ):
            best_recall = recall
            best_tp = tp
            best_fn = fn
    return best_tp, best_fn


def _branch_targets_by_group(df: pd.DataFrame) -> dict[tuple[str, str], list[str]]:
    """Collect unique branch targets for each prefix/source group."""
    if not {"seq_source", "seq_target"}.issubset(df.columns):
        return {}

    group_cols = ["seq_source"]
    if "branch_target_gids" in df.columns:
        group_cols.append("branch_target_gids")

    targets_by_group: dict[tuple[str, str], list[str]] = {}
    for key, group in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        src = str(key[0]) if key else ""
        gid_key = str(key[1]) if len(key) > 1 else ""
        targets: list[str] = []
        seen: set[str] = set()
        for tgt in group["seq_target"].dropna().astype(str):
            if tgt and tgt not in seen:
                seen.add(tgt)
                targets.append(tgt)
        targets_by_group[(src, gid_key)] = targets or [""]
    return targets_by_group


def add_best_branch_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Per-row TP/FN when scoring against the best-matching branch target."""
    out = df.copy()
    decoded_col = (
        "decoded_prediction_sequence"
        if "decoded_prediction_sequence" in out.columns
        else "decoded_prediction"
    )
    if decoded_col not in out.columns or "seq_source" not in out.columns:
        return out

    targets_by_group = _branch_targets_by_group(out)

    dec_tp: list[Optional[int]] = []
    dec_fn: list[Optional[int]] = []
    knn_tp: list[Optional[int]] = []
    knn_fn: list[Optional[int]] = []

    for _, row in out.iterrows():
        src = str(row.get("seq_source", ""))
        gid_key = (
            str(row.get("branch_target_gids", ""))
            if "branch_target_gids" in out.columns
            else ""
        )
        branch_targets = targets_by_group.get(
            (src, gid_key), [str(row.get("seq_target", ""))]
        )
        decoded = row.get(decoded_col, "")
        knn = row.get("knn_closest_sequence", "")
        decoded = "" if pd.isna(decoded) else str(decoded)
        knn = "" if pd.isna(knn) else str(knn)

        tp_d, fn_d = _best_branch_tp_fn(src, decoded, branch_targets, metric="decoder")
        tp_k, fn_k = _best_branch_tp_fn(src, knn, branch_targets, metric="closest_knn")
        dec_tp.append(tp_d)
        dec_fn.append(fn_d)
        knn_tp.append(tp_k)
        knn_fn.append(fn_k)

    # Build all Series on out.index so assignment aligns by position, not by a
    # fresh RangeIndex (which breaks for filtered/non-contiguous indices).
    dec_tp_s = pd.Series(dec_tp, index=out.index, dtype="float")
    dec_fn_s = pd.Series(dec_fn, index=out.index, dtype="float")
    knn_tp_s = pd.Series(knn_tp, index=out.index, dtype="float")
    knn_fn_s = pd.Series(knn_fn, index=out.index, dtype="float")
    out["decoder_tp_best_branch"] = dec_tp_s
    out["decoder_fn_best_branch"] = dec_fn_s
    out["closest_knn_tp_best_branch"] = knn_tp_s
    out["closest_knn_fn_best_branch"] = knn_fn_s
    out["decoder_recall_best_branch"] = _recall(dec_tp_s, dec_fn_s)
    out["closest_knn_recall_best_branch"] = _recall(knn_tp_s, knn_fn_s)
    return out


def filter_unique_continuations(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows whose prefix/source has a single distinct ground-truth target.

    The report stores ``n_branch_targets_for_prefix`` (number of distinct target
    sequences among series sharing the same prefix, or the same source sequence
    when prefix info was unavailable). Rows with exactly one target are kept.
    Missing values (or a missing column) fall back to counting distinct
    ``seq_target`` values per ``seq_source``.
    """
    if "n_branch_targets_for_prefix" in df.columns:
        n_targets = pd.to_numeric(df["n_branch_targets_for_prefix"], errors="coerce")
    else:
        n_targets = pd.Series(np.nan, index=df.index, dtype="float")

    missing = n_targets.isna()
    if missing.any():
        if not {"seq_source", "seq_target"}.issubset(df.columns):
            raise ValueError(
                "Cannot determine unique continuations: need either "
                "'n_branch_targets_for_prefix' or both 'seq_source' and 'seq_target'."
            )
        fallback = df.groupby("seq_source", dropna=False)["seq_target"].transform(
            "nunique"
        )
        n_targets = n_targets.where(~missing, fallback.astype("float"))

    keep = n_targets == 1
    # Reset the index: downstream code builds pd.Series from python lists
    # (default RangeIndex) and assigns them as columns, which silently
    # misaligns on a filtered (non-contiguous) index.
    out = df.loc[keep].reset_index(drop=True)
    logger.info(
        "Unique-continuation filter: kept %d / %d rows (%d dropped with multiple GT targets)",
        len(out),
        len(df),
        len(df) - len(out),
    )
    if out.empty:
        raise ValueError("Unique-continuation filter removed all rows.")
    return out


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived recall/difficulty columns used across plots."""
    out = df.copy()
    numeric_cols = [
        "decoder_tp",
        "decoder_fn",
        "decoder_fp",
        "decoder_tn",
        "decoder_position_tp",
        "decoder_position_fn",
        "decoder_position_fp",
        "decoder_position_tn",
        "layer_consensus_tp",
        "layer_consensus_fn",
        "layer_consensus_fp",
        "layer_consensus_tn",
        "layer_consensus_position_tp",
        "layer_consensus_position_fn",
        "layer_consensus_position_fp",
        "layer_consensus_position_tn",
        "closest_knn_tp",
        "closest_knn_fn",
        "closest_knn_fp",
        "knn_tp",
        "knn_fn",
        "knn_fp",
        "decoder_tp_any_branch",
        "decoder_fn_any_branch",
        "layer_consensus_tp_any_branch",
        "layer_consensus_fn_any_branch",
        "layer_consensus_tp_all_branches",
        "layer_consensus_fn_all_branches",
        "closest_knn_tp_any_branch",
        "closest_knn_fn_any_branch",
        "decoder_tp_all_branches",
        "decoder_fn_all_branches",
        "decoder_tp_alleles_all_branches",
        "decoder_fn_alleles_all_branches",
        "layer_consensus_tp_alleles_all_branches",
        "layer_consensus_fn_alleles_all_branches",
        "closest_knn_tp_all_branches",
        "closest_knn_fn_all_branches",
        "gt_hamming_source_target",
        "decoded_hamming_to_target",
        "num_branches",
        "num_identical_prefix_series",
        "n_branch_targets_for_prefix",
        "blosum_distance_source_to_target",
        "blast_identity_source_to_target",
        "blast_similarity_source_to_target",
        "blast_similarity_decoded_to_target",
        "blast_identity_decoded_to_target",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # The report writes 0/0 (not NaN) for the *_all_branches closest-kNN columns
    # when no kNN sequence exists, which inflates n_rows vs. the labeled/any-branch
    # columns. Treat those rows as missing so all closest-kNN rows use the same cohort.
    if "closest_knn_tp" in out.columns:
        knn_missing = out["closest_knn_tp"].isna()
        for col in ("closest_knn_tp_all_branches", "closest_knn_fn_all_branches"):
            if col in out.columns:
                out.loc[knn_missing, col] = np.nan

    if {"decoder_tp", "decoder_fn"}.issubset(out.columns):
        out["decoder_recall"] = _recall(out["decoder_tp"], out["decoder_fn"])
    if {"decoder_position_tp", "decoder_position_fn"}.issubset(out.columns):
        out["decoder_position_recall"] = _recall(
            out["decoder_position_tp"], out["decoder_position_fn"]
        )
    if {"layer_consensus_tp", "layer_consensus_fn"}.issubset(out.columns):
        out["layer_consensus_recall"] = _recall(
            out["layer_consensus_tp"], out["layer_consensus_fn"]
        )
    if {"layer_consensus_position_tp", "layer_consensus_position_fn"}.issubset(
        out.columns
    ):
        out["layer_consensus_position_recall"] = _recall(
            out["layer_consensus_position_tp"], out["layer_consensus_position_fn"]
        )
    if {"closest_knn_tp", "closest_knn_fn"}.issubset(out.columns):
        out["closest_knn_recall"] = _recall(
            out["closest_knn_tp"], out["closest_knn_fn"]
        )
    if {"knn_tp", "knn_fn"}.issubset(out.columns):
        out["knn_pool_recall"] = _recall(out["knn_tp"], out["knn_fn"])
    if {"decoder_tp_any_branch", "decoder_fn_any_branch"}.issubset(out.columns):
        out["decoder_recall_any_branch"] = _recall(
            out["decoder_tp_any_branch"], out["decoder_fn_any_branch"]
        )
    if {"layer_consensus_tp_any_branch", "layer_consensus_fn_any_branch"}.issubset(
        out.columns
    ):
        out["layer_consensus_recall_any_branch"] = _recall(
            out["layer_consensus_tp_any_branch"], out["layer_consensus_fn_any_branch"]
        )
    if {"closest_knn_tp_any_branch", "closest_knn_fn_any_branch"}.issubset(out.columns):
        out["closest_knn_recall_any_branch"] = _recall(
            out["closest_knn_tp_any_branch"], out["closest_knn_fn_any_branch"]
        )
    if {"decoder_tp_all_branches", "decoder_fn_all_branches"}.issubset(out.columns):
        out["decoder_recall_all_branches"] = _recall(
            out["decoder_tp_all_branches"], out["decoder_fn_all_branches"]
        )
    if {"decoder_tp_alleles_all_branches", "decoder_fn_alleles_all_branches"}.issubset(
        out.columns
    ):
        out["decoder_recall_alleles_all_branches"] = _recall(
            out["decoder_tp_alleles_all_branches"],
            out["decoder_fn_alleles_all_branches"],
        )
    if {
        "layer_consensus_tp_alleles_all_branches",
        "layer_consensus_fn_alleles_all_branches",
    }.issubset(out.columns):
        out["layer_consensus_recall_alleles_all_branches"] = _recall(
            out["layer_consensus_tp_alleles_all_branches"],
            out["layer_consensus_fn_alleles_all_branches"],
        )
    if {"layer_consensus_tp_all_branches", "layer_consensus_fn_all_branches"}.issubset(
        out.columns
    ):
        out["layer_consensus_recall_all_branches"] = _recall(
            out["layer_consensus_tp_all_branches"],
            out["layer_consensus_fn_all_branches"],
        )
    if {"closest_knn_tp_all_branches", "closest_knn_fn_all_branches"}.issubset(
        out.columns
    ):
        out["closest_knn_recall_all_branches"] = _recall(
            out["closest_knn_tp_all_branches"], out["closest_knn_fn_all_branches"]
        )

    if "decoder_recall" in out.columns and "closest_knn_recall" in out.columns:
        out["recall_delta_decoder_minus_knn"] = (
            out["decoder_recall"] - out["closest_knn_recall"]
        )

    if "gt_hamming_source_target" in out.columns:
        out["gt_edit_bin"] = pd.cut(
            out["gt_hamming_source_target"],
            bins=GT_EDIT_BIN_EDGES,
            labels=GT_EDIT_BIN_LABELS,
            right=True,
        )

    out = add_best_branch_metrics(out)
    out = add_paired_metrics(out)
    out = add_best_branch_level_metrics(out)
    return out


def cohort_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Aggregate TP/FN/FP/recall across the cohort."""
    allele_specs = (
        LABELED_METHOD_SPECS
        + BRANCH_METHOD_SPECS
        + LAYER_CONSENSUS_BRANCH_METHOD_SPECS
    )
    position_specs = (
        LABELED_POSITION_METHOD_SPECS
        + BRANCH_POSITION_METHOD_SPECS
        + LAYER_CONSENSUS_BRANCH_POSITION_METHOD_SPECS
    )
    methods_allele = _methods_from_specs(df, allele_specs)
    methods_position = _methods_from_specs(df, position_specs)
    summary: dict[str, Any] = {
        "n_series": int(len(df)),
        "methods": methods_allele,
        "methods_allele": methods_allele,
        "methods_position": methods_position,
    }
    if "decoder_recall" in df.columns and "closest_knn_recall" in df.columns:
        summary["decoder_wins_recall"] = int(
            (df["decoder_recall"] > df["closest_knn_recall"]).sum()
        )
        summary["knn_wins_recall"] = int(
            (df["closest_knn_recall"] > df["decoder_recall"]).sum()
        )
        summary["ties_recall"] = int(
            (df["decoder_recall"] == df["closest_knn_recall"]).sum()
        )
    if "decoder_matches_any_branch" in df.columns:
        summary["decoder_matches_any_branch_rate"] = float(
            df["decoder_matches_any_branch"].astype(bool).mean()
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

    labels = [
        m.get("label", METHOD_LABELS.get(m["method"], m["method"])) for m in methods
    ]
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
            name="Macro recall",
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
        width=max(800, 120 * len(labels)),
    )
    fig.update_yaxes(title_text="Count (sum over series)", secondary_y=False)
    fig.update_yaxes(title_text="Macro recall", range=[0, 1], secondary_y=True)
    return _write_fig(fig, path)


def _methods_for_specs(
    methods: list[dict[str, Any]],
    specs: list[tuple[str, str, str, str, str, str]],
) -> list[dict[str, Any]]:
    method_ids = {spec[0] for spec in specs}
    return [m for m in methods if m["method"] in method_ids]


def _append_cohort_plot_bundle(
    df: pd.DataFrame,
    methods: list[dict[str, Any]],
    specs: list[tuple[str, str, str, str, str, str]],
    out_dir: Path,
    *,
    cohort_path: Path,
    violin_path: Path,
    violin_by_bin_path: Path,
    title_prefix: str,
    metric_note: str,
) -> list[Path]:
    """Cohort bar chart + per-series violins for one metric family."""
    selected = _methods_for_specs(methods, specs)
    paths: list[Path] = []
    if not selected:
        return paths

    paths.append(
        _plot_cohort_tp_fn_recall(
            selected,
            title=f"{title_prefix} ({metric_note})",
            path=cohort_path,
        )
    )
    v = _plot_per_series_tp_fn_fp_violins(
        df,
        specs,
        title=(
            f"{title_prefix} — per-series TP / FN / FP ({metric_note}; "
            "grouped by method; markers: mean, median, quantiles)"
        ),
        path=violin_path,
    )
    if v is not None:
        paths.append(v)
    v_bin = _plot_per_series_tp_fn_fp_violins_by_edit_bin(
        df,
        specs,
        title=(
            f"{title_prefix} — per-series TP / FN / FP by GT edit distance "
            f"({metric_note}; grouped by method; markers: mean, median, quantiles)"
        ),
        path=violin_by_bin_path,
    )
    if v_bin is not None:
        paths.append(v_bin)
    return paths


def plot_cohort_tp_fn(df: pd.DataFrame, summary: dict[str, Any], out_dir: Path) -> list[Path]:
    methods_allele = summary.get("methods_allele", summary.get("methods", []))
    methods_position = summary.get("methods_position", [])

    paths: list[Path] = []
    paths.extend(
        _append_cohort_plot_bundle(
            df,
            methods_allele,
            LABELED_METHOD_SPECS,
            out_dir,
            cohort_path=out_dir / "01_cohort_tp_fn_summary.html",
            violin_path=out_dir / "01a_per_series_tp_fn_fp_violin_summary.html",
            violin_by_bin_path=out_dir / "01a_by_bin_per_series_tp_fn_fp_violin_summary.html",
            title_prefix="Cohort TP / FN / FP / Recall",
            metric_note="allele-exact",
        )
    )
    paths.extend(
        _append_cohort_plot_bundle(
            df,
            methods_position,
            LABELED_POSITION_METHOD_SPECS,
            out_dir,
            cohort_path=out_dir / "01p_cohort_tp_fn_summary_position.html",
            violin_path=out_dir / "01pa_per_series_tp_fn_fp_violin_summary_position.html",
            violin_by_bin_path=out_dir / "01pa_by_bin_per_series_tp_fn_fp_violin_summary_position.html",
            title_prefix="Cohort TP / FN / FP / Recall",
            metric_note="position-level, change-from-source",
        )
    )
    paths.extend(
        _append_cohort_plot_bundle(
            df,
            methods_allele,
            LAYER_CONSENSUS_BRANCH_METHOD_SPECS,
            out_dir,
            cohort_path=out_dir / "01c_cohort_tp_fn_layer_consensus_branch_summary.html",
            violin_path=out_dir / "01c2_per_series_tp_fn_fp_violin_layer_consensus_branch.html",
            violin_by_bin_path=out_dir / "01c2_by_bin_per_series_tp_fn_fp_violin_layer_consensus_branch.html",
            title_prefix="Cohort TP / FN / FP / Recall — layer33+34 consensus (branch targets)",
            metric_note="allele-exact",
        )
    )
    paths.extend(
        _append_cohort_plot_bundle(
            df,
            methods_position,
            LAYER_CONSENSUS_BRANCH_POSITION_METHOD_SPECS,
            out_dir,
            cohort_path=out_dir / "01cp_cohort_tp_fn_layer_consensus_branch_summary_position.html",
            violin_path=out_dir / "01cp2_per_series_tp_fn_fp_violin_layer_consensus_branch_position.html",
            violin_by_bin_path=out_dir / "01cp2_by_bin_per_series_tp_fn_fp_violin_layer_consensus_branch_position.html",
            title_prefix="Cohort TP / FN / FP / Recall — layer33+34 consensus (branch targets)",
            metric_note="position-level, change-from-source",
        )
    )
    paths.extend(
        _append_cohort_plot_bundle(
            df,
            methods_allele,
            BRANCH_METHOD_SPECS,
            out_dir,
            cohort_path=out_dir / "01b_cohort_tp_fn_branch_summary.html",
            violin_path=out_dir / "01b2_per_series_tp_fn_fp_violin_branch_summary.html",
            violin_by_bin_path=out_dir / "01b2_by_bin_per_series_tp_fn_fp_violin_branch_summary.html",
            title_prefix="Cohort TP / FN / FP / Recall (branch targets)",
            metric_note="allele-exact where applicable",
        )
    )
    paths.extend(
        _append_cohort_plot_bundle(
            df,
            methods_position,
            BRANCH_POSITION_METHOD_SPECS,
            out_dir,
            cohort_path=out_dir / "01bp_cohort_tp_fn_branch_summary_position.html",
            violin_path=out_dir / "01bp2_per_series_tp_fn_fp_violin_branch_summary_position.html",
            violin_by_bin_path=out_dir / "01bp2_by_bin_per_series_tp_fn_fp_violin_branch_summary_position.html",
            title_prefix="Cohort TP / FN / FP / Recall (branch targets)",
            metric_note="position-level, change-from-source",
        )
    )
    if not paths:
        raise ValueError("No TP/FN columns found for cohort summary plot.")
    return paths


def plot_decoder_vs_knn_scatter(df: pd.DataFrame, out_dir: Path) -> Optional[Path]:
    if not {"decoder_recall", "closest_knn_recall"}.issubset(df.columns):
        return None

    plot_df = df.dropna(subset=["decoder_recall", "closest_knn_recall"])
    if plot_df.empty:
        return None

    color = (
        plot_df["gt_hamming_source_target"]
        if "gt_hamming_source_target" in plot_df.columns
        else None
    )
    fig = go.Figure(
        go.Scatter(
            x=plot_df["closest_knn_recall"],
            y=plot_df["decoder_recall"],
            mode="markers",
            marker=dict(
                size=7,
                opacity=0.55,
                color=color,
                colorscale="Viridis",
                showscale=color is not None,
                colorbar=dict(title="GT edit distance") if color is not None else None,
            ),
            text=plot_df.get("row", plot_df.index),
            hovertemplate=(
                "Closest kNN recall=%{x:.3f}<br>"
                "Decoder recall=%{y:.3f}<br>"
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
    fig.update_layout(
        **_LAYOUT,
        title="Per-series recall: decoder vs closest kNN",
        xaxis_title="Closest kNN recall (labeled target)",
        yaxis_title="Decoder recall (labeled target)",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        height=560,
        width=620,
    )
    return _write_fig(fig, out_dir / "02_decoder_vs_closest_knn_recall.html")


def plot_stratified_recall(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    bin_col: str,
    title: str,
    filename: str,
    metrics: list[tuple[str, str, str]],
    common_cohort: bool = True,
) -> Optional[Path]:
    """Grouped bars of mean per-series recall by bin.

    With ``common_cohort`` (default) only series for which *every* plotted metric
    is defined are used, so all bars in a bin are averaged over the same series
    (e.g. series without a kNN sequence are dropped for the decoder bars too).
    Bin labels show n; error bars are the standard error of the mean.
    """
    if bin_col not in df.columns:
        return None
    present = [(n, c, color) for n, c, color in metrics if c in df.columns]
    if not present:
        return None

    sub = df
    if common_cohort:
        sub = df.dropna(subset=[c for _, c, _ in present])
    grouped = sub.groupby(bin_col, observed=False)
    sizes = grouped.size()
    bins = [f"{b} (n={int(sizes[b])})" for b in sizes.index]

    fig = go.Figure()
    for name, col, color in present:
        means = grouped[col].mean().reindex(sizes.index)
        sem = grouped[col].sem().reindex(sizes.index).fillna(0.0)
        fig.add_trace(
            go.Bar(
                name=name,
                x=bins,
                y=means,
                marker_color=color,
                error_y=dict(type="data", array=sem.to_numpy(), visible=True),
            )
        )

    cohort_note = (
        f"common cohort: {len(sub)}/{len(df)} series defined for all metrics"
        if common_cohort
        else f"{len(sub)} series"
    )
    fig.update_layout(
        **_LAYOUT,
        title=f"{title}<br><sup>{cohort_note}; error bars = SEM</sup>",
        xaxis_title=(
            "GT edit distance bin "
            "(1=1 edit, 2=2, 3=3, 4-5=4–5, 6+=6+)"
        ),
        yaxis=dict(title="Mean recall", range=[0, 1]),
        barmode="group",
        height=500,
        width=1000,
    )
    return _write_fig(fig, out_dir / filename)


# ---------------------------------------------------------------------------
# Paired, like-for-like decoder vs closest-kNN comparison
# ---------------------------------------------------------------------------
_PAIRED_DEFS: tuple[tuple[str, str], ...] = (
    ("allele", "Allele-exact (right amino acid at right site)"),
    ("pos", "Position-level (site changed from source)"),
)
_PAIRED_METRICS: tuple[tuple[str, str], ...] = (("recall", "Recall"), ("f1", "F1"))


def _tp_fn_fp_to_recall_f1(tp: int, fn: int, fp: int) -> tuple[float, float]:
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    den = 2 * tp + fp + fn
    f1 = (2 * tp / den) if den > 0 else float("nan")
    return recall, f1


def _paired_counts(
    source: str, target: str, pred: str
) -> Optional[dict[str, tuple[int, int, int]]]:
    """(TP, FN, FP) of one prediction vs source->target, under both definitions."""
    if not source or not target or not pred:
        return None
    _, gt_alleles = extract_mutations(source, target)
    _, pr_alleles = extract_mutations(source, pred)
    g, p = set(gt_alleles), set(pr_alleles)
    pos = change_from_source_confusion_stats(source, target, pred)
    if pos["tp"] is None:
        return None
    return {
        "allele": (len(g & p), len(g - p), len(p - g)),
        "pos": (int(pos["tp"]), int(pos["fn"]), int(pos["fp"])),
    }


def add_paired_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Per-series recall and F1 for decoder and closest kNN under the SAME definitions.

    Both predictions are scored from their sequences against their best branch
    target (the labeled target when there is a single branch), allele-exact and
    position-level. Columns:
    ``paired_{dec|knn}_{allele|pos}_{recall|f1}`` and boolean ``paired_cohort``
    (series where both predictions exist).
    """
    out = df.copy()
    decoded_col = (
        "decoded_prediction_sequence"
        if "decoded_prediction_sequence" in out.columns
        else "decoded_prediction"
    )
    needed = {"seq_source", "seq_target", "knn_closest_sequence", decoded_col}
    if not needed.issubset(out.columns):
        return out

    def _s(v: Any) -> str:
        return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)

    cols: dict[str, list[float]] = {
        f"paired_{m}_{d}_{k}": []
        for m in ("dec", "knn")
        for d, _ in _PAIRED_DEFS
        for k, _ in _PAIRED_METRICS
    }
    targets_by_group = _branch_targets_by_group(out)
    has_gids = "branch_target_gids" in out.columns
    for _, row in out.iterrows():
        src, tgt = _s(row["seq_source"]), _s(row["seq_target"])
        gid_key = str(row.get("branch_target_gids", "")) if has_gids else ""
        # With several ground-truth branches, score each prediction against its
        # best branch (max recall, ties: more TP), per definition.
        targets = targets_by_group.get((src, gid_key)) or [tgt]
        for m, pred in (
            ("dec", _s(row[decoded_col])),
            ("knn", _s(row["knn_closest_sequence"])),
        ):
            best: dict[str, tuple[tuple[float, int], tuple[int, int, int]]] = {}
            for t in targets:
                counts = _paired_counts(src, t, pred) if t else None
                if counts is None:
                    continue
                for d, (tp, fn, fp) in counts.items():
                    rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
                    if np.isnan(rec):
                        continue
                    key = (rec, tp)
                    if d not in best or key > best[d][0]:
                        best[d] = (key, (tp, fn, fp))
            for d, _ in _PAIRED_DEFS:
                if d in best:
                    rec, f1 = _tp_fn_fp_to_recall_f1(*best[d][1])
                else:
                    rec = f1 = float("nan")
                cols[f"paired_{m}_{d}_recall"].append(rec)
                cols[f"paired_{m}_{d}_f1"].append(f1)
    for name, vals in cols.items():
        out[name] = pd.Series(vals, index=out.index, dtype="float")
    out["paired_cohort"] = out[list(cols)].notna().all(axis=1)
    return out


# Methods shown in the per-level 04 plots: (column key, sequence column(s), label, color)
_LEVEL_PLOT_METHODS: tuple[tuple[str, tuple[str, ...], str, str], ...] = (
    ("dec", ("decoded_prediction_sequence", "decoded_prediction"), "Decoder", "#1f77b4"),
    ("knn_closest", ("knn_closest_sequence",), "Closest kNN", "#ff7f0e"),
    ("knn_pool", ("knn_consensus_sequence",), "kNN pool consensus", "#9467bd"),
)


def _best_branch_level_recall(
    source: str, pred: str, branch_targets: list[str]
) -> dict[str, float]:
    """Best recall over branch targets for a prediction, per scoring level.

    The best target is chosen independently for each level (allele-exact and
    position-level) as the one maximizing recall (ties: more TP).
    """
    out = {"allele": float("nan"), "pos": float("nan")}
    if not source or not pred or not branch_targets:
        return out
    best: dict[str, tuple[float, int]] = {}
    for tgt in branch_targets:
        counts = _paired_counts(source, tgt, pred) if tgt else None
        if counts is None:
            continue
        for level, (tp, fn, _fp) in counts.items():
            if tp + fn == 0:
                continue
            key = (tp / (tp + fn), tp)
            if level not in best or key > best[level]:
                best[level] = key
    for level, (rec, _tp) in best.items():
        out[level] = float(rec)
    return out


def add_best_branch_level_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Columns ``bb_{dec|knn_closest|knn_pool}_{allele|pos}_recall``.

    Every method is scored against the *best* of the branch targets sharing the
    row's source (same grouping as ``add_best_branch_metrics``), using the same
    scoring definition, so allele-exact and position-level plots compare
    like with like. Missing predictions give NaN.
    """
    out = df.copy()
    if "seq_source" not in out.columns or "seq_target" not in out.columns:
        return out
    targets_by_group = _branch_targets_by_group(out)

    def _s(v: Any) -> str:
        return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)

    methods = []
    for key, cols, _label, _color in _LEVEL_PLOT_METHODS:
        col = next((c for c in cols if c in out.columns), None)
        if col is not None:
            methods.append((key, col))
    if not methods:
        return out

    has_gids = "branch_target_gids" in out.columns
    values: dict[str, list[float]] = {
        f"bb_{key}_{lvl}_recall": [] for key, _ in methods for lvl in ("allele", "pos")
    }
    for _, row in out.iterrows():
        src = _s(row.get("seq_source"))
        gid_key = str(row.get("branch_target_gids", "")) if has_gids else ""
        targets = targets_by_group.get((src, gid_key)) or [_s(row.get("seq_target"))]
        for key, col in methods:
            rec = _best_branch_level_recall(src, _s(row.get(col)), targets)
            values[f"bb_{key}_allele_recall"].append(rec["allele"])
            values[f"bb_{key}_pos_recall"].append(rec["pos"])
    for name, vals in values.items():
        out[name] = pd.Series(vals, index=out.index, dtype="float")
    return out


def plot_recall_by_gt_edits_by_level(df: pd.DataFrame, out_dir: Path) -> list[Optional[Path]]:
    """04: mean recall per GT-edit bin, one plot per scoring level.

    Each plot shows decoder, closest kNN and kNN pool consensus scored with the
    SAME definition, each against its best branch target, on the same series.
    """
    paths: list[Optional[Path]] = []
    for level, level_title, filename in (
        ("allele", "exact allele level", "04_recall_by_gt_edits_allele.html"),
        ("pos", "position level (site changed from source)", "04_recall_by_gt_edits_position.html"),
    ):
        metrics = [
            (f"{label} (best branch)", f"bb_{key}_{level}_recall", color)
            for key, cols, label, color in _LEVEL_PLOT_METHODS
            if f"bb_{key}_{level}_recall" in df.columns
        ]
        paths.append(
            plot_stratified_recall(
                df,
                out_dir,
                bin_col="gt_edit_bin",
                title=f"Mean recall by GT edit distance — {level_title}",
                filename=filename,
                metrics=metrics,
            )
        )
    return paths


def _bootstrap_mean_ci(
    x: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(x.mean())
    if n == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return mean, float(lo), float(hi)


def paired_comparison_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Per bin (+ 'All'), definition and metric: paired mean(decoder - kNN) with 95% CI."""
    if "paired_cohort" not in df.columns or "gt_edit_bin" not in df.columns:
        return []
    sub = df[df["paired_cohort"]]
    if sub.empty:
        return []
    bin_names = [str(b) for b in GT_EDIT_BIN_LABELS]
    groups: list[tuple[str, pd.DataFrame]] = [
        (b, sub[sub["gt_edit_bin"].astype(str) == b]) for b in bin_names
    ]
    groups.append(("All", sub))
    rows: list[dict[str, Any]] = []
    for bin_name, g in groups:
        for d, _ in _PAIRED_DEFS:
            for k, _ in _PAIRED_METRICS:
                dec = g[f"paired_dec_{d}_{k}"].to_numpy(dtype=float)
                knn = g[f"paired_knn_{d}_{k}"].to_numpy(dtype=float)
                diff = dec - knn
                mean, lo, hi = _bootstrap_mean_ci(diff)
                rows.append(
                    {
                        "bin": bin_name,
                        "definition": d,
                        "metric": k,
                        "n": int(len(g)),
                        "mean_decoder": float(dec.mean()) if len(g) else float("nan"),
                        "mean_knn": float(knn.mean()) if len(g) else float("nan"),
                        "mean_diff": mean,
                        "ci_lo": lo,
                        "ci_hi": hi,
                        "decoder_wins": int((diff > 1e-12).sum()),
                        "ties": int((np.abs(diff) <= 1e-12).sum()),
                        "knn_wins": int((diff < -1e-12).sum()),
                    }
                )
    return rows


def plot_paired_decoder_vs_knn(df: pd.DataFrame, out_dir: Path) -> Optional[Path]:
    """Paired per-series (decoder - closest kNN) difference by GT-edit bin."""
    rows = paired_comparison_rows(df)
    if not rows:
        return None
    table = pd.DataFrame(rows)
    n_by_bin = table.drop_duplicates("bin").set_index("bin")["n"].to_dict()
    bin_order = [str(b) for b in GT_EDIT_BIN_LABELS] + ["All"]
    x_labels = [f"{b} (n={n_by_bin[b]})" for b in bin_order]

    fig = make_subplots(
        rows=len(_PAIRED_METRICS),
        cols=len(_PAIRED_DEFS),
        subplot_titles=[
            f"{mlabel} — {dlabel}"
            for _, mlabel in _PAIRED_METRICS
            for _, dlabel in _PAIRED_DEFS
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.16,
    )
    for r, (k, _) in enumerate(_PAIRED_METRICS, start=1):
        for c, (d, _) in enumerate(_PAIRED_DEFS, start=1):
            sub = table[(table["metric"] == k) & (table["definition"] == d)].set_index("bin")
            sub = sub.reindex(bin_order)
            colors = [
                "#2ca02c" if lo > 0 else ("#d62728" if hi < 0 else "#9e9e9e")
                for lo, hi in zip(sub["ci_lo"], sub["ci_hi"])
            ]
            hover = [
                f"bin {b}<br>n={int(row.n)}"
                f"<br>decoder mean={row.mean_decoder:.3f}"
                f"<br>kNN mean={row.mean_knn:.3f}"
                f"<br>diff={row.mean_diff:+.3f} [{row.ci_lo:+.3f}, {row.ci_hi:+.3f}]"
                f"<br>decoder wins/ties/kNN wins="
                f"{int(row.decoder_wins)}/{int(row.ties)}/{int(row.knn_wins)}"
                for b, row in zip(bin_order, sub.itertuples())
            ]
            fig.add_trace(
                go.Bar(
                    x=x_labels,
                    y=sub["mean_diff"],
                    marker_color=colors,
                    error_y=dict(
                        type="data",
                        symmetric=False,
                        array=(sub["ci_hi"] - sub["mean_diff"]).to_numpy(),
                        arrayminus=(sub["mean_diff"] - sub["ci_lo"]).to_numpy(),
                        visible=True,
                    ),
                    hovertext=hover,
                    hoverinfo="text",
                    text=[
                        f"{int(w)}/{int(t)}/{int(l)}"
                        for w, t, l in zip(sub["decoder_wins"], sub["ties"], sub["knn_wins"])
                    ],
                    textposition="outside",
                    showlegend=False,
                ),
                row=r,
                col=c,
            )
            fig.add_hline(y=0, line_width=1, line_color="#333", row=r, col=c)
            fig.update_yaxes(title_text="decoder − kNN", row=r, col=c)
            fig.update_xaxes(tickangle=-30, row=r, col=c)

    n_cohort = int(df["paired_cohort"].sum())
    fig.update_layout(
        **dict(_LAYOUT_BASE, margin=dict(l=60, r=40, t=130, b=90)),
        title=(
            "Paired decoder vs closest-kNN difference by GT edit distance "
            f"(same {n_cohort} series, best branch target, same metric definition)"
            "<br><sup>Bar = mean per-series (decoder − kNN); whiskers = 95% bootstrap CI; "
            "green = decoder better, red = kNN better, grey = CI includes 0; "
            "labels = decoder wins / ties / kNN wins</sup>"
        ),
        height=900,
        width=1200,
    )
    return _write_fig(fig, out_dir / "04b_paired_decoder_vs_knn_by_gt_edits.html")


def plot_hamming_calibration(df: pd.DataFrame, out_dir: Path) -> Optional[Path]:
    needed = {"decoded_hamming_to_target", "decoder_fn", "gt_hamming_source_target"}
    if not needed.issubset(df.columns):
        return None

    plot_df = df.dropna(subset=list(needed))
    if plot_df.empty:
        return None

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Distance calibration", "FN vs distance to labeled target"))

    fig.add_trace(
        go.Scatter(
            x=plot_df["gt_hamming_source_target"],
            y=plot_df["decoded_hamming_to_target"],
            mode="markers",
            marker=dict(
                size=6,
                opacity=0.5,
                color=plot_df["decoder_tp"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Decoder TP", x=0.45),
            ),
            name="series",
        ),
        row=1,
        col=1,
    )
    mx = max(
        float(plot_df["gt_hamming_source_target"].max()),
        float(plot_df["decoded_hamming_to_target"].max()),
    )
    fig.add_trace(
        go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(dash="dash", color="gray"), showlegend=False),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["decoded_hamming_to_target"],
            y=plot_df["decoder_fn"],
            mode="markers",
            marker=dict(size=6, opacity=0.45, color="#d62728"),
            name="FN",
        ),
        row=1,
        col=2,
    )

    x1_range = _axis_range(
        plot_df["gt_hamming_source_target"],
        plot_df["decoded_hamming_to_target"],
        clamp_lo=0,
    )
    x2_range = _axis_range(plot_df["decoded_hamming_to_target"], clamp_lo=0)
    y2_range = _axis_range(plot_df["decoder_fn"], clamp_lo=0)

    fig.update_xaxes(title_text="GT edit distance (source → target)", range=x1_range, row=1, col=1)
    fig.update_yaxes(title_text="Prediction edit distance (prediction → labeled target)", range=x1_range, row=1, col=1)
    fig.update_xaxes(title_text="Prediction edit distance (prediction → labeled target)", range=x2_range, row=1, col=2)
    fig.update_yaxes(title_text="Decoder FN (labeled target)", range=y2_range, row=1, col=2)
    fig.update_layout(**_LAYOUT, height=500, width=1100, showlegend=False)
    return _write_fig(fig, out_dir / "06_hamming_calibration.html")


def plot_blast_similarity_calibration(
    df: pd.DataFrame, out_dir: Path
) -> Optional[Path]:
    """Source→target vs decoded→target BLAST similarity (BLOSUM62 positives)."""
    needed = {
        "blast_similarity_source_to_target",
        "blast_similarity_decoded_to_target",
    }
    if not needed.issubset(df.columns):
        return None

    plot_df = df.dropna(subset=list(needed))
    if plot_df.empty:
        return None

    xcol = "blast_similarity_source_to_target"
    ycol = "blast_similarity_decoded_to_target"
    sim_range = _axis_range(plot_df[xcol], plot_df[ycol], clamp_lo=0, clamp_hi=1)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("BLAST similarity calibration", "FN vs BLAST similarity gap"),
    )

    marker_kwargs: dict[str, Any] = dict(size=6, opacity=0.55)
    if "decoder_tp" in plot_df.columns:
        marker_kwargs.update(
            color=plot_df["decoder_tp"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Decoder TP", x=0.45),
        )
    else:
        marker_kwargs["color"] = "#1f77b4"

    fig.add_trace(
        go.Scatter(
            x=plot_df[xcol],
            y=plot_df[ycol],
            mode="markers",
            marker=marker_kwargs,
            name="series",
        ),
        row=1,
        col=1,
    )
    if sim_range is not None:
        fig.add_trace(
            go.Scatter(
                x=sim_range,
                y=sim_range,
                mode="lines",
                line=dict(dash="dash", color="gray"),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    similarity_gap = plot_df[ycol] - plot_df[xcol]
    if "decoder_fn" in plot_df.columns:
        y_gap = plot_df["decoder_fn"]
        y_title = "Decoder FN (labeled target)"
    elif "decoder_recall" in plot_df.columns:
        y_gap = plot_df["decoder_recall"]
        y_title = "Decoder recall (labeled target)"
    else:
        fig.add_trace(
            go.Histogram(x=similarity_gap, marker_color="#9467bd", name="gap"),
            row=1,
            col=2,
        )
        y_title = "Series count"

    if "decoder_fn" in plot_df.columns or "decoder_recall" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=similarity_gap,
                y=y_gap,
                mode="markers",
                marker=dict(size=6, opacity=0.45, color="#d62728"),
                name="gap vs metric",
            ),
            row=1,
            col=2,
        )

    gap_range = _axis_range(similarity_gap)
    if "decoder_fn" in plot_df.columns:
        y2_range = _axis_range(plot_df["decoder_fn"], clamp_lo=0)
    elif "decoder_recall" in plot_df.columns:
        y2_range = [0, 1]
    else:
        y2_range = None

    fig.update_xaxes(
        title_text="BLAST similarity (source → target)",
        range=sim_range,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="BLAST similarity (decoded → target)",
        range=sim_range,
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title_text="BLAST similarity gap (decoded − source)",
        range=gap_range,
        row=1,
        col=2,
    )
    fig.update_yaxes(title_text=y_title, range=y2_range, row=1, col=2)
    fig.update_layout(**_LAYOUT, height=500, width=1100, showlegend=False)
    return _write_fig(fig, out_dir / "08_blast_similarity_calibration.html")


def plot_distance_vs_recall(
    df: pd.DataFrame, out_dir: Path, *, recall_label: Optional[str] = None
) -> Optional[Path]:
    recall_col = (
        "decoder_recall_best_branch"
        if "decoder_recall_best_branch" in df.columns
        else "decoder_recall"
    )
    if recall_col not in df.columns:
        return None

    panels: list[tuple[str, str]] = []
    if "blosum_distance_source_to_target" in df.columns:
        panels.append(
            ("BLOSUM distance (source → target)", "blosum_distance_source_to_target")
        )
    if "blast_identity_source_to_target" in df.columns:
        panels.append(
            ("BLAST identity (source → target)", "blast_identity_source_to_target")
        )
    if not panels:
        return None

    if recall_label is None:
        recall_label = (
            "Decoder recall (best branch)"
            if recall_col == "decoder_recall_best_branch"
            else "Decoder recall (labeled target)"
        )

    fig = make_subplots(
        rows=1,
        cols=len(panels),
        subplot_titles=[p[0] for p in panels],
    )

    for col_idx, (xlabel, xcol) in enumerate(panels, start=1):
        sub = df.dropna(subset=[xcol, recall_col])
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub[xcol],
                y=sub[recall_col],
                mode="markers",
                marker=dict(size=6, opacity=0.45, color="#1f77b4"),
                name=xlabel,
                showlegend=False,
            ),
            row=1,
            col=col_idx,
        )
        x_range = _axis_range(sub[xcol], clamp_lo=0 if "identity" in xcol else None)
        y_range = _axis_range(sub[recall_col], clamp_lo=0, clamp_hi=1)
        fig.update_xaxes(title_text=xlabel, range=x_range, row=1, col=col_idx)
        fig.update_yaxes(title_text=recall_label, range=y_range, row=1, col=col_idx)

    fig.update_layout(
        **_LAYOUT,
        title=f"{recall_label} vs evolutionary distance",
        height=500,
        width=620 * len(panels),
    )
    return _write_fig(fig, out_dir / "07_distance_vs_recall.html")


def _render_cohort_summary_table(methods: list[dict[str, Any]]) -> list[str]:
    lines = [
        "<table><tr><th>Method</th><th>TP</th><th>FN</th><th>FP</th>"
        "<th>Macro recall</th><th>Micro recall</th></tr>",
    ]
    for m in methods:
        macro_recall = m.get("macro_recall", m.get("recall"))
        micro_recall = m.get("micro_recall")
        macro_s = f"{macro_recall:.3f}" if macro_recall == macro_recall else "n/a"
        micro_s = f"{micro_recall:.3f}" if micro_recall == micro_recall else "n/a"
        label = m.get("label", METHOD_LABELS.get(m["method"], m["method"]))
        fp = int(m.get("fp", 0))
        lines.append(
            f"<tr><td>{html_module.escape(label)}</td><td>{int(m['tp'])}</td>"
            f"<td>{int(m['fn'])}</td><td>{fp}</td>"
            f"<td>{macro_s}</td><td>{micro_s}</td></tr>"
        )
    lines.append("</table>")
    return lines


def write_html_index(
    out_dir: Path,
    summary: dict[str, Any],
    plot_paths: list[Path],
) -> Path:
    lines = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'><title>Evaluation report analytics</title>",
        "<style>",
        "body{font-family:sans-serif;max-width:1200px;margin:24px auto;padding:0 16px;}",
        "iframe{width:100%;height:560px;border:1px solid #ddd;margin:12px 0;}",
        "table{border-collapse:collapse;margin:12px 0;}td,th{border:1px solid #ccc;padding:6px 10px;}",
        ".note{color:#444;font-size:0.95em;}",
        "a.plot-link{display:block;margin:4px 0;}",
        "</style></head><body>",
        "<h1>Evaluation report analytics</h1>",
        f"<p>Series rows: <b>{summary.get('n_series', 0)}</b></p>",
        "<h2>Cohort recall summary (allele-exact)</h2>",
        "<p class='note'>Allele TP/FN: exact required mutation match "
        "(e.g. E191Q). Macro recall = mean per-series recall. "
        "Micro recall = sum(TP) / (sum(TP) + sum(FN)).</p>",
    ]
    lines.extend(_render_cohort_summary_table(summary.get("methods_allele", [])))
    lines.append("<h2>Cohort recall summary (position-level)</h2>")
    lines.append(
        "<p class='note'>Position TP/FN/FP: change-from-source vs labeled target "
        "(site changed from source counts as TP even if allele differs). "
        "kNN metrics are always position-level.</p>"
    )
    lines.extend(_render_cohort_summary_table(summary.get("methods_position", [])))
    lines.append(f"<p class='note'>{html_module.escape(GT_EDIT_BIN_LEGEND)}</p>")
    if "decoder_wins_recall" in summary:
        lines.append(
            "<p>"
            f"Decoder higher recall (labeled target): <b>{summary['decoder_wins_recall']}</b> | "
            f"Closest kNN higher: <b>{summary['knn_wins_recall']}</b> | "
            f"Ties: <b>{summary['ties_recall']}</b>"
            "</p>"
        )
    if "decoder_matches_any_branch_rate" in summary:
        lines.append(
            "<p>Decoder matches any branch target: "
            f"<b>{summary['decoder_matches_any_branch_rate']:.1%}</b></p>"
        )
    lines.append("<h2>Plots</h2>")
    for p in plot_paths:
        if p is not None and p.is_file():
            name = html_module.escape(p.name)
            stem = html_module.escape(p.stem)
            if p.suffix == ".html":
                lines.append(f"<h3>{stem}</h3>")
                lines.append(
                    f'<iframe src="{name}" title="{stem}" loading="lazy"></iframe>'
                )
                lines.append(f'<a class="plot-link" href="{name}" target="_blank">Open {name}</a>')
            else:
                lines.append(f"<h3>{stem}</h3>")
                lines.append(f'<img src="{name}" alt="{stem}" style="max-width:100%"/>')
    lines.append("</body></html>")
    index_path = out_dir / "index.html"
    index_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", index_path)
    return index_path


def plot_evaluation_report(
    report_csv: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    decoded_fasta: Optional[Union[str, Path]] = None,
    json_dir: Optional[Union[str, Path]] = None,
    position_heatmaps: bool = True,
    examples_per_bucket: int = 3,
    max_positions: int = 60,
    top_k_positions: int = 40,
    unique_continuations_only: bool = False,
) -> dict[str, Any]:
    report_csv = Path(report_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(report_csv, low_memory=False)
    if unique_continuations_only:
        df = filter_unique_continuations(df)
    df = prepare_dataframe(df)
    paired_rows = paired_comparison_rows(df)
    if paired_rows:
        pd.DataFrame(paired_rows).to_csv(
            out_dir / "paired_decoder_vs_knn_by_bin.csv", index=False
        )
    summary = cohort_summary(df)
    (out_dir / "cohort_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    plot_paths: list[Path] = []
    plot_paths.extend(plot_cohort_tp_fn(df, summary, out_dir))
    optional_plots = [
        plot_decoder_vs_knn_scatter(df, out_dir),
        *plot_recall_by_gt_edits_by_level(df, out_dir),
        plot_paired_decoder_vs_knn(df, out_dir),
        plot_hamming_calibration(df, out_dir),
        plot_distance_vs_recall(df, out_dir),
        plot_blast_similarity_calibration(df, out_dir),
    ]
    plot_paths.extend(p for p in optional_plots if p is not None)

    if position_heatmaps:
        try:
            pos_paths = plot_position_heatmaps(
                df,
                out_dir,
                decoded_fasta_path=Path(decoded_fasta) if decoded_fasta else None,
                json_dir=Path(json_dir) if json_dir else None,
                examples_per_bucket=examples_per_bucket,
                max_positions=max_positions,
                top_k_positions=top_k_positions,
            )
            plot_paths.extend(pos_paths)
        except Exception:
            logger.exception("Position heatmap generation failed")

    index_path = write_html_index(out_dir, summary, plot_paths)
    return {
        "report_csv": str(report_csv),
        "out_dir": str(out_dir),
        "index_html": str(index_path),
        "summary": summary,
        "plots": [str(p) for p in plot_paths if p is not None],
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Plot visual analytics for evaluation_report.csv (Plotly HTML)."
    )
    p.add_argument(
        "--report-csv",
        type=Path,
        required=True,
        help="Path to evaluation_report.csv from generate_prediction_report.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory for HTML plots, cohort_summary.json, and index.html.",
    )
    p.add_argument(
        "--fasta",
        type=Path,
        default=None,
        help="Decoded prediction FASTA (row= in header) when sequences are not in CSV.",
    )
    p.add_argument(
        "--json-dir",
        type=Path,
        default=None,
        help="kNN analysis dir with knn_baseline_row*.fasta for closest/consensus sequences.",
    )
    p.add_argument(
        "--no-position-heatmaps",
        action="store_true",
        help="Skip position-level example tracks and cohort FN spike plots.",
    )
    p.add_argument(
        "--unique-continuations-only",
        action="store_true",
        help=(
            "Analyse only series with a single ground-truth target (drop rows whose "
            "prefix/source has multiple distinct continuations in the test series)."
        ),
    )
    p.add_argument("--examples-per-bucket", type=int, default=3)
    p.add_argument("--max-positions", type=int, default=60)
    p.add_argument("--top-k-positions", type=int, default=40)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    _setup_logging(args.log_level)
    result = plot_evaluation_report(
        args.report_csv,
        args.out_dir,
        decoded_fasta=args.fasta,
        json_dir=args.json_dir,
        position_heatmaps=not args.no_position_heatmaps,
        examples_per_bucket=args.examples_per_bucket,
        max_positions=args.max_positions,
        top_k_positions=args.top_k_positions,
        unique_continuations_only=args.unique_continuations_only,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "summary"}, indent=2))


if __name__ == "__main__":
    main()
