#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     plot_prefix_swap_position_heatmaps.py
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
"""Position-level heatmaps for prefix-swap evaluation reports."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from recomb2027.pipelines.generate_evaluation_report import extract_mutations
from recomb2027.pipelines.plot_position_heatmaps import (
    _parse_json_list,
    _row_id,
    plot_cohort_fn_spikes,
    plot_cohort_position_rates,
    plot_example_position_track,
    position_labels,
    select_positions,
)
from recomb2027.pipelines.prefix_swap_metrics import parse_prefix_swap_decoded_fasta

logger = logging.getLogger(__name__)


def load_prefix_swap_decoded_by_row(fasta_path: Optional[Path]) -> dict[int, dict[str, str]]:
    """Map row id → {query_id: sequence} from prefix-swap decoded FASTA."""
    if not fasta_path or not Path(fasta_path).is_file():
        return {}
    parsed = parse_prefix_swap_decoded_fasta(str(fasta_path))
    out: dict[int, dict[str, str]] = {}
    for row_id, queries in parsed.items():
        out[row_id] = {
            str(qid): str(meta.get("sequence", ""))
            for qid, meta in queries.items()
            if meta.get("sequence")
        }
    return out


def load_prefix_swap_row_sequences(
    row: pd.Series,
    *,
    decoded_by_row: dict[int, dict[str, str]],
) -> dict[str, str]:
    """Resolve sequences for one prefix-swap evaluation row."""
    rid = _row_id(row)
    queries = decoded_by_row.get(rid, {})
    best_qid = str(row.get("pred_best_slot_query_id", "") or "")
    best_slot = queries.get(best_qid, "")
    baseline = queries.get("baseline", "")

    knn_pooled = row.get("knn_aa_consensus_pooled_sequence", "")
    knn_pooled = "" if pd.isna(knn_pooled) else str(knn_pooled)
    cloud_cons = row.get("pred_cloud_consensus_sequence", "")
    cloud_cons = "" if pd.isna(cloud_cons) else str(cloud_cons)
    branch_cons = row.get("seq_target_consensus_branches", "")
    branch_cons = "" if pd.isna(branch_cons) else str(branch_cons)

    return {
        "source": str(row.get("seq_source", "")),
        "target": str(row.get("seq_target", "")),
        "decoder": best_slot,
        "closest_knn": baseline,
        "knn_pool_consensus": knn_pooled,
        "branch_targets_consensus": branch_cons,
        "pred_cloud_consensus": cloud_cons,
        "_track_labels": {
            "status": "best-slot status (labeled target)",
            "decoder": "best slot",
            "closest_knn": "baseline",
            "pred_cloud_consensus": "pred cloud consensus",
            "knn_pool_consensus": "kNN pooled consensus",
        },
    }


def select_prefix_swap_example_rows(
    df: pd.DataFrame,
    *,
    examples_per_bucket: int = 3,
) -> list[tuple[str, int]]:
    """Return (bucket_name, row_id) examples for prefix-swap track plots."""
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

    if "pred_best_slot_fn" in df.columns:
        _add(
            "worst_fn_best_slot",
            df.sort_values(["pred_best_slot_fn", "pred_best_slot_tp"], ascending=[False, True]),
            examples_per_bucket,
        )
    if {"pred_best_slot_tp", "pred_best_slot_fn"}.issubset(df.columns):
        best = df[(df["pred_best_slot_fn"] == 0) & (df["pred_best_slot_tp"] > 0)].sort_values(
            "pred_best_slot_tp", ascending=False
        )
        _add("best_tp_best_slot", best, examples_per_bucket)
    if "pred_best_slot_vs_baseline_delta_recall" in df.columns:
        _add(
            "best_slot_beats_baseline",
            df.sort_values("pred_best_slot_vs_baseline_delta_recall", ascending=False),
            examples_per_bucket,
        )
    if "pred_pooled_all_vs_knn_pooled_delta_recall" in df.columns:
        _add(
            "cloud_beats_knn",
            df.sort_values("pred_pooled_all_vs_knn_pooled_delta_recall", ascending=False),
            examples_per_bucket,
        )
    return chosen


def build_prefix_swap_cohort_position_table(
    df: pd.DataFrame,
    *,
    decoded_by_row: dict[int, dict[str, str]],
) -> pd.DataFrame:
    """Long table: one record per (row, position, method, label)."""
    records: list[dict[str, Any]] = []
    method_keys = [
        ("baseline", "closest_knn"),
        ("best_slot", "decoder"),
        ("cloud_consensus", "pred_cloud_consensus"),
        ("knn_pooled", "knn_pool_consensus"),
    ]
    for _, row in df.iterrows():
        seqs = load_prefix_swap_row_sequences(row, decoded_by_row=decoded_by_row)
        source, target = seqs["source"], seqs["target"]
        if not source or not target:
            continue
        required = _parse_json_list(row.get("branch_required_sites_union"))
        if not required:
            required, _ = extract_mutations(source, target)

        for method, query_key in method_keys:
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


def plot_prefix_swap_position_heatmaps(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    decoded_fasta_path: Optional[Path] = None,
    examples_per_bucket: int = 3,
    max_positions: int = 60,
    top_k_positions: int = 40,
) -> list[Path]:
    """Generate example tracks and cohort position summaries for prefix-swap."""
    out_dir = Path(out_dir)
    decoded_by_row = load_prefix_swap_decoded_by_row(decoded_fasta_path)

    paths: list[Path] = []
    examples = select_prefix_swap_example_rows(df, examples_per_bucket=examples_per_bucket)
    row_by_id = {_row_id(r): r for _, r in df.iterrows()}

    for bucket, rid in examples:
        row = row_by_id.get(rid)
        if row is None:
            continue
        seqs = load_prefix_swap_row_sequences(row, decoded_by_row=decoded_by_row)
        if not seqs["source"] or not seqs["target"]:
            continue
        required = _parse_json_list(row.get("branch_required_sites_union"))
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
        out_path = out_dir / f"10_position_track_row{rid}_{bucket}.html"
        paths.append(
            plot_example_position_track(
                row,
                seqs,
                positions=positions,
                bucket=bucket,
                out_path=out_path,
            )
        )

    long_df = build_prefix_swap_cohort_position_table(
        df, decoded_by_row=decoded_by_row
    )
    if not long_df.empty:
        csv_path = out_dir / "prefix_swap_position_level_long.csv"
        long_df.to_csv(csv_path, index=False)
        logger.info("Wrote %s", csv_path)

        decoder_like = long_df.copy()
        decoder_like["method"] = decoder_like["method"].replace(
            {"best_slot": "decoder", "baseline": "closest_knn"}
        )
        spike = plot_cohort_fn_spikes(decoder_like, out_dir, top_k=top_k_positions)
        if spike is not None:
            paths.append(spike)
        rates = plot_cohort_position_rates(decoder_like, out_dir, top_k=top_k_positions)
        if rates is not None:
            paths.append(rates)

    return paths
