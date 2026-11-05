#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     generate_prefix_swap_evaluation_report.py
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
"""Compile prefix-swap cloud evaluation reports (CSV + consensus FASTA).

Joins validation pairs, multi-query decoded FASTA (baseline + prefix-swap slots),
optional kNN mutation-stats JSON from ``analyse_lineage_subst_knn``, and branch
metadata from series NPZ + MSA.

Outputs:
* ``prefix_swap_evaluation_report.csv`` — one row per series (wide summary)
* ``prefix_swap_slots_long.csv`` — one row per (series, query_id)
* ``prefix_swap_consensus.fasta`` — cohort FASTA with ``row=`` in headers

Example::

    PYTHONPATH=src python -m recomb2027.pipelines.generate_prefix_swap_evaluation_report \\
        --val-pairs validation_pairs_test.csv \\
        --decoded-fasta b3_lineage_subst_prefix_swap_test_decoded.fasta \\
        --json-dir kNN_prefix_swap_analysis/ \\
        --series-npz test_dataset.npz --msa filtered_msa_S1.fasta \\
        --metadata-csv /mnt/.../phylotree_matched/df_labels.csv \\
        --output-dir evaluation_prefix_swap/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from recomb2027.latent.predictors.visualize_infer_umap import load_gid_lineage_map
from recomb2027.pipelines.generate_evaluation_report import (
    _load_json_if_exists,
    _row_id_from_pairs_row,
    _serialize_cell,
    build_branch_groups,
    build_branch_metadata_by_row,
    build_target_consensus,
    compute_blast_alignment_metrics,
    compute_blosum_distance,
    compute_hamming_distance,
    compute_series_prefix_keys,
    load_blosum62_matrix,
)
from recomb2027.pipelines.prefix_swap_esm_cloud import (
    METADATA_CSV_DEFAULT,
    resolve_lineage_for_gid,
)
from recomb2027.pipelines.prefix_swap_metrics import (
    best_slot_by_position_recall,
    build_pred_cloud_consensus,
    cloud_diversity_metrics,
    flatten_confusion_prefix,
    parse_prefix_swap_decoded_fasta,
    per_query_position_stats,
    pool_predictions_position_confusion,
    split_slot_and_baseline_queries,
)

try:
    from recomb2027.evaluation.knn_decode_utils import fasta_header, gid_str
except ImportError:  # pragma: no cover
    fasta_header = None
    gid_str = str


def _baseline_from_mut_json(mut_json: dict[str, Any]) -> dict[str, Any]:
    for entry in mut_json.get("per_query") or []:
        if isinstance(entry, dict) and str(entry.get("label", "")) == "baseline":
            return entry
    return {}


def _flatten_knn_block(block: Optional[dict[str, Any]], prefix: str) -> dict[str, Any]:
    if not isinstance(block, dict):
        return {}
    out: dict[str, Any] = {}
    change = block.get("confusion_change_from_source") or {}
    if isinstance(change, dict):
        for k in ("tp", "fp", "tn", "fn", "precision", "recall", "f1", "specificity"):
            if k in change:
                out[f"{prefix}_change_{k}"] = change[k]
    allele = block.get("confusion_target_allele") or {}
    if isinstance(allele, dict):
        for k in ("tp", "fp", "tn", "fn", "precision", "recall", "f1"):
            if k in allele:
                out[f"{prefix}_allele_{k}"] = allele[k]
    for k in (
        "n_knn_seqs",
        "n_required_source_to_target",
        "fraction_required_recovered",
        "label",
    ):
        if k in block:
            out[f"{prefix}_{k}"] = block[k]
    if block.get("sequence"):
        out[f"{prefix}_aa_consensus_sequence"] = block["sequence"]
    return out


def _extract_knn_consensus_from_mut_json(data: dict[str, Any]) -> dict[str, str]:
    """Pull AA kNN consensus sequences from enriched mutation-stats JSON."""
    out: dict[str, str] = {}
    for key in (
        "knn_consensus_baseline",
        "knn_consensus_slots",
        "knn_consensus_pooled",
    ):
        block = data.get(key)
        if isinstance(block, dict) and block.get("sequence"):
            out[key.replace("knn_consensus_", "")] = str(block["sequence"])
    return out


def build_row_summary(
    *,
    row_id: int,
    pairs_row: pd.Series,
    queries: dict[str, dict[str, Any]],
    branch_meta: dict[str, Any],
    prefix_count: Optional[int],
    mut_json: dict[str, Any],
    lineage_map: Optional[dict[str, str]] = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    """Build wide summary row, long slot rows, and FASTA lines for one series."""
    src_seq = str(pairs_row.get("seq_source", ""))
    tgt_seq = str(pairs_row.get("seq_target", ""))
    src_gid = gid_str(pairs_row.get("genome_id_source", ""))
    tgt_gid = gid_str(pairs_row.get("genome_id_target", ""))
    lin_src = resolve_lineage_for_gid(
        src_gid,
        pairs_row.get("lineage_source") if "lineage_source" in pairs_row.index else None,
        lineage_map,
    )
    lin_tgt = resolve_lineage_for_gid(
        tgt_gid,
        pairs_row.get("lineage_target") if "lineage_target" in pairs_row.index else None,
        lineage_map,
    )

    all_preds, baseline_preds, slot_preds = split_slot_and_baseline_queries(queries)
    all_seqs = list(all_preds.values())
    slot_seqs = list(slot_preds.values())

    pred_div_all = cloud_diversity_metrics(all_seqs)
    pred_div_slots = cloud_diversity_metrics(slot_seqs)

    branch_targets = list(branch_meta.get("branch_target_sequences") or [tgt_seq])
    branch_consensus = (
        branch_meta.get("branch_target_consensus")
        or build_target_consensus(branch_targets)
        or tgt_seq
    )

    pooled_all = pool_predictions_position_confusion(src_seq, tgt_seq, all_seqs)
    pooled_slots = pool_predictions_position_confusion(src_seq, tgt_seq, slot_seqs)
    baseline_seq = next(iter(baseline_preds.values()), "")
    pooled_baseline = (
        pool_predictions_position_confusion(src_seq, tgt_seq, [baseline_seq])
        if baseline_seq
        else {}
    )

    pred_consensus_all = build_pred_cloud_consensus(all_seqs)
    pred_consensus_slots = build_pred_cloud_consensus(slot_seqs)
    pred_consensus_stats = pool_predictions_position_confusion(
        src_seq, tgt_seq, [pred_consensus_all] if pred_consensus_all else []
    )

    best_qid, best_stats = best_slot_by_position_recall(
        src_seq, tgt_seq, slot_preds
    )

    blosum62 = load_blosum62_matrix()
    blast_src_tgt = compute_blast_alignment_metrics(src_seq, tgt_seq)

    summary: dict[str, Any] = {
        "row": row_id,
        "genome_id_source": src_gid,
        "genome_id_target": tgt_gid,
        "lineage_source": lin_src,
        "lineage_target": lin_tgt,
        "seq_source": src_seq,
        "seq_target": tgt_seq,
        "gt_hamming_source_target": compute_hamming_distance(src_seq, tgt_seq),
        "blosum_distance_source_to_target": round(
            compute_blosum_distance(src_seq, tgt_seq, blosum62), 4
        ),
        "blast_identity_source_to_target": round(blast_src_tgt["identity"], 6),
        "blast_similarity_source_to_target": round(blast_src_tgt["similarity"], 6),
        # Branching from labeled targets sharing prefix (MSA / pairs grouping)
        "n_branch_targets_same_prefix": branch_meta.get(
            "n_branch_targets_for_prefix", len(branch_targets)
        ),
        "n_branch_variable_sites_between_targets": len(
            branch_meta.get("branch_variable_sites_between_targets") or []
        ),
        "num_identical_prefix_series": prefix_count,
        "branch_required_sites_union": _serialize_cell(
            branch_meta.get("branch_required_sites_union", [])
        ),
        "branch_variable_sites_between_targets": _serialize_cell(
            branch_meta.get("branch_variable_sites_between_targets", [])
        ),
        "seq_target_consensus_branches": branch_consensus,
        # Branching from prefix-swap *predictions* (decoded cloud disagreement)
        "n_pred_queries": pred_div_all["n_sequences"],
        "n_pred_unique_sequences": pred_div_all["n_unique_sequences"],
        "n_pred_variable_sites": pred_div_all["n_variable_sites"],
        "n_pred_slots_unique_sequences": pred_div_slots["n_unique_sequences"],
        "n_pred_slots_variable_sites": pred_div_slots["n_variable_sites"],
        "pred_pairwise_hamming_mean": pred_div_all["pairwise_hamming_mean"],
        "pred_per_site_entropy_mean": pred_div_all["per_site_entropy_mean"],
        "pred_cloud_consensus_sequence": pred_consensus_all,
        "pred_slots_consensus_sequence": pred_consensus_slots,
        "pred_best_slot_query_id": best_qid,
        "pred_best_slot_recall": best_stats.get("recall"),
        "pred_best_slot_tp": best_stats.get("tp"),
        "pred_best_slot_fn": best_stats.get("fn"),
        "pred_best_slot_fp": best_stats.get("fp"),
        "pred_best_slot_tn": best_stats.get("tn"),
        "pred_best_slot_f1": best_stats.get("f1"),
    }
    summary.update(flatten_confusion_prefix(pooled_all, "pred_pooled_all"))
    summary.update(flatten_confusion_prefix(pooled_slots, "pred_pooled_slots"))
    summary.update(flatten_confusion_prefix(pooled_baseline, "pred_baseline"))
    summary.update(flatten_confusion_prefix(pred_consensus_stats, "pred_consensus_all"))

    if pooled_baseline.get("recall") is not None and pooled_all.get("recall") is not None:
        summary["pred_pooled_all_vs_baseline_delta_recall"] = (
            float(pooled_all["recall"]) - float(pooled_baseline["recall"])
        )
    if pooled_slots.get("recall") is not None and pooled_baseline.get("recall") is not None:
        summary["pred_pooled_slots_vs_baseline_delta_recall"] = (
            float(pooled_slots["recall"]) - float(pooled_baseline["recall"])
        )

    # kNN blocks from analyse_lineage_subst_knn mutation-stats JSON
    summary.update(_flatten_knn_block(_baseline_from_mut_json(mut_json), "knn_baseline"))
    summary.update(_flatten_knn_block(mut_json.get("union_all_slots"), "knn_slots"))
    summary.update(
        _flatten_knn_block(
            mut_json.get("union_all_queries")
            or mut_json.get("pooled_unique_latent_knn"),
            "knn_pooled",
        )
    )
    summary.update(_flatten_knn_block(mut_json.get("gt_blosum"), "knn_gt_blosum"))

    knn_cons = _extract_knn_consensus_from_mut_json(mut_json)
    for name, seq in knn_cons.items():
        summary[f"knn_aa_consensus_{name}_sequence"] = seq
        if seq:
            knn_pool = pool_predictions_position_confusion(src_seq, tgt_seq, [seq])
            summary.update(
                flatten_confusion_prefix(knn_pool, f"knn_aa_consensus_{name}")
            )

    if (
        summary.get("knn_pooled_change_recall") is not None
        and summary.get("knn_baseline_change_recall") is not None
    ):
        summary["knn_pooled_vs_baseline_delta_recall"] = (
            float(summary["knn_pooled_change_recall"])
            - float(summary["knn_baseline_change_recall"])
        )

    long_rows: list[dict[str, Any]] = []
    for qid, meta in sorted(
        queries.items(),
        key=lambda x: (
            0 if x[0] == "baseline" else 1,
            x[1].get("slot") if x[1].get("slot") is not None else 0,
            x[0],
        ),
    ):
        seq = str(meta.get("sequence", ""))
        qstats = per_query_position_stats(src_seq, tgt_seq, qid, seq)
        qstats.pop("query_id", None)
        long_rows.append(
            {
                "row": row_id,
                "genome_id_source": src_gid,
                "genome_id_target": tgt_gid,
                "lineage_source": lin_src,
                "lineage_target": lin_tgt,
                "query_id": qid,
                "slot": meta.get("slot"),
                "source_gid": meta.get("source_gid", ""),
                **qstats,
                "is_baseline": qid == "baseline",
            }
        )

    fasta_lines: list[str] = []
    if fasta_header is not None:
        fasta_lines.extend(
            _fasta_records_for_row(
                row_id=row_id,
                src_gid=src_gid,
                tgt_gid=tgt_gid,
                src_seq=src_seq,
                tgt_seq=tgt_seq,
                all_preds=all_preds,
                pred_consensus_all=pred_consensus_all,
                pred_consensus_slots=pred_consensus_slots,
                pooled_all=pooled_all,
                pooled_slots=pooled_slots,
                branch_consensus=branch_consensus,
                knn_cons=knn_cons,
            )
        )

    return summary, long_rows, fasta_lines


def _fasta_records_for_row(
    *,
    row_id: int,
    src_gid: str,
    tgt_gid: str,
    src_seq: str,
    tgt_seq: str,
    all_preds: dict[str, str],
    pred_consensus_all: str,
    pred_consensus_slots: str,
    pooled_all: dict[str, Any],
    pooled_slots: dict[str, Any],
    branch_consensus: str,
    knn_cons: dict[str, str],
) -> list[str]:
    branch_stats = pool_predictions_position_confusion(
        src_seq, tgt_seq, [branch_consensus] if branch_consensus else []
    )
    lines: list[str] = []
    lines.append(fasta_header("source", src_gid, extras=[f"row={row_id}"]))
    lines.append(src_seq)
    lines.append(fasta_header("target_labeled", tgt_gid, extras=[f"row={row_id}"]))
    lines.append(tgt_seq)
    lines.append(
        fasta_header(
            "branch_targets_consensus",
            "",
            extras=[
                f"row={row_id}",
                f"tp={branch_stats.get('tp', '')}",
                f"fn={branch_stats.get('fn', '')}",
            ],
        )
    )
    lines.append(branch_consensus)

    for qid, seq in sorted(all_preds.items()):
        st = pool_predictions_position_confusion(src_seq, tgt_seq, [seq])
        lines.append(
            fasta_header(
                f"pred_{qid}",
                "",
                extras=[
                    f"row={row_id}",
                    f"query_id={qid}",
                    f"tp={st.get('tp', '')}",
                    f"fn={st.get('fn', '')}",
                    f"fp={st.get('fp', '')}",
                    f"tn={st.get('tn', '')}",
                ],
            )
        )
        lines.append(seq)

    if pred_consensus_all:
        lines.append(
            fasta_header(
                "pred_cloud_consensus",
                "",
                extras=[
                    f"row={row_id}",
                    f"n_members={len(all_preds)}",
                    f"tp={pooled_all.get('tp', '')}",
                    f"fn={pooled_all.get('fn', '')}",
                ],
            )
        )
        lines.append(pred_consensus_all)

    if pred_consensus_slots and pred_consensus_slots != pred_consensus_all:
        lines.append(
            fasta_header(
                "pred_slots_consensus",
                "",
                extras=[
                    f"row={row_id}",
                    f"tp={pooled_slots.get('tp', '')}",
                    f"fn={pooled_slots.get('fn', '')}",
                ],
            )
        )
        lines.append(pred_consensus_slots)

    for name, seq in knn_cons.items():
        if not seq:
            continue
        st = pool_predictions_position_confusion(src_seq, tgt_seq, [seq])
        lines.append(
            fasta_header(
                f"knn_aa_consensus_{name}",
                "",
                extras=[
                    f"row={row_id}",
                    f"tp={st.get('tp', '')}",
                    f"fn={st.get('fn', '')}",
                ],
            )
        )
        lines.append(seq)

    return lines


def build_prefix_swap_report(
    *,
    val_pairs_path: Union[str, Path],
    decoded_fasta_path: Union[str, Path],
    output_dir: Union[str, Path],
    json_dir: Optional[Union[str, Path]] = None,
    series_npz: Optional[Union[str, Path]] = None,
    msa_path: Optional[Union[str, Path]] = None,
    metadata_csv: Optional[Union[str, Path]] = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lineage_map: Optional[dict[str, str]] = None
    meta_path = Path(metadata_csv) if metadata_csv is not None else None
    if meta_path is None and Path(METADATA_CSV_DEFAULT).is_file():
        meta_path = Path(METADATA_CSV_DEFAULT)
        print(f"Using default metadata CSV: {meta_path}")
    if meta_path is not None and meta_path.is_file():
        lineage_map = load_gid_lineage_map(meta_path)
        print(f"Loaded {len(lineage_map)} genome→lineage entries from {meta_path}")

    df_pairs = pd.read_csv(
        val_pairs_path,
        dtype={"genome_id_source": "string", "genome_id_target": "string"},
        low_memory=False,
    )
    decoded_by_row = parse_prefix_swap_decoded_fasta(str(decoded_fasta_path))
    json_root = Path(json_dir) if json_dir else None
    if json_root is not None and not json_root.is_dir():
        raise SystemExit(
            f"--json-dir {json_root} does not exist. Without it every kNN column "
            "would be silently empty; fix the path or omit --json-dir."
        )

    prefix_counts: dict[int, int] = {}
    row_id_to_prefix: dict[int, tuple[str, ...]] = {}
    if series_npz and msa_path:
        row_id_to_prefix, prefix_counts = compute_series_prefix_keys(
            series_npz, msa_path
        )
    elif series_npz or msa_path:
        raise SystemExit("Both --series-npz and --msa are required for branch metadata.")

    branch_groups = build_branch_groups(df_pairs, row_id_to_prefix)
    branch_metadata_by_row = build_branch_metadata_by_row(df_pairs, branch_groups)

    summary_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    fasta_lines: list[str] = []

    n_no_queries = 0
    n_json_found = 0
    for idx, row in df_pairs.iterrows():
        row_id = _row_id_from_pairs_row(row, idx)
        if row_id % 100 == 0:
            print(f"Processing row {row_id} / {len(df_pairs)}.")

        queries = decoded_by_row.get(row_id, {})
        if not queries:
            n_no_queries += 1
            continue

        mut_path = None
        if json_root is not None:
            for lookup_id in (row_id, int(idx)):
                for pattern in (
                    f"knn_row{lookup_id}_mutation_stats.json",
                    f"lineage_subst_row{lookup_id}_mutation_stats.json",
                ):
                    candidate = json_root / pattern
                    if candidate.is_file():
                        mut_path = candidate
                        break
                if mut_path is not None:
                    break
        mut_json = _load_json_if_exists(mut_path) if mut_path else {}
        n_json_found += int(bool(mut_json))

        summary, slot_long, row_fasta = build_row_summary(
            row_id=row_id,
            pairs_row=row,
            queries=queries,
            branch_meta=branch_metadata_by_row.get(row_id, {}),
            prefix_count=prefix_counts.get(row_id),
            mut_json=mut_json,
            lineage_map=lineage_map,
        )
        summary_rows.append(summary)
        long_rows.extend(slot_long)
        fasta_lines.extend(row_fasta)

    print(
        f"Decoded predictions cover {len(df_pairs) - n_no_queries}/{len(df_pairs)} "
        "validation rows."
    )
    if n_no_queries:
        print(
            f"WARNING: {n_no_queries} validation rows have no decoded queries and "
            "are omitted from the report."
        )
    if json_root is not None:
        print(f"kNN mutation-stats JSON found for {n_json_found}/{len(summary_rows)} rows.")
        if n_json_found == 0:
            print(
                f"WARNING: no knn_row*_mutation_stats.json / "
                f"lineage_subst_row*_mutation_stats.json found in {json_root}; "
                "kNN columns will be missing."
            )
    wide_path = output_dir / "prefix_swap_evaluation_report.csv"
    long_path = output_dir / "prefix_swap_slots_long.csv"
    fasta_path = output_dir / "prefix_swap_consensus.fasta"

    pd.DataFrame(summary_rows).to_csv(wide_path, index=False)
    pd.DataFrame(long_rows).to_csv(long_path, index=False)
    fasta_path.write_text("\n".join(fasta_lines) + ("\n" if fasta_lines else ""), encoding="utf-8")

    print(f"Wrote {wide_path} ({len(summary_rows)} rows)")
    print(f"Wrote {long_path} ({len(long_rows)} rows)")
    print(f"Wrote {fasta_path}")

    return {
        "wide_csv": str(wide_path),
        "long_csv": str(long_path),
        "consensus_fasta": str(fasta_path),
        "n_series": len(summary_rows),
        "n_slot_rows": len(long_rows),
        "metadata_csv": str(meta_path) if meta_path is not None else None,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Prefix-swap cloud evaluation report.")
    p.add_argument("--val-pairs", type=Path, required=True)
    p.add_argument("--decoded-fasta", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--json-dir",
        type=Path,
        default=None,
        help="Directory with knn_row*_mutation_stats.json from analyse_lineage_subst_knn.",
    )
    p.add_argument("--series-npz", type=Path, default=None)
    p.add_argument("--msa", type=Path, default=None)
    p.add_argument(
        "--metadata-csv",
        type=Path,
        default=None,
        help=(
            "Genome metadata with lineage for source/target columns in CSV output. "
            f"Default: {METADATA_CSV_DEFAULT} when present."
        ),
    )
    args = p.parse_args()
    result = build_prefix_swap_report(
        val_pairs_path=args.val_pairs,
        decoded_fasta_path=args.decoded_fasta,
        output_dir=args.output_dir,
        json_dir=args.json_dir,
        series_npz=args.series_npz,
        msa_path=args.msa,
        metadata_csv=args.metadata_csv,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
