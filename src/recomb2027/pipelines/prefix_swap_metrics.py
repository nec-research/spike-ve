#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     prefix_swap_metrics.py
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
"""Metrics for prefix-swap prediction clouds (decoded baseline + slots)."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Optional

import numpy as np

from recomb2027.pipelines.generate_evaluation_report import (
    build_target_consensus,
    closest_knn_confusion_stats,
    compute_hamming_distance,
    extract_mutations,
)

_HEADER_KV_RE = re.compile(r"(\w+)=([^\s]+)")


def _safe_div(num: float, den: float) -> Optional[float]:
    if den <= 0:
        return None
    return float(num) / float(den)


def parse_header_kv(header: str) -> dict[str, str]:
    """Parse ``key=value`` tokens from a FASTA header line."""
    return {m.group(1): m.group(2) for m in _HEADER_KV_RE.finditer(header)}


def parse_prefix_swap_decoded_fasta(
    fasta_path: str,
) -> dict[int, dict[str, dict[str, Any]]]:
    """Parse multi-query decoded FASTA into ``row -> query_id -> {seq, slot, ...}``."""
    by_row: dict[int, dict[str, dict[str, Any]]] = {}
    current_row: Optional[int] = None
    current_meta: dict[str, Any] = {}
    current_seq: list[str] = []

    def _flush() -> None:
        nonlocal current_row, current_meta, current_seq
        if current_row is None:
            return
        qid = str(current_meta.get("query_id", "unknown"))
        by_row.setdefault(current_row, {})[qid] = {
            **current_meta,
            "sequence": "".join(current_seq),
        }
        current_seq = []

    with open(fasta_path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                _flush()
                meta = parse_header_kv(line[1:])
                if "row" not in meta:
                    current_row = None
                    current_meta = {}
                    current_seq = []
                    continue
                try:
                    current_row = int(meta["row"])
                except ValueError:
                    current_row = None
                    current_meta = {}
                    current_seq = []
                    continue
                current_meta = meta
                if "slot" in meta:
                    try:
                        current_meta["slot"] = int(meta["slot"])
                    except ValueError:
                        pass
                current_seq = []
            else:
                current_seq.append(line)
        _flush()
    if not by_row:
        raise ValueError(
            f"No FASTA records with a 'row=<int>' header found in {fasta_path}. "
            "Pass the decoded *.fasta (not the *_decoded.csv summary)."
        )
    return by_row


def variable_sites_among_sequences(seqs: list[str]) -> list[int]:
    """1-based positions where sequences disagree."""
    seqs = [s for s in seqs if s]
    if len(seqs) < 2:
        return []
    n = min(len(s) for s in seqs)
    sites: list[int] = []
    for i in range(n):
        chars = {s[i] for s in seqs}
        if len(chars) > 1:
            sites.append(i + 1)
    return sites


def cloud_diversity_metrics(seqs: list[str]) -> dict[str, Any]:
    """Diversity statistics for a list of aligned sequences."""
    seqs = [s for s in seqs if s]
    if not seqs:
        return {
            "n_sequences": 0,
            "n_unique_sequences": 0,
            "pairwise_hamming_mean": None,
            "pairwise_hamming_max": None,
            "per_site_entropy_mean": None,
            "n_variable_sites": 0,
        }

    unique = list(dict.fromkeys(seqs))
    pairwise: list[int] = []
    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            pairwise.append(compute_hamming_distance(unique[i], unique[j]))

    n = min(len(s) for s in seqs)
    entropies: list[float] = []
    for i in range(n):
        counts = Counter(s[i] for s in seqs if len(s) > i)
        total = sum(counts.values())
        if total <= 1:
            continue
        ent = 0.0
        for c in counts.values():
            p = c / total
            ent -= p * math.log2(p)
        entropies.append(ent)

    var_sites = variable_sites_among_sequences(seqs)
    return {
        "n_sequences": len(seqs),
        "n_unique_sequences": len(unique),
        "pairwise_hamming_mean": (
            float(np.mean(pairwise)) if pairwise else 0.0
        ),
        "pairwise_hamming_max": (
            int(max(pairwise)) if pairwise else 0
        ),
        "per_site_entropy_mean": (
            float(np.mean(entropies)) if entropies else 0.0
        ),
        "n_variable_sites": len(var_sites),
        "variable_sites": var_sites,
    }


def pool_predictions_position_confusion(
    source_seq: str,
    target_seq: str,
    pred_seqs: list[str],
) -> dict[str, Any]:
    """Pooled TP/FN/FP/TN across a prediction cloud (any-slot logic).

    At each site, the pool is considered to have **changed** from source if
    any member sequence differs from source at that site (same as pooled kNN
    change-from-source logic).
    """
    pred_seqs = [s for s in pred_seqs if s]
    if not source_seq or not target_seq or not pred_seqs:
        return {
            "tp": None,
            "fn": None,
            "fp": None,
            "tn": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "specificity": None,
        }

    n = min(len(source_seq), len(target_seq), min(len(s) for s in pred_seqs))
    tp = fp = tn = fn = 0
    for i in range(n):
        required = source_seq[i] != target_seq[i]
        changed = any(s[i] != source_seq[i] for s in pred_seqs if len(s) > i)
        if required and changed:
            tp += 1
        elif required and not changed:
            fn += 1
        elif (not required) and changed:
            fp += 1
        else:
            tn += 1

    prec = _safe_div(tp, tp + fp)
    rec = _safe_div(tp, tp + fn)
    spec = _safe_div(tn, tn + fp)
    f1 = None
    if prec is not None and rec is not None and (prec + rec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec)

    return {
        "tp": int(tp),
        "fn": int(fn),
        "fp": int(fp),
        "tn": int(tn),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "specificity": spec,
    }


def best_slot_by_position_recall(
    source_seq: str,
    target_seq: str,
    queries: dict[str, str],
    *,
    slots_only: bool = True,
) -> tuple[str, dict[str, Any]]:
    """Pick query with highest position-level recall vs labeled target.

    When ``slots_only`` is true (default), ``baseline`` and slot=-1 queries are
    excluded so the winner is the best prefix-swap slot.
    """
    best_qid = ""
    best_stats: dict[str, Any] = {}
    best_key: Optional[tuple[float, float, int]] = None

    for qid, seq in queries.items():
        if not seq:
            continue
        if slots_only and (qid == "baseline" or qid.startswith("baseline")):
            continue
        stats = closest_knn_confusion_stats(source_seq, target_seq, seq)
        tp = int(stats.get("closest_knn_tp") or 0)
        fn = int(stats.get("closest_knn_fn") or 0)
        rec = _safe_div(float(tp), float(tp + fn))
        recall = float(rec) if rec is not None else -1.0
        f1 = stats.get("closest_knn_f1")
        f1_val = float(f1) if f1 is not None else -1.0
        # Rank by recall, then F1, then TP. (Compare with explicit None checks:
        # an F1 of 0.0 is a valid value and must not be treated as missing.)
        key = (recall, f1_val, tp)
        if best_key is None or key > best_key:
            best_key = key
            best_qid = qid
            best_stats = {
                "tp": stats.get("closest_knn_tp"),
                "fn": stats.get("closest_knn_fn"),
                "fp": stats.get("closest_knn_fp"),
                "tn": stats.get("closest_knn_tn"),
                "recall": recall,
                "f1": stats.get("closest_knn_f1"),
            }
    return best_qid, best_stats


def per_query_position_stats(
    source_seq: str,
    target_seq: str,
    query_id: str,
    seq: str,
) -> dict[str, Any]:
    """Position-level confusion for one decoded query."""
    stats = closest_knn_confusion_stats(source_seq, target_seq, seq)
    return {
        "query_id": query_id,
        "tp": stats.get("closest_knn_tp"),
        "fn": stats.get("closest_knn_fn"),
        "fp": stats.get("closest_knn_fp"),
        "tn": stats.get("closest_knn_tn"),
        "recall": _safe_div(
            float(stats.get("closest_knn_tp") or 0),
            float((stats.get("closest_knn_tp") or 0) + (stats.get("closest_knn_fn") or 0)),
        ),
        "f1": stats.get("closest_knn_f1"),
        "hamming_to_target": compute_hamming_distance(seq, target_seq) if seq else None,
        "hamming_to_source": compute_hamming_distance(seq, source_seq) if seq else None,
    }


def build_pred_cloud_consensus(seqs: list[str]) -> str:
    """Unanimity consensus: any column where members disagree becomes ``X``.

    (Not a majority vote.) At position level an ``X`` counts as "changed from
    source", so its TP/FN/FP equal the pooled any-member logic.
    """
    return build_target_consensus(seqs)


def split_slot_and_baseline_queries(
    queries: dict[str, dict[str, Any]],
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Return (all, baseline_only, slots_only) query_id -> sequence maps."""
    all_q: dict[str, str] = {}
    baseline_q: dict[str, str] = {}
    slots_q: dict[str, str] = {}
    for qid, meta in queries.items():
        seq = str(meta.get("sequence", ""))
        if not seq:
            continue
        all_q[qid] = seq
        if qid == "baseline":
            baseline_q[qid] = seq
            continue
        try:
            slot_i = int(meta.get("slot", 0))
        except (TypeError, ValueError):
            slot_i = 0
        if slot_i < 0:
            baseline_q[qid] = seq
        else:
            slots_q[qid] = seq
    return all_q, baseline_q, slots_q


def flatten_confusion_prefix(
    stats: dict[str, Any],
    prefix: str,
) -> dict[str, Any]:
    """Flatten pooled confusion dict with a column prefix."""
    return {
        f"{prefix}_tp": stats.get("tp"),
        f"{prefix}_fn": stats.get("fn"),
        f"{prefix}_fp": stats.get("fp"),
        f"{prefix}_tn": stats.get("tn"),
        f"{prefix}_precision": stats.get("precision"),
        f"{prefix}_recall": stats.get("recall"),
        f"{prefix}_f1": stats.get("f1"),
        f"{prefix}_specificity": stats.get("specificity"),
    }


def mutation_lists_for_sequence(
    source_seq: str,
    target_seq: str,
    pred_seq: str,
) -> tuple[list[int], list[str], list[str]]:
    """Required sites/alleles and which are recovered in pred (allele-exact)."""
    gt_pos, gt_alleles = extract_mutations(source_seq, target_seq)
    _, pred_alleles = extract_mutations(source_seq, pred_seq)
    pred_set = set(pred_alleles)
    tp_alleles = sorted(set(gt_alleles) & pred_set)
    fn_alleles = sorted(set(gt_alleles) - pred_set)
    return gt_pos, tp_alleles, fn_alleles
