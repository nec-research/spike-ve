#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     generate_evaluation_report.py
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
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from Bio.Align import substitution_matrices

from recomb2027.data.seq_pairwise_distance import fast_pairwise_blosum_distance


def load_blosum62_matrix():
    """Load BLOSUM62 matrix for distance calculations."""
    try:
        return substitution_matrices.load("BLOSUM62")
    except Exception:
        # Fallback if Biopython matrix load differs by version
        from Bio.SubsMat import MatrixInfo
        return MatrixInfo.blosum62


def compute_hamming_distance(seq1: str, seq2: str) -> int:
    """Compute Hamming distance between two equal-length aligned sequences."""
    min_len = min(len(seq1), len(seq2))
    dist = sum(1 for a, b in zip(seq1[:min_len], seq2[:min_len]) if a != b)
    dist += abs(len(seq1) - len(seq2))
    return dist


def compute_blosum_distance(seq1: str, seq2: str, blosum_matrix = None) -> float:
    """Computes a normalized distance."""    
    return fast_pairwise_blosum_distance(seq1, seq2)


def compute_blosum_distance_(seq1: str, seq2: str, blosum_matrix) -> float:
    """Computes a normalized BLOSUM62 distance."""
    score = 0.0
    max_score = 0.0
    min_len = min(len(seq1), len(seq2))

    for i in range(min_len):
        a, b = seq1[i], seq2[i]
        pair = (a, b)
        rev_pair = (b, a)

        if pair in blosum_matrix:
            s = blosum_matrix[pair]
        elif rev_pair in blosum_matrix:
            s = blosum_matrix[rev_pair]
        else:
            s = 0.0

        score += s

    for i in range(min_len):
        a = seq1[i]
        score_a = blosum_matrix.get((a, a), 4.0)
        max_score += score_a

    return max(0.0, float(max_score - score))


def parse_fasta(fasta_path: str) -> dict:
    """Parses FASTA containing predicted sequences indexed by row number in header."""
    decoded_seqs = {}
    current_row = None
    current_seq = []

    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_row is not None:
                    decoded_seqs[current_row] = "".join(current_seq)
                header_parts = line[1:].split()
                row_part = [p for p in header_parts if p.startswith("row=")][0]
                current_row = int(row_part.split("=")[1])
                current_seq = []
            else:
                current_seq.append(line)
        if current_row is not None:
            decoded_seqs[current_row] = "".join(current_seq)

    if not decoded_seqs:
        raise ValueError(
            f"No FASTA records with a 'row=<int>' header found in {fasta_path}. "
            "Is this a FASTA file (e.g. *_decoded.fasta, not the *_decoded.csv "
            "summary)?"
        )
    return decoded_seqs


def extract_mutations(src_seq: str, tgt_seq: str):
    """Generates a list of mutation notation strings (e.g., 'E191Q') 1-indexed."""
    positions = []
    mutations = []
    min_len = min(len(src_seq), len(tgt_seq))
    for i in range(min_len):
        if src_seq[i] != tgt_seq[i]:
            positions.append(i + 1)
            mutations.append(f"{src_seq[i]}{i+1}{tgt_seq[i]}")
    return positions, mutations


def _gid_str(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace")
    text = str(value).strip()
    if text.lower() in {"nan", "<na>", "none"}:
        return ""
    return text


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, dict) else {}


def parse_knn_analysis_fasta(path: Path) -> dict[str, str]:
    """Parse ``knn_baseline_row*.fasta`` into named sequences.

    Returns keys such as ``source``, ``target``, ``knn_0``, ``knn_consensus``.
    The closest-to-prediction neighbor is ``knn_0`` (rank 0).
    """
    out: dict[str, str] = {}
    if not path.is_file():
        return out

    current_key: Optional[str] = None
    current_seq: list[str] = []

    def _flush() -> None:
        nonlocal current_key, current_seq
        if current_key is not None:
            out[current_key] = "".join(current_seq)
        current_key = None
        current_seq = []

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                _flush()
                header = line[1:]
                kind = header.split("|", 1)[0].strip()
                current_key = kind
                current_seq = []
            else:
                if current_key is not None:
                    current_seq.append(line)
        _flush()
    return out


def compute_identical_prefix_series_counts(
    series_npz: Union[str, Path],
    msa_path: Union[str, Path],
) -> dict[int, int]:
    """Count series that share an identical sequence prefix."""
    _, counts = compute_series_prefix_keys(series_npz, msa_path)
    return counts


def _source_gids_array_from_npz(data: np.lib.npyio.NpzFile) -> Optional[np.ndarray]:
    """Return per-row source genome ids from lineage-subst or GA prediction NPZ."""
    for key in (
        "source_gids",
        "test_source_gids",
        "val_source_gids",
        "cal_source_gids",
    ):
        if key in data:
            return np.asarray(data[key], dtype=object)
    return None


def compute_series_prefix_keys(
    series_npz: Union[str, Path],
    msa_path: Union[str, Path],
) -> tuple[dict[int, tuple[str, ...]], dict[int, int]]:
    """Map series row index → prefix sequence tuple and prefix multiplicity.

    The prefix is the MSA sequences of all steps except the final target step
    (``gids[:-1]``). The last prefix step is the source / penultimate genome.

    For lineage-substitution prediction NPZ files that only store ``source_gids``
    (no ``sample_gids`` path), the prefix is the MSA sequence of the source
    genome for that row (a single-step prefix).
    """
    from recomb2027.latent.data_series.latent_decoder_io import (
        parse_msa_gids,
        resolve_genome_id,
    )

    series_npz = Path(series_npz)
    data = np.load(series_npz, allow_pickle=True)

    print(f"Loading MSA for prefix keys from {msa_path} ...")
    gid_to_seq = parse_msa_gids(msa_path)

    row_to_prefix: dict[int, tuple[str, ...]] = {}
    prefix_keys: list[Optional[tuple[str, ...]]] = []

    if "sample_gids" in data:
        sample_gids = np.asarray(data["sample_gids"])
        for i, row in enumerate(sample_gids):
            gids = [_gid_str(g) for g in np.asarray(row).tolist()]
            if len(gids) < 2:
                prefix_keys.append(None)
                continue
            seqs: list[str] = []
            ok = True
            for gid in gids[:-1]:
                key = resolve_genome_id(gid, gid_to_seq) or gid
                seq = gid_to_seq.get(key)
                if not seq:
                    ok = False
                    break
                seqs.append(seq)
            prefix = tuple(seqs) if ok else None
            prefix_keys.append(prefix)
            if prefix is not None:
                row_to_prefix[i] = prefix
    else:
        source_gids = _source_gids_array_from_npz(data)
        if source_gids is None:
            data.close()
            raise KeyError(
                f"{series_npz} missing sample_gids and source_gids; "
                "cannot compute prefix keys."
            )
        print(
            f"Using source_gids from {series_npz.name} "
            f"(lineage-subst format, n={len(source_gids)})"
        )
        for i, gid in enumerate(source_gids):
            gid_s = _gid_str(gid)
            key = resolve_genome_id(gid_s, gid_to_seq) or gid_s
            seq = gid_to_seq.get(key)
            prefix = (seq,) if seq else None
            prefix_keys.append(prefix)
            if prefix is not None:
                row_to_prefix[i] = prefix

    data.close()

    counts = Counter(k for k in prefix_keys if k is not None)
    row_counts = {
        i: int(counts[key])
        for i, key in enumerate(prefix_keys)
        if key is not None
    }
    return row_to_prefix, row_counts


def build_target_consensus(target_seqs: list[str]) -> str:
    """Consensus over branch targets; disagreeing columns become ``X``."""
    seqs = [s for s in target_seqs if s]
    if not seqs:
        return ""
    seq_len = len(seqs[0])
    consensus: list[str] = []
    for i in range(seq_len):
        chars = {seq[i] for seq in seqs if len(seq) > i}
        if len(chars) == 1:
            consensus.append(chars.pop())
        else:
            consensus.append("X")
    return "".join(consensus)


def branch_union_mutations(
    source_seq: str,
    branch_targets: list[str],
) -> tuple[list[int], list[str], list[int]]:
    """Union required sites/alleles and variable sites across branch targets."""
    sites_union: set[int] = set()
    alleles_union: set[str] = set()
    per_target_sites: list[set[int]] = []

    for tgt in branch_targets:
        if not tgt:
            continue
        sites, alleles = extract_mutations(source_seq, tgt)
        site_set = set(sites)
        per_target_sites.append(site_set)
        sites_union.update(site_set)
        alleles_union.update(alleles)

    variable_between_targets: set[int] = set()
    if len(per_target_sites) >= 2:
        for i in range(min(len(source_seq), max((len(t) for t in branch_targets), default=0))):
            alleles = {
                tgt[i]
                for tgt in branch_targets
                if tgt and i < len(tgt)
            }
            if len(alleles) >= 2:
                variable_between_targets.add(i + 1)

    return (
        sorted(sites_union),
        sorted(alleles_union),
        sorted(variable_between_targets),
    )


def branch_intersection_mutations(
    source_seq: str,
    branch_targets: list[str],
) -> tuple[list[int], list[str]]:
    """Intersection of required sites/alleles shared by every branch target."""
    if not branch_targets:
        return [], []

    site_sets: list[set[int]] = []
    allele_sets: list[set[str]] = []
    for tgt in branch_targets:
        if not tgt:
            continue
        sites, alleles = extract_mutations(source_seq, tgt)
        site_sets.append(set(sites))
        allele_sets.append(set(alleles))

    if not site_sets:
        return [], []

    sites_all = set.intersection(*site_sets)
    alleles_all = set.intersection(*allele_sets) if allele_sets else set()
    return sorted(sites_all), sorted(alleles_all)


def any_branch_allele_metrics(
    source_seq: str,
    pred_seq: str,
    branch_targets: list[str],
) -> dict[str, Any]:
    """Allele-exact TP/FN against the union of branch target mutations."""
    empty: dict[str, Any] = {
        "decoder_tp_any": None,
        "decoder_fn_any": None,
        "decoder_recall_any": None,
        "decoder_matches_any_branch": False,
        "min_hamming_pred_to_branch_targets": None,
    }
    if not source_seq or not branch_targets:
        return empty

    _, union_alleles, _ = branch_union_mutations(source_seq, branch_targets)
    union_set = set(union_alleles)
    if not pred_seq:
        return {
            **empty,
            "decoder_tp_any": 0,
            "decoder_fn_any": len(union_set),
            "decoder_recall_any": 0.0 if union_set else None,
        }

    _, pred_alleles = extract_mutations(source_seq, pred_seq)
    pred_set = set(pred_alleles)
    tp_any = len(union_set & pred_set)
    fn_any = len(union_set - pred_set)
    recall_any = _safe_div(tp_any, tp_any + fn_any)
    min_ham = min(compute_hamming_distance(pred_seq, tgt) for tgt in branch_targets)
    return {
        "decoder_tp_any": int(tp_any),
        "decoder_fn_any": int(fn_any),
        "decoder_recall_any": recall_any,
        "decoder_matches_any_branch": any(pred_seq == tgt for tgt in branch_targets),
        "min_hamming_pred_to_branch_targets": int(min_ham),
    }


def all_branch_allele_metrics(
    source_seq: str,
    pred_seq: str,
    branch_targets: list[str],
) -> dict[str, Optional[int]]:
    """Allele-exact TP/FN against mutations present in ALL branch targets."""
    empty: dict[str, Optional[int]] = {
        "decoder_tp_all_alleles": None,
        "decoder_fn_all_alleles": None,
    }
    if not source_seq or not branch_targets:
        return empty

    _, alleles_all = branch_intersection_mutations(source_seq, branch_targets)
    all_set = set(alleles_all)
    if not pred_seq:
        return {
            "decoder_tp_all_alleles": 0,
            "decoder_fn_all_alleles": len(all_set),
        }

    _, pred_alleles = extract_mutations(source_seq, pred_seq)
    pred_set = set(pred_alleles)
    return {
        "decoder_tp_all_alleles": int(len(all_set & pred_set)),
        "decoder_fn_all_alleles": int(len(all_set - pred_set)),
    }


def _intersection_position_tp_fn(
    source_seq: str,
    pred_seq: str,
    branch_targets: list[str],
) -> tuple[int, int]:
    """TP/FN at sites required in every branch (position changes from source)."""
    if not source_seq or not pred_seq or not branch_targets:
        return 0, 0

    sites_all, _ = branch_intersection_mutations(source_seq, branch_targets)
    if not sites_all:
        return 0, 0

    n = min(len(source_seq), len(pred_seq))
    tp_all = fn_all = 0
    for pos in sites_all:
        idx = pos - 1
        if idx < 0 or idx >= n:
            continue
        if pred_seq[idx] != source_seq[idx]:
            tp_all += 1
        else:
            fn_all += 1
    return tp_all, fn_all


def all_branch_position_metrics(
    source_seq: str,
    decoder_seq: str,
    knn_seq: str,
    branch_targets: list[str],
) -> dict[str, Optional[int]]:
    """Position-level TP/FN against sites required in ALL branch targets."""
    empty = {
        "decoder_tp_all": None,
        "decoder_fn_all": None,
        "closest_knn_tp_all": None,
        "closest_knn_fn_all": None,
    }
    if not source_seq or not branch_targets:
        return empty

    dec_tp, dec_fn = _intersection_position_tp_fn(
        source_seq, decoder_seq or "", branch_targets
    )
    knn_tp, knn_fn = _intersection_position_tp_fn(
        source_seq, knn_seq or "", branch_targets
    )
    return {
        "decoder_tp_all": int(dec_tp),
        "decoder_fn_all": int(dec_fn),
        "closest_knn_tp_all": int(knn_tp),
        "closest_knn_fn_all": int(knn_fn),
    }


def any_branch_position_metrics(
    source_seq: str,
    pred_seq: str,
    branch_targets: list[str],
) -> dict[str, Optional[float]]:
    """Position-level TP/FN against union of branch-required sites."""
    empty = {
        "closest_knn_tp_any": None,
        "closest_knn_fn_any": None,
        "closest_knn_recall_any": None,
    }
    if not source_seq or not pred_seq or not branch_targets:
        return empty

    sites_union, _, _ = branch_union_mutations(source_seq, branch_targets)
    union_set = set(sites_union)
    if not union_set:
        return {
            "closest_knn_tp_any": 0,
            "closest_knn_fn_any": 0,
            "closest_knn_recall_any": None,
        }

    n = min(len(source_seq), len(pred_seq))
    tp_any = fn_any = 0
    for pos in union_set:
        idx = pos - 1
        if idx < 0 or idx >= n:
            continue
        if pred_seq[idx] != source_seq[idx]:
            tp_any += 1
        else:
            fn_any += 1

    return {
        "closest_knn_tp_any": int(tp_any),
        "closest_knn_fn_any": int(fn_any),
        "closest_knn_recall_any": _safe_div(tp_any, tp_any + fn_any),
    }


def _row_id_from_pairs_row(row: pd.Series, idx: int) -> int:
    if "series_row" in row.index and pd.notna(row.get("series_row")):
        return int(row["series_row"])
    if "index" in row.index and pd.notna(row.get("index")):
        return int(row["index"])
    return int(idx)


def _lookup_pairs_row(df_pairs: pd.DataFrame, row_id: int) -> Optional[pd.Series]:
    if "series_row" in df_pairs.columns:
        match = df_pairs[df_pairs["series_row"].astype("Int64") == row_id]
        if not match.empty:
            return match.iloc[0]
    if "index" in df_pairs.columns:
        match = df_pairs[df_pairs["index"].astype("Int64") == row_id]
        if not match.empty:
            return match.iloc[0]
    return None


def build_branch_groups(
    df_pairs: pd.DataFrame,
    row_id_to_prefix: Optional[dict[int, tuple[str, ...]]] = None,
) -> dict[tuple[Any, ...], list[int]]:
    """Group validation rows that share the same evolutionary prefix."""
    groups: dict[tuple[Any, ...], list[int]] = {}
    for idx, row in df_pairs.iterrows():
        row_id = _row_id_from_pairs_row(row, idx)
        src_seq = row.get("seq_source", "")
        if not isinstance(src_seq, str):
            src_seq = "" if pd.isna(src_seq) else str(src_seq)
        prefix = (row_id_to_prefix or {}).get(row_id)
        if prefix is not None:
            key: tuple[Any, ...] = ("prefix",) + prefix
        else:
            key = ("source", src_seq)
        groups.setdefault(key, []).append(row_id)
    return groups


def build_branch_metadata_by_row(
    df_pairs: pd.DataFrame,
    groups: dict[tuple[Any, ...], list[int]],
) -> dict[int, dict[str, Any]]:
    """Precompute branch metadata once per prefix/source group."""
    metadata_by_row: dict[int, dict[str, Any]] = {}
    for member_ids in groups.values():
        targets: list[str] = []
        target_gids: list[str] = []
        seen_tgt: set[str] = set()
        src_seq = ""
        for mid in member_ids:
            row = _lookup_pairs_row(df_pairs, mid)
            if row is None:
                continue
            if not src_seq:
                src_seq = str(row.get("seq_source", ""))
            tgt = str(row.get("seq_target", ""))
            gid = _gid_str(row.get("genome_id_target", ""))
            if tgt and tgt not in seen_tgt:
                seen_tgt.add(tgt)
                targets.append(tgt)
                if gid:
                    target_gids.append(gid)

        sites_union, alleles_union, variable_sites = branch_union_mutations(
            src_seq, targets
        )
        sites_all, alleles_all = branch_intersection_mutations(src_seq, targets)
        meta = {
            "n_branch_targets_for_prefix": len(targets),
            "branch_target_gids": target_gids,
            "branch_target_sequences": targets,
            "branch_target_consensus": build_target_consensus(targets),
            "branch_required_sites_union": sites_union,
            "branch_required_alleles_union": alleles_union,
            "branch_required_sites_intersection": sites_all,
            "branch_required_alleles_intersection": alleles_all,
            "branch_variable_sites_between_targets": variable_sites,
        }
        for mid in member_ids:
            metadata_by_row[mid] = meta
    return metadata_by_row


def _serialize_cell(value: Any) -> Any:
    """Make nested JSON values CSV-friendly."""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def _safe_div(num: float, den: float) -> Optional[float]:
    if den <= 0:
        return None
    return float(num) / float(den)


_blast_metrics_cache: dict[tuple[str, str], dict[str, float]] = {}


def _blosum_pair_score(matrix: Any, a: str, b: str) -> float:
    pair = (a, b)
    if pair in matrix:
        return float(matrix[pair])
    rev = (b, a)
    if rev in matrix:
        return float(matrix[rev])
    return 0.0


def compute_blast_alignment_metrics(
    seq1: str,
    seq2: str,
) -> dict[str, float]:
    """BLAST-like local alignment metrics using BLOSUM62 (protein).

    Returns fractions in [0, 1] for identity and similarity (positives), plus
    the raw alignment score summed over aligned columns (gaps penalized via the
    aligner). Similarity counts identical residues and conservative substitutions
    (pairs with positive BLOSUM62 score).
    """
    key = (seq1, seq2)
    if key in _blast_metrics_cache:
        return _blast_metrics_cache[key]

    from Bio import Align

    query = seq1.replace("-", "").replace(".", "")
    subject = seq2.replace("-", "").replace(".", "")
    empty = {
        "identity": 0.0,
        "similarity": 0.0,
        "raw_score": 0.0,
        "aligned_length": 0.0,
    }
    if not query or not subject:
        _blast_metrics_cache[key] = empty
        return empty

    matrix = substitution_matrices.load("BLOSUM62")
    aligner = Align.PairwiseAligner()
    aligner.substitution_matrix = matrix
    aligner.mode = "local"
    aligner.open_gap_score = -11
    aligner.extend_gap_score = -1

    try:
        alignments = aligner.align(query, subject)
        best_alignment = alignments[0]
    except (IndexError, StopIteration, OverflowError):
        _blast_metrics_cache[key] = empty
        return empty

    query_ranges, subject_ranges = best_alignment.aligned
    identical = 0
    positives = 0
    aligned_positions = 0
    raw_score = float(best_alignment.score)
    for (qs, qe), (ss, se) in zip(query_ranges, subject_ranges):
        seg_len = qe - qs
        aligned_positions += seg_len
        for i in range(seg_len):
            a = query[qs + i]
            b = subject[ss + i]
            if a == b:
                identical += 1
                positives += 1
            elif _blosum_pair_score(matrix, a, b) > 0:
                positives += 1

    identity = identical / aligned_positions if aligned_positions else 0.0
    similarity = positives / aligned_positions if aligned_positions else 0.0
    result = {
        "identity": float(identity),
        "similarity": float(similarity),
        "raw_score": raw_score,
        "aligned_length": float(aligned_positions),
    }
    _blast_metrics_cache[key] = result
    return result


def compute_blast_identity(seq1: str, seq2: str) -> Optional[float]:
    """Backward-compatible wrapper returning alignment identity fraction."""
    return compute_blast_alignment_metrics(seq1, seq2)["identity"]


def change_from_source_confusion_stats(
    source_seq: str,
    target_seq: str,
    pred_seq: str,
) -> dict[str, Optional[int | float]]:
    """Position-level TP/FN/TN/FP/F1 for a prediction vs source→target changes.

    Same change-from-source definitions as ``compute_mutation_site_stats``:

      R  = sites where source ≠ target
      C  = sites where prediction differs from source
      TP = |R ∩ C|, FN = |R − C|, FP = |C − R|, TN = rest
    """
    empty: dict[str, Optional[int | float]] = {
        "tp": None,
        "fn": None,
        "tn": None,
        "fp": None,
        "f1": None,
    }
    if not source_seq or not target_seq or not pred_seq:
        return empty

    n = min(len(source_seq), len(target_seq), len(pred_seq))
    if n == 0:
        return empty

    src = source_seq[:n]
    tgt = target_seq[:n]
    pred = pred_seq[:n]

    tp = fp = tn = fn = 0
    for i in range(n):
        required = src[i] != tgt[i]
        changed = pred[i] != src[i]
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
    f1: Optional[float] = None
    if prec is not None and rec is not None and (prec + rec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec)

    return {
        "tp": int(tp),
        "fn": int(fn),
        "tn": int(tn),
        "fp": int(fp),
        "f1": f1,
    }


def closest_knn_confusion_stats(
    source_seq: str,
    target_seq: str,
    closest_seq: str,
) -> dict[str, Optional[float]]:
    """TP/FN/TN/FP/F1 for one closest-kNN sequence vs source→target changes."""
    stats = change_from_source_confusion_stats(source_seq, target_seq, closest_seq)
    if stats["tp"] is None:
        return {
            "closest_knn_tp": None,
            "closest_knn_fn": None,
            "closest_knn_tn": None,
            "closest_knn_fp": None,
            "closest_knn_f1": None,
        }
    return {
        "closest_knn_tp": stats["tp"],
        "closest_knn_fn": stats["fn"],
        "closest_knn_tn": stats["tn"],
        "closest_knn_fp": stats["fp"],
        "closest_knn_f1": stats["f1"],
    }


def decoder_confusion_stats(
    source_seq: str,
    target_seq: str,
    decoded_seq: str,
) -> dict[str, Optional[float]]:
    """TP/FN/TN/FP/F1 for decoded prediction vs source→target changes (position-level)."""
    stats = change_from_source_confusion_stats(source_seq, target_seq, decoded_seq)
    if stats["tp"] is None:
        return {
            "decoder_position_tp": None,
            "decoder_position_fn": None,
            "decoder_position_tn": None,
            "decoder_position_fp": None,
            "decoder_position_f1": None,
        }
    return {
        "decoder_position_tp": stats["tp"],
        "decoder_position_fn": stats["fn"],
        "decoder_position_tn": stats["tn"],
        "decoder_position_fp": stats["fp"],
        "decoder_position_f1": stats["f1"],
    }


def build_prediction_report_fields(
    source_seq: str,
    target_seq: str,
    pred_seq: str,
    branch_targets: list[str],
    knn_closest_sequence: str,
    *,
    metric_prefix: str,
    sequence_column: str,
    hamming_prefix: str,
    blast_target_prefix: str,
) -> dict[str, Any]:
    """Decoder-style per-row metrics for an arbitrary predicted sequence."""
    gt_positions, gt_mutations = (
        extract_mutations(source_seq, target_seq) if (source_seq and target_seq) else ([], [])
    )
    _, pred_mutations = (
        extract_mutations(source_seq, pred_seq)
        if (source_seq and pred_seq)
        else ([], [])
    )

    tp_alleles = sorted(set(gt_mutations) & set(pred_mutations))
    fn_alleles = sorted(set(gt_mutations) - set(pred_mutations))
    fp_alleles = sorted(set(pred_mutations) - set(gt_mutations))
    position_stats = change_from_source_confusion_stats(source_seq, target_seq, pred_seq)

    pred_any = any_branch_allele_metrics(source_seq, pred_seq, branch_targets)
    pred_all_alleles = all_branch_allele_metrics(source_seq, pred_seq, branch_targets)
    all_position = all_branch_position_metrics(
        source_seq,
        pred_seq,
        knn_closest_sequence,
        branch_targets,
    )

    hamming_src = (
        compute_hamming_distance(source_seq, pred_seq) if pred_seq else None
    )
    hamming_tgt = (
        compute_hamming_distance(target_seq, pred_seq) if pred_seq else None
    )
    blast_tgt = (
        compute_blast_alignment_metrics(pred_seq, target_seq) if pred_seq else None
    )

    return {
        sequence_column: pred_seq,
        f"{hamming_prefix}_hamming_to_source": hamming_src,
        f"{hamming_prefix}_hamming_to_target": hamming_tgt,
        f"blast_identity_{blast_target_prefix}_to_target": (
            round(blast_tgt["identity"], 6) if blast_tgt else None
        ),
        f"blast_similarity_{blast_target_prefix}_to_target": (
            round(blast_tgt["similarity"], 6) if blast_tgt else None
        ),
        f"{metric_prefix}_tp": len(tp_alleles),
        f"{metric_prefix}_fn": len(fn_alleles),
        f"{metric_prefix}_fp": position_stats["fp"],
        f"{metric_prefix}_tn": position_stats["tn"],
        f"{metric_prefix}_f1": position_stats["f1"],
        f"{metric_prefix}_tp_alleles": tp_alleles,
        f"{metric_prefix}_fn_alleles": fn_alleles,
        f"{metric_prefix}_fp_alleles": fp_alleles,
        f"{metric_prefix}_fp_alleles_count": len(fp_alleles),
        f"{metric_prefix}_position_tp": position_stats["tp"],
        f"{metric_prefix}_position_fn": position_stats["fn"],
        f"{metric_prefix}_position_fp": position_stats["fp"],
        f"{metric_prefix}_position_tn": position_stats["tn"],
        f"{metric_prefix}_position_f1": position_stats["f1"],
        f"{metric_prefix}_tp_any_branch": pred_any.get("decoder_tp_any"),
        f"{metric_prefix}_fn_any_branch": pred_any.get("decoder_fn_any"),
        f"{metric_prefix}_recall_any_branch": pred_any.get("decoder_recall_any"),
        f"{metric_prefix}_tp_all_branches": all_position.get("decoder_tp_all"),
        f"{metric_prefix}_fn_all_branches": all_position.get("decoder_fn_all"),
        f"{metric_prefix}_tp_alleles_all_branches": pred_all_alleles.get(
            "decoder_tp_all_alleles"
        ),
        f"{metric_prefix}_fn_alleles_all_branches": pred_all_alleles.get(
            "decoder_fn_all_alleles"
        ),
        f"{metric_prefix}_matches_any_branch": pred_any.get(
            "decoder_matches_any_branch"
        ),
        f"{metric_prefix}_min_hamming_to_branch_targets": pred_any.get(
            "min_hamming_pred_to_branch_targets"
        ),
    }


def build_report(
    val_pairs_path: str,
    fasta_path: str,
    json_dir: str,
    output_csv: str,
    include_full_seqs: bool = False,
    series_npz: Optional[str] = None,
    msa_path: Optional[str] = None,
    knn_dir: Optional[str] = None,
    layers_consensus_fasta: Optional[str] = None,
):
    blosum62 = load_blosum62_matrix()
    knn_root = Path(knn_dir or json_dir)

    # Keep genome ids as text; float parsing corrupts BV-BRC style ids.
    df_pairs = pd.read_csv(
        val_pairs_path,
        dtype={
            "genome_id_source": "string",
            "genome_id_target": "string",
        },
        low_memory=False,
    )
    decoded_dict = parse_fasta(fasta_path)
    _pair_ids = [_row_id_from_pairs_row(r, i) for i, r in df_pairs.iterrows()]
    _n_missing = sum(1 for rid in _pair_ids if rid not in decoded_dict)
    print(
        f"Decoded predictions: {len(decoded_dict)} sequences in {fasta_path}; "
        f"{len(_pair_ids) - _n_missing}/{len(_pair_ids)} validation rows covered."
    )
    if _n_missing:
        print(
            f"WARNING: {_n_missing} validation rows have no decoded prediction; "
            "they will be scored as complete misses (TP=0)."
        )
    layers_consensus_dict: dict[int, str] = {}
    if layers_consensus_fasta:
        layers_consensus_dict = parse_fasta(layers_consensus_fasta)
        print(
            f"Loaded {len(layers_consensus_dict)} layer-consensus sequences from "
            f"{layers_consensus_fasta}"
        )

    # Pre-compute number of unique target sequence branches per unique source sequence
    branch_counts = df_pairs.groupby("seq_source")["seq_target"].nunique().to_dict()

    prefix_counts: dict[int, int] = {}
    row_id_to_prefix: dict[int, tuple[str, ...]] = {}
    if series_npz and msa_path:
        row_id_to_prefix, prefix_counts = compute_series_prefix_keys(
            series_npz, msa_path
        )
        print(
            f"Computed identical-prefix series counts for {len(prefix_counts)} / "
            f"{len(df_pairs)} rows from {series_npz}"
        )
    elif series_npz or msa_path:
        raise SystemExit("Both --series-npz and --msa are required for prefix counts.")

    branch_groups = build_branch_groups(df_pairs, row_id_to_prefix)
    branch_metadata_by_row = build_branch_metadata_by_row(df_pairs, branch_groups)
    print(f"Built branch metadata for {len(branch_metadata_by_row)} rows.")

    report_rows = []

    for idx, row in df_pairs.iterrows():
        row_id = _row_id_from_pairs_row(row, idx)

        if row_id % 100 == 0:
            print(f"Processing row {row_id} / {len(df_pairs)}.")

        src_seq = row.get("seq_source", "")
        tgt_seq = row.get("seq_target", "")
        if not isinstance(src_seq, str):
            src_seq = "" if pd.isna(src_seq) else str(src_seq)
        if not isinstance(tgt_seq, str):
            tgt_seq = "" if pd.isna(tgt_seq) else str(tgt_seq)

        if not src_seq or not tgt_seq:
            continue

        num_branches = branch_counts.get(src_seq, 0)
        num_identical_prefix_series = prefix_counts.get(row_id)

        mut_json = _load_json_if_exists(
            Path(json_dir) / f"knn_baseline_mutations_row{row_id}.json"
        )
        knn_json = _load_json_if_exists(knn_root / f"knn_baseline_row{row_id}.json")
        knn_metrics = knn_json.get("metrics", {}) if isinstance(knn_json, dict) else {}

        latent_knn = mut_json.get("latent_knn", {})
        pos_stat = mut_json.get("position_stat", {})
        confusion_change = latent_knn.get("confusion_change_from_source", {})
        confusion_target_allele = latent_knn.get("confusion_target_allele", {})

        knn_target_sequence_found = knn_metrics.get("knn_target_sequence_found")
        knn_target_sequence_gids = knn_metrics.get("knn_target_sequence_gids", [])
        if knn_target_sequence_found is None and mut_json:
            # Older mutation-only JSONs may not carry these fields.
            knn_target_sequence_found = mut_json.get("knn_target_sequence_found")
            knn_target_sequence_gids = mut_json.get("knn_target_sequence_gids", [])

        knn_fasta = parse_knn_analysis_fasta(
            knn_root / f"knn_baseline_row{row_id}.fasta"
        )
        knn_closest_sequence = knn_fasta.get("knn_0", "")
        knn_consensus_sequence = knn_fasta.get("knn_consensus", "")
        closest_stats = closest_knn_confusion_stats(
            src_seq, tgt_seq, knn_closest_sequence
        )

        gt_blosum_dist = compute_blosum_distance(src_seq, tgt_seq, blosum62)
        blast_src_tgt = compute_blast_alignment_metrics(src_seq, tgt_seq)
        decoded_seq = decoded_dict.get(row_id, "")

        branch_meta = branch_metadata_by_row.get(row_id, {})
        branch_targets = list(branch_meta.get("branch_target_sequences") or [tgt_seq])
        target_consensus_sequence = (
            branch_meta.get("branch_target_consensus")
            or build_target_consensus(branch_targets)
            or tgt_seq
        )

        gt_positions, gt_mutations = (
            extract_mutations(src_seq, tgt_seq) if (src_seq and tgt_seq) else ([], [])
        )

        decoder_fields = build_prediction_report_fields(
            src_seq,
            tgt_seq,
            decoded_seq,
            branch_targets,
            knn_closest_sequence,
            metric_prefix="decoder",
            sequence_column="decoded_prediction_sequence",
            hamming_prefix="decoded",
            blast_target_prefix="decoded",
        )
        layer_consensus_seq = layers_consensus_dict.get(row_id, "")
        layer_consensus_fields: dict[str, Any] = {}
        if layers_consensus_dict:
            layer_consensus_fields = build_prediction_report_fields(
                src_seq,
                tgt_seq,
                layer_consensus_seq,
                branch_targets,
                knn_closest_sequence,
                metric_prefix="layer_consensus",
                sequence_column="layer_consensus_sequence",
                hamming_prefix="layer_consensus",
                blast_target_prefix="layer_consensus",
            )

        closest_any = any_branch_position_metrics(
            src_seq, knn_closest_sequence, branch_targets
        )
        all_position = all_branch_position_metrics(
            src_seq, decoded_seq, knn_closest_sequence, branch_targets
        )

        lineage_source = knn_json.get("lineage_source") or mut_json.get("lineage_source", "")
        lineage_target = knn_json.get("lineage_target") or mut_json.get("lineage_target", "")
        date_source = knn_json.get("date_source") or mut_json.get("date_source", "")
        date_target = knn_json.get("date_target") or mut_json.get("date_target", "")

        report_row = {
            "row": row_id,
            "num_branches": num_branches,
            "num_identical_prefix_series": num_identical_prefix_series,
            "seq_source": src_seq,
            "seq_target": tgt_seq,
            "seq_target_consensus_branches": target_consensus_sequence,
            "genome_id_source": _gid_str(row.get("genome_id_source", "")),
            "genome_id_target": _gid_str(row.get("genome_id_target", "")),
            "lineage_source": lineage_source,
            "lineage_target": lineage_target,
            "date_source": date_source,
            "date_target": date_target,
            "gt_hamming_source_target": compute_hamming_distance(src_seq, tgt_seq),
            "blosum_distance_source_to_target": round(gt_blosum_dist, 4),
            "blast_identity_source_to_target": round(
                blast_src_tgt["identity"], 6
            ),
            "blast_similarity_source_to_target": round(
                blast_src_tgt["similarity"], 6
            ),
            "required_mutation_sites": gt_positions,
            "required_mutations": gt_mutations,
            **decoder_fields,
            **layer_consensus_fields,
            "knn_consensus_sequence": knn_consensus_sequence,
            "knn_tp": confusion_change.get("tp", None),
            "knn_fn": confusion_change.get("fn", None),
            "knn_fp": confusion_change.get("fp", None),
            "knn_tn": confusion_change.get("tn", None),
            "knn_f1_allele": confusion_target_allele.get("f1", None),
            "knn_tp_alleles": pos_stat.get("true_positive_alleles", []),
            "knn_fn_alleles": pos_stat.get("false_negative_alleles", []),
            "knn_target_sequence_found": knn_target_sequence_found,
            "knn_target_sequence_gids": _serialize_cell(knn_target_sequence_gids or []),
            "knn_closest_sequence": knn_closest_sequence,
            "closest_knn_tp": closest_stats["closest_knn_tp"],
            "closest_knn_fn": closest_stats["closest_knn_fn"],

            "closest_knn_tn": closest_stats["closest_knn_tn"],
            "closest_knn_fp": closest_stats["closest_knn_fp"],
            "closest_knn_f1": closest_stats["closest_knn_f1"],
            "n_branch_targets_for_prefix": branch_meta.get(
                "n_branch_targets_for_prefix", len(branch_targets)
            ),
            "branch_target_gids": _serialize_cell(
                branch_meta.get("branch_target_gids", [])
            ),
            "branch_required_sites_union": _serialize_cell(
                branch_meta.get("branch_required_sites_union", [])
            ),
            "branch_required_alleles_union": _serialize_cell(
                branch_meta.get("branch_required_alleles_union", [])
            ),
            "branch_required_sites_intersection": _serialize_cell(
                branch_meta.get("branch_required_sites_intersection", [])
            ),
            "branch_required_alleles_intersection": _serialize_cell(
                branch_meta.get("branch_required_alleles_intersection", [])
            ),
            "branch_variable_sites_between_targets": _serialize_cell(
                branch_meta.get("branch_variable_sites_between_targets", [])
            ),
            "decoder_matches_any_branch": decoder_fields.get(
                "decoder_matches_any_branch"
            ),
            "min_hamming_pred_to_branch_targets": decoder_fields.get(
                "decoder_min_hamming_to_branch_targets"
            ),
            "closest_knn_tp_any_branch": closest_any.get("closest_knn_tp_any"),
            "closest_knn_fn_any_branch": closest_any.get("closest_knn_fn_any"),
            "closest_knn_recall_any_branch": closest_any.get("closest_knn_recall_any"),
            "closest_knn_tp_all_branches": all_position.get("closest_knn_tp_all"),
            "closest_knn_fn_all_branches": all_position.get("closest_knn_fn_all"),
        }
        report_rows.append(report_row)

    df_report = pd.DataFrame(report_rows)

    # Always keep source/target and branch metadata. Drop bulky sequences unless requested.
    if not include_full_seqs:
        columns_to_remove = [
            "decoded_prediction",
            "layer_consensus_sequence",
            "knn_closest_sequence",
            "knn_consensus_sequence",
        ]
        existing_cols_to_drop = [
            col for col in columns_to_remove if col in df_report.columns
        ]
        df_report = df_report.drop(columns=existing_cols_to_drop)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df_report.to_csv(output_csv, index=False)
    print(f"Report compiled and saved to {output_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compile metrics into a structured CSV report.")
    p.add_argument(
        "--val_pairs",
        type=str,
        default="validation_pairs_test.csv",
        help="Path to ground truth validation pairs CSV.",
    )
    p.add_argument(
        "--fasta",
        type=str,
        default="b3_mean_res_cal_test_decoded.fasta",
        help="Path to FASTA decoded predictions.",
    )
    p.add_argument(
        "--json_dir",
        type=str,
        default=".",
        help="Directory containing knn_baseline_mutations_row*.json files.",
    )
    p.add_argument(
        "--knn_dir",
        type=str,
        default=None,
        help=(
            "Directory with knn_baseline_row*.json / *.fasta "
            "(defaults to --json_dir)."
        ),
    )
    p.add_argument(
        "--series-npz",
        type=str,
        default=None,
        help=(
            "Series NPZ (e.g. test_dataset.npz). With --msa, adds "
            "num_identical_prefix_series counts."
        ),
    )
    p.add_argument(
        "--msa",
        type=str,
        default=None,
        help="MSA FASTA used to resolve series genome ids to sequences.",
    )
    p.add_argument(
        "--output",
        type=str,
        default="evaluation_report.csv",
        help="Path to save the output CSV report.",
    )
    p.add_argument(
        "--include-full-seqs",
        action="store_true",
        default=True,
        help="Also include decoded_prediction in the CSV (seq_source/seq_target are always kept).",
    )
    p.add_argument(
        "--layers-consensus-fasta",
        type=str,
        default=None,
        help=(
            "Optional FASTA with per-row layer33+34 consensus sequences (row= in header). "
            "When set, adds layer_consensus_* metrics parallel to decoder_*."
        ),
    )
    args = p.parse_args()
    build_report(
        args.val_pairs,
        args.fasta,
        args.json_dir,
        args.output,
        args.include_full_seqs,
        series_npz=args.series_npz,
        msa_path=args.msa,
        knn_dir=args.knn_dir,
        layers_consensus_fasta=args.layers_consensus_fasta,
    )
