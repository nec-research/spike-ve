#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     deduplicate_msa.py
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
"""Deduplicate MSA FASTA records and subset a matching phylogenetic tree.

Among records with identical aligned sequences, keep one record chosen by a
metadata field (default: earliest ``collection_date``). Metadata comes from an
optional labels CSV (e.g. ``df_labels.csv``); otherwise fields are parsed from
FASTA headers. When no duplicate in a group has the requested field, one record
is chosen at random (seeded for reproducibility).

Optionally prune a Newick tree to the retained genome IDs, preserving branch
lengths (tree leaves are expected as ``genome_id-<genome_id>`` by default).

Example
-------

    python -m recomb2027.data.deduplicate_msa \\
        --input-fasta /mnt/.../filtered_msa_S1.fasta \\
        --output-fasta /mnt/.../filtered_msa_S1_dedup.fasta \\
        --metadata-csv /mnt/.../df_labels.csv \\
        --dedup-field collection_date \\
        --dedup-criterion min \\
        --tree /mnt/.../pruned_tree.nwk \\
        --output-tree /mnt/.../pruned_tree_S1_dedup.nwk
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

GENOME_ID_RE = re.compile(r"(\d+\.\d+)")
DEFAULT_MSA = (
    ""
)
DEFAULT_METADATA = (
    ""
)
DEFAULT_TREE = (
    ""
)


@dataclass(frozen=True)
class FastaRecord:
    header: str
    sequence: str
    genome_id: str
    header_fields: dict[str, str]


def parse_header_fields(header: str) -> dict[str, str]:
    """Parse ``field:value|field:value`` FASTA headers."""
    line = header[1:] if header.startswith(">") else header
    fields: dict[str, str] = {}
    for part in line.split("|"):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        fields[key.strip()] = value.strip()
    return fields


def genome_id_from_header(header: str) -> str:
    fields = parse_header_fields(header)
    raw = fields.get("genome_id", "")
    match = GENOME_ID_RE.search(raw)
    if match:
        return match.group(1)
    matches = GENOME_ID_RE.findall(header)
    if not matches:
        raise ValueError(f"Could not parse genome_id from header: {header[:120]!r}")
    return matches[0]


def load_fasta_records(path: Path) -> list[FastaRecord]:
    records: list[FastaRecord] = []
    header: str | None = None
    seq_parts: list[str] = []

    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith(">"):
                if header is not None:
                    records.append(
                        FastaRecord(
                            header=header,
                            sequence="".join(seq_parts),
                            genome_id=genome_id_from_header(header),
                            header_fields=parse_header_fields(header),
                        )
                    )
                header = line.rstrip("\n")
                seq_parts = []
            else:
                seq_parts.append(line.strip())

    if header is not None:
        records.append(
            FastaRecord(
                header=header,
                sequence="".join(seq_parts),
                genome_id=genome_id_from_header(header),
                header_fields=parse_header_fields(header),
            )
        )

    return records


def load_metadata_lookup(
    metadata_csv: Path | None,
    *,
    genome_id_col: str = "genome_id",
) -> dict[str, dict[str, str]]:
    if metadata_csv is None:
        return {}
    if not metadata_csv.is_file():
        raise FileNotFoundError(metadata_csv)

    df = pd.read_csv(metadata_csv, dtype=str)
    if genome_id_col not in df.columns:
        raise ValueError(
            f"Metadata CSV missing required column {genome_id_col!r}: {metadata_csv}"
        )

    lookup: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        gid = str(row[genome_id_col]).strip()
        if not gid or gid == "nan":
            continue
        lookup[gid] = {
            col: ("" if pd.isna(row[col]) else str(row[col]).strip())
            for col in df.columns
        }
    return lookup


def _coerce_sort_key(raw: str, field: str) -> tuple[int, Any]:
    """Return (rank, value) where rank 0 means valid sortable value."""
    if raw is None:
        return 1, None
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "null", ""}:
        return 1, None

    if field == "collection_date" or re.search(r"date", field, re.I):
        parsed = pd.to_datetime(text, errors="coerce")
        if pd.isna(parsed):
            return 1, None
        if isinstance(parsed, pd.Timestamp):
            return 0, parsed.to_pydatetime()
        return 0, parsed

    try:
        return 0, float(text)
    except ValueError:
        return 0, text


def field_value_for_record(
    record: FastaRecord,
    field: str,
    metadata_lookup: Mapping[str, Mapping[str, str]],
) -> str | None:
    meta = metadata_lookup.get(record.genome_id)
    if meta is not None and field in meta:
        value = meta.get(field, "")
        if value:
            return value
    header_value = record.header_fields.get(field, "")
    return header_value or None


def select_keeper(
    group: list[FastaRecord],
    *,
    field: str,
    criterion: str,
    metadata_lookup: Mapping[str, Mapping[str, str]],
    rng: np.random.Generator,
) -> FastaRecord:
    if len(group) == 1:
        return group[0]

    scored: list[tuple[tuple[int, Any], FastaRecord]] = []
    for record in group:
        raw = field_value_for_record(record, field, metadata_lookup)
        scored.append((_coerce_sort_key(raw, field), record))

    valid = [(key, rec) for key, rec in scored if key[0] == 0]
    if not valid:
        return group[int(rng.integers(0, len(group)))]

    reverse = criterion == "max"
    _, keeper = min(valid, key=lambda item: item[0][1], default=(None, group[0]))
    if reverse:
        _, keeper = max(valid, key=lambda item: item[0][1])
    return keeper


def deduplicate_records(
    records: list[FastaRecord],
    *,
    field: str,
    criterion: str,
    metadata_lookup: Mapping[str, Mapping[str, str]],
    seed: int,
) -> tuple[list[FastaRecord], dict[str, int]]:
    rng = np.random.default_rng(seed)
    groups: dict[str, list[FastaRecord]] = defaultdict(list)
    for record in records:
        groups[record.sequence].append(record)

    kept: list[FastaRecord] = []
    n_duplicate_groups = 0
    n_removed = 0
    n_random_tiebreak = 0

    for group in groups.values():
        if len(group) > 1:
            n_duplicate_groups += 1
            n_removed += len(group) - 1
            scored_valid = [
                _coerce_sort_key(
                    field_value_for_record(rec, field, metadata_lookup),
                    field,
                )[0]
                == 0
                for rec in group
            ]
            if not any(scored_valid):
                n_random_tiebreak += 1
        kept.append(
            select_keeper(
                group,
                field=field,
                criterion=criterion,
                metadata_lookup=metadata_lookup,
                rng=rng,
            )
        )

    stats = {
        "input_records": len(records),
        "output_records": len(kept),
        "duplicate_groups": n_duplicate_groups,
        "records_removed": n_removed,
        "random_tiebreak_groups": n_random_tiebreak,
    }
    return kept, stats


def write_fasta_records(records: Iterable[FastaRecord], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for record in records:
            header = record.header
            if not header.startswith(">"):
                header = ">" + header
            fh.write(f"{header}\n{record.sequence}\n")


def tree_leaf_label(genome_id: str, leaf_prefix: str) -> str:
    return f"{leaf_prefix}-{genome_id}"


def _genome_id_from_leaf_label(label: str) -> str:
    return label.split("-", 1)[-1]


def _leaf_labels_by_genome_id(leaf_labels: Iterable[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for label in leaf_labels:
        mapping[_genome_id_from_leaf_label(label)] = label
    return mapping


def _subset_tree_with_dendropy(
    tree_path: Path,
    keep_labels: set[str],
    remove_labels: set[str],
    out_path: Path,
) -> None:
    import dendropy

    tree = dendropy.Tree.get(path=str(tree_path), schema="newick")
    tree.prune_taxa_with_labels(remove_labels)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path=str(out_path), schema="newick")


def subset_phylogenetic_tree(
    tree_path: Path,
    keep_genome_ids: set[str],
    out_path: Path,
) -> dict[str, int]:
    if not tree_path.is_file():
        raise FileNotFoundError(tree_path)

    log.info("Loading tree from %s", tree_path)

    try:
        from ete3 import Tree as EteTree

        probe = EteTree(str(tree_path), format=1)
        leaf_labels = {leaf.name for leaf in probe.get_leaves()}
        backend = "ete3"
    except ImportError:
        import dendropy

        probe = dendropy.Tree.get(path=str(tree_path), schema="newick")
        leaf_labels = {node.taxon.label for node in probe.leaf_node_iter()}
        backend = "dendropy"
        probe = None

    leaf_by_gid = _leaf_labels_by_genome_id(leaf_labels)
    keep_labels_list = sorted(
        leaf_by_gid[gid] for gid in keep_genome_ids if gid in leaf_by_gid
    )
    keep_labels = set(keep_labels_list)
    missing_in_tree = sorted(gid for gid in keep_genome_ids if gid not in leaf_by_gid)
    if missing_in_tree:
        log.warning(
            "%d deduplicated genome IDs are absent from the tree (showing up to 5): %s",
            len(missing_in_tree),
            missing_in_tree[:5],
        )

    remove_labels = leaf_labels - keep_labels
    log.info(
        "Pruning tree with %s: keep %d / %d leaves (%d removed)",
        backend,
        len(keep_labels),
        len(leaf_labels),
        len(remove_labels),
    )
    if not keep_labels:
        raise ValueError("No deduplicated genome IDs matched any leaf in the tree.")

    if backend == "ete3":
        assert probe is not None
        input_leaves = len(probe.get_leaves())
        probe.prune(keep_labels_list, preserve_branch_length=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        probe.write(format=1, outfile=str(out_path))
        output_leaves = len(probe.get_leaves())
    else:
        _subset_tree_with_dendropy(
            tree_path,
            keep_labels,
            remove_labels,
            out_path,
        )
        input_leaves = len(leaf_labels)
        output_leaves = len(keep_labels)

    stats = {
        "tree_input_leaves": input_leaves,
        "tree_output_leaves": output_leaves,
        "tree_missing_genome_ids": len(missing_in_tree),
        "tree_removed_leaves": len(remove_labels),
    }
    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deduplicate an MSA FASTA by sequence identity and optionally "
            "subset a phylogenetic tree to the retained genomes."
        ),
    )
    parser.add_argument(
        "--input-fasta",
        type=Path,
        default=Path(DEFAULT_MSA),
        help="Input MSA FASTA.",
    )
    parser.add_argument(
        "--output-fasta",
        type=Path,
        required=True,
        help="Output deduplicated FASTA path.",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=None,
        help=(
            "Optional metadata CSV (e.g. df_labels.csv). When set, field values "
            "are read from this file first, then FASTA headers as fallback."
        ),
    )
    parser.add_argument(
        "--dedup-field",
        default="collection_date",
        help="Metadata field used to choose among duplicate sequences.",
    )
    parser.add_argument(
        "--dedup-criterion",
        choices=("min", "max"),
        default="min",
        help=(
            "Keep the duplicate with the lowest (min) or highest (max) field "
            "value. Default min keeps the earliest collection_date."
        ),
    )
    parser.add_argument(
        "--metadata-genome-id-col",
        default="genome_id",
        help="Genome ID column name in --metadata-csv.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed when duplicate groups lack the dedup field.",
    )
    parser.add_argument(
        "--tree",
        type=Path,
        default=None,
        help="Input phylogenetic tree (Newick). Defaults to pruned_tree.nwk when present.",
    )
    parser.add_argument(
        "--output-tree",
        type=Path,
        default=None,
        help="Output pruned tree path. Required when --tree is set unless inferred.",
    )
    parser.add_argument(
        "--skip-tree",
        action="store_true",
        help="Skip phylogenetic tree subsetting.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)

    if not args.input_fasta.is_file():
        raise SystemExit(f"Input FASTA not found: {args.input_fasta}")

    metadata_csv = args.metadata_csv
    if metadata_csv is None and Path(DEFAULT_METADATA).is_file():
        metadata_csv = Path(DEFAULT_METADATA)
        log.info("Using default metadata CSV: %s", metadata_csv)

    log.info("Loading FASTA records from %s", args.input_fasta)
    records = load_fasta_records(args.input_fasta)
    metadata_lookup = load_metadata_lookup(
        metadata_csv,
        genome_id_col=args.metadata_genome_id_col,
    )
    if metadata_lookup:
        log.info("Loaded metadata for %d genome IDs", len(metadata_lookup))

    kept, dedup_stats = deduplicate_records(
        records,
        field=args.dedup_field,
        criterion=args.dedup_criterion,
        metadata_lookup=metadata_lookup,
        seed=args.seed,
    )

    write_fasta_records(kept, args.output_fasta)
    log.info("Wrote deduplicated FASTA: %s", args.output_fasta)
    for key, value in dedup_stats.items():
        log.info("  %s: %d", key, value)

    if args.skip_tree:
        return 0

    tree_path = args.tree
    if tree_path is None and Path(DEFAULT_TREE).is_file():
        tree_path = Path(DEFAULT_TREE)
        log.info("Using default tree: %s", tree_path)

    if tree_path is None:
        log.info("No tree provided; skipping tree subsetting.")
        return 0

    output_tree = args.output_tree
    if output_tree is None:
        output_tree = args.output_fasta.with_suffix(".nwk")
        log.info("Inferred --output-tree: %s", output_tree)

    keep_gids = {rec.genome_id for rec in kept}
    tree_stats = subset_phylogenetic_tree(
        tree_path,
        keep_gids,
        output_tree,
    )
    log.info("Wrote pruned tree: %s", output_tree)
    for key, value in tree_stats.items():
        log.info("  %s: %d", key, value)

    return 0


if __name__ == "__main__":
    sys.exit(main())
