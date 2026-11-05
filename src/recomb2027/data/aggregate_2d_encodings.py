#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     aggregate_2d_encodings.py
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
"""Post-hoc pooling of precomputed 2D ESM token embeddings.

Reads flat ``<label>.pt`` files produced by ``encode_proteins_from_fasta --save-2d``
(each record: ``{"label", "encodings": {layer: [L+1, D]}, "aligned_seq"?}``) and
writes 1D pooled vectors.  Does **not** load an ESM model or run inference.

Gap columns (``-`` in ``aligned_seq``; ``.`` treated as gap) are **excluded** from
``average_token``, block means, and extra stats by default.  Pass
``--include-gap-columns`` to mean over all MSA columns.

Examples
--------

Mean-pool all layers, including gaps (separate output tree)::

    python -m recomb2027.data.aggregate_2d_encodings \\
        --input-dir /mnt/.../encodings/ESM1B/filtered_msa \\
        --output-dir /mnt/.../encodings/ESM1B/filtered_msa_mean_with_gaps \\
        --aggregate-pooling average_token \\
        --include-gap-columns

Layer 33 only, gap-excluded (default), plus domain blocks::

    python -m recomb2027.data.aggregate_2d_encodings \\
        --input-dir /mnt/.../encodings/ESM1B/filtered_msa \\
        --output-dir /mnt/.../encodings/ESM1B/filtered_msa \\
        --aggregate-pooling average_token \\
        --layers 33 \\
        --block-boundaries configs/latent/shared/spike_block_boundaries.yaml \\
        --block-pooling average_token \\
        --skip-existing

Output layout (same as encode_proteins_from_fasta)::

    <output-dir>/<pooling_strategy>/<layer>/<file_id>.pt
    <output-dir>/<stat>/<layer>/<file_id>.pt
    <output-dir>/block_<name>/<block_pooling>/<layer>/<file_id>.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

import torch
from tqdm import tqdm

from recomb2027.core.paths import repo_root

REPO_ROOT = repo_root(__file__)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recomb2027.data.esm_pooling import (
    EXTRA_STAT_NAMES,
    mean_pool_aa_tokens,
    mean_pool_sequence_dim,
    msa_non_gap_mask,
    pool_aa_tokens_stats,
    first_non_gap_row,
    pool_token_repr_single_stat,
    pool_token_repr_slice,
)
from recomb2027.data.spike_blocks import (
    block_output_dirname,
    clamp_block_slice,
    load_spike_block_boundaries,
    max_block_msa_end,
    validate_block_boundaries,
)

SUPPORTED_POOLING = [
    "cls_token",
    "first_token",
    "average_token",
    "average_sequence",
    "stats_summary",
]
MEAN_POOLING_STRATEGIES = ("average_token", "average_sequence")
MAX_FILENAME_LEN = 100


def _dedupe_preserve_order(names: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def label_to_filename(label: str, max_len: int = MAX_FILENAME_LEN) -> str:
    """Sanitise a sequence label for use as a filename."""
    return (
        label.replace("/", "-")
        .replace(":", "_")
        .replace(" ", "")
        .replace("'", "")
        .replace(",", "_")
    )[:max_len]


def aggregate_encodings(
    input_dir: Path,
    output_dir: Path,
    pooling_strategies: list,
    skip_existing: bool = False,
    extra_pooling_stats: list | None = None,
    block_specs: list | None = None,
    block_pooling: str | None = None,
    normalize_position_stats: bool = True,
    exclude_gap_columns: bool = True,
    layers: list[int] | None = None,
):
    """Pool saved 2D token-level embeddings and write 1D files.

    Output layout: ``output_dir / pooling_strategy / layer / <file_id>.pt``

    Each output .pt file contains ``{"label": str, "encodings": Tensor}``.
    When ``exclude_gap_columns`` is True, ``aligned_seq`` stored in 2D records
    is used to omit ``-`` columns from mean/stat pooling.

    Args:
        layers: If set, only pool these layer keys (ints or matching keys in the
            encodings dict).  ``None`` → all layers present in each file.
    """
    pt_files = sorted(f for f in input_dir.iterdir() if f.suffix == ".pt")
    logging.info(
        f"Aggregating {len(pt_files)} files from '{input_dir}' "
        f"with strategies {pooling_strategies}."
    )
    if layers is not None:
        logging.info(f"  Layer filter : {layers}")
    gap_fallback_warned = False
    layer_filter = None if layers is None else {int(x) for x in layers}

    for fpath in tqdm(pt_files, desc="Aggregating"):
        try:
            record = torch.load(fpath, map_location="cpu")
        except Exception as exc:
            logging.error(f"Cannot load '{fpath.name}': {exc}")
            continue

        label = record.get("label", fpath.stem)
        encodings_rec = record.get("encodings")

        if not isinstance(encodings_rec, dict):
            logging.warning(f"Skipping '{fpath.name}': 'encodings' is not a dict.")
            continue

        all_layers = list(encodings_rec.keys())
        if not all_layers:
            logging.warning(f"Skipping '{fpath.name}': no layers found.")
            continue

        if layer_filter is not None:
            file_layers = []
            for layer in all_layers:
                try:
                    key = int(layer)
                except (TypeError, ValueError):
                    key = layer
                if key in layer_filter or layer in layer_filter:
                    file_layers.append(layer)
            if not file_layers:
                logging.warning(
                    f"Skipping '{fpath.name}': no requested layers "
                    f"{sorted(layer_filter)} (have {all_layers})."
                )
                continue
        else:
            file_layers = all_layers

        file_id = label_to_filename(label)
        aligned_seq = None
        if exclude_gap_columns:
            aligned_seq = record.get("aligned_seq")
            if not aligned_seq and not gap_fallback_warned:
                logging.warning(
                    "2D record %s has no aligned_seq; gap columns included in pooling. "
                    "Re-encode with --save-2d to store aligned_seq for gap-aware aggregation.",
                    fpath.name,
                )
                gap_fallback_warned = True

        strategies = list(pooling_strategies or [])
        for pooling_strategy, layer in product(strategies, file_layers):
            enc = encodings_rec[layer]

            if enc.ndim == 1:
                logging.debug(
                    f"'{fpath.name}' layer {layer} already 1D, skipping."
                )
                continue

            if enc.ndim != 2:
                logging.warning(
                    f"Unexpected shape {enc.shape} in '{fpath.name}' layer {layer}."
                )
                continue

            if enc.shape[0] < 2:
                logging.warning(
                    f"Skipping '{fpath.name}' layer {layer}: seq dim {enc.shape[0]} < 2."
                )
                continue

            layer_truncate = int(enc.shape[0]) - 1
            aa = enc[1 : layer_truncate + 1, :]
            gap_mask = (
                msa_non_gap_mask(str(aligned_seq), layer_truncate)
                if aligned_seq
                else None
            )

            if pooling_strategy == "cls_token":
                pooled = enc[0, :]
            elif pooling_strategy == "first_token":
                pooled = first_non_gap_row(aa, gap_mask)
            elif pooling_strategy == "average_sequence":
                pooled = mean_pool_sequence_dim(aa)
            elif pooling_strategy == "average_token":
                pooled = mean_pool_aa_tokens(aa, gap_mask)
            elif pooling_strategy == "stats_summary":
                pooled = pool_aa_tokens_stats(aa, non_gap_mask=gap_mask)
            elif pooling_strategy in EXTRA_STAT_NAMES:
                pooled = pool_token_repr_single_stat(
                    enc,
                    pooling_strategy,
                    layer_truncate,
                    normalize_positions=normalize_position_stats,
                    non_gap_mask=gap_mask,
                )
            else:
                raise ValueError(f"Unknown pooling strategy: '{pooling_strategy}'")

            # clone() so torch.save does not persist the full 2D storage via a view
            # (same rationale as ESMProteinEncoder._apply_pooling).
            pooled = pooled.detach().float().contiguous().clone()

            out_subdir = output_dir / str(pooling_strategy) / str(layer)
            out_subdir.mkdir(parents=True, exist_ok=True)
            out_file = out_subdir / f"{file_id}.pt"

            if skip_existing and out_file.exists():
                continue

            torch.save({"label": label, "encodings": pooled}, out_file)

        for stat in _dedupe_preserve_order(extra_pooling_stats or []):
            for layer in file_layers:
                enc = encodings_rec[layer]
                if enc.ndim != 2 or enc.shape[0] < 2:
                    continue
                layer_truncate = int(enc.shape[0]) - 1
                gap_mask = (
                    msa_non_gap_mask(str(aligned_seq), layer_truncate)
                    if aligned_seq
                    else None
                )
                pooled = pool_token_repr_single_stat(
                    enc,
                    stat,
                    layer_truncate,
                    normalize_positions=normalize_position_stats,
                    non_gap_mask=gap_mask,
                )
                pooled = pooled.detach().float().contiguous().clone()
                out_subdir = output_dir / str(stat) / str(layer)
                out_subdir.mkdir(parents=True, exist_ok=True)
                out_file = out_subdir / f"{file_id}.pt"
                if skip_existing and out_file.exists():
                    continue
                torch.save({"label": label, "encodings": pooled}, out_file)

        block_clamp_warned: set[str] = set()
        if block_specs and block_pooling:
            for block in block_specs:
                top = block_output_dirname(str(block["name"]))
                name = str(block["name"])
                for layer in file_layers:
                    enc = encodings_rec[layer]
                    if enc.ndim != 2 or enc.shape[0] < 2:
                        continue
                    layer_truncate = int(enc.shape[0]) - 1
                    gap_mask = (
                        msa_non_gap_mask(str(aligned_seq), layer_truncate)
                        if aligned_seq
                        else None
                    )
                    msa_slice = clamp_block_slice(
                        int(block["msa_start"]),
                        int(block["msa_end"]),
                        layer_truncate,
                        block_name=name,
                        _warned=block_clamp_warned,
                    )
                    if msa_slice is None:
                        logging.error(
                            "Skipping block %s in aggregate pass for %s: "
                            "msa_start=%s >= truncate_len=%s",
                            name,
                            file_id,
                            block["msa_start"],
                            layer_truncate,
                        )
                        continue
                    try:
                        pooled = pool_token_repr_slice(
                            enc,
                            msa_slice[0],
                            msa_slice[1],
                            block_pooling,
                            normalize_positions=normalize_position_stats,
                            non_gap_mask=gap_mask,
                        )
                    except ValueError as exc:
                        logging.error(
                            "Skipping block %s in aggregate pass for %s: %s",
                            name,
                            file_id,
                            exc,
                        )
                        continue
                    pooled = pooled.detach().float().contiguous().clone()
                    out_subdir = output_dir / top / str(block_pooling) / str(layer)
                    out_subdir.mkdir(parents=True, exist_ok=True)
                    out_file = out_subdir / f"{file_id}.pt"
                    if skip_existing and out_file.exists():
                        continue
                    torch.save({"label": label, "encodings": pooled}, out_file)


def setup_logging(output_dir: Path, level: str):
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"aggregate_2d_{timestamp}.log"

    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(str(log_file)),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logging.info(f"Log file: {log_file}")


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description=(
            "Pool precomputed 2D ESM token embeddings to 1D vectors "
            "(no model load / inference)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        metavar="DIR",
        help=(
            "Directory of flat 2D ``<label>.pt`` files from "
            "``encode_proteins_from_fasta --save-2d``."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        metavar="DIR",
        help=(
            "Root for pooled outputs "
            "(``<strategy>/<layer>/``, optional ``block_*/``, extra stats)."
        ),
    )
    parser.add_argument(
        "--aggregate-pooling",
        nargs="+",
        default=None,
        choices=SUPPORTED_POOLING,
        metavar="STRATEGY",
        help=(
            "Pooling strategies to apply. "
            f"Choices: {SUPPORTED_POOLING}. "
            "If omitted and ``--block-boundaries`` is set, "
            "``average_token`` is enabled automatically."
        ),
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=None,
        metavar="L",
        help="Only pool these layer keys (default: all layers in each file).",
    )
    parser.add_argument(
        "--extra-pooling-stats",
        nargs="+",
        default=None,
        metavar="STAT",
        choices=list(EXTRA_STAT_NAMES),
        help=(
            "Additional per-residue stats under ``<output-dir>/<stat>/<layer>/``. "
            f"Choices: {', '.join(EXTRA_STAT_NAMES)}."
        ),
    )
    parser.add_argument(
        "--no-normalize-position-stats",
        action="store_true",
        help=(
            "Keep raw 0-based residue indices for ``argmax_pos`` / ``argmin_pos`` "
            "instead of scaling to [0, 1]."
        ),
    )
    parser.add_argument(
        "--block-boundaries",
        default=None,
        metavar="YAML",
        help=(
            "YAML with Spike domain blocks (``name``, ``msa_start``, ``msa_end``). "
            "Writes ``<output-dir>/block_<name>/<block-pooling>/<layer>/``."
        ),
    )
    parser.add_argument(
        "--block-pooling",
        default="average_token",
        choices=SUPPORTED_POOLING,
        metavar="STRATEGY",
        help="Pooling applied within each domain block slice.",
    )
    parser.add_argument(
        "--include-gap-columns",
        action="store_true",
        help=(
            "Include alignment gap columns when mean/stat pooling. "
            "Default: exclude gaps using ``aligned_seq`` in each 2D record."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip outputs that already exist on disk.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir, args.log_level)

    if not input_dir.is_dir():
        logging.error(f"Input directory not found: {input_dir}")
        raise FileNotFoundError(input_dir)

    pooling_strategies = list(args.aggregate_pooling or [])
    extra_stats = _dedupe_preserve_order(args.extra_pooling_stats or [])

    block_specs: list[dict] = []
    if args.block_boundaries:
        block_cfg = load_spike_block_boundaries(args.block_boundaries)
        block_specs = list(block_cfg.get("blocks") or [])
        msa_len = int(block_cfg.get("msa_length") or 0)
        if msa_len:
            validate_block_boundaries(block_cfg, msa_length=msa_len)
        else:
            validate_block_boundaries(block_cfg)
        max_end = max_block_msa_end(block_cfg)
        if max_end:
            logging.info(
                "Block YAML %s (%d blocks, max msa_end=%s)",
                args.block_boundaries,
                len(block_specs),
                max_end,
            )

    has_mean = any(s in MEAN_POOLING_STRATEGIES for s in pooling_strategies)
    if block_specs and not has_mean and "average_token" not in pooling_strategies:
        logging.warning(
            "--block-boundaries without global mean pooling: auto-enabling "
            "average_token alongside block outputs."
        )
        pooling_strategies = ["average_token", *pooling_strategies]

    if not pooling_strategies and not extra_stats and not block_specs:
        logging.error(
            "Nothing to do: set --aggregate-pooling and/or --extra-pooling-stats "
            "and/or --block-boundaries."
        )
        raise SystemExit(2)

    logging.info("=" * 60)
    logging.info("Post-hoc 2D embedding aggregation (no ESM)")
    logging.info(f"  Input dir     : {input_dir}")
    logging.info(f"  Output dir    : {output_dir}")
    logging.info(f"  Pooling       : {pooling_strategies or 'none'}")
    logging.info(f"  Layers        : {args.layers or 'all'}")
    logging.info(f"  Extra stats   : {extra_stats or 'none'}")
    logging.info(f"  Exclude gaps  : {not args.include_gap_columns}")
    if block_specs:
        logging.info(
            f"  Blocks        : {len(block_specs)} ({args.block_pooling})"
        )
    else:
        logging.info("  Blocks        : none")
    logging.info(f"  Skip existing : {args.skip_existing}")
    logging.info("=" * 60)

    aggregate_encodings(
        input_dir=input_dir,
        output_dir=output_dir,
        pooling_strategies=pooling_strategies,
        skip_existing=args.skip_existing,
        extra_pooling_stats=extra_stats,
        block_specs=block_specs,
        block_pooling=args.block_pooling if block_specs else None,
        normalize_position_stats=not args.no_normalize_position_stats,
        exclude_gap_columns=not args.include_gap_columns,
        layers=args.layers,
    )
    logging.info("All done.")


if __name__ == "__main__":
    main()
