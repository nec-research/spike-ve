#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     fit_pca2d_from_2d_encodings.py
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
"""Fit a feature-PCA (pca2d_k) basis from precomputed 2D ESM encodings.

Produces the same artifact schema as
``recomb2027.generation.fit_pca2d_feature_basis`` for use with
JEPA-GPT ``pca2d_s1`` training and the GA ``pca2d_euclidean`` metric.

Example (S1 2D encodings, seq_len from MSA)::

    python -m recomb2027.latent.datasets.fit_pca2d_from_2d_encodings \\
        --enc-2d-dir /mnt/.../encodings/ESM1B/filtered_msa_S1/2d/33 \\
        --msa /mnt/.../phylotree_matched/filtered_msa_S1.fasta \\
        --train-dataset /mnt/.../series/.../train_dataset.npz \\
        --output /mnt/.../ga/pca2d_k8_layer33_S1_train.pt \\
        --k 8 --layer 33 --max-gids 1000

``seq_len`` resolution order: ``--seq-len`` > ``--msa`` > ``--region`` >
median length of 2D ``.pt`` matrices.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence, Set

import numpy as np
import torch
from Bio import SeqIO
from sklearn.decomposition import PCA

from recomb2027.core.paths import repo_root
from recomb2027.latent.data_series.esm_2d_io import (
    build_gid_file_index,
    load_2d_encoding,
)
from recomb2027.latent.data_series.pca2d_features import load_aa_matrix_for_basis

REPO_ROOT = repo_root(__file__)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def seq_len_from_msa(msa_path: Path) -> int:
    """Return alignment column count from the first FASTA record."""
    msa_path = Path(msa_path)
    if not msa_path.is_file():
        raise FileNotFoundError(f"MSA FASTA not found: {msa_path}")
    for record in SeqIO.parse(str(msa_path), "fasta"):
        n = len(record.seq)
        if n <= 0:
            raise ValueError(f"Empty sequence in MSA: {msa_path} id={record.id}")
        return int(n)
    raise ValueError(f"No sequences found in MSA: {msa_path}")


def resolve_seq_len(
    *,
    seq_len: Optional[int],
    msa: Optional[Path],
    region: Optional[str],
    fit_gids: Sequence[str],
    index: dict[str, Path],
    layer: int,
) -> tuple[int, str]:
    """Resolve AA ``seq_len`` and a short source label for logging."""
    if seq_len is not None:
        return int(seq_len), "cli --seq-len"

    if msa is not None:
        n = seq_len_from_msa(msa)
        return n, f"msa:{msa}"

    if region:
        from recomb2027.generation.utils.data_utils.msa_length_cut import (
            resolve_msa_slice,
        )

        sl = resolve_msa_slice(sequence_region=region)
        if sl is None:
            raise SystemExit(f"Could not resolve region={region!r}")
        return int(sl[1] - sl[0]), f"region:{region} slice={sl}"

    if not fit_gids:
        raise SystemExit("Cannot infer seq_len: empty fit gid list")
    raw_lengths = []
    for gid in list(fit_gids)[: min(100, len(fit_gids))]:
        arr = load_2d_encoding(index[gid], layer, drop_cls=False)
        raw_lengths.append(int(arr.shape[0]))
    return int(np.median(raw_lengths)), "median 2d .pt row count"


def unique_gids_from_train_npz(train_dataset: Path) -> Set[str]:
    """Collect unique genome ids from a series ``train_dataset.npz``."""
    train_dataset = Path(train_dataset)
    if not train_dataset.is_file():
        raise FileNotFoundError(f"Train dataset not found: {train_dataset}")
    data = np.load(train_dataset, allow_pickle=True)
    if "sample_gids" not in data:
        raise KeyError(f"{train_dataset} has no 'sample_gids' array")
    gids = np.asarray(data["sample_gids"], dtype=object).ravel()
    return {str(g) for g in gids}


def select_fit_gids(
    index: dict[str, Path],
    *,
    train_dataset: Optional[Path],
    max_gids: int,
    seed: int,
) -> tuple[list[str], int, Optional[int]]:
    """Return ``(fit_gids, pool_size, n_train_unique)``.

    When ``train_dataset`` is set, the candidate pool is the intersection of
    unique train ``sample_gids`` with keys present in ``index``.
    ``max_gids <= 0`` means use the full candidate pool (no subsample).
    """
    if train_dataset is not None:
        train_gids = unique_gids_from_train_npz(train_dataset)
        candidates = sorted(g for g in train_gids if g in index)
        n_train = len(train_gids)
        n_missing = n_train - len(candidates)
        if not candidates:
            raise SystemExit(
                f"No overlap between train gids ({n_train} unique) and "
                f"2D encodings ({len(index)} files)"
            )
        if n_missing:
            print(
                f"Train split: {n_train} unique gids; "
                f"{len(candidates)} have 2D encodings "
                f"({n_missing} missing from enc-2d-dir)",
                flush=True,
            )
        else:
            print(
                f"Train split: {n_train} unique gids; all have 2D encodings",
                flush=True,
            )
    else:
        candidates = sorted(index)
        n_train = None

    pool_size = len(candidates)
    if max_gids is not None and int(max_gids) > 0:
        n_take = min(int(max_gids), pool_size)
        if n_take < pool_size:
            rng = np.random.default_rng(seed)
            fit_gids = sorted(rng.choice(candidates, size=n_take, replace=False))
        else:
            fit_gids = list(candidates)
    else:
        fit_gids = list(candidates)

    return fit_gids, pool_size, n_train


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fit pca2d_k feature PCA from 2D ESM encoding directory."
    )
    p.add_argument("--enc-2d-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--layer", type=int, default=33)
    p.add_argument("--k", type=int, default=8)
    p.add_argument(
        "--max-gids",
        type=int,
        default=200,
        help=(
            "Max genomes to fit on after pool selection. "
            "Use 0 to fit on all candidates (recommended with --train-dataset)."
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--train-dataset",
        type=Path,
        default=None,
        metavar="NPZ",
        help=(
            "Train series NPZ (must contain sample_gids). Restrict the fit pool "
            "to unique genome ids from this split that also have 2D encodings."
        ),
    )
    p.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="AA rows after optional CLS strip (overrides --msa / --region).",
    )
    p.add_argument(
        "--msa",
        type=Path,
        default=None,
        metavar="FASTA",
        help=(
            "Aligned FASTA used to build the 2D encodings. When --seq-len is "
            "omitted, seq_len = length of the first record (MSA columns)."
        ),
    )
    p.add_argument(
        "--region",
        type=str,
        default=None,
        help=(
            "Optional spike region (e.g. S1) to set seq_len from block boundaries "
            "when --seq-len and --msa are omitted."
        ),
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.enc_2d_dir.is_dir():
        raise SystemExit(f"2D encoding dir not found: {args.enc_2d_dir}")
    if args.k < 1:
        raise SystemExit("--k must be >= 1")
    if args.msa is not None and args.region and args.seq_len is None:
        print(
            "Note: both --msa and --region set; using --msa for seq_len "
            "(pass --seq-len to override).",
            flush=True,
        )

    index = build_gid_file_index(args.enc_2d_dir)
    if not index:
        raise SystemExit(f"No .pt files under {args.enc_2d_dir}")
    print(f"Index built: {len(index)} gids found in enc-2d-dir", flush=True)

    fit_gids, pool_size, n_train = select_fit_gids(
        index,
        train_dataset=args.train_dataset,
        max_gids=int(args.max_gids),
        seed=int(args.seed),
    )
    n_take = len(fit_gids)

    args.seq_len, seq_len_source = resolve_seq_len(
        seq_len=args.seq_len,
        msa=args.msa,
        region=args.region,
        fit_gids=fit_gids,
        index=index,
        layer=int(args.layer),
    )
    print(f"seq_len={args.seq_len} (from {seq_len_source})", flush=True)

    pool_desc = f"pool={pool_size}"
    if n_train is not None:
        pool_desc = f"train_unique≈{n_train}, pool_with_2d={pool_size}"
    print(
        f"Fitting PCA on {n_take} genomes ({pool_desc}), k={args.k}, "
        f"layer={args.layer}, seq_len={args.seq_len}",
        flush=True,
    )

    rows: list[np.ndarray] = []
    lengths: list[int] = []
    for i, gid in enumerate(fit_gids):
        raw = load_2d_encoding(index[gid], args.layer, drop_cls=False)
        lengths.append(int(raw.shape[0]))
        mat = load_aa_matrix_for_basis(
            index[gid], layer=args.layer, seq_len=int(args.seq_len)
        )
        rows.append(mat)
        if (i + 1) % 50 == 0:
            print(f"  loaded {i + 1}/{n_take}", flush=True)

    data = np.vstack(rows).astype(np.float64)
    feat_dim = int(data.shape[1])
    if args.k > feat_dim:
        raise SystemExit(f"--k={args.k} > feature_dim={feat_dim}")
    print(f"  PCA data shape={data.shape}", flush=True)

    pca = PCA(n_components=int(args.k), random_state=int(args.seed))
    pca.fit(data)
    var = float(pca.explained_variance_ratio_.sum())
    seq_len = int(args.seq_len)
    print(
        f"  variance explained={var * 100:.1f}%  seq_len={seq_len}  "
        f"raw L range=[{min(lengths)}, {max(lengths)}]",
        flush=True,
    )

    payload = {
        "W": torch.from_numpy(pca.components_.astype(np.float32)),
        "mu": torch.from_numpy(pca.mean_.astype(np.float32)),
        "k": int(args.k),
        "feature_dim": feat_dim,
        "seq_len": seq_len,
        "seq_len_source": seq_len_source,
        "layer": int(args.layer),
        "variance_explained": var,
        "n_fit_gids": n_take,
        "n_pool_gids": pool_size,
        "mode": "feature_pca",
        "region": args.region,
        "msa": str(args.msa) if args.msa else None,
        "enc_2d_dir": str(args.enc_2d_dir),
        "train_dataset": str(args.train_dataset) if args.train_dataset else None,
        "fit_restricted_to_train_split": bool(args.train_dataset is not None),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"Wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
