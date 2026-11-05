#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     build_msa_substitution_sidecar.py
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
"""B1: build sparse per-sample MSA substitution sidecar from series NPZs."""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
from omegaconf import OmegaConf
from recomb2027.config_io import load_config

from recomb2027.data.series import SeriesEncodingsDataset
from recomb2027.latent.data_series.msa_substitution_features import (
    build_substitution_sidecar_from_gids,
    save_substitution_sidecar,
)
from recomb2027.latent.data_series.latent_decoder_io import parse_msa_gids
from recomb2027.latent.data_series.pca2d_features import load_pca2d_basis
from recomb2027.latent.data_series.stem_predecessor import resolve_n_jobs

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = "configs/latent/diagnostics/b1_substitution_sidecar.yaml"
_DEFAULT_DATASET = (
    "configs/latent/datasets/by_nearest_ancestor/"
    "dataset_nearest_ancestor_deterministic_tau4_2022-12-01_S1_cal.yaml"
)


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )


def _split_npz_paths(processed_dir: str, splits: tuple[str, ...]) -> dict[str, Path]:
    root = Path(processed_dir)
    mapping = {
        "train": root / "train_dataset.npz",
        "cal": root / "cal_dataset.npz",
        "test": root / "test_dataset.npz",
    }
    out: dict[str, Path] = {}
    for name in splits:
        p = mapping.get(name)
        if p is not None and p.is_file():
            out[name] = p
            continue
        logger.warning("split %s NPZ not found under %s; skipping", name, root)
    return out


def mirror_heldout_sidecar(
    out_path: Path,
    alt_path: Path,
    *,
    this_source: Path,
    alt_source: Optional[Path],
    alt_built_this_run: bool = False,
) -> str:
    """Keep ``msa_substitution_sidecar_{val,test}.npz`` consistent.

    Loaders resolve ``test`` or ``val`` depending on the dataset leaf split name. When
    both names are served by the *same* series NPZ (the usual case: no separate
    ``test_dataset.npz``), the alternate file is (re)written whenever it is missing or
    differs from the freshly built sidecar, so a rebuild can never leave a stale copy.

    Returns one of ``written | updated | unchanged | skipped_built_this_run |
    skipped_distinct_source``.
    """
    if alt_built_this_run:
        return "skipped_built_this_run"  # the alternate split builds its own file
    if alt_source is not None and Path(alt_source).resolve() != Path(this_source).resolve():
        logger.warning(
            "Not mirroring %s over %s: alternate split has its own series NPZ (%s); "
            "rebuild it with --splits including that split.",
            out_path.name,
            alt_path.name,
            alt_source,
        )
        return "skipped_distinct_source"
    new_bytes = out_path.read_bytes()
    if not alt_path.is_file():
        alt_path.write_bytes(new_bytes)
        logger.info("Mirrored %s -> %s (new)", out_path.name, alt_path.name)
        return "written"
    if alt_path.read_bytes() == new_bytes:
        return "unchanged"
    alt_path.write_bytes(new_bytes)
    logger.warning(
        "Updated stale %s from %s (contents differed).", alt_path.name, out_path.name
    )
    return "updated"


def build_sidecar_for_split(
    npz_path: Path,
    *,
    msa_path: str,
    seq_len: int,
    out_path: Path,
    split_name: str = "",
    gid_to_seq: Optional[dict[str, str]] = None,
    n_jobs: int = 1,
) -> dict[str, Any]:
    label = split_name or npz_path.stem
    logger.info("[%s] loading series NPZ: %s", label, npz_path)
    t0 = time.perf_counter()
    ds = SeriesEncodingsDataset.load_from_file(str(npz_path), load_scaler=False)
    gids = np.asarray(ds.sample_gids, dtype=object)
    logger.info(
        "[%s] loaded %d series in %.1fs",
        label,
        len(gids),
        time.perf_counter() - t0,
    )
    payload, stats = build_substitution_sidecar_from_gids(
        gids,
        msa_path=msa_path if gid_to_seq is None else None,
        gid_to_seq=gid_to_seq,
        seq_len=seq_len,
        n_jobs=n_jobs,
    )
    save_substitution_sidecar(payload, out_path, stats=stats)
    stats["sidecar_path"] = str(out_path)
    stats["n_samples_loaded"] = float(len(gids))
    return stats


def run_build_sidecar(
    cfg,
    data_cfg,
    *,
    splits: tuple[str, ...] = ("train", "cal", "test"),
    json_out: Optional[str] = None,
    n_jobs: Optional[int] = None,
) -> dict[str, Any]:
    msa_path = str(cfg.get("msa_path") or data_cfg.get("msa_path") or "")
    if not msa_path:
        raise ValueError("msa_path required in config or dataset config")

    if n_jobs is None:
        n_jobs = int(cfg.get("n_jobs", data_cfg.get("n_jobs", 1)) or 1)
    n_workers = resolve_n_jobs(n_jobs)

    basis_path = cfg.get("pca2d_basis_path")
    seq_len = int(cfg.get("seq_len", 0) or 0)
    if seq_len <= 0 and basis_path:
        logger.info("Resolving seq_len from pca2d basis: %s", basis_path)
        basis = load_pca2d_basis(basis_path)
        seq_len = int(basis["seq_len"])
        logger.info("Using seq_len=%d from basis", seq_len)

    processed_dir = str(data_cfg["processed_dir"])
    out_dir = Path(cfg.get("sidecar_dir") or processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_paths = _split_npz_paths(processed_dir, splits)
    if not split_paths:
        raise FileNotFoundError(
            f"No split NPZs found under {processed_dir} for splits={splits}"
        )

    logger.info("B1 sidecar: loading MSA once for all splits: %s", msa_path)
    t_msa = time.perf_counter()
    gid_to_seq = parse_msa_gids(msa_path)
    logger.info(
        "B1 sidecar: MSA ready — %d genome_ids in %.1fs",
        len(gid_to_seq),
        time.perf_counter() - t_msa,
    )

    logger.info(
        "B1 sidecar build: seq_len=%d out_dir=%s workers=%d splits=%s",
        seq_len,
        out_dir,
        n_workers,
        ", ".join(split_paths.keys()),
    )
    report: dict[str, Any] = {
        "msa_path": msa_path,
        "seq_len": seq_len,
        "processed_dir": processed_dir,
        "splits": {},
    }

    for split_idx, (name, npz_path) in enumerate(split_paths.items(), start=1):
        logger.info(
            "=== Split %d/%d: %s ===",
            split_idx,
            len(split_paths),
            name,
        )
        out_path = out_dir / f"msa_substitution_sidecar_{name}.npz"
        t_split = time.perf_counter()
        stats = build_sidecar_for_split(
            npz_path,
            msa_path=msa_path,
            seq_len=seq_len,
            out_path=out_path,
            split_name=name,
            gid_to_seq=gid_to_seq,
            n_jobs=n_workers,
        )
        # Also mirror held-out sidecar under the alternate val/test name so
        # loaders that report split="val" or "test" both resolve.
        if name in ("test", "val"):
            alt = "val" if name == "test" else "test"
            alt_path = out_dir / f"msa_substitution_sidecar_{alt}.npz"
            # The held-out series is test_dataset.npz; there is no separate val series NPZ.
            outcome = mirror_heldout_sidecar(
                out_path,
                alt_path,
                this_source=npz_path,
                alt_source=None,
                alt_built_this_run=alt in split_paths,
            )
            stats[f"mirror_{alt}"] = outcome
            if outcome in ("written", "updated"):
                stats["also_wrote"] = str(alt_path)
        report["splits"][name] = stats
        elapsed = time.perf_counter() - t_split
        print(
            f"\n=== B1 substitution sidecar: {name} ({elapsed:.1f}s) ===\n"
            f"  n_samples={int(stats['n_samples_loaded'])}\n"
            f"  n_transitions={int(stats['n_transitions'])}\n"
            f"  hamming_mean={stats['hamming_mean']:.3f}\n"
            f"  hamming_median={stats['hamming_median']:.3f}\n"
            f"  hamming_p95={stats['hamming_p95']:.3f}\n"
            f"  → {out_path}",
            flush=True,
        )
        logger.info("[%s] split complete in %.1fs", name, elapsed)

    if json_out:
        out = Path(json_out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote {out}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(
        description="B1: build sparse MSA substitution sidecar (prefix-only, no leakage)."
    )
    parser.add_argument("--config", default=_DEFAULT_CONFIG)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument(
        "--splits",
        default=None,
        help="Comma-separated splits (default: train,cal,test from config).",
    )
    parser.add_argument("--json-out", default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="CPU workers for prefix record build (1=serial, -1=all CPUs). "
        "Overrides config n_jobs.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    _setup_logging(args.log_level)

    cfg = load_config(args.config)
    ds_path = args.dataset_config or cfg.get("dataset_config") or _DEFAULT_DATASET
    data_cfg = load_config(str(ds_path))
    if args.splits:
        splits = tuple(s.strip() for s in str(args.splits).split(",") if s.strip())
    else:
        cfg_splits = cfg.get("splits")
        if cfg_splits is None:
            splits = ("train", "cal", "test")
        elif OmegaConf.is_list(cfg_splits) or isinstance(cfg_splits, (list, tuple)):
            splits = tuple(str(s).strip() for s in cfg_splits if str(s).strip())
        else:
            splits = tuple(
                s.strip() for s in str(cfg_splits).split(",") if s.strip()
            )
    json_out = args.json_out or cfg.get("json_out")

    logger.info(
        "Starting B1 substitution sidecar build (splits=%s, dataset=%s, workers=%s)",
        ", ".join(splits),
        ds_path,
        args.workers if args.workers is not None else cfg.get("n_jobs", 1),
    )
    run_build_sidecar(
        cfg,
        data_cfg,
        splits=splits,
        json_out=json_out,
        n_jobs=args.workers,
    )
    logger.info("B1 substitution sidecar build finished.")


if __name__ == "__main__":
    main()
