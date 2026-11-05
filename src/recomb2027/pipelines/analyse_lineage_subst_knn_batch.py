#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     analyse_lineage_subst_knn_batch.py
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
"""Run ``analyse_lineage_subst_knn`` over many series rows from a pairs CSV.

All row-independent data (MSA, kNN encoding indexes, metadata, prediction NPZ) is
loaded **once**; rows are then processed sequentially or by a pool of forked
worker processes (``--num-workers``) that share that data copy-on-write. For
each row all query vectors (baseline + sample slots) are scored against the
kNN index in a single batched pass.

Example::

    PYTHONPATH=src python -m recomb2027.pipelines.analyse_lineage_subst_knn_batch \\
        --pairs-csv validation_pairs_test.csv \\
        --pred-npz b3_lineage_subst_prefix_swap_test.npz \\
        --msa filtered_msa_S1.fasta \\
        --out-dir kNN_prefix_swap_analysis/ \\
        --all-rows --num-workers 14 --continue-on-error

Outputs per row ``i`` (same names as the former one-process-per-row runs):
``knn_row{i}_report.json`` / ``.fasta`` and ``knn_row{i}_mutation_stats.json``.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from recomb2027.evaluation import knn_decode_utils as _knn
from recomb2027.pipelines.analyse_baseline_knn import (
    pairwise_knn_distances as _fast_pairwise_single,
    pairwise_knn_distances_multi as _fast_pairwise_multi,
)
from recomb2027.pipelines.analyse_lineage_subst_knn import (
    SharedContext,
    build_parser as build_row_parser,
    prepare_shared,
    process_row,
    read_pairs_csv,
)

logger = logging.getLogger(__name__)

_LOG_ROW: Optional[int] = None  # row being processed (log-line prefix)


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    for h in logging.getLogger().handlers:
        h.addFilter(_RowLogFilter())


class _RowLogFilter(logging.Filter):
    """Prefix log lines with ``[row N]`` while a row is being processed."""

    def filter(self, record: logging.LogRecord) -> bool:
        if _LOG_ROW is not None and not getattr(record, "_row_prefixed", False):
            record.msg = f"[row {_LOG_ROW}] {record.getMessage()}"
            record.args = None
            record._row_prefixed = True
        return True


# ---------------------------------------------------------------------------
# Batched kNN distances: precompute all query↔index distances of a row at once
# and serve them to the (unchanged) per-query ``analyse_row`` code path.
# ---------------------------------------------------------------------------

_DIST_CACHE: dict[tuple, np.ndarray] = {}


def _dist_key(query: np.ndarray, index: np.ndarray, metric: str) -> tuple:
    q32 = np.ascontiguousarray(np.asarray(query, dtype=np.float32).reshape(-1))
    return (id(index), metric, hashlib.blake2b(q32.tobytes(), digest_size=16).digest())


def _cached_pairwise_knn_distances(
    query: np.ndarray, index: np.ndarray, *, metric: str = "euclidean"
) -> np.ndarray:
    hit = _DIST_CACHE.get(_dist_key(query, index, metric))
    if hit is not None:
        return hit
    return _fast_pairwise_single(query, index, metric=metric)


def _install_fast_distances() -> None:
    """Make ``knn_decode_utils`` kNN code use cached / faster distances."""
    _knn.pairwise_knn_distances = _cached_pairwise_knn_distances


def _prime_row_distances(
    queries: list[dict[str, Any]], ctx: SharedContext, args: argparse.Namespace
) -> None:
    """Compute distances of every query of this row to the index in one pass."""
    _DIST_CACHE.clear()
    metric = str(args.knn_metric)
    for field, index, needed in (
        ("pred_mean", ctx.enc_mean, ctx.need_mean),
        ("pred_pca2d", ctx.enc_pca2d, ctx.need_pca),
    ):
        if not needed or index is None:
            continue
        vecs: list[np.ndarray] = []
        keys: list[tuple] = []
        for q in queries:
            v = q.get(field)
            if v is None:
                continue
            k = _dist_key(v, index, metric)
            if k in _DIST_CACHE or k in keys:
                continue
            keys.append(k)
            vecs.append(np.asarray(v, dtype=np.float32).reshape(-1))
        if not vecs:
            continue
        dists = _fast_pairwise_multi(np.stack(vecs, axis=0), index, metric=metric)
        for k, d in zip(keys, dists):
            _DIST_CACHE[k] = d


# ---------------------------------------------------------------------------
# Row selection / per-row argv
# ---------------------------------------------------------------------------


def resolve_row_indices(
    df: pd.DataFrame,
    *,
    all_rows: bool,
    rows: Optional[list[int]],
    row_start: Optional[int],
    row_end: Optional[int],
) -> list[int]:
    """Return iloc indices into ``df`` to process."""
    n = len(df)
    if all_rows:
        return list(range(n))
    if rows:
        return [int(r) for r in rows if 0 <= int(r) < n]
    if row_start is not None or row_end is not None:
        start = int(row_start) if row_start is not None else 0
        end = int(row_end) if row_end is not None else n
        return list(range(max(0, start), min(n, end)))
    return [0]


def _row_argv(args: argparse.Namespace, extra: list[str], iloc: int, out_dir: Path) -> list[str]:
    out_json = out_dir / f"knn_row{iloc}_report.json"
    out_mut = out_dir / f"knn_row{iloc}_mutation_stats.json"
    argv_row = [
        "--pairs-csv", str(args.pairs_csv),
        "--row", str(iloc),
        "--pred-npz", str(args.pred_npz),
        "--msa", str(args.msa),
        "--out-json", str(out_json),
        "--out-mutation-stats", str(out_mut),
        "--knn-k", str(args.knn_k),
        "--knn-space", str(args.knn_space),
        "--knn-metric", str(args.knn_metric),
        "--knn-mean-weight", str(args.knn_mean_weight),
        "--knn-pca2d-weight", str(args.knn_pca2d_weight),
        "--max-index-files", str(args.max_index_files),
        "--gt-blosum-k", str(args.gt_blosum_k),
    ]
    if args.pred_split:
        argv_row += ["--pred-split", str(args.pred_split)]
    if args.encodings_dir:
        argv_row += ["--encodings-dir", str(args.encodings_dir)]
    if args.enc_2d_dir:
        argv_row += ["--enc-2d-dir", str(args.enc_2d_dir)]
    if args.pca2d_basis_path:
        argv_row += ["--pca2d-basis-path", str(args.pca2d_basis_path)]
    if args.dataset_config:
        argv_row += ["--dataset-config", str(args.dataset_config)]
    if args.metadata_csv:
        argv_row += ["--metadata-csv", str(args.metadata_csv)]
    if args.knn_before_date:
        argv_row += ["--knn-before-date", str(args.knn_before_date)]
    if args.no_knn_date_filter:
        argv_row.append("--no-knn-date-filter")
    if args.pca2d_index_cache:
        argv_row += ["--pca2d-index-cache", str(args.pca2d_index_cache)]
    if args.no_pca2d_index_cache:
        argv_row.append("--no-pca2d-index-cache")
    argv_row.extend(extra)
    return argv_row


# ---------------------------------------------------------------------------
# Execution (sequential or forked pool)
# ---------------------------------------------------------------------------

_STATE: Optional[tuple[SharedContext, argparse.Namespace, list[str], Path]] = None


def _run_one(
    ctx: SharedContext,
    args: argparse.Namespace,
    extra: list[str],
    out_dir: Path,
    iloc: int,
) -> Optional[str]:
    """Process one row; return an error string or None."""
    global _LOG_ROW
    _LOG_ROW = int(iloc)
    try:
        row_args = build_row_parser().parse_args(_row_argv(args, extra, iloc, out_dir))
        process_row(row_args, ctx, pairs_iloc=iloc, on_queries=_prime_row_distances)
        return None
    except (Exception, SystemExit) as exc:
        msg = f"{type(exc).__name__}: {exc}"
        logger.error("row %d failed: %s", iloc, msg, exc_info=not isinstance(exc, SystemExit))
        return msg
    finally:
        _DIST_CACHE.clear()
        _LOG_ROW = None


def _worker_init() -> None:
    # Avoid BLAS/torch thread oversubscription with many workers.
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=1)
    except Exception:
        pass


def _worker(iloc: int) -> tuple[int, Optional[str]]:
    assert _STATE is not None
    ctx, args, extra, out_dir = _STATE
    return iloc, _run_one(ctx, args, extra, out_dir, iloc)


def run_batch(argv: Optional[list[str]] = None) -> dict:
    global _STATE
    p = argparse.ArgumentParser(
        description="Batch wrapper for analyse_lineage_subst_knn over pairs CSV rows."
    )
    p.add_argument("--pairs-csv", type=Path, required=True)
    p.add_argument("--pred-npz", type=Path, required=True)
    p.add_argument("--msa", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--all-rows", action="store_true")
    p.add_argument("--rows", type=int, nargs="*", default=None)
    p.add_argument("--row-start", type=int, default=None)
    p.add_argument("--row-end", type=int, default=None)
    p.add_argument("--pred-split", default=None)
    p.add_argument("--encodings-dir", type=Path, default=None)
    p.add_argument("--enc-2d-dir", type=Path, default=None)
    p.add_argument("--pca2d-basis-path", type=Path, default=None)
    p.add_argument("--knn-k", type=int, default=5)
    p.add_argument("--knn-space", default="combined")
    p.add_argument("--knn-metric", default="euclidean")
    p.add_argument("--knn-mean-weight", type=float, default=1.0)
    p.add_argument("--knn-pca2d-weight", type=float, default=0.75)
    p.add_argument("--max-index-files", type=int, default=5000)
    p.add_argument("--dataset-config", type=Path, default=None)
    p.add_argument("--metadata-csv", type=Path, default=None)
    p.add_argument("--knn-before-date", default=None)
    p.add_argument("--no-knn-date-filter", action="store_true")
    p.add_argument("--pca2d-index-cache", type=Path, default=None)
    p.add_argument("--no-pca2d-index-cache", action="store_true")
    p.add_argument("--gt-blosum-k", type=int, default=0)
    p.add_argument("--log-level", default="INFO")
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Log failures and continue with remaining rows.",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Worker processes (fork; the MSA / kNN index / predictions are loaded "
        "once and shared copy-on-write). Default 1 (sequential).",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip rows whose knn_row{i}_report.json already exists (resume).",
    )
    args, extra = p.parse_known_args(argv)
    _setup_logging(args.log_level)
    if int(args.num_workers) < 1:
        raise SystemExit("--num-workers must be >= 1")

    df = read_pairs_csv(args.pairs_csv)
    row_indices = resolve_row_indices(
        df,
        all_rows=args.all_rows,
        rows=args.rows,
        row_start=args.row_start,
        row_end=args.row_end,
    )
    if not row_indices:
        raise SystemExit("No rows selected.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_existing:
        todo = [r for r in row_indices if not (out_dir / f"knn_row{r}_report.json").is_file()]
        logger.info("skip-existing: %d / %d rows already done", len(row_indices) - len(todo), len(row_indices))
        row_indices_run = todo
    else:
        row_indices_run = list(row_indices)

    ok: list[int] = []
    failed: list[tuple[int, str]] = []

    if row_indices_run:
        # Shared heavy data, loaded once (parsed with the first row's arguments).
        first_args = build_row_parser().parse_args(
            _row_argv(args, extra, row_indices_run[0], out_dir)
        )
        _install_fast_distances()
        ctx = prepare_shared(first_args, preload_pred=True, check_row=False)

        n = len(row_indices_run)
        t0 = datetime.now()

        def _progress(iloc: int, err: Optional[str]) -> None:
            if err is None:
                ok.append(iloc)
            else:
                failed.append((iloc, err))
            logger.info(
                "batch progress: %d / %d rows done (row %d %s, failures=%d, elapsed=%s)",
                len(ok) + len(failed),
                n,
                iloc,
                "ok" if err is None else "FAILED",
                len(failed),
                str(datetime.now() - t0).split(".")[0],
            )

        nw = min(int(args.num_workers), n)
        if nw <= 1:
            for iloc in row_indices_run:
                logger.info("Processing row iloc=%d (%d/%d)", iloc, len(ok) + len(failed) + 1, n)
                err = _run_one(ctx, args, extra, out_dir, iloc)
                _progress(iloc, err)
                if err is not None and not args.continue_on_error:
                    raise SystemExit(f"row {iloc} failed: {err}")
        else:
            import multiprocessing as mp

            try:
                mp_ctx = mp.get_context("fork")
            except ValueError as e:
                raise SystemExit(
                    "--num-workers > 1 requires the 'fork' start method (Linux/macOS)"
                ) from e
            _STATE = (ctx, args, extra, out_dir)
            gc.collect()
            gc.freeze()  # keep GC from touching (and copying) shared pages
            logger.info("batch: %d rows, %d worker processes", n, nw)
            try:
                with mp_ctx.Pool(processes=nw, initializer=_worker_init) as pool:
                    for iloc, err in pool.imap_unordered(_worker, row_indices_run, chunksize=1):
                        _progress(iloc, err)
                        if err is not None and not args.continue_on_error:
                            pool.terminate()
                            raise SystemExit(f"row {iloc} failed: {err}")
            finally:
                _STATE = None

    summary = {
        "n_requested": len(row_indices),
        "n_ok": len(ok),
        "n_failed": len(failed),
        "n_skipped_existing": len(row_indices) - len(row_indices_run),
        "out_dir": str(out_dir),
        "failed_rows": sorted(failed),
    }
    logger.info(
        "Batch complete: %d ok, %d failed → %s",
        len(ok),
        len(failed),
        out_dir,
    )
    return summary


def main(argv: Optional[list[str]] = None) -> dict:
    return run_batch(argv)


if __name__ == "__main__":
    result = main()
    if result.get("n_failed"):
        sys.exit(1)
