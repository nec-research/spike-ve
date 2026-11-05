#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     predictor_utils.py
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
"""Helpers shared by the anchor / residual trainers and the inference scripts.

Device selection, PCA-2D dataset construction, MSA parsing and mutation-bucket metrics, PCA-2D evaluation
metrics, and the top-mode kNN lookup used by the prefix-swap inference."""
from __future__ import annotations

import sys
import torch
import logging
import re
import numpy as np

from recomb2027.core.paths import repo_root
from recomb2027.latent.io_utils import setup_logging
from recomb2027.core.gpu_utils import select_best_gpu
from pathlib import Path
from recomb2027.data.series import SeriesEncodingsDataset
from recomb2027.data.series_pca2d import SeriesPca2dEncodingsDataset
from recomb2027.latent.data_series.pca2d_features import PCA2D_S1_TARGET, load_pca2d_basis
from recomb2027.data.seq_pairwise_distance import pairwise_jump_distance
from typing import Any, Optional


# ============================================================================
# (merged from train_predictor.py)
# ============================================================================

REPO_ROOT = repo_root(__file__)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


logger = setup_logging()


def resolve_device(cfg) -> str:
    """Return a device string from config, auto-selecting the best GPU when
    ``device`` is ``"auto"`` or omitted."""
    device_cfg = cfg.get("device", "auto")
    if device_cfg == "auto":
        gpu_id, free_mem = select_best_gpu()
        if gpu_id is not None and torch.cuda.is_available():
            device = f"cuda:{gpu_id}"
        else:
            device = "cpu"
        logger.info(f"Auto-selected device: {device}")
        return device
    return device_cfg


# ============================================================================
# (merged from train_predictor_jepa_pca2d.py)
# ============================================================================

logger = logging.getLogger(__name__)


def _pca2d_sidecar_matches_basis(
    ds: SeriesPca2dEncodingsDataset,
    basis_path: Path,
    expected_flat_dim: int,
) -> bool:
    """True if loaded sidecar was built with the same basis path and flat dim."""
    got_flat = int(getattr(ds, "pca2d_flat_dim", 0) or 0)
    if got_flat != int(expected_flat_dim):
        return False
    stored = str(getattr(ds, "pca2d_basis_path", None) or "").strip()
    if not stored:
        return False
    try:
        return Path(stored).resolve() == Path(basis_path).resolve()
    except OSError:
        return stored == str(basis_path)


def _load_pca2d_dataset(
    npz_path: Path,
    *,
    enc_2d_dir: Path,
    basis_path: Path,
    pca2d_name: str,
    load_scaler: bool,
    force_rebuild: bool,
    fit_pca2d_scaler: bool = True,
    pca2d_scaler_mode: str = "flat",
) -> SeriesPca2dEncodingsDataset:
    basis = load_pca2d_basis(basis_path)
    expected_flat = int(basis["flat_dim"])
    base = SeriesEncodingsDataset.load_from_file(str(npz_path), load_scaler=load_scaler)
    ds = SeriesPca2dEncodingsDataset.from_series_dataset(base)
    ds.pca2d_scaler_mode = str(pca2d_scaler_mode or "flat")
    ds._pca2d_k = int(basis.get("k", 0) or 0)
    loaded = False if force_rebuild else ds.load_pca2d_sidecar(npz_path, load_scaler=load_scaler)
    if loaded and not _pca2d_sidecar_matches_basis(ds, basis_path, expected_flat):
        logger.warning(
            "Stale pca2d sidecar for %s (stored basis=%r flat_dim=%s; "
            "expected basis=%s flat_dim=%s); rebuilding",
            npz_path,
            getattr(ds, "pca2d_basis_path", None),
            getattr(ds, "pca2d_flat_dim", None),
            basis_path,
            expected_flat,
        )
        loaded = False
    if loaded and fit_pca2d_scaler and not ds.pca2d_scaler_matches_mode(pca2d_scaler_mode):
        logger.warning(
            "pca2d scaler incompatible with mode=%r (n_features_in_=%s) for %s; "
            "refitting",
            pca2d_scaler_mode,
            getattr(ds.pca2d_scaler, "n_features_in_", None),
            npz_path,
        )
        if "train" in str(npz_path):
            ds.fit_pca2d_scaler(mode=pca2d_scaler_mode)
            ds.save_pca2d_sidecar(npz_path, save_scaler=True)
        else:
            ds.pca2d_scaler = None
    if not loaded:
        ds.match_pca2d_encodings_from_2d_dir(
            enc_2d_dir, basis_path, name=pca2d_name
        )
        if fit_pca2d_scaler and (
            ds.split == "train" or "train" in str(npz_path)
        ):
            ds.fit_pca2d_scaler(mode=pca2d_scaler_mode)
        ds.save_pca2d_sidecar(npz_path, save_scaler=fit_pca2d_scaler)
    elif (
        fit_pca2d_scaler
        and ds.pca2d_scaler is None
        and "train" in str(npz_path)
    ):
        ds.fit_pca2d_scaler(mode=pca2d_scaler_mode)
        ds.save_pca2d_sidecar(npz_path, save_scaler=True)
    if not fit_pca2d_scaler:
        # Keep raw pca2d coords (aligns train MSE with GA pca2d_euclidean).
        ds.pca2d_scaler = None
    return ds


def build_datasets(cfg, data_cfg):
    processed = Path(data_cfg.processed_dir)
    train_path = processed / "train_dataset.npz"
    val_path = processed / "test_dataset.npz"
    cal_path = processed / "cal_dataset.npz"
    if not train_path.is_file():
        raise FileNotFoundError(train_path)
    if not val_path.is_file():
        raise FileNotFoundError(val_path)

    enc_2d_raw = str(
        cfg.get("enc_2d_dir") or data_cfg.get("enc_2d_dir") or ""
    ).strip()
    basis_raw = str(
        cfg.get("pca2d_basis_path") or data_cfg.get("pca2d_basis_path") or ""
    ).strip()
    if not enc_2d_raw:
        raise FileNotFoundError(
            "enc_2d_dir is missing; set it in the predictor or dataset config"
        )
    if not basis_raw:
        raise FileNotFoundError(
            "pca2d_basis_path is missing; set it in the predictor or dataset config"
        )
    enc_2d_dir = Path(enc_2d_raw)
    basis_path = Path(basis_raw)
    if not enc_2d_dir.is_dir():
        raise FileNotFoundError(f"enc_2d_dir not found: {enc_2d_dir}")
    if not basis_path.is_file():
        raise FileNotFoundError(f"pca2d_basis_path not found: {basis_path}")

    pca2d_name = str(cfg.get("pca2d_target_name", PCA2D_S1_TARGET))
    force = bool(cfg.get("force_rebuild_pca2d", False))
    scale_pca2d = bool(cfg.get("scale_pca2d", True))
    pca2d_scaler_mode = str(cfg.get("pca2d_scaler_mode", "flat") or "flat")

    train_ds = _load_pca2d_dataset(
        train_path,
        enc_2d_dir=enc_2d_dir,
        basis_path=basis_path,
        pca2d_name=pca2d_name,
        load_scaler=True,
        force_rebuild=force,
        fit_pca2d_scaler=scale_pca2d,
        pca2d_scaler_mode=pca2d_scaler_mode,
    )
    val_ds = _load_pca2d_dataset(
        val_path,
        enc_2d_dir=enc_2d_dir,
        basis_path=basis_path,
        pca2d_name=pca2d_name,
        load_scaler=True,
        force_rebuild=force,
        fit_pca2d_scaler=scale_pca2d,
        pca2d_scaler_mode=pca2d_scaler_mode,
    )
    cal_ds = None
    if cal_path.is_file():
        cal_ds = _load_pca2d_dataset(
            cal_path,
            enc_2d_dir=enc_2d_dir,
            basis_path=basis_path,
            pca2d_name=pca2d_name,
            load_scaler=True,
            force_rebuild=force,
            fit_pca2d_scaler=False,
            pca2d_scaler_mode=pca2d_scaler_mode,
        )
    if scale_pca2d and train_ds.pca2d_scaler is not None:
        # Keep cal/val on the train-fitted scaler *and* its mode/k (avoids
        # flat-mode transform against a per_pc scaler after sidecar reloads).
        train_mode = str(
            getattr(train_ds, "pca2d_scaler_mode", pca2d_scaler_mode) or pca2d_scaler_mode
        )
        train_k = int(getattr(train_ds, "_pca2d_k", 0) or 0)
        val_ds.pca2d_scaler = train_ds.pca2d_scaler
        val_ds.pca2d_scaler_mode = train_mode
        if train_k > 0:
            val_ds._pca2d_k = train_k
        if cal_ds is not None:
            cal_ds.pca2d_scaler = train_ds.pca2d_scaler
            cal_ds.pca2d_scaler_mode = train_mode
            if train_k > 0:
                cal_ds._pca2d_k = train_k
        pca2d_scaler_mode = train_mode
    else:
        train_ds.pca2d_scaler = None
        val_ds.pca2d_scaler = None
        if cal_ds is not None:
            cal_ds.pca2d_scaler = None
        logger.info("pca2d StandardScaler disabled (raw coords; scale_pca2d=false)")

    ifeature = cfg.get("input_feature_mode", "gatt_plus")
    for ds in (train_ds, val_ds, cal_ds):
        if ds is None:
            continue
        ds.input_feature_mode = ifeature
        ds.pca2d_gatt_plus = bool(cfg.get("pca2d_gatt_plus", True))
        ds.pca2d_scaler_mode = pca2d_scaler_mode
        if hasattr(train_ds, "_pca2d_k") and int(getattr(train_ds, "_pca2d_k", 0) or 0) > 0:
            ds._pca2d_k = int(train_ds._pca2d_k)
        ds.return_horizon_meta = bool(cfg.get("return_horizon_meta", False))

    sched = cfg.get("epoch_subsample_schedule")
    if sched:
        train_ds.epoch_subsample_schedule = tuple(int(x) for x in sched)
    else:
        train_ds.epoch_subsample_schedule = None
        train_ds.reset_timeline_indices()
    val_ds.reset_timeline_indices()
    if cal_ds is not None:
        cal_ds.reset_timeline_indices()

    if cal_ds is not None:
        return train_ds, cal_ds, val_ds, basis_path
    return train_ds, val_ds, basis_path


def parse_pca2d_splits(
    splits: tuple,
) -> tuple:
    """Normalize ``build_datasets`` return to ``(train, cal, test, basis_path)``."""
    if len(splits) == 4:
        train_ds, cal_ds, test_ds, basis_path = splits
        return train_ds, cal_ds, test_ds, basis_path
    train_ds, test_ds, basis_path = splits
    return train_ds, None, test_ds, basis_path


# ============================================================================
# (merged from gated_lstm_utils.py)
# ============================================================================

# Default (non-deduplicated) S1 MSA. Callers normally pass an explicit ``--msa``;
# this is only the fallback used when none is given.
MSA_PATH = (
    ""
)


# ---------------------------------------------------------------------------
# MSA parsing + mutation counting
# ---------------------------------------------------------------------------
def parse_msa(path):
    gid_to_seq = {}
    cur_ids, cur_parts = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if cur_ids and cur_parts:
                    seq = "".join(cur_parts)
                    for gid in cur_ids:
                        gid_to_seq[gid] = seq
                m = re.search(r"genome_id:([^|]+)", line[1:])
                cur_ids = re.findall(r"(\d+\.\d+)", m.group(1)) if m else []
                cur_parts = []
            else:
                cur_parts.append(line)
    if cur_ids and cur_parts:
        seq = "".join(cur_parts)
        for gid in cur_ids:
            gid_to_seq[gid] = seq
    return gid_to_seq


def compute_jump_distances(gids_src, gids_tgt, gid_to_seq, metric="blosum"):
    """Per-example distance (last predictive jump): Hamming count or BLOSUM dissimilarity.

    ``metric`` is ``"hamming"`` or ``"blosum"`` (default: BLOSUM62-based
    ``fast_pairwise_blosum_distance`` semantics).
    """
    return pairwise_jump_distance(
        gids_src, gids_tgt, gid_to_seq, metric=metric)


# Backwards-compatible name (returns floats when metric is blosum).
compute_hammings = compute_jump_distances


def compute_series_dates(gids_array, gid_to_date):
    """Return ``(N, K)`` float32 array of days for every gid in the series.

    Missing dates are linearly interpolated across the series; if interpolation
    is impossible (e.g. all-missing), zeros are used and the row is flagged.
    """
    gids = np.asarray(gids_array)
    N, K = gids_array.shape
    flat = gids.reshape(-1).astype(str)
    mapped = np.fromiter(
        (gid_to_date.get(g, np.nan) for g in flat),
        dtype=np.float32,
        count=flat.size,
    ).reshape(N, K)
    out = mapped.copy()
    xs = np.arange(K, dtype=np.float32)
    for i in range(N):
        row = out[i]
        mask = ~np.isnan(row)
        if mask.sum() == 0:
            row[:] = 0.0
        elif mask.sum() < K:
            row[:] = np.interp(xs, xs[mask], row[mask])
    return out


# ============================================================================
# (merged from pca2d_eval_metrics.py)
# ============================================================================

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Global R² over all elements."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def r2_per_axis(
    y_true: np.ndarray, y_pred: np.ndarray, axis: int
) -> list[float]:
    """R² per slice along ``axis`` (aggregate other dims)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    n = y_true.shape[axis]
    out: list[float] = []
    for i in range(n):
        t = np.take(y_true, i, axis=axis)
        p = np.take(y_pred, i, axis=axis)
        out.append(r2_score(t, p))
    return out


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation over flattened arrays."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt(np.sum(a**2) * np.sum(b**2)))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(a * b) / denom)


def delta_norm_stats(delta: np.ndarray, *, eps: float = 1e-4) -> dict[str, float]:
    """Summary stats of per-sample ‖Δ‖."""
    norms = np.linalg.norm(delta.reshape(delta.shape[0], -1), axis=1)
    return {
        "delta_norm_mean": float(norms.mean()),
        "delta_norm_std": float(norms.std()),
        "delta_norm_p95": float(np.percentile(norms, 95)),
        "delta_near_zero_frac": float(np.mean(norms < eps)),
    }


def aleatoric_floor_from_groups(
    delta: np.ndarray, group_ids: list[str]
) -> dict[str, float]:
    """Mean within-group variance of Δ (irreducible noise proxy)."""
    if len(group_ids) != delta.shape[0]:
        raise ValueError("group_ids length must match batch rows")
    groups: dict[str, list[np.ndarray]] = {}
    flat = delta.reshape(delta.shape[0], -1)
    for gid, row in zip(group_ids, flat):
        groups.setdefault(str(gid), []).append(row)
    vars_: list[float] = []
    for rows in groups.values():
        if len(rows) < 2:
            continue
        arr = np.stack(rows, axis=0)
        vars_.append(float(arr.var(axis=0).mean()))
    if not vars_:
        return {"aleatoric_floor": float("nan"), "n_groups_multi": 0.0}
    return {
        "aleatoric_floor": float(np.mean(vars_)),
        "n_groups_multi": float(len(vars_)),
    }


def summarize_ceiling_batch(
    *,
    target: torch.Tensor,
    mu_flat: torch.Tensor,
    seq_len: int,
    k: int,
    pred_flat: Optional[torch.Tensor] = None,
    pred_r_flat: Optional[torch.Tensor] = None,
    group_ids: Optional[list[str]] = None,
    eps: float = 1e-4,
) -> dict[str, Any]:
    """Aggregate ceiling metrics for one batch (CPU numpy internals)."""
    target_np = target.detach().cpu().numpy()
    mu_np = mu_flat.detach().cpu().numpy()
    n = target_np.shape[0]
    persist_mse = float(np.mean(target_np**2))
    mu_mse = float(np.mean((mu_np - target_np) ** 2))
    skill_mu = 1.0 - mu_mse / max(persist_mse, 1e-12)

    r_true = target_np - mu_np
    d_norm = np.linalg.norm(target_np, axis=1)
    r_norm = np.linalg.norm(r_true, axis=1)
    residual_energy_frac = float(np.mean(r_norm / np.clip(d_norm, 1e-8, None)))

    sites = target_np.reshape(n, seq_len, k)
    mu_sites = mu_np.reshape(n, seq_len, k)
    r2_pc = r2_per_axis(sites, mu_sites, axis=-1)
    r2_site = r2_per_axis(sites, mu_sites, axis=1)

    out: dict[str, Any] = {
        "n": float(n),
        "persist_mse": persist_mse,
        "mu_mse": mu_mse,
        "skill_mu": skill_mu,
        "residual_energy_frac": residual_energy_frac,
        "r2_per_pc": r2_pc,
        "r2_per_site_mean": float(np.nanmean(r2_site)),
        "r2_pc_mean": float(np.nanmean(r2_pc)),
        **delta_norm_stats(target_np, eps=eps),
    }
    if group_ids is not None:
        out.update(aleatoric_floor_from_groups(target_np, group_ids))

    if pred_flat is not None:
        pf = pred_flat.detach().cpu().numpy()
        out["full_mse"] = float(np.mean((pf - target_np) ** 2))
        out["delta_full_minus_mu"] = out["full_mse"] - mu_mse

    if pred_r_flat is not None:
        rh = pred_r_flat.detach().cpu().numpy()
        out["corr_r"] = pearson_corr(rh, r_true)
        out["norm_ratio"] = float(
            np.mean(np.linalg.norm(rh, axis=1) / np.clip(r_norm, 1e-8, None))
        )
    return out


def merge_metric_dicts(acc: dict[str, Any], batch: dict[str, Any]) -> dict[str, Any]:
    """Online merge for scalar sums; lists concatenated."""
    if not acc:
        acc = {"n": 0.0}
        for key in (
            "persist_mse",
            "mu_mse",
            "residual_energy_frac",
            "full_mse",
            "corr_r",
            "norm_ratio",
            "delta_norm_mean",
            "delta_near_zero_frac",
            "aleatoric_floor",
        ):
            acc[f"_sum_{key}"] = 0.0
        acc["_r2_pc"] = []
        acc["_r2_site"] = []
        acc["_delta_norms"] = []
        acc["_aleatoric"] = []

    bn = float(batch["n"])
    n0 = acc["n"]
    n1 = n0 + bn
    acc["n"] = n1

    for key in (
        "persist_mse",
        "mu_mse",
        "residual_energy_frac",
        "full_mse",
        "corr_r",
        "norm_ratio",
        "delta_norm_mean",
        "delta_near_zero_frac",
        "aleatoric_floor",
    ):
        sk = f"_sum_{key}"
        if key in batch and sk in acc:
            acc[sk] += float(batch[key]) * bn

    acc["_r2_pc"].extend(batch.get("r2_per_pc", []))
    acc["_r2_site"].append(batch.get("r2_per_site_mean", float("nan")))
    acc["_delta_norms"].extend(
        [batch.get("delta_norm_mean", 0.0)] * int(bn)
    )
    if "aleatoric_floor" in batch and not np.isnan(batch["aleatoric_floor"]):
        acc["_aleatoric"].append(batch["aleatoric_floor"])
    return acc


def finalize_merged_metrics(acc: dict[str, Any]) -> dict[str, float]:
    """Convert merged accumulator to report dict."""
    n = max(acc.get("n", 0.0), 1.0)
    out: dict[str, float] = {"n": acc.get("n", 0.0)}
    for key in (
        "persist_mse",
        "mu_mse",
        "residual_energy_frac",
        "full_mse",
        "corr_r",
        "norm_ratio",
        "delta_norm_mean",
        "delta_near_zero_frac",
        "aleatoric_floor",
    ):
        sk = f"_sum_{key}"
        if sk in acc:
            out[key] = acc[sk] / n
    if "persist_mse" in out and "mu_mse" in out:
        out["skill_mu"] = 1.0 - out["mu_mse"] / max(out["persist_mse"], 1e-12)
    if "_r2_pc" in acc and acc["_r2_pc"]:
        out["r2_pc_mean"] = float(np.nanmean(acc["_r2_pc"]))
    if "_r2_site" in acc and acc["_r2_site"]:
        out["r2_site_mean"] = float(np.nanmean(acc["_r2_site"]))
    if "_delta_norms" in acc and acc["_delta_norms"]:
        norms = np.asarray(acc["_delta_norms"], dtype=np.float64)
        out["delta_norm_std"] = float(norms.std())
        out["delta_norm_p95"] = float(np.percentile(norms, 95))
    if "_aleatoric" in acc and acc["_aleatoric"]:
        out["aleatoric_floor"] = float(np.mean(acc["_aleatoric"]))
    return out


# ============================================================================
# (merged from select_diffusion_top_modes_knn.py)
# ============================================================================

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)


def knn_density_scores(
    samples: np.ndarray,
    k: int,
    *,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-sample k-th neighbour distance and density ``1/(d_k+eps)``.

    Parameters
    ----------
    samples
        (S, D) float array of one series' diffusion draws.
    k
        Neighbour order ``k`` for ``d_k`` (must be >= 1). Capped at ``S - 1``.
    """
    from sklearn.neighbors import NearestNeighbors

    x = np.asarray(samples, dtype=np.float64)
    s, d = x.shape
    if s < 2:
        raise ValueError("Need at least 2 samples for k-NN density.")
    k_eff = int(min(max(1, k), s - 1))
    # Column index k_eff in kneighbors distances: self @ 0, then 1..k_eff-th other
    n_neighbors = k_eff + 1
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", algorithm="auto")
    nn.fit(x)
    dist, _ = nn.kneighbors(x)
    dk = dist[:, k_eff]
    rho = 1.0 / (dk + eps)
    return dk.astype(np.float64), rho.astype(np.float64)


def top_m_modes(
    samples: np.ndarray,
    k: int,
    m: int,
    *,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Indices of top ``m`` samples by density, their d_k and ρ (descending ρ)."""
    dk, rho = knn_density_scores(samples, k, eps=eps)
    s = int(samples.shape[0])
    m_eff = int(min(max(1, m), s))
    order = np.argsort(-rho, kind="stable")
    top_ix = order[:m_eff]
    return top_ix, dk[top_ix], rho[top_ix]
