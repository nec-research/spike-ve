#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     infer_residual.py
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
"""Infer / evaluate B4 (residual) checkpoints on cal and test and export the raw PCA-2D / mean predictions.

Run (this is the ``infer`` stage of ``scripts/run_experiment.sh``)::

    python -m recomb2027.latent.predictors.infer_residual \\
        --checkpoint .../Pca2dSiteUnifiedV4_best_model.pt \\
        --predictor-config configs/latent/predictors/residual_model_conscurr.yaml \\
        --dataset-config configs/latent/datasets/by_nearest_ancestor/split_2022-12-01_k5_esm1_layer33.yaml \\
        --splits cal,test --save-predictions --absolute --unscaled \\
        --npz-out .../pred_cal_test.npz --json-out .../pred_cal_test.json
"""
from __future__ import annotations

import numpy as np
import json
import logging
import torch
import argparse

from typing import Optional, Any
from recomb2027.latent.predictors.train_residual import (
    _iter_leaf_datasets,
    _active_timeline_indices,
    _edge_phylo_for_transition,
    _load_conservation_stats,
    _select_step_tensor,
    _unpack_v3_batch,
    build_unified_v3_model_from_cfg,
    build_v3_loaders,
    build_model_from_cfg,
)
from pathlib import Path
from torch.utils.data import DataLoader
from recomb2027.core.constants import PRIMARY_TARGET
from recomb2027.latent.data_series.horizon_collate import horizon_from_batch
from recomb2027.latent.models.residual_model import low_rank_site_nll, site_total_variance
from recomb2027.latent.predictors.predictor_utils import (
    finalize_merged_metrics,
    merge_metric_dicts,
    summarize_ceiling_batch,
    resolve_device,
)
from omegaconf import OmegaConf
from recomb2027.config_io import load_config


# ============================================================================
# (merged from ga_export_utils.py)
# ============================================================================

def leaf_dataset(ds):
    leaves = _iter_leaf_datasets(ds)
    return leaves[0] if leaves else ds


def source_gids_from_dataset(ds, n: Optional[int] = None) -> np.ndarray:
    """Prefix-last (parent) genome ids aligned to dataset order."""
    leaf = leaf_dataset(ds)
    gids = getattr(leaf, "sample_gids", None)
    if gids is None:
        raise AttributeError("dataset has no sample_gids")
    arr = np.asarray(gids)
    if arr.ndim == 2:
        out = arr[:, -2].astype(object)
    else:
        out = arr.astype(object)
    if n is not None:
        out = out[: int(n)]
    return out


def inverse_mean_rows(ds, rows: np.ndarray) -> np.ndarray:
    """Unscale ``[N, D]`` mean predictions with the series mean scaler."""
    leaf = leaf_dataset(ds)
    rows = np.asarray(rows, dtype=np.float64)
    if not hasattr(leaf, "inverse_scaling"):
        return rows.astype(np.float32)
    try:
        return np.asarray(leaf.inverse_scaling(rows), dtype=np.float32)
    except ValueError:
        # No scaler fitted
        return rows.astype(np.float32)


def unscale_pca2d_flat(ds, flat: np.ndarray) -> np.ndarray:
    """Unscale ``[N, L*k]`` pca2d flats with the pca2d StandardScaler."""
    leaf = leaf_dataset(ds)
    flat = np.asarray(flat, dtype=np.float32)
    if not hasattr(leaf, "_unscale_pca2d_cube"):
        return flat
    # _unscale expects last dim = flat_dim features
    return leaf._unscale_pca2d_cube(flat)


def reshape_pca2d_sites(
    flat: np.ndarray, *, seq_len: int, k: int
) -> np.ndarray:
    """``[N, L*k]`` → ``[N, L, k]``."""
    flat = np.asarray(flat, dtype=np.float32)
    if flat.ndim != 2:
        raise ValueError(f"expected [N, L*k], got {flat.shape}")
    n, d = flat.shape
    if int(seq_len) * int(k) != d:
        raise ValueError(
            f"flat_dim {d} != seq_len*k ({seq_len}*{k}={seq_len * k})"
        )
    return flat.reshape(n, int(seq_len), int(k))


def export_raw_pca2d_bundle(
    ds,
    flat_scaled: np.ndarray,
    *,
    seq_len: int,
    k: int,
) -> dict[str, np.ndarray]:
    raw_flat = unscale_pca2d_flat(ds, flat_scaled)
    sites = reshape_pca2d_sites(raw_flat, seq_len=seq_len, k=k)
    return {
        "pred_pca2d_flat_raw": raw_flat.astype(np.float32),
        "pred_pca2d_sites_raw": sites.astype(np.float32),
    }


# ============================================================================
# (merged from pca2d_v3_diagnostics.py)
# ============================================================================

def _histogram(values: np.ndarray, *, bins: int = 20) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "bins": [], "counts": []}
    counts, edges = np.histogram(arr, bins=int(bins))
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "bins": [float(x) for x in edges.tolist()],
        "counts": [int(x) for x in counts.tolist()],
    }


# ============================================================================
# (merged from infer_unified_pca2d_site_v3.py)
# ============================================================================

logger = logging.getLogger(__name__)

_DEFAULT_DATASET = (
    "configs/latent/datasets/by_nearest_ancestor/split_2022-12-01_k5_esm1_layer33.yaml"
)


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )


def _load_checkpoint(model, checkpoint: str, device: torch.device) -> None:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("trainable_state_dict") or ckpt.get("model_state_dict")
    try:
        model.load_state_dict(state, strict=False)
    except RuntimeError as exc:
        msg = str(exc)
        if "size mismatch" in msg and "site_gpt" in msg:
            raise RuntimeError(
                f"Failed to load {checkpoint}: frozen site-GPT shapes in the "
                "B3 checkpoint do not match site_gpt_checkpoint in the predictor "
                "YAML (e.g. decode_rank=64 flat vs decode_rank=256 site_gate). "
                "Point --predictor-config / site_gpt_checkpoint at the same "
                "anchor used when this B3 run was trained.\n"
                f"Original error: {msg}"
            ) from exc
        raise


def _primary_abs_last(xb: torch.Tensor, site_gpt, mean_dim: int) -> torch.Tensor:
    """Last-step absolute mean latent from dual-branch input ``[B,T,Din]``."""
    layout = site_gpt.hparams.get("mean_pca2d_layout") or {}
    sl = layout.get("primary_slice")
    if sl:
        s, e = int(sl[0]), int(sl[1])
        chunk = xb[:, -1, s:e]
    else:
        chunk = xb[:, -1, :mean_dim]
    if chunk.size(-1) == mean_dim:
        return chunk
    if chunk.size(-1) == 2 * mean_dim:
        return chunk[..., :mean_dim]
    raise ValueError(
        f"primary slice width {chunk.size(-1)} incompatible with mean_dim={mean_dim}"
    )


def _pca2d_abs_last(xb: torch.Tensor, site_gpt) -> torch.Tensor:
    return site_gpt.extract_abs_pca2d(xb)[:, -1, :]


EDGE_OVERRIDE_MODES = ("none", "median", "prev_hop", "zero", "const")


def _unwrap_substitution(ds):
    """Strip outermost ``SubstitutionFeatureDataset`` wrappers."""
    while type(ds).__name__ == "SubstitutionFeatureDataset" and hasattr(ds, "base"):
        ds = ds.base
    return ds


def _per_sample_edge_tau(dataset, *, hops_back: int = 0) -> np.ndarray:
    """Per-sample ``log1p`` BLOSUM distance of a prefix/terminal hop.

    ``hops_back=0`` is the terminal hop g3->g4 (the value the head sees today:
    source -> *future target*). ``hops_back=1`` is the last observed prefix hop
    g2->g3, which is available at forecast time. Same code path as the train /
    eval collate (``_edge_phylo_for_transition``), so the scale matches exactly.
    """
    ds = _unwrap_substitution(dataset)
    out = np.zeros(len(ds), dtype=np.float32)
    for i in range(len(ds)):
        n_trans = max(0, len(_active_timeline_indices(ds, i)) - 1)
        t = n_trans - 1 - int(hops_back)
        out[i] = _edge_phylo_for_transition(ds, i, t) if t >= 0 else 0.0
    return out


def _train_terminal_hop_median(train_dataset) -> float:
    """Median ``log1p`` BLOSUM terminal-hop distance over the training split."""
    vals = []
    for leaf in _iter_leaf_datasets(_unwrap_substitution(train_dataset)):
        edge = getattr(leaf, "edge_distances", None)
        if edge is None or edge.size == 0:
            continue
        vals.append(np.asarray(edge, dtype=np.float32)[:, -1])
    if not vals:
        raise RuntimeError("train dataset has no edge_distances for median override")
    raw = np.concatenate(vals)
    return float(np.log1p(np.maximum(np.median(raw), 0.0)))


def _apply_edge_override(
    edge_phylo: Optional[torch.Tensor],
    horizon: Optional[torch.Tensor],
    values: torch.Tensor,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Overwrite ``edge_phylo`` and ``horizon[..., 0]`` (edge_tau) with ``values [B]``."""
    if edge_phylo is not None:
        edge_phylo = values.to(edge_phylo).reshape(-1, 1).expand_as(edge_phylo).clone()
    if horizon is not None:
        horizon = horizon.clone()
        horizon[..., 0] = values.to(horizon)
    return edge_phylo, horizon


@torch.no_grad()
def export_v3_split_predictions(
    model,
    loader: DataLoader,
    *,
    pca2d_name: str,
    device: torch.device,
    cfg,
    max_batches: Optional[int] = None,
    save_predictions: bool = True,
    absolute: bool = False,
    unscaled: bool = False,
    hist_bins: int = 20,
    edge_override_values: Optional[np.ndarray] = None,
    true_edge_values: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Run v3 diagnostics and optionally collect prediction tensors for GA.

    ``edge_override_values`` (``[N]``, dataset order; loaders are unshuffled)
    replaces ``edge_phylo`` and ``horizon[..., 0]`` (the terminal-hop BLOSUM
    ``edge_tau`` that otherwise leaks the source->target distance).
    ``true_edge_values`` is only used to sanity-check alignment.
    """
    model.eval()
    site_gpt = model.site_gpt
    target_mode = str(cfg.get("target_mode", "diff") or "diff")
    want_abs = bool(absolute) and target_mode == "diff"
    want_raw = bool(unscaled)

    acc: dict[str, Any] = {}
    var_vals: list[np.ndarray] = []
    nll_vals: list[float] = []

    pred_pca2d_rows: list[np.ndarray] = []
    mu_pca2d_rows: list[np.ndarray] = []
    target_pca2d_rows: list[np.ndarray] = []
    pred_mean_rows: list[np.ndarray] = []
    target_mean_rows: list[np.ndarray] = []
    pred_pca2d_abs_rows: list[np.ndarray] = []
    mu_pca2d_abs_rows: list[np.ndarray] = []
    pred_mean_abs_rows: list[np.ndarray] = []
    activation_rows: list[np.ndarray] = []
    sample_count = 0
    row_offset = 0

    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        xb, yb, _, edge_phylo, horizon, substitution_feats, activation_target = (
            _unpack_v3_batch(batch, device, cfg=cfg)
        )
        if horizon is None:
            horizon = horizon_from_batch(batch, cfg=cfg, device=device)
        else:
            horizon = _select_step_tensor(horizon, -1, last_only=True)
        if edge_phylo is not None:
            edge_phylo = _select_step_tensor(edge_phylo, -1, last_only=True)
        if substitution_feats is not None:
            substitution_feats = _select_step_tensor(
                substitution_feats, -1, last_only=True
            )
        if activation_target is not None:
            activation_target = _select_step_tensor(
                activation_target, -1, last_only=True
            )

        bsz = int(xb.shape[0])
        lo, hi = row_offset, row_offset + bsz
        row_offset = hi
        if edge_override_values is not None:
            if true_edge_values is not None and bi == 0:
                ref = edge_phylo if edge_phylo is not None else (
                    horizon[:, :1] if horizon is not None else None
                )
                if ref is not None:
                    diff = float(
                        np.abs(
                            ref.detach().cpu().numpy().reshape(-1)
                            - true_edge_values[lo:hi]
                        ).max()
                    )
                    logger.info(
                        "edge-override alignment check (batch 0): max|batch - dataset|=%.3e",
                        diff,
                    )
                    if diff > 1e-5:
                        raise RuntimeError(
                            "edge-override: dataset-order edge values do not match "
                            f"batch edge_phylo (max diff {diff:.3e}); aborting"
                        )
            vals = torch.as_tensor(
                edge_override_values[lo:hi], dtype=torch.float32, device=device
            )
            edge_phylo, horizon = _apply_edge_override(edge_phylo, horizon, vals)

        target_pca = yb[pca2d_name].to(device)[:, -1, :]
        has_mean = PRIMARY_TARGET in yb
        target_mean = (
            yb[PRIMARY_TARGET].to(device)[:, -1, :] if has_mean else None
        )

        dist = model.predict_distribution(
            xb,
            horizon=horizon,
            edge_phylo=edge_phylo,
            substitution_feats=substitution_feats,
            activation_target=activation_target,
        )
        pred_flat = dist["pred_flat"]
        mu_flat = dist["mu_flat"]

        if "pred_mean" in dist:
            mean_pred = dist["pred_mean"]
        else:
            h = site_gpt.encode_trunk(xb)
            out_z = site_gpt._decode_diff(h, expand_pca2d=False)
            mean_pred = out_z[PRIMARY_TARGET][:, -1, :]

        nll = low_rank_site_nll(
            target_pca,
            mu_flat,
            dist["mu_r"],
            dist["log_sigma"],
            dist["U"],
            seq_len=model.seq_len,
            k=model.k,
        )
        nll_vals.append(float(nll.item()))
        total_var = site_total_variance(dist["log_sigma"], dist["U"])
        var_vals.append(total_var.detach().cpu().numpy().reshape(-1))

        batch_m = summarize_ceiling_batch(
            target=target_pca,
            mu_flat=mu_flat,
            pred_flat=pred_flat,
            seq_len=model.seq_len,
            k=model.k,
        )
        batch_m["nll"] = float(nll.item())
        batch_m["n"] = float(target_pca.shape[0])
        acc = merge_metric_dicts(acc, batch_m)

        if save_predictions:
            pred_pca2d_rows.append(pred_flat.detach().cpu().numpy())
            mu_pca2d_rows.append(mu_flat.detach().cpu().numpy())
            target_pca2d_rows.append(target_pca.detach().cpu().numpy())
            pred_mean_rows.append(mean_pred.detach().cpu().numpy())
            if target_mean is not None:
                target_mean_rows.append(target_mean.detach().cpu().numpy())
            if "activation_probs" in dist:
                activation_rows.append(
                    dist["activation_probs"].detach().cpu().numpy()
                )

            if want_abs:
                base_pca = _pca2d_abs_last(xb, site_gpt)
                pred_pca2d_abs_rows.append(
                    (base_pca + pred_flat).detach().cpu().numpy()
                )
                mu_pca2d_abs_rows.append(
                    (base_pca + mu_flat).detach().cpu().numpy()
                )
                mean_dim = int(mean_pred.shape[-1])
                base_mean = _primary_abs_last(xb, site_gpt, mean_dim)
                pred_mean_abs_rows.append(
                    (base_mean + mean_pred).detach().cpu().numpy()
                )
            sample_count += int(target_pca.shape[0])

    out = finalize_merged_metrics(acc)
    out["full_x0_mse"] = out.get("full_mse", out.get("full_x0_mse", float("nan")))
    if "full_mse" in out and "persist_mse" in out:
        out["skill_full"] = 1.0 - out["full_mse"] / max(out["persist_mse"], 1e-12)
    out["nll_mean"] = float(np.mean(nll_vals)) if nll_vals else float("nan")
    var_concat = np.concatenate(var_vals) if var_vals else np.array([])
    out["var_hist"] = _histogram(var_concat, bins=hist_bins)
    out["var_mean"] = out["var_hist"].get("mean", float("nan"))

    result: dict[str, Any] = {"metrics": out, "n_samples": sample_count}
    if save_predictions and pred_pca2d_rows:
        result["pred_pca2d"] = np.concatenate(pred_pca2d_rows, axis=0)
        result["mu_pca2d"] = np.concatenate(mu_pca2d_rows, axis=0)
        result["target_pca2d"] = np.concatenate(target_pca2d_rows, axis=0)
        result["pred_mean"] = np.concatenate(pred_mean_rows, axis=0)
        if target_mean_rows:
            result["target_mean"] = np.concatenate(target_mean_rows, axis=0)
        if activation_rows:
            result["activation_probs"] = np.concatenate(activation_rows, axis=0)
        if want_abs and pred_pca2d_abs_rows:
            result["pred_pca2d_abs"] = np.concatenate(pred_pca2d_abs_rows, axis=0)
            result["mu_pca2d_abs"] = np.concatenate(mu_pca2d_abs_rows, axis=0)
            result["pred_mean_abs"] = np.concatenate(pred_mean_abs_rows, axis=0)
            result["absolute"] = True
        else:
            result["absolute"] = False
        result["target_mode"] = target_mode

        try:
            result["source_gids"] = source_gids_from_dataset(
                loader.dataset, n=int(result["pred_pca2d"].shape[0])
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not export source_gids: %s", exc)

        if want_raw:
            ds = loader.dataset
            mean_src = result.get("pred_mean_abs", result["pred_mean"])
            pca_src = result.get("pred_pca2d_abs", result["pred_pca2d"])
            result["pred_mean_raw"] = inverse_mean_rows(ds, mean_src)
            raw_bundle = export_raw_pca2d_bundle(
                ds,
                pca_src,
                seq_len=int(model.seq_len),
                k=int(model.k),
            )
            result.update(raw_bundle)
            result["unscaled"] = True
        else:
            result["unscaled"] = False
    return result


def run_v3_inference(
    cfg,
    data_cfg,
    *,
    checkpoint: str,
    splits: tuple[str, ...] = ("cal", "test"),
    batch_size: Optional[int] = None,
    max_batches: Optional[int] = None,
    json_out: Optional[str] = None,
    npz_out: Optional[str] = None,
    absolute: bool = False,
    unscaled: bool = False,
    build_model_fn=None,
    edge_override: str = "none",
    edge_override_value: Optional[float] = None,
) -> dict[str, Any]:
    device = torch.device(resolve_device(cfg))
    site_ckpt = str(cfg.get("site_gpt_checkpoint", ""))
    bs = int(batch_size or cfg.get("batch_size", 32))
    save_preds = bool(npz_out)
    if unscaled and not save_preds:
        raise ValueError("--unscaled requires --npz-out")
    if edge_override not in EDGE_OVERRIDE_MODES:
        raise ValueError(f"edge_override must be one of {EDGE_OVERRIDE_MODES}")
    if edge_override == "const" and edge_override_value is None:
        raise ValueError("edge_override=const requires edge_override_value")

    train_loader, cal_loader, test_loader, pca2d_name, _ = build_v3_loaders(
        cfg, data_cfg, batch_size=bs, num_workers=0
    )

    median_value: Optional[float] = None
    if edge_override == "median":
        median_value = _train_terminal_hop_median(train_loader.dataset)
        logger.info(
            "edge-override=median: train terminal-hop log1p(BLOSUM) median=%.6f",
            median_value,
        )
    split_loaders = {
        "train": train_loader,
        "cal": cal_loader,
        "test": test_loader,
    }

    builder = build_model_fn or build_unified_v3_model_from_cfg
    model = builder(cfg, site_gpt_checkpoint=site_ckpt, device=device, data_cfg=data_cfg)
    _load_checkpoint(model, checkpoint, device)
    if bool(cfg.get("append_msa_conservation", False)):
        # Match training: conservation is often a zero-init buffer in the ckpt;
        # always re-load MSA stats after state_dict so infer matches train.
        conservation = _load_conservation_stats(
            cfg, data_cfg, seq_len=int(model.seq_len)
        )
        if conservation is not None:
            model.set_conservation_stats(conservation)
            logger.info(
                "Loaded MSA conservation stats for inference shape=%s",
                tuple(conservation.shape),
            )
        else:
            logger.warning(
                "append_msa_conservation=true but conservation stats could not "
                "be loaded; site features will use zeros (train/infer mismatch)."
            )
    print(
        f"Loaded v3 checkpoint from {checkpoint} "
        f"(substitution_dim={model.substitution_dim}, "
        f"activation={model.use_activation_supervision}, "
        f"mean_residual={model.use_mean_residual}, "
        f"anchor={site_ckpt})",
        flush=True,
    )

    report: dict[str, Any] = {
        "checkpoint": checkpoint,
        "site_gpt_checkpoint": site_ckpt,
        "pca2d_target_name": pca2d_name,
        "substitution_dim": int(model.substitution_dim),
        "use_activation_supervision": bool(model.use_activation_supervision),
        "use_mean_residual": bool(model.use_mean_residual),
        "absolute": bool(absolute),
        "unscaled": bool(unscaled),
        "edge_override": edge_override,
        "edge_override_train_median_log1p": median_value,
        "edge_override_value": (
            None if edge_override_value is None else float(edge_override_value)
        ),
        "splits": {},
    }
    basis_path = str(cfg.get("pca2d_basis_path", "") or "")
    npz_payload: dict[str, Any] = {
        "checkpoint": np.asarray(checkpoint),
        "site_gpt_checkpoint": np.asarray(site_ckpt),
        "pca2d_target_name": np.asarray(pca2d_name),
        "absolute": np.asarray(bool(absolute)),
        "unscaled": np.asarray(bool(unscaled)),
        "basis_path": np.asarray(basis_path),
        "edge_override": np.asarray(edge_override),
        "seq_len": np.asarray(int(model.seq_len)),
        "k": np.asarray(int(model.k)),
    }

    export_keys = (
        "pred_pca2d",
        "mu_pca2d",
        "target_pca2d",
        "pred_mean",
        "target_mean",
        "pred_pca2d_abs",
        "mu_pca2d_abs",
        "pred_mean_abs",
        "pred_mean_raw",
        "pred_pca2d_flat_raw",
        "pred_pca2d_sites_raw",
        "activation_probs",
        "source_gids",
    )

    for name in splits:
        loader = split_loaders.get(name)
        if loader is None:
            logger.warning("split %s not available; skipping", name)
            continue
        override_vals = true_vals = None
        if edge_override != "none":
            n_ds = len(loader.dataset)
            true_vals = _per_sample_edge_tau(loader.dataset, hops_back=0)
            if edge_override == "median":
                override_vals = np.full(n_ds, median_value, dtype=np.float32)
            elif edge_override == "prev_hop":
                override_vals = _per_sample_edge_tau(loader.dataset, hops_back=1)
            elif edge_override == "zero":
                override_vals = np.zeros(n_ds, dtype=np.float32)
            else:
                override_vals = np.full(
                    n_ds, float(edge_override_value), dtype=np.float32
                )
            logger.info(
                "[%s] edge-override=%s: true terminal-hop mean=%.5f -> used mean=%.5f "
                "(corr=%.3f)",
                name,
                edge_override,
                float(true_vals.mean()),
                float(override_vals.mean()),
                float(np.corrcoef(true_vals, override_vals)[0, 1])
                if override_vals.std() > 0
                else float("nan"),
            )
        split_out = export_v3_split_predictions(
            model,
            loader,
            pca2d_name=pca2d_name,
            device=device,
            cfg=cfg,
            max_batches=max_batches,
            save_predictions=save_preds,
            absolute=absolute,
            unscaled=unscaled,
            edge_override_values=override_vals,
            true_edge_values=true_vals,
        )
        metrics = split_out["metrics"]
        report["splits"][name] = metrics
        full_mse = float(metrics.get("full_mse", metrics.get("full_x0_mse", float("nan"))))
        print(
            f"\n=== v3 inference: {name} (n={int(metrics.get('n', 0))}) ===\n"
            f"  full_mse={full_mse:.6f}\n"
            f"  skill_full={metrics.get('skill_full', float('nan')):.4f}\n"
            f"  skill_mu={metrics.get('skill_mu', float('nan')):.4f}",
            flush=True,
        )
        if save_preds:
            for key in export_keys:
                if key in split_out:
                    npz_payload[f"{name}_{key}"] = split_out[key]
                    arr = split_out[key]
                    shape = getattr(arr, "shape", None)
                    print(f"  exported {name}_{key} shape={shape}", flush=True)

    if npz_out:
        out_npz = Path(npz_out).expanduser()
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_npz, **npz_payload)
        print(f"\nWrote predictions → {out_npz}", flush=True)
        report["npz_out"] = str(out_npz)

    if json_out:
        out_path = Path(json_out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote metrics → {out_path}", flush=True)
        report["json_out"] = str(out_path)

    return report


# ============================================================================
# (merged from infer_unified_pca2d_site_v4.py)
# ============================================================================

_DEFAULT_PREDICTOR = "configs/latent/predictors/residual_model_conscurr.yaml"


def main():
    parser = argparse.ArgumentParser(description="Infer B4 unified pca2d site v4.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--predictor-config", default=_DEFAULT_PREDICTOR)
    parser.add_argument("--dataset-config", default=_DEFAULT_DATASET)
    parser.add_argument("--splits", default="cal,test")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--npz-out", default=None)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--absolute", action="store_true")
    parser.add_argument("--unscaled", action="store_true")
    parser.add_argument("--site-gpt-checkpoint", default=None,
                        help="anchor checkpoint (A4); default: site_gpt_checkpoint of the YAML")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="same overrides that were used for training")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    _setup_logging(args.log_level)
    if args.save_predictions and not args.npz_out:
        parser.error("--save-predictions requires --npz-out")

    cfg = OmegaConf.create(
        OmegaConf.to_container(load_config(args.predictor_config), resolve=True)
    )
    if args.set:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(args.set)))
    if args.site_gpt_checkpoint:
        cfg["site_gpt_checkpoint"] = str(args.site_gpt_checkpoint)
    data_cfg = load_config(args.dataset_config)
    splits = tuple(s.strip() for s in str(args.splits).split(",") if s.strip())
    run_v3_inference(
        cfg,
        data_cfg,
        checkpoint=args.checkpoint,
        splits=splits,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        json_out=args.json_out,
        npz_out=args.npz_out,
        absolute=args.absolute,
        unscaled=args.unscaled,
        build_model_fn=build_model_from_cfg,
    )


if __name__ == "__main__":
    main()
