#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     predictor_trainer.py
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
"""
Train finals prediction model to do the inference of variant positions in the latent space.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List

import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import matplotlib.pyplot as plt

from recomb2027.latent.models.model_utils import (
    load_training_resume_state,
    save_model_w_hyperparams,
)

import logging

from recomb2027.latent.plotting import plot_training_loss

logger = logging.getLogger(__name__)
from sklearn.preprocessing import StandardScaler

from recomb2027.core.constants import PRIMARY_TARGET


# ---- Utils -----------

def _move_yb_to_device(yb, device):
    if isinstance(yb, dict):
        return {k: v.to(device).float() for k, v in yb.items()}
    return yb.to(device).float()


def resolve_val_selection_score(
    val_metrics: dict,
    *,
    early_stopping_dict=None,
    loss_kwargs=None,
    scale_state: dict | None = None,
) -> tuple[float, str]:
    """Scalar used for best-ckpt / early-stop / LR schedule.

    ``early_stopping.metric``:
      - ``mean`` (default): primary head only
      - ``<target>``: that key from ``val_metrics`` (e.g. ``pca2d_s1``)
      - ``joint`` / ``weighted``: sum of ``w_t * val_mse_t`` (optionally scaled)

    For joint mode, weights are **fixed** (not curriculum-modulated), from:
      1. ``early_stopping.target_weights``, else
      2. ``loss_kwargs["selection_target_weights"]`` (static snapshot), else
      3. ``{mean: 1.0}``

    Optional scale fairness (when heads live in different units)::

        early_stopping.normalize: true
        early_stopping.target_scales: {mean: 1.0, pca2d_s1: 10.0}  # optional

    With ``normalize: true``, each term is ``w_t * mse_t / scale_t``.
    If ``target_scales`` omit a key, the first observed ``mse_t`` (stored in
    ``scale_state``) becomes that target's scale and is then frozen.
    Live ``target_loss_weights`` are intentionally ignored so scores stay
    comparable across mean-only → ramp epochs.
    """
    es = dict(early_stopping_dict or {})
    metric = str(es.get("metric", PRIMARY_TARGET)).lower()
    metrics = {str(k): float(v) for k, v in val_metrics.items()}

    if metric in ("joint_site", "site_capture"):
        # Latent site-level selection (``site_capture`` from ``site_metrics.evaluate_site_metrics``).
        if "site_capture" not in metrics:
            raise KeyError(f"early_stopping.metric={metric!r} needs site_capture in val metrics")
        sw = float(es.get("site_weight", 1.0))
        site_term = sw * (1.0 - metrics["site_capture"])
        if metric == "site_capture":
            return float(site_term), "site_capture"
        base, label = resolve_val_selection_score(
            val_metrics,
            early_stopping_dict={**es, "metric": "joint"},
            loss_kwargs=loss_kwargs,
            scale_state=scale_state,
        )
        return float(base + site_term), f"{label}+site*{sw:g}"

    if metric in ("joint", "weighted"):
        weights = es.get("target_weights")
        if weights is None and loss_kwargs:
            weights = loss_kwargs.get("selection_target_weights")
        if not weights:
            weights = {PRIMARY_TARGET: 1.0}
        normalize = bool(es.get("normalize", False))
        fixed_scales = {
            str(k): float(v)
            for k, v in dict(es.get("target_scales") or {}).items()
        }
        score = 0.0
        used = []
        for tname, w in weights.items():
            w = float(w)
            if w == 0.0:
                continue
            key = PRIMARY_TARGET if str(tname) in ("mean", PRIMARY_TARGET) else str(tname)
            if key not in metrics and str(tname) in metrics:
                key = str(tname)
            if key not in metrics:
                continue
            mse = metrics[key]
            if normalize:
                if key in fixed_scales and fixed_scales[key] > 0:
                    scale = fixed_scales[key]
                else:
                    if scale_state is None:
                        scale = max(mse, 1e-6)
                    else:
                        if key not in scale_state or float(scale_state[key]) <= 0:
                            scale_state[key] = max(mse, 1e-6)
                        scale = float(scale_state[key])
                term = w * (mse / scale)
                used.append(f"{key}*{w:g}/s{scale:.4g}")
            else:
                term = w * mse
                used.append(f"{key}*{w:g}")
            score += term
        label = "joint[" + "+".join(used) + "]" if used else "joint"
        return float(score), label

    if metric in ("mean", PRIMARY_TARGET):
        return float(metrics[PRIMARY_TARGET]), PRIMARY_TARGET

    if metric == "raw_jump_ratio":
        if "raw_jump_ratio_deviation" in metrics:
            return float(metrics["raw_jump_ratio_deviation"]), "raw_jump_ratio_dev"
        if "raw_jump_ratio" not in metrics:
            raise KeyError(
                "early_stopping.metric='raw_jump_ratio' requires "
                "raw_jump_ratio in val_metrics"
            )
        ratio = float(metrics["raw_jump_ratio"])
        return abs(ratio - 1.0), "raw_jump_ratio"

    if metric not in metrics:
        raise KeyError(
            f"early_stopping.metric={metric!r} not in val_metrics {list(metrics)}"
        )
    return float(metrics[metric]), metric


def _yb_mean_last(yb, model):
    """Last-step mean target from tensor or dict batch labels."""
    from recomb2027.core.constants import PRIMARY_TARGET

    if isinstance(yb, dict):
        return yb[PRIMARY_TARGET][:, -1, :]
    if yb.dim() == 3:
        return yb[:, -1, :]
    mean_idx = list(getattr(model, "prediction_targets", (PRIMARY_TARGET,))).index(
        PRIMARY_TARGET
    )
    return yb[:, -1, mean_idx, :]


def _yb_target_last(yb, model, tname: str):
    """Last-step target tensor for a named head."""
    from recomb2027.core.constants import PRIMARY_TARGET

    if isinstance(yb, dict):
        if tname not in yb:
            return None
        return yb[tname][:, -1, :]
    if tname == PRIMARY_TARGET:
        return _yb_mean_last(yb, model)
    if yb.dim() == 4:
        names = list(getattr(model, "prediction_targets", (PRIMARY_TARGET,)))
        if tname not in names:
            return None
        return yb[:, -1, names.index(tname), :]
    return None


@torch.no_grad()
def evaluate_mse_on_validation(model, val_loader, device, n_samples=1):
    """Compute last-step MSE on the validation set.

    Returns a dict with at least ``mean`` (used for early stopping / LR schedule).
    When the model is multi-target, also includes ``mse_<target>`` for each head.
    For variational models, ``n_samples > 1`` keeps best-of-N on the mean head.
    """
    from recomb2027.core.constants import PRIMARY_TARGET

    model.eval()
    is_variational = hasattr(model, "sample")
    target_names = list(getattr(model, "prediction_targets", (PRIMARY_TARGET,)))

    totals = {PRIMARY_TARGET: 0.0}
    for tname in target_names:
        totals[tname] = 0.0

    n_batches = 0
    for batch in val_loader:
        if len(batch) == 3:
            xb, yb, _ = batch
        else:
            xb, yb = batch
        xb = xb.to(device).float()
        yb = _move_yb_to_device(yb, device)
        yb_last = _yb_mean_last(yb, model)

        if is_variational and n_samples > 1:
            preds = model.sample(xb, n_samples=n_samples)
            sq_err = (preds - yb_last.unsqueeze(1)).pow(2).mean(dim=-1)
            batch_loss = sq_err.min(dim=1).values.mean()
            totals[PRIMARY_TARGET] += batch_loss.item()
        else:
            y_pred = model(xb, return_dict=True) if len(target_names) > 1 else model(xb)
            if isinstance(y_pred, tuple):
                y_pred = y_pred[0]
            if not isinstance(y_pred, dict):
                y_pred = {PRIMARY_TARGET: y_pred}

            for tname in target_names:
                if tname not in y_pred:
                    continue
                y_t = _yb_target_last(yb, model, tname)
                if y_t is None:
                    continue
                p_t = y_pred[tname][:, -1, :]
                totals[tname] += F.mse_loss(p_t, y_t).item()

        n_batches += 1

    denom = max(n_batches, 1)
    metrics = {tname: totals[tname] / denom for tname in totals}
    # Alias used by callers / TensorBoard
    metrics["mean"] = metrics[PRIMARY_TARGET]
    return metrics


@torch.no_grad()
def evaluate_coverage(model, val_loader, device, n_samples=20):
    """Evaluate generative quality metrics for variational models.

    Returns a dict with:
        best_of_n_mse : mean best-of-N MSE across samples
        diversity      : mean pairwise L2 distance between sampled predictions
        coverage_50    : fraction of targets where at least one sample falls
                         within the 50th percentile of per-sample L2 distance
        coverage_90    : same at the 90th percentile
        mean_target_l2 : mean L2 distance to the nearest sample (across targets)
    """
    if not hasattr(model, 'sample'):
        return {}

    model.eval()

    all_min_dists = []
    all_diversities = []

    for xb, yb in val_loader:
        xb = xb.to(device).float()
        yb = yb.to(device).float()
        yb_last = yb[:, -1, :]                                 # [B, D]

        preds = model.sample(xb, n_samples=n_samples)           # [B, N, D]
        B, N, D = preds.shape

        # -- best-of-N L2 distance to target --
        diffs = preds - yb_last.unsqueeze(1)                    # [B, N, D]
        l2_dists = diffs.norm(dim=-1)                           # [B, N]
        min_dist = l2_dists.min(dim=1).values                   # [B]
        all_min_dists.append(min_dist.cpu())

        # -- diversity: mean pairwise L2 between samples --
        # Efficient: for each pair (i,j), L2 between preds[:,i,:] and preds[:,j,:]
        # Use broadcasting: [B, N, 1, D] - [B, 1, N, D] -> [B, N, N, D]
        pw_diffs = preds.unsqueeze(2) - preds.unsqueeze(1)      # [B, N, N, D]
        pw_l2 = pw_diffs.norm(dim=-1)                           # [B, N, N]
        # Upper-triangle mean (excluding diagonal)
        mask = torch.triu(torch.ones(N, N, device=device), diagonal=1).bool()
        pw_mean = pw_l2[:, mask].mean(dim=-1)                   # [B]
        all_diversities.append(pw_mean.cpu())

    min_dists = torch.cat(all_min_dists)                        # [total_samples]
    diversities = torch.cat(all_diversities)

    # coverage thresholds based on the distribution of min distances
    p50 = min_dists.median().item()
    p90 = min_dists.quantile(0.9).item()

    return {
        "mean_target_l2": min_dists.mean().item(),
        "median_target_l2": p50,
        "p90_target_l2": p90,
        "diversity": diversities.mean().item(),
    }


# ------ Training loops ----------

def train_prediction_model(
        model,
        model_loss_func,
        train_dataset,
        val_dataset,
        batch_size,
        device,
        lr=1e-3,
        weight_decay=0.0,
        epochs=100,
        clip_grad_norm=1.0,
        early_stopping_dict=None,
        scheduler_dict=None,
        save_dir="./checkpoints",
        tensorboard_dir="./runs",
        kl_annealing_dict=None,
        train_collate_fn=None,
        loss_kwargs=None,
        resume_checkpoint=None,
        all_step_loss=None,
        epoch_callback=None,
        amp_dtype="fp16",
        ema_decay=None,
        input_noise_std=0.0,
        site_metric_cfg=None,
):
    """ Train models.

    Optional additions (all default-off): ``ema_decay`` (validate / checkpoint with EMA weights),
    ``input_noise_std`` (Gaussian latent noise on the input channels),
    ``site_metric_cfg`` (``{seq_len, k, pca2d_name, topk}``; adds ``site_capture`` to the selection score).

    When ``resume_checkpoint`` is set, restores model/optimizer state and continues
    from the next epoch up to ``epochs``.

    ``all_step_loss``: if True, supervise every prefix step; if False, only the
    last step. When ``None``, falls back to ``kl_annealing.all_step_loss``
    (default False if no kl_annealing block — prefer passing explicitly for GPT).

    ``epoch_callback``: optional ``fn(model, epoch, loss_kwargs)`` invoked at the
    start of each epoch (e.g. pca2d loss curriculum / head freeze).

    ``amp_dtype``: ``"fp16"`` (default GradScaler AMP), ``"bf16"`` (autocast
    bfloat16, no GradScaler), or ``"off"`` / ``"fp32"`` (full precision).
    """
    # --- Setup ---
    device  = torch.device(device)

    amp_key = str(amp_dtype or "fp16").lower().strip()
    if amp_key in ("off", "fp32", "none", "false", "0"):
        amp_enabled = False
        autocast_dtype = torch.float32
        use_grad_scaler = False
    elif amp_key in ("bf16", "bfloat16"):
        amp_enabled = device.type == "cuda"
        autocast_dtype = torch.bfloat16
        use_grad_scaler = False
    elif amp_key in ("fp16", "float16", "half"):
        amp_enabled = device.type == "cuda"
        autocast_dtype = torch.float16
        use_grad_scaler = amp_enabled
    else:
        raise ValueError(
            f"amp_dtype must be fp16, bf16, or off; got {amp_dtype!r}"
        )
    logger.info(
        "AMP: enabled=%s dtype=%s grad_scaler=%s",
        amp_enabled,
        amp_key if amp_enabled else "fp32",
        use_grad_scaler,
    )

    # Mutable copy so curriculum callbacks can update weights in-place.
    loss_kwargs = dict(loss_kwargs or {})

    # Include all parameters (including currently-frozen ones) so curriculum
    # unfreeze can resume updates without rebuilding the optimizer.
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    initial_lrs = [float(pg["lr"]) for pg in optimizer.param_groups]

    scheduler = None
    reduce_lr_min_epoch = 0
    if scheduler_dict is not None:
        # ReduceLROnPlateau (same family as ``train_gated_lstm``).
        factor = float(scheduler_dict.get(
            "reduce_lr_factor", scheduler_dict.get("gamma", 0.5)))
        patience = int(scheduler_dict.get(
            "reduce_lr_patience", scheduler_dict.get("step_size", 5)))
        reduce_lr_min_epoch = int(scheduler_dict.get("reduce_lr_min_epoch", 0) or 0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=factor, patience=patience)
        if reduce_lr_min_epoch > 0:
            logger.info(
                "ReduceLROnPlateau held until epoch >= %d",
                reduce_lr_min_epoch,
            )

    early_stop_enabled = False
    early_stop_patience = epochs
    if early_stopping_dict is not None:
        early_stop_enabled = bool(early_stopping_dict.get("enabled", True))
        if early_stop_enabled:
            early_stop_patience = int(early_stopping_dict.get("patience", 20))
        else:
            logger.info("early_stopping.enabled=false; running all %d epochs", epochs)

    selection_metric = str(
        (early_stopping_dict or {}).get("metric", PRIMARY_TARGET)
    ).lower()
    logger.info(
        "Val selection metric=%s (best ckpt / early-stop / LR schedule)",
        selection_metric,
    )

    # AMP scaler: only for fp16; bf16 / full fp32 do not need it.
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

    model = model.to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=train_collate_fn if train_collate_fn is not None else None,
    )

    # TensorBoard setup
    writer = None
    if tensorboard_dir is not None:
        run_name = datetime.now().strftime(f"{model.__class__.__name__}_%Y%m%d-%H%M%S")
        writer = SummaryWriter(log_dir=f"{tensorboard_dir}/{run_name}")

    # Tracking variables
    train_model_losses = []
    train_mse_losses, val_mse_losses = [], []
    train_mse_per_target: dict[str, list[float]] = {}
    val_mse_per_target: dict[str, list[float]] = {}
    selection_scale_state: dict[str, float] = {}
    last_sel_label = PRIMARY_TARGET
    best_val_loss = float("inf")
    best_epoch = -1
    early_stop_counter = 0
    start_epoch = 0

    if resume_checkpoint:
        resume_path = os.path.expanduser(str(resume_checkpoint))
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"resume_checkpoint not found: {resume_path}")
        resume_info = load_training_resume_state(
            resume_path, model, optimizer, device=device,
        )
        start_epoch = resume_info["start_epoch"]
        best_val_loss = resume_info["best_val_loss"]
        best_epoch = resume_info["best_epoch"]
        early_stop_counter = resume_info["early_stop_counter"]
        logger.info(
            "Resumed from %s: epoch=%d, best_epoch=%d, best_val_loss=%.6f, "
            "early_stop_counter=%d; training epochs %d..%d",
            resume_path,
            start_epoch,
            best_epoch,
            best_val_loss,
            early_stop_counter,
            start_epoch + 1,
            epochs,
        )
        curriculum = getattr(model, "_pca2d_curriculum", None) or {}
        mean_only = int(curriculum.get("mean_only_epochs", 0) or 0)
        if (
            start_epoch > mean_only
            and bool(curriculum.get("reset_early_stop_on_unfreeze", True))
        ):
            selection_scale_state.clear()
            best_val_loss = float("inf")
            best_epoch = -1
            early_stop_counter = 0
            logger.info(
                "Resume past pca2d unfreeze (epoch %d > mean_only %d): "
                "reset selection state",
                start_epoch,
                mean_only,
            )
        if start_epoch >= epochs:
            logger.warning(
                "resume epoch (%d) >= epochs (%d); nothing to train",
                start_epoch,
                epochs,
            )
            return {
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "train_mse_losses": train_mse_losses,
                "val_mse_losses": val_mse_losses,
            }

    # --- KL annealing setup (for variational models) ---
    kl_warmup_epochs = 0
    kl_free_bits = 0.0
    kl_beta_max = 1.0
    recon_loss_type = "mse"
    log_mse_eps = 1e-4
    # Prefer explicit all_step_loss; else kl_annealing; else False (legacy GPT default).
    use_all_step_loss = False
    if all_step_loss is not None:
        use_all_step_loss = bool(all_step_loss)
    if kl_annealing_dict is not None:
        kl_warmup_epochs = kl_annealing_dict.get("warmup_epochs", 20)
        kl_free_bits = kl_annealing_dict.get("free_bits", 0.0)
        kl_beta_max = kl_annealing_dict.get("beta_max", 1.0)
        recon_loss_type = kl_annealing_dict.get("recon_loss", "mse")
        log_mse_eps = kl_annealing_dict.get("log_mse_eps", 1e-4)
        if all_step_loss is None:
            use_all_step_loss = bool(kl_annealing_dict.get("all_step_loss", True))
    logger.info(
        "Training with all_step_loss=%s (only_last=%s)",
        use_all_step_loss,
        not use_all_step_loss,
    )

    ema = None
    if ema_decay:
        from recomb2027.latent.models.model_utils import ModelEMA

        ema = ModelEMA(model, decay=float(ema_decay), trainable_only=False, warmup=True)
        logger.info("EMA enabled (decay=%g); validation/checkpoints use EMA weights", float(ema_decay))
    input_noise_std = float(input_noise_std or 0.0)

    # --- Training Loop ---
    for epoch in range(start_epoch + 1, epochs + 1):
        if epoch_callback is not None:
            epoch_callback(model, epoch, loss_kwargs)
            if getattr(model, "_curriculum_reset_early_stop", False):
                early_stop_counter = 0
                # Mean-only best scores are not comparable to joint scores once
                # pca2d enters the selection metric; allow a fresh best.
                best_val_loss = float("inf")
                best_epoch = -1
                selection_scale_state.clear()
                model._curriculum_reset_early_stop = False
                # Mean-only plateau often trips ReduceLROnPlateau; restore base LR
                # so the pca2d phase does not start at a crushed learning rate.
                for pg, base_lr in zip(optimizer.param_groups, initial_lrs):
                    pg["lr"] = base_lr
                if scheduler is not None:
                    if hasattr(scheduler, "num_bad_epochs"):
                        scheduler.num_bad_epochs = 0
                    if hasattr(scheduler, "cooldown_counter"):
                        scheduler.cooldown_counter = 0
                    if hasattr(scheduler, "best"):
                        scheduler.best = float("inf") if getattr(
                            scheduler, "mode", "min"
                        ) == "min" else float("-inf")
                logger.info(
                    "Reset early-stop/best/scheduler at epoch %d (pca2d unfreeze); lr=%s",
                    epoch,
                    [pg["lr"] for pg in optimizer.param_groups],
                )

        if hasattr(train_dataset, "advance_epoch_subsample"):
            train_dataset.advance_epoch_subsample(epoch)

        model.train()
        total_train_loss = 0.0
        total_train_mse_loss = 0.0
        finite_batches = 0
        nonfinite_batches = 0
        max_nonfinite_batches = 20

        beta = min(epoch / kl_warmup_epochs, 1.0) * kl_beta_max if kl_warmup_epochs > 0 else kl_beta_max
        epoch_diagnostics = {}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch", leave=False)

        for batch in pbar:
            jepa_aux = None
            if len(batch) == 3:
                xb, yb, jepa_aux = batch
            else:
                xb, yb = batch
            xb, yb = xb.to(device).float(), _move_yb_to_device(yb, device)
            if input_noise_std > 0.0:
                xb = xb + input_noise_std * torch.randn_like(xb)

            optimizer.zero_grad(set_to_none=True)

            loss_kw = dict(loss_kwargs or {})
            if jepa_aux is not None:
                loss_kw["jepa_aux"] = jepa_aux

            # --- Forward + Backward with AMP ---
            with torch.amp.autocast(
                "cuda",
                enabled=amp_enabled,
                dtype=autocast_dtype,
            ):
                loss_out = model_loss_func(
                    model,
                    xb, yb,
                    alpha=0,
                    only_last=not use_all_step_loss,
                    beta=beta,
                    free_bits=kl_free_bits,
                    recon_loss_type=recon_loss_type,
                    log_mse_eps=log_mse_eps,
                    **loss_kw,
                )

            if len(loss_out) == 3:
                model_loss, mse_loss, batch_diag = loss_out
            else:
                model_loss, mse_loss = loss_out
                batch_diag = None

            if not torch.isfinite(model_loss):
                nonfinite_batches += 1
                logger.warning(
                    "Non-finite train loss at epoch %d (batch skip %d); "
                    "skipping optimizer step",
                    epoch,
                    nonfinite_batches,
                )
                optimizer.zero_grad(set_to_none=True)
                if nonfinite_batches >= max_nonfinite_batches:
                    raise RuntimeError(
                        f"Too many non-finite losses ({nonfinite_batches}) at epoch "
                        f"{epoch}; aborting to avoid poisoning weights"
                    )
                continue

            scaler.scale(model_loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=clip_grad_norm)

            # Skip step if grads are non-finite after unscale.
            grads_ok = True
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    grads_ok = False
                    break
            if not grads_ok:
                nonfinite_batches += 1
                logger.warning(
                    "Non-finite gradients at epoch %d; skipping step (%d)",
                    epoch,
                    nonfinite_batches,
                )
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                if nonfinite_batches >= max_nonfinite_batches:
                    raise RuntimeError(
                        f"Too many non-finite grad steps ({nonfinite_batches}) "
                        f"at epoch {epoch}"
                    )
                continue

            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)

            total_train_loss += model_loss.item()
            total_train_mse_loss += mse_loss.item()
            finite_batches += 1
            if batch_diag is not None:
                for k, v in batch_diag.items():
                    if isinstance(v, float) and not (v == v):  # NaN check
                        continue
                    epoch_diagnostics[k] = epoch_diagnostics.get(k, 0.0) + v

            pbar.set_postfix(loss=total_train_loss / (finite_batches + 1e-9))

        denom_batches = max(finite_batches, 1)
        avg_train_loss = total_train_loss / denom_batches
        train_model_losses.append(avg_train_loss)

        avg_train_mse_loss = total_train_mse_loss / denom_batches
        train_mse_losses.append(avg_train_mse_loss)

        # --- Validation ---
        if ema is not None:
            ema.apply(model)
        val_n_samples = kl_annealing_dict.get("val_n_samples", 1) if kl_annealing_dict else 1
        val_metrics = evaluate_mse_on_validation(
            model, val_loader, device, n_samples=val_n_samples)
        if not isinstance(val_metrics, dict):
            val_metrics = {PRIMARY_TARGET: float(val_metrics)}
        if selection_metric == "raw_jump_ratio":
            from recomb2027.latent.training.latent_delta_loss import (
                evaluate_raw_jump_metrics,
            )

            raw_jump_cfg = dict((early_stopping_dict or {}).get("raw_jump") or {})
            max_batches = raw_jump_cfg.get("max_batches")
            if max_batches is not None:
                max_batches = int(max_batches)
            val_metrics.update(
                evaluate_raw_jump_metrics(
                    model,
                    val_loader,
                    device,
                    mean_weight=float(raw_jump_cfg.get("mean_weight", 1.0)),
                    pca2d_weight=float(raw_jump_cfg.get("pca2d_weight", 0.75)),
                    max_batches=max_batches,
                )
            )
        # Keep logging primary Val MSE separately from the selection score.
        val_mse_mean = float(val_metrics[PRIMARY_TARGET])
        site_m = {}
        if site_metric_cfg:
            from recomb2027.latent.training.site_metrics import evaluate_site_metrics

            site_m = evaluate_site_metrics(model, val_loader, device, **dict(site_metric_cfg))
        avg_val_loss, sel_label = resolve_val_selection_score(
            {**val_metrics, **site_m},
            early_stopping_dict=early_stopping_dict,
            loss_kwargs=loss_kwargs,
            scale_state=selection_scale_state,
        )
        last_sel_label = sel_label
        val_mse_losses.append(avg_val_loss)

        if scheduler is not None and epoch >= reduce_lr_min_epoch:
            scheduler.step(avg_val_loss)

        # --- Average diagnostics over finite batches ---
        n_batches = max(finite_batches, 1)
        avg_diag = {k: v / n_batches for k, v in epoch_diagnostics.items()}
        for tname, v in val_metrics.items():
            if tname == PRIMARY_TARGET:
                continue
            avg_diag[f"val_mse_{tname}"] = float(v)
        if sel_label != PRIMARY_TARGET:
            avg_diag["val_select"] = float(avg_val_loss)
        for _k, _v in site_m.items():
            avg_diag[f"val_{_k}"] = float(_v)
        if "raw_jump_ratio" in val_metrics:
            avg_diag["raw_jump_ratio"] = float(val_metrics["raw_jump_ratio"])
            if "raw_jump_pred_median" in val_metrics:
                avg_diag["raw_jump_pred_median"] = float(
                    val_metrics["raw_jump_pred_median"]
                )
                avg_diag["raw_jump_true_median"] = float(
                    val_metrics["raw_jump_true_median"]
                )

        # Per-target histories: val from eval; train from mse_* diagnostics.
        # Pad missing train targets (e.g. mean-only curriculum) with nan so
        # epoch indices stay aligned with train_mse_losses.
        for tname, v in val_metrics.items():
            if tname == "mean" and PRIMARY_TARGET in val_metrics and tname != PRIMARY_TARGET:
                continue
            val_mse_per_target.setdefault(tname, []).append(float(v))
        train_mse_per_target.setdefault(PRIMARY_TARGET, []).append(avg_train_mse_loss)
        for tname in val_metrics:
            if tname in ("mean", PRIMARY_TARGET):
                continue
            train_v = avg_diag.get(f"mse_{tname}")
            train_mse_per_target.setdefault(tname, []).append(
                float(train_v) if train_v is not None else float("nan")
            )

        # --- TensorBoard Logging ---
        if writer is not None:
            writer.add_scalar("Loss/train", avg_train_loss, epoch)
            writer.add_scalar("Loss/train_mse", avg_train_mse_loss, epoch)
            writer.add_scalar("Loss/val", val_mse_mean, epoch)
            writer.add_scalar("Loss/val_select", avg_val_loss, epoch)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)
            if kl_annealing_dict is not None:
                writer.add_scalar("KL/beta", beta, epoch)
            for k, v in avg_diag.items():
                writer.add_scalar(f"Diagnostics/{k}", v, epoch)
            for tname, v in val_metrics.items():
                writer.add_scalar(f"Loss/val_{tname}", float(v), epoch)

        # --- Print progress ---
        diag_str = " | ".join(f"{k}: {v:.4f}" for k, v in avg_diag.items())
        logger.info(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Train MSE: {avg_train_mse_loss:.6f} | "
            f"Val MSE: {val_mse_mean:.6f}"
            + (f" | val_select({sel_label}): {avg_val_loss:.6f}" if sel_label != PRIMARY_TARGET else "")
            + (f" | {diag_str}" if diag_str else "")
        )

        # --- Best model tracking ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            save_model_w_hyperparams(
                os.path.join(save_dir, f"{model.__class__.__name__}_best_model.pt"),
                model,
                optimizer,
                epoch,
                best_val_loss,
                best_epoch=best_epoch,
                early_stop_counter=early_stop_counter,
            )
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # --- Periodic checkpoint + coverage evaluation ---
        if epoch % 10 == 0:
            save_model_w_hyperparams(
                os.path.join(save_dir, f"{model.__class__.__name__}_model_epoch-{epoch}.pt"),
                model,
                optimizer,
                epoch,
                best_val_loss,
                best_epoch=best_epoch,
                early_stop_counter=early_stop_counter,
            )
            cov = evaluate_coverage(model, val_loader, device, n_samples=20)
            if cov:
                cov_str = " | ".join(f"{k}: {v:.4f}" for k, v in cov.items())
                logger.info(f"  [Coverage@{epoch}] {cov_str}")
                if writer is not None:
                    for k, v in cov.items():
                        writer.add_scalar(f"Coverage/{k}", v, epoch)

        if ema is not None:
            ema.restore(model)

        # --- Early stopping ---
        if early_stop_enabled and early_stop_counter >= early_stop_patience:
            logger.info(
                f"Early stopping at epoch {epoch} "
                f"(best at epoch {best_epoch}, val_select {best_val_loss:.6f})"
            )
            break

    # --- Final coverage evaluation ---
    final_cov = evaluate_coverage(model, val_loader, device, n_samples=20)
    if final_cov:
        cov_str = " | ".join(f"{k}: {v:.4f}" for k, v in final_cov.items())
        logger.info(f"  [Final Coverage] {cov_str}")

    # --- Wrap up ---
    if writer is not None:
        if final_cov:
            for k, v in final_cov.items():
                writer.add_scalar(f"Coverage/{k}", v, epochs)
        writer.close()

    logger.info(f"\nTraining complete. Best model at epoch {best_epoch} with val loss {best_val_loss:.6f}.")

    results = {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "train_mse_losses": train_mse_losses,
        "val_mse_losses": val_mse_losses,
        "train_mse_per_target": train_mse_per_target,
        "val_mse_per_target": val_mse_per_target,
        "selection_label": last_sel_label,
    }

    # --- Move model back to CPU and clear GPU memory ---
    model.to("cpu")
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    logger.info("Model moved to CPU and GPU cache cleared.")

    # Plot curves and output into the folder
    plot_dir = os.path.join(save_dir, f"{model.__class__.__name__}_model")
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    plot_training_loss(
        train_mse_losses,
        val_mse_losses,
        plot_dir,
        train_per_target=train_mse_per_target,
        val_per_target=val_mse_per_target,
        val_select_history=val_mse_losses if last_sel_label != PRIMARY_TARGET else None,
        select_label=f"val_select ({last_sel_label})",
    )
    logger.info(f"Train/Validation plots saved to '{plot_dir}'")

    return results

