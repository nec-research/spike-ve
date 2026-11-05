#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     latent_delta_loss.py
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
"""Scaled / relative / raw delta losses and raw jump validation metrics."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from recomb2027.core.constants import PRIMARY_TARGET
from recomb2027.latent.data_series.pca2d_features import is_pca2d_target
from recomb2027.latent.predictors.infer_residual import leaf_dataset

MEAN_W = 1.0
PCA2D_W = 0.75


def register_latent_scaler_buffers(model, dataset) -> None:
    """Attach sklearn ``StandardScaler`` stats as frozen buffers on ``model``."""
    leaf = leaf_dataset(dataset)
    pca_scaler = getattr(leaf, "pca2d_scaler", None)
    if pca_scaler is not None:
        scale = torch.as_tensor(pca_scaler.scale_, dtype=torch.float32)
        mean = torch.as_tensor(pca_scaler.mean_, dtype=torch.float32)
        model.register_buffer("pca2d_inv_scale", scale, persistent=False)
        model.register_buffer("pca2d_inv_mean", mean, persistent=False)
    mean_scaler = getattr(leaf, "scaler", None)
    if mean_scaler is not None:
        scale = torch.as_tensor(mean_scaler.scale_, dtype=torch.float32)
        mean = torch.as_tensor(mean_scaler.mean_, dtype=torch.float32)
        model.register_buffer("mean_inv_scale", scale, persistent=False)
        model.register_buffer("mean_inv_mean", mean, persistent=False)


def _inverse_standard(
    x: torch.Tensor,
    scale: Optional[torch.Tensor],
    mean: Optional[torch.Tensor],
) -> torch.Tensor:
    if scale is None or mean is None:
        return x
    return x * scale + mean


def extract_abs_mean_from_x(
    xb: torch.Tensor,
    model,
    mean_dim: int,
) -> torch.Tensor:
    """Last-step absolute mean latent from dual-branch input ``[B,T,Din]``."""
    layout = getattr(model, "hparams", {}).get("mean_pca2d_layout") or {}
    sl = layout.get("primary_slice")
    if sl:
        s, e = int(sl[0]), int(sl[1])
        chunk = xb[..., s:e]
    else:
        chunk = xb[..., :mean_dim]
    if chunk.size(-1) == mean_dim:
        return chunk
    if chunk.size(-1) == 2 * mean_dim:
        return chunk[..., :mean_dim]
    raise ValueError(
        f"primary slice width {chunk.size(-1)} incompatible with mean_dim={mean_dim}"
    )


def _align_base_to_pred(
    base: torch.Tensor,
    pred: torch.Tensor,
) -> torch.Tensor:
    """Match ``base`` time dimension to ``pred`` (last step when ``only_last``)."""
    if base.shape[1] == pred.shape[1]:
        return base
    return base[:, -pred.shape[1] :, :]


def compute_delta_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    mode: str = "scaled",
    eps: float = 1e-6,
    xb: Optional[torch.Tensor] = None,
    model=None,
    tname: str = PRIMARY_TARGET,
) -> torch.Tensor:
    """Per-target delta loss in scaled, relative, or raw space."""
    mode = str(mode or "scaled").lower()
    p = pred.float()
    y_t = target.float()

    if mode == "scaled":
        return F.mse_loss(p, y_t)

    if mode == "relative":
        p_flat = p.reshape(-1, p.size(-1))
        y_flat = y_t.reshape(-1, y_t.size(-1))
        denom = y_flat.pow(2).mean(dim=-1, keepdim=True).clamp_min(float(eps))
        return ((p_flat - y_flat).pow(2) / denom).mean()

    if mode != "raw":
        raise ValueError(f"delta_loss_mode must be scaled, relative, or raw; got {mode!r}")

    if xb is None or model is None:
        raise ValueError("raw delta_loss_mode requires xb and model")

    if is_pca2d_target(tname):
        if not hasattr(model, "extract_abs_pca2d"):
            return F.mse_loss(p, y_t)
        base = model.extract_abs_pca2d(xb).float()
        base = _align_base_to_pred(base, p)
        scale = getattr(model, "pca2d_inv_scale", None)
        mean = getattr(model, "pca2d_inv_mean", None)
    else:
        mean_dim = int(y_t.shape[-1])
        base = extract_abs_mean_from_x(xb, model, mean_dim).float()
        base = _align_base_to_pred(base, p)
        scale = getattr(model, "mean_inv_scale", None)
        mean = getattr(model, "mean_inv_mean", None)

    if scale is None or mean is None:
        return F.mse_loss(p, y_t)

    pred_abs = base + p
    tgt_abs = base + y_t
    pred_raw = _inverse_standard(pred_abs, scale, mean)
    tgt_raw = _inverse_standard(tgt_abs, scale, mean)
    return F.mse_loss(pred_raw, tgt_raw)


@torch.no_grad()
def evaluate_raw_jump_metrics(
    model,
    val_loader,
    device,
    *,
    mean_weight: float = MEAN_W,
    pca2d_weight: float = PCA2D_W,
    max_batches: Optional[int] = None,
) -> dict[str, float]:
    """Median raw jump ratio and related metrics on a validation loader."""
    from recomb2027.latent.data_series.pca2d_features import PCA2D_S1_TARGET

    if not hasattr(model, "extract_abs_pca2d"):
        return {
            "raw_jump_ratio": float("nan"),
            "raw_jump_pred_median": float("nan"),
            "raw_jump_true_median": float("nan"),
        }

    model.eval()
    target_names = list(getattr(model, "prediction_targets", (PRIMARY_TARGET,)))
    pca_name = next((t for t in target_names if is_pca2d_target(t)), PCA2D_S1_TARGET)

    ratios: list[float] = []
    pred_jumps: list[float] = []
    true_jumps: list[float] = []
    n_batches = 0

    pca_scale = getattr(model, "pca2d_inv_scale", None)
    pca_mean = getattr(model, "pca2d_inv_mean", None)
    mean_scale = getattr(model, "mean_inv_scale", None)
    mean_mean = getattr(model, "mean_inv_mean", None)

    for batch in val_loader:
        if max_batches is not None and n_batches >= max_batches:
            break
        if len(batch) == 3:
            xb, yb, _ = batch
        else:
            xb, yb = batch
        xb = xb.to(device).float()
        if isinstance(yb, dict):
            yb = {k: v.to(device).float() for k, v in yb.items()}
        else:
            yb = yb.to(device).float()

        pred_dict = model(xb, return_dict=True)
        base_pca = model.extract_abs_pca2d(xb)[:, -1, :]
        pred_pca = pred_dict[pca_name][:, -1, :]
        tgt_pca = (
            yb[pca_name][:, -1, :]
            if isinstance(yb, dict)
            else yb[:, -1, :]
        )

        if pca_scale is not None and pca_mean is not None:
            src_pca = _inverse_standard(base_pca, pca_scale, pca_mean)
            pred_raw_pca = _inverse_standard(base_pca + pred_pca, pca_scale, pca_mean)
            tgt_raw_pca = _inverse_standard(base_pca + tgt_pca, pca_scale, pca_mean)
        else:
            src_pca = base_pca
            pred_raw_pca = base_pca + pred_pca
            tgt_raw_pca = base_pca + tgt_pca

        mean_dim = int(pred_dict[PRIMARY_TARGET].shape[-1])
        base_mean = extract_abs_mean_from_x(xb, model, mean_dim)[:, -1, :]
        pred_mean = pred_dict[PRIMARY_TARGET][:, -1, :]
        tgt_mean = (
            yb[PRIMARY_TARGET][:, -1, :]
            if isinstance(yb, dict)
            else yb[:, -1, :]
        )

        if mean_scale is not None and mean_mean is not None:
            src_mean = _inverse_standard(base_mean, mean_scale, mean_mean)
            pred_raw_mean = _inverse_standard(
                base_mean + pred_mean, mean_scale, mean_mean
            )
            tgt_raw_mean = _inverse_standard(
                base_mean + tgt_mean, mean_scale, mean_mean
            )
        else:
            src_mean = base_mean
            pred_raw_mean = base_mean + pred_mean
            tgt_raw_mean = base_mean + tgt_mean

        pred_jump = (
            mean_weight * (pred_raw_mean - src_mean).norm(dim=-1)
            + pca2d_weight * (pred_raw_pca - src_pca).norm(dim=-1)
        )
        true_jump = (
            mean_weight * (tgt_raw_mean - src_mean).norm(dim=-1)
            + pca2d_weight * (tgt_raw_pca - src_pca).norm(dim=-1)
        )
        pred_jumps.extend(pred_jump.detach().cpu().tolist())
        true_jumps.extend(true_jump.detach().cpu().tolist())
        ratios.extend(
            (pred_jump / true_jump.clamp_min(1e-9)).detach().cpu().tolist()
        )
        n_batches += 1

    if not ratios:
        return {
            "raw_jump_ratio": float("nan"),
            "raw_jump_pred_median": float("nan"),
            "raw_jump_true_median": float("nan"),
        }

    ratio_t = torch.tensor(ratios, dtype=torch.float64)
    pred_t = torch.tensor(pred_jumps, dtype=torch.float64)
    true_t = torch.tensor(true_jumps, dtype=torch.float64)
    med_ratio = float(ratio_t.median().item())
    return {
        "raw_jump_ratio": med_ratio,
        "raw_jump_ratio_deviation": float(abs(med_ratio - 1.0)),
        "raw_jump_pred_median": float(pred_t.median().item()),
        "raw_jump_true_median": float(true_t.median().item()),
    }
