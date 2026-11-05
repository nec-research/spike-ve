#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     model_utils.py
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
"""Model factory, loss functions, EMA and checkpoint helpers shared by the trainers.

Contains ``get_predictor`` (builds the anchor model), the anchor/GPT losses (``gpt_losses``), ``ModelEMA`` and
the checkpoint / seeding utilities."""
from __future__ import annotations

import torch
import torch.nn.functional as F
import torch.nn as nn
import inspect
import random
import numpy as np
import logging

from typing import Optional, Sequence, Union, Dict, Type
from recomb2027.core.constants import PRIMARY_TARGET
from recomb2027.latent.models.anchor_model import GPTModel, GPTModelPca2dSite


# ============================================================================
# (merged from losses.py)
# ============================================================================

# ------ Loss functions ---------

def _gpt_is_multitarget(model) -> bool:
    targets = getattr(model, "prediction_targets", None)
    return targets is not None and len(targets) > 1


def _gpt_target_names(model) -> tuple[str, ...]:
    return tuple(getattr(model, "prediction_targets", (PRIMARY_TARGET,)))


def consistency_curriculum_scale(
    mean_pca2d_v4: dict, weights: dict[str, float]
) -> float:
    """Multiplier in [0, 1] applied to ``mean_pca2d_v4.consistency_weight``.

    ``consistency_schedule: constant`` (default) keeps the historical behaviour.
    ``curriculum`` follows the PCA curriculum: scale = w_pca_now / w_pca_ref, so the
    cross-view term is off while the PCA head is frozen / untrained (mean-only epochs)
    and ramps up together with the PCA loss weight. ``pca2d_weight_ref`` is the final
    PCA weight (set by the trainer from the base ``target_loss_weights``).
    """
    sched = str(mean_pca2d_v4.get("consistency_schedule", "constant") or "constant").lower()
    if sched == "constant":
        return 1.0
    if sched != "curriculum":
        raise ValueError(
            f"consistency_schedule must be constant|curriculum; got {sched!r}"
        )
    from recomb2027.latent.data_series.pca2d_features import PCA2D_S1_TARGET

    name = str(mean_pca2d_v4.get("pca2d_target_name", PCA2D_S1_TARGET))
    ref = float(mean_pca2d_v4.get("pca2d_weight_ref") or 0.0)
    cur = float(weights.get(name, ref))
    if ref <= 0.0:
        return 1.0 if cur > 0.0 else 0.0
    return float(min(max(cur / ref, 0.0), 1.0))


def _gpt_target_weights(model, override: Optional[dict]) -> dict[str, float]:
    if override:
        return {str(k): float(v) for k, v in override.items()}
    raw = getattr(model, "hparams", {}).get("target_loss_weights")
    if raw:
        return {str(k): float(v) for k, v in raw.items()}
    names = _gpt_target_names(model)
    return {n: 1.0 for n in names}


def _slice_gpt_batch(
    pred_dict: dict[str, torch.Tensor],
    yb: Union[torch.Tensor, dict[str, torch.Tensor]],
    target_names: Sequence[str],
    *,
    only_last: bool,
) -> tuple[dict[str, torch.Tensor], Union[torch.Tensor, dict[str, torch.Tensor]]]:
    if only_last:
        pred_dict = {k: v[:, -1:, :] for k, v in pred_dict.items()}
        if isinstance(yb, dict):
            yb = {k: v[:, -1:, :] for k, v in yb.items()}
        elif yb.dim() == 4:
            yb = yb[:, -1:, :, :]
        else:
            yb = yb[:, -1:, :]
    return pred_dict, yb


def _yb_for_target(
    yb: Union[torch.Tensor, dict[str, torch.Tensor]],
    target_names: Sequence[str],
    tname: str,
) -> torch.Tensor:
    if isinstance(yb, dict):
        if tname not in yb:
            raise KeyError(
                f"target {tname!r} missing from yb dict; keys={list(yb.keys())}"
            )
        return yb[tname]
    if yb.dim() == 4:
        idx = target_names.index(tname)
        return yb[:, :, idx, :]
    if len(target_names) == 1:
        return yb
    raise ValueError(
        f"Expected yb [B,T,K,D] for multi-target loss; got shape {tuple(yb.shape)}"
    )


def _gather_at_mask_indices(
    values: torch.Tensor,
    mask_idx: torch.Tensor,
) -> torch.Tensor:
    """Index ``values[B,T,D]`` at ``mask_idx[B,M]`` (``-1`` → zero row)."""
    if values.dim() != 3:
        raise ValueError(f"values must be [B,T,D]; got {tuple(values.shape)}")
    out = values.new_zeros(mask_idx.size(0), mask_idx.size(1), values.size(-1))
    valid = mask_idx >= 0
    for b in range(values.size(0)):
        idx = mask_idx[b, valid[b]].long()
        if idx.numel() > 0:
            out[b, valid[b]] = values[b, idx]
    return out


def sample_jepa_mask_indices(
    batch_size: int,
    seq_len: int,
    *,
    mask_prob: float = 0.25,
    n_mask_per_sample: int = 1,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Sample masked input positions ``[B, M]`` with ``-1`` padding."""
    device = device or torch.device("cpu")
    n_mask = max(1, int(n_mask_per_sample))
    max_candidates = max(1, seq_len - 1)
    n_pick = min(n_mask, max_candidates)
    out = torch.full((batch_size, n_mask), -1, dtype=torch.long, device=device)
    for b in range(batch_size):
        if torch.rand(1, device=device).item() > mask_prob:
            continue
        perm = torch.randperm(max_candidates, device=device)[:n_pick]
        out[b, : perm.numel()] = perm
    return out


def gpt_losses(
        model,
        xb, yb,
        alpha=0.4,
        only_last=False,
        target_loss_weights=None,
        jepa_aux=None,
        **kwargs
):
    """Autoregressive loss for GPT (single- or multi-target diff prediction).

    Optional JEPA-style auxiliary loss masks prefix steps and predicts absolute
    encodings via ``model.forward_jepa_aux`` (requires ``jepa_mask_token``).

    Numerics (AMP-safe):
      - Targets with weight ``0`` are skipped (avoids ``0 * inf = nan``).
      - MSE is accumulated in float32.
      - pca2d decode is skipped when all pca2d target weights are 0.
    """
    from recomb2027.latent.data_series.pca2d_features import is_pca2d_target

    target_names = _gpt_target_names(model)
    weights = _gpt_target_weights(model, target_loss_weights)

    mean_pca2d_v4 = kwargs.get("mean_pca2d_v4")
    cons_scale = 1.0
    if mean_pca2d_v4 and mean_pca2d_v4.get("enabled"):
        cons_scale = consistency_curriculum_scale(mean_pca2d_v4, weights)
    w_cons_eff = (
        float(mean_pca2d_v4.get("consistency_weight", 0.2)) * cons_scale
        if mean_pca2d_v4
        else 0.0
    )
    need_consistency_expand = bool(
        mean_pca2d_v4
        and mean_pca2d_v4.get("enabled")
        and float(mean_pca2d_v4.get("consistency_weight", 0.0)) > 0.0
        and w_cons_eff > 0.0
    )
    need_expand_pca2d = need_consistency_expand or any(
        is_pca2d_target(t) and float(weights.get(t, 1.0)) != 0.0
        for t in target_names
    )

    if _gpt_is_multitarget(model):
        pred_dict = model(xb, return_dict=True, expand_pca2d=need_expand_pca2d)
    else:
        pred_dict = {target_names[0]: model(xb)}

    pred_dict, yb = _slice_gpt_batch(pred_dict, yb, target_names, only_last=only_last)

    # Primary last-step monitor in fp32 (AMP-safe).
    mse_last = F.mse_loss(
        pred_dict[PRIMARY_TARGET][:, -1, :].float(),
        _yb_for_target(yb, target_names, PRIMARY_TARGET)[:, -1, :].float(),
    )

    recon = pred_dict[PRIMARY_TARGET].new_zeros((), dtype=torch.float32)
    per_target: dict[str, float] = {}
    cosine_sum = pred_dict[PRIMARY_TARGET].new_zeros((), dtype=torch.float32)
    cosine_weight = 0.0

    delta_loss_mode = str(kwargs.get("delta_loss_mode", "scaled") or "scaled").lower()
    delta_loss_eps = float(kwargs.get("delta_loss_eps", 1e-6) or 1e-6)
    if delta_loss_mode not in ("scaled", "relative", "raw"):
        raise ValueError(
            f"delta_loss_mode must be scaled, relative, or raw; got {delta_loss_mode!r}"
        )
    if delta_loss_mode != "scaled":
        from recomb2027.latent.training.latent_delta_loss import (
            compute_delta_loss,
        )

    for tname in target_names:
        w = float(weights.get(tname, 1.0))
        if w == 0.0:
            # Do not evaluate MSE: under AMP, huge errors → inf and 0*inf → nan.
            continue
        if tname not in pred_dict:
            continue
        p = pred_dict[tname].float()
        if isinstance(yb, dict) and tname not in yb:
            raise KeyError(
                f"target {tname!r} missing from yb dict; keys={list(yb.keys())}. "
                "Ensure dataset.prediction_targets includes all model "
                "prediction_targets (train script must set this on train/val ds)."
            )
        y_t = _yb_for_target(yb, target_names, tname).float()
        if delta_loss_mode == "scaled":
            lt = F.mse_loss(p, y_t)
        else:
            lt = compute_delta_loss(
                p,
                y_t,
                mode=delta_loss_mode,
                eps=delta_loss_eps,
                xb=xb,
                model=model,
                tname=tname,
            )
        recon = recon + w * lt
        per_target[tname] = float(lt.detach().item())

        if alpha > 0:
            if only_last:
                p_flat = p.reshape(-1, p.size(-1))
                y_flat = y_t.reshape(-1, y_t.size(-1))
            else:
                b, t, d = p.shape
                p_flat = p.reshape(b * t, d)
                y_flat = y_t.reshape(b * t, d)
            cos = 1 - F.cosine_similarity(p_flat, y_flat, dim=1).mean()
            cosine_sum = cosine_sum + w * cos
            cosine_weight += w

    model_loss = recon
    if alpha > 0 and cosine_weight > 0:
        model_loss = model_loss + alpha * (cosine_sum / cosine_weight)

    jepa_loss_val = 0.0
    if (
        jepa_aux
        and jepa_aux.get("enabled")
        and model.training
        and getattr(model, "jepa_heads", None) is not None
    ):
        abs_targets = jepa_aux.get("absolute_targets")
        if abs_targets is None:
            raise ValueError("jepa_aux.enabled requires absolute_targets in batch kwargs")
        mask_idx = jepa_aux.get("mask_indices")
        if mask_idx is None:
            mask_idx = sample_jepa_mask_indices(
                xb.size(0),
                xb.size(1),
                mask_prob=float(jepa_aux.get("mask_prob", 0.25)),
                n_mask_per_sample=int(jepa_aux.get("n_mask_per_sample", 1)),
                device=xb.device,
            )
        predict_targets = tuple(
            jepa_aux.get("predict_targets") or getattr(model, "jepa_predict_targets", (PRIMARY_TARGET,))
        )
        jepa_preds = model.forward_jepa_aux(
            xb, mask_idx.to(xb.device), predict_targets=predict_targets
        )
        jepa_sum = xb.new_zeros((), dtype=torch.float32)
        for tname in predict_targets:
            if tname not in jepa_preds:
                raise ValueError(
                    f"JEPA target {tname!r} missing from model.forward_jepa_aux output; "
                    f"got {list(jepa_preds.keys())}"
                )
            if tname not in abs_targets:
                raise ValueError(
                    f"JEPA target {tname!r} missing from batch absolute_targets; "
                    f"got {list(abs_targets.keys())}. "
                    "Ensure dataset.jepa_absolute_stats includes all predict_targets."
                )
            pred_t = jepa_preds[tname].float()
            tgt_full = abs_targets[tname].to(xb.device).float()
            tgt_t = _gather_at_mask_indices(tgt_full, mask_idx.to(xb.device))
            valid = (mask_idx.to(xb.device) >= 0).unsqueeze(-1).float()
            sq = (pred_t - tgt_t).pow(2) * valid
            denom = valid.sum().clamp(min=1.0) * pred_t.size(-1)
            jepa_sum = jepa_sum + sq.sum() / denom
        w_aux = float(jepa_aux.get("loss_weight", 0.2))
        model_loss = model_loss + w_aux * jepa_sum
        jepa_loss_val = float(jepa_sum.detach().item())

    consistency_loss_val = 0.0
    if need_consistency_expand:
        from recomb2027.latent.data_series.pca2d_features import PCA2D_S1_TARGET
        from recomb2027.latent.models.consistency_loss import (
            gpt_mean_pca2d_consistency_loss,
        )

        readout = mean_pca2d_v4.get("readout")
        if readout is None:
            raise ValueError("mean_pca2d_v4.enabled requires readout module in loss kwargs")
        pca_name = str(mean_pca2d_v4.get("pca2d_target_name", PCA2D_S1_TARGET))
        cons = gpt_mean_pca2d_consistency_loss(
            model,
            xb,
            pred_dict,
            readout=readout,
            pca2d_target_name=pca_name,
            only_last=only_last,
            stop_grad_pca2d=bool(mean_pca2d_v4.get("stop_grad_pca2d", True)),
        )
        model_loss = model_loss + w_cons_eff * cons
        consistency_loss_val = float(cons.detach().item())

    diagnostics = {
        "recon": float(recon.detach().item()),
        "jepa": jepa_loss_val,
        "consistency": consistency_loss_val,
        "consistency_scale": float(cons_scale),
        **{f"mse_{k}": v for k, v in per_target.items()},
    }
    return model_loss, mse_last, diagnostics


# ============================================================================
# (merged from ema.py)
# ============================================================================

class ModelEMA:
    """Maintain EMA shadow weights for trainable parameters."""

    def __init__(
        self,
        model: nn.Module,
        *,
        decay: float = 0.999,
        trainable_only: bool = True,
        warmup: bool = False,
    ):
        self.warmup = bool(warmup)
        self.num_updates = 0
        self.decay = float(decay)
        self.shadow: dict[str, torch.Tensor] = {}
        self._backup: Optional[dict[str, torch.Tensor]] = None
        for name, param in model.named_parameters():
            if trainable_only and not param.requires_grad:
                continue
            self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        n = self.num_updates
        d = min(self.decay, (1.0 + n) / (10.0 + n)) if self.warmup else self.decay
        for name, param in model.named_parameters():
            if name not in self.shadow:
                continue
            self.shadow[name].mul_(d).add_(param.data, alpha=1.0 - d)

    def apply(self, model: nn.Module) -> None:
        """Swap live weights with EMA shadow (stores backup for restore)."""
        self._backup = {}
        for name, param in model.named_parameters():
            if name not in self.shadow:
                continue
            self._backup[name] = param.data.clone()
            param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        if self._backup is None:
            return
        for name, param in model.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup = None

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.clone() for k, v in state.items()}


# ============================================================================
# (merged from model_utils.py)
# ============================================================================

logger = logging.getLogger(__name__)

# Checkpoint ``model_cls_name`` → import path (for eval / load helpers).
_PREDICTOR_CLS_MODULES: Dict[str, str] = {
    "GPTModel": "recomb2027.latent.models.anchor_model",
    "GPTModelPca2dSite": "recomb2027.latent.models.anchor_model",
}


def filter_model_ctor_kwargs(model_cls: Type, hp: dict) -> dict:
    """Drop hparams keys that are not ``__init__`` parameters.

    Walks the MRO so subclasses that only declare ``*args, **kwargs`` plus a few
    extras (e.g. :class:`GPTModelPca2dSite`) still accept parent GPT kwargs.
    """
    allowed: set[str] = set()
    for cls in model_cls.__mro__:
        if cls is object:
            break
        init = getattr(cls, "__init__", None)
        if init is None:
            continue
        for name, param in inspect.signature(init).parameters.items():
            if name == "self":
                continue
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            allowed.add(name)
    return {k: v for k, v in hp.items() if k in allowed}


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # deterministic flags (optional, may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# def get_predictor(name: str, **cfg):
#     spec = REGISTRY[name]
#
#     ctor = spec["ctor"]
#     allowed = spec.get("hyperparam_keys", [])
#
#     ctor_kwargs = {k: v for k, v in cfg.items() if k in allowed}
#
#     model = ctor(**ctor_kwargs)
#     default_loss = spec["default_loss"]
#     return model, default_loss


def get_predictor(cfg: Dict):
    """
    Factory for predictor models.
    """
    name = cfg.get("name", "gpt_pca2d_site")

    if name == "gpt":
        model_cfg = {
            "input_dim": cfg.get("input_dim"),
            "embed_dim": cfg.get("embed_dim"),
            "ff_dim": cfg.get("ff_dim"),
            "output_dim": cfg.get("output_dim"),
            "num_heads": cfg.get("num_heads"),
            "num_gpt_blocks": cfg.get("num_gpt_blocks"),
            "dropout": cfg.get("dropout", 0.1),
            "max_seq_len": cfg.get("max_seq_len", 32),
            "extra_stat_projection": cfg.get("extra_stat_projection", False),
            "extra_stat_projection_layernorm": cfg.get(
                "extra_stat_projection_layernorm", False
            ),
            "position_stat_slices": cfg.get("position_stat_slices"),
            "extra_stat_dual_branch": cfg.get("extra_stat_dual_branch", False),
            "extra_stat_dual_branch_layernorm": cfg.get(
                "extra_stat_dual_branch_layernorm", False
            ),
            "dual_branch_layout": cfg.get("dual_branch_layout"),
            "block_branch": cfg.get("block_branch", False),
            "triple_branch_layout": cfg.get("triple_branch_layout"),
            "pca2d_branch": cfg.get("pca2d_branch", False),
            "mean_pca2d_layout": cfg.get("mean_pca2d_layout"),
            "mean_pca2d_layernorm": cfg.get("mean_pca2d_layernorm", True),
            "pca2d_bottleneck_dim": cfg.get("pca2d_bottleneck_dim"),
            "pca2d_flat_dim": cfg.get("pca2d_flat_dim"),
            "pca2d_bottleneck_hidden": cfg.get("pca2d_bottleneck_hidden"),
            "prediction_targets": cfg.get("prediction_targets"),
            "target_output_dims": cfg.get("target_output_dims"),
            "jepa_mask_token": cfg.get("jepa_mask_token", False),
            "jepa_predict_targets": cfg.get("jepa_predict_targets"),
        }
        model = GPTModel(**model_cfg)
        return model, gpt_losses

    if name == "gpt_pca2d_site":
        model_cfg = {
            "input_dim": cfg.get("input_dim"),
            "embed_dim": cfg.get("embed_dim"),
            "ff_dim": cfg.get("ff_dim"),
            "output_dim": cfg.get("output_dim"),
            "num_heads": cfg.get("num_heads"),
            "num_gpt_blocks": cfg.get("num_gpt_blocks"),
            "dropout": cfg.get("dropout", 0.1),
            "max_seq_len": cfg.get("max_seq_len", 32),
            "pca2d_branch": cfg.get("pca2d_branch", True),
            "mean_pca2d_layout": cfg.get("mean_pca2d_layout"),
            "mean_pca2d_layernorm": cfg.get("mean_pca2d_layernorm", True),
            "pca2d_bottleneck_dim": cfg.get("pca2d_bottleneck_dim"),
            "pca2d_flat_dim": cfg.get("pca2d_flat_dim"),
            "pca2d_bottleneck_hidden": cfg.get("pca2d_bottleneck_hidden"),
            "pca2d_seq_len": cfg.get("pca2d_seq_len"),
            "pca2d_k": cfg.get("pca2d_k"),
            "pca2d_site_hidden": cfg.get("pca2d_site_hidden", 64),
            "pca2d_site_dim": cfg.get("pca2d_site_dim", 32),
            "pca2d_site_mixer_layers": cfg.get("pca2d_site_mixer_layers", 1),
            "pca2d_site_pool": cfg.get("pca2d_site_pool", "mean"),
            "pca2d_decode_rank": cfg.get("pca2d_decode_rank", 0),
            "pca2d_mean_cond_attn": cfg.get("pca2d_mean_cond_attn", False),
            "pca2d_gated_residual": cfg.get("pca2d_gated_residual", False),
            "pca2d_site_resolved_gate": cfg.get("pca2d_site_resolved_gate", False),
            "pca2d_residual": cfg.get("pca2d_residual", True),
            "prediction_targets": cfg.get("prediction_targets"),
            "target_output_dims": cfg.get("target_output_dims"),
            "jepa_mask_token": cfg.get("jepa_mask_token", False),
            "jepa_predict_targets": cfg.get("jepa_predict_targets"),
            "extra_stat_projection": False,
            "extra_stat_dual_branch": False,
            "block_branch": False,
        }
        model = GPTModelPca2dSite(**model_cfg)
        return model, gpt_losses

    raise ValueError(f"Unknown predictor '{name}'")


def save_model_w_hyperparams(
    checkpoint_path,
    model,
    optimizer,
    epoch,
    loss,
    *,
    best_epoch=None,
    early_stop_counter=0,
):
    torch.save({
        "model_cls_name": model.__class__.__name__,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "best_epoch": int(best_epoch if best_epoch is not None else epoch),
        "early_stop_counter": int(early_stop_counter),
        "hyperparameters": model.hparams,
    }, checkpoint_path)
    logger.info(f"Model saved to: '{checkpoint_path}'.")


def load_training_resume_state(
    checkpoint_path,
    model,
    optimizer,
    *,
    device=None,
) -> dict:
    """Restore model/optimizer weights and training counters from a checkpoint."""
    map_loc = device if device is not None else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=map_loc, weights_only=False)

    ckpt_cls = checkpoint.get("model_cls_name")
    if ckpt_cls is not None and ckpt_cls != model.__class__.__name__:
        raise ValueError(
            f"Checkpoint model class {ckpt_cls!r} != {model.__class__.__name__!r}"
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if device is not None:
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

    start_epoch = int(checkpoint.get("epoch", 0))
    return {
        "start_epoch": start_epoch,
        "best_val_loss": float(checkpoint.get("loss", float("inf"))),
        "best_epoch": int(checkpoint.get("best_epoch", start_epoch)),
        "early_stop_counter": int(checkpoint.get("early_stop_counter", 0)),
    }
