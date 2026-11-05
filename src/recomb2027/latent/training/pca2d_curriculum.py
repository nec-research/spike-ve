#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     pca2d_curriculum.py
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
"""pca2d loss-weight curriculum and head freeze helpers."""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Sequence

import torch.nn as nn

from recomb2027.core.constants import PRIMARY_TARGET
from recomb2027.latent.data_series.pca2d_features import PCA2D_S1_TARGET, is_pca2d_target

logger = logging.getLogger(__name__)


def _as_float_map(raw: Optional[Mapping[str, Any]]) -> Dict[str, float]:
    if not raw:
        return {}
    return {str(k): float(v) for k, v in raw.items()}


def resolve_pca2d_weight(
    epoch: int,
    *,
    curriculum: Optional[Mapping[str, Any]],
    base_weights: Mapping[str, float],
    pca2d_name: str = PCA2D_S1_TARGET,
) -> Dict[str, float]:
    """Return target loss weights for ``epoch`` (1-indexed).

    Curriculum keys (all optional)::

        mean_only_epochs: int   # pca2d weight forced to 0
        weight_start: float     # after mean-only (default 0.1)
        weight_end: float       # after ramp (default base weight or 0.25)
        ramp_epochs: int        # linear ramp length; 0 = jump to weight_end
    """
    weights = dict(base_weights)
    if PRIMARY_TARGET not in weights:
        weights[PRIMARY_TARGET] = 1.0
    if pca2d_name not in weights:
        weights[pca2d_name] = float(base_weights.get(pca2d_name, 0.25))

    if not curriculum:
        return weights

    mean_only = int(curriculum.get("mean_only_epochs", 0) or 0)
    w_end = float(
        curriculum.get(
            "weight_end",
            base_weights.get(pca2d_name, weights[pca2d_name]),
        )
    )
    w_start = float(curriculum.get("weight_start", 0.1))
    ramp = int(curriculum.get("ramp_epochs", 0) or 0)

    if epoch <= mean_only:
        weights[pca2d_name] = 0.0
        return weights

    if ramp <= 0:
        weights[pca2d_name] = w_end
        return weights

    # epoch mean_only+1 → w_start; mean_only+ramp → w_end; after → w_end
    t = epoch - mean_only
    if t > ramp:
        weights[pca2d_name] = w_end
    elif ramp == 1:
        # Single ramp epoch at weight_start; next epoch jumps to weight_end.
        weights[pca2d_name] = w_start
    else:
        alpha = (t - 1) / (ramp - 1)
        weights[pca2d_name] = w_start + alpha * (w_end - w_start)
    return weights


def pca2d_head_param_modules(
    model: nn.Module,
    pca2d_name: str = PCA2D_S1_TARGET,
) -> list[nn.Module]:
    """Modules that implement pca2d Δ prediction (head + flat decoder)."""
    mods: list[nn.Module] = []
    lm_heads = getattr(model, "lm_heads", None)
    if lm_heads is not None and hasattr(lm_heads, "heads"):
        heads = lm_heads.heads
        if pca2d_name in heads:
            mods.append(heads[pca2d_name])
    bottleneck = getattr(model, "pca2d_bottleneck", None)
    if bottleneck is not None and hasattr(bottleneck, "decoder"):
        mods.append(bottleneck.decoder)
    return mods


def set_pca2d_head_trainable(
    model: nn.Module,
    trainable: bool,
    *,
    pca2d_name: str = PCA2D_S1_TARGET,
) -> int:
    """Freeze/unfreeze pca2d prediction head (+ bottleneck decoder if present).

    Input-side pca2d encoder stays trainable so mean can still condition on
    pca2d context during the mean-only phase.
    """
    n = 0
    for mod in pca2d_head_param_modules(model, pca2d_name=pca2d_name):
        for p in mod.parameters():
            p.requires_grad = bool(trainable)
            n += p.numel()
    return n


def apply_pca2d_curriculum_epoch(
    model: nn.Module,
    epoch: int,
    *,
    curriculum: Optional[Mapping[str, Any]],
    base_weights: Mapping[str, float],
    loss_kwargs: Optional[MutableMapping[str, Any]],
    pca2d_name: str = PCA2D_S1_TARGET,
) -> Dict[str, float]:
    """Update loss weights and optional head freeze for this epoch.

    Returns the active ``target_loss_weights`` map.
    """
    weights = resolve_pca2d_weight(
        epoch,
        curriculum=curriculum,
        base_weights=base_weights,
        pca2d_name=pca2d_name,
    )
    if loss_kwargs is not None:
        loss_kwargs["target_loss_weights"] = dict(weights)
    if hasattr(model, "hparams") and isinstance(model.hparams, dict):
        model.hparams["target_loss_weights"] = dict(weights)

    if curriculum and bool(curriculum.get("freeze_pca2d_head_during_mean_only", True)):
        mean_only = int(curriculum.get("mean_only_epochs", 0) or 0)
        trainable = epoch > mean_only
        n = set_pca2d_head_trainable(model, trainable, pca2d_name=pca2d_name)
        if epoch == 1 or epoch == mean_only + 1:
            logger.info(
                "pca2d curriculum epoch=%d: weight(%s)=%.4f head_trainable=%s (%d params)",
                epoch,
                pca2d_name,
                weights.get(pca2d_name, 0.0),
                trainable,
                n,
            )
        if epoch == mean_only + 1 and bool(
            curriculum.get("reset_early_stop_on_unfreeze", True)
        ):
            # Signal trainer to clear early-stop counter when pca2d phase begins.
            setattr(model, "_curriculum_reset_early_stop", True)
    elif epoch == 1 or (
        curriculum and epoch == int(curriculum.get("mean_only_epochs", 0) or 0) + 1
    ):
        logger.info(
            "pca2d curriculum epoch=%d: weight(%s)=%.4f",
            epoch,
            pca2d_name,
            weights.get(pca2d_name, 0.0),
        )
        if (
            curriculum
            and epoch == int(curriculum.get("mean_only_epochs", 0) or 0) + 1
            and bool(curriculum.get("reset_early_stop_on_unfreeze", True))
        ):
            setattr(model, "_curriculum_reset_early_stop", True)
    return weights


def make_pca2d_curriculum_callback(
    curriculum: Optional[Mapping[str, Any]],
    base_weights: Mapping[str, float],
    *,
    pca2d_name: str = PCA2D_S1_TARGET,
) -> Optional[Callable[[nn.Module, int, Optional[MutableMapping[str, Any]]], Dict[str, float]]]:
    """Build a trainer epoch hook, or ``None`` if curriculum is disabled."""
    if not curriculum:
        return None
    # Enable if any meaningful key is set
    mean_only = int(curriculum.get("mean_only_epochs", 0) or 0)
    ramp = int(curriculum.get("ramp_epochs", 0) or 0)
    if mean_only <= 0 and ramp <= 0 and curriculum.get("weight_start") is None:
        return None

    base = _as_float_map(base_weights)

    def _cb(model, epoch, loss_kwargs):
        return apply_pca2d_curriculum_epoch(
            model,
            epoch,
            curriculum=curriculum,
            base_weights=base,
            loss_kwargs=loss_kwargs,
            pca2d_name=pca2d_name,
        )

    return _cb


def find_pca2d_target_name(targets: Sequence[str]) -> str:
    for t in targets:
        if is_pca2d_target(t):
            return str(t)
    return PCA2D_S1_TARGET
