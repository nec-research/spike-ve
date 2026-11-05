#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     train_anchor.py
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
"""Train the anchor model A4 (site-aware GPT, mean + PCA-2D targets, mean<->PCA-2D consistency loss).

Run (this is the ``train_anchor`` stage of ``scripts/run_experiment.sh``)::

    python -m recomb2027.latent.predictors.train_anchor \\
        --predictor-config configs/latent/predictors/anchor_model_conscurr.yaml \\
        --dataset-config configs/latent/datasets/by_nearest_ancestor/split_2022-12-01_k5_esm1_layer33.yaml
"""
from __future__ import annotations

import logging
import torch
import torch.nn.functional as F
import argparse

from pathlib import Path
from omegaconf import OmegaConf
from recomb2027.config_io import load_config
from torch.utils.data import DataLoader
from tqdm import tqdm
from recomb2027.core.constants import PRIMARY_TARGET
from recomb2027.latent.data_series.jepa_collate import (
    make_dict_target_collate_fn,
    make_jepa_collate_fn,
)
from recomb2027.latent.data_series.pca2d_features import (
    PCA2D_S1_TARGET,
    expected_mean_pca2d_input_dim,
    load_pca2d_basis,
    mean_pca2d_input_layout,
)
from recomb2027.latent.models.anchor_model import GPTModelPca2dSite, extract_pca2d_flats_from_batch
from recomb2027.latent.models.model_utils import get_predictor, set_seed
from recomb2027.latent.predictors.predictor_utils import (
    resolve_device,
    build_datasets,
    parse_pca2d_splits,
)
from recomb2027.latent.training.pca2d_curriculum import make_pca2d_curriculum_callback
from recomb2027.latent.training.predictor_trainer import train_prediction_model


# ============================================================================
# (merged from train_predictor_jepa_pca2d_site.py)
# ============================================================================

logger = logging.getLogger(__name__)


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )


def run_pca2d_ae_warmup(
    model: GPTModelPca2dSite,
    train_dataset,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    collate_fn=None,
    freeze_encoder: bool = True,
) -> None:
    """Reconstruct absolute pca2d flats with the site bottleneck only.

    After warmup, all model params are re-enabled. When ``freeze_encoder`` is
    True the site encode path is frozen immediately; otherwise it stays
    trainable (use ``freeze_encoder_after_epochs`` in the train loop to
    delay freezing).
    """
    if epochs <= 0:
        return
    assert model.pca2d_bottleneck is not None
    bn = model.pca2d_bottleneck
    for p in model.parameters():
        p.requires_grad = False
    for p in bn.parameters():
        p.requires_grad = True

    opt = torch.optim.AdamW(bn.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    logger.info(
        "pca2d AE warmup: %d epochs, lr=%g, params=%d",
        epochs,
        lr,
        sum(p.numel() for p in bn.parameters()),
    )
    bn.train()
    for ep in range(1, epochs + 1):
        total = 0.0
        n = 0
        pbar = tqdm(loader, desc=f"AE warmup {ep}/{epochs}", leave=False)
        for batch in pbar:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                xb = batch[0]
            else:
                xb = batch
            xb = xb.to(device).float()
            flats = extract_pca2d_flats_from_batch(model, xb)
            # Flatten batch×time for AE
            flat = flats.reshape(-1, flats.size(-1))
            opt.zero_grad(set_to_none=True)
            recon = bn(flat)
            loss = F.mse_loss(recon.float(), flat.float())
            if not torch.isfinite(loss):
                continue
            loss.backward()
            opt.step()
            total += float(loss.item())
            n += 1
            pbar.set_postfix(ae_mse=total / max(n, 1))
        logger.info("AE warmup epoch %d: mse=%.6f", ep, total / max(n, 1))

    # Re-enable all params; optionally freeze encoder path immediately.
    for p in model.parameters():
        p.requires_grad = True
    if freeze_encoder:
        n_frozen = model.set_pca2d_encoder_trainable(False)
        logger.info(
            "Froze pca2d site encoder after AE warmup (%d params requires_grad=False)",
            n_frozen,
        )
    else:
        logger.info(
            "Left pca2d site encoder trainable after AE warmup "
            "(delayed freeze via freeze_encoder_after_epochs)"
        )


def train_predictor_jepa_pca2d_site(cfg, data_cfg):
    set_seed(cfg.get("seed", 42))
    device = resolve_device(cfg)

    train_dataset, cal_dataset, test_dataset, basis_path = parse_pca2d_splits(
        build_datasets(cfg, data_cfg)
    )
    val_dataset = cal_dataset if cal_dataset is not None else test_dataset
    train_dataset.target_mode = cfg.get("target_mode", "diff")
    val_dataset.target_mode = cfg.get("target_mode", "diff")

    basis = load_pca2d_basis(basis_path)
    pca2d_dim = int(basis["flat_dim"])
    pca2d_name = str(cfg.get("pca2d_target_name", PCA2D_S1_TARGET))
    seq_len = int(basis["seq_len"])
    k = int(basis["k"])

    model_cfg = OmegaConf.to_container(load_config(cfg.model_cfg), resolve=True)
    if cfg.get("model_overrides"):  # e.g. {embed_dim: 192, num_gpt_blocks: 2}
        model_cfg.update(OmegaConf.to_container(cfg.model_overrides, resolve=True))
    primary_d = int(train_dataset.get_primary_latent_dim())

    gatt_plus = cfg.get("input_feature_mode", "gatt_plus") in ("gatt_plus", "diff")
    pca2d_gatt = bool(cfg.get("pca2d_gatt_plus", True))

    prediction_targets = list(
        cfg.get("prediction_targets")
        or model_cfg.get("prediction_targets")
        or [PRIMARY_TARGET, pca2d_name]
    )
    if PRIMARY_TARGET not in prediction_targets:
        raise ValueError("prediction_targets must include 'mean'")
    if pca2d_name not in prediction_targets:
        prediction_targets.append(pca2d_name)
    # Required so __getitem__ builds yb[pca2d] (otherwise KeyError at curriculum unfreeze).
    train_dataset.prediction_targets = list(prediction_targets)
    val_dataset.prediction_targets = list(prediction_targets)
    model_cfg["prediction_targets"] = list(prediction_targets)
    logger.info("Dataset prediction_targets=%s", prediction_targets)

    model_cfg["target_output_dims"] = {
        PRIMARY_TARGET: primary_d,
        pca2d_name: pca2d_dim,
    }

    bn_cfg = cfg.get("pca2d_bottleneck")
    if bn_cfg is None:
        bn_cfg = {}
    else:
        bn_cfg = OmegaConf.to_container(bn_cfg, resolve=True) or {}
    bn_enabled = bool(bn_cfg.get("enabled", True))
    bn_dim = int(bn_cfg.get("dim", 0) or model_cfg.get("pca2d_bottleneck_dim") or 128)
    if not bn_enabled:
        raise ValueError("gpt_pca2d_site requires pca2d_bottleneck.enabled=true")
    model_cfg["pca2d_bottleneck_dim"] = bn_dim
    model_cfg["pca2d_flat_dim"] = pca2d_dim
    # Prefer basis geometry over model YAML (avoids L/k vs flat_dim mismatch).
    yaml_L = model_cfg.get("pca2d_seq_len")
    yaml_k = model_cfg.get("pca2d_k")
    if yaml_L is not None and int(yaml_L) != seq_len:
        logger.warning(
            "Overriding model pca2d_seq_len=%s with basis seq_len=%s", yaml_L, seq_len
        )
    if yaml_k is not None and int(yaml_k) != k:
        logger.warning(
            "Overriding model pca2d_k=%s with basis k=%s", yaml_k, k
        )
    model_cfg["pca2d_seq_len"] = seq_len
    model_cfg["pca2d_k"] = k
    if seq_len * k != pca2d_dim:
        raise ValueError(
            f"basis flat_dim={pca2d_dim} != seq_len*k={seq_len}*{k}={seq_len * k}"
        )
    if bn_cfg.get("site_hidden") is not None:
        model_cfg["pca2d_site_hidden"] = int(bn_cfg["site_hidden"])
    if bn_cfg.get("site_dim") is not None:
        model_cfg["pca2d_site_dim"] = int(bn_cfg["site_dim"])
    if bn_cfg.get("mixer_layers") is not None:
        model_cfg["pca2d_site_mixer_layers"] = int(bn_cfg["mixer_layers"])
    if bn_cfg.get("site_pool") is not None:
        model_cfg["pca2d_site_pool"] = str(bn_cfg["site_pool"])
    if bn_cfg.get("decode_rank") is not None:
        model_cfg["pca2d_decode_rank"] = int(bn_cfg["decode_rank"])
    if bn_cfg.get("mean_cond_attn") is not None:
        model_cfg["pca2d_mean_cond_attn"] = bool(bn_cfg["mean_cond_attn"])
    if bn_cfg.get("gated_residual") is not None:
        model_cfg["pca2d_gated_residual"] = bool(bn_cfg["gated_residual"])
    if bn_cfg.get("site_resolved_gate") is not None:
        model_cfg["pca2d_site_resolved_gate"] = bool(bn_cfg["site_resolved_gate"])
    model_cfg["pca2d_residual"] = bool(
        cfg.get("pca2d_residual", model_cfg.get("pca2d_residual", True))
    )

    target_loss_weights = cfg.get("target_loss_weights")
    if target_loss_weights is None:
        target_loss_weights = model_cfg.get("target_loss_weights")
    loss_kwargs = {}
    base_tlw = {
        PRIMARY_TARGET: 1.0,
        pca2d_name: 0.5,
    }
    if target_loss_weights:
        base_tlw.update(
            {str(k): float(v) for k, v in dict(target_loss_weights).items()}
        )
    loss_kwargs["target_loss_weights"] = dict(base_tlw)
    loss_kwargs["selection_target_weights"] = dict(base_tlw)
    delta_loss_mode = str(cfg.get("delta_loss_mode", "scaled") or "scaled").lower()
    loss_kwargs["delta_loss_mode"] = delta_loss_mode
    loss_kwargs["delta_loss_eps"] = float(cfg.get("delta_loss_eps", 1e-6) or 1e-6)
    if delta_loss_mode not in ("scaled", "relative", "raw"):
        raise ValueError(
            f"delta_loss_mode must be scaled, relative, or raw; got {delta_loss_mode!r}"
        )
    logger.info("delta_loss_mode=%s eps=%.1e", delta_loss_mode, loss_kwargs["delta_loss_eps"])

    v4_raw = cfg.get("mean_pca2d_v4")
    if v4_raw is not None:
        v4 = (
            OmegaConf.to_container(v4_raw, resolve=True)
            if OmegaConf.is_config(v4_raw)
            else dict(v4_raw)
        )
        if bool(v4.get("enabled", False)):
            from recomb2027.latent.models.consistency_loss import (
                resolve_ridge_readout,
            )

            readout = resolve_ridge_readout(cfg, device=device, data_cfg=data_cfg)
            loss_kwargs["mean_pca2d_v4"] = {
                "enabled": True,
                "consistency_weight": float(v4.get("consistency_weight", 0.2)),
                "stop_grad_pca2d": bool(v4.get("stop_grad_pca2d", True)),
                "pca2d_target_name": pca2d_name,
                "readout": readout,
                "consistency_schedule": str(
                    v4.get("consistency_schedule", "constant") or "constant"
                ).lower(),
                "pca2d_weight_ref": float(base_tlw.get(pca2d_name, 0.0)),
            }
            logger.info(
                "A4 mean/pca2d consistency enabled weight=%.3f schedule=%s "
                "(pca2d_weight_ref=%.3f) ridge_sha256=%s",
                loss_kwargs["mean_pca2d_v4"]["consistency_weight"],
                loss_kwargs["mean_pca2d_v4"]["consistency_schedule"],
                loss_kwargs["mean_pca2d_v4"]["pca2d_weight_ref"],
                readout.sha256,
            )

    curriculum_raw = cfg.get("pca2d_curriculum")
    curriculum = (
        OmegaConf.to_container(curriculum_raw, resolve=True)
        if curriculum_raw is not None
        else None
    )
    epoch_callback = make_pca2d_curriculum_callback(
        curriculum, base_tlw, pca2d_name=pca2d_name
    )

    # AE warmup + optional delayed encoder freeze.
    # freeze_encoder_after_epochs>0: keep encoder trainable for that many GPT
    # epochs after AE, then freeze (AE → unfreeze window → freeze).
    ae_raw = cfg.get("pca2d_ae_warmup")
    if ae_raw is None:
        ae_raw = {}
    else:
        ae_raw = OmegaConf.to_container(ae_raw, resolve=True) or {}
    freeze_encoder = bool(ae_raw.get("freeze_encoder", True))
    ae_epochs = int(ae_raw.get("epochs", 0) or 0)
    freeze_after = int(ae_raw.get("freeze_encoder_after_epochs", 0) or 0)
    freeze_immediately = bool(freeze_encoder and freeze_after <= 0)

    def _combined_epoch_cb(model, epoch, loss_kwargs_inner):
        out = {}
        if epoch_callback is not None:
            out = epoch_callback(model, epoch, loss_kwargs_inner) or {}
        if (
            freeze_encoder
            and ae_epochs > 0
            and isinstance(model, GPTModelPca2dSite)
        ):
            if freeze_after <= 0 or epoch > freeze_after:
                n = model.set_pca2d_encoder_trainable(False)
                if epoch == max(freeze_after, 0) + 1 or (
                    freeze_after <= 0 and epoch == 1
                ):
                    logger.info(
                        "pca2d site encoder frozen at GPT epoch %d (%d params)",
                        epoch,
                        n,
                    )
            else:
                model.set_pca2d_encoder_trainable(True)
        return out

    jepa_raw = cfg.get("jepa_aux")
    jepa_aux_cfg = (
        OmegaConf.to_container(jepa_raw, resolve=True) if jepa_raw is not None else {}
    )
    train_collate_fn = make_dict_target_collate_fn()
    if jepa_aux_cfg.get("enabled"):
        jepa_predict = list(jepa_aux_cfg.get("predict_targets") or [PRIMARY_TARGET])
        model_cfg["jepa_mask_token"] = True
        model_cfg["jepa_predict_targets"] = jepa_predict
        train_dataset.jepa_return_abs = True
        train_dataset.jepa_absolute_stats = jepa_predict
        val_dataset.jepa_return_abs = False
        train_collate_fn = make_jepa_collate_fn(jepa_aux_cfg, jepa_predict)
    else:
        model_cfg["jepa_mask_token"] = False
        model_cfg.pop("jepa_predict_targets", None)

    exp_in = expected_mean_pca2d_input_dim(
        primary_dim=primary_d,
        pca2d_dim=pca2d_dim,
        gatt_plus=gatt_plus,
        pca2d_gatt_plus=pca2d_gatt,
    )
    model_cfg["input_dim"] = exp_in
    layout = mean_pca2d_input_layout(
        primary_dim=primary_d,
        pca2d_dim=pca2d_dim,
        gatt_plus=gatt_plus,
        pca2d_gatt_plus=pca2d_gatt,
    )
    model_cfg["pca2d_branch"] = True
    model_cfg["mean_pca2d_layout"] = {
        "primary_slice": list(layout["primary_slice"]),
        "pca2d_slice": list(layout["pca2d_slice"]),
    }
    model_cfg["mean_pca2d_layernorm"] = bool(
        cfg.get("mean_pca2d_layernorm", model_cfg.get("mean_pca2d_layernorm", True))
    )
    model_cfg["block_branch"] = False
    model_cfg["extra_stat_dual_branch"] = False
    model_cfg["extra_stat_projection"] = False
    model_cfg.pop("target_loss_weights", None)
    model_cfg["name"] = "gpt_pca2d_site"

    prediction_model, loss_func = get_predictor(model_cfg)
    if not isinstance(prediction_model, GPTModelPca2dSite):
        raise TypeError(
            f"expected GPTModelPca2dSite, got {type(prediction_model).__name__}"
        )
    if loss_kwargs.get("target_loss_weights") and hasattr(prediction_model, "hparams"):
        prediction_model.hparams["target_loss_weights"] = loss_kwargs[
            "target_loss_weights"
        ]
    prediction_model.hparams["pca2d_basis_path"] = str(basis_path)
    prediction_model.hparams["pca2d_dim"] = pca2d_dim
    prediction_model.hparams["scale_pca2d"] = bool(cfg.get("scale_pca2d", True))
    prediction_model.hparams["pca2d_gatt_plus"] = bool(pca2d_gatt)
    prediction_model.hparams["pca2d_ae_freeze_encoder_after_epochs"] = freeze_after
    prediction_model.hparams["delta_loss_mode"] = delta_loss_mode
    _v4_loss = loss_kwargs.get("mean_pca2d_v4")
    if _v4_loss:
        _ro = _v4_loss["readout"]
        prediction_model.hparams["ridge_readout_sha256"] = _ro.sha256
        prediction_model.hparams["ridge_readout_source"] = str(_ro.source or "")
        prediction_model.hparams["consistency_weight"] = float(_v4_loss["consistency_weight"])
        prediction_model.hparams["consistency_schedule"] = _v4_loss["consistency_schedule"]

    es_metric = str((cfg.get("early_stopping") or {}).get("metric", "joint")).lower()
    if delta_loss_mode == "raw" or es_metric == "raw_jump_ratio":
        from recomb2027.latent.training.latent_delta_loss import (
            register_latent_scaler_buffers,
        )

        register_latent_scaler_buffers(prediction_model, train_dataset)
        logger.info(
            "Registered mean/pca2d scaler buffers on model "
            "(delta_loss_mode=%s early_stop_metric=%s)",
            delta_loss_mode,
            es_metric,
        )

    n_params = sum(p.numel() for p in prediction_model.parameters()) / 1e6
    logger.info(
        "Site-residual model: %.3f M params (input_dim=%s flat=%s bn=%s "
        "L=%s k=%s residual=%s)",
        n_params,
        model_cfg["input_dim"],
        pca2d_dim,
        bn_dim,
        model_cfg["pca2d_seq_len"],
        model_cfg["pca2d_k"],
        model_cfg["pca2d_residual"],
    )

    prediction_model = prediction_model.to(device)
    resume_ckpt = cfg.get("resume_checkpoint")
    if resume_ckpt:
        # Resume restores weights; AE on a fresh init would be discarded/wasted.
        # requires_grad is not in the checkpoint — epoch_callback re-applies
        # freeze once epoch > freeze_encoder_after_epochs.
        logger.info(
            "Skipping pca2d AE warmup because resume_checkpoint is set (%s)",
            resume_ckpt,
        )
        if freeze_immediately and ae_epochs > 0:
            n_frozen = prediction_model.set_pca2d_encoder_trainable(False)
            logger.info(
                "Re-applied pca2d encoder freeze after resume (%d params)",
                n_frozen,
            )
    else:
        run_pca2d_ae_warmup(
            prediction_model,
            train_dataset,
            device=device,
            epochs=ae_epochs,
            batch_size=int(cfg.get("batch_size", 64)),
            lr=float(ae_raw.get("lr", 1e-3)),
            collate_fn=train_collate_fn,
            freeze_encoder=freeze_immediately,
        )
        if freeze_encoder and freeze_after > 0 and ae_epochs > 0:
            logger.info(
                "pca2d site encoder will stay trainable for %d GPT epochs, "
                "then freeze",
                freeze_after,
            )

    checkpoint_dir = cfg.get("checkpoint_dir", "./checkpoints")
    tensorboard_dir = cfg.get("tensorboard_dir", "./runs")
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    if tensorboard_dir is not None:
        Path(tensorboard_dir).mkdir(parents=True, exist_ok=True)

    results = train_prediction_model(
        prediction_model,
        loss_func,
        train_dataset,
        val_dataset,
        batch_size=cfg.get("batch_size", 32),
        device=device,
        lr=cfg.get("learning_rate", 1e-4),
        weight_decay=cfg.get("weight_decay", 0.0),
        epochs=cfg.get("epochs", 100),
        clip_grad_norm=cfg.get("clip_grad_norm", 1.0),
        early_stopping_dict=cfg.get("early_stopping", None),
        scheduler_dict=cfg.get("scheduler", None),
        save_dir=checkpoint_dir,
        tensorboard_dir=tensorboard_dir,
        kl_annealing_dict=cfg.get("kl_annealing", None),
        train_collate_fn=train_collate_fn,
        loss_kwargs=loss_kwargs or None,
        resume_checkpoint=resume_ckpt,
        all_step_loss=bool(cfg.get("all_step_loss", False)),
        epoch_callback=_combined_epoch_cb,
        amp_dtype=cfg.get("amp_dtype", "bf16"),
        ema_decay=cfg.get("ema_decay", None),
        input_noise_std=float(cfg.get("input_noise_std", 0.0) or 0.0),
        site_metric_cfg=(
            {
                "seq_len": seq_len,
                "k": k,
                "pca2d_name": pca2d_name,
                "topk": int((cfg.get("early_stopping") or {}).get("site_topk", 20)),
            }
            if str((cfg.get("early_stopping") or {}).get("metric", "")).lower()
            in ("joint_site", "site_capture")
            else None
        ),
    )
    return results


# ============================================================================
# (merged from train_predictor_jepa_pca2d_site_cond.py)
# ============================================================================

def _require_option_c(cfg) -> None:
    bn = cfg.get("pca2d_bottleneck") or {}
    if OmegaConf.is_config(bn):
        bn = OmegaConf.to_container(bn, resolve=True) or {}
    mean_cond = bool(bn.get("mean_cond_attn", False))
    gated = bool(bn.get("gated_residual", False))
    if not mean_cond or not gated:
        raise ValueError(
            "train_predictor_jepa_pca2d_site_cond requires "
            "pca2d_bottleneck.mean_cond_attn=true and "
            "pca2d_bottleneck.gated_residual=true "
            f"(got mean_cond_attn={mean_cond}, gated_residual={gated})"
        )


# ============================================================================
# (merged from train_predictor_jepa_pca2d_site_v4.py)
# ============================================================================

_DEFAULT_PREDICTOR = "configs/latent/predictors/anchor_model_conscurr.yaml"
_DEFAULT_DATASET = (
    "configs/latent/datasets/by_nearest_ancestor/split_2022-12-01_k5_esm1_layer33.yaml"
)


def main():
    parser = argparse.ArgumentParser(
        description="Train A4 site GPT (A3 + mean/pca2d consistency loss)."
    )
    parser.add_argument("--predictor-config", default=_DEFAULT_PREDICTOR)
    parser.add_argument("--dataset-config", default=_DEFAULT_DATASET)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--force-rebuild-pca2d", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    _setup_logging(args.log_level)

    predictor_config = load_config(args.predictor_config)
    dataset_config = load_config(args.dataset_config)
    _require_option_c(predictor_config)

    if args.resume_checkpoint:
        predictor_config["resume_checkpoint"] = args.resume_checkpoint
    if args.force_rebuild_pca2d:
        predictor_config["force_rebuild_pca2d"] = True

    train_predictor_jepa_pca2d_site(predictor_config, dataset_config)


if __name__ == "__main__":
    main()
