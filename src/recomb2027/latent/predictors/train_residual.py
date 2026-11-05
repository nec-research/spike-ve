#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     train_residual.py
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
"""Train the residual head B4 (unified site model, mean read out from the gated residual) on a frozen anchor.

Run (this is the ``train_residual`` stage of ``scripts/run_experiment.sh``)::

    python -m recomb2027.latent.predictors.train_residual \\
        --predictor-config configs/latent/predictors/residual_model_conscurr.yaml \\
        --dataset-config configs/latent/datasets/by_nearest_ancestor/split_2022-12-01_k5_esm1_layer33.yaml \\
        --mode mse --tag a4b4_conscurr

Contains the data-loader construction, the training loop with EMA / early stopping, and the model builder.
"""
from __future__ import annotations

import logging
import numpy as np
import torch
import math
import time
import argparse

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from collections import Counter
from typing import Any, Optional
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from recomb2027.core.constants import PRIMARY_TARGET
from recomb2027.latent.data_series.edge_phylo_features import attach_edge_phylo_from_msa
from recomb2027.latent.data_series.jepa_collate import make_dict_target_collate_fn
from recomb2027.latent.data_series.msa_conservation_features import load_or_compute_conservation
from recomb2027.latent.data_series.pca2d_features import PCA2D_S1_TARGET
from recomb2027.latent.predictors.predictor_utils import (
    _load_pca2d_dataset,
    parse_pca2d_splits,
    build_datasets as build_pca2d_datasets,
    finalize_merged_metrics,
    merge_metric_dicts,
    summarize_ceiling_batch,
    resolve_device,
)
from recomb2027.latent.data_series.horizon_collate import (
    horizon_from_batch,
    make_horizon_collate_fn,
)
from recomb2027.latent.data_series.msa_substitution_features import PREFIX_FEATURE_DIM
from recomb2027.latent.data_series.substitution_dataset import attach_substitution_sidecars
from recomb2027.latent.models.model_utils import ModelEMA, set_seed
from recomb2027.latent.models.residual_model import (
    load_frozen_site_gpt_anchor,
    Pca2dSiteUnifiedV3,
    gate_ranking_metrics,
    Pca2dSiteUnifiedV4,
)
from recomb2027.config_io import load_config
from recomb2027.latent.models.consistency_loss import mean_pca2d_v4_cfg, resolve_ridge_readout


# ============================================================================
# (merged from train_diffusion_pca2d_site_residual.py)
# ============================================================================

logger = logging.getLogger(__name__)


_SAMPLE_EARLY_STOP = frozenset({"one_sample", "best_of_k", "mean_of_k"})
_ALL_EARLY_STOP = _SAMPLE_EARLY_STOP | frozenset({"full_x0", "total", "residual"})


class _TeeStdout:
    def __init__(self, stream, log_file):
        self._stream = stream
        self._log_file = log_file

    def write(self, data):
        self._stream.write(data)
        self._log_file.write(data)

    def flush(self):
        self._stream.flush()
        self._log_file.flush()

    def isatty(self):
        return getattr(self._stream, "isatty", lambda: False)()


@contextmanager
def run_logging(log_dir: str | None, tag: str, *, use_tensorboard: bool = True):
    import sys

    orig_stdout = sys.stdout
    log_file = None
    writer = None
    paths: dict[str, Path] = {}
    try:
        if not log_dir:
            yield None, paths
            return
        log_root = Path(log_dir).expanduser() / tag
        log_root.mkdir(parents=True, exist_ok=True)
        run_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        text_log = log_root / f"train_{run_stamp}.log"
        log_file = open(text_log, "w", encoding="utf-8")
        sys.stdout = _TeeStdout(orig_stdout, log_file)
        paths["text_log"] = text_log
        paths["log_root"] = log_root
        print(f"Text log → {text_log}", flush=True)
        if use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            tb_dir = log_root / "tensorboard" / f"run_{run_stamp}"
            tb_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir=str(tb_dir))
            paths["tensorboard"] = tb_dir
            print(f"TensorBoard → {tb_dir}", flush=True)
        yield writer, paths
    finally:
        sys.stdout = orig_stdout
        if log_file is not None:
            log_file.close()
        if writer is not None:
            writer.close()


# ============================================================================
# (merged from train_residual_pca2d_site_v2.py)
# ============================================================================

_EARLY_STOP = frozenset({"full_x0", "residual", "total", "composite"})


def _training_cfg(cfg) -> dict[str, Any]:
    tr = cfg.get("training") or {}
    if OmegaConf.is_config(tr):
        return OmegaConf.to_container(tr, resolve=True) or {}
    return dict(tr)


def _active_timeline_indices(ds, idx: int) -> list[int]:
    ds_leaf = ds
    local_idx = idx
    if isinstance(ds, ConcatDataset):
        offset = 0
        for sub in ds.datasets:
            n = len(sub)
            if local_idx < offset + n:
                return _active_timeline_indices(sub, local_idx - offset)
            offset += n
        return [0]
    if isinstance(ds, _WeightedDictTargetDataset):
        return _active_timeline_indices(ds.base, local_idx)
    if isinstance(ds, _EdgePhyloHeadDataset):
        return _active_timeline_indices(ds.base, local_idx)
    active = getattr(ds_leaf, "_timeline_indices", None)
    enc = getattr(ds_leaf, "sample_encodings", None)
    n = int(enc[local_idx].shape[0]) if enc is not None else 1
    if active is None:
        return list(range(n))
    return [int(x) for x in np.asarray(active).tolist()]


def _edge_phylo_for_transition(dataset, idx: int, trans_idx: int) -> float:
    ds = dataset
    local_idx = idx
    if isinstance(ds, ConcatDataset):
        offset = 0
        for sub in ds.datasets:
            n = len(sub)
            if local_idx < offset + n:
                return _edge_phylo_for_transition(sub, local_idx - offset, trans_idx)
            offset += n
        return 0.0
    if isinstance(ds, _WeightedDictTargetDataset):
        return _edge_phylo_for_transition(ds.base, local_idx, trans_idx)
    if isinstance(ds, _EdgePhyloHeadDataset):
        return _edge_phylo_for_transition(ds.base, local_idx, trans_idx)
    edge = getattr(ds, "edge_distances", None)
    if edge is None:
        return 0.0
    row = np.asarray(edge[local_idx], dtype=np.float32).reshape(-1)
    active = _active_timeline_indices(ds, local_idx)
    if len(active) < 2:
        return 0.0
    t0 = min(int(trans_idx), len(active) - 2)
    t1 = t0 + 1
    lo, hi = int(active[t0]), int(active[t1])
    if row.size == 0:
        return 0.0
    d = float(row[lo:hi].sum()) if hi > lo else float(row[min(lo, row.size - 1)])
    if getattr(ds, "edge_phylo_use_log1p", True):
        d = float(np.log1p(max(d, 0.0)))
    return d


def _last_step_edge_phylo(dataset, idx: int) -> float:
    active = _active_timeline_indices(dataset, idx)
    n_trans = max(0, len(active) - 1)
    if n_trans <= 0:
        return 0.0
    return _edge_phylo_for_transition(dataset, idx, n_trans - 1)


def _all_step_edge_phylo(dataset, idx: int) -> torch.Tensor:
    active = _active_timeline_indices(dataset, idx)
    n_trans = max(0, len(active) - 1)
    vals = [
        _edge_phylo_for_transition(dataset, idx, t) for t in range(n_trans)
    ]
    if not vals:
        vals = [0.0]
    return torch.tensor(vals, dtype=torch.float32).unsqueeze(-1)


class _EdgePhyloHeadDataset(Dataset):
    """Yield edge phylo for head conditioning (not trunk input)."""

    def __init__(self, base: Dataset, *, all_step: bool = False):
        self.base = base
        self.all_step = bool(all_step)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        item = self.base[idx]
        if self.all_step:
            ep = _all_step_edge_phylo(self.base, idx)
        else:
            ep = torch.tensor(
                [_last_step_edge_phylo(self.base, idx)], dtype=torch.float32
            )
        if len(item) == 4:
            x, y, a, b = item
            if isinstance(b, dict):
                return x, y, a, ep, b
            if isinstance(a, torch.Tensor) and a.dim() == 0:
                return x, y, a, ep, b
            return x, y, ep, a, b
        if len(item) == 3:
            x, y, third = item
            if isinstance(third, dict):
                return x, y, ep, third
            return x, y, third, ep
        x, y = item
        return x, y, ep


def _iter_leaf_datasets(ds) -> list:
    if isinstance(ds, ConcatDataset):
        out: list = []
        for sub in ds.datasets:
            out.extend(_iter_leaf_datasets(sub))
        return out
    if isinstance(ds, _WeightedDictTargetDataset):
        return _iter_leaf_datasets(ds.base)
    if isinstance(ds, _EdgePhyloHeadDataset):
        return _iter_leaf_datasets(ds.base)
    # SubstitutionFeatureDataset (and similar outermost feature wrappers).
    if type(ds).__name__ == "SubstitutionFeatureDataset" and hasattr(ds, "base"):
        return _iter_leaf_datasets(ds.base)
    return [ds]


def _attach_head_edge_phylo(datasets: list, cfg, data_cfg) -> None:
    if not bool(cfg.get("append_edge_phylo_to_head", False)):
        return
    msa_path = cfg.get("msa_path") or data_cfg.get("msa_path")
    if not msa_path:
        raise ValueError(
            "append_edge_phylo_to_head requires msa_path in predictor or dataset config"
        )
    cache_dir = cfg.get("edge_phylo_cache_dir")
    if cache_dir is None:
        cache_dir = str(Path(data_cfg["processed_dir"]) / ".edge_phylo_cache")
    use_log1p = bool(cfg.get("edge_phylo_use_log1p", True))
    for ds in datasets:
        if ds is None:
            continue
        for leaf in _iter_leaf_datasets(ds):
            split = getattr(leaf, "split", "data")
            attach_edge_phylo_from_msa(
                leaf,
                msa_path,
                cache_dir=cache_dir,
                split=split,
                use_log1p=use_log1p,
                log_prefix=f"[{split}] head ",
            )
            leaf.append_edge_phylo_distance = False


def _load_conservation_stats(cfg, data_cfg, *, seq_len: int) -> Optional[torch.Tensor]:
    if not bool(cfg.get("append_msa_conservation", False)):
        return None
    msa_path = cfg.get("msa_path") or data_cfg.get("msa_path")
    if not msa_path:
        raise ValueError(
            "append_msa_conservation requires msa_path in predictor or dataset config"
        )
    cache_path = cfg.get("conservation_cache_path")
    if cache_path is None:
        cache_path = (
            Path(data_cfg["processed_dir"]) / "msa_conservation_sidecar.npz"
        )
    arr = load_or_compute_conservation(
        msa_path, seq_len=seq_len, cache_path=cache_path
    )
    return torch.from_numpy(arr)


def compute_inverse_predecessor_weights(dataset) -> torch.Tensor:
    """``1 / count(predecessor_gid)`` per sample."""
    gids = getattr(dataset, "sample_gids", None)
    if gids is None:
        return torch.ones(0, dtype=torch.float32)
    preds = [str(row[-2]) for row in gids]
    counts = Counter(preds)
    weights = [1.0 / max(counts[p], 1) for p in preds]
    return torch.tensor(weights, dtype=torch.float32)


def compute_train_sample_weights(train_ds) -> Optional[torch.Tensor]:
    """Sample weights for train set, including ``ConcatDataset`` unions."""
    if isinstance(train_ds, ConcatDataset):
        parts = [compute_inverse_predecessor_weights(ds) for ds in train_ds.datasets]
        if not parts:
            return None
        return torch.cat(parts, dim=0)
    return compute_inverse_predecessor_weights(train_ds)


class _WeightedDictTargetDataset(Dataset):
    """Attach per-sample weights without breaking dict-target collate."""

    def __init__(self, base: Dataset, weights: torch.Tensor):
        self.base = base
        self.weights = weights

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        item = self.base[idx]
        if len(item) == 3:
            x, y, meta = item
            return x, y, self.weights[idx], meta
        x, y = item
        return x, y, self.weights[idx]


def _make_train_collate(
    use_weights: bool,
    use_edge_phylo: bool,
    *,
    return_horizon_meta: bool = False,
    horizon_embed_dim: int = 4,
):
    if return_horizon_meta:
        from recomb2027.latent.data_series.horizon_collate import (
            make_horizon_collate_fn,
        )

        return make_horizon_collate_fn(
            use_weights=use_weights,
            use_edge_phylo=use_edge_phylo,
            horizon_embed_dim=max(int(horizon_embed_dim), 1),
        )

    base_collate = make_dict_target_collate_fn()

    if not use_weights and not use_edge_phylo:
        return base_collate

    def collate(batch):
        if use_weights and use_edge_phylo:
            weights = torch.stack([item[2] for item in batch])
            edge = torch.stack([item[3] for item in batch])
            xb, yb = base_collate([(item[0], item[1]) for item in batch])
            return xb, yb, weights, edge
        if use_weights:
            weights = torch.stack([item[2] for item in batch])
            xb, yb = base_collate([(item[0], item[1]) for item in batch])
            return xb, yb, weights
        edge = torch.stack([item[2] for item in batch])
        xb, yb = base_collate([(item[0], item[1]) for item in batch])
        return xb, yb, edge

    return collate


def _eval_collate(
    use_edge_phylo: bool,
    *,
    return_horizon_meta: bool = False,
    horizon_embed_dim: int = 4,
):
    if return_horizon_meta:
        from recomb2027.latent.data_series.horizon_collate import (
            make_horizon_collate_fn,
        )

        return make_horizon_collate_fn(
            use_weights=False,
            use_edge_phylo=use_edge_phylo,
            horizon_embed_dim=max(int(horizon_embed_dim), 1),
        )

    base_collate = make_dict_target_collate_fn()
    if not use_edge_phylo:
        return base_collate

    def collate(batch):
        edge = torch.stack([item[2] for item in batch])
        xb, yb = base_collate([(item[0], item[1]) for item in batch])
        return xb, yb, edge

    return collate


def _maybe_concat_train_dataset(train_ds, cfg, data_cfg):
    paths = cfg.get("multi_horizon_paths")
    if not paths:
        return train_ds
    extra = []
    processed = Path(str(data_cfg.processed_dir)).parent
    enc_2d_dir = Path(str(cfg.get("enc_2d_dir") or data_cfg.get("enc_2d_dir") or ""))
    basis_path = Path(
        str(cfg.get("pca2d_basis_path") or data_cfg.get("pca2d_basis_path") or "")
    )
    pca2d_name = str(cfg.get("pca2d_target_name", PCA2D_S1_TARGET))
    force = bool(cfg.get("force_rebuild_pca2d", False))
    scale_pca2d = bool(cfg.get("scale_pca2d", True))
    for rel in paths:
        npz = Path(str(rel))
        if not npz.is_file():
            npz = processed / str(rel) / "train_dataset.npz"
        if not npz.is_file():
            logger.warning("Skipping missing multi_horizon path: %s", rel)
            continue
        ds = _load_pca2d_dataset(
            npz,
            enc_2d_dir=enc_2d_dir,
            basis_path=basis_path,
            pca2d_name=pca2d_name,
            load_scaler=True,
            force_rebuild=force,
            fit_pca2d_scaler=False,
            pca2d_scaler_mode=str(
                getattr(train_ds, "pca2d_scaler_mode", "flat") or "flat"
            ),
        )
        ds.pca2d_scaler = train_ds.pca2d_scaler
        ds.pca2d_scaler_mode = str(
            getattr(train_ds, "pca2d_scaler_mode", "flat") or "flat"
        )
        if int(getattr(train_ds, "_pca2d_k", 0) or 0) > 0:
            ds._pca2d_k = int(train_ds._pca2d_k)
        ds.input_feature_mode = train_ds.input_feature_mode
        ds.pca2d_gatt_plus = train_ds.pca2d_gatt_plus
        ds.target_mode = train_ds.target_mode
        ds.prediction_targets = train_ds.prediction_targets
        extra.append(ds)
    if not extra:
        return train_ds
    return ConcatDataset([train_ds, *extra])


def _reset_eval_timelines(datasets: list) -> None:
    for ds in datasets:
        if ds is None:
            continue
        for leaf in _iter_leaf_datasets(ds):
            if hasattr(leaf, "reset_timeline_indices"):
                leaf.reset_timeline_indices()


def _advance_train_subsample(train_ds, epoch: int) -> None:
    for leaf in _iter_leaf_datasets(train_ds):
        if hasattr(leaf, "advance_epoch_subsample"):
            leaf.advance_epoch_subsample(epoch)


def build_loaders_with_options(cfg, data_cfg, *, batch_size: int, num_workers: int):
    """Like ``build_pca2d_site_loaders`` but supports multi-horizon train union."""
    train_ds, cal_ds, test_ds, _ = parse_pca2d_splits(
        build_pca2d_datasets(cfg, data_cfg)
    )
    train_ds = _maybe_concat_train_dataset(train_ds, cfg, data_cfg)
    pca2d_name = str(cfg.get("pca2d_target_name", PCA2D_S1_TARGET))
    target_mode = cfg.get("target_mode", "diff")
    prediction_targets = list(
        cfg.get("prediction_targets") or [PRIMARY_TARGET, pca2d_name]
    )
    if pca2d_name not in prediction_targets:
        prediction_targets.append(pca2d_name)

    for ds in (train_ds, cal_ds, test_ds):
        if ds is None:
            continue
        if isinstance(ds, ConcatDataset):
            for sub in ds.datasets:
                sub.target_mode = target_mode
                sub.prediction_targets = prediction_targets
        else:
            ds.target_mode = target_mode
            ds.prediction_targets = prediction_targets

    use_weights = bool(cfg.get("redundancy_weighting", False))
    sample_weights = compute_train_sample_weights(train_ds) if use_weights else None
    if sample_weights is not None:
        train_ds = _WeightedDictTargetDataset(train_ds, sample_weights)

    _attach_head_edge_phylo([train_ds, cal_ds, test_ds], cfg, data_cfg)
    use_edge_phylo = bool(cfg.get("append_edge_phylo_to_head", False))
    all_step_loss = bool(cfg.get("all_step_loss", False))
    if use_edge_phylo:
        if train_ds is not None:
            train_ds = _EdgePhyloHeadDataset(train_ds, all_step=all_step_loss)
        if cal_ds is not None:
            cal_ds = _EdgePhyloHeadDataset(cal_ds, all_step=False)
        if test_ds is not None:
            test_ds = _EdgePhyloHeadDataset(test_ds, all_step=False)

    return_horizon_meta = bool(cfg.get("return_horizon_meta", False))
    horizon_embed_dim = int(cfg.get("horizon_embed_dim", 4) or 4)
    collate = _make_train_collate(
        sample_weights is not None,
        use_edge_phylo,
        return_horizon_meta=return_horizon_meta,
        horizon_embed_dim=horizon_embed_dim,
    )
    eval_collate = _eval_collate(
        use_edge_phylo,
        return_horizon_meta=return_horizon_meta,
        horizon_embed_dim=horizon_embed_dim,
    )

    def _loader(ds, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate if shuffle else eval_collate,
            num_workers=num_workers,
            drop_last=shuffle,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=num_workers,
        drop_last=True,
    )
    cal_loader = _loader(cal_ds, shuffle=False) if cal_ds is not None else None
    test_loader = _loader(test_ds, shuffle=False)
    _reset_eval_timelines([cal_ds, test_ds])
    return train_loader, cal_loader, test_loader, pca2d_name, sample_weights


# ============================================================================
# (merged from train_unified_pca2d_site_v3.py)
# ============================================================================

def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )


def _unified_cfg(cfg) -> dict[str, Any]:
    u = cfg.get("unified_v3") or {}
    if OmegaConf.is_config(u):
        return OmegaConf.to_container(u, resolve=True) or {}
    return dict(u)


def build_v3_loaders(cfg, data_cfg, *, batch_size: int, num_workers: int):
    train_loader, cal_loader, test_loader, pca2d_name, sample_weights = (
        build_loaders_with_options(cfg, data_cfg, batch_size=batch_size, num_workers=num_workers)
    )
    use_weights = sample_weights is not None
    use_edge = bool(cfg.get("append_edge_phylo_to_head", False))
    use_substitution = bool(cfg.get("append_msa_substitution", False))
    use_target_activation = bool(cfg.get("substitution_target_activation", False))
    all_step_loss = bool(cfg.get("all_step_loss", False))
    embed_dim = int(cfg.get("horizon_embed_dim", 4) or 4) if bool(
        cfg.get("horizon_embedding", True)
    ) else 0
    graph_n = 1 + len(cfg.get("multi_horizon_paths") or [])

    train_wrapped, cal_wrapped, test_wrapped = attach_substitution_sidecars(
        [
            train_loader.dataset,
            cal_loader.dataset if cal_loader else None,
            test_loader.dataset,
        ],
        cfg,
        data_cfg,
    )

    for leaf in _iter_leaf_datasets(train_wrapped):
        leaf.return_horizon_meta = embed_dim > 0
        leaf.return_all_step_horizon_meta = bool(all_step_loss and embed_dim > 0)
    for ds in (cal_wrapped, test_wrapped):
        if ds is not None:
            for leaf in _iter_leaf_datasets(ds):
                leaf.return_horizon_meta = embed_dim > 0
                leaf.return_all_step_horizon_meta = False
    collate = make_horizon_collate_fn(
        use_weights=use_weights,
        use_edge_phylo=use_edge,
        use_substitution=use_substitution,
        use_target_activation=use_target_activation,
        horizon_embed_dim=max(embed_dim, 1),
        graph_n=graph_n,
    )
    eval_collate = make_horizon_collate_fn(
        use_weights=False,
        use_edge_phylo=use_edge,
        use_substitution=use_substitution,
        use_target_activation=use_target_activation,
        horizon_embed_dim=max(embed_dim, 1),
        graph_n=graph_n,
    )

    def _make_loader(ds, *, shuffle: bool) -> DataLoader:
        pin_memory = bool(cfg.get("pin_memory", num_workers > 0))
        persistent_workers = bool(cfg.get("persistent_workers", num_workers > 0))
        loader_kwargs: dict[str, Any] = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "collate_fn": collate if shuffle else eval_collate,
            "num_workers": num_workers,
            "drop_last": shuffle,
        }
        if num_workers > 0:
            loader_kwargs["pin_memory"] = pin_memory
            loader_kwargs["persistent_workers"] = persistent_workers
        return DataLoader(ds, **loader_kwargs)

    train_loader = _make_loader(train_wrapped, shuffle=True)
    cal_loader = _make_loader(cal_wrapped, shuffle=False) if cal_wrapped is not None else None
    test_loader = _make_loader(test_wrapped, shuffle=False)
    return train_loader, cal_loader, test_loader, pca2d_name, sample_weights


def build_unified_v3_model_from_cfg(
    cfg,
    *,
    site_gpt_checkpoint: str,
    device: torch.device,
    conservation_stats: Optional[torch.Tensor] = None,
    data_cfg: Optional[Any] = None,
) -> Pca2dSiteUnifiedV3:
    site_gpt = load_frozen_site_gpt_anchor(site_gpt_checkpoint, device)
    u = _unified_cfg(cfg)
    edge_dim = 1 if bool(cfg.get("append_edge_phylo_to_head", False)) else 0
    cons_dim = 3 if bool(cfg.get("append_msa_conservation", False)) else 0
    subst_dim = int(PREFIX_FEATURE_DIM) if bool(cfg.get("append_msa_substitution", False)) else 0
    use_activation = bool(cfg.get("substitution_target_activation", False))
    horizon_dim = int(cfg.get("horizon_embed_dim", 4) or 0) if bool(
        cfg.get("horizon_embedding", True)
    ) else 0
    pca2d_name = str(cfg.get("pca2d_target_name", PCA2D_S1_TARGET))
    use_mean_residual = bool(u.get("use_mean_residual", False))
    mean_aux_weight = float(u.get("mean_aux_weight", 1.0))
    mean_dim = u.get("mean_dim", None)
    model = Pca2dSiteUnifiedV3(
        site_gpt,
        cond_dim=int(u.get("cond_dim", 256)),
        mem_dim=int(u.get("mem_dim", 128)),
        site_channels=int(u.get("site_channels", 96)),
        dit_layers=int(u.get("dit_layers", 4)),
        dit_heads=int(u.get("dit_heads", 4)),
        dropout=float(u.get("dropout", 0.1)),
        cov_rank=int(u.get("cov_rank", 4)),
        horizon_dim=horizon_dim,
        edge_phylo_dim=edge_dim,
        use_temporal_pca2d=bool(u.get("use_temporal_pca2d", True)),
        temporal_proj_dim=int(u.get("temporal_proj_dim", 16)),
        conservation_dim=cons_dim,
        substitution_dim=subst_dim,
        use_activation_supervision=use_activation,
        activation_aux_weight=float(u.get("activation_aux_weight", 0.5)),
        activation_head_input=str(u.get("activation_head_input", "site_feats")),
        activation_pos_weight=float(u.get("activation_pos_weight", 1.0)),
        use_mean_residual=use_mean_residual,
        mean_aux_weight=mean_aux_weight,
        mean_dim=int(mean_dim) if mean_dim is not None else None,
        bridge_sigma=float(u.get("bridge_sigma", 0.1)),
        pca2d_name=pca2d_name,
    )
    if conservation_stats is not None:
        model.set_conservation_stats(conservation_stats)
    model.to(device)
    return model


def _unpack_v3_batch(batch, device: torch.device, *, cfg=None):
    """Unpack collated v3 batch tensors.

    Collate order (when present): weights, edge_phylo, horizon, substitution,
    activation. Substitution is ``[B,T,L,F]`` or ``[B,L,F]``; activation is
    ``[B,T,L]`` or ``[B,L]`` — never confuse activation with substitution.
    """
    xb = batch[0].to(device)
    yb = batch[1]
    sw = edge_phylo = horizon = substitution_feats = activation_target = None
    embed_dim = int((cfg or {}).get("horizon_embed_dim", 4) or 4)

    for item in batch[2:]:
        if not isinstance(item, torch.Tensor):
            continue
        if item.dim() == 1:
            sw = item.to(device)
            continue
        if item.dim() == 4:
            # [B, T, L, F] substitution
            substitution_feats = item.to(device)
            continue
        if item.dim() == 3:
            last = int(item.shape[-1])
            if last == 1:
                edge_phylo = item.to(device)
            elif last == embed_dim:
                horizon = item.to(device)
            elif last == 3 and bool((cfg or {}).get("use_drift_jump_labels", False)):
                # [B, T, 3] drift/jump meta — handled by v5 unpack
                continue
            elif last == PREFIX_FEATURE_DIM:
                # [B, L, F] single-step substitution
                substitution_feats = item.to(device)
            else:
                # [B, T, L] multi-step activation masks
                activation_target = item.to(device)
            continue
        if item.dim() == 2:
            last = int(item.shape[1])
            if last == 1:
                edge_phylo = item.to(device)
            elif last == embed_dim:
                horizon = item.to(device)
            elif last == PREFIX_FEATURE_DIM and item.shape[0] > 0:
                # Ambiguous only if L == PREFIX_FEATURE_DIM; prefer subst when
                # append_msa_substitution is on and L is tiny (unit tests).
                substitution_feats = item.to(device)
            else:
                activation_target = item.to(device)

    return xb, yb, sw, edge_phylo, horizon, substitution_feats, activation_target


def _select_step_tensor(
    tensor: Optional[torch.Tensor], step_index: int, *, last_only: bool = False
) -> Optional[torch.Tensor]:
    """Select one step from ``[B,T,...]`` tensors; pass through step-free tensors."""
    if tensor is None:
        return None
    # Substitution: [B, T, L, F]
    if tensor.dim() == 4:
        idx = -1 if last_only else int(step_index)
        return tensor[:, idx]
    # Multi-step: horizon/edge [B,T,D], activation [B,T,L], or single-step
    # substitution [B,L,F] (F == PREFIX_FEATURE_DIM).
    if tensor.dim() == 3:
        if tensor.shape[-1] == PREFIX_FEATURE_DIM:
            # [B, L, F] — already step-selected
            return tensor
        idx = -1 if last_only else int(step_index)
        return tensor[:, idx]
    return tensor


def _forward_batch_loss(
    model: Pca2dSiteUnifiedV3,
    xb: torch.Tensor,
    y_pca: torch.Tensor,
    *,
    mode: str,
    horizon: Optional[torch.Tensor],
    edge_phylo: Optional[torch.Tensor],
    substitution_feats: Optional[torch.Tensor],
    activation_target: Optional[torch.Tensor],
    mse_aux: float,
    sample_weight: Optional[torch.Tensor],
    all_step_loss: bool,
    y_mean: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not all_step_loss or y_pca.dim() != 3 or y_pca.shape[1] <= 1:
        loss, _, info = model(
            xb,
            y_pca,
            mode=mode,
            horizon=_select_step_tensor(horizon, -1, last_only=True),
            edge_phylo=_select_step_tensor(edge_phylo, -1, last_only=True),
            substitution_feats=_select_step_tensor(
                substitution_feats, -1, last_only=True
            ),
            activation_target=_select_step_tensor(
                activation_target, -1, last_only=True
            ),
            mse_aux_weight=mse_aux,
            sample_weight=sample_weight,
            y_mean=y_mean,
        )
        return loss, info

    n_steps = int(y_pca.shape[1])
    total = xb.new_zeros(())
    last_info: dict[str, torch.Tensor] = {}
    for step_t in range(n_steps):
        x_prefix = xb[:, : step_t + 1]
        loss_t, _, info_t = model(
            x_prefix,
            y_pca,
            mode=mode,
            horizon=_select_step_tensor(horizon, step_t),
            edge_phylo=_select_step_tensor(edge_phylo, step_t),
            substitution_feats=_select_step_tensor(substitution_feats, step_t),
            activation_target=_select_step_tensor(activation_target, step_t),
            mse_aux_weight=mse_aux,
            sample_weight=sample_weight,
            step_index=step_t,
            y_mean=y_mean,
        )
        total = total + loss_t
        if step_t == n_steps - 1:
            last_info = info_t
    return total / float(n_steps), last_info


@torch.no_grad()
def eval_v3_metrics(
    model: Pca2dSiteUnifiedV3,
    loader: DataLoader,
    *,
    pca2d_name: str,
    device: torch.device,
    cfg,
    mode: str = "nll",
    max_batches: Optional[int] = None,
) -> dict[str, float]:
    model.eval()
    acc: dict[str, Any] = {}
    gate_scores: list = []
    gate_changed: list = []
    gate_prefix: list = []
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
        target = yb[pca2d_name].to(device)[:, -1, :]
        dist = model.predict_distribution(
            xb,
            horizon=horizon,
            edge_phylo=edge_phylo,
            substitution_feats=substitution_feats,
            activation_target=activation_target,
        )
        pred_flat = dist["pred_flat"]
        mu_flat = dist["mu_flat"]
        if activation_target is not None and (
            "activation_logits" in dist or "activation_probs" in dist
        ):
            sc = dist.get("activation_logits", dist.get("activation_probs"))
            gate_scores.append(sc.detach().float().cpu().numpy())
            gate_changed.append((activation_target.detach().cpu().numpy() > 0.5))
            if substitution_feats is not None:
                # channel 1 = cumulative number of prefix substitutions at the site
                gate_prefix.append(substitution_feats[..., 1].detach().cpu().numpy())
        batch_m = summarize_ceiling_batch(
            target=target,
            mu_flat=mu_flat,
            pred_flat=pred_flat,
            seq_len=model.seq_len,
            k=model.k,
        )
        batch_m["full_x0_mse"] = float((pred_flat - target).pow(2).mean().item())
        batch_m["n"] = float(target.shape[0])
        acc = merge_metric_dicts(acc, batch_m)
    out = finalize_merged_metrics(acc)
    out["full_x0_mse"] = out.get("full_mse", out.get("full_x0_mse", float("nan")))
    if "full_mse" in out and "persist_mse" in out:
        out["skill_full"] = 1.0 - out["full_mse"] / max(out["persist_mse"], 1e-12)
    if gate_scores:
        import numpy as _np

        try:
            out.update(
                gate_ranking_metrics(
                    _np.concatenate(gate_scores),
                    _np.concatenate(gate_changed),
                    _np.concatenate(gate_prefix) if len(gate_prefix) == len(gate_scores) else None,
                    topk=int((cfg or {}).get("gate_eval_topk", 20) or 20),
                )
            )
        except Exception as exc:  # noqa: BLE001 - metrics must never kill training
            logger.warning("gate ranking metrics failed: %s", exc)
    return out


def _gate_log(label: str, m: dict) -> str:
    if "gate_auroc_novel" not in m:
        return ""
    return (
        f" | {label} gate novel auroc={m['gate_auroc_novel']:.3f} "
        f"ap={m.get('gate_ap_novel', float('nan')):.4f} "
        f"rec@k={m.get('gate_recall_novel', float('nan')):.3f}"
    )


def _early_stop_combo(training_cfg: dict) -> dict[str, float]:
    """``training.early_stop_combo`` -> ``{metric_name: weight}`` (empty if unset)."""
    raw = training_cfg.get("early_stop_combo") or {}
    if OmegaConf.is_config(raw):
        raw = OmegaConf.to_container(raw, resolve=True) or {}
    return {str(k): float(v) for k, v in dict(raw).items()}


def train_loop(
    model: Pca2dSiteUnifiedV3,
    train_loader: DataLoader,
    stop_loader: DataLoader,
    test_loader: Optional[DataLoader],
    *,
    pca2d_name: str,
    device: torch.device,
    cfg,
    mode: str = "nll",
    epochs: int = 120,
    lr: float = 2e-4,
    weight_decay: float = 1e-2,
    grad_clip: float = 1.0,
    patience: int = 12,
    log_every: int = 5,
    save_dir: Path,
    early_stop_metric: str = "skill_mu",
    tb_writer=None,
    has_cal_split: bool = False,
    training_cfg: Optional[dict] = None,
    ckpt_basename: str = "Pca2dSiteUnifiedV3_best_model.pt",
    model_cls_name: str = "Pca2dSiteUnifiedV3",
) -> tuple[Pca2dSiteUnifiedV3, float]:
    training_cfg = training_cfg or {}
    u = _unified_cfg(cfg)
    mse_aux = float(u.get("mse_aux_weight", 0.0))
    all_step_loss = bool(cfg.get("all_step_loss", False))
    ema_decay = float(training_cfg.get("ema_decay", 0.999))
    warmup_steps = int(training_cfg.get("lr_warmup_steps", 0) or 0)

    params = list(model.trainable_parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(len(train_loader), 1)
    total_steps = steps_per_epoch * epochs
    global_step = 0
    if warmup_steps > 0:
        def _lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(max(warmup_steps, 1))
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=5
        )

    ema = ModelEMA(model, decay=ema_decay, trainable_only=True) if ema_decay > 0 else None
    higher_better = early_stop_metric.startswith(("skill", "gate_", "full_skill", "combo"))
    # An epoch only counts as an improvement if it beats the best score by ``min_delta`` (absolute,
    # in the metric's units; 0 = any strict improvement).  Cal skill_full moves by ~1e-3 between
    # epochs, so ~1e-3 keeps noise from resetting the patience counter.
    min_delta = float(training_cfg.get("min_delta", 0.0) or 0.0)
    combo = _early_stop_combo(training_cfg) if early_stop_metric == "combo" else None
    if early_stop_metric == "combo" and not combo:
        raise ValueError(
            "early_stop_metric='combo' needs training.early_stop_combo, e.g. "
            "{skill_full: 1.0, gate_ap_novel: 0.01}"
        )
    best_val = float("-inf") if higher_better else float("inf")
    best_state = None
    patience_counter = 0
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        _advance_train_subsample(train_loader.dataset, epoch)
        model.train()
        model.site_gpt.eval()
        tr_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in train_loader:
            xb, yb, sw, edge_phylo, horizon, substitution_feats, activation_target = (
                _unpack_v3_batch(batch, device, cfg=cfg)
            )
            if horizon is None:
                horizon = horizon_from_batch(batch, cfg=cfg, device=device)
            y_pca = yb[pca2d_name].to(device)
            y_mean = None
            u_cfg = _unified_cfg(cfg)
            uv4 = cfg.get("unified_v4") or {}
            if OmegaConf.is_config(uv4):
                uv4 = OmegaConf.to_container(uv4, resolve=True) or {}
            need_mean = bool(u_cfg.get("use_mean_residual", False)) or float(
                uv4.get("mean_aux_weight", 0.0) or 0.0
            ) > 0.0
            if PRIMARY_TARGET in yb and need_mean:
                y_mean = yb[PRIMARY_TARGET].to(device)
            loss, info = _forward_batch_loss(
                model,
                xb,
                y_pca,
                mode=mode,
                horizon=horizon,
                edge_phylo=edge_phylo,
                substitution_feats=substitution_feats,
                activation_target=activation_target,
                mse_aux=mse_aux,
                sample_weight=sw,
                all_step_loss=all_step_loss,
                y_mean=y_mean,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()
            if ema is not None:
                ema.update(model)
            if warmup_steps > 0:
                scheduler.step()
                global_step += 1
            tr_loss += float(loss.item())
            n_batches += 1

        avg_tr = tr_loss / max(1, n_batches)
        if ema is not None:
            ema.apply(model)
        try:
            stop_metrics = eval_v3_metrics(
                model, stop_loader, pca2d_name=pca2d_name, device=device, cfg=cfg, mode=mode
            )
            test_metrics = (
                eval_v3_metrics(
                    model, test_loader, pca2d_name=pca2d_name, device=device, cfg=cfg, mode=mode
                )
                if test_loader is not None and test_loader is not stop_loader
                else stop_metrics
            )
        finally:
            if ema is not None:
                ema.restore(model)

        if early_stop_metric == "combo":
            # weighted sum of cal metrics (higher is better), e.g. skill_full + 0.01 * gate_ap_novel
            stop_score = sum(
                w * float(stop_metrics.get(name, float("nan"))) for name, w in combo.items()
            )
            improved = stop_score > best_val + min_delta
        elif early_stop_metric == "skill_mu":
            stop_score = float(stop_metrics.get("skill_mu", float("nan")))
            improved = stop_score > best_val + min_delta
        elif early_stop_metric in ("skill_full", "full_skill"):
            stop_score = float(stop_metrics.get("skill_full", float("nan")))
            improved = stop_score > best_val + min_delta
        elif early_stop_metric.startswith("gate_"):
            # gate_auroc_novel | gate_ap_novel | gate_recall_novel | ..._all (higher is better)
            stop_score = float(stop_metrics.get(early_stop_metric, float("nan")))
            improved = stop_score > best_val + min_delta
        else:
            stop_score = float(stop_metrics.get("full_x0_mse", stop_metrics.get("full_mse", 1.0)))
            improved = stop_score < best_val - min_delta

        if improved:
            best_val = stop_score
            patience_counter = 0
            if ema is not None:
                ema.apply(model)
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
                if not k.startswith("site_gpt.") or "lora_" in k
            }
            if ema is not None:
                ema.restore(model)
            torch.save(
                {
                    "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "trainable_state_dict": best_state,
                    "hyperparameters": dict(model.hparams),
                    "model_cls_name": model_cls_name,
                    "epoch": epoch,
                    "best_val": best_val,
                    "mode": mode,
                },
                save_dir / ckpt_basename,
            )
        else:
            patience_counter += 1

        if warmup_steps <= 0:
            scheduler.step(-stop_score if higher_better else stop_score)

        split_label = "cal" if has_cal_split else "test"
        dt = time.time() - t0
        print(
            f"Epoch {epoch}/{epochs} ({dt:.1f}s) mode={mode} | "
            f"train loss={avg_tr:.6f} | {split_label} skill_mu="
            f"{stop_metrics.get('skill_mu', float('nan')):.4f} "
            f"skill_full={stop_metrics.get('skill_full', float('nan')):.4f} "
            f"full_x0={stop_metrics.get('full_x0_mse', float('nan')):.6f} | "
            f"test skill_mu={test_metrics.get('skill_mu', float('nan')):.4f} "
            f"test skill_full={test_metrics.get('skill_full', float('nan')):.4f} | "
            f"best({early_stop_metric})={best_val:.6f}"
            f"{' *' if improved else ''}"
            f"{_gate_log(split_label, stop_metrics)}{_gate_log('test', test_metrics) if test_metrics is not stop_metrics else ''}",
            flush=True,
        )
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    if best_state is not None:
        cur = model.state_dict()
        cur.update({k: v.to(device) for k, v in best_state.items()})
        model.load_state_dict(cur, strict=False)
    return model, best_val


# ============================================================================
# (merged from train_unified_pca2d_site_v4.py)
# ============================================================================

_DEFAULT_PREDICTOR = "configs/latent/predictors/residual_model_conscurr.yaml"
_DEFAULT_DATASET = (
    "configs/latent/datasets/by_nearest_ancestor/split_2022-12-01_k5_esm1_layer33.yaml"
)
_CKPT_BASENAME = "Pca2dSiteUnifiedV4_best_model.pt"


def _unified_v4_cfg(cfg) -> dict[str, Any]:
    u = cfg.get("unified_v4") or cfg.get("unified_v3") or {}
    if OmegaConf.is_config(u):
        return OmegaConf.to_container(u, resolve=True) or {}
    return dict(u)


def build_model_from_cfg(
    cfg,
    *,
    site_gpt_checkpoint: str,
    device: torch.device,
    conservation_stats: Optional[torch.Tensor] = None,
    data_cfg: Optional[Any] = None,
) -> Pca2dSiteUnifiedV4:
    site_gpt = load_frozen_site_gpt_anchor(site_gpt_checkpoint, device)
    u = _unified_v4_cfg(cfg)
    v4 = mean_pca2d_v4_cfg(cfg)
    edge_dim = 1 if bool(cfg.get("append_edge_phylo_to_head", False)) else 0
    cons_dim = 3 if bool(cfg.get("append_msa_conservation", False)) else 0
    subst_dim = int(PREFIX_FEATURE_DIM) if bool(cfg.get("append_msa_substitution", False)) else 0
    use_activation = bool(cfg.get("substitution_target_activation", False))
    horizon_dim = int(cfg.get("horizon_embed_dim", 4) or 0) if bool(
        cfg.get("horizon_embedding", True)
    ) else 0
    pca2d_name = str(cfg.get("pca2d_target_name", PCA2D_S1_TARGET))
    use_ridge = bool(u.get("use_ridge_readout", True))
    readout = resolve_ridge_readout(cfg, device=device, data_cfg=data_cfg) if use_ridge else None
    use_independent = bool(u.get("use_independent_mean_mlp", False))
    if use_ridge and not v4:
        logger.warning(
            "build_model_from_cfg: no mean_pca2d_v4 block in predictor YAML; "
            "ridge readout will be refit on the fly unless unified_v*.ridge is cached. "
            "Add mean_pca2d_v4.ridge_coef_cache/ridge_coef_path for reproducibility."
        )
    if bool(u.get("use_mean_residual", False)) and not use_independent:
        logger.warning(
            "YAML unified_*.use_mean_residual=true is ignored by Pca2dSiteUnifiedV4 "
            "(requires use_independent_mean_mlp=true)."
        )
    model = Pca2dSiteUnifiedV4(
        site_gpt,
        mean_readout=readout,
        consistency_weight=(
            float(v4.get("consistency_weight", u.get("consistency_weight", 0.2))) if use_ridge else 0.0
        ),
        use_independent_mean_mlp=use_independent,
        use_ridge_readout=use_ridge,
        stop_grad_pca2d_for_consistency=bool(
            v4.get("stop_grad_pca2d", u.get("stop_grad_pca2d", True))
        ),
        cond_dim=int(u.get("cond_dim", 256)),
        mem_dim=int(u.get("mem_dim", 128)),
        site_channels=int(u.get("site_channels", 96)),
        dit_layers=int(u.get("dit_layers", 4)),
        dit_heads=int(u.get("dit_heads", 4)),
        dropout=float(u.get("dropout", 0.1)),
        cov_rank=int(u.get("cov_rank", 4)),
        horizon_dim=horizon_dim,
        edge_phylo_dim=edge_dim,
        use_temporal_pca2d=bool(u.get("use_temporal_pca2d", True)),
        temporal_proj_dim=int(u.get("temporal_proj_dim", 16)),
        conservation_dim=cons_dim,
        substitution_dim=subst_dim,
        use_activation_supervision=use_activation,
        activation_aux_weight=float(u.get("activation_aux_weight", 0.5)),
        activation_head_input=str(u.get("activation_head_input", "site_feats")),
        activation_pos_weight=float(u.get("activation_pos_weight", 1.0)),
        use_mean_residual=use_independent,
        mean_aux_weight=float(u.get("mean_aux_weight", 0.2)),
        mean_dim=int(u.get("mean_dim")) if u.get("mean_dim") is not None else None,
        bridge_sigma=float(u.get("bridge_sigma", 0.1)),
        pca2d_name=pca2d_name,
    )
    if conservation_stats is not None:
        model.set_conservation_stats(conservation_stats)
    model.to(device)
    return model


def train_from_configs(
    cfg,
    data_cfg,
    *,
    site_gpt_checkpoint: str,
    tag: str,
    mode: str = "mse",
):
    set_seed(int(cfg.get("seed", 42)))
    if site_gpt_checkpoint:  # keep the anchor path in the config for downstream readers
        cfg["site_gpt_checkpoint"] = str(site_gpt_checkpoint)
    device = torch.device(resolve_device(cfg))
    bs = int(cfg.get("batch_size", 256))
    nw = int(cfg.get("num_workers", 0))
    train_loader, cal_loader, test_loader, pca2d_name, _ = build_v3_loaders(
        cfg, data_cfg, batch_size=bs, num_workers=nw
    )
    stop_loader = cal_loader if cal_loader is not None else test_loader

    site_gpt_tmp = load_frozen_site_gpt_anchor(site_gpt_checkpoint, device)
    seq_len = int(getattr(site_gpt_tmp, "pca2d_seq_len", 0) or 0)
    del site_gpt_tmp
    conservation = _load_conservation_stats(cfg, data_cfg, seq_len=seq_len)
    model = build_model_from_cfg(
        cfg,
        site_gpt_checkpoint=site_gpt_checkpoint,
        device=device,
        conservation_stats=conservation,
        data_cfg=data_cfg,
    )

    u = _unified_v4_cfg(cfg)
    gen_mode = str(u.get("generative_mode", "none") or "none")
    if mode == "nll" and gen_mode == "bridge_flow":
        mode = "flow"

    root = Path(str(cfg.get("root", ".")))
    if cfg.get("checkpoint_dir"):
        ckpt_dir = Path(str(cfg.get("checkpoint_dir")))
    else:
        ckpt_dir = root / "models" / "checkpoints" / "layer33" / f"unified_pca2d_site_v4_{tag}"
    log_dir = str(cfg.get("logs_dir") or (root / "logs"))
    tr_cfg = _training_cfg(cfg)
    es = cfg.get("early_stopping") or {}
    if OmegaConf.is_config(es):
        es = OmegaConf.to_container(es, resolve=True) or {}
    early_metric = str(tr_cfg.get("early_stop_metric") or es.get("metric", "skill_full"))
    patience = int(tr_cfg.get("patience") or es.get("patience", 12))

    with run_logging(log_dir, f"unified_pca2d_site_v4_{tag}") as (writer, _paths):
        print(
            f"v4 unified model trainable={model.n_trainable()} mode={mode} "
            f"consistency_weight={model.consistency_weight}\n"
            f"ckpt_dir={ckpt_dir}",
            flush=True,
        )
        model, best = train_loop(
            model,
            train_loader,
            stop_loader,
            test_loader if cal_loader is not None else None,
            pca2d_name=pca2d_name,
            device=device,
            cfg=cfg,
            mode=mode,
            epochs=int(cfg.get("epochs", 120)),
            lr=float(cfg.get("learning_rate", 2e-4)),
            weight_decay=float(cfg.get("weight_decay", 1e-2)),
            grad_clip=float(cfg.get("clip_grad_norm", 1.0)),
            patience=patience,
            save_dir=ckpt_dir,
            early_stop_metric=early_metric,
            tb_writer=writer,
            has_cal_split=cal_loader is not None,
            training_cfg=tr_cfg,
            ckpt_basename=_CKPT_BASENAME,
            model_cls_name="Pca2dSiteUnifiedV4",
        )
        print(f"Done best({early_metric})={best:.6f}", flush=True)
    return model, best


def main():
    parser = argparse.ArgumentParser(description="Train B4 unified pca2d site v4.")
    parser.add_argument("--predictor-config", default=_DEFAULT_PREDICTOR)
    parser.add_argument("--dataset-config", default=_DEFAULT_DATASET)
    parser.add_argument("--site-gpt-checkpoint", default=None)
    parser.add_argument("--tag", default="b2_b3_mean_res_readout")
    parser.add_argument("--mode", choices=("nll", "flow", "mse"), default="mse")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="OmegaConf dotlist override of the predictor YAML (repeatable)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    _setup_logging(args.log_level)
    cfg = OmegaConf.create(
        OmegaConf.to_container(load_config(args.predictor_config), resolve=True)
    )
    if args.set:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(args.set)))
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    data_cfg = load_config(args.dataset_config)
    site = args.site_gpt_checkpoint or str(cfg.get("site_gpt_checkpoint", ""))
    train_from_configs(
        cfg,
        data_cfg,
        site_gpt_checkpoint=site,
        tag=args.tag,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
