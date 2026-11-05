#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     series.py
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
import os
from collections.abc import Sized
from typing import Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

import logging

from recomb2027.data.encodings import EncodingMatcher
from recomb2027.data.extra_encoding_features import (
    concat_prefix_step_features,
    extra_stats_npz_keys,
    is_block_target,
    parse_block_names,
    parse_block_target_name,
    parse_extra_encoding_stats,
    scale_block_array,
    scale_extra_stat_array,
    stat_encodings_dir,
)
from recomb2027.data.spike_blocks import block_encodings_dir
from recomb2027.core.constants import PRIMARY_TARGET
from recomb2027.data.utils import joblib_load_numpy2_compat

logger = logging.getLogger(__name__)


# ------------------------------
# 0. General purpose functions
# ------------------------------
def _flatten(data):
    assert len(data.shape) > 1
    D = data.shape[-1]

    return data.reshape(-1, D)


def standard_scaler_transform_encoding_cube(
    enc: np.ndarray,
    scaler: StandardScaler,
) -> np.ndarray:
    """Apply a fitted ``StandardScaler.transform`` to the last axis of ``enc``.

    ``enc`` shape ``[..., D]``; transform uses rows ``(-1, D)`` matching
    sklearn's convention (same as ``SeriesEncodingsDataset._apply_scaler``).
    """
    if scaler is None:
        raise ValueError("scaler must not be None")
    if enc.ndim < 2:
        raise ValueError(f"Expected rank ≥2 encoding cube; got {enc.shape!r}")
    dims = enc.shape
    d = int(dims[-1])
    flat = enc.reshape(-1, d)
    out = scaler.transform(flat.astype(np.float64))
    return out.reshape(dims).astype(np.float32)


# ------------------------------
# 1. Base dataset for shared logic
# ------------------------------
class SeriesDataset(Dataset):
    """ Base series class that stores only gids (genome ids) of the series.
    """
    def __init__(self, series_gids, split="train"):
        self.sample_gids = series_gids
        self.split = split

    def __len__(self):
        return len(self.sample_gids) if self.sample_gids is not None else 0

    def __getitem__(self, idx):
        """ Return X, Y for genome ids for the series"""
        return self.sample_gids[idx][:-1], self.sample_gids[idx][1:]

    # ---------- Meta info ----------
    def get_block_size(self):
        if (self.sample_gids is not None) and len(self.sample_gids) > 0:
            return len(self.sample_gids[0])
        else:
            return 0

    def get_xy_gids(self):
        x_gid = np.stack(self.sample_gids)[:,:-1]
        y_gid = np.stack(self.sample_gids)[:,1:]

        return x_gid, y_gid

    def get_x_gids(self):
        x_gid = np.stack(self.sample_gids)[:,:-1]
        return x_gid

    def get_y_gids(self):
        y_gid = np.stack(self.sample_gids)[:,-1]
        return y_gid

    # ---------- Save / Load ----------
    def save_to_file(self, filepath):
        logger.info(f"Saving dataset to: '{filepath}'")
        np.savez_compressed(
            filepath,
            sample_gids=self.sample_gids,
            split = self.split
        )

    @classmethod
    def load_from_file(cls, filepath):
        logger.info(f"Loading dataset from: '{filepath}'")
        data = np.load(filepath, allow_pickle=True)
        obj = cls.__new__(cls)
        obj.sample_gids = data["sample_gids"]
        obj.split = data["split"]

        return obj


# ------------------------------
# 2. Dataset for prediction model
# ------------------------------
class SeriesEncodingsDataset(SeriesDataset):
    """ Class that works with the encodings of the series.
    It also stores information about the genome ids of the series."""
    def __init__(self,
                 series_gids,
                 sample_encodings=None,
                 df_encodings=None,
                 split="train",
                 target_mode="value",
                 input_feature_mode="absolute",
                 ):
        super().__init__(series_gids, split)
        # Assign true series of vectors
        self.sample_encodings = sample_encodings
        # Work with Ys (targets) as differences between time steps or absolute next-step values
        self.target_mode = target_mode
        # "absolute": model input is latent x_t. "gatt_plus": [x_t, v_t] with v_t = x_t - x_{t-1}
        # (same convention as ``train_gated_lstm.AugmentedDataset`` feature_mode ``diff``).
        self.input_feature_mode = input_feature_mode
        # Optional timeline subsampling (full-series indices). None = use all timepoints.
        self._timeline_indices = None
        self.epoch_subsample_schedule = None  # tuple of n_drop values, cycled each epoch (train only)
        # Optional ``(start, end)`` half-open channel slice applied only when building targets ``y``.
        # Use with fused inputs ``[..., :primary,...,:cross ]`` when ``y`` should live in primary latent
        # (e.g. layer 33 only) while ``x`` consumes the whole fused vector.
        self.target_primary_slice: Optional[Tuple[int, int]] = None
        # Optional per-series consecutive edge distances ``(N, K-1)`` from MSA (BLOSUM / phylo).
        self.edge_distances: Optional[np.ndarray] = None
        self.append_edge_phylo_distance: bool = False
        self.edge_phylo_use_log1p: bool = True
        # Optional extra ESM stat blocks (std, median, argmax_pos, …) per timestep.
        self.extra_encoding_stats: list[str] = []
        self.extra_stat_encodings: Optional[dict[str, np.ndarray]] = None
        # Optional domain block mean encodings (NTD, RBD_a, …) per timestep.
        self.block_encoding_names: list[str] = []
        self.block_encodings: Optional[dict[str, np.ndarray]] = None
        self.block_pooling: str = "average_token"
        self.block_scaler: Optional[StandardScaler] = None
        # Multi-target diff prediction (Phase 1 JEPA); always includes ``mean``.
        self.prediction_targets: list[str] = [PRIMARY_TARGET]
        # When True, ``__getitem__`` returns ``(x, y, jepa_abs)`` for train collate.
        self.jepa_return_abs: bool = False
        # Match encodings to the genome ids
        if df_encodings is not None:
            self._match_encodings(df_encodings)
        # Scaler
        self.scaler = None

    @staticmethod
    def _concat_velocity_features(x_abs: np.ndarray) -> np.ndarray:
        velocity = np.zeros_like(x_abs)
        if x_abs.shape[0] > 1:
            velocity[1:] = x_abs[1:] - x_abs[:-1]
        return np.concatenate([x_abs, velocity], axis=-1).astype(np.float32)

    def _active_timeline_indices(self, sub_len: int):
        active = getattr(self, "_timeline_indices", None)
        if active is None:
            return list(range(sub_len))
        return list(active)

    def _input_step_edge_distances(self, idx: int, active_indices) -> np.ndarray:
        """Per input step ``t``: phylo distance on the same edge as ``v_t`` — ``(t−1) → t``.

        ``edge_distances[idx, k]`` stores the consecutive distance ``k → k+1`` on the
        full series. For active timeline indices ``a[0], …, a[T]`` and input step
        ``t ∈ {0,…,T−1}`` we use ``d_0 = 0`` and for ``t ≥ 1`` sum
        ``edge_distances[idx, a[t−1] : a[t]]`` (aggregate over dropped interior steps).
        """
        if self.edge_distances is None:
            raise RuntimeError("edge_distances is None; call attach_edge_phylo_from_msa first")
        T = len(active_indices) - 1
        out = np.zeros(T, dtype=np.float32)
        row = self.edge_distances[idx]
        for t in range(1, T):
            lo, hi = active_indices[t - 1], active_indices[t]
            out[t] = float(row[lo:hi].sum())
        return out

    def get_primary_latent_dim(self) -> int:
        """Mean-embedding width (target / velocity / scaler dimension)."""
        if self.sample_encodings is None:
            return 0
        return int(self.sample_encodings.shape[-1])

    def get_extra_encoding_stats(self) -> list[str]:
        return list(getattr(self, "extra_encoding_stats", None) or [])

    def _scale_mean_cube(self, data: np.ndarray, scaler, *, fit: bool = False) -> np.ndarray:
        return self._apply_scaler(data, scaler, fit=fit)

    def _scale_extra_cube(
        self, stat: str, data: np.ndarray, scaler, *, fit: bool = False
    ) -> np.ndarray:
        return scale_extra_stat_array(data, stat, scaler, fit=fit)

    def _active_extra_prefix_steps(
        self, idx: int, active_indices, scaled_extra: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Deprecated helper; kept for callers that pass full-series extra cubes."""
        out: dict[str, np.ndarray] = {}
        for stat, cube in scaled_extra.items():
            row = cube[idx] if cube.ndim == 3 else cube
            if getattr(self, "_timeline_indices", None) is not None:
                row = row[np.asarray(active_indices, dtype=np.int64)]
            out[stat] = row[:-1].astype(np.float32)
        return out

    def _build_model_input_features(
        self,
        x_abs: np.ndarray,
        idx: int,
        active_indices,
        extra_prefix: Optional[dict[str, np.ndarray]] = None,
        block_prefix: Optional[dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        edge_feat = None
        if getattr(self, "append_edge_phylo_distance", False):
            edge_d = self._input_step_edge_distances(idx, active_indices)
            if getattr(self, "edge_phylo_use_log1p", True):
                edge_feat = np.log1p(edge_d, dtype=np.float32)
            else:
                edge_feat = edge_d.astype(np.float32)

        return concat_prefix_step_features(
            x_abs,
            extra_prefix,
            self.get_extra_encoding_stats(),
            input_feature_mode=getattr(self, "input_feature_mode", "absolute"),
            edge_feat=edge_feat,
            block_steps=block_prefix,
            block_names=self.get_block_encoding_names(),
        )

    def reset_timeline_indices(self):
        """Use the full series timeline (disables random middle-step drops)."""
        self._timeline_indices = None

    def set_timeline_random_drop(self, n_drop: int):
        """Drop ``n_drop`` random interior timepoints (keep ends), shared across samples.

        Mirrors ``AugmentedDataset.subsample_steps`` in ``train_gated_lstm.py``.
        """
        K = int(self.sample_encodings.shape[1])
        if n_drop <= 0 or K - n_drop < 2:
            self._timeline_indices = list(range(K))
            return
        middle = list(range(1, K - 1))
        nd = min(int(n_drop), len(middle))
        drop = set(np.random.choice(middle, nd, replace=False))
        self._timeline_indices = sorted(set(range(K)) - drop)

    def advance_epoch_subsample(self, epoch: int):
        """Apply subsampling schedule for this epoch (1-based), matching
        ``train_model_augmented`` indexing: ``n_drop = schedule[(epoch-1) % len]``."""
        sched = self.epoch_subsample_schedule
        if not sched:
            return
        i = (epoch - 1) % len(sched)
        self.set_timeline_random_drop(int(sched[i]))

    def get_block_encoding_names(self) -> list[str]:
        return list(getattr(self, "block_encoding_names", None) or [])

    def get_prediction_targets(self) -> list[str]:
        return list(getattr(self, "prediction_targets", [PRIMARY_TARGET]))

    def _absolute_stat_trajectory(
        self,
        stat: str,
        sub_mean: np.ndarray,
        sub_extra_row: dict[str, np.ndarray],
        sub_block_row: Optional[dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        if stat == PRIMARY_TARGET:
            traj = sub_mean.astype(np.float32)
            if getattr(self, "target_primary_slice", None) is not None:
                s, e = self.target_primary_slice
                traj = traj[..., s:e]
            return traj
        if is_block_target(stat):
            bname = parse_block_target_name(stat)
            blocks = sub_block_row or {}
            if bname not in blocks:
                raise KeyError(
                    f"prediction target {stat!r} missing from block encodings "
                    f"(available: {list(blocks.keys())})"
                )
            return blocks[bname].astype(np.float32)
        if stat not in sub_extra_row:
            raise KeyError(
                f"prediction target {stat!r} missing from extra_stat_encodings "
                f"(available: {list(sub_extra_row.keys())})"
            )
        return sub_extra_row[stat].astype(np.float32)

    def _build_multitarget_y(
        self,
        sub_mean: np.ndarray,
        sub_extra_row: dict[str, np.ndarray],
        sub_block_row: Optional[dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        parts = []
        for stat in self.get_prediction_targets():
            traj = self._absolute_stat_trajectory(
                stat, sub_mean, sub_extra_row, sub_block_row=sub_block_row
            )
            y_prefix = traj[:-1]
            y_suffix = traj[1:]
            if self.target_mode == "diff":
                parts.append(y_suffix - y_prefix)
            else:
                parts.append(y_suffix.astype(np.float32))
        if len(parts) == 1:
            return parts[0]
        return np.stack(parts, axis=1).astype(np.float32)

    def _build_jepa_absolute_targets(
        self,
        sub_mean: np.ndarray,
        sub_extra_row: dict[str, np.ndarray],
        stats: Sequence[str],
        sub_block_row: Optional[dict[str, np.ndarray]] = None,
    ) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for stat in stats:
            traj = self._absolute_stat_trajectory(
                stat, sub_mean, sub_extra_row, sub_block_row=sub_block_row
            )
            out[stat] = traj[:-1].astype(np.float32)
        return out

    def __getitem__(self, idx):
        mean_row = self.sample_encodings[idx]
        if self.scaler is not None:
            mean_row = self._scale_mean_cube(mean_row, self.scaler, fit=False)
        else:
            mean_row = mean_row.astype(np.float32)

        scaled_extra_row: dict[str, np.ndarray] = {}
        if self.extra_stat_encodings:
            for stat in self.get_extra_encoding_stats():
                cube = self.extra_stat_encodings[stat][idx]
                scaled_extra_row[stat] = self._scale_extra_cube(
                    stat, cube, self.scaler, fit=False,
                )

        scaled_block_row: dict[str, np.ndarray] = {}
        if self.block_encodings:
            for name in self.get_block_encoding_names():
                cube = self.block_encodings[name][idx]
                scaled_block_row[name] = self._scale_block_cube(cube, fit=False)

        active = getattr(self, "_timeline_indices", None)

        if active is None:
            sub_mean = mean_row
            sub_extra_row = scaled_extra_row
            sub_block_row = scaled_block_row
        else:
            active_idx = np.asarray(active, dtype=np.int64)
            sub_mean = mean_row[active_idx]
            sub_extra_row = {s: v[active_idx] for s, v in scaled_extra_row.items()}
            sub_block_row = {s: v[active_idx] for s, v in scaled_block_row.items()}

        active_indices = self._active_timeline_indices(sub_mean.shape[0])
        x_abs_full = sub_mean[:-1].astype(np.float32)
        extra_prefix = (
            {s: sub_extra_row[s][:-1].astype(np.float32) for s in self.get_extra_encoding_stats()}
            if self.get_extra_encoding_stats()
            else None
        )
        block_prefix = (
            {s: sub_block_row[s][:-1].astype(np.float32) for s in self.get_block_encoding_names()}
            if self.get_block_encoding_names()
            else None
        )

        traj_for_y = (
            sub_mean[..., self.target_primary_slice[0] : self.target_primary_slice[1]].astype(
                np.float32
            )
            if getattr(self, "target_primary_slice", None) is not None
            else sub_mean.astype(np.float32)
        )
        if len(self.get_prediction_targets()) > 1:
            y = self._build_multitarget_y(sub_mean, sub_extra_row, sub_block_row=sub_block_row)
        else:
            y_prefix = traj_for_y[:-1]
            y_suffix = traj_for_y[1:]
            if self.target_mode == "diff":
                y = y_suffix - y_prefix
            else:
                y = y_suffix.astype(np.float32)

        x = self._build_model_input_features(
            x_abs_full,
            idx,
            active_indices,
            extra_prefix=extra_prefix,
            block_prefix=block_prefix,
        )

        if getattr(self, "jepa_return_abs", False):
            jepa_stats = getattr(self, "jepa_absolute_stats", None) or (PRIMARY_TARGET,)
            jepa_abs = self._build_jepa_absolute_targets(
                sub_mean, sub_extra_row, jepa_stats, sub_block_row=sub_block_row
            )
            return x, y, jepa_abs

        return x, y

    def _encoding_dict_from_dir(self, encodings_dir, encoding_key="encodings", gid_key="genome_id"):
        df_encodings = EncodingMatcher.read_from_dir(
            encodings_dir, encoding_key=encoding_key,
        ).set_index(gid_key)
        return df_encodings["encodings"].to_dict()

    def _match_encodings(self, df_encodings):
        encoding_dict = df_encodings["encodings"].to_dict()
        # Match encodings to genome ids
        self.sample_encodings = np.stack([
            np.stack([encoding_dict[gid] for gid in row]) for row in self.sample_gids
        ])

    def match_extra_encodings_from_dir(
        self,
        primary_encodings_dir: str,
        extra_stats: Sequence[str],
    ) -> None:
        """Load optional per-stat encoding cubes aligned with ``sample_gids``."""
        stats = parse_extra_encoding_stats(extra_stats)
        if not stats:
            self.extra_encoding_stats = []
            self.extra_stat_encodings = None
            return
        cubes: dict[str, np.ndarray] = {}
        for stat in stats:
            stat_dir = stat_encodings_dir(primary_encodings_dir, stat)
            logger.info("Matching extra encoding stat %s from %s", stat, stat_dir)
            enc_dict = self._encoding_dict_from_dir(str(stat_dir))
            cubes[stat] = np.stack([
                np.stack([enc_dict[gid] for gid in row]) for row in self.sample_gids
            ]).astype(np.float32)
        self.extra_encoding_stats = stats
        self.extra_stat_encodings = cubes

    def match_block_encodings_from_dir(
        self,
        primary_encodings_dir: str,
        block_names: Sequence[str],
        *,
        block_pooling: str = "average_token",
    ) -> None:
        """Load optional per-block mean encoding cubes aligned with ``sample_gids``."""
        names = parse_block_names(block_names)
        if not names:
            self.block_encoding_names = []
            self.block_encodings = None
            return
        cubes: dict[str, np.ndarray] = {}
        for name in names:
            block_dir = block_encodings_dir(
                primary_encodings_dir, name, block_pooling=block_pooling
            )
            logger.info("Matching block encoding %s from %s", name, block_dir)
            enc_dict = self._encoding_dict_from_dir(str(block_dir))
            cubes[name] = np.stack([
                np.stack([enc_dict[gid] for gid in row]) for row in self.sample_gids
            ]).astype(np.float32)
        self.block_encoding_names = names
        self.block_encodings = cubes
        self.block_pooling = str(block_pooling)

    def get_latent_dim(self):
        if self.sample_encodings is not None:
            return self.sample_encodings.shape[-1]
        else:
            return 0

    def match_encodings_from_dir(self, encodings_dir, encoding_key="encodings", gid_key="genome_id"):
        """ Load and match encodings from directory."""
        df_encodings = EncodingMatcher.read_from_dir(encodings_dir, encoding_key=encoding_key).set_index(gid_key)
        self._match_encodings(df_encodings)

    def get_x(self):
        return self.sample_encodings[:, :-1]

    def get_y(self):
        return self.sample_encodings[:, -1]

    def get_xy(self):
        return self.sample_encodings[:, :-1], self.sample_encodings[:, 1:]

    def get_xy_scaled(self):
        sample_encodings = self._apply_scaler(
            data = self.sample_encodings, scaler = self.scaler, fit = False)

        return sample_encodings[:, :-1], sample_encodings[:, 1:]

    def remove_by_index(self, ids_list):
        """ Remove samples with specified index from the dataset. """
        if self.sample_encodings.shape[0] != len(self.sample_gids):
            raise ValueError("Embedding dimensions of sample_gids and sample_encodings must match.")

        n_samples = self.sample_encodings.shape[0]

        mask = np.ones(n_samples, dtype=bool)
        mask[ids_list] = False

        self.sample_encodings = self.sample_encodings[mask]
        self.sample_gids = [gid for i, gid in enumerate(self.sample_gids) if i not in set(ids_list)]
        if self.extra_stat_encodings:
            self.extra_stat_encodings = {
                stat: enc[mask] for stat, enc in self.extra_stat_encodings.items()
            }
        if self.block_encodings:
            self.block_encodings = {
                name: enc[mask] for name, enc in self.block_encodings.items()
            }

        # ---------- Scale helpers ----------

    def _scale_block_cube(self, data: np.ndarray, *, fit: bool = False) -> np.ndarray:
        scaler = self.block_scaler if self.block_scaler is not None else self.scaler
        return scale_block_array(data, scaler, fit=fit)

    def fit_scaler(self, scaler=None):
        """Fit mean scaler on primary encodings and block scaler on block cubes."""
        # Add scaler
        self.scaler = scaler if scaler is not None else StandardScaler()
        # Data used to fit scaler
        xb = self.sample_encodings[:,:-1]
        assert len(xb.shape) == 3
        B, T, D = xb.shape
        flat = xb.reshape(-1, D)
        self.scaler.fit(flat)

        self.block_scaler = None
        if self.block_encodings and self.get_block_encoding_names():
            self.block_scaler = StandardScaler()
            parts = [
                self.block_encodings[name].reshape(-1, D)
                for name in self.get_block_encoding_names()
            ]
            block_flat = np.concatenate(parts, axis=0)
            self.block_scaler.fit(block_flat)

    def _apply_scaler(self, data, scaler, fit=False):
        """ Apply provided scaler to the provided data."""
        dims = data.shape
        flat = data.reshape(-1, dims[-1])
        if scaler is not None:
            flat = scaler.fit_transform(flat) if fit else scaler.transform(flat)
        return flat.reshape(*dims)

    def scale_x(self, fit=False):
        if self.scaler is not None:
            return self._apply_scaler(self.sample_encodings[:, :-1], scaler=self.scaler, fit=fit)
        return None

    def scale_y(self):
        if self.scaler is not None:
            return self._apply_scaler(self.sample_encodings[:, 1:], scaler=self.scaler, fit=False)
        return None

    def scale_xy(self):
        if self.scaler is not None:
            self.sample_encodings = self._apply_scaler(
                data=self.sample_encodings, scaler=self.scaler, fit=False)
        else:
            raise ValueError("No scaler available")

    def inverse_scaling(self, data):
        if self.scaler is not None:
            return self.scaler.inverse_transform(data)
        else:
            raise ValueError("No scaler available")

    # ---------- Transform helpers ----------
    def _apply_transform(self, data, func):
        assert len(data.shape) == 3
        B, T, D = data.shape
        flat = data.reshape(-1, D)
        flat = func(flat)

        return flat.reshape(B, T, D)

    def transform_x(self, transform_func):
        return self._apply_transform(self.sample_encodings[:, :-1], transform_func)

    def transform_y(self, transform_func):
        return self._apply_transform(self.sample_encodings[:, 1:], transform_func)

    # ---------- Save / Load ----------
    def save_to_file(self, filepath, save_scaler=True, save_split=True):
        logger.info(f"Saving dataset to: '{filepath}'")
        payload = {
            "sample_gids": self.sample_gids,
            "sample_encodings": self.sample_encodings,
        }
        if save_split:
            payload["split"] = self.split
        stats = self.get_extra_encoding_stats()
        if stats and self.extra_stat_encodings:
            payload["extra_encoding_stats"] = np.array(stats, dtype=object)
            for stat in stats:
                payload[f"extra_enc_{stat}"] = self.extra_stat_encodings[stat]
        blocks = self.get_block_encoding_names()
        if blocks and self.block_encodings:
            payload["block_encoding_names"] = np.array(blocks, dtype=object)
            payload["block_pooling"] = np.array(self.block_pooling, dtype=object)
            for name in blocks:
                payload[f"block_enc_{name}"] = self.block_encodings[name]
        np.savez_compressed(filepath, **payload)
        # Serialize scaler
        if save_scaler and self.scaler is not None:
            logger.info(f"Saving scaler to: '{filepath}.scaler'")
            joblib.dump(self.scaler, f"{filepath}.scaler")
        if save_scaler and self.block_scaler is not None:
            logger.info(f"Saving block scaler to: '{filepath}.block_scaler'")
            joblib.dump(self.block_scaler, f"{filepath}.block_scaler")

    @classmethod
    def load_from_file(cls, filepath, split=None, load_scaler=True):
        logger.info(f"Loading dataset from: '{filepath}'")
        data = np.load(filepath, allow_pickle=True)
        obj = cls.__new__(cls)
        obj.sample_gids = data["sample_gids"]
        obj.sample_encodings = data["sample_encodings"]
        if (split is None) and ("split" in data):
            obj.split = data["split"]
        else:
            obj.split = split
        obj.scaler = (
            joblib_load_numpy2_compat(f"{filepath}.scaler")
            if load_scaler and os.path.exists(f"{filepath}.scaler")
            else None
        )
        obj.target_mode = "value"
        obj.input_feature_mode = "absolute"
        obj.extra_encoding_stats = []
        obj.extra_stat_encodings = None
        obj.block_encoding_names = []
        obj.block_encodings = None
        obj.block_pooling = "average_token"
        obj.block_scaler = None
        block_scaler_path = f"{filepath}.block_scaler"
        if load_scaler and os.path.exists(block_scaler_path):
            obj.block_scaler = joblib_load_numpy2_compat(block_scaler_path)
        if "extra_encoding_stats" in data:
            stats = [str(s) for s in np.asarray(data["extra_encoding_stats"]).tolist()]
            obj.extra_encoding_stats = parse_extra_encoding_stats(stats)
            cubes = {}
            for stat in obj.extra_encoding_stats:
                key = f"extra_enc_{stat}"
                if key not in data:
                    raise KeyError(f"NPZ missing {key!r} for extra_encoding_stats")
                cubes[stat] = data[key].astype(np.float32)
            obj.extra_stat_encodings = cubes
        if "block_encoding_names" in data:
            blocks = [str(s) for s in np.asarray(data["block_encoding_names"]).tolist()]
            obj.block_encoding_names = parse_block_names(blocks)
            obj.block_pooling = str(np.asarray(data.get("block_pooling", "average_token")).item())
            bcubes = {}
            for name in obj.block_encoding_names:
                key = f"block_enc_{name}"
                if key not in data:
                    raise KeyError(f"NPZ missing {key!r} for block_encoding_names")
                bcubes[name] = data[key].astype(np.float32)
            obj.block_encodings = bcubes
        return obj


def _dataset_gid_row_tuple(row):
    """Hashable genome-id row (one series timestep path)."""
    return tuple(str(x) for x in np.asarray(row, dtype=object).ravel().tolist())


def _reorder_secondary_to_primary(primary_gids, secondary_gids, secondary_encodings: np.ndarray) -> np.ndarray:
    """Return ``secondary_encodings`` rows permuted so row *i* matches ``primary_gids[i]*."""
    pg = [_dataset_gid_row_tuple(primary_gids[i]) for i in range(primary_gids.shape[0])]
    sg = {_dataset_gid_row_tuple(secondary_gids[i]): i for i in range(secondary_gids.shape[0])}
    idx = []
    missing = []
    for k in pg:
        j = sg.get(k)
        if j is None:
            missing.append(k)
        idx.append(j)
    if missing:
        raise ValueError(
            f"{len(missing)} primary series gid rows missing in secondary NPZ "
            f"(showing first 2): {missing[:2]}"
        )
    return secondary_encodings[np.array(idx, dtype=np.int64)]


def fuse_series_concat_secondary_npz_inplace(
    primary_ds: SeriesEncodingsDataset,
    secondary_npz_path: Union[str, os.PathLike],
    *,
    standardize_streams_before_concat: bool = False,
) -> None:
    """Append secondary-layer latents to each primary row (channels last).

    Both NPZs must use the **same phylogenetic series orderings** modulo row shuffle;
    rows are aligned by ``sample_gids``.

    By default, raw ``sample_encodings`` arrays are concatenated and ``primary_ds.scaler``
    is cleared so ``__getitem__`` does not transform (historic ``gpt_cross_stream`` behaviour).

    When ``standardize_streams_before_concat`` is ``True``, the **primary** block is transformed
    with ``primary_ds.scaler`` (fit on primary train NPZ), the secondary block with the scaler
    stored next to ``secondary_npz_path``, **then** concatenated. Resulting fused tensors carry
    per-stream z-scores; ``primary_ds.scaler`` is still cleared afterward to avoid double
    transformation in ``__getitem__``. Typical: primary = layer33 ``[N,T,D₁]``, secondary = layer12
    ``[N,T,D₂]`` ⇒ fused ``[N,T,D₁+D₂]``. With ``input_feature_mode: gatt_plus`` the dataloader
    builds velocities on those fused channels.
    """
    sec_ds = SeriesEncodingsDataset.load_from_file(str(secondary_npz_path), load_scaler=True)
    enc_p = np.asarray(primary_ds.sample_encodings, dtype=np.float32)
    pri_scaler = primary_ds.scaler
    gid_p = primary_ds.sample_gids
    enc_s = np.asarray(sec_ds.sample_encodings, dtype=np.float32)
    gid_s = sec_ds.sample_gids

    enc_s_ord = _reorder_secondary_to_primary(gid_p, gid_s, enc_s)
    if enc_p.ndim != 3 or enc_s_ord.ndim != 3:
        raise ValueError(
            f"Expected rank-3 sample_encodings; got primary {enc_p.shape} secondary {enc_s_ord.shape}"
        )
    if enc_p.shape[:-1] != enc_s_ord.shape[:-1]:
        raise ValueError(
            f"(N,T) mismatch primary {enc_p.shape[:-1]} vs secondary {enc_s_ord.shape[:-1]}"
        )

    if standardize_streams_before_concat:
        if pri_scaler is None:
            raise ValueError(
                "standardize_streams_before_concat=True requires primary_ds.scaler "
                "(load primary NPZ with load_scaler=True and ensure train_dataset.npz.scaler exists)."
            )
        if sec_ds.scaler is None:
            raise ValueError(
                f"standardize_streams_before_concat=True requires a scaler next to {secondary_npz_path!r}"
            )
        enc_p = standard_scaler_transform_encoding_cube(enc_p, pri_scaler)
        enc_s_ord = standard_scaler_transform_encoding_cube(enc_s_ord, sec_ds.scaler)
        logger.info(
            "Cross-stream fuse: standardized primary and secondary cubes with respective "
            "train scalers before channel concat (%s)",
            secondary_npz_path,
        )

    fused = np.concatenate([enc_p.astype(np.float32), enc_s_ord.astype(np.float32)], axis=-1)
    primary_ds.sample_encodings = fused
    primary_ds.scaler = None
    logger.info(
        "Fused secondary latent file %s → sample_encodings shape %s "
        "(per-timestep width %s; scaler cleared on fused dataset)",
        secondary_npz_path,
        fused.shape,
        fused.shape[-1],
    )


class SeriesTransformedEncodingsDataset(SeriesEncodingsDataset):
    """ Specialized class that works with the encodings of the series
    and the transformations done on the X part of the series.
    """
    def __init__(self,
                 series_gids,
                 df_encodings=None,
                 split="train",
                 target_mode="value",
                 input_feature_mode="absolute",
                 ):
        super().__init__(series_gids, df_encodings, split, target_mode,
                         input_feature_mode=input_feature_mode)
        # Transformed encodings
        self.sample_encodings_transformed_x = None
        self.sample_encodings_transformed_y = None
        # Transformation functions
        self.ext_scaler = None

    def __getitem__(self, idx):
        if self.scaler is not None:
            # Rescale sample is scaler is set up
            sample = self._apply_scaler(
                data = self.sample_encodings[idx],
                scaler = self.scaler,
                fit = False)
        else:
            sample = self.sample_encodings[idx]

        active = getattr(self, "_timeline_indices", None)
        if active is None:
            sub = sample
            tx = self.sample_encodings_transformed_x[idx]
        else:
            a = np.asarray(active, dtype=np.int64)
            sub = sample[a]
            inp_rows = a[:-1]
            tx = self.sample_encodings_transformed_x[idx][inp_rows]

        active_indices = (
            list(range(sub.shape[0]))
            if active is None
            else list(active)
        )

        sample_x_orig = sub[:-1]
        sample_y = sub[1:]
        if self.target_mode == "diff":
            y = (sample_y - sample_x_orig).astype(np.float32)
        else:
            y = sample_y.astype(np.float32)

        x_abs = tx.astype(np.float32)
        x = self._build_model_input_features(x_abs, idx, active_indices)

        return x, y

    def adjust_encodings_x(self, transform_func, ext_scaler=None):
        """ Adjust the encoding of the X part of the series.

        The following chain of transformations will be applied:
        1. Transform with internal scaler (must be already fitted).
        2. Apply external transformation function.
        3. Fit extra scaler on the transformed data (in case of train split).
        4. Transform the fitted scaler.
        """
        # Rescale with existing scaler
        encodings_scaled = self._apply_scaler(
            self.sample_encodings[:, :-1], self.scaler, fit=False)
        # Transform
        encodings_transformed = self._apply_transform(encodings_scaled, transform_func)
        # Fit (for train split) and rescale again the transformed data
        if self.split == "train":
            self.ext_scaler = ext_scaler if ext_scaler is not None else StandardScaler()
            self.sample_encodings_transformed_x = self._apply_scaler(
                encodings_transformed, self.ext_scaler, fit=True)
        else:
            if ext_scaler is None:
                raise ValueError("Extra scaler not available")
            self.ext_scaler = ext_scaler
            self.sample_encodings_transformed_x = self._apply_scaler(
                encodings_transformed, self.ext_scaler, fit=False)

    def get_transformed_x(self):
        return self.sample_encodings_transformed_x

    # ---------- Save / Load ----------
    def save_to_file(self, filepath, save_scaler=True):
        logger.info(f"Saving dataset to: '{filepath}'")
        np.savez_compressed(
            filepath,
            sample_gids=self.sample_gids,
            sample_encodings=self.sample_encodings,
            sample_encodings_transformed_x = self.sample_encodings_transformed_x,
            sample_encodings_transformed_y = self.sample_encodings_transformed_y,
            split=self.split
        )
        if save_scaler and self.scaler is not None:
            joblib.dump(self.scaler, f"{filepath}.scaler")
        if save_scaler and self.ext_scaler is not None:
            joblib.dump(self.ext_scaler, f"{filepath}.ext_scaler")

    @classmethod
    def load_from_file(cls, filepath, split=None, load_scaler=True):
        logger.info(f"Loading dataset from: '{filepath}'")
        data = np.load(filepath, allow_pickle=True)
        obj = cls.__new__(cls)
        obj.sample_gids = data["sample_gids"]
        obj.sample_encodings = data["sample_encodings"]
        if "sample_encodings_transformed_x" in data:
            obj.sample_encodings_transformed_x = data["sample_encodings_transformed_x"]
        else:
            obj.sample_encodings_transformed_x = None
        if "sample_encodings_transformed_y" in data:
            obj.sample_encodings_transformed_y = data["sample_encodings_transformed_y"]
        else:
            obj.sample_encodings_transformed_y = None
        if (split is None) and ("split" in data):
            obj.split = data["split"]
        else:
            obj.split = split
        obj.scaler = (
            joblib_load_numpy2_compat(f"{filepath}.scaler")
            if load_scaler and os.path.exists(f"{filepath}.scaler")
            else None
        )
        obj.ext_scaler = (
            joblib_load_numpy2_compat(f"{filepath}.ext_scaler")
            if load_scaler and os.path.exists(f"{filepath}.ext_scaler")
            else None
        )

        return obj


class VectorAsSequenceDataset(Dataset):
    """
    Dataset wrapper that converts flat vector encodings into
    sequence format suitable for sequence models (e.g. BiLSTM).

    Each sample is returned as:
        X: [embedding_dim, 1]
        Y: [embedding_dim, 1]  (or X if reconstruct_self=True)
    """

    def __init__(
        self,
        dataset: SeriesEncodingsDataset,
        *,
        split: str,
        reconstruct_self: bool = False,
    ) -> None:
        super().__init__()

        if split not in {"train", "val"}:
            raise ValueError(f"Invalid split '{split}'. Expected 'train' or 'val'.")

        self.reconstruct_self = reconstruct_self

        # Build tensors
        self.X, self.Y = self._build_tensors(dataset, split)

    # ------------------------------------------------------------------
    # Dataset construction
    # ------------------------------------------------------------------
    def _build_tensors(
        self,
        dataset: SeriesEncodingsDataset,
        split: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Construct input (X) and target (Y) tensors.

        Returns:
            X: Tensor of shape [n_samples, embedding_dim, 1]
            Y: Tensor of shape [n_samples, embedding_dim, 1]
        """
        encodings = dataset.sample_encodings

        if split == "train":
            x_raw = encodings[:, :-1]
            y_raw = encodings[:, :-1] if self.reconstruct_self else encodings[:, 1:]

        else:  # validation
            if encodings.shape[-1] <= 1:
                raise ValueError("Series length must be > 1 for validation split")

            x_raw = encodings[:, -1] if self.reconstruct_self else encodings[:, -2]
            y_raw = x_raw if self.reconstruct_self else encodings[:, -1]

        X = self._to_sequence_tensor(x_raw)
        Y = self._to_sequence_tensor(y_raw)

        return X, Y

    @staticmethod
    def _to_sequence_tensor(array) -> torch.Tensor:
        """
        Convert flattened encodings into a tensor of shape:
            [n_samples, embedding_dim, 1]
        """
        tensor = torch.as_tensor(_flatten(array), dtype=torch.float32)
        return tensor.unsqueeze(-1)

    # ------------------------------------------------------------------
    # PyTorch Dataset API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            X[idx], Y[idx] each of shape [embedding_dim, 1]
        """
        return self.X[idx], self.Y[idx]
