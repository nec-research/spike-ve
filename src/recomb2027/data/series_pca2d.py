#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     series_pca2d.py
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
"""Series dataset with mean + pca2d_s1 cubes for JEPA-GPT training."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from recomb2027.core.constants import PRIMARY_TARGET
from recomb2027.data.utils import joblib_load_numpy2_compat
from recomb2027.data.series import SeriesEncodingsDataset
from recomb2027.latent.data_series.pca2d_features import (
    PCA2D_S1_TARGET,
    build_pca2d_cube_for_series,
    concat_mean_pca2d_prefix_features,
    is_pca2d_target,
    load_pca2d_basis,
)

logger = logging.getLogger(__name__)


class SeriesPca2dEncodingsDataset(SeriesEncodingsDataset):
    """Extends ``SeriesEncodingsDataset`` with a pca2d flat cube and dual inputs.

    ``__getitem__`` returns:
      - without JEPA: ``(x, y_dict)`` where ``y_dict[target]`` is ``[T, D_t]``
      - with JEPA: ``(x, y_dict, jepa_abs_dict)``
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pca2d_encoding_names: list[str] = []
        self.pca2d_encodings: Optional[dict[str, np.ndarray]] = None
        self.pca2d_scaler: Optional[StandardScaler] = None
        self.pca2d_basis_path: Optional[str] = None
        self.pca2d_flat_dim: int = 0
        self.pca2d_gatt_plus: bool = True
        self.return_dict_targets: bool = True
        self.pca2d_scaler_mode: str = "flat"  # flat | per_pc
        self.return_horizon_meta: bool = False
        self.return_all_step_horizon_meta: bool = False
        self.horizon_graph_id: int = 0

    @classmethod
    def from_series_dataset(
        cls,
        base: SeriesEncodingsDataset,
    ) -> "SeriesPca2dEncodingsDataset":
        """Wrap an already-loaded mean series NPZ as a pca2d-capable dataset."""
        obj = cls.__new__(cls)
        obj.__dict__.update(base.__dict__)
        obj.pca2d_encoding_names = []
        obj.pca2d_encodings = None
        obj.pca2d_scaler = getattr(base, "pca2d_scaler", None)
        obj.pca2d_basis_path = None
        obj.pca2d_flat_dim = 0
        obj.pca2d_gatt_plus = True
        obj.return_dict_targets = True
        obj.prediction_targets = list(
            getattr(base, "prediction_targets", [PRIMARY_TARGET])
        )
        obj.jepa_return_abs = bool(getattr(base, "jepa_return_abs", False))
        obj.jepa_absolute_stats = getattr(base, "jepa_absolute_stats", None)
        return obj

    def get_pca2d_encoding_names(self) -> list[str]:
        return list(getattr(self, "pca2d_encoding_names", None) or [])

    def get_pca2d_flat_dim(self) -> int:
        if self.pca2d_flat_dim:
            return int(self.pca2d_flat_dim)
        names = self.get_pca2d_encoding_names()
        if names and self.pca2d_encodings:
            return int(self.pca2d_encodings[names[0]].shape[-1])
        return 0

    def match_pca2d_encodings_from_2d_dir(
        self,
        enc_2d_dir: Union[str, Path],
        basis_path: Union[str, Path],
        *,
        name: str = PCA2D_S1_TARGET,
    ) -> None:
        basis = load_pca2d_basis(basis_path)
        logger.info(
            "Building pca2d cube %s from %s (seq_len=%s k=%s flat=%s)",
            name,
            enc_2d_dir,
            basis["seq_len"],
            basis["k"],
            basis["flat_dim"],
        )
        k_series = self.get_block_size()
        if k_series <= 0 and getattr(self, "sample_encodings", None) is not None:
            enc = np.asarray(self.sample_encodings)
            if enc.ndim >= 2:
                k_series = int(enc.shape[1])
        cube = build_pca2d_cube_for_series(
            self.sample_gids,
            enc_2d_dir,
            basis,
            k_series=k_series or None,
        )
        self.pca2d_encoding_names = [str(name)]
        self.pca2d_encodings = {str(name): cube}
        self.pca2d_basis_path = str(basis_path)
        self.pca2d_flat_dim = int(basis["flat_dim"])

    def fit_pca2d_scaler(self, mode: str | None = None) -> None:
        names = self.get_pca2d_encoding_names()
        if not names or not self.pca2d_encodings:
            self.pca2d_scaler = None
            return
        scaler_mode = str(mode or getattr(self, "pca2d_scaler_mode", "flat") or "flat")
        self.pca2d_scaler_mode = scaler_mode
        self.pca2d_scaler = StandardScaler()
        parts = [
            self.pca2d_encodings[n].reshape(-1, self.pca2d_encodings[n].shape[-1])
            for n in names
        ]
        flat = np.concatenate(parts, axis=0)
        if scaler_mode == "per_pc":
            k = self._infer_pca2d_k(flat.shape[-1])
            if k <= 0:
                raise ValueError("per_pc scaler requires inferable k from flat_dim")
            flat_pc = flat.reshape(-1, k)
            self.pca2d_scaler.fit(flat_pc)
            self.pca2d_flat_dim = int(flat.shape[-1])
            self._pca2d_k = int(k)
            return
        self.pca2d_scaler.fit(flat)
        self.pca2d_flat_dim = int(flat.shape[-1])

    def _infer_pca2d_k(self, flat_dim: int) -> int:
        k_attr = int(getattr(self, "_pca2d_k", 0) or 0)
        if k_attr > 0:
            return k_attr
        if flat_dim % 712 == 0:
            return flat_dim // 712
        if flat_dim % 4 == 0:
            return 4
        return 0

    def pca2d_scaler_matches_mode(self, mode: str | None = None) -> bool:
        """True when the loaded scaler matches the requested normalization mode."""
        scaler = self.pca2d_scaler
        if scaler is None:
            return True
        scaler_mode = str(mode or getattr(self, "pca2d_scaler_mode", "flat") or "flat")
        n_features = getattr(scaler, "n_features_in_", None)
        if n_features is None:
            return False
        flat_dim = int(self.get_pca2d_flat_dim() or 0)
        if scaler_mode == "per_pc":
            k = self._infer_pca2d_k(flat_dim)
            return k > 0 and int(n_features) == k
        return flat_dim > 0 and int(n_features) == flat_dim

    def _scale_pca2d_cube(self, data: np.ndarray, *, fit: bool = False) -> np.ndarray:
        dims = data.shape
        flat = data.reshape(-1, dims[-1])
        scaler = self.pca2d_scaler
        if scaler is None:
            return data.astype(np.float32)
        mode = str(getattr(self, "pca2d_scaler_mode", "flat") or "flat")
        n_in = getattr(scaler, "n_features_in_", None)
        # Prefer scaler geometry over the mode string: sidecars can leave
        # mode="flat" while the shared train scaler is per-PC (k features).
        use_per_pc = mode == "per_pc"
        if n_in is not None and not fit:
            if int(n_in) == int(dims[-1]):
                use_per_pc = False
            elif int(n_in) > 0 and int(dims[-1]) % int(n_in) == 0:
                use_per_pc = True
                self.pca2d_scaler_mode = "per_pc"
                self._pca2d_k = int(n_in)
        if use_per_pc:
            k = int(n_in) if (n_in is not None and int(n_in) > 0) else self._infer_pca2d_k(
                dims[-1]
            )
            if k <= 0 or int(dims[-1]) % k != 0:
                raise ValueError(
                    f"per_pc pca2d scaler expects divisible flat_dim; "
                    f"got flat_dim={dims[-1]}, k={k}, n_features_in_={n_in}"
                )
            flat_pc = flat.reshape(-1, k)
            if fit:
                flat_pc = scaler.fit_transform(flat_pc)
            else:
                flat_pc = scaler.transform(flat_pc)
            return flat_pc.reshape(dims).astype(np.float32)
        if fit:
            flat = scaler.fit_transform(flat)
        else:
            flat = scaler.transform(flat)
        return flat.reshape(dims).astype(np.float32)

    def _unscale_pca2d_cube(self, data: np.ndarray) -> np.ndarray:
        """Inverse of ``_scale_pca2d_cube`` (same flat / per_pc geometry)."""
        dims = data.shape
        flat = data.reshape(-1, dims[-1])
        scaler = self.pca2d_scaler
        if scaler is None:
            return data.astype(np.float32)
        mode = str(getattr(self, "pca2d_scaler_mode", "flat") or "flat")
        n_in = getattr(scaler, "n_features_in_", None)
        use_per_pc = mode == "per_pc"
        if n_in is not None:
            if int(n_in) == int(dims[-1]):
                use_per_pc = False
            elif int(n_in) > 0 and int(dims[-1]) % int(n_in) == 0:
                use_per_pc = True
                self.pca2d_scaler_mode = "per_pc"
                self._pca2d_k = int(n_in)
        if use_per_pc:
            k = int(n_in) if (n_in is not None and int(n_in) > 0) else self._infer_pca2d_k(
                dims[-1]
            )
            if k <= 0 or int(dims[-1]) % k != 0:
                raise ValueError(
                    f"per_pc pca2d unscale expects divisible flat_dim; "
                    f"got flat_dim={dims[-1]}, k={k}, n_features_in_={n_in}"
                )
            flat_pc = flat.reshape(-1, k)
            flat_pc = scaler.inverse_transform(flat_pc)
            return flat_pc.reshape(dims).astype(np.float32)
        flat = scaler.inverse_transform(flat)
        return flat.reshape(dims).astype(np.float32)

    def _horizon_meta_for_transition(
        self, idx: int, active_indices: list[int], trans_idx: int
    ) -> dict[str, float | int]:
        """Horizon features for y-step ``trans_idx`` (prefix ``trans_idx`` → ``trans_idx+1``)."""
        if len(active_indices) < 2:
            return {
                "edge_tau": 0.0,
                "n_edges": 1,
                "dt_days": 0.0,
                "graph_id": int(getattr(self, "horizon_graph_id", 0) or 0),
            }
        t0 = int(trans_idx)
        t1 = t0 + 1
        if t1 >= len(active_indices):
            t0 = len(active_indices) - 2
            t1 = len(active_indices) - 1
        lo = int(active_indices[t0])
        hi = int(active_indices[t1])
        n_edges = max(1, hi - lo)
        edge_tau = 0.0
        edge = getattr(self, "edge_distances", None)
        if edge is not None:
            row = np.asarray(edge[idx], dtype=np.float32).reshape(-1)
            if row.size > 0:
                edge_tau = float(row[lo:hi].sum()) if hi > lo else float(row[min(lo, row.size - 1)])
            if getattr(self, "edge_phylo_use_log1p", True):
                edge_tau = float(np.log1p(max(edge_tau, 0.0)))
        dt_days = 0.0
        dates = getattr(self, "sample_dates", None)
        if dates is not None:
            try:
                d0 = float(dates[idx][lo])
                d1 = float(dates[idx][hi])
                if d0 > 0 and d1 > 0:
                    dt_days = max(0.0, (d1 - d0) / 86400.0)
            except (TypeError, IndexError, ValueError):
                pass
        return {
            "edge_tau": edge_tau,
            "n_edges": int(n_edges),
            "dt_days": float(dt_days),
            "graph_id": int(getattr(self, "horizon_graph_id", 0) or 0),
        }

    def _build_horizon_meta(
        self, idx: int, active_indices: list[int]
    ) -> dict[str, float | int]:
        """Per-sample horizon features for the last predicted delta."""
        n_trans = max(0, len(active_indices) - 1)
        if n_trans <= 0:
            return self._horizon_meta_for_transition(idx, active_indices, 0)
        return self._horizon_meta_for_transition(idx, active_indices, n_trans - 1)

    def _build_all_step_horizon_meta(
        self, idx: int, active_indices: list[int]
    ) -> list[dict[str, float | int]]:
        """One horizon dict per y-step (length ``len(active_indices) - 1``)."""
        n_trans = max(0, len(active_indices) - 1)
        return [
            self._horizon_meta_for_transition(idx, active_indices, t)
            for t in range(n_trans)
        ]

    def _absolute_stat_trajectory(
        self,
        stat: str,
        sub_mean: np.ndarray,
        sub_extra_row: dict[str, np.ndarray],
        sub_block_row: Optional[dict[str, np.ndarray]] = None,
        sub_pca2d_row: Optional[dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        if is_pca2d_target(stat):
            row = sub_pca2d_row or {}
            if stat not in row:
                raise KeyError(
                    f"prediction target {stat!r} missing from pca2d encodings "
                    f"(available: {list(row.keys())})"
                )
            return row[stat].astype(np.float32)
        return super()._absolute_stat_trajectory(
            stat, sub_mean, sub_extra_row, sub_block_row=sub_block_row
        )

    def _build_multitarget_y_dict(
        self,
        sub_mean: np.ndarray,
        sub_extra_row: dict[str, np.ndarray],
        sub_block_row: Optional[dict[str, np.ndarray]] = None,
        sub_pca2d_row: Optional[dict[str, np.ndarray]] = None,
    ) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for stat in self.get_prediction_targets():
            traj = self._absolute_stat_trajectory(
                stat,
                sub_mean,
                sub_extra_row,
                sub_block_row=sub_block_row,
                sub_pca2d_row=sub_pca2d_row,
            )
            y_prefix = traj[:-1]
            y_suffix = traj[1:]
            if self.target_mode == "diff":
                out[stat] = (y_suffix - y_prefix).astype(np.float32)
            else:
                out[stat] = y_suffix.astype(np.float32)
        return out

    def _build_jepa_absolute_targets(
        self,
        sub_mean: np.ndarray,
        sub_extra_row: dict[str, np.ndarray],
        stats: Sequence[str],
        sub_block_row: Optional[dict[str, np.ndarray]] = None,
        sub_pca2d_row: Optional[dict[str, np.ndarray]] = None,
    ) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for stat in stats:
            traj = self._absolute_stat_trajectory(
                stat,
                sub_mean,
                sub_extra_row,
                sub_block_row=sub_block_row,
                sub_pca2d_row=sub_pca2d_row,
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
                    stat, cube, self.scaler, fit=False
                )

        scaled_block_row: dict[str, np.ndarray] = {}
        if self.block_encodings:
            for name in self.get_block_encoding_names():
                cube = self.block_encodings[name][idx]
                scaled_block_row[name] = self._scale_block_cube(cube, fit=False)

        scaled_pca2d_row: dict[str, np.ndarray] = {}
        if self.pca2d_encodings:
            for name in self.get_pca2d_encoding_names():
                cube = self.pca2d_encodings[name][idx]
                scaled_pca2d_row[name] = self._scale_pca2d_cube(cube, fit=False)

        active = getattr(self, "_timeline_indices", None)
        if active is None:
            sub_mean = mean_row
            sub_extra_row = scaled_extra_row
            sub_block_row = scaled_block_row
            sub_pca2d_row = scaled_pca2d_row
        else:
            active_idx = np.asarray(active, dtype=np.int64)
            sub_mean = mean_row[active_idx]
            sub_extra_row = {s: v[active_idx] for s, v in scaled_extra_row.items()}
            sub_block_row = {s: v[active_idx] for s, v in scaled_block_row.items()}
            sub_pca2d_row = {s: v[active_idx] for s, v in scaled_pca2d_row.items()}

        active_indices = self._active_timeline_indices(sub_mean.shape[0])
        x_abs_mean = sub_mean[:-1].astype(np.float32)

        pca_name = (
            self.get_pca2d_encoding_names()[0]
            if self.get_pca2d_encoding_names()
            else None
        )
        if pca_name is None:
            raise RuntimeError(
                "SeriesPca2dEncodingsDataset requires pca2d encodings; "
                "call match_pca2d_encodings_from_2d_dir first"
            )
        x_abs_pca = sub_pca2d_row[pca_name][:-1].astype(np.float32)

        gatt = getattr(self, "input_feature_mode", "absolute") in ("gatt_plus", "diff")
        x = concat_mean_pca2d_prefix_features(
            x_abs_mean,
            x_abs_pca,
            gatt_plus=gatt,
            pca2d_gatt_plus=bool(getattr(self, "pca2d_gatt_plus", True)),
        )

        y = self._build_multitarget_y_dict(
            sub_mean,
            sub_extra_row,
            sub_block_row=sub_block_row,
            sub_pca2d_row=sub_pca2d_row,
        )

        if getattr(self, "jepa_return_abs", False):
            jepa_stats = getattr(self, "jepa_absolute_stats", None) or (PRIMARY_TARGET,)
            jepa_abs = self._build_jepa_absolute_targets(
                sub_mean,
                sub_extra_row,
                jepa_stats,
                sub_block_row=sub_block_row,
                sub_pca2d_row=sub_pca2d_row,
            )
            if getattr(self, "return_horizon_meta", False):
                meta = self._horizon_meta_payload(idx, active_indices)
                return x, y, jepa_abs, meta
            return x, y, jepa_abs

        if getattr(self, "return_horizon_meta", False):
            meta = self._horizon_meta_payload(idx, active_indices)
            return x, y, meta
        return x, y

    def _horizon_meta_payload(
        self, idx: int, active_indices: list[int]
    ) -> dict[str, float | int] | list[dict[str, float | int]]:
        if getattr(self, "return_all_step_horizon_meta", False):
            return self._build_all_step_horizon_meta(idx, active_indices)
        return self._build_horizon_meta(idx, active_indices)

    def remove_by_index(self, ids_list):
        n_samples = self.sample_encodings.shape[0]
        mask = np.ones(n_samples, dtype=bool)
        mask[list(ids_list)] = False
        if self.pca2d_encodings:
            self.pca2d_encodings = {
                name: enc[mask] for name, enc in self.pca2d_encodings.items()
            }
        super().remove_by_index(ids_list)

    def save_pca2d_sidecar(self, filepath: Union[str, Path], *, save_scaler: bool = True) -> None:
        """Save pca2d cubes next to a mean series NPZ."""
        filepath = Path(filepath)
        names = self.get_pca2d_encoding_names()
        if not names or not self.pca2d_encodings:
            raise RuntimeError("No pca2d encodings to save")
        payload: dict[str, Any] = {
            "pca2d_encoding_names": np.array(names, dtype=object),
            "pca2d_basis_path": np.array(self.pca2d_basis_path or "", dtype=object),
            "pca2d_flat_dim": np.array(self.get_pca2d_flat_dim()),
            "pca2d_scaler_mode": np.array(
                str(getattr(self, "pca2d_scaler_mode", "flat") or "flat"), dtype=object
            ),
        }
        for name in names:
            payload[f"pca2d_enc_{name}"] = self.pca2d_encodings[name]
        out = Path(str(filepath) + ".pca2d.npz")
        np.savez_compressed(out, **payload)
        logger.info("Saved pca2d sidecar → %s", out)
        if save_scaler and self.pca2d_scaler is not None:
            sp = Path(str(filepath) + ".pca2d_scaler")
            joblib.dump(self.pca2d_scaler, sp)
            logger.info("Saved pca2d scaler → %s", sp)

    def load_pca2d_sidecar(self, filepath: Union[str, Path], *, load_scaler: bool = True) -> bool:
        """Load pca2d cubes from ``{filepath}.pca2d.npz`` if present."""
        filepath = Path(filepath)
        side = Path(str(filepath) + ".pca2d.npz")
        if not side.is_file():
            return False
        data = np.load(side, allow_pickle=True)
        names = [str(s) for s in np.asarray(data["pca2d_encoding_names"]).tolist()]
        cubes = {}
        for name in names:
            key = f"pca2d_enc_{name}"
            if key not in data:
                raise KeyError(f"pca2d sidecar missing {key}")
            cubes[name] = data[key].astype(np.float32)
        self.pca2d_encoding_names = names
        self.pca2d_encodings = cubes
        if "pca2d_flat_dim" in data:
            self.pca2d_flat_dim = int(np.asarray(data["pca2d_flat_dim"]).item())
        else:
            self.pca2d_flat_dim = int(cubes[names[0]].shape[-1])
        if "pca2d_basis_path" in data:
            self.pca2d_basis_path = str(np.asarray(data["pca2d_basis_path"]).item())
        if "pca2d_scaler_mode" in data:
            self.pca2d_scaler_mode = str(np.asarray(data["pca2d_scaler_mode"]).item())
        if load_scaler:
            sp = Path(str(filepath) + ".pca2d_scaler")
            if sp.is_file():
                self.pca2d_scaler = joblib_load_numpy2_compat(str(sp))
        logger.info("Loaded pca2d sidecar from %s (names=%s)", side, names)
        return True
