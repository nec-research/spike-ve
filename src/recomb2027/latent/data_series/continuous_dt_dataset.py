#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     continuous_dt_dataset.py
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
"""Continuous-Δt dataset for prefix → next-variant diffusion training.

Default behaviour (``pair_mode=random``, ``prefix_mode=contiguous``):

1. Samples ``last_idx ∈ [last_idx_min, last_idx_max]``.
2. Builds a fixed-length prefix from consecutive series steps ending at ``last_idx``.
3. Target diff ``series[target_idx] - series[last_idx]`` with ``dt`` from dates.

Extended modes for ablations:

- **Prefix (Q3):** ``fixed`` — explicit timeline e.g. ``[0,1,2]`` vs ``[0,1,2,3]``;
  ``random_gap`` — random strict subsequence ending at a sampled anchor (≥1 gap).
- **Pairs (Q4):** ``all_to_final`` — all anchors → ``target_idx``;
  ``all_pairs`` — all ``(i, j)`` with ``i < j``;
  ``all_pairs_dt_weighted`` — one random pair per series weighted by ``Δt``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

PAIR_MODES = (
    "random",
    "all_to_final",
    "all_pairs",
    "all_pairs_dt_weighted",
)
PREFIX_MODES = ("contiguous", "fixed", "random_gap")


@dataclass
class DiffusionTrainSample:
    """One continuous-Δt training/eval example for ``DiffusionLSTM``."""

    prefix: torch.Tensor
    target_diff: torch.Tensor
    dt_days: torch.Tensor
    edge_phylo: torch.Tensor | None = None
    target_calendar_days: torch.Tensor | None = None
    lineage_id: torch.Tensor | None = None
    country_id: torch.Tensor | None = None


def collate_diffusion_train_batch(
    samples: list[DiffusionTrainSample],
) -> DiffusionTrainSample:

    def _stack_optional(field: str):
        vals = [getattr(s, field) for s in samples]
        if vals[0] is None:
            return None
        return torch.stack(vals, dim=0)

    return DiffusionTrainSample(
        prefix=torch.stack([s.prefix for s in samples], dim=0),
        target_diff=torch.stack([s.target_diff for s in samples], dim=0),
        dt_days=torch.stack([s.dt_days for s in samples], dim=0),
        edge_phylo=_stack_optional("edge_phylo"),
        target_calendar_days=_stack_optional("target_calendar_days"),
        lineage_id=_stack_optional("lineage_id"),
        country_id=_stack_optional("country_id"),
    )


class DiffusionTrainSnapshot(Dataset):
    """Frozen val snapshot as indexable ``DiffusionTrainSample`` rows."""

    def __init__(self, snap: DiffusionTrainSample):
        self.snap = snap
        self.n = int(snap.prefix.shape[0])

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> DiffusionTrainSample:
        s = self.snap
        return DiffusionTrainSample(
            prefix=s.prefix[idx],
            target_diff=s.target_diff[idx],
            dt_days=s.dt_days[idx],
            edge_phylo=None if s.edge_phylo is None else s.edge_phylo[idx],
            target_calendar_days=(
                None if s.target_calendar_days is None else s.target_calendar_days[idx]
            ),
            lineage_id=None if s.lineage_id is None else s.lineage_id[idx],
            country_id=None if s.country_id is None else s.country_id[idx],
        )


def _validate_strictly_increasing(indices: Sequence[int], *, name: str) -> list[int]:
    out = [int(x) for x in indices]
    if len(out) < 1:
        raise ValueError(f"{name} must contain at least one index")
    for a, b in zip(out[:-1], out[1:]):
        if b <= a:
            raise ValueError(f"{name} must be strictly increasing; got {out}")
    return out


def build_prefix_from_timeline(
    enc_row: np.ndarray,
    timeline: Sequence[int],
    prefix_len: int,
) -> tuple[np.ndarray, list[int]]:
    """Encode ``timeline`` into a left-padded ``(prefix_len, D)`` prefix."""
    timeline = _validate_strictly_increasing(timeline, name="timeline")
    encs = enc_row[timeline]
    if len(timeline) >= prefix_len:
        slot_timeline = timeline[-prefix_len:]
        prefix = enc_row[slot_timeline]
    else:
        pad_n = prefix_len - len(timeline)
        slot_timeline = [timeline[0]] * pad_n + timeline
        pad = np.tile(encs[:1], (pad_n, 1))
        prefix = np.concatenate([pad, encs], axis=0)
    return np.ascontiguousarray(prefix, dtype=np.float32), slot_timeline


def sample_random_gap_timeline(
    rng: np.random.Generator,
    anchor_idx: int,
    *,
    min_obs: int,
    max_obs: int | None,
) -> list[int]:
    """Strict subsequence of ``[0..anchor]`` ending at ``anchor`` with ≥1 gap when possible."""
    anchor_idx = int(anchor_idx)
    if anchor_idx < 0:
        raise ValueError(f"anchor_idx must be >= 0, got {anchor_idx}")
    max_obs = int(max_obs if max_obs is not None else anchor_idx + 1)
    max_obs = min(max_obs, anchor_idx + 1)
    min_obs = max(2, int(min_obs))
    min_obs = min(min_obs, max_obs)
    n_obs = int(rng.integers(min_obs, max_obs + 1))
    if n_obs >= anchor_idx + 1:
        return list(range(anchor_idx + 1))
    if n_obs == 1:
        return [anchor_idx]
    interior = rng.choice(
        np.arange(0, anchor_idx, dtype=np.int64),
        size=n_obs - 1,
        replace=False,
    )
    timeline = sorted(int(x) for x in interior) + [anchor_idx]
    if len(timeline) >= 2 and timeline[-1] - timeline[-2] == 1 and anchor_idx >= 2:
        # Force at least one gap when anchor > 0 and we'd otherwise be contiguous tail.
        if n_obs < anchor_idx + 1:
            return timeline
    return timeline


def enumerate_pairs(
    *,
    pair_mode: str,
    last_idx_min: int,
    target_idx: int,
) -> list[tuple[int, int]]:
    """Return ``(anchor_idx, target_idx)`` pairs for expanded datasets."""
    if pair_mode == "random":
        return []
    if pair_mode == "all_to_final":
        return [
            (a, target_idx)
            for a in range(last_idx_min, target_idx)
        ]
    if pair_mode in ("all_pairs", "all_pairs_dt_weighted"):
        return [
            (i, j)
            for i in range(last_idx_min, target_idx)
            for j in range(i + 1, target_idx + 1)
        ]
    raise ValueError(f"unknown pair_mode {pair_mode!r}")


class ContinuousDtDataset(Dataset):
    def __init__(
        self,
        encs_scaled: np.ndarray,
        dates: np.ndarray,
        max_prefix_len: int = 5,
        last_idx_min: int = 1,
        last_idx_max: int | None = None,
        target_idx: int | None = None,
        seed: int | None = None,
        edge_distances: np.ndarray | None = None,
        target_calendar_days: np.ndarray | None = None,
        lineage_ids: np.ndarray | None = None,
        country_ids: np.ndarray | None = None,
        *,
        pair_mode: str = "random",
        pair_dt_weight_power: float = 1.0,
        prefix_mode: str = "contiguous",
        prefix_fixed_indices: Sequence[int] | None = None,
        random_gap_min_obs: int = 3,
        random_gap_max_obs: int | None = None,
        step_calendar_days: np.ndarray | None = None,
        step_lineage_ids: np.ndarray | None = None,
        step_country_ids: np.ndarray | None = None,
    ) -> None:
        encs = np.asarray(encs_scaled, dtype=np.float32)
        if encs.ndim != 3:
            raise ValueError(f"encs_scaled must be (N, K, D), got {encs.shape}")
        self.data = encs
        self.N, self.K, self.D = encs.shape

        if dates is None:
            self.dates = np.zeros((self.N, self.K), dtype=np.float32)
        else:
            self.dates = np.asarray(dates, dtype=np.float32)
            if self.dates.shape != (self.N, self.K):
                raise ValueError(
                    f"dates shape {self.dates.shape} != ({self.N}, {self.K})"
                )

        self.L = int(max_prefix_len)
        if self.L < 2:
            raise ValueError("max_prefix_len must be >= 2")

        self.target_idx = (self.K - 1) if target_idx is None else int(target_idx)
        if not (0 <= self.target_idx < self.K):
            raise ValueError(f"target_idx {self.target_idx} out of range [0, {self.K})")

        self.last_idx_min = max(0, int(last_idx_min))
        self.last_idx_max = (
            self.target_idx - 1 if last_idx_max is None else int(last_idx_max)
        )
        if self.last_idx_max < self.last_idx_min:
            raise ValueError(
                f"last_idx_max {self.last_idx_max} < last_idx_min {self.last_idx_min}"
            )
        if self.last_idx_max >= self.target_idx:
            raise ValueError(
                f"last_idx_max {self.last_idx_max} must be < target_idx {self.target_idx}"
            )

        if pair_mode not in PAIR_MODES:
            raise ValueError(f"pair_mode must be one of {PAIR_MODES}; got {pair_mode!r}")
        if prefix_mode not in PREFIX_MODES:
            raise ValueError(
                f"prefix_mode must be one of {PREFIX_MODES}; got {prefix_mode!r}"
            )
        self.pair_mode = pair_mode
        self.pair_dt_weight_power = float(pair_dt_weight_power)
        self.prefix_mode = prefix_mode
        self.random_gap_min_obs = int(random_gap_min_obs)
        self.random_gap_max_obs = random_gap_max_obs

        if prefix_fixed_indices is not None:
            fixed = _validate_strictly_increasing(
                prefix_fixed_indices, name="prefix_fixed_indices"
            )
            if fixed[-1] >= self.target_idx:
                raise ValueError(
                    f"prefix_fixed_indices last index {fixed[-1]} must be < "
                    f"target_idx {self.target_idx}"
                )
            self.prefix_fixed_indices: list[int] | None = fixed
        else:
            self.prefix_fixed_indices = None

        if prefix_mode == "fixed":
            if self.prefix_fixed_indices is None:
                raise ValueError(
                    "prefix_mode='fixed' requires prefix_fixed_indices "
                    "(e.g. 0 1 2 for gap prefix ending at anchor 2)"
                )
            if pair_mode != "random":
                raise ValueError(
                    "prefix_mode='fixed' is only supported with pair_mode='random' "
                    "(anchor is implied by prefix_fixed_indices[-1])"
                )

        if prefix_mode == "random_gap" and pair_mode != "random":
            raise ValueError(
                "prefix_mode='random_gap' is only supported with pair_mode='random'"
            )

        self._pairs = enumerate_pairs(
            pair_mode=pair_mode,
            last_idx_min=self.last_idx_min,
            target_idx=self.target_idx,
        )
        if pair_mode in ("all_pairs", "all_pairs_dt_weighted") and not self._pairs:
            raise ValueError("all_pairs modes require at least one valid pair")

        self._rng = np.random.default_rng(seed)

        if edge_distances is not None:
            edge = np.asarray(edge_distances, dtype=np.float32)
            if edge.shape != (self.N, self.K - 1):
                raise ValueError(
                    f"edge_distances shape {edge.shape} != ({self.N}, {self.K - 1})"
                )
            self.edge_distances = edge
        else:
            self.edge_distances = None

        self._target_calendar = self._validate_sidecar(
            target_calendar_days, "target_calendar_days"
        )
        self._lineage_ids = self._validate_sidecar(lineage_ids, "lineage_ids", int)
        self._country_ids = self._validate_sidecar(country_ids, "country_ids", int)

        self._step_calendar = self._validate_step_sidecar(
            step_calendar_days, "step_calendar_days"
        )
        self._step_lineage = self._validate_step_sidecar(
            step_lineage_ids, "step_lineage_ids", int
        )
        self._step_country = self._validate_step_sidecar(
            step_country_ids, "step_country_ids", int
        )

    def _validate_sidecar(self, arr, name: str, dtype=np.float32):
        if arr is None:
            return None
        out = np.asarray(arr)
        if out.shape[0] != self.N:
            raise ValueError(f"{name} length {out.shape[0]} != N={self.N}")
        if dtype is int:
            return out.astype(np.int64)
        return out.astype(np.float32)

    def _validate_step_sidecar(self, arr, name: str, dtype=np.float32):
        if arr is None:
            return None
        out = np.asarray(arr)
        if out.shape != (self.N, self.K):
            raise ValueError(f"{name} shape {out.shape} != ({self.N}, {self.K})")
        if dtype is int:
            return out.astype(np.int64)
        return out.astype(np.float32)

    def __len__(self) -> int:
        if self.pair_mode in ("all_to_final", "all_pairs"):
            return self.N * len(self._pairs)
        return self.N

    @property
    def latent_dim(self) -> int:
        return self.D

    @property
    def pairs_per_series(self) -> int:
        return len(self._pairs)

    def describe_config(self) -> str:
        parts = [
            f"pair_mode={self.pair_mode}",
            f"prefix_mode={self.prefix_mode}",
            f"last_idx=[{self.last_idx_min},{self.last_idx_max}]",
            f"target_idx={self.target_idx}",
            f"len={len(self)} ({self.N} series",
        ]
        if self.pair_mode in ("all_to_final", "all_pairs"):
            parts.append(f"× {len(self._pairs)} pairs")
        parts.append(")")
        if self.prefix_fixed_indices is not None:
            parts.append(f" fixed_timeline={self.prefix_fixed_indices}")
        if self.pair_mode == "all_pairs_dt_weighted":
            parts.append(f" dt_weight_power={self.pair_dt_weight_power:g}")
        return "".join(parts)

    def _resolve_index(self, idx: int) -> tuple[int, int, int]:
        """Map flat index → ``(series_idx, anchor_idx, target_j)``."""
        if self.pair_mode in ("all_to_final", "all_pairs"):
            if len(self._pairs) == 0:
                raise RuntimeError("expanded pair list is empty")
            series_idx = int(idx) // len(self._pairs)
            pair_idx = int(idx) % len(self._pairs)
            anchor, target_j = self._pairs[pair_idx]
            return series_idx, anchor, target_j

        series_idx = int(idx)
        if self.prefix_mode == "fixed":
            assert self.prefix_fixed_indices is not None
            anchor = self.prefix_fixed_indices[-1]
            return series_idx, anchor, self.target_idx

        if self.pair_mode == "all_pairs_dt_weighted":
            weights = []
            for a, t in self._pairs:
                dt = float(self.dates[series_idx, t] - self.dates[series_idx, a])
                weights.append(max(dt, 1e-3) ** self.pair_dt_weight_power)
            w = np.asarray(weights, dtype=np.float64)
            w /= w.sum()
            pair_idx = int(self._rng.choice(len(self._pairs), p=w))
            anchor, target_j = self._pairs[pair_idx]
            return series_idx, anchor, target_j

        anchor = int(
            self._rng.integers(self.last_idx_min, self.last_idx_max + 1)
        )
        return series_idx, anchor, self.target_idx

    def _build_contiguous_prefix(
        self, series_idx: int, anchor_idx: int
    ) -> tuple[np.ndarray, list[int]]:
        """Legacy contiguous window with left-pad (pad slots repeat index 0)."""
        from recomb2027.latent.data_series.edge_phylo_features import (
            prefix_timeline_indices,
        )

        n_obs = anchor_idx + 1
        if n_obs >= self.L:
            prefix = self.data[series_idx, anchor_idx - self.L + 1 : anchor_idx + 1]
        else:
            pad = np.tile(self.data[series_idx, 0:1], (self.L - n_obs, 1))
            prefix = np.concatenate(
                [pad, self.data[series_idx, :n_obs]], axis=0
            )
        slot_timeline = prefix_timeline_indices(anchor_idx, self.L)
        return np.ascontiguousarray(prefix, dtype=np.float32), slot_timeline

    def _strict_prefix_timeline(self, anchor_idx: int) -> list[int]:
        """Strictly increasing timeline for fixed / random_gap modes."""
        if self.prefix_mode == "fixed":
            assert self.prefix_fixed_indices is not None
            if self.prefix_fixed_indices[-1] != anchor_idx:
                raise ValueError(
                    f"fixed prefix ends at {self.prefix_fixed_indices[-1]} "
                    f"but anchor_idx={anchor_idx}"
                )
            return list(self.prefix_fixed_indices)
        return sample_random_gap_timeline(
            self._rng,
            anchor_idx,
            min_obs=self.random_gap_min_obs,
            max_obs=self.random_gap_max_obs,
        )

    def _prefix_timeline(self, anchor_idx: int) -> list[int]:
        if self.prefix_mode == "contiguous":
            from recomb2027.latent.data_series.edge_phylo_features import (
                prefix_timeline_indices,
            )

            return prefix_timeline_indices(anchor_idx, self.L)
        return self._strict_prefix_timeline(anchor_idx)

    def _metadata_at_target(
        self, series_idx: int, target_j: int
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        cal_t = lin_t = cty_t = None
        if self._step_calendar is not None:
            cal_t = torch.tensor(
                float(self._step_calendar[series_idx, target_j]), dtype=torch.float32
            )
        elif self._target_calendar is not None and target_j == self.target_idx:
            cal_t = torch.tensor(
                float(self._target_calendar[series_idx]), dtype=torch.float32
            )
        elif self.dates is not None:
            cal_t = torch.tensor(
                float(self.dates[series_idx, target_j]), dtype=torch.float32
            )

        if self._step_lineage is not None:
            lin_t = torch.tensor(
                int(self._step_lineage[series_idx, target_j]), dtype=torch.long
            )
        elif self._lineage_ids is not None and target_j == self.target_idx:
            lin_t = torch.tensor(int(self._lineage_ids[series_idx]), dtype=torch.long)

        if self._step_country is not None:
            cty_t = torch.tensor(
                int(self._step_country[series_idx, target_j]), dtype=torch.long
            )
        elif self._country_ids is not None and target_j == self.target_idx:
            cty_t = torch.tensor(int(self._country_ids[series_idx]), dtype=torch.long)
        return cal_t, lin_t, cty_t

    def make_sample(
        self,
        series_idx: int,
        anchor_idx: int,
        target_j: int,
    ) -> DiffusionTrainSample:
        """Build one example (used by ``__getitem__`` and val snapshots)."""
        if self.prefix_mode == "contiguous":
            prefix, slot_timeline = self._build_contiguous_prefix(
                series_idx, anchor_idx
            )
        else:
            timeline = self._strict_prefix_timeline(anchor_idx)
            prefix, slot_timeline = build_prefix_from_timeline(
                self.data[series_idx], timeline, self.L
            )
        target_diff = self.data[series_idx, target_j] - self.data[series_idx, anchor_idx]
        dt = float(self.dates[series_idx, target_j] - self.dates[series_idx, anchor_idx])

        edge_t = None
        if self.edge_distances is not None:
            from recomb2027.latent.data_series.edge_phylo_features import (
                prefix_step_edge_distances_from_timeline,
            )

            edge_d = prefix_step_edge_distances_from_timeline(
                self.edge_distances[series_idx], slot_timeline
            )
            edge_t = torch.from_numpy(edge_d)

        cal_t, lin_t, cty_t = self._metadata_at_target(series_idx, target_j)

        return DiffusionTrainSample(
            prefix=torch.from_numpy(prefix),
            target_diff=torch.from_numpy(np.ascontiguousarray(target_diff, dtype=np.float32)),
            dt_days=torch.tensor(dt, dtype=torch.float32),
            edge_phylo=edge_t,
            target_calendar_days=cal_t,
            lineage_id=lin_t,
            country_id=cty_t,
        )

    def __getitem__(self, idx: int) -> DiffusionTrainSample:
        series_idx, anchor_idx, target_j = self._resolve_index(idx)
        return self.make_sample(series_idx, anchor_idx, target_j)

    def deterministic_view(self, last_idx: int) -> "ContinuousDtDataset":
        """Return a copy that always uses the same ``last_idx`` (for eval)."""
        if not (self.last_idx_min <= last_idx <= self.last_idx_max):
            raise ValueError(
                f"last_idx {last_idx} outside [{self.last_idx_min}, {self.last_idx_max}]"
            )
        view = ContinuousDtDataset(
            encs_scaled=self.data,
            dates=self.dates,
            max_prefix_len=self.L,
            last_idx_min=last_idx,
            last_idx_max=last_idx,
            target_idx=self.target_idx,
            seed=None,
            edge_distances=self.edge_distances,
            target_calendar_days=self._target_calendar,
            lineage_ids=self._lineage_ids,
            country_ids=self._country_ids,
            pair_mode="random",
            prefix_mode="contiguous",
            step_calendar_days=self._step_calendar,
            step_lineage_ids=self._step_lineage,
            step_country_ids=self._step_country,
        )
        return view


def diffusion_dataset_kwargs_from_args(
    args,
    *,
    for_val: bool = False,
) -> dict:
    """Build ``ContinuousDtDataset`` keyword args from training CLI ``args``."""
    pair_mode = "random" if for_val else str(getattr(args, "pair_mode", "random"))
    prefix_mode = str(getattr(args, "prefix_mode", "contiguous"))
    val_prefix_mode = str(getattr(args, "val_prefix_mode", "contiguous"))
    if for_val and val_prefix_mode == "contiguous":
        prefix_mode = "contiguous"
    elif for_val and val_prefix_mode == "same":
        prefix_mode = str(getattr(args, "prefix_mode", "contiguous"))

    fixed = getattr(args, "prefix_fixed_indices", None)
    if fixed is not None and len(fixed) == 0:
        fixed = None

    return dict(
        max_prefix_len=args.max_prefix_len,
        last_idx_min=args.last_idx_min,
        last_idx_max=args.last_idx_max,
        pair_mode=pair_mode,
        pair_dt_weight_power=float(getattr(args, "pair_dt_weight_power", 1.0)),
        prefix_mode=prefix_mode,
        prefix_fixed_indices=fixed,
        random_gap_min_obs=int(getattr(args, "random_gap_min_obs", 3)),
        random_gap_max_obs=getattr(args, "random_gap_max_obs", None),
    )


def build_continuous_dt_dataset(
    encs_scaled: np.ndarray,
    gids_array: np.ndarray,
    gid_to_date: dict | None,
    max_prefix_len: int = 5,
    last_idx_min: int = 1,
    last_idx_max: int | None = None,
    target_idx: int | None = None,
    seed: int | None = None,
) -> ContinuousDtDataset:
    """Convenience factory that computes per-step dates from gids if available."""
    from recomb2027.latent.predictors.predictor_utils import compute_series_dates

    if gid_to_date is None:
        dates = None
    else:
        dates = compute_series_dates(gids_array, gid_to_date)
    return ContinuousDtDataset(
        encs_scaled=encs_scaled,
        dates=dates,
        max_prefix_len=max_prefix_len,
        last_idx_min=last_idx_min,
        last_idx_max=last_idx_max,
        target_idx=target_idx,
        seed=seed,
    )
