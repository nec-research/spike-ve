#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     gpt_lineage_substitution.py
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
"""Multi-draw GPT inference via same-lineage substitution of the last prefix step.

For each validation series, replace the last seen prefix genome with up to ``M``
alternative genomes from the same lineage (ordered by BLOSUM distance from the
penultimate step), then run the frozen GPT to obtain ``M`` next-step predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from recomb2027.data.series import SeriesEncodingsDataset
from recomb2027.data.extra_encoding_features import (
    concat_prefix_step_features,
)
from recomb2027.data.seq_pairwise_distance import (
    fast_pairwise_blosum_distance,
)


def to_gid_str(g) -> str:
    if isinstance(g, bytes):
        return g.decode()
    return str(g)


def series_last_prefix_idx(series_len: int) -> int:
    """Series index of the last prefix element (``x[:, -1]``) for full timelines."""
    if series_len < 2:
        raise ValueError(f"series_len must be >= 2; got {series_len}")
    return int(series_len - 2)


@dataclass(frozen=True)
class SubstituteCandidate:
    gid: str
    distance: float


@dataclass(frozen=True)
class LineageSubstitutePlan:
    """Up to ``m`` substitute genomes for one validation series."""

    series_idx: int
    last_idx: int
    ref_gid: str
    anchor_gid: str
    lineage: str
    candidates: tuple[SubstituteCandidate, ...]


def build_lineage_pools(
    gid_to_lineage: Mapping[str, str],
    *,
    gid_filter: set[str] | None = None,
) -> dict[str, list[str]]:
    """Map normalized lineage label → sorted genome ids."""
    pools: dict[str, list[str]] = {}
    for gid, lin in gid_to_lineage.items():
        g = to_gid_str(gid)
        if gid_filter is not None and g not in gid_filter:
            continue
        label = str(lin).strip()
        if not label or label.lower() in {"nan", "none"}:
            continue
        pools.setdefault(label, []).append(g)
    for label in pools:
        pools[label] = sorted(set(pools[label]))
    return pools


def build_gid_to_encoding_from_datasets(
    *datasets: SeriesEncodingsDataset,
) -> dict[str, np.ndarray]:
    """Collect ``gid → global mean encoding`` from one or more series NPZ datasets."""
    out: dict[str, np.ndarray] = {}
    for ds in datasets:
        enc = np.asarray(ds.sample_encodings, dtype=np.float32)
        for row_gids, row_enc in zip(ds.sample_gids, enc):
            for gid, vec in zip(row_gids, row_enc):
                g = to_gid_str(gid)
                out[g] = np.asarray(vec, dtype=np.float32)
    return out


def build_gid_to_extra_from_datasets(
    *datasets: SeriesEncodingsDataset,
) -> dict[str, dict[str, np.ndarray]]:
    """Collect ``stat → {gid → vector}`` for optional extra encoding stats."""
    out: dict[str, dict[str, np.ndarray]] = {}
    for ds in datasets:
        stats = ds.get_extra_encoding_stats()
        if not stats or not ds.extra_stat_encodings:
            continue
        for stat in stats:
            cube = ds.extra_stat_encodings[stat]
            stat_map = out.setdefault(stat, {})
            for row_gids, row_enc in zip(ds.sample_gids, cube):
                for gid, vec in zip(row_gids, row_enc):
                    stat_map[to_gid_str(gid)] = np.asarray(vec, dtype=np.float32)
    return out


def build_gid_to_blocks_from_datasets(
    *datasets: SeriesEncodingsDataset,
) -> dict[str, dict[str, np.ndarray]]:
    """Collect ``block_name → {gid → vector}`` for optional domain block means."""
    out: dict[str, dict[str, np.ndarray]] = {}
    for ds in datasets:
        names = ds.get_block_encoding_names()
        if not names or not ds.block_encodings:
            continue
        for name in names:
            cube = ds.block_encodings[name]
            block_map = out.setdefault(name, {})
            for row_gids, row_enc in zip(ds.sample_gids, cube):
                for gid, vec in zip(row_gids, row_enc):
                    block_map[to_gid_str(gid)] = np.asarray(vec, dtype=np.float32)
    return out


def load_gid_metadata_for_substitution(
    metadata_path: str,
    *,
    lineage_col: str = "lineage",
) -> tuple[dict[str, str], dict[str, pd.Timestamp]]:
    """Return ``(gid_to_lineage, gid_to_collection_date)`` from labels CSV."""
    df = pd.read_csv(metadata_path, dtype=str)
    need = {"genome_id", "collection_date", lineage_col}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"metadata missing columns {sorted(missing)} in {metadata_path}")

    df = df.drop_duplicates(subset=["genome_id"], keep="first")
    gid_to_lineage: dict[str, str] = {}
    gid_to_date: dict[str, pd.Timestamp] = {}
    for _, row in df.iterrows():
        gid = to_gid_str(row["genome_id"])
        gid_to_lineage[gid] = str(row[lineage_col]).strip()
        dt = pd.to_datetime(row["collection_date"], errors="coerce")
        gid_to_date[gid] = dt
    return gid_to_lineage, gid_to_date


class LineageSubstitutionIndex:
    """Candidate lookup for same-lineage prefix substitution."""

    def __init__(
        self,
        *,
        gid_to_lineage: Mapping[str, str],
        gid_to_date: Mapping[str, pd.Timestamp],
        gid_to_seq: Mapping[str, str],
        gid_to_encoding: Mapping[str, np.ndarray],
        gid_to_extra: Mapping[str, Mapping[str, np.ndarray]] | None = None,
        gid_to_blocks: Mapping[str, Mapping[str, np.ndarray]] | None = None,
        lineage_pools: Mapping[str, Sequence[str]] | None = None,
    ):
        self.gid_to_lineage = {to_gid_str(k): str(v).strip() for k, v in gid_to_lineage.items()}
        self.gid_to_date = {to_gid_str(k): v for k, v in gid_to_date.items()}
        self.gid_to_seq = {to_gid_str(k): v for k, v in gid_to_seq.items()}
        self.gid_to_encoding = {
            to_gid_str(k): np.asarray(v, dtype=np.float32) for k, v in gid_to_encoding.items()
        }
        self.gid_to_extra = {
            str(stat): {to_gid_str(g): np.asarray(v, dtype=np.float32) for g, v in m.items()}
            for stat, m in (gid_to_extra or {}).items()
        }
        self.gid_to_blocks = {
            str(name): {to_gid_str(g): np.asarray(v, dtype=np.float32) for g, v in m.items()}
            for name, m in (gid_to_blocks or {}).items()
        }
        enc_filter = set(self.gid_to_encoding)
        if lineage_pools is None:
            self.lineage_pools = build_lineage_pools(
                self.gid_to_lineage, gid_filter=enc_filter,
            )
        else:
            self.lineage_pools = {
                k: [g for g in v if g in enc_filter]
                for k, v in lineage_pools.items()
            }

    def has_full_substitute_encoding(
        self,
        gid: str,
        *,
        extra_stats: Sequence[str] = (),
        block_names: Sequence[str] = (),
    ) -> bool:
        g = to_gid_str(gid)
        if g not in self.gid_to_encoding:
            return False
        for stat in extra_stats:
            if stat not in self.gid_to_extra or g not in self.gid_to_extra[stat]:
                return False
        for name in block_names:
            if name not in self.gid_to_blocks or g not in self.gid_to_blocks[name]:
                return False
        return True

    def select_substitutes(
        self,
        series_gids: Sequence,
        last_idx: int,
        m: int,
        *,
        extra_stats: Sequence[str] = (),
        block_names: Sequence[str] = (),
    ) -> list[SubstituteCandidate]:
        """Return up to ``m`` eligible substitutes ordered by evolutionary distance."""
        if m <= 0:
            return []
        if last_idx < 1:
            raise ValueError(
                f"last_idx must be >= 1 for anchor date/distance; got {last_idx}"
            )

        gids = [to_gid_str(g) for g in series_gids]
        if last_idx >= len(gids):
            raise IndexError(
                f"last_idx={last_idx} out of range for series length {len(gids)}"
            )

        ref_gid = gids[last_idx]
        anchor_gid = gids[last_idx - 1]
        lineage = self.gid_to_lineage.get(ref_gid, "")
        prefix_before_last = set(gids[:last_idx])

        anchor_date = self.gid_to_date.get(anchor_gid, pd.NaT)
        anchor_seq = self.gid_to_seq.get(anchor_gid)

        pool = self.lineage_pools.get(lineage, [])
        scored: list[SubstituteCandidate] = []
        for cand_gid in pool:
            if cand_gid in prefix_before_last or cand_gid == ref_gid:
                continue
            if not self.has_full_substitute_encoding(
                cand_gid, extra_stats=extra_stats, block_names=block_names
            ):
                continue
            cand_date = self.gid_to_date.get(cand_gid, pd.NaT)
            if pd.isna(anchor_date) or pd.isna(cand_date) or cand_date <= anchor_date:
                continue
            if anchor_seq is None or cand_gid not in self.gid_to_seq:
                continue
            dist = float(
                fast_pairwise_blosum_distance(
                    anchor_seq, self.gid_to_seq[cand_gid],
                )
            )
            scored.append(SubstituteCandidate(gid=cand_gid, distance=dist))

        scored.sort(key=lambda c: (c.distance, c.gid))
        return scored[:m]

    def plan_for_series(
        self,
        series_idx: int,
        series_gids: Sequence,
        last_idx: int,
        m: int,
        *,
        extra_stats: Sequence[str] = (),
        block_names: Sequence[str] = (),
    ) -> LineageSubstitutePlan:
        gids = [to_gid_str(g) for g in series_gids]
        ref_gid = gids[last_idx]
        return LineageSubstitutePlan(
            series_idx=int(series_idx),
            last_idx=int(last_idx),
            ref_gid=ref_gid,
            anchor_gid=gids[last_idx - 1],
            lineage=self.gid_to_lineage.get(ref_gid, ""),
            candidates=tuple(
                self.select_substitutes(
                    gids,
                    last_idx,
                    m,
                    extra_stats=extra_stats,
                    block_names=block_names,
                )
            ),
        )


def prefix_edge_distances_with_substitute(
    edge_row: np.ndarray | None,
    *,
    last_idx: int,
    sub_distance: float,
) -> np.ndarray:
    """Per-slot phylo distances for prefix ``0..last_idx`` with a substituted last step."""
    n_slots = last_idx + 1
    out = np.zeros(n_slots, dtype=np.float32)
    if n_slots <= 1:
        return out
    if edge_row is not None:
        edge_row = np.asarray(edge_row, dtype=np.float32).reshape(-1)
        for t in range(1, last_idx):
            out[t] = float(edge_row[t - 1 : t].sum())
    out[last_idx] = float(sub_distance)
    return out


def scaled_prefix_abs_from_dataset(
    dataset: SeriesEncodingsDataset,
    series_idx: int,
) -> np.ndarray:
    """Return scaled absolute prefix mean encodings ``[K-1, D]``."""
    enc = np.asarray(dataset.sample_encodings[series_idx], dtype=np.float32)
    if dataset.scaler is not None:
        enc = dataset._apply_scaler(enc, dataset.scaler, fit=False)
    return enc[:-1].astype(np.float32)


def scaled_prefix_components_from_dataset(
    dataset: SeriesEncodingsDataset,
    series_idx: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return scaled prefix rows: mean ``[T,D]``, extra stats, block means."""
    enc = np.asarray(dataset.sample_encodings[series_idx], dtype=np.float32)
    if dataset.scaler is not None:
        mean_row = dataset._apply_scaler(enc, dataset.scaler, fit=False)
    else:
        mean_row = enc.astype(np.float32)

    x_abs = mean_row[:-1].astype(np.float32)
    extra_prefix: dict[str, np.ndarray] = {}
    for stat in dataset.get_extra_encoding_stats():
        cube = dataset.extra_stat_encodings[stat][series_idx]
        scaled = dataset._scale_extra_cube(stat, cube, dataset.scaler, fit=False)
        extra_prefix[stat] = scaled[:-1].astype(np.float32)

    block_prefix: dict[str, np.ndarray] = {}
    for name in dataset.get_block_encoding_names():
        cube = dataset.block_encodings[name][series_idx]
        scaled = dataset._scale_block_cube(cube, fit=False)
        block_prefix[name] = scaled[:-1].astype(np.float32)

    return x_abs, extra_prefix, block_prefix


def scale_substitute_mean(
    dataset: SeriesEncodingsDataset,
    raw: np.ndarray,
) -> np.ndarray:
    scaler = getattr(dataset, "substitution_mean_scaler", dataset.scaler)
    vec = np.asarray(raw, dtype=np.float32).reshape(1, -1)
    if scaler is not None:
        return dataset._apply_scaler(vec, scaler, fit=False).reshape(-1)
    return vec.reshape(-1)


def scale_substitute_extra(
    dataset: SeriesEncodingsDataset,
    stat: str,
    raw: np.ndarray,
) -> np.ndarray:
    scaler = getattr(dataset, "substitution_mean_scaler", dataset.scaler)
    return dataset._scale_extra_cube(
        stat, np.asarray(raw, dtype=np.float32).reshape(1, -1), scaler, fit=False
    ).reshape(-1)


def scale_substitute_block(
    dataset: SeriesEncodingsDataset,
    block_name: str,
    raw: np.ndarray,
) -> np.ndarray:
    from recomb2027.data.extra_encoding_features import (
        scale_block_array,
    )

    scaler = getattr(dataset, "substitution_block_scaler", None)
    if scaler is None:
        scaler = dataset.block_scaler if dataset.block_scaler is not None else dataset.scaler
    return scale_block_array(
        np.asarray(raw, dtype=np.float32).reshape(1, -1), scaler, fit=False
    ).reshape(-1)


def build_substituted_model_input(
    x_abs: np.ndarray,
    *,
    substitute_encoding: np.ndarray,
    last_idx: int,
    input_feature_mode: str = "absolute",
    edge_distances: np.ndarray | None = None,
    edge_phylo_use_log1p: bool = True,
    extra_prefix: dict[str, np.ndarray] | None = None,
    substitute_extra: dict[str, np.ndarray] | None = None,
    extra_stats: Sequence[str] = (),
    block_prefix: dict[str, np.ndarray] | None = None,
    substitute_blocks: dict[str, np.ndarray] | None = None,
    block_names: Sequence[str] = (),
) -> np.ndarray:
    """Build full model input with the last prefix step replaced (mean + optional stats)."""
    x_abs = np.asarray(x_abs, dtype=np.float32).copy()
    if last_idx != x_abs.shape[0] - 1:
        raise ValueError(
            f"last_idx={last_idx} must index the last row of x_abs (shape {x_abs.shape})"
        )
    x_abs[last_idx] = np.asarray(substitute_encoding, dtype=np.float32).reshape(-1)

    extra_steps = None
    if extra_stats and extra_prefix:
        extra_steps = {s: extra_prefix[s].copy() for s in extra_stats}
        if substitute_extra:
            for stat in extra_stats:
                if stat not in substitute_extra:
                    raise KeyError(
                        f"substitute_extra missing stat {stat!r} required by extra_stats"
                    )
                extra_steps[stat][last_idx] = np.asarray(
                    substitute_extra[stat], dtype=np.float32
                ).reshape(-1)

    block_steps = None
    if block_names and block_prefix:
        block_steps = {n: block_prefix[n].copy() for n in block_names}
        if substitute_blocks:
            for name in block_names:
                if name not in substitute_blocks:
                    raise KeyError(
                        f"substitute_blocks missing block {name!r} required by block_names"
                    )
                block_steps[name][last_idx] = np.asarray(
                    substitute_blocks[name], dtype=np.float32
                ).reshape(-1)

    x = concat_prefix_step_features(
        x_abs,
        extra_steps,
        list(extra_stats),
        input_feature_mode=input_feature_mode,
        block_steps=block_steps,
        block_names=list(block_names),
    )

    if edge_distances is not None:
        edge_d = np.asarray(edge_distances, dtype=np.float32).reshape(-1)
        if edge_d.shape[0] != x_abs.shape[0]:
            raise ValueError(
                f"edge_distances length {edge_d.shape[0]} != prefix length {x_abs.shape[0]}"
            )
        if edge_phylo_use_log1p:
            edge_feat = np.log1p(edge_d, dtype=np.float32).reshape(-1, 1)
        else:
            edge_feat = edge_d.reshape(-1, 1)
        x = np.concatenate([x, edge_feat], axis=-1)

    return x.astype(np.float32)
