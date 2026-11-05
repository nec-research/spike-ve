#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     diffusion_metadata.py
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
"""Target-context metadata for diffusion conditioning (calendar, lineage, country).

The GPT / GAttn encoder summarises the **prefix trajectory**. These fields
describe the **prediction target context** (epidemic period, variant background,
geography) that modulates evolutionary rate at a given horizon Δt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Union

import numpy as np

from recomb2027.latent.data_series.geo_aggregation import (
    GEO_AGGREGATION_CHOICES,
    GeoAggregation,
    METADATA_UNK as _GEO_UNK,
    normalize_geo_label,
)

METADATA_UNK = _GEO_UNK
CALENDAR_EPOCH = "2019-01-01"


def parse_calendar_days(
    value: Union[str, float, int, None],
    *,
    epoch: str = CALENDAR_EPOCH,
) -> float:
    """Convert ISO date or day index to days since *epoch*."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    import pandas as pd

    ts = pd.to_datetime(str(value), errors="coerce")
    if pd.isna(ts):
        return 0.0
    origin = pd.Timestamp(epoch)
    return float((ts - origin).days)


def normalize_lineage_label(raw: object, *, depth: int = 0) -> str:
    """Optional Pango compression (``depth=2`` → ``BA.5`` from ``BA.5.2.1``)."""
    s = str(raw).strip()
    if not s or s.lower() in {"nan", "none", ""}:
        return METADATA_UNK
    if depth <= 0:
        return s
    parts = [p for p in s.split(".") if p]
    if not parts:
        return METADATA_UNK
    return ".".join(parts[:depth])


def normalize_country_label(
    raw: object,
    *,
    geo_aggregation: GeoAggregation = "country",
) -> str:
    return normalize_geo_label(raw, geo_aggregation=geo_aggregation)


def build_str_vocab(
    values: Iterable[str],
    *,
    min_count: int = 1,
) -> Dict[str, int]:
    """Build ``{label: id}`` with ``<unk>`` at index 0."""
    from collections import Counter

    counts = Counter(str(v) for v in values)
    vocab: Dict[str, int] = {METADATA_UNK: 0}
    for label, cnt in sorted(counts.items()):
        if label == METADATA_UNK:
            continue
        if cnt >= min_count:
            vocab[label] = len(vocab)
    return vocab


def vocab_lookup(vocab: Mapping[str, int], label: str) -> int:
    return int(vocab.get(label, vocab.get(METADATA_UNK, 0)))


def load_gid_metadata_maps(
    metadata_path: Union[str, "Path"],
    *,
    lineage_col: str = "lineage",
    country_col: str = "isolation_country",
    epoch: str = CALENDAR_EPOCH,
) -> tuple[dict, dict, dict]:
    """Return ``(gid_to_calendar_days, gid_to_lineage, gid_to_country)``."""
    from pathlib import Path

    import pandas as pd

    path = Path(metadata_path)
    df = pd.read_csv(
        path,
        usecols=["genome_id", "collection_date", lineage_col, country_col],
    )
    dates = pd.to_datetime(df["collection_date"], errors="coerce")
    origin = pd.Timestamp(epoch)
    days = (dates - origin).dt.days.astype(np.float32)
    gids = df["genome_id"].astype(str).to_numpy()
    lineages = df[lineage_col].map(lambda x: str(x).strip()).to_numpy()
    countries = df[country_col].map(lambda x: str(x).strip()).to_numpy()

    valid = ~np.isnan(days)
    gid_to_date = {
        g: float(d) for g, d in zip(gids[valid], days[valid])
    }
    gid_to_lineage = dict(zip(gids, lineages))
    gid_to_country = dict(zip(gids, countries))
    return gid_to_date, gid_to_lineage, gid_to_country


@dataclass(frozen=True)
class DiffusionMetadataVocabs:
    lineage: Dict[str, int]
    country: Dict[str, int]
    lineage_depth: int = 0
    geo_aggregation: GeoAggregation = "country"

    def to_dict(self) -> dict:
        return {
            "lineage": dict(self.lineage),
            "country": dict(self.country),
            "lineage_depth": int(self.lineage_depth),
            "geo_aggregation": str(self.geo_aggregation),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DiffusionMetadataVocabs":
        geo = str(d.get("geo_aggregation", "country"))
        if geo not in GEO_AGGREGATION_CHOICES:
            geo = "country"
        return cls(
            lineage={str(k): int(v) for k, v in d["lineage"].items()},
            country={str(k): int(v) for k, v in d["country"].items()},
            lineage_depth=int(d.get("lineage_depth", 0)),
            geo_aggregation=geo,  # type: ignore[arg-type]
        )


def build_metadata_vocabs_from_gids(
    gids_array: np.ndarray,
    gid_to_lineage: Mapping[str, str],
    gid_to_country: Mapping[str, str],
    *,
    target_idx: int = -1,
    lineage_min_count: int = 1,
    lineage_depth: int = 0,
    geo_aggregation: GeoAggregation = "country",
) -> DiffusionMetadataVocabs:
    """Fit lineage / geography vocabs on target-step labels in *gids_array*."""
    gids = np.asarray(gids_array)
    tgt = gids[:, target_idx].astype(str)
    lineages = [
        normalize_lineage_label(gid_to_lineage.get(g, METADATA_UNK), depth=lineage_depth)
        for g in tgt
    ]
    countries = [
        normalize_country_label(
            gid_to_country.get(g, METADATA_UNK),
            geo_aggregation=geo_aggregation,
        )
        for g in tgt
    ]
    return DiffusionMetadataVocabs(
        lineage=build_str_vocab(lineages, min_count=lineage_min_count),
        country=build_str_vocab(countries, min_count=lineage_min_count),
        lineage_depth=int(lineage_depth),
        geo_aggregation=geo_aggregation,
    )


def compute_series_target_metadata(
    gids_array: np.ndarray,
    gid_to_calendar: Mapping[str, float],
    gid_to_lineage: Mapping[str, str],
    gid_to_country: Mapping[str, str],
    vocabs: DiffusionMetadataVocabs,
    *,
    target_idx: int = -1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-series target context: ``(calendar_days, lineage_id, country_id)`` each ``(N,)``."""
    import pandas as pd

    gids = np.asarray(gids_array)
    tgt = pd.Series(gids[:, target_idx].astype(str), copy=False)
    n = len(tgt)
    depth = vocabs.lineage_depth
    geo = vocabs.geo_aggregation
    unk_lin = vocab_lookup(vocabs.lineage, METADATA_UNK)
    unk_geo = vocab_lookup(vocabs.country, METADATA_UNK)

    cal = tgt.map(gid_to_calendar).fillna(0.0).astype(np.float32).to_numpy()

    lin_labels = tgt.map(gid_to_lineage).fillna(METADATA_UNK).map(
        lambda x: normalize_lineage_label(x, depth=depth)
    )
    lin_ids = (
        lin_labels.map(vocabs.lineage).fillna(unk_lin).astype(np.int64).to_numpy()
    )

    geo_labels = tgt.map(gid_to_country).fillna(METADATA_UNK).map(
        lambda x: normalize_country_label(x, geo_aggregation=geo)
    )
    cty_ids = (
        geo_labels.map(vocabs.country).fillna(unk_geo).astype(np.int64).to_numpy()
    )

    if cal.shape[0] != n or lin_ids.shape[0] != n or cty_ids.shape[0] != n:
        raise RuntimeError(
            f"metadata length mismatch for N={n}: "
            f"cal={cal.shape}, lin={lin_ids.shape}, geo={cty_ids.shape}"
        )
    return cal, lin_ids, cty_ids


def compute_series_step_metadata(
    gids_array: np.ndarray,
    gid_to_calendar: Mapping[str, float],
    gid_to_lineage: Mapping[str, str],
    gid_to_country: Mapping[str, str],
    vocabs: DiffusionMetadataVocabs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-step metadata arrays ``(N, K)`` for variable target indices."""
    gids = np.asarray(gids_array)
    if gids.ndim != 2:
        raise ValueError(f"gids_array must be (N, K); got {gids.shape}")
    n, k = gids.shape
    cal = np.zeros((n, k), dtype=np.float32)
    lin = np.zeros((n, k), dtype=np.int64)
    cty = np.zeros((n, k), dtype=np.int64)
    for step in range(k):
        c, l, g = compute_series_target_metadata(
            gids,
            gid_to_calendar,
            gid_to_lineage,
            gid_to_country,
            vocabs,
            target_idx=step,
        )
        cal[:, step] = c
        lin[:, step] = l
        cty[:, step] = g
    return cal, lin, cty
