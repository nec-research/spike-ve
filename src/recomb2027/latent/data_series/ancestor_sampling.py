#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     ancestor_sampling.py
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
"""Weighted predecessor sampling for ``by_nearest_ancestor``."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

NS_PER_DAY = 86_400 * 1_000_000_000


@dataclass(frozen=True)
class AncestorSamplingConfig:
    """How to pick one predecessor from eligible nested-clade candidates."""

    selection: str = "nearest"
    top_m: int = 10
    tau_patristic: float = 3.0
    patristic_band: float = 3.0
    tau_days: Optional[float] = 120.0
    seed: int = 0

    @classmethod
    def from_mapping(cls, raw: Optional[Mapping[str, Any]]) -> "AncestorSamplingConfig":
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise ValueError("predecessor_sampling must be a mapping/object")
        selection = str(raw.get("selection", "nearest")).strip().lower()
        if selection not in ("nearest", "sample"):
            raise ValueError(
                "predecessor_sampling.selection must be 'nearest' or 'sample'; "
                f"got {selection!r}"
            )
        top_m = int(raw.get("top_m", 10))
        if top_m < 1:
            raise ValueError(f"predecessor_sampling.top_m must be >= 1, got {top_m}")
        tau_patristic = float(raw.get("tau_patristic", 3.0))
        if tau_patristic <= 0:
            raise ValueError("predecessor_sampling.tau_patristic must be > 0")
        patristic_band = float(raw.get("patristic_band", 3.0))
        if patristic_band < 0:
            raise ValueError("predecessor_sampling.patristic_band must be >= 0")
        tau_days_raw = raw.get("tau_days", 120.0)
        tau_days = None if tau_days_raw is None else float(tau_days_raw)
        if tau_days is not None and tau_days <= 0:
            raise ValueError("predecessor_sampling.tau_days must be > 0 or null")
        seed = int(raw.get("seed", 0))
        return cls(
            selection=selection,
            top_m=top_m,
            tau_patristic=tau_patristic,
            patristic_band=patristic_band,
            tau_days=tau_days,
            seed=seed,
        )

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)


CandidateRecord = Tuple[str, float, int]


def _rng_for_focal(seed: int, cur_gid: str) -> np.random.Generator:
    """Stable per-focal RNG (reproducible across parallel workers)."""
    mix = hash((int(seed), str(cur_gid))) & 0xFFFFFFFFFFFFFFFF
    return np.random.default_rng(mix)


def _truncate_top_m(
        eligible: Sequence[CandidateRecord],
        top_m: int,
) -> List[CandidateRecord]:
    ordered = sorted(eligible, key=lambda t: (t[1], t[0]))
    return list(ordered[:top_m])


def sample_predecessor_weighted(
        eligible: Sequence[CandidateRecord],
        *,
        config: AncestorSamplingConfig,
        cur_gid: str,
        cur_date_ns: int,
        predecessor_rank: int = 1,
) -> Optional[str]:
    """Pick one predecessor from eligible ``(gid, patristic, date_ns)`` rows."""
    if not eligible:
        return None

    if config.selection == "nearest":
        ordered = sorted(eligible, key=lambda t: (t[1], t[0]))
        idx = min(max(1, int(predecessor_rank)) - 1, len(ordered) - 1)
        return ordered[idx][0]

    pool = _truncate_top_m(eligible, config.top_m)
    d_min = min(p[1] for p in pool)
    banded = [row for row in pool if row[1] <= d_min + config.patristic_band]
    if not banded:
        banded = pool

    scores: List[float] = []
    for _gid, pdist, cand_ns in banded:
        dt_days = max(0.0, (cur_date_ns - cand_ns) / NS_PER_DAY)
        score = -pdist / config.tau_patristic
        if config.tau_days is not None:
            score -= dt_days / config.tau_days
        scores.append(score)

    logits = np.asarray(scores, dtype=np.float64)
    logits -= logits.max()
    weights = np.exp(logits)
    total = weights.sum()
    if total <= 0 or not np.isfinite(total):
        ordered = sorted(banded, key=lambda t: (t[1], t[0]))
        return ordered[0][0]

    rng = _rng_for_focal(config.seed, cur_gid)
    idx = int(rng.choice(len(banded), p=weights / total))
    return banded[idx][0]
