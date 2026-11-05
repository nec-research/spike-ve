#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     substitution_sidecar_store.py
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
"""Load B1 substitution sidecar and materialize prefix / target site features."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from recomb2027.latent.data_series.msa_substitution_features import (
    PREFIX_FEATURE_DIM,
    load_substitution_sidecar,
    materialize_prefix_features,
)
from recomb2027.latent.data_series.msa_substitution_features import (
    PrefixSubstitutionState,
)

__all__ = [
    "PREFIX_FEATURE_DIM",
    "SubstitutionSidecarStore",
    "resolve_substitution_sidecar_path",
]


def resolve_substitution_sidecar_path(
    data_cfg,
    *,
    split: str,
    override: Optional[str] = None,
) -> Path:
    if override:
        return Path(str(override)).expanduser()
    processed = Path(str(data_cfg["processed_dir"]))
    split = str(split)
    candidates = [processed / f"msa_substitution_sidecar_{split}.npz"]
    # Dataset leaf split is often "val" while B1 writes "..._test.npz" (or vice versa).
    if split in ("test", "val"):
        alt = "val" if split == "test" else "test"
        candidates.append(processed / f"msa_substitution_sidecar_{alt}.npz")
    for p in candidates:
        if p.is_file():
            return p
    return candidates[0]


class SubstitutionSidecarStore:
    """Fast lookup of prefix features and target activation masks per (sample, step)."""

    def __init__(self, payload: dict[str, Any]):
        self.seq_len = int(payload["seq_len"])
        self.max_steps = int(payload.get("max_steps", 0))
        self.n_samples = int(payload.get("n_samples", 0))

        self._prefix_entry_ptr = np.asarray(payload["prefix_entry_ptr"], dtype=np.int64)
        self._prefix_entry_sample = np.asarray(
            payload["prefix_entry_sample"], dtype=np.int32
        )
        self._prefix_entry_step = np.asarray(payload["prefix_entry_step"], dtype=np.uint8)
        self._prefix_site = np.asarray(payload["prefix_site"], dtype=np.int16)
        self._prefix_from = np.asarray(payload["prefix_from"], dtype=np.uint8)
        self._prefix_to = np.asarray(payload["prefix_to"], dtype=np.uint8)
        self._prefix_cum = np.asarray(payload["prefix_cum_count"], dtype=np.uint16)
        self._prefix_since = np.asarray(payload["prefix_steps_since"], dtype=np.uint8)
        self._prefix_sub_last = np.asarray(payload["prefix_sub_last"], dtype=np.uint8)

        self._target_entry_ptr = np.asarray(payload["target_entry_ptr"], dtype=np.int64)
        self._target_entry_sample = np.asarray(
            payload["target_entry_sample"], dtype=np.int32
        )
        self._target_entry_step = np.asarray(payload["target_entry_step"], dtype=np.uint8)
        self._target_site = np.asarray(payload["target_site"], dtype=np.int16)
        self._has_target_aa = "target_from" in payload and "target_to" in payload
        if self._has_target_aa:
            self._target_from = np.asarray(payload["target_from"], dtype=np.uint8)
            self._target_to = np.asarray(payload["target_to"], dtype=np.uint8)
        else:
            self._target_from = np.asarray([], dtype=np.uint8)
            self._target_to = np.asarray([], dtype=np.uint8)

        self._prefix_index: dict[tuple[int, int], int] = {}
        for ei in range(len(self._prefix_entry_sample)):
            key = (int(self._prefix_entry_sample[ei]), int(self._prefix_entry_step[ei]))
            self._prefix_index[key] = ei

        self._target_index: dict[tuple[int, int], tuple[int, int]] = {}
        n_tgt = len(self._target_entry_sample)
        for ti in range(n_tgt):
            key = (
                int(self._target_entry_sample[ti]),
                int(self._target_entry_step[ti]),
            )
            lo = int(self._target_entry_ptr[ti])
            hi = int(self._target_entry_ptr[ti + 1])
            self._target_index[key] = (lo, hi)

    @classmethod
    def from_path(cls, path: Union[str, Path]) -> "SubstitutionSidecarStore":
        return cls(load_substitution_sidecar(path))

    def n_transition_steps(self, sample_idx: int) -> int:
        """Number of y-steps (transitions) for a sample.

        Prefix entries are stored at steps ``0..n_trans`` (inclusive), so the
        max prefix step index equals ``n_trans``. Falling back to the global
        ``max_steps`` would over-generate features for short / empty samples.
        """
        steps = [s for (si, s) in self._prefix_index if si == int(sample_idx)]
        if not steps:
            return 0
        return int(max(steps))

    def _prefix_state(self, sample_idx: int, step_idx: int) -> PrefixSubstitutionState:
        key = (int(sample_idx), int(step_idx))
        ei = self._prefix_index.get(key)
        st = PrefixSubstitutionState.empty(self.seq_len)
        if ei is None:
            return st
        lo = int(self._prefix_entry_ptr[ei])
        hi = int(self._prefix_entry_ptr[ei + 1])
        for j in range(lo, hi):
            site = int(self._prefix_site[j])
            st.substituted_last[site] = int(self._prefix_sub_last[j])
            st.cumulative_count[site] = int(self._prefix_cum[j])
            st.steps_since_last[site] = int(self._prefix_since[j])
            st.last_from[site] = int(self._prefix_from[j])
            st.last_to[site] = int(self._prefix_to[j])
        return st

    def prefix_features(self, sample_idx: int, step_idx: int) -> np.ndarray:
        """Dense prefix-only features ``[L, F]`` for predicting step ``step_idx``."""
        st = self._prefix_state(sample_idx, step_idx)
        return materialize_prefix_features(st, seq_len=self.seq_len)

    def prefix_features_all_steps(self, sample_idx: int) -> list[np.ndarray]:
        n = self.n_transition_steps(sample_idx)
        return [self.prefix_features(sample_idx, t) for t in range(n + 1)]

    def target_activation_mask(self, sample_idx: int, step_idx: int) -> np.ndarray:
        """Binary target mask ``[L]`` for auxiliary activation (label only, not input)."""
        mask = np.zeros(self.seq_len, dtype=np.float32)
        key = (int(sample_idx), int(step_idx))
        span = self._target_index.get(key)
        if span is None:
            return mask
        lo, hi = span
        for j in range(lo, hi):
            mask[int(self._target_site[j])] = 1.0
        return mask

    def target_activation_mask_span(
        self, sample_idx: int, lo_step: int, hi_step: int
    ) -> np.ndarray:
        """Binary mask ``[L]`` of sites that differ between genome ``lo_step`` and ``hi_step``.

        Used when epoch subsampling drops interior timepoints so that one supervised
        transition spans several sidecar edges ``lo_step -> ... -> hi_step``. With target
        amino-acid indices the *net* change is used (a site mutated and then reverted
        inside the span is not active); otherwise the union of edited sites is returned.
        """
        lo, hi = int(lo_step), int(hi_step)
        if hi - lo <= 1:
            return self.target_activation_mask(sample_idx, lo)
        if not self._has_target_aa:
            mask = np.zeros(self.seq_len, dtype=np.float32)
            for step in range(lo, hi):
                mask = np.maximum(mask, self.target_activation_mask(sample_idx, step))
            return mask
        net: dict[int, list[int]] = {}
        for step in range(lo, hi):
            for site, from_idx, to_idx in self.target_substitutions(sample_idx, step):
                if site in net:
                    net[site][1] = to_idx
                else:
                    net[site] = [from_idx, to_idx]
        mask = np.zeros(self.seq_len, dtype=np.float32)
        for site, (from_idx, to_idx) in net.items():
            if from_idx != to_idx and 0 <= site < self.seq_len:
                mask[site] = 1.0
        return mask

    def target_activation_all_steps(self, sample_idx: int) -> list[np.ndarray]:
        n = self.n_transition_steps(sample_idx)
        return [self.target_activation_mask(sample_idx, t) for t in range(n)]

    def target_substitutions(
        self, sample_idx: int, step_idx: int
    ) -> list[tuple[int, int, int]]:
        """Ground-truth substitutions ``[(site, from_idx, to_idx), ...]`` for one edge."""
        if not self._has_target_aa:
            raise KeyError(
                "Sidecar missing target_from/target_to arrays; rebuild with "
                "recomb2027.latent.predictors.build_msa_substitution_sidecar"
            )
        key = (int(sample_idx), int(step_idx))
        span = self._target_index.get(key)
        if span is None:
            return []
        lo, hi = span
        return [
            (
                int(self._target_site[j]),
                int(self._target_from[j]),
                int(self._target_to[j]),
            )
            for j in range(lo, hi)
        ]

    def target_child_tokens(
        self,
        sample_idx: int,
        step_idx: int,
        parent_tokens: np.ndarray,
    ) -> np.ndarray:
        """Copy parent tokens and apply target substitutions → ``int64 [L]``."""
        child = np.asarray(parent_tokens, dtype=np.int64).reshape(-1).copy()
        for site, _from_idx, to_idx in self.target_substitutions(
            sample_idx, step_idx
        ):
            si = int(site)
            if 0 <= si < child.shape[0]:
                child[si] = int(to_idx)
        return child
