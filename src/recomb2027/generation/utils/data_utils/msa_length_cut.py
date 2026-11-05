#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     msa_length_cut.py
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
"""Truncate MSA-aligned sequences, position weights, and mutation priors.

Used to run GA ablations on a prefix/slice of the alignment (e.g. S1 only:
MSA columns ``[0, 714)`` through ``S1_C`` in ``spike_block_boundaries.yaml``).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import yaml

from recomb2027.generation.utils.data_utils.mutation_priors import (
    InsertionBoundaryPrior,
    MutationPriors,
)

# Spike MSA presets (0-based half-open), matching
# configs/latent/shared/spike_block_boundaries.yaml defaults.
_DEFAULT_REGION_SLICES: Dict[str, Tuple[int, int]] = {
    "S1": (0, 714),       # N-term through S1_C (excludes S2)
    "S2": (714, 1300),
    "RBD": (347, 570),    # RBD_a..RBD_c
    "full": (0, 1300),
}

ArrayLike = Union[np.ndarray, Sequence[float]]


def resolve_msa_slice(
    *,
    sequence_length: Optional[int] = None,
    sequence_start: int = 0,
    sequence_region: Optional[str] = None,
    spike_boundaries_path: Optional[Path] = None,
) -> Optional[Tuple[int, int]]:
    """Return ``(start, end)`` half-open MSA slice, or ``None`` if no cut.

    Precedence: ``sequence_region`` (if set) overrides ``sequence_start`` /
    ``sequence_length``. Region ``S1`` is MSA ``[0, S1_C.msa_end)``.
    """
    if sequence_region:
        name = str(sequence_region).strip().upper().replace("-", "_")
        aliases = {"FULL": "full"}
        key = aliases.get(name, name)
        # Prefer YAML block boundaries when available
        from_yaml = _region_from_spike_yaml(spike_boundaries_path, preset=key)
        if from_yaml is not None:
            return from_yaml
        if key in _DEFAULT_REGION_SLICES:
            start, end = _DEFAULT_REGION_SLICES[key]
            return int(start), int(end)
        raise ValueError(
            f"Unknown sequence_region={sequence_region!r}. "
            f"Valid: {sorted(_DEFAULT_REGION_SLICES)}"
        )

    if sequence_length is None:
        return None
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be > 0, got {sequence_length}")
    if sequence_start < 0:
        raise ValueError(f"sequence_start must be >= 0, got {sequence_start}")
    return int(sequence_start), int(sequence_start) + int(sequence_length)


def _region_from_spike_yaml(
    path: Optional[Path], *, preset: str
) -> Optional[Tuple[int, int]]:
    if path is None:
        path = (
            Path(__file__).resolve().parents[4]
            / "recomb2027.latent"
            / "config"
            / "spike_block_boundaries.yaml"
        )
    path = Path(path)
    if not path.is_file():
        return None
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    blocks = {b["name"]: b for b in data.get("blocks", [])}
    preset = preset.upper() if preset != "full" else "full"
    if preset == "S1":
        if "S1_C" not in blocks:
            return None
        return 0, int(blocks["S1_C"]["msa_end"])
    if preset == "S2":
        if "S2" not in blocks:
            return None
        return int(blocks["S2"]["msa_start"]), int(blocks["S2"]["msa_end"])
    if preset == "RBD":
        if not {"RBD_a", "RBD_c"}.issubset(blocks):
            return None
        return int(blocks["RBD_a"]["msa_start"]), int(blocks["RBD_c"]["msa_end"])
    if preset == "full":
        msa_len = data.get("msa_length")
        if msa_len is not None:
            return 0, int(msa_len)
    return None


def cut_string(seq: Optional[str], start: int, end: int) -> Optional[str]:
    if seq is None:
        return None
    if end > len(seq):
        raise ValueError(
            f"Cannot cut sequence of length {len(seq)} to slice [{start}, {end})"
        )
    return seq[start:end]


def cut_position_weights(weights: ArrayLike, start: int, end: int) -> np.ndarray:
    arr = np.asarray(weights, dtype=np.float64)
    if end > len(arr):
        raise ValueError(
            f"Cannot cut position weights of length {len(arr)} to slice [{start}, {end})"
        )
    out = arr[start:end].copy()
    if out.sum() <= 0:
        raise ValueError("Cut position weights sum to 0; check sequence_length / region")
    return out


def cut_mutation_priors(priors: MutationPriors, start: int, end: int) -> MutationPriors:
    """Slice position arrays and keep insertion boundaries fully inside the window."""
    width = priors.alignment_width
    if end > width:
        raise ValueError(
            f"Cannot cut mutation priors width={width} to slice [{start}, {end})"
        )
    sub_w = np.asarray(priors.substitution_position_weights, dtype=np.float64)[start:end]
    del_w = np.asarray(priors.deletion_start_weights, dtype=np.float64)[start:end]
    if sub_w.sum() > 0:
        sub_w = sub_w / sub_w.sum()
    if del_w.sum() > 0:
        del_w = del_w / del_w.sum()

    kept: List[InsertionBoundaryPrior] = []
    for b in priors.insertion_boundaries:
        cols = [c - start for c in b.msa_columns if start <= c < end]
        if not cols:
            continue
        # Drop boundaries that only partially overlap the window
        if len(cols) != len(b.msa_columns):
            continue
        kept.append(
            InsertionBoundaryPrior(
                boundary=b.boundary,
                msa_columns=cols,
                start_weight=b.start_weight,
                length_prior=dict(b.length_prior),
                haplotypes=list(b.haplotypes),
                occupancy_patterns=list(b.occupancy_patterns),
            )
        )
    if kept:
        sw = np.asarray([b.start_weight for b in kept], dtype=np.float64)
        if sw.sum() > 0:
            sw = sw / sw.sum()
            kept = [
                replace(b, start_weight=float(sw[i])) for i, b in enumerate(kept)
            ]

    meta = dict(priors.meta)
    meta["msa_cut"] = {"start": int(start), "end": int(end), "width": int(end - start)}

    return MutationPriors(
        meta=meta,
        operator_prior=dict(priors.operator_prior),
        substitution_position_weights=sub_w,
        deletion_start_weights=del_w,
        deletion_length_prior_global=dict(priors.deletion_length_prior_global),
        insertion_boundaries=kept,
        max_deletion_length=priors.max_deletion_length,
        max_insertion_length=priors.max_insertion_length,
    )


def apply_msa_length_cut(
    *,
    source_sequence: str,
    target_sequence: Optional[str],
    pos_weights: ArrayLike,
    sequence_length: Optional[int] = None,
    sequence_start: int = 0,
    sequence_region: Optional[str] = None,
    target_encoding: Any = None,
    spike_boundaries_path: Optional[Path] = None,
) -> Tuple[str, Optional[str], np.ndarray, Any, Optional[Tuple[int, int]]]:
    """Cut source/target strings and position weights; clear target_encoding if cut.

    Returns:
        source_sequence, target_sequence, pos_weights, target_encoding, slice_or_none
    """
    sl = resolve_msa_slice(
        sequence_length=sequence_length,
        sequence_start=sequence_start,
        sequence_region=sequence_region,
        spike_boundaries_path=spike_boundaries_path,
    )
    if sl is None:
        return (
            source_sequence,
            target_sequence,
            np.asarray(pos_weights, dtype=np.float64),
            target_encoding,
            None,
        )

    start, end = sl
    src = cut_string(source_sequence, start, end)
    tgt = cut_string(target_sequence, start, end) if target_sequence is not None else None
    weights = cut_position_weights(pos_weights, start, end)
    assert src is not None
    if len(weights) != len(src):
        raise ValueError(
            f"After cut, pos_weights length {len(weights)} != sequence length {len(src)}"
        )
    if tgt is not None and len(tgt) != len(src):
        raise ValueError(
            f"After cut, target length {len(tgt)} != source length {len(src)}"
        )
    # Precomputed full-length embeddings are invalid for the truncated sequences.
    return src, tgt, weights, None, (start, end)
