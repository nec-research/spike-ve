#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     mutation_priors.py
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
"""Mutation prior artifact for typed GA operators (substitution / insertion / deletion)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

GAP_CHARS = frozenset({"-", "."})
STANDARD_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
MUTATION_PRIORS_FORMAT = "mutation_priors_v1"


def is_gap(char: str) -> bool:
    return char in GAP_CHARS


def is_residue(char: str) -> bool:
    return char not in GAP_CHARS


@dataclass
class InsertionBoundaryPrior:
    """Insertion hotspot anchored to an ungapped boundary in the reference."""

    boundary: int
    msa_columns: List[int]
    start_weight: float = 0.0
    length_prior: Dict[int, float] = field(default_factory=dict)
    haplotypes: List[str] = field(default_factory=list)
    occupancy_patterns: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "boundary": int(self.boundary),
            "msa_columns": [int(c) for c in self.msa_columns],
            "start_weight": float(self.start_weight),
            "length_prior": {int(k): float(v) for k, v in self.length_prior.items()},
            "haplotypes": list(self.haplotypes),
            "occupancy_patterns": list(self.occupancy_patterns),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InsertionBoundaryPrior":
        length_prior = {int(k): float(v) for k, v in data.get("length_prior", {}).items()}
        return cls(
            boundary=int(data["boundary"]),
            msa_columns=[int(c) for c in data["msa_columns"]],
            start_weight=float(data.get("start_weight", 0.0)),
            length_prior=length_prior,
            haplotypes=list(data.get("haplotypes", [])),
            occupancy_patterns=list(data.get("occupancy_patterns", [])),
        )


@dataclass
class MutationPriors:
    """Precomputed operator priors for typed mutant generation."""

    meta: Dict[str, Any]
    operator_prior: Dict[str, float]
    substitution_position_weights: np.ndarray
    deletion_start_weights: np.ndarray
    deletion_length_prior_global: Dict[int, float]
    insertion_boundaries: List[InsertionBoundaryPrior]
    max_deletion_length: int = 10
    max_insertion_length: int = 5

    @property
    def alignment_width(self) -> int:
        return int(len(self.substitution_position_weights))

    def validate(self) -> List[str]:
        errors: List[str] = []
        width = self.alignment_width

        if width == 0:
            errors.append("substitution_position_weights is empty")
            return errors

        for name, arr in (
            ("substitution_position_weights", self.substitution_position_weights),
            ("deletion_start_weights", self.deletion_start_weights),
        ):
            if len(arr) != width:
                errors.append(f"{name} length {len(arr)} != alignment width {width}")

        op = self.operator_prior
        for key in ("substitution", "insertion", "deletion"):
            if key not in op:
                errors.append(f"operator_prior missing '{key}'")
        if op:
            total = sum(float(op.get(k, 0.0)) for k in ("substitution", "insertion", "deletion"))
            if not np.isclose(total, 1.0, atol=1e-4):
                errors.append(f"operator_prior must sum to 1, got {total}")

        for b in self.insertion_boundaries:
            if not b.msa_columns:
                errors.append(f"insertion boundary {b.boundary} has empty msa_columns")
            if any(c < 0 or c >= width for c in b.msa_columns):
                errors.append(
                    f"insertion boundary {b.boundary} has out-of-range msa_columns"
                )

        if not self.deletion_length_prior_global:
            errors.append("deletion_length_prior_global is empty")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "_format": MUTATION_PRIORS_FORMAT,
            "meta": dict(self.meta),
            "operator_prior": {k: float(v) for k, v in self.operator_prior.items()},
            "substitution_position_weights": self.substitution_position_weights.astype(float).tolist(),
            "deletion_start_weights": self.deletion_start_weights.astype(float).tolist(),
            "deletion_length_prior_global": {
                int(k): float(v) for k, v in self.deletion_length_prior_global.items()
            },
            "insertion_boundaries": [b.to_dict() for b in self.insertion_boundaries],
            "max_deletion_length": int(self.max_deletion_length),
            "max_insertion_length": int(self.max_insertion_length),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MutationPriors":
        return cls(
            meta=dict(data.get("meta", {})),
            operator_prior={k: float(v) for k, v in data["operator_prior"].items()},
            substitution_position_weights=np.asarray(
                data["substitution_position_weights"], dtype=np.float64
            ),
            deletion_start_weights=np.asarray(data["deletion_start_weights"], dtype=np.float64),
            deletion_length_prior_global={
                int(k): float(v) for k, v in data["deletion_length_prior_global"].items()
            },
            insertion_boundaries=[
                InsertionBoundaryPrior.from_dict(b) for b in data.get("insertion_boundaries", [])
            ],
            max_deletion_length=int(data.get("max_deletion_length", 10)),
            max_insertion_length=int(data.get("max_insertion_length", 5)),
        )


def save_mutation_priors(priors: MutationPriors, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = priors.to_dict()
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)


def load_mutation_priors(path: str | Path) -> MutationPriors:
    with open(path, "r") as f:
        data = json.load(f)
    if data.get("_format") != MUTATION_PRIORS_FORMAT:
        raise ValueError(
            f"Unsupported mutation priors format: {data.get('_format')!r}; "
            f"expected {MUTATION_PRIORS_FORMAT!r}"
        )
    priors = MutationPriors.from_dict(data)
    errors = priors.validate()
    if errors:
        raise ValueError("Invalid mutation priors: " + "; ".join(errors))
    return priors


def mask_and_renormalize(
    weights: np.ndarray, mask: np.ndarray, min_weight: float = 0.0
) -> Optional[np.ndarray]:
    """Apply a boolean mask and renormalize to a probability vector."""
    w = np.asarray(weights, dtype=np.float64).copy()
    m = np.asarray(mask, dtype=bool)
    if w.shape != m.shape:
        raise ValueError(f"weights shape {w.shape} != mask shape {m.shape}")
    w[~m] = 0.0
    if min_weight > 0:
        w[m & (w <= 0)] = min_weight
    total = float(w.sum())
    if total <= 0:
        return None
    return w / total


def ungapped_length(sequence: str) -> int:
    return sum(1 for c in sequence if is_residue(c))


def boundary_from_msa_column(reference: str, column: int) -> int:
    """Ungapped boundary index before the residue/gap block at *column*."""
    if column < 0 or column >= len(reference):
        raise IndexError(column)
    return sum(1 for c in reference[:column] if is_residue(c))


def discover_insertion_blocks(reference: str) -> List[Tuple[int, List[int]]]:
    """Return (ungapped_boundary, msa_columns) for each reference gap block."""
    blocks: List[Tuple[int, List[int]]] = []
    i = 0
    n = len(reference)
    while i < n:
        if is_gap(reference[i]):
            boundary = boundary_from_msa_column(reference, i)
            cols: List[int] = []
            while i < n and is_gap(reference[i]):
                cols.append(i)
                i += 1
            blocks.append((boundary, cols))
        else:
            i += 1
    return blocks
