#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     ancestor_reversion.py
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
"""Asymmetric branch / non-reversion penalty for ``by_nearest_ancestor``.

Why
---
Every edge of the series is between two *observed leaves* ``P`` (candidate parent) and ``F`` (focal child).
With ``M = LCA(P, F)`` the transition ``P -> F`` is the composition of "undo P's private terminal branch"
(``M -> P`` reversed) and "walk F's own branch" (``M -> F``).  The plain patristic distance

    Delta = b_F + b_P,   b_F = d(F) - d(M),   b_P = d(P) - d(M)

treats both parts alike, so a cousin with private mutations scores as well as a clean ancestor-like leaf, and
roughly half of all observed substitutions become *reversions of the parent's private mutations* (measured on
the current series: ~43 % of the true substitutions restore the parent-lineage consensus residue).

Effective distance
------------------
    d_eff(P -> F) = b_F + alpha * b_P + beta * n_rev(P -> F)

* ``b_P`` (tree units, mutation counts) penalises candidates with a long private terminal branch.  The tree is
  built on the whole genome, so ``b_P`` also counts mutations outside S1: keep ``alpha`` moderate.
* ``n_rev`` is the *S1 amino-acid* analogue and the term that matters: the number of alignment columns where
  ``P`` and ``F`` differ (both non-gap) and ``F``'s residue equals the reference residue, i.e. ``F`` restores

      - the consensus residue of ``P``'s lineage (column conserved in that lineage, freq >= ``min_cons_freq``), and/or
      - the root (Wuhan) residue.

  ``reference`` selects ``consensus`` / ``root`` / ``both`` (union, default).

Scoring all candidates by ``n_rev`` would cost O(candidates * L); instead the tree-only key
``b_F + alpha * b_P`` pre-selects a shortlist of ``shortlist`` candidates and ``n_rev`` is computed (vectorised)
only for those.  If every shortlisted candidate has a reversion the best ``d_eff`` is still chosen, so **no focal
leaf is dropped** (optional hard cap: ``max_n_rev``).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

AA_ALPHABET = "-ACDEFGHIKLMNPQRSTVWY"
_REFERENCES = ("consensus", "root", "both")

_LUT = np.zeros(256, dtype=np.uint8)
for _i, _c in enumerate(AA_ALPHABET):
    _LUT[ord(_c)] = _i
    _LUT[ord(_c.lower())] = _i


@dataclass(frozen=True)
class ReversionPenaltyConfig:
    """Parameters of the asymmetric / non-reversion candidate score."""

    enabled: bool = True
    alpha: float = 2.0
    beta: float = 5.0
    reference: str = "both"
    min_cons_freq: float = 0.8
    min_lineage_n: int = 10
    shortlist: int = 100
    max_n_rev: Optional[int] = None
    root_gid: str = "MAT_NODE_1"
    seq_len: Optional[int] = None
    # Minimum tree branch of the child (whole-genome mutation units); 0 = off.
    min_b_child: float = 0.0
    # Slope of the lower hinge on the tree term; must be >= 1 so every candidate key is >= min_b_child
    # (1 = flat floor at min_b_child, 2 = V-shape).  Ignored when min_b_child == 0.
    gamma_tree: float = 2.0
    # Minimum forward S1 amino-acid substitutions P -> C (= Hamming_S1 - n_rev); 0 = off.
    min_n_fwd: int = 0
    # Soft penalty per missing forward substitution below min_n_fwd.
    gamma_fwd: float = 6.0

    # ---- tree term / lower bound used by the stem walk --------------------------------------------
    def tree_term(self, b_child: float) -> float:
        """``T(b) = b + gamma_tree * max(0, b_min - b)``; equals ``b`` when ``b_min == 0``."""
        if self.min_b_child <= 0.0:
            return float(b_child)
        return float(b_child) + self.gamma_tree * max(0.0, self.min_b_child - float(b_child))

    def tree_lower_bound(self, b_child: float) -> float:
        """Lower bound of ``T(b')`` over all ``b' >= b_child`` (valid because deeper LCAs have larger b_child).

        ``min_{b' >= b} T(b') = min(T(b), max(b_min, b))`` for ``gamma_tree >= 1``.
        """
        if self.min_b_child <= 0.0:
            return float(b_child)
        return min(self.tree_term(b_child), max(float(self.min_b_child), float(b_child)))

    def fwd_penalty(self, n_fwd: int) -> float:
        """``gamma_fwd * max(0, m_min - n_fwd)``."""
        if self.min_n_fwd <= 0:
            return 0.0
        return float(self.gamma_fwd) * max(0, int(self.min_n_fwd) - int(n_fwd))

    @classmethod
    def from_mapping(cls, raw: Optional[Mapping[str, Any]]) -> Optional["ReversionPenaltyConfig"]:
        """``None`` / ``enabled: false`` -> ``None`` (legacy scoring)."""
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise ValueError("reversion_penalty must be a mapping/object")
        enabled = bool(raw.get("enabled", True))
        if not enabled:
            return None
        alpha = float(raw.get("alpha", 2.0))
        beta = float(raw.get("beta", 5.0))
        if alpha < 0:
            raise ValueError(f"reversion_penalty.alpha must be >= 0, got {alpha}")
        if beta < 0:
            raise ValueError(f"reversion_penalty.beta must be >= 0, got {beta}")
        reference = str(raw.get("reference", "both")).strip().lower()
        if reference not in _REFERENCES:
            raise ValueError(f"reversion_penalty.reference must be one of {_REFERENCES}; got {reference!r}")
        min_cons_freq = float(raw.get("min_cons_freq", 0.8))
        if not 0.5 < min_cons_freq <= 1.0:
            raise ValueError("reversion_penalty.min_cons_freq must be in (0.5, 1]")
        min_lineage_n = int(raw.get("min_lineage_n", 10))
        if min_lineage_n < 1:
            raise ValueError("reversion_penalty.min_lineage_n must be >= 1")
        shortlist = int(raw.get("shortlist", 100))
        if shortlist < 1:
            raise ValueError("reversion_penalty.shortlist must be >= 1")
        mnr = raw.get("max_n_rev", None)
        max_n_rev = None if mnr is None else int(mnr)
        sl = raw.get("seq_len", None)
        min_b_child = float(raw.get("min_b_child", 0.0))
        if min_b_child < 0:
            raise ValueError(f"reversion_penalty.min_b_child must be >= 0, got {min_b_child}")
        gamma_tree = float(raw.get("gamma_tree", 2.0))
        if min_b_child > 0 and gamma_tree < 1.0:
            raise ValueError(
                "reversion_penalty.gamma_tree must be >= 1 when min_b_child > 0 "
                f"(otherwise close candidates can rank below the floor), got {gamma_tree}"
            )
        min_n_fwd = int(raw.get("min_n_fwd", 0))
        if min_n_fwd < 0:
            raise ValueError(f"reversion_penalty.min_n_fwd must be >= 0, got {min_n_fwd}")
        gamma_fwd = float(raw.get("gamma_fwd", 6.0))
        if gamma_fwd < 0:
            raise ValueError(f"reversion_penalty.gamma_fwd must be >= 0, got {gamma_fwd}")
        return cls(
            enabled=True,
            alpha=alpha,
            beta=beta,
            reference=reference,
            min_cons_freq=min_cons_freq,
            min_lineage_n=min_lineage_n,
            shortlist=shortlist,
            max_n_rev=max_n_rev,
            root_gid=str(raw.get("root_gid", "MAT_NODE_1")),
            seq_len=None if sl is None else int(sl),
            min_b_child=min_b_child,
            gamma_tree=gamma_tree,
            min_n_fwd=min_n_fwd,
            gamma_fwd=gamma_fwd,
        )

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)


def encode_sequence(seq: str, seq_len: int) -> np.ndarray:
    """Aligned amino-acid string -> int8 tokens ``[seq_len]`` (gap / unknown = 0, padded / truncated)."""
    raw = np.frombuffer(seq.encode("latin-1", "ignore"), dtype=np.uint8)
    tok = _LUT[raw].astype(np.int8)
    if tok.shape[0] >= seq_len:
        return tok[:seq_len]
    out = np.zeros(seq_len, dtype=np.int8)
    out[: tok.shape[0]] = tok
    return out


class ReversionIndex:
    """Token matrix + lineage consensus used to count reversions ``n_rev(P -> F)``."""

    def __init__(
        self,
        tok: np.ndarray,
        gid_to_row: Dict[str, int],
        gid_to_lineage: Mapping[str, str],
        cfg: ReversionPenaltyConfig,
        root: Optional[np.ndarray] = None,
    ):
        self.tok = tok
        self.gid_to_row = gid_to_row
        self.gid_to_lineage = dict(gid_to_lineage)
        self.cfg = cfg
        self.root = root
        self.seq_len = int(tok.shape[1])
        self._zero_cons = np.zeros(self.seq_len, dtype=np.int8)
        self._false = np.zeros(self.seq_len, dtype=bool)
        self.exact: Dict[str, Tuple[np.ndarray, np.ndarray, int]] = {}
        self._ref_cache: Dict[str, Optional[Tuple[np.ndarray, np.ndarray]]] = {}
        self._build_consensus()

    # ------------------------------------------------------------------ build
    def _build_consensus(self) -> None:
        if self.cfg.reference == "root":
            return
        by_lin: Dict[str, List[int]] = defaultdict(list)
        for gid, row in self.gid_to_row.items():
            lin = self.gid_to_lineage.get(gid)
            if isinstance(lin, str) and lin and lin != "nan":
                by_lin[lin].append(row)
        for lin, rows in by_lin.items():
            if len(rows) < self.cfg.min_lineage_n:
                continue
            M = self.tok[rows]
            cnt = np.stack([(M == a).sum(0) for a in range(len(AA_ALPHABET))])  # [21, L]
            self.exact[lin] = (
                cnt.argmax(0).astype(np.int8),
                (cnt.max(0) / len(rows)) >= self.cfg.min_cons_freq,
                len(rows),
            )
        logger.info(
            "reversion index: consensus for %d lineages (n>=%d) from %d genomes; root=%s; reference=%s",
            len(self.exact), self.cfg.min_lineage_n, len(self.gid_to_row),
            "yes" if self.root is not None else "no", self.cfg.reference,
        )

    def lineage_ref(self, lin: Optional[str]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """``(consensus[L], conserved_mask[L])`` of a lineage, backing off to dotted parents."""
        if not isinstance(lin, str) or not lin:
            return None
        if lin in self._ref_cache:
            return self._ref_cache[lin]
        ref: Optional[Tuple[np.ndarray, np.ndarray]] = None
        parts = lin.split(".")
        while parts:
            key = ".".join(parts)
            hit = self.exact.get(key)
            if hit is not None:
                ref = (hit[0], hit[1])
                break
            parts = parts[:-1]
        self._ref_cache[lin] = ref
        return ref

    # ------------------------------------------------------------------ query
    def has(self, gid: str) -> bool:
        return gid in self.gid_to_row

    def n_rev(self, focal_gid: str, cand_gids: Sequence[str]) -> Tuple[np.ndarray, int]:
        """Reversion counts of ``cand -> focal`` for every candidate (+ #candidates without sequence)."""
        n_rev, _n_fwd, missing = self.score(focal_gid, cand_gids)
        return n_rev, missing

    def score(self, focal_gid: str, cand_gids: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, int]:
        """``(n_rev, n_fwd, n_missing)`` per candidate ``cand -> focal``.

        ``n_fwd = Hamming_S1(cand, focal) - n_rev`` is the number of *forward* S1 amino-acid substitutions
        (valid columns where the focal residue is not the reference residue).  Candidates without sequence
        get ``n_rev = n_fwd = 0``.
        """
        k = len(cand_gids)
        f_row = self.gid_to_row.get(focal_gid)
        if f_row is None or k == 0:
            return np.zeros(k, dtype=np.int32), np.zeros(k, dtype=np.int32), k
        rows = np.array([self.gid_to_row.get(g, -1) for g in cand_gids], dtype=np.int64)
        missing = rows < 0
        P = self.tok[np.where(missing, 0, rows)].copy()
        P[missing] = 0                                    # all-gap => no valid column => n_rev = 0
        C = self.tok[f_row]
        valid = (P != 0) & (C[None, :] != 0) & (P != C[None, :])
        hit = np.zeros_like(valid)
        if self.cfg.reference in ("consensus", "both"):
            cons = np.empty((k, self.seq_len), dtype=np.int8)
            pure = np.zeros((k, self.seq_len), dtype=bool)
            for i, g in enumerate(cand_gids):
                ref = self.lineage_ref(self.gid_to_lineage.get(g))
                if ref is None:
                    cons[i] = self._zero_cons
                else:
                    cons[i], pure[i] = ref
            hit |= valid & pure & (cons == C[None, :])
        if self.cfg.reference in ("root", "both") and self.root is not None:
            hit |= valid & (self.root != 0)[None, :] & (self.root[None, :] == C[None, :])
        n_rev = hit.sum(1).astype(np.int32)
        n_fwd = (valid.sum(1) - n_rev).astype(np.int32)
        return n_rev, n_fwd, int(missing.sum())


def build_reversion_index(
    seq_lookup: Mapping[str, str],
    gid_to_lineage: Mapping[str, str],
    cfg: ReversionPenaltyConfig,
) -> ReversionIndex:
    """Encode the aligned S1 sequences and precompute lineage consensus."""
    if not seq_lookup:
        raise ValueError(
            "reversion_penalty needs genome sequences (seq_lookup is empty): load the MSA via msa_path."
        )
    if cfg.seq_len is not None:
        L = int(cfg.seq_len)
    else:
        lens = np.fromiter((len(s) for s in seq_lookup.values()), dtype=np.int64)
        L = int(np.bincount(lens).argmax())
    gids = list(seq_lookup.keys())
    tok = np.zeros((len(gids), L), dtype=np.int8)
    for i, g in enumerate(gids):
        tok[i] = encode_sequence(seq_lookup[g], L)
    gid_to_row = {g: i for i, g in enumerate(gids)}

    root = None
    root_key = next((g for g in gids if g == cfg.root_gid), None) or next(
        (g for g in gids if cfg.root_gid in g), None
    )
    if root_key is not None:
        root = tok[gid_to_row[root_key]].copy()
        logger.info("reversion index: root genome = %s", root_key)
    elif cfg.reference in ("root", "both"):
        logger.warning(
            "reversion_penalty: root genome %r not found in sequences; root reference disabled "
            "(consensus only).", cfg.root_gid,
        )
    if root is None and cfg.reference == "root":
        raise ValueError(f"reversion_penalty.reference='root' but root genome {cfg.root_gid!r} not found")
    return ReversionIndex(tok, gid_to_row, gid_to_lineage, cfg, root=root)
