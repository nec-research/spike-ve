#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     pango_anchor_path.py
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
"""PANGO milestones + deterministic sequence/tree-capped interpolation.

``PredecessorFinder.by_pango_anchors`` merges each genome's canonical root→focal path
into ``predecessors_dict``. The path mixes PANGO ancestry milestones with optional
bisection intermediates capped by Spike Hamming and (if a tree is loaded) patristic
distance - see :func:`anchored_genome_path_for_focal`.

``pangonet`` is optional: if import succeeds it drives parent lineage order; otherwise
dotted prefixes from ``alias_key`` approximate ancestry.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from recomb2027.latent.data_series import lineage_utils

logger = logging.getLogger(__name__)

_NA_LINEAGE = {"", "n/a", "na", "nan", "none", "unassigned", "unknown", "?"}


def _is_na_lineage(x: Optional[str]) -> bool:
    if x is None:
        return True
    return str(x).strip().lower() in _NA_LINEAGE


def hamming_aa_pair(seq_a: str, seq_b: str) -> Tuple[int, int]:
    muts = 0
    comparable = 0
    for a, b in zip(seq_a, seq_b):
        if a == "-" and b == "-":
            continue
        comparable += 1
        if a != b:
            muts += 1
    return muts, comparable


def normalize_lineage_string(
        name: Optional[str],
        *,
        pango,
        alias_key: Optional[Mapping[str, str]],
) -> str:
    if _is_na_lineage(name):
        return ""
    raw = str(name).strip()
    if pango is not None:
        try:
            u = pango.uncompress(raw)
            return str(pango.compress(u)).strip()
        except Exception:
            pass
    d = lineage_utils.dealias(raw, alias_key if alias_key else None)
    return (d if d else raw).strip()


def dotted_prefix_chain_root_to_tip(
        lineage: Optional[str],
        alias_key: Optional[Mapping[str, str]],
) -> List[str]:
    base = lineage_utils.dealias(lineage, alias_key if alias_key else None)
    parts: List[str] = []
    if base is not None:
        parts = str(base).strip().split(".")
    return [".".join(parts[:i]) for i in range(1, len(parts) + 1)]


def pango_parents_chain_tip_to_root(pango, lineage_compressed: str, *, recomb_pick: str) -> List[str]:
    milestones_rev: List[str] = []
    cur = lineage_compressed.strip()
    seen: set[str] = set()
    while cur and cur not in seen:
        seen.add(cur)
        milestones_rev.append(cur)
        try:
            pars = list(pango.get_parents(cur) or [])
        except Exception:
            pars = []
        if not pars:
            break
        cur = pars[0] if recomb_pick == "first" else min(str(p) for p in pars)
    return list(reversed(milestones_rev))


def ancestry_milestones_root_to_tip(
        lineage_raw: Optional[str],
        *,
        pango,
        alias_key: Optional[Mapping[str, str]],
        fallback_dotted_chain: bool = True,
        recomb_pick: str = "lex_min",
) -> List[str]:
    if _is_na_lineage(lineage_raw):
        return []
    cmp_key = normalize_lineage_string(lineage_raw, pango=pango, alias_key=alias_key)

    milestones: List[str] = []
    used_pango = False
    if pango is not None and cmp_key:
        try:
            milestones = pango_parents_chain_tip_to_root(pango, cmp_key, recomb_pick=recomb_pick)
            used_pango = len(milestones) > 0
        except Exception as e:
            logger.debug("pangonet ancestry failed for %r: %s", lineage_raw, e)

    if not milestones and fallback_dotted_chain:
        milestones = []
        for d in dotted_prefix_chain_root_to_tip(lineage_raw, alias_key if alias_key else None):
            nk = normalize_lineage_string(d, pango=pango, alias_key=alias_key)
            if nk:
                milestones.append(nk)

    dedup: List[str] = []
    for m in milestones:
        if m and (not dedup or dedup[-1] != m):
            dedup.append(m)

    if used_pango and cmp_key:
        if not dedup or dedup[-1] != cmp_key:
            dedup.append(cmp_key)
    elif cmp_key:
        if not dedup:
            dedup = [cmp_key]
        elif dedup[-1] != cmp_key:
            dedup.append(cmp_key)
    return dedup


def canonical_rep_gid(
        lineage_buckets: Mapping[str, Sequence[Tuple[pd.Timestamp, str]]],
        milestone: str,
        *,
        pango,
        alias_key: Optional[Mapping[str, str]],
) -> Optional[str]:
    nk = normalize_lineage_string(milestone, pango=pango, alias_key=alias_key)

    cand = lineage_buckets.get(milestone)
    if cand is None and nk:
        cand = lineage_buckets.get(nk)

    if cand:
        pick_list_sorted = sorted(cand, key=lambda x: (x[0], x[1]))
        return str(pick_list_sorted[0][1])

    best: Optional[Tuple[pd.Timestamp, str]] = None

    for lin_key, seq in lineage_buckets.items():
        if nk and (lin_key == nk or lin_key.startswith(nk + ".")):
            for tup in seq:
                if best is None or tup < best:
                    best = tup

    return str(best[1]) if best else None


class PatristicIndex:
    """Two-leaf distances via PhyloDM from a Dendropy tree."""

    def __init__(self, phylotree, taxa_sep: str = "-") -> None:
        from phylodm import PhyloDM

        pdm = PhyloDM.load_from_dendropy(phylotree)
        self._dm = pdm.dm(norm=False)
        self.gid_to_idx: Dict[str, int] = {}
        for i, lbl in enumerate(pdm.taxa()):
            gid = str(lbl).split(taxa_sep)[-1]
            self.gid_to_idx.setdefault(str(gid), i)

    def __call__(self, a_gid: str, b_gid: str) -> Optional[float]:
        ia = self.gid_to_idx.get(str(a_gid))
        ib = self.gid_to_idx.get(str(b_gid))
        if ia is None or ib is None:
            return None
        return float(self._dm[ia, ib])


def build_lineage_date_buckets(
        df_labels: pd.DataFrame,
        *,
        genome_col: str,
        lineage_col: str,
        timestamp_field: str,
        gid_filter: Optional[set[str]],
        pango,
        alias_key: Optional[Mapping[str, str]],
) -> Tuple[Dict[str, List[Tuple[pd.Timestamp, str]]], Dict[str, pd.Timestamp]]:
    buckets: Dict[str, List[Tuple[pd.Timestamp, str]]] = {}
    gid_to_date: Dict[str, pd.Timestamp] = {}

    cols = [genome_col, lineage_col, timestamp_field]
    subset = df_labels[cols].copy()
    subset[genome_col] = subset[genome_col].astype(str)
    subset["_dt"] = pd.to_datetime(subset[timestamp_field], errors="coerce")

    for _, row in subset.iterrows():
        gid = str(row[genome_col])
        if gid_filter is not None and gid not in gid_filter:
            continue
        dt = row["_dt"]
        if dt is pd.NaT or pd.isna(dt):
            dt = pd.Timestamp.min
        gid_to_date[gid] = dt
        ln = normalize_lineage_string(row[lineage_col], pango=pango, alias_key=alias_key)
        if not ln:
            continue
        buckets.setdefault(ln, []).append((dt, gid))

    for lk in buckets:
        buckets[lk].sort(key=lambda x: (x[0], x[1]))
    return buckets, gid_to_date


def chronological_pool_sorted(
        gid_to_date: Mapping[str, pd.Timestamp],
) -> List[Tuple[pd.Timestamp, str]]:
    return sorted(((d, gid) for gid, d in gid_to_date.items()),
                  key=lambda x: (x[0], x[1]))


def _needs_split(
        gid_a: str,
        gid_b: str,
        seq_lookup: Mapping[str, str],
        *,
        max_h_edge: Optional[int],
        max_pat_edge: Optional[float],
        pat_fn: Optional[Callable[[str, str], Optional[float]]],
) -> Tuple[bool, int]:
    """Return ``(needs_midpoint?, ham_ab)``."""
    sa = seq_lookup.get(gid_a)
    sb = seq_lookup.get(gid_b)
    if sa is None or sb is None or sa == sb:
        return False, 0
    ham, _ = hamming_aa_pair(sa, sb)
    if ham <= 0:
        return False, ham

    if max_h_edge is not None and ham > max_h_edge:
        return True, ham

    if max_pat_edge is not None and pat_fn is not None:
        pv = pat_fn(gid_a, gid_b)
        if pv is None:
            return False, ham
        if pv > max_pat_edge + 1e-9:
            return True, ham

    return False, ham


def _best_midpoint(
        gid_a: str,
        gid_b: str,
        *,
        chronological_pool: Sequence[Tuple[pd.Timestamp, str]],
        seq_lookup: Mapping[str, str],
        gid_to_date: Mapping[str, pd.Timestamp],
        max_h_edge: Optional[int],
        max_pat_edge: Optional[float],
        pat_fn: Optional[Callable[[str, str], Optional[float]]],
        ham_ab_total: int,
        dt_lo: pd.Timestamp,
        dt_hi: pd.Timestamp,
) -> Optional[str]:
    sa = seq_lookup.get(gid_a)
    sb = seq_lookup.get(gid_b)
    if sa is None or sb is None:
        return None

    best: Optional[str] = None
    best_rank: Optional[Tuple[int, int, str]] = None

    for dt_c, cid in chronological_pool:
        if cid in (gid_a, gid_b):
            continue
        if dt_c <= dt_lo or dt_c >= dt_hi:
            continue
        sc = seq_lookup.get(cid)
        if sc is None or sc == sa or sc == sb:
            continue

        ha_c, _ = hamming_aa_pair(sa, sc)
        hc_b, _ = hamming_aa_pair(sc, sb)
        if ha_c <= 0 or hc_b <= 0:
            continue

        wing_max = max(ha_c, hc_b)
        # Require a strictly shorter worst wing than AB so recursion terminates.
        if wing_max >= ham_ab_total:
            continue

        mx = wing_max
        imb = abs(ha_c - hc_b)

        if pat_fn is not None and max_pat_edge is not None:
            pa = pat_fn(gid_a, cid)
            pb = pat_fn(cid, gid_b)
            if pa is None or pb is None:
                continue
            if pa > max_pat_edge + 1e-9 or pb > max_pat_edge + 1e-9:
                continue

        rank = (mx, imb, cid)
        if best_rank is None or rank[:2] < best_rank[:2] or (
                rank[:2] == best_rank[:2] and cid < best_rank[2]):
            best_rank = rank
            best = cid

    return best


def interpolate_between_genomes(
        gid_a: str,
        gid_b: str,
        *,
        chronological_pool: Sequence[Tuple[pd.Timestamp, str]],
        seq_lookup: Mapping[str, str],
        gid_to_date: Mapping[str, pd.Timestamp],
        max_h_edge: Optional[int],
        max_pat_edge: Optional[float],
        pat_fn: Optional[Callable[[str, str], Optional[float]]],
        depth_guard: int = 48,
        _depth: int = 0,
) -> List[str]:
    split, ham_ab = _needs_split(
        gid_a, gid_b, seq_lookup,
        max_h_edge=max_h_edge,
        max_pat_edge=max_pat_edge,
        pat_fn=pat_fn,
    )
    if not split:
        return []
    if _depth >= depth_guard:
        logger.debug("interpolation depth_guard hit (%s,%s)", gid_a, gid_b)
        return []

    dt_a = gid_to_date.get(gid_a, pd.Timestamp.min)
    dt_b = gid_to_date.get(gid_b, pd.Timestamp.min)
    dt_lo, dt_hi = (dt_b, dt_a) if dt_b < dt_a else (dt_a, dt_b)

    mid = _best_midpoint(
        gid_a, gid_b,
        chronological_pool=chronological_pool,
        seq_lookup=seq_lookup,
        gid_to_date=gid_to_date,
        max_h_edge=max_h_edge,
        max_pat_edge=max_pat_edge,
        pat_fn=pat_fn,
        ham_ab_total=max(ham_ab, 1),
        dt_lo=dt_lo,
        dt_hi=dt_hi,
    )
    if mid is None:
        return []

    left = interpolate_between_genomes(
        gid_a, mid,
        chronological_pool=chronological_pool,
        seq_lookup=seq_lookup,
        gid_to_date=gid_to_date,
        max_h_edge=max_h_edge,
        max_pat_edge=max_pat_edge,
        pat_fn=pat_fn,
        depth_guard=depth_guard,
        _depth=_depth + 1,
    )
    right = interpolate_between_genomes(
        mid, gid_b,
        chronological_pool=chronological_pool,
        seq_lookup=seq_lookup,
        gid_to_date=gid_to_date,
        max_h_edge=max_h_edge,
        max_pat_edge=max_pat_edge,
        pat_fn=pat_fn,
        depth_guard=depth_guard,
        _depth=_depth + 1,
    )

    interior: List[str] = [*left, mid, *right]

    collapsed: List[str] = []
    for x in interior:
        if not collapsed or collapsed[-1] != x:
            collapsed.append(x)

    return collapsed


def anchored_genome_path_for_focal(
        focal_gid: str,
        focal_lineage_raw: str,
        *,
        pango,
        alias_key: Optional[Mapping[str, str]],
        lineage_buckets: Mapping[str, Sequence[Tuple[pd.Timestamp, str]]],
        seq_lookup: Mapping[str, str],
        gid_to_date: Mapping[str, pd.Timestamp],
        chronological_pool_sorted_arg: Sequence[Tuple[pd.Timestamp, str]],
        max_h_edge: Optional[int],
        max_pat_edge: Optional[float],
        pat_fn: Optional[Callable[[str, str], Optional[float]]],
        recomb_pick: str,
        interpolate_depth: int = 48,
) -> List[str]:
    """Canonical root→``focal_gid`` genome path (milestones + interpolation)."""

    milestones = ancestry_milestones_root_to_tip(
        focal_lineage_raw,
        pango=pango,
        alias_key=alias_key,
        fallback_dotted_chain=True,
        recomb_pick=recomb_pick,
    )

    anchors: List[str] = []
    focal_norm = normalize_lineage_string(focal_lineage_raw, pango=pango, alias_key=alias_key)

    for milestone in milestones[:-1]:
        rep = canonical_rep_gid(
            lineage_buckets, milestone, pango=pango, alias_key=alias_key)
        if rep:
            anchors.append(rep)

    last_lin = milestones[-1] if milestones else ""
    if last_lin:
        last_key = normalize_lineage_string(last_lin, pango=pango, alias_key=alias_key)
        if focal_norm and last_key == focal_norm:
            anchors.append(str(focal_gid))
        else:
            rep = canonical_rep_gid(lineage_buckets, last_lin, pango=pango, alias_key=alias_key)
            if rep:
                anchors.append(rep)
            anchors.append(str(focal_gid))
    elif not anchors:
        return [str(focal_gid)]

    uniq: List[str] = []
    for g in anchors:
        if not uniq or uniq[-1] != g:
            uniq.append(g)

    expanded: List[str] = []
    for i_seg in range(len(uniq) - 1):
        a, b = uniq[i_seg], uniq[i_seg + 1]

        mids = interpolate_between_genomes(
            a, b,
            chronological_pool=chronological_pool_sorted_arg,
            seq_lookup=seq_lookup,
            gid_to_date=gid_to_date,
            max_h_edge=max_h_edge,
            max_pat_edge=max_pat_edge,
            pat_fn=pat_fn,
            depth_guard=interpolate_depth,
        )

        if not expanded:
            expanded.append(a)
        elif expanded[-1] != a:
            expanded.append(a)
        expanded.extend([m for m in mids if not expanded or m != expanded[-1]])
        if not expanded or expanded[-1] != b:
            expanded.append(b)

    if not expanded:
        expanded = list(uniq)

    out: List[str] = []
    for g in expanded:
        sg = seq_lookup.get(g)
        if out and sg and seq_lookup.get(out[-1]) == sg:
            continue
        if out and g == out[-1]:
            continue
        out.append(str(g))

    if not out or out[-1] != focal_gid:
        sg_f = seq_lookup.get(str(focal_gid))
        if out and sg_f and seq_lookup.get(out[-1]) == sg_f:
            out[-1] = str(focal_gid)
        else:
            out.append(str(focal_gid))

    return out


def accumulate_predecessor_edges(paths: Mapping[str, List[str]]) -> Dict[str, str]:
    """Project root→leaf paths onto ``{{child : parent}}``. First writer wins."""
    pred: Dict[str, str] = {}
    n_conflict = 0
    for focal, path in paths.items():
        if len(path) < 2:
            continue
        for i in range(1, len(path)):
            child = path[i]
            parent = path[i - 1]
            prev_p = pred.get(child)
            if prev_p == parent:
                continue
            if prev_p is not None:
                n_conflict += 1
                if n_conflict <= 5:
                    logger.warning(
                        "Predecessor conflict for %s: %s vs %s "
                        "(focal keyed by %s; keeping earliest assignment)",
                        child, prev_p, parent, focal,
                    )
                continue
            pred[child] = parent

    if n_conflict:
        logger.warning("Total predecessor conflicts suppressed: %d", n_conflict)
    return pred
