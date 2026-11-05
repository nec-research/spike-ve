#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     lineage_utils.py
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
"""Pango lineage helpers for predecessor compatibility checks.

This module is intentionally dependency-free: it implements just enough of the
Pango aliasing logic to support hierarchical sublineage matching plus an
explicit recombinant-parent whitelist. It does **not** require the third-party
``pango_aliasor`` package, but it is fully compatible with the official
``alias_key.json`` shipped by `pango-designation`.

Key concepts
------------
- **De-aliasing**: Pango uses short aliases (``BA``, ``BQ``, ``XBB``) as
  prefixes for very long lineage names. Two aliases form an
  ancestor/descendant relationship iff one *de-aliased* dotted name is a
  prefix of the other.
- **Recombinant lineages** (``X*``): map to multiple parents and break the
  prefix relation. We treat them via an explicit whitelist
  ``recombinant_parents: {child_lineage: [parent_lineage, ...]}``.

Public API
----------
- ``load_alias_key(path)``: read ``alias_key.json`` -> simple alias dict
  + auto-extracted recombinant parents map.
- ``dealias(name, alias_key)``: recursively expand the leading alias.
- ``shared_prefix_depth(a, b, alias_key)``: number of dotted segments shared
  by ``dealias(a)`` and ``dealias(b)``.
- ``is_pango_compatible(parent_lin, child_lin, *, min_shared_depth,
  recombinant_parents, alias_key)``: True iff ``parent_lin`` is an allowed
  predecessor lineage for ``child_lin``.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Tuple

# Sentinel for missing / unknown lineage values commonly seen in metadata.
_NA_VALUES = {"", "n/a", "na", "nan", "none", "unassigned", "unknown", "?"}


def _is_na(name: Optional[str]) -> bool:
    if name is None:
        return True
    return str(name).strip().lower() in _NA_VALUES


# ---------------------------------------------------------------------------
# Alias key loading & de-aliasing
# ---------------------------------------------------------------------------

def load_alias_key(path: str) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """Load Pango ``alias_key.json`` into two dicts.

    Returns
    -------
    simple_aliases : Dict[str, str]
        ``{alias: full_dotted_form}`` for non-recombinant aliases (e.g.
        ``"BA": "B.1.1.529"``). The empty-string aliases used by Pango as
        placeholders for top-level letters (e.g. ``"A": ""``) are skipped.
    recombinant_parents : Dict[str, List[str]]
        ``{recombinant_alias: [parent_lineage, ...]}`` auto-extracted from
        list-valued entries (e.g. ``"XBB": ["BJ.1", "BM.1.1.1"]``). These can
        be used directly as the ``recombinant_parents`` argument of
        :func:`is_pango_compatible`.
    """
    with open(path, "r") as f:
        raw = json.load(f)

    simple: Dict[str, str] = {}
    recomb: Dict[str, List[str]] = {}
    for alias, val in raw.items():
        if isinstance(val, list):
            if val:
                recomb[alias] = [str(v) for v in val]
            continue
        if not isinstance(val, str) or not val:
            continue
        simple[alias] = val
    return simple, recomb


def dealias(name: Optional[str], alias_key: Optional[Dict[str, str]]) -> Optional[str]:
    """Recursively expand the leading alias of a Pango lineage name.

    >>> dealias("BA.2.12.1", {"BA": "B.1.1.529"})
    'B.1.1.529.2.12.1'
    >>> dealias("B.1.1.7", {"BA": "B.1.1.529"})
    'B.1.1.7'
    >>> dealias(None, {})
    None

    Recombinant aliases (e.g. ``"XBB"``) have no canonical single-parent
    prefix and are returned unchanged.
    """
    if _is_na(name):
        return None
    if not alias_key:
        return name
    parts = str(name).split(".")
    head = parts[0]
    if head not in alias_key:
        return name
    full = alias_key[head]
    if not full:
        return name
    expanded_head = dealias(full, alias_key) or full
    if len(parts) == 1:
        return expanded_head
    return expanded_head + "." + ".".join(parts[1:])


def shared_prefix_depth(
        a: Optional[str],
        b: Optional[str],
        alias_key: Optional[Dict[str, str]] = None,
) -> int:
    """Number of leading dotted segments shared by ``dealias(a)`` and ``dealias(b)``.

    >>> shared_prefix_depth("BA.2.12.1", "BA.2.5", {"BA": "B.1.1.529"})
    4   # B, 1, 1, 529, 2 -> wait, four equal: B==B, 1==1, 1==1, 529==529 -> 4, then 2==2 -> 5? Let's recount:
        # dealias("BA.2.12.1") = "B.1.1.529.2.12.1" -> ['B','1','1','529','2','12','1']
        # dealias("BA.2.5")    = "B.1.1.529.2.5"    -> ['B','1','1','529','2','5']
        # shared prefix: B,1,1,529,2 -> 5

    Returns 0 when either name is missing/unknown.
    """
    da = dealias(a, alias_key)
    db = dealias(b, alias_key)
    if da is None or db is None:
        return 0
    pa = da.split(".")
    pb = db.split(".")
    n = 0
    for x, y in zip(pa, pb):
        if x == y:
            n += 1
        else:
            break
    return n


# ---------------------------------------------------------------------------
# Compatibility check used by predecessor search
# ---------------------------------------------------------------------------

def _normalize_recomb_keys(
        recomb: Optional[Dict[str, List[str]]],
        alias_key: Optional[Dict[str, str]],
) -> Optional[Dict[str, List[str]]]:
    """De-alias every key and value in a ``recombinant_parents`` dict so the
    downstream prefix matching is consistent with internal de-aliased names."""
    if not recomb:
        return None
    out: Dict[str, List[str]] = {}
    for k, vs in recomb.items():
        dk = dealias(k, alias_key) or k
        dvs: List[str] = []
        for v in vs:
            dv = dealias(v, alias_key) or v
            dvs.append(dv)
        out[dk] = dvs
    return out


def _matches_recombinant_parent(
        cand_dealiased: str,
        cur_dealiased: str,
        recomb_dealiased: Dict[str, List[str]],
) -> bool:
    """True iff ``cur_dealiased`` (or one of its Pango ancestors) lists
    ``cand_dealiased`` (or any of its Pango ancestors) as a designated
    parent in the de-aliased recombinant table.
    """
    cur_parts = cur_dealiased.split(".")
    cand_parts = cand_dealiased.split(".")
    # Walk every prefix of cur (e.g. XBB.1.5 -> XBB.1.5, XBB.1, XBB, ...)
    for i in range(len(cur_parts), 0, -1):
        cur_prefix = ".".join(cur_parts[:i])
        parents = recomb_dealiased.get(cur_prefix)
        if not parents:
            continue
        for parent_d in parents:
            parent_parts = parent_d.split(".")
            if len(cand_parts) < len(parent_parts):
                continue
            if cand_parts[: len(parent_parts)] == parent_parts:
                return True
    return False


def is_pango_compatible(
        parent_lin: Optional[str],
        child_lin: Optional[str],
        *,
        min_shared_depth: int = 2,
        recombinant_parents: Optional[Dict[str, List[str]]] = None,
        alias_key: Optional[Dict[str, str]] = None,
) -> bool:
    """True iff ``parent_lin`` is an allowed predecessor lineage for ``child_lin``.

    The candidate is accepted when **any** of the following holds:

    1. **Hierarchical match (Pango ancestor / sublineage of ancestor).**
       After de-aliasing, the two names share at least ``min_shared_depth``
       leading dotted segments. With ``min_shared_depth = inf`` this becomes
       a strict ancestor/descendant relationship; with ``min_shared_depth = 1``
       it accepts any two names sharing the same root (very permissive).

    2. **Recombinant whitelist.** ``recombinant_parents`` lists Pango parents
       for recombinant child lineages (the ``X*`` lineages). For example
       ``{"XBB": ["BJ.1", "BM.1.1.1"]}`` lets any candidate inside
       ``BJ.1.*`` or ``BM.1.1.1.*`` be accepted as a parent of ``XBB`` (or any
       ``XBB.*`` sublineage).

    Missing or "n/a"-like lineage values are conservatively treated as
    *non-compatible* so that ambiguous candidates are filtered out.

    Parameters
    ----------
    parent_lin : str or None
        Pango lineage of the candidate predecessor leaf.
    child_lin : str or None
        Pango lineage of the focal leaf ``L``.
    min_shared_depth : int, default 2
        Minimum number of de-aliased dotted segments the two lineages must
        share. Set to a large number (e.g. ``999``) to require a strict
        ancestor/descendant relation. Set to ``0`` to disable the
        hierarchical check entirely.
    recombinant_parents : dict, optional
        Whitelist of designated parents per recombinant lineage. Keys and
        values may use Pango aliases; they are de-aliased internally.
    alias_key : dict, optional
        Pango ``alias_key.json`` simple-aliases dict from
        :func:`load_alias_key`. Without it, names are compared literally and
        sublineage matching may be inaccurate for aliased names.
    """
    if _is_na(parent_lin) or _is_na(child_lin):
        return False

    # 1. Hierarchical / shared-prefix match
    if min_shared_depth > 0:
        depth = shared_prefix_depth(parent_lin, child_lin, alias_key)
        if depth >= min_shared_depth:
            return True

    # 2. Recombinant whitelist
    if recombinant_parents:
        recomb_d = _normalize_recomb_keys(recombinant_parents, alias_key)
        cur_d = dealias(child_lin, alias_key)
        cand_d = dealias(parent_lin, alias_key)
        if cur_d and cand_d and recomb_d and _matches_recombinant_parent(cand_d, cur_d, recomb_d):
            return True

    return False


def is_strict_prefix_ancestor_lineage(
        ancestor_candidate_lin: Optional[str],
        descendant_focal_lin: Optional[str],
        *,
        alias_key: Optional[Dict[str, str]] = None,
        recombinant_parents: Optional[Dict[str, List[str]]] = None,
) -> bool:
    """True iff de-aliased ``ancestor_candidate_lin`` is a dotted prefix of focal.

    Equal-length identical names qualify (cousins assigned the same lineage).
    ``recombinant_parents`` bypasses pure prefix equality when designated parents
    match (same mechanism as :func:`is_pango_compatible`).
    """
    if _is_na(ancestor_candidate_lin) or _is_na(descendant_focal_lin):
        return False

    if recombinant_parents:
        recomb_d = _normalize_recomb_keys(recombinant_parents, alias_key)
        cur_d = dealias(descendant_focal_lin, alias_key)
        cand_d = dealias(ancestor_candidate_lin, alias_key)
        if cur_d and cand_d and recomb_d and _matches_recombinant_parent(cand_d, cur_d, recomb_d):
            return True

    cand_d = dealias(ancestor_candidate_lin, alias_key)
    focal_d = dealias(descendant_focal_lin, alias_key)
    if cand_d is None or focal_d is None:
        return False

    ca = cand_d.split(".")
    cb = focal_d.split(".")
    if not ca:
        return False
    if len(ca) > len(cb):
        return False
    return ca == cb[: len(ca)]


def maybe_load_aliases(path: Optional[str]) -> Tuple[Optional[Dict[str, str]], Optional[Dict[str, List[str]]]]:
    """Convenience wrapper: load ``alias_key.json`` if a path is provided,
    otherwise return ``(None, None)`` and log a warning.
    """
    if not path:
        return None, None
    try:
        simple, recomb = load_alias_key(path)
        logging.info(
            "Loaded Pango alias_key: %d simple aliases, %d recombinant aliases (from %s)",
            len(simple), len(recomb), path,
        )
        return simple, recomb
    except Exception as e:
        logging.warning(
            "Failed to load alias_key from %s: %s. "
            "Falling back to literal-string lineage comparison.",
            path, e,
        )
        return None, None


def nearest_leaf_lineage_predecessor_ok(
        cand_lin: Optional[str],
        focal_lin: Optional[str],
        *,
        lineage_match: str,
        lineage_min_shared_depth: int,
        recombinant_parents: Optional[Dict[str, List[str]]],
        alias_key: Optional[Dict[str, str]],
        focal_ancestors_compressed: frozenset,
        pangonet_obj,
) -> bool:
    """Lineage filter shared by ``by_nearestleaf`` and ``by_nearest_stem``."""
    from recomb2027.latent.data_series.pango_anchor_path import (
        normalize_lineage_string,
    )

    mode = lineage_match.strip().lower()
    if mode == "compat":
        return is_pango_compatible(
            cand_lin, focal_lin,
            min_shared_depth=lineage_min_shared_depth,
            recombinant_parents=recombinant_parents,
            alias_key=alias_key,
        )
    if mode == "ancestor":
        return is_strict_prefix_ancestor_lineage(
            cand_lin,
            focal_lin,
            alias_key=alias_key,
            recombinant_parents=recombinant_parents,
        )
    if mode == "pangonet":
        cand_key = normalize_lineage_string(
            str(cand_lin), pango=pangonet_obj, alias_key=alias_key,
        )
        if cand_key and cand_key in focal_ancestors_compressed:
            return True
        return is_strict_prefix_ancestor_lineage(
            cand_lin,
            focal_lin,
            alias_key=alias_key,
            recombinant_parents=recombinant_parents,
        )
    raise ValueError(f"Unknown nearest-leaf lineage_match mode {lineage_match!r}")
