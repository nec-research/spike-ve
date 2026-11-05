#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     ancestor_predecessor.py
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
"""Nested (stem-child) predecessor candidate search on a phylogenetic tree.

``by_nearest_ancestor`` inverts ``by_nearestleaf``: at each ancestor on the
focal→root path it searches **only** the child branch that continues toward the
root (the nested / ancestor clade), not sibling subtrees (parallel cousins).

Performance notes
-----------------
* The union of nested scans up the stem is indexed as date-sorted subtree
  clades at each ``prev_child`` on the focal→root path (depth-many lookups
  per focal, not a full-tree rescan).
* Per-clade arrays are sorted by collection date for early-date cutoffs.
* Leaf→root node-id paths are precomputed for O(depth) LCA patristic distance.
* Parallel search mirrors ``stem_predecessor.run_stem_predecessor_search``.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import heapq
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from recomb2027.latent.data_series.ancestor_reversion import (
    ReversionIndex,
    ReversionPenaltyConfig,
)
from recomb2027.latent.data_series.ancestor_sampling import (
    AncestorSamplingConfig,
    sample_predecessor_weighted,
)
from recomb2027.latent.data_series.lineage_utils import (
    nearest_leaf_lineage_predecessor_ok,
)
from recomb2027.latent.data_series.stem_predecessor import (
    StemPickleTree,
    _chunk_list,
    build_dist_from_root,
    build_stem_pickle_tree,
    gid_from_tree_leaf,
    iter_subtree_leaf_gids,
    resolve_n_jobs,
)

logger = logging.getLogger(__name__)

AttachmentNode = Any
CandidateLeaf = Any

_ANCESTOR_WORKER_CTX: Optional["AncestorSearchContext"] = None


@dataclass
class AncestorSearchContext:
    """Picklable read-only context for parallel nested-clade search."""

    gid_to_d_root: Dict[str, float]
    gid_to_date_ns: Dict[str, int]
    gid_to_path_ids: Dict[str, Tuple[int, ...]]
    gid_to_path_set: Dict[str, frozenset]
    node_id_dist: Dict[int, float]
    pickle_tree: StemPickleTree
    seq_lookup: Dict[str, str]
    max_patristic_distance: Optional[float]
    min_patristic_distance: Optional[float]
    predecessor_rank: int
    lineage_filter_on: bool
    lineage_match: str
    lineage_min_shared_depth: int
    gid_to_lineage: Dict[str, str]
    lineage_to_focal_ancestors: Dict[str, frozenset]
    gid_to_lineage_normalized: Dict[str, str]
    alias_key: Optional[Dict[str, str]]
    recombinant_parents: Optional[Dict[str, List[str]]]
    sampling: AncestorSamplingConfig = field(default_factory=AncestorSamplingConfig)
    # Asymmetric branch / non-reversion penalty (``None`` => legacy scoring by patristic distance).
    reversion: Optional[ReversionPenaltyConfig] = None
    reversion_index: Optional[ReversionIndex] = None
    # Memo of the stem lineage filter per (candidate lineage, focal lineage). The filter is a pure function of the
    # two lineage strings; with alias_key it costs ~0.8 ms per call (de-aliasing) versus ~4 us without, and the
    # walk evaluates it for every candidate leaf.  Per-process (each pool worker keeps its own copy).
    lineage_ok_cache: Dict[Tuple[Optional[str], Optional[str]], bool] = field(default_factory=dict)


@dataclass
class NearestAncestorStats:
    no_date: int = 0
    no_candidate: int = 0
    skipped_identical: int = 0
    skipped_lineage: int = 0
    skipped_max_dist: int = 0
    skipped_min_dist: int = 0
    skipped_not_rootward: int = 0
    skipped_not_earlier: int = 0
    # Diagnostics of the chosen parents (reversion-penalty mode only).
    picked: int = 0
    picked_n_rev_sum: float = 0.0
    picked_n_rev_zero: int = 0
    picked_b_parent_sum: float = 0.0
    picked_b_parent_zero: int = 0
    picked_b_child_sum: float = 0.0
    picked_not_nearest_patristic: int = 0
    picked_n_fwd_sum: float = 0.0
    picked_n_fwd_below_min: int = 0
    picked_b_child_below_min: int = 0
    shortlist_no_sequence: int = 0
    # Search effort: stem depths visited and leaves iterated (diagnostics for slow runs).
    walk_depths: int = 0
    walk_leaves: int = 0


@dataclass
class NearestAncestorPick:
    predecessor_gid: Optional[str] = None
    stats: NearestAncestorStats = field(default_factory=NearestAncestorStats)


@dataclass
class AncestorSearchResult:
    links: Dict[str, str] = field(default_factory=dict)
    stats: NearestAncestorStats = field(default_factory=NearestAncestorStats)


def leaf_path_toward_root_set(focal_leaf: Any) -> set:
    """Nodes on the focal leaf → root path (inclusive)."""
    path: set = set()
    node = focal_leaf
    while node is not None:
        path.add(node)
        node = node.parent_node
    return path


def patristic_distance_leaves(
        focal_leaf: Any,
        cand_leaf: Any,
        dist_from_root: Mapping[Any, float],
) -> float:
    """Patristic distance using the true LCA of the two leaves (test / legacy)."""
    focal_path = leaf_path_toward_root_set(focal_leaf)
    node = cand_leaf
    while node not in focal_path:
        node = node.parent_node
    lca = node
    return (
        float(dist_from_root[focal_leaf])
        + float(dist_from_root[cand_leaf])
        - 2.0 * float(dist_from_root[lca])
    )


def patristic_distance_path_ids(
        focal_path_ids: Tuple[int, ...],
        cand_path_ids: Tuple[int, ...],
        focal_d_root: float,
        cand_d_root: float,
        node_id_dist: Mapping[int, float],
        focal_path_set: Optional[frozenset] = None,
) -> float:
    """Patristic distance from precomputed leaf→root node-id paths."""
    path_set = focal_path_set if focal_path_set is not None else frozenset(focal_path_ids)
    for nid in cand_path_ids:
        if nid in path_set:
            return focal_d_root + cand_d_root - 2.0 * float(node_id_dist[nid])
    return focal_d_root + cand_d_root


def iter_nested_stem_candidate_leaves(
        focal_leaf: Any,
        node_leaves: Mapping[Any, List[Any]],
) -> Iterable[Tuple[AttachmentNode, CandidateLeaf]]:
    """Yield ``(attachment_node, cand_leaf)`` on stem-child branches toward root."""
    prev_child = focal_leaf
    node = focal_leaf.parent_node
    while node is not None:
        for child in node.child_node_iter():
            if child is not prev_child:
                continue
            for cand_leaf in node_leaves.get(child, ()):
                if cand_leaf is focal_leaf:
                    continue
                yield node, cand_leaf
        prev_child = node
        node = node.parent_node


def precompute_ancestor_search_index(
        phylotree,
        node_to_id: Mapping[Any, int],
        dist_from_root: Mapping[Any, float],
        gid_to_date_ns: Mapping[str, int],
        *,
        log_every: int = 10000,
) -> Tuple[
    Dict[str, float],
    Dict[str, Tuple[int, ...]],
    Dict[str, frozenset],
    Dict[int, float],
]:
    """One-pass index: leaf paths and node distances (subtrees walked on demand)."""
    n_leaves = len(phylotree.leaf_nodes())
    logger.info(
        "by_nearest_ancestor: building search index for %d leaves (log_every=%d) ...",
        n_leaves, log_every,
    )
    t0 = time.time()

    node_id_dist: Dict[int, float] = {
        node_to_id[node]: float(dist_from_root[node])
        for node in dist_from_root
        if node in node_to_id
    }

    gid_to_d_root: Dict[str, float] = {}
    gid_to_path_ids: Dict[str, Tuple[int, ...]] = {}
    gid_to_path_set: Dict[str, frozenset] = {}

    for leaf_idx, leaf in enumerate(phylotree.leaf_node_iter(), start=1):
        gid = gid_from_tree_leaf(leaf)
        gid_to_d_root[gid] = float(dist_from_root[leaf])
        path_ids: List[int] = []
        node = leaf
        while node is not None:
            path_ids.append(node_to_id[node])
            node = node.parent_node
        gid_to_path_ids[gid] = tuple(path_ids)
        gid_to_path_set[gid] = frozenset(path_ids)

        if log_every > 0 and leaf_idx % log_every == 0:
            elapsed = time.time() - t0
            rate = leaf_idx / elapsed if elapsed > 0 else 0.0
            logger.info(
                "by_nearest_ancestor: path index %d / %d (%.1fs, %.0f leaves/s)",
                leaf_idx, n_leaves, elapsed, rate,
            )

    logger.info(
        "by_nearest_ancestor: index complete in %.1fs — %d leaves "
        "(subtree clades built on demand)",
        time.time() - t0,
        n_leaves,
    )
    return (
        gid_to_d_root,
        gid_to_path_ids,
        gid_to_path_set,
        node_id_dist,
    )


def _is_identical_seq(
        seq_lookup: Mapping[str, str],
        gid_a: Optional[str],
        gid_b: Optional[str],
) -> bool:
    if not seq_lookup or gid_a is None or gid_b is None:
        return False
    sa = seq_lookup.get(gid_a)
    sb = seq_lookup.get(gid_b)
    return sa is not None and sb is not None and sa == sb


def iter_leaves_new_at_depth(tree: Any, path_ids: Sequence[int], depth: int) -> Iterable[str]:
    """Leaves whose LCA with ``path_ids[0]`` is ``path_ids[depth]``: the subtree of ``path_ids[depth]`` MINUS the
    subtree of ``path_ids[depth-1]``.

    Walking the full subtree of ``path_ids[depth]`` at every depth (the previous behaviour) re-visits all leaves of
    the lower depths again and again: the cost per focal is the sum of the subtree sizes along the stem, which
    explodes once the walk climbs toward the root (``min_b_child`` makes it climb further).  The union over
    ``depth = 0 .. d`` is exactly the full subtree of ``path_ids[d]``, each leaf yielded once.  Falls back to the full
    subtree if the path is not a parent chain in ``tree.children`` (correct, only slower).
    """
    if depth <= 0:
        yield from iter_subtree_leaf_gids(tree, path_ids[0])
        return
    node = path_ids[depth]
    prev = path_ids[depth - 1]
    kids = tree.children.get(node, ())
    if prev not in kids:
        yield from iter_subtree_leaf_gids(tree, node)
        return
    for child in kids:
        if child != prev:
            yield from iter_subtree_leaf_gids(tree, child)


def _stem_lineage_ok(ctx: AncestorSearchContext, cand_gid: str, cur_gid: str) -> bool:
    if not ctx.lineage_filter_on:
        return True
    cand_lin = ctx.gid_to_lineage.get(cand_gid)
    cur_lin = ctx.gid_to_lineage.get(cur_gid)
    key = (cand_lin, cur_lin)
    cached = ctx.lineage_ok_cache.get(key)
    if cached is not None:
        return cached
    focal_compressed = frozenset()
    if ctx.lineage_match == "pangonet" and cur_lin is not None:
        focal_compressed = ctx.lineage_to_focal_ancestors.get(cur_lin, frozenset())
    ok = nearest_leaf_lineage_predecessor_ok(
        cand_lin,
        cur_lin,
        lineage_match=ctx.lineage_match,
        lineage_min_shared_depth=ctx.lineage_min_shared_depth,
        recombinant_parents=ctx.recombinant_parents,
        alias_key=ctx.alias_key,
        focal_ancestors_compressed=focal_compressed,
        pangonet_obj=None,
    )
    ctx.lineage_ok_cache[key] = bool(ok)
    return bool(ok)


def pick_nearest_ancestor_for_gid(
        cur_gid: str,
        ctx: AncestorSearchContext,
) -> NearestAncestorPick:
    """Select predecessor for one focal genome id (stream subtrees like stem)."""
    if ctx.reversion is not None and ctx.reversion_index is not None:
        return pick_nearest_ancestor_penalized(cur_gid, ctx)
    out = NearestAncestorPick()
    cur_date_ns = ctx.gid_to_date_ns.get(cur_gid)
    if cur_date_ns is None:
        out.stats.no_date = 1
        return out

    cur_d_root = ctx.gid_to_d_root.get(cur_gid)
    if cur_d_root is None:
        out.stats.no_date = 1
        return out

    focal_path_ids = ctx.gid_to_path_ids.get(cur_gid)
    if focal_path_ids is None or len(focal_path_ids) < 2:
        out.stats.no_candidate = 1
        return out

    focal_path_set = ctx.gid_to_path_set[cur_gid]
    best_by_gid: Dict[str, Tuple[float, int]] = {}
    best_dist = np.inf
    seen_gids: set = set()

    for depth in range(len(focal_path_ids) - 1):
        ancestor_id = focal_path_ids[depth + 1]
        lb = float(cur_d_root) - float(ctx.node_id_dist.get(ancestor_id, 0.0))
        if lb >= best_dist:
            break
        if ctx.max_patristic_distance is not None and lb > ctx.max_patristic_distance:
            break

        out.stats.walk_depths += 1
        for cand_gid in iter_leaves_new_at_depth(ctx.pickle_tree, focal_path_ids, depth):
            out.stats.walk_leaves += 1
            if cand_gid == cur_gid or cand_gid in seen_gids:
                continue
            seen_gids.add(cand_gid)

            cand_ns = ctx.gid_to_date_ns.get(cand_gid)
            if cand_ns is None:
                continue
            if cand_ns >= cur_date_ns:
                out.stats.skipped_not_earlier += 1
                continue

            cand_d_root = ctx.gid_to_d_root.get(cand_gid)
            if cand_d_root is None or cand_d_root >= cur_d_root:
                out.stats.skipped_not_rootward += 1
                continue

            if _is_identical_seq(ctx.seq_lookup, cand_gid, cur_gid):
                out.stats.skipped_identical += 1
                continue

            cand_path_ids = ctx.gid_to_path_ids.get(cand_gid)
            if cand_path_ids is None:
                continue

            pdist = patristic_distance_path_ids(
                focal_path_ids,
                cand_path_ids,
                float(cur_d_root),
                float(cand_d_root),
                ctx.node_id_dist,
                focal_path_set=focal_path_set,
            )

            if ctx.max_patristic_distance is not None and pdist > ctx.max_patristic_distance:
                out.stats.skipped_max_dist += 1
                continue
            if ctx.min_patristic_distance is not None and pdist < ctx.min_patristic_distance:
                out.stats.skipped_min_dist += 1
                continue

            if not _stem_lineage_ok(ctx, cand_gid, cur_gid):
                out.stats.skipped_lineage += 1
                continue

            prev = best_by_gid.get(cand_gid)
            if prev is None or pdist < prev[0]:
                best_by_gid[cand_gid] = (float(pdist), int(cand_ns))
                if pdist < best_dist:
                    best_dist = float(pdist)

    if not best_by_gid:
        out.stats.no_candidate = 1
        return out

    eligible = [
        (gid, pdist, date_ns)
        for gid, (pdist, date_ns) in best_by_gid.items()
    ]
    out.predecessor_gid = sample_predecessor_weighted(
        eligible,
        config=ctx.sampling,
        cur_gid=cur_gid,
        cur_date_ns=int(cur_date_ns),
        predecessor_rank=ctx.predecessor_rank,
    )
    return out


def pick_nearest_ancestor_penalized(
        cur_gid: str,
        ctx: AncestorSearchContext,
) -> NearestAncestorPick:
    """Predecessor choice with the asymmetric branch / non-reversion score.

    ``d_eff = T(b_F) + alpha * b_P + beta * n_rev + gamma_fwd * max(0, m_min - n_fwd)`` with
    ``T(b) = b + gamma_tree * max(0, b_min - b)`` (see :mod:`ancestor_reversion`; ``b_min = m_min = 0``
    reduces to ``b_F + alpha * b_P + beta * n_rev``).  Candidates are the same eligible leaves as in the
    legacy search (earlier, rootward, non-identical, lineage-compatible).

    Candidates found under ``focal_path[depth]`` (and not under a shallower path node) have
    ``LCA = focal_path[depth]``, hence ``b_F = d(F) - d(focal_path[depth])`` is known per subtree.  All
    candidates at that depth or higher have tree key ``>= cfg.tree_lower_bound(b_F)``; the walk up the stem
    stops once that bound cannot improve the ``shortlist``-th best key.
    """
    cfg = ctx.reversion
    index = ctx.reversion_index
    assert cfg is not None and index is not None
    out = NearestAncestorPick()
    cur_date_ns = ctx.gid_to_date_ns.get(cur_gid)
    if cur_date_ns is None:
        out.stats.no_date = 1
        return out
    cur_d_root = ctx.gid_to_d_root.get(cur_gid)
    if cur_d_root is None:
        out.stats.no_date = 1
        return out
    focal_path_ids = ctx.gid_to_path_ids.get(cur_gid)
    if focal_path_ids is None or len(focal_path_ids) < 2:
        out.stats.no_candidate = 1
        return out

    K = int(cfg.shortlist)
    alpha = float(cfg.alpha)
    heap: List[float] = []          # negated keys; heap[0] = -(K-th smallest key)
    seen_gids: set = set()
    # gid -> (key, b_F, b_P, patristic, date_ns)
    cands: Dict[str, Tuple[float, float, float, float, int]] = {}

    # Same candidate universe as the legacy search: path nodes 1 .. len-2 (the root's subtree is not walked).
    for depth in range(1, len(focal_path_ids) - 1):
        lca_id = focal_path_ids[depth]
        d_lca = float(ctx.node_id_dist.get(lca_id, 0.0))
        b_f = float(cur_d_root) - d_lca
        kth = -heap[0] if len(heap) >= K else np.inf
        # Every candidate at this depth or higher has key >= tree_lower_bound(b_f) (b_P >= 0, b_F grows
        # along the stem).  With min_b_child > 0 the lower bound is min(T(b_f), max(b_min, b_f)), so the
        # walk does not stop while the shortlist is still filled by too-close ancestors.
        if cfg.tree_lower_bound(b_f) >= kth:
            break
        if ctx.max_patristic_distance is not None and b_f > ctx.max_patristic_distance:
            break
        out.stats.walk_depths += 1
        for cand_gid in iter_leaves_new_at_depth(ctx.pickle_tree, focal_path_ids, depth):
            out.stats.walk_leaves += 1
            if cand_gid == cur_gid or cand_gid in seen_gids:
                continue
            seen_gids.add(cand_gid)
            cand_ns = ctx.gid_to_date_ns.get(cand_gid)
            if cand_ns is None:
                continue
            if cand_ns >= cur_date_ns:
                out.stats.skipped_not_earlier += 1
                continue
            cand_d_root = ctx.gid_to_d_root.get(cand_gid)
            if cand_d_root is None or cand_d_root >= cur_d_root:
                out.stats.skipped_not_rootward += 1
                continue
            if _is_identical_seq(ctx.seq_lookup, cand_gid, cur_gid):
                out.stats.skipped_identical += 1
                continue
            b_p = max(0.0, float(cand_d_root) - d_lca)
            pdist = b_f + b_p
            if ctx.max_patristic_distance is not None and pdist > ctx.max_patristic_distance:
                out.stats.skipped_max_dist += 1
                continue
            if ctx.min_patristic_distance is not None and pdist < ctx.min_patristic_distance:
                out.stats.skipped_min_dist += 1
                continue
            if not _stem_lineage_ok(ctx, cand_gid, cur_gid):
                out.stats.skipped_lineage += 1
                continue
            key = cfg.tree_term(b_f) + alpha * b_p
            cands[cand_gid] = (key, b_f, b_p, pdist, int(cand_ns))
            if len(heap) < K:
                heapq.heappush(heap, -key)
            elif key < -heap[0]:
                heapq.heapreplace(heap, -key)

    if not cands:
        out.stats.no_candidate = 1
        return out

    short = sorted(cands.items(), key=lambda kv: (kv[1][0], kv[0]))[:K]
    short_gids = [g for g, _ in short]
    n_rev, n_fwd, n_missing = index.score(cur_gid, short_gids)
    out.stats.shortlist_no_sequence += n_missing

    # rows: (d_eff, n_rev, b_P, date_ns, gid, b_F, patristic, n_fwd)
    rows: List[Tuple[float, int, float, int, str, float, float, int]] = []
    for (gid, (_key, b_f, b_p, pdist, date_ns)), nr, nf in zip(short, n_rev, n_fwd):
        d_eff = (cfg.tree_term(b_f) + alpha * b_p + float(cfg.beta) * float(nr)
                 + cfg.fwd_penalty(int(nf)))
        rows.append((d_eff, int(nr), b_p, date_ns, gid, b_f, pdist, int(nf)))

    if cfg.max_n_rev is not None:
        capped = [r for r in rows if r[1] <= int(cfg.max_n_rev)]
        if capped:                        # never drop the focal: fall back to the full shortlist
            rows = capped

    if ctx.sampling.selection == "nearest":
        # d_eff, then fewest reversions, then smallest parent branch, then MOST RECENT parent, then gid
        ordered = sorted(rows, key=lambda r: (r[0], r[1], r[2], -r[3], r[4]))
        idx = min(max(1, int(ctx.predecessor_rank)) - 1, len(ordered) - 1)
        chosen = ordered[idx]
    else:
        gid_pick = sample_predecessor_weighted(
            [(r[4], r[0], r[3]) for r in rows],
            config=ctx.sampling,
            cur_gid=cur_gid,
            cur_date_ns=int(cur_date_ns),
            predecessor_rank=ctx.predecessor_rank,
        )
        chosen = next(r for r in rows if r[4] == gid_pick)

    st = out.stats
    st.picked = 1
    st.picked_n_rev_sum = float(chosen[1])
    st.picked_n_rev_zero = int(chosen[1] == 0)
    st.picked_b_parent_sum = float(chosen[2])
    st.picked_b_parent_zero = int(chosen[2] == 0.0)
    st.picked_b_child_sum = float(chosen[5])
    st.picked_n_fwd_sum = float(chosen[7])
    st.picked_n_fwd_below_min = int(cfg.min_n_fwd > 0 and chosen[7] < cfg.min_n_fwd)
    st.picked_b_child_below_min = int(cfg.min_b_child > 0 and chosen[5] < cfg.min_b_child)
    nearest_pat = min(r[6] for r in rows)
    st.picked_not_nearest_patristic = int(chosen[6] > nearest_pat + 1e-12)
    out.predecessor_gid = chosen[4]
    return out


def pick_nearest_ancestor_for_focal(
        focal_leaf: Any,
        *,
        cur_gid: str,
        cur_date,
        cur_d_root: float,
        cur_lin: Optional[str],
        node_leaves: Mapping[Any, List[Any]],
        dist_from_root: Mapping[Any, float],
        gid_to_date: Mapping[str, Any],
        gid_to_lineage: Optional[Mapping[str, str]],
        is_identical_seq,
        lineage_filter_on: bool,
        lineage_match: str,
        lineage_min_shared_depth: int,
        recombinant_parents: Optional[Dict[str, List[str]]],
        alias_key: Optional[Dict[str, str]],
        focal_compressed_ancestors: frozenset,
        pangonet_obj,
        max_patristic_distance: Optional[float],
        min_patristic_distance: Optional[float],
        predecessor_rank: int,
) -> NearestAncestorPick:
    """Legacy leaf-object API used by unit tests on small mock trees."""
    out = NearestAncestorPick()
    if cur_date is None or (hasattr(cur_date, "__bool__") and np.isnan(cur_date)):
        out.stats.no_date = 1
        return out

    rank = max(1, int(predecessor_rank))
    best_by_gid: Dict[str, float] = {}

    for _attach_node, cand_leaf in iter_nested_stem_candidate_leaves(
            focal_leaf, node_leaves,
    ):
        cand_gid = cand_leaf.taxon.label.split("-")[-1]
        if cand_gid == cur_gid:
            continue

        cand_d_root = dist_from_root.get(cand_leaf)
        if cand_d_root is None or cand_d_root >= cur_d_root:
            out.stats.skipped_not_rootward += 1
            continue

        cand_date = gid_to_date.get(cand_gid)
        if cand_date is None or (hasattr(cand_date, "__bool__") and np.isnan(cand_date)):
            continue
        if cand_date >= cur_date:
            out.stats.skipped_not_earlier += 1
            continue

        if is_identical_seq(cand_gid, cur_gid):
            out.stats.skipped_identical += 1
            continue

        pdist = patristic_distance_leaves(focal_leaf, cand_leaf, dist_from_root)

        if max_patristic_distance is not None and pdist > max_patristic_distance:
            out.stats.skipped_max_dist += 1
            continue
        if min_patristic_distance is not None and pdist < min_patristic_distance:
            out.stats.skipped_min_dist += 1
            continue

        if lineage_filter_on:
            cand_lin = gid_to_lineage.get(cand_gid) if gid_to_lineage else None
            if not nearest_leaf_lineage_predecessor_ok(
                    cand_lin,
                    cur_lin,
                    lineage_match=lineage_match,
                    lineage_min_shared_depth=lineage_min_shared_depth,
                    recombinant_parents=recombinant_parents,
                    alias_key=alias_key,
                    focal_ancestors_compressed=focal_compressed_ancestors,
                    pangonet_obj=pangonet_obj,
            ):
                out.stats.skipped_lineage += 1
                continue

        prev = best_by_gid.get(cand_gid)
        if prev is None or pdist < prev:
            best_by_gid[cand_gid] = float(pdist)

    if not best_by_gid:
        out.stats.no_candidate = 1
        return out

    eligible = sorted(((d, g) for g, d in best_by_gid.items()), key=lambda t: (t[0], t[1]))
    idx = min(rank - 1, len(eligible) - 1)
    out.predecessor_gid = eligible[idx][1]
    return out


def _merge_ancestor_batch_result(merged: AncestorSearchResult, part: AncestorSearchResult) -> None:
    merged.links.update(part.links)
    for field_name in (
        "no_date", "no_candidate", "skipped_identical", "skipped_lineage",
        "skipped_max_dist", "skipped_min_dist", "skipped_not_rootward",
        "skipped_not_earlier", "picked", "picked_n_rev_sum", "picked_n_rev_zero",
        "picked_b_parent_sum", "picked_b_parent_zero", "picked_b_child_sum",
        "picked_not_nearest_patristic", "shortlist_no_sequence",
        "picked_n_fwd_sum", "picked_n_fwd_below_min", "picked_b_child_below_min",
        "walk_depths", "walk_leaves",
    ):
        setattr(merged.stats, field_name,
                getattr(merged.stats, field_name) + getattr(part.stats, field_name))


def _serial_ancestor_batch(
        focal_gids: List[str],
        ctx: AncestorSearchContext,
        *,
        log_every: int = 1000,
) -> AncestorSearchResult:
    out = AncestorSearchResult()
    n_total = len(focal_gids)
    t0 = time.time()
    for i, cur_gid in enumerate(focal_gids, start=1):
        pick = pick_nearest_ancestor_for_gid(cur_gid, ctx)
        if pick.predecessor_gid is not None:
            out.links[cur_gid] = pick.predecessor_gid
        _merge_ancestor_batch_result(out, AncestorSearchResult(stats=pick.stats))
        if log_every > 0 and i % log_every == 0:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0.0
            st = out.stats
            logger.info(
                "by_nearest_ancestor: serial search %d / %d (%.1fs, %.0f focals/s, %d links) | "
                "stem depths/focal=%.1f, leaves scanned/focal=%.0f | lineage memo=%d pairs",
                i, n_total, elapsed, rate, len(out.links),
                st.walk_depths / i, st.walk_leaves / i, len(ctx.lineage_ok_cache),
            )
    return out


def _process_ancestor_focal_batch(focal_gids: List[str]) -> AncestorSearchResult:
    if _ANCESTOR_WORKER_CTX is None:
        raise RuntimeError("ancestor worker context not initialized")
    out = AncestorSearchResult()
    n = len(focal_gids)
    every = max(1, n // 10)
    t0 = time.time()
    pid = os.getpid()
    for i, cur_gid in enumerate(focal_gids, start=1):
        pick = pick_nearest_ancestor_for_gid(cur_gid, _ANCESTOR_WORKER_CTX)
        if pick.predecessor_gid is not None:
            out.links[cur_gid] = pick.predecessor_gid
        _merge_ancestor_batch_result(out, AncestorSearchResult(stats=pick.stats))
        if i % every == 0 or i == n:
            elapsed = time.time() - t0
            st = out.stats
            logger.info(
                "by_nearest_ancestor worker pid=%d: %d/%d focals in %.0fs (%.2f focals/s) | links=%d | "
                "stem depths/focal=%.1f, leaves scanned/focal=%.0f | lineage memo=%d pairs",
                pid, i, n, elapsed, i / max(elapsed, 1e-9), len(out.links),
                st.walk_depths / max(i, 1), st.walk_leaves / max(i, 1),
                len(_ANCESTOR_WORKER_CTX.lineage_ok_cache),
            )
    return out


def _init_ancestor_worker(ctx: AncestorSearchContext) -> None:
    global _ANCESTOR_WORKER_CTX
    _ANCESTOR_WORKER_CTX = ctx


def run_ancestor_predecessor_search(
        focal_gids: Sequence[str],
        ctx: AncestorSearchContext,
        *,
        n_jobs: int = 1,
        log_every: int = 1000,
) -> AncestorSearchResult:
    """Run nested-clade search serially or via :class:`ProcessPoolExecutor`."""
    n_workers = resolve_n_jobs(n_jobs)
    focal_list = list(focal_gids)
    if not focal_list:
        return AncestorSearchResult()

    if n_workers == 1:
        logger.info("by_nearest_ancestor: serial search over %d leaves", len(focal_list))
        return _serial_ancestor_batch(focal_list, ctx, log_every=log_every)

    chunks = _chunk_list(focal_list, n_workers)
    n_pool_workers = min(n_workers, len(chunks))
    logger.info(
        "by_nearest_ancestor: starting process pool (%d workers, %d chunks) ...",
        n_pool_workers, len(chunks),
    )
    merged = AncestorSearchResult()
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(
            max_workers=n_pool_workers,
            initializer=_init_ancestor_worker,
            initargs=(ctx,),
    ) as pool:
        futures = {
            pool.submit(_process_ancestor_focal_batch, chunk): (ci, len(chunk))
            for ci, chunk in enumerate(chunks)
        }
        # completion order (not submission order): a slow early chunk no longer hides progress of the others
        for fut in as_completed(futures):
            ci, n_chunk = futures[fut]
            part = fut.result()
            _merge_ancestor_batch_result(merged, part)
            done += 1
            logger.info(
                "by_nearest_ancestor: chunk %d (%d focals) done -> %d / %d chunks | +%d links, total %d | "
                "%.1fs elapsed",
                ci, n_chunk, done, len(chunks), len(part.links), len(merged.links), time.time() - t0,
            )
    n_all = max(1, len(focal_list))
    logger.info(
        "by_nearest_ancestor: parallel search finished in %.1fs | links=%d | stem depths/focal=%.1f, "
        "leaves scanned/focal=%.0f",
        time.time() - t0, len(merged.links),
        merged.stats.walk_depths / n_all, merged.stats.walk_leaves / n_all,
    )
    return merged
