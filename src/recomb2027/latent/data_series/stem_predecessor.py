#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     stem_predecessor.py
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
"""Stem-local predecessor candidate enumeration on a phylogenetic tree."""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from recomb2027.latent.data_series.lineage_utils import (
    is_strict_prefix_ancestor_lineage,
    nearest_leaf_lineage_predecessor_ok,
)

logger = logging.getLogger(__name__)

_STEM_WORKER_CTX: Optional[Any] = None

# (attachment dist_from_root, side-branch subtree root node id)
StemSideRef = Tuple[float, int]


def gid_from_tree_leaf(leaf) -> str:
    return leaf.taxon.label.split("-")[-1]


def leaf_path_toward_root(leaf) -> List[Any]:
    """``[focal_leaf, parent, ..., root]`` along the tree."""
    path: List[Any] = [leaf]
    node = leaf.parent_node
    while node is not None:
        path.append(node)
        node = node.parent_node
    return path


@dataclass
class StemPickleTree:
    """Picklable tree topology for on-the-fly subtree leaf iteration."""

    children: Dict[int, List[int]]
    leaf_gids: Dict[int, str]
    seed_node_id: int


def build_stem_pickle_tree(phylotree) -> Tuple[Dict[Any, int], StemPickleTree]:
    """Assign stable int ids and extract adjacency + leaf genome ids."""
    node_to_id: Dict[Any, int] = {}
    children: Dict[int, List[int]] = {}
    leaf_gids: Dict[int, str] = {}

    # Two passes: preorder visits parents before children, so child ids are
    # not available when building adjacency in a single pass.
    nodes = list(phylotree.preorder_node_iter())
    for node in nodes:
        nid = id(node)
        node_to_id[node] = nid
        if node.is_leaf():
            leaf_gids[nid] = gid_from_tree_leaf(node)

    for node in nodes:
        if node.is_leaf():
            continue
        nid = node_to_id[node]
        children[nid] = [node_to_id[ch] for ch in node.child_node_iter()]

    seed_id = node_to_id[phylotree.seed_node]
    return node_to_id, StemPickleTree(
        children=children,
        leaf_gids=leaf_gids,
        seed_node_id=seed_id,
    )


def build_dist_from_root(phylotree, *, log_every: int = 0) -> Dict[Any, float]:
    """Root distances for every node (no descendant-leaf lists)."""
    n_leaves = len(phylotree.leaf_nodes())
    logger.info(
        "by_nearest_stem: building root distances for %d leaves ...",
        n_leaves,
    )
    t0 = time.time()
    dist_from_root: Dict[Any, float] = {phylotree.seed_node: 0.0}
    leaf_idx = 0
    for node in phylotree.preorder_node_iter():
        if node is phylotree.seed_node:
            continue
        parent_d = dist_from_root.get(node.parent_node, 0.0)
        edge_len = node.edge_length if node.edge_length is not None else 0.0
        dist_from_root[node] = parent_d + edge_len
        if node.is_leaf():
            leaf_idx += 1
            if log_every > 0 and leaf_idx % log_every == 0:
                logger.info(
                    "by_nearest_stem: root distances %d / %d (%.1fs)",
                    leaf_idx, n_leaves, time.time() - t0,
                )
    logger.info(
        "by_nearest_stem: root distances done in %.1fs",
        time.time() - t0,
    )
    return dist_from_root


def iter_stem_side_branch_refs(
        path_toward_root: List[Any],
        node_to_id: Mapping[Any, int],
        dist_from_root: Mapping[Any, float],
        *,
        seed_node: Any,
) -> Iterable[StemSideRef]:
    """Yield ``(attach_d_root, side_child_node_id)`` for side branches on the stem."""
    for idx in range(1, len(path_toward_root)):
        ancestor = path_toward_root[idx]
        toward_focal = path_toward_root[idx - 1]

        if toward_focal.is_leaf():
            if ancestor.is_leaf():
                continue
            attach_d = float(dist_from_root[ancestor])
            for child in ancestor.child_node_iter():
                if child is toward_focal:
                    continue
                yield attach_d, node_to_id[child]
        else:
            if ancestor is seed_node:
                continue
            inner_toward_focal = path_toward_root[idx - 2]
            attach_d = float(dist_from_root[toward_focal])
            for child in toward_focal.child_node_iter():
                if child is inner_toward_focal:
                    continue
                yield attach_d, node_to_id[child]


def iter_stem_side_candidate_leaves(
        path_toward_root: List[Any],
        node_leaves: Dict[Any, List[Any]],
        *,
        seed_node: Any,
) -> Iterable[Tuple[Any, Any]]:
    """Yield ``(attachment_node, cand_leaf)`` on side branches along focal's stem."""
    for idx in range(1, len(path_toward_root)):
        ancestor = path_toward_root[idx]
        toward_focal = path_toward_root[idx - 1]

        if toward_focal.is_leaf():
            if ancestor.is_leaf():
                continue
            for child in ancestor.child_node_iter():
                if child is toward_focal:
                    continue
                for cand_leaf in node_leaves.get(child, ()):
                    yield ancestor, cand_leaf
        else:
            if ancestor is seed_node:
                continue
            inner_toward_focal = path_toward_root[idx - 2]
            for child in toward_focal.child_node_iter():
                if child is inner_toward_focal:
                    continue
                for cand_leaf in node_leaves.get(child, ()):
                    yield toward_focal, cand_leaf


def iter_subtree_leaf_gids(
        tree: StemPickleTree,
        subtree_root_id: int,
) -> Iterable[str]:
    """Depth-first walk of leaves under ``subtree_root_id``."""
    stack = [subtree_root_id]
    children = tree.children
    leaf_gids = tree.leaf_gids
    while stack:
        nid = stack.pop()
        if nid in leaf_gids:
            yield leaf_gids[nid]
            continue
        for child_id in children.get(nid, ()):
            stack.append(child_id)


def _summarize_side_ref_index(
        focal_side_refs: Mapping[str, List[StemSideRef]],
) -> Dict[str, float]:
    if not focal_side_refs:
        return {"n_focal": 0, "total_refs": 0, "avg_refs": 0.0, "max_refs": 0}
    counts = [len(v) for v in focal_side_refs.values()]
    total = sum(counts)
    return {
        "n_focal": len(focal_side_refs),
        "total_refs": total,
        "avg_refs": total / len(focal_side_refs),
        "max_refs": max(counts),
    }


def precompute_stem_side_ref_index(
        phylotree,
        dist_from_root: Mapping[Any, float],
        node_to_id: Mapping[Any, int],
        *,
        seed_node: Any,
        log_every: int = 10000,
) -> Tuple[Dict[str, float], Dict[str, List[StemSideRef]]]:
    """Build compact per-focal side-branch descriptors (O(leaves * depth) memory)."""
    n_leaves = len(phylotree.leaf_nodes())
    logger.info(
        "by_nearest_stem: precomputing stem side-ref index for %d leaves "
        "(log_every=%d) ...",
        n_leaves, log_every,
    )
    t0 = time.time()
    gid_to_d_root: Dict[str, float] = {}
    focal_side_refs: Dict[str, List[StemSideRef]] = {}

    for leaf_idx, current_leaf in enumerate(phylotree.leaf_node_iter(), start=1):
        cur_gid = gid_from_tree_leaf(current_leaf)
        gid_to_d_root[cur_gid] = float(dist_from_root[current_leaf])
        path = leaf_path_toward_root(current_leaf)
        seen_refs: set[StemSideRef] = set()
        refs: List[StemSideRef] = []
        for ref in iter_stem_side_branch_refs(
                path, node_to_id, dist_from_root, seed_node=seed_node,
        ):
            if ref not in seen_refs:
                seen_refs.add(ref)
                refs.append(ref)
        focal_side_refs[cur_gid] = refs

        if log_every > 0 and leaf_idx % log_every == 0:
            elapsed = time.time() - t0
            rate = leaf_idx / elapsed if elapsed > 0 else 0.0
            logger.info(
                "by_nearest_stem: side-ref index %d / %d (%.1fs, %.0f leaves/s, "
                "last focal had %d side branches)",
                leaf_idx, n_leaves, elapsed, rate, len(refs),
            )

    summary = _summarize_side_ref_index(focal_side_refs)
    logger.info(
        "by_nearest_stem: side-ref index complete in %.1fs — "
        "%d focals, %d total side-branch refs (avg=%.1f, max=%d per focal)",
        time.time() - t0,
        int(summary["n_focal"]),
        int(summary["total_refs"]),
        summary["avg_refs"],
        int(summary["max_refs"]),
    )
    return gid_to_d_root, focal_side_refs


def resolve_n_jobs(n_jobs: Optional[int]) -> int:
    """``1`` = serial; ``-1`` = all CPUs; ``None``/``0`` treated as ``1``."""
    if n_jobs is None or n_jobs == 0:
        return 1
    if n_jobs < 0:
        return max(1, os.cpu_count() or 1)
    return max(1, int(n_jobs))


@dataclass
class StemFocalStats:
    no_date: int = 0
    no_candidate: int = 0
    skipped_identical: int = 0
    skipped_lineage: int = 0
    skipped_max_dist: int = 0


@dataclass
class StemSearchContext:
    """Picklable read-only context for per-focal stem predecessor search."""

    tree: StemPickleTree
    focal_side_refs: Dict[str, List[StemSideRef]]
    gid_to_d_root: Dict[str, float]
    gid_to_date_ns: Dict[str, int]
    seq_lookup: Dict[str, str]
    max_patristic_distance: Optional[float]
    lineage_filter_on: bool
    lineage_match: str
    lineage_min_shared_depth: int
    gid_to_lineage: Dict[str, str]
    lineage_to_focal_ancestors: Dict[str, frozenset]
    alias_key: Optional[Dict[str, str]]
    recombinant_parents: Optional[Dict[str, List[str]]]
    gid_to_lineage_normalized: Dict[str, str] = field(default_factory=dict)


@dataclass
class StemSearchResult:
    links: Dict[str, str] = field(default_factory=dict)
    stats: StemFocalStats = field(default_factory=StemFocalStats)


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


def _stem_lineage_predecessor_ok(
        ctx: StemSearchContext,
        cand_gid: str,
        cur_lin: Optional[str],
        focal_compressed_ancestors: frozenset,
) -> bool:
    """Lineage gate using precomputed pangonet-normalized candidate keys when set."""
    if not ctx.lineage_filter_on:
        return True
    cand_lin = ctx.gid_to_lineage.get(cand_gid)
    if ctx.lineage_match == "pangonet" and ctx.gid_to_lineage_normalized:
        cand_key = ctx.gid_to_lineage_normalized.get(cand_gid, "")
        if cand_key and cand_key in focal_compressed_ancestors:
            return True
        return is_strict_prefix_ancestor_lineage(
            cand_lin,
            cur_lin,
            alias_key=ctx.alias_key,
            recombinant_parents=ctx.recombinant_parents,
        )
    return nearest_leaf_lineage_predecessor_ok(
        cand_lin,
        cur_lin,
        lineage_match=ctx.lineage_match,
        lineage_min_shared_depth=ctx.lineage_min_shared_depth,
        recombinant_parents=ctx.recombinant_parents,
        alias_key=ctx.alias_key,
        focal_ancestors_compressed=focal_compressed_ancestors,
        pangonet_obj=None,
    )


def pick_stem_predecessor_for_focal(
        cur_gid: str,
        ctx: StemSearchContext,
) -> Tuple[Optional[str], StemFocalStats]:
    """Select the best stem predecessor for one focal genome id."""
    stats = StemFocalStats()
    date_ns = ctx.gid_to_date_ns.get(cur_gid)
    if date_ns is None:
        stats.no_date = 1
        return None, stats

    cur_d_root = ctx.gid_to_d_root.get(cur_gid)
    if cur_d_root is None:
        stats.no_date = 1
        return None, stats

    cur_lin = ctx.gid_to_lineage.get(cur_gid) if ctx.lineage_filter_on else None
    focal_compressed_ancestors = frozenset()
    if ctx.lineage_filter_on and ctx.lineage_match == "pangonet":
        if cur_lin is not None:
            focal_compressed_ancestors = ctx.lineage_to_focal_ancestors.get(
                cur_lin, frozenset(),
            )

    best_dist = np.inf
    best_gid: Optional[str] = None

    for attach_d_root, side_child_id in ctx.focal_side_refs.get(cur_gid, ()):
        for cand_gid in iter_subtree_leaf_gids(ctx.tree, side_child_id):
            if cand_gid == cur_gid:
                continue
            cand_ns = ctx.gid_to_date_ns.get(cand_gid)
            if cand_ns is None or cand_ns >= date_ns:
                continue
            if _is_identical_seq(ctx.seq_lookup, cand_gid, cur_gid):
                stats.skipped_identical += 1
                continue

            cand_d_root = ctx.gid_to_d_root.get(cand_gid)
            if cand_d_root is None:
                continue
            pdist = cur_d_root + cand_d_root - 2.0 * attach_d_root
            if (ctx.max_patristic_distance is not None
                    and pdist > ctx.max_patristic_distance):
                stats.skipped_max_dist += 1
                continue

            if ctx.lineage_filter_on:
                if not _stem_lineage_predecessor_ok(
                        ctx, cand_gid, cur_lin, focal_compressed_ancestors,
                ):
                    stats.skipped_lineage += 1
                    continue

            if (pdist < best_dist
                    or (pdist == best_dist
                        and (best_gid is None or cand_gid < best_gid))):
                best_dist = pdist
                best_gid = cand_gid

    if best_gid is None:
        stats.no_candidate = 1
    return best_gid, stats


def _merge_stem_batch_result(
        merged: StemSearchResult,
        part: StemSearchResult,
) -> None:
    merged.links.update(part.links)
    merged.stats.no_date += part.stats.no_date
    merged.stats.no_candidate += part.stats.no_candidate
    merged.stats.skipped_identical += part.stats.skipped_identical
    merged.stats.skipped_lineage += part.stats.skipped_lineage
    merged.stats.skipped_max_dist += part.stats.skipped_max_dist


def _serial_stem_batch(
        focal_gids: List[str],
        ctx: StemSearchContext,
        *,
        log_every: int = 1000,
) -> StemSearchResult:
    out = StemSearchResult()
    n_total = len(focal_gids)
    t0 = time.time()
    for i, cur_gid in enumerate(focal_gids, start=1):
        pred_gid, focal_stats = pick_stem_predecessor_for_focal(cur_gid, ctx)
        if pred_gid is not None:
            out.links[cur_gid] = pred_gid
        out.stats.no_date += focal_stats.no_date
        out.stats.no_candidate += focal_stats.no_candidate
        out.stats.skipped_identical += focal_stats.skipped_identical
        out.stats.skipped_lineage += focal_stats.skipped_lineage
        out.stats.skipped_max_dist += focal_stats.skipped_max_dist
        if log_every > 0 and i % log_every == 0:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0.0
            logger.info(
                "by_nearest_stem: serial search %d / %d (%.1fs, %.0f leaves/s, "
                "%d links so far)",
                i, n_total, elapsed, rate, len(out.links),
            )
    return out


def _process_stem_focal_batch(focal_gids: List[str]) -> StemSearchResult:
    if _STEM_WORKER_CTX is None:
        raise RuntimeError("stem worker context not initialized")
    out = StemSearchResult()
    for cur_gid in focal_gids:
        pred_gid, focal_stats = pick_stem_predecessor_for_focal(
            cur_gid, _STEM_WORKER_CTX,
        )
        if pred_gid is not None:
            out.links[cur_gid] = pred_gid
        out.stats.no_date += focal_stats.no_date
        out.stats.no_candidate += focal_stats.no_candidate
        out.stats.skipped_identical += focal_stats.skipped_identical
        out.stats.skipped_lineage += focal_stats.skipped_lineage
        out.stats.skipped_max_dist += focal_stats.skipped_max_dist
    return out


def _init_stem_worker(ctx: StemSearchContext) -> None:
    global _STEM_WORKER_CTX
    _STEM_WORKER_CTX = ctx


def _chunk_list(items: Sequence[str], n_chunks: int) -> List[List[str]]:
    n_chunks = max(1, min(n_chunks, len(items)))
    chunks: List[List[str]] = [[] for _ in range(n_chunks)]
    for i, item in enumerate(items):
        chunks[i % n_chunks].append(item)
    return [c for c in chunks if c]


def _estimate_context_mb(ctx: StemSearchContext) -> float:
    side_ref_count = sum(len(v) for v in ctx.focal_side_refs.values())
    n_nodes = len(ctx.tree.children) + len(ctx.tree.leaf_gids)
    return (
        len(ctx.focal_side_refs) * 64
        + side_ref_count * 24
        + n_nodes * 48
        + len(ctx.gid_to_d_root) * 48
    ) / (1024 * 1024)


def run_stem_predecessor_search(
        focal_gids: Sequence[str],
        ctx: StemSearchContext,
        *,
        n_jobs: int = 1,
        log_every: int = 1000,
) -> StemSearchResult:
    """Run stem search serially or via :class:`ProcessPoolExecutor`."""
    n_workers = resolve_n_jobs(n_jobs)
    focal_list = list(focal_gids)
    if not focal_list:
        return StemSearchResult()

    if n_workers == 1:
        logger.info("by_nearest_stem: serial search over %d leaves", len(focal_list))
        return _serial_stem_batch(focal_list, ctx, log_every=log_every)

    chunks = _chunk_list(focal_list, n_workers)
    n_pool_workers = min(n_workers, len(chunks))
    ctx_mb = _estimate_context_mb(ctx)
    logger.info(
        "by_nearest_stem: starting process pool (%d workers, %d chunks, "
        "~%.0f MB context to unpickle per worker) ...",
        n_pool_workers, len(chunks), ctx_mb,
    )
    logger.info(
        "by_nearest_stem: parallel search over %d leaves",
        len(focal_list),
    )
    merged = StemSearchResult()
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(
            max_workers=n_pool_workers,
            initializer=_init_stem_worker,
            initargs=(ctx,),
    ) as pool:
        futures = [
            pool.submit(_process_stem_focal_batch, chunk)
            for chunk in chunks
        ]
        logger.info(
            "by_nearest_stem: submitted %d chunks; waiting for workers ...",
            len(futures),
        )
        for fut in futures:
            part = fut.result()
            _merge_stem_batch_result(merged, part)
            done += 1
            logger.info(
                "by_nearest_stem: chunk %d / %d done (+%d links, total %d, %.1fs)",
                done, len(chunks), len(part.links), len(merged.links),
                time.time() - t0,
            )
    logger.info(
        "by_nearest_stem: parallel search finished in %.1fs",
        time.time() - t0,
    )
    return merged
