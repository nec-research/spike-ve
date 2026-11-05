#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     predecessor_finder.py
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
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .ancestor_predecessor import (
    AncestorSearchContext,
    precompute_ancestor_search_index,
    run_ancestor_predecessor_search,
)
from .ancestor_reversion import ReversionPenaltyConfig, build_reversion_index
from .ancestor_sampling import AncestorSamplingConfig
from .predecessors_json import load_predecessors_json, write_predecessors_json
from .distances import DistanceMatrixHandler
from .lineage_utils import (
    dealias,
    is_pango_compatible,
    is_strict_prefix_ancestor_lineage,
    maybe_load_aliases,
    nearest_leaf_lineage_predecessor_ok,
)
from .pango_anchor_path import (
    PatristicIndex,
    accumulate_predecessor_edges,
    anchored_genome_path_for_focal,
    build_lineage_date_buckets,
    chronological_pool_sorted,
    normalize_lineage_string,
)
from .stem_predecessor import (
    StemSearchContext,
    build_dist_from_root,
    build_stem_pickle_tree,
    precompute_stem_side_ref_index,
    run_stem_predecessor_search,
)
from .phylotree import PhylotreeHandler, get_leaf_date


def _dotted_prefix_dealiased_suffixes(focal_lin: Optional[str], alias_key: Optional[Dict[str, str]]) -> List[str]:
    """Leading dotted prefixes of ``dealias(focal_lin)``."""
    if focal_lin is None or str(focal_lin).strip() == "":
        return []
    base = dealias(str(focal_lin).strip(), alias_key)
    if base is None or not base:
        return []
    parts = base.split(".")
    return [".".join(parts[:i]) for i in range(1, len(parts) + 1)]


def focal_ancestor_normalized_set_cached(
        focal_lin_raw: Optional[str],
        *,
        match_mode_resolved: str,
        pangonet_obj,
        alias_key: Optional[Dict[str, str]],
        memo_by_focal_key: Dict[str, frozenset],
) -> frozenset:
    """Compressed/normalized tags on the curator ancestry ladder (pangonet + focal)."""

    nk = normalize_lineage_string(str(focal_lin_raw), pango=pangonet_obj, alias_key=alias_key)
    if not nk:
        return frozenset()

    memo_key = f"{nk}|{match_mode_resolved}"
    if memo_key in memo_by_focal_key:
        return memo_by_focal_key[memo_key]

    tags: set[str] = set()

    if match_mode_resolved == "pangonet" and pangonet_obj is not None:
        try:
            ancestors = pangonet_obj.get_ancestors(nk)
            ancestors = ancestors or ()
            for name in tuple(ancestors) + (nk,):
                tag = normalize_lineage_string(str(name), pango=pangonet_obj, alias_key=alias_key)
                if tag:
                    tags.add(tag)
        except Exception as e:
            logging.debug(
                "pangonet.get_ancestors failed for %r: %s; using dotted prefixes.",
                nk,
                e,
            )

    if not tags:
        for d in _dotted_prefix_dealiased_suffixes(focal_lin_raw, alias_key):
            tag = normalize_lineage_string(d, pango=pangonet_obj, alias_key=alias_key)
            if tag:
                tags.add(tag)

    fz = frozenset(tags)
    memo_by_focal_key[memo_key] = fz
    return fz


class PredecessorFinder:
    def __init__(self, data_manager, seq_lookup: Optional[Dict[str, str]] = None):
        self.distances = DistanceMatrixHandler()
        # self.tree = PhylotreeHandler(
        #     data_manager.phylotree, data_manager.df_labels, data_manager.timestamp_field
        # )
        self.phylotree = data_manager.phylotree
        self.df_labels = data_manager.df_labels
        self.timestamp_field = data_manager.timestamp_field
        self.candidate_predecessors = {}
        self.predecessors_metadata: Optional[Dict[str, Any]] = None

        # Optional genome_id -> protein sequence map. When available, every
        # predecessor-search routine in this class skips (parent, child) pairs
        # whose Spike sequences are byte-identical so that downstream series
        # never contain zero-mutation jumps. We try to build it from the
        # DataManager's MSA contents if it isn't passed in explicitly.
        if seq_lookup is None:
            seq_lookup = self._try_build_seq_lookup(data_manager)
        self.seq_lookup: Dict[str, str] = seq_lookup or {}
        if self.seq_lookup:
            logging.info(
                "PredecessorFinder: identical-sequence filter ENABLED "
                "(seq_lookup has %d genome_ids).",
                len(self.seq_lookup),
            )
        else:
            logging.info(
                "PredecessorFinder: identical-sequence filter DISABLED "
                "(no seq_lookup available)."
            )

    @staticmethod
    def _try_build_seq_lookup(data_manager) -> Dict[str, str]:
        """Best-effort genome_id -> sequence map from DataManager's MSA."""
        labels = getattr(data_manager, "sequence_label_dicts", None)
        seqs = getattr(data_manager, "sequence_strs", None)
        if not labels or not seqs:
            return {}
        out: Dict[str, str] = {}
        for ld, s in zip(labels, seqs):
            gid = ld.get("genome_id") if isinstance(ld, dict) else None
            if gid:
                out[gid] = s
        return out

    def _is_identical_seq(self, gid_a: Optional[str], gid_b: Optional[str]) -> bool:
        """True iff both gids are present in seq_lookup and map to identical sequences.

        When the sequence of either side is unknown we conservatively return False
        so the predecessor link is kept; the downstream trivial-copy filter will
        catch any remaining duplicates.
        """
        if not self.seq_lookup or gid_a is None or gid_b is None:
            return False
        sa = self.seq_lookup.get(gid_a)
        sb = self.seq_lookup.get(gid_b)
        return sa is not None and sb is not None and sa == sb

    def by_timeslice(self, final_timeslice=None):
        """ Arrange the data samples into series that go through the time slices.
        Use the distance_matrix to get the distance values. here should be minimum one time slice in collection_date
        between current genome_id and it's predecessor.

        The output is stored as a dictionary of the following structure
            {genome_id : predecessor_id}
        """
        if 'Timeslice' not in self.df_labels.columns:
            raise ValueError("Do the time slicing first.")
        if self.distances.matrix is None or self.distances.labels is None:
            raise ValueError("Distance matrix required.")

        # get unique time slices in a descending order
        timeslices = self.df_labels['Timeslice'].tolist()
        sorted_timeslices = sorted(list(set(timeslices)), reverse=True)
        # keep only time slices earlier than a given time slice
        if final_timeslice is not None:
            idx = 0
            while sorted_timeslices[idx] > final_timeslice:
                idx += 1
            sorted_timeslices = sorted_timeslices[idx:]

        if len(timeslices) < 2:
            raise Exception("Less than 2 time slices, can't proceed.")

        logging.info(f"Start searching predecessor candidates.")

        start = time.time()

        self.candidate_predecessors = {}
        n_skipped_identical = 0
        for idx, t in enumerate(sorted_timeslices):
            print(f"Time slice '{t}'")
            # with points within current time slice
            cur_t_ids = self.df_labels.index[self.df_labels['Timeslice'] == t].tolist()
            # entries with earlier time slices
            prev_t_ids = self.df_labels.index[self.df_labels['Timeslice'] < t].tolist()
            # exit if no sequences in previous time slices
            if len(prev_t_ids) < 1:
                break

            prev_t_seq_ids = set(self.df_labels.loc[prev_t_ids]['genome_id'])
            for t_idx in cur_t_ids:
                # pick a sequence
                cur_t_seq_id = self.df_labels.loc[t_idx]['genome_id']
                if (cur_t_seq_id in self.candidate_predecessors) or (cur_t_seq_id not in self.distances.labels):
                    # skip if sequence already preprocessed
                    continue
                # find predecessor for the sequence
                cur_t_seq_id_index = self.distances.labels.index(cur_t_seq_id)
                distances = self.distances.matrix[cur_t_seq_id_index]

                # Mask out invalid candidates
                mask = [
                    (i != cur_t_seq_id_index) and (self.distances.labels[i] in prev_t_seq_ids)
                    for i in range(len(distances))
                ]
                valid_indices = np.where(mask)[0]

                if valid_indices.size == 0:
                    continue

                # Walk valid candidates from closest to farthest and pick the
                # first one whose Spike sequence is NOT identical to the current
                # leaf. This avoids zero-mutation predecessor links.
                order = valid_indices[np.argsort(distances[valid_indices])]
                closest_id = None
                for cand_idx in order:
                    cand_id = self.distances.labels[cand_idx]
                    if self._is_identical_seq(cand_id, cur_t_seq_id):
                        n_skipped_identical += 1
                        continue
                    closest_id = cand_id
                    break
                if closest_id is None:
                    continue

                self.candidate_predecessors[cur_t_seq_id] = closest_id
                print(f"{closest_id} <- {cur_t_seq_id}")

        end = time.time()

        logging.info(
            f"Predecessor candidates have been calculated in {end - start:.1f} sec; "
            f"links={len(self.candidate_predecessors)}, "
            f"skipped-identical-seq candidates={n_skipped_identical}"
        )

        return self.candidate_predecessors

    def by_timestep(self, time_step="10 days"):
        """
        Use the distance_matrix to get the distance values. There should be minimum time_step in collection_date between
        current genome_id and it's predecessor.

        The output is stored as a dictionary of the following structure
            {genome_id : predecessor_id}
        """
        if (self.distances.matrix is None) or (self.distances.labels is None):
            raise ValueError("Precalculate sequence-to-sequence distance matrix first.")

        logging.info(f"Start calculating predecessors by time step = {time_step}...")

        start = time.time()

        self.candidate_predecessors = {}
        n_skipped_identical = 0
        # sort unique time slices in a descending order
        seq_counter = 0
        for t_idx, row in self.df_labels.iterrows():
            if seq_counter % 100 == 0:
                logging.info(f"Seq. {seq_counter} / {len(self.df_labels)}")
            seq_counter += 1
            t = row[self.timestamp_field]
            # entries with earlier time slices
            prev_t_ids = self.df_labels.index[
                self.df_labels[self.timestamp_field] <= t - pd.Timedelta(time_step)].tolist()
            # exit if no sequences in previous time slices
            if len(prev_t_ids) < 1:
                # skip if nothing to calculate
                continue

            prev_t_seq_ids = set(self.df_labels.loc[prev_t_ids]['genome_id'])
            # pick a sequence
            cur_t_seq_id = row['genome_id']
            if (cur_t_seq_id in self.candidate_predecessors) or (cur_t_seq_id not in self.distances.labels):
                # skip if sequence already preprocessed
                continue
            # find predecessor for the sequence
            cur_t_seq_id_index = self.distances.labels.index(cur_t_seq_id)
            distances = self.distances.matrix[cur_t_seq_id_index]

            # Mask out invalid candidates
            mask = [
                (i != cur_t_seq_id_index) and (self.distances.labels[i] in prev_t_seq_ids)
                for i in range(len(distances))
            ]
            valid_indices = np.where(mask)[0]

            if valid_indices.size == 0:
                continue
            # Walk valid candidates from closest to farthest and pick the first
            # one whose Spike sequence is NOT identical to the current leaf to
            # avoid zero-mutation predecessor links.
            order = valid_indices[np.argsort(distances[valid_indices])]
            closest_id = None
            for cand_idx in order:
                cand_id = self.distances.labels[cand_idx]
                if self._is_identical_seq(cand_id, cur_t_seq_id):
                    n_skipped_identical += 1
                    continue
                closest_id = cand_id
                break
            if closest_id is None:
                continue

            self.candidate_predecessors[cur_t_seq_id] = closest_id

        end = time.time()

        logging.info(
            f"Predecessor candidates have been calculated in {end - start:.1f} sec; "
            f"links={len(self.candidate_predecessors)}, "
            f"skipped-identical-seq candidates={n_skipped_identical}"
        )

    def by_phylotree(self, closest_by_distance=False, time_step="7 days"):
        """ Arrange the data samples into series that go through all the time slices.
        Use phylogenetic tree to find predecessors - pick a real sequence closest (by time or distance)
        to each internal node of the phylogenetic tree.

        The output is stored as a dictionary of the following structure:

        """
        if 'Timeslice' not in list(self.df_labels.columns):
            raise ValueError("Do the time slicing first.")

        timeslices = self.df_labels['Timeslice'].tolist()
        if len(timeslices) < 2:
            raise Exception("Less than 2 time slices, can't proceed.")

        logging.info(f"Start calculating predecessors using phylogenetic tree...")

        start = time.time()

        # links to closes leaf from each internal node in the tree
        node_to_leaf_dict = {}
        # reconstructed predecessor links between leaves
        leaf_to_leaf_dict = {}

        # 1. Traverse the tree first time to fill node-to-leaf links
        logging.info(f"Traverse tree to calculate links for internal nodes.")
        node_idx = 0
        num_nodes = len(list(self.phylotree.nodes()))
        for node in self.phylotree.postorder_node_iter():
            node_idx += 1
            logging.info(f"Node {node_idx} / {num_nodes}...")

            if node.is_leaf():
                node_to_leaf_dict[node] = node
            else:
                # node is NOT a leaf
                candidate_leaves = [
                    node_to_leaf_dict[l] for l in node.child_node_iter()
                ]
                if candidate_leaves:
                    if not closest_by_distance:
                        best_leaf = min(
                            candidate_leaves,
                            key=lambda x: get_leaf_date(x.taxon.label, self.df_labels, self.timestamp_field),
                        )
                    else:
                        best_leaf = min(
                            candidate_leaves,
                            key=lambda x: self.phylotree.phylogenetic_distance(node, x),
                        )
                    node_to_leaf_dict[node] = best_leaf
                else:
                    logging.info(f"No candidate leaves for node {node.label}")

        # 2. Iterate over leaves and for each leave find candidate predecessor from above links
        logging.info(f"Traverse tree to calculate predecessors.")
        leaf_idx = 0
        n_skipped_identical = 0
        num_leaves = len(self.phylotree.leaf_nodes())
        for current_leaf in self.phylotree.leaf_node_iter():
            leaf_idx += 1
            logging.info(f"Leaf {leaf_idx} / {num_leaves}...")
            current_leaf_date = get_leaf_date(current_leaf.taxon.label, self.df_labels, self.timestamp_field)
            cur_gid = current_leaf.taxon.label.split('-')[-1]
            node = current_leaf
            while node is not None:
                closest_leaf = node_to_leaf_dict[node]
                if ((closest_leaf is not current_leaf)
                        and (get_leaf_date(closest_leaf.taxon.label, self.df_labels, self.timestamp_field)
                             < current_leaf_date)):
                    cand_gid = closest_leaf.taxon.label.split('-')[-1]
                    # Skip identical-sequence (zero-mutation) parent->child links
                    # by walking further up the tree to look for a non-identical
                    # ancestor leaf.
                    if self._is_identical_seq(cand_gid, cur_gid):
                        n_skipped_identical += 1
                        node = node.parent_node
                        continue
                    leaf_to_leaf_dict[current_leaf.taxon.label] = closest_leaf.taxon.label
                    break
                node = node.parent_node

        logging.info(f"Trim label prefixes.")
        self.candidate_predecessors = {}
        for key, val in leaf_to_leaf_dict.items():
            self.candidate_predecessors[key.split('-')[-1]] = val.split('-')[-1]

        end = time.time()
        logging.info(
            f"Predecessor candidates have been calculated in {end - start:.1f} sec; "
            f"links={len(self.candidate_predecessors)}, "
            f"skipped-identical-seq ancestors={n_skipped_identical}"
        )

    def by_nearestleaf(
            self,
            max_patristic_distance: Optional[float] = None,
            lineage_col: Optional[str] = None,
            lineage_min_shared_depth: int = 2,
            lineage_aliases_path: Optional[str] = None,
            lineage_recombinant_parents: Optional[Dict[str, List[str]]] = None,
            distance_fallback_factor: Optional[float] = None,
            log_every: int = 1000,
            lineage_match: str = "compat",
            lineage_ascent_exclusive: bool = False,
    ):
        """L-anchored nearest-earlier-leaf predecessor finder over a phylogenetic tree.

        For each focal leaf ``L`` with collection date ``t_L`` we find the
        observed leaf that best approximates its predecessor, defined as:

            argmin_{C} patristic_distance(L, C)
            subject to:
                t_C  <  t_L                                  (earlier in time)
                C    !=  L                                   (not self)
                C    in  smallest_subtree_of(L) containing
                          at least one earlier-dated leaf    (clade-restricted)
                seq(C) != seq(L)                             (when seq_lookup set)
                lineage(C) obeys ``lineage_match`` vs lineage(L)  (when lineage_col set)
                patristic_distance(L, C) <= max_patristic_distance  (when set)

        Pango-aware lineage filter (opt-in)
        -----------------------------------
        When ``lineage_col`` is ``None`` (the default) the lineage filter is
        **disabled** and every earlier leaf in the chosen subtree is eligible.

        When ``lineage_col`` is provided, lineage rules depend on ``lineage_match``
        (see Parameters). Default ``compat`` uses
        :func:`lineage_utils.is_pango_compatible`. Modes ``ancestor`` (strict
        de-aliased prefix) and ``pangonet`` (curator ancestry set from
        [pangonet](https://github.com/phac-nml/pangonet), with dotted-prefix
        fallback when unavailable) tighten naming consistency.

        ``lineage_ascent_exclusive``
        ---------------------------
        When True (requires ``lineage_col``), incompatible candidates never
        relax the traversal early-stop bound: the walker keeps ascending toward
        the root until lineage-qualified predecessors are exhausted.

        Distance escape hatch
        ---------------------

        ``distance_fallback_factor`` (in (0, 1]) lets the search keep a
        secondary "best any-lineage" candidate. If no lineage-compatible
        match is found within ``max_patristic_distance``, the search falls
        back to the any-lineage candidate provided its distance is below
        ``distance_fallback_factor * max_patristic_distance``. This catches
        genuine but undocumented lateral jumps without opening the floodgates.
        Disabled when ``None`` or when ``max_patristic_distance`` is ``None``.

        Conceptual caveats that *cannot* be removed without ancestral sequence
        reconstruction:
          * The selected ``C`` is still an *observed leaf* used as a proxy for
            an unobserved ancestor (so it is technically a cousin, not a
            parent).
          * Collection date is used as a proxy for branching time.

        Parameters
        ----------
        max_patristic_distance : float, optional
            Hard cap on patristic distance to ``L``. ``None`` = no cap.
        lineage_col : str, optional
            Column in ``df_labels`` holding the Pango/WHO lineage per genome
            (e.g. ``"lineage"`` or ``"who_lineage"``). When ``None`` the
            lineage filter is disabled entirely.
        lineage_min_shared_depth : int, default 2
            Minimum number of de-aliased dotted segments two lineages must
            share to be considered Pango-compatible. Set to ``999`` for strict
            ancestor/descendant; set to ``0`` to disable the hierarchical
            check (and rely solely on the recombinant whitelist).
        lineage_aliases_path : str, optional
            Path to Pango ``alias_key.json``. Strongly recommended whenever
            ``lineage_col`` is set so aliased names like ``BA.2`` are
            de-aliased to ``B.1.1.529.2`` before prefix comparison.
        lineage_recombinant_parents : dict, optional
            ``{child_lineage: [parent_lineage, ...]}`` whitelist of
            cross-lineage transitions (most useful for ``X*`` recombinants).
        distance_fallback_factor : float, optional
            See "Distance escape hatch" above.
        lineage_match : str, default "compat"
            ``compat`` uses :func:`lineage_utils.is_pango_compatible`; ``ancestor``
            enforces strict de-aliased dotted prefix ancestry; ``pangonet``
            requires candidate tags in the focal's curator ancestor set via
            [pangonet](https://github.com/phac-nml/pangonet) (fallback: dotted prefixes).
        log_every : int, optional
            Log leaf progress interval (use ``0`` to disable).
        lineage_ascent_exclusive : bool, default False
            When lineage filtering is enabled, disallow lineage-incompatible
            cousins from widening the pruning ``ceiling`` (keeps ascending the
            tree for eligible matches first).
        """
        if self.phylotree is None:
            raise ValueError("Phylogenetic tree is required for by_nearestleaf.")
        if self.timestamp_field not in self.df_labels.columns:
            raise ValueError(
                f"timestamp_field '{self.timestamp_field}' missing from df_labels."
            )
        if (distance_fallback_factor is not None and
                (distance_fallback_factor <= 0 or distance_fallback_factor > 1)):
            raise ValueError(
                f"distance_fallback_factor must be in (0, 1] or None, "
                f"got {distance_fallback_factor!r}"
            )
        if (distance_fallback_factor is not None and max_patristic_distance is None):
            logging.warning(
                "distance_fallback_factor set but max_patristic_distance is None; "
                "fallback has no effect (no distance cap to scale against)."
            )

        # ---- 0. Lineage filter setup --------------------------------------
        gid_to_lineage: Optional[Dict[str, str]] = None
        alias_key: Optional[Dict[str, str]] = None
        recomb_parents: Optional[Dict[str, List[str]]] = None
        lineage_filter_on = lineage_col is not None
        if lineage_filter_on:
            if lineage_col not in self.df_labels.columns:
                raise ValueError(
                    f"lineage_col='{lineage_col}' not in df_labels columns."
                )
            alias_key, _auto_recomb = maybe_load_aliases(lineage_aliases_path)
            recomb_parents = lineage_recombinant_parents
            if recomb_parents is None and _auto_recomb:
                # Use auto-extracted recombinant table from alias_key.json
                recomb_parents = _auto_recomb

        ascent_exclusive_effective = bool(lineage_ascent_exclusive and lineage_filter_on)
        if lineage_ascent_exclusive and not lineage_filter_on:
            logging.warning(
                "lineage_ascent_exclusive requires lineage_col; ignoring ascent-exclusive mode.",
            )

        match_raw = str(lineage_match or "compat").strip().lower()
        if match_raw not in ("compat", "ancestor", "pangonet"):
            raise ValueError(
                "lineage_match must be one of {'compat','ancestor','pangonet'}; "
                f"got {lineage_match!r}"
            )
        lineage_match_effective = match_raw

        pangonet_for_nearest: Any = None
        if lineage_filter_on and lineage_match_effective == "pangonet":
            try:
                from pangonet.pangonet import PangoNet  # type: ignore[import-untyped]

                pangonet_for_nearest = PangoNet().build()
                logging.info("by_nearestleaf: loaded pangonet for lineage_match=pangonet.")
            except Exception as exc:  # ImportError, or a pangonet data/build error (e.g. KeyError)
                logging.warning("pangonet unavailable (%r); see fallback message below.", exc)
                logging.warning(
                    "lineage_match='pangonet' but pangonet is not installed or failed to build; "
                    "falling back to lineage_match='ancestor' (prefix ancestry). "
                    'Install pangonet via: pip install "pangonet @ git+https://github.com/phac-nml/pangonet.git"'
                )
                lineage_match_effective = "ancestor"

        ancestry_memo: Dict[str, frozenset] = {}

        logging.info(
            "by_nearestleaf: max_patristic_distance=%s, lineage_filter=%s, "
            "lineage_min_shared_depth=%s, lineage_match=%s, ascent_exclusive=%s, "
            "pangonet_loaded=%s, recombinant_parents=%s, "
            "distance_fallback_factor=%s, seq_lookup=%s",
            max_patristic_distance,
            lineage_col if lineage_filter_on else "OFF",
            lineage_min_shared_depth if lineage_filter_on else "n/a",
            lineage_match_effective if lineage_filter_on else "n/a",
            ascent_exclusive_effective,
            lineage_match_effective == "pangonet" and pangonet_for_nearest is not None,
            (len(recomb_parents) if recomb_parents else 0) if lineage_filter_on else "n/a",
            distance_fallback_factor,
            "ON" if self.seq_lookup else "OFF",
        )
        start = time.time()

        # ---- 1. Pre-build lookups -----------------------------------------
        gids = self.df_labels['genome_id'].astype(str)
        dates = pd.to_datetime(self.df_labels[self.timestamp_field], errors='coerce')
        gid_to_date: Dict[str, pd.Timestamp] = dict(zip(gids, dates))

        if lineage_filter_on:
            gid_to_lineage = dict(zip(
                gids,
                self.df_labels[lineage_col].astype(str),
            ))

        # ---- 2. dist_from_root for every node (preorder) ------------------
        dist_from_root: Dict = {self.phylotree.seed_node: 0.0}
        for node in self.phylotree.preorder_node_iter():
            if node is self.phylotree.seed_node:
                continue
            parent_d = dist_from_root.get(node.parent_node, 0.0)
            edge_len = node.edge_length if node.edge_length is not None else 0.0
            dist_from_root[node] = parent_d + edge_len

        # ---- 3. node -> list of descendant leaves (postorder) -------------
        # We share leaf-node references across levels; each leaf appears in
        # every ancestor's list. Total memory is O(sum of subtree sizes).
        node_leaves: Dict = {}
        for node in self.phylotree.postorder_node_iter():
            if node.is_leaf():
                node_leaves[node] = [node]
            else:
                descendants: List[Any] = []
                for child in node.child_node_iter():
                    descendants.extend(node_leaves.get(child, []))
                node_leaves[node] = descendants

        n_leaves_total = len(self.phylotree.leaf_nodes())
        logging.info(
            "by_nearestleaf: pre-build complete in %.1fs (nodes=%d, leaves=%d)",
            time.time() - start, len(dist_from_root), n_leaves_total,
        )

        # ---- 4. Per-leaf nearest-earlier search ---------------------------
        fallback_cap: Optional[float] = None
        if (lineage_filter_on
                and distance_fallback_factor is not None
                and max_patristic_distance is not None):
            fallback_cap = float(distance_fallback_factor) * float(max_patristic_distance)

        self.candidate_predecessors = {}
        n_no_date = 0
        n_no_candidate = 0
        n_skipped_identical = 0
        n_skipped_lineage = 0
        n_skipped_max_dist = 0
        n_used_fallback = 0
        leaf_idx = 0

        for current_leaf in self.phylotree.leaf_node_iter():
            leaf_idx += 1
            if log_every > 0 and leaf_idx % log_every == 0:
                logging.info("by_nearestleaf: leaf %d / %d", leaf_idx, n_leaves_total)

            cur_gid = current_leaf.taxon.label.split('-')[-1]
            cur_date = gid_to_date.get(cur_gid)
            if cur_date is None or pd.isna(cur_date):
                n_no_date += 1
                continue

            cur_lin = gid_to_lineage.get(cur_gid) if lineage_filter_on else None
            cur_d_root = dist_from_root[current_leaf]

            focal_compressed_ancestors = frozenset()
            if lineage_filter_on and lineage_match_effective == "pangonet":
                focal_compressed_ancestors = focal_ancestor_normalized_set_cached(
                    cur_lin,
                    match_mode_resolved="pangonet",
                    pangonet_obj=pangonet_for_nearest,
                    alias_key=alias_key,
                    memo_by_focal_key=ancestry_memo,
                )

            best_dist = np.inf
            best_gid: Optional[str] = None
            best_fb_dist = np.inf
            best_fb_gid: Optional[str] = None

            # Walk up; at each ancestor `node`, candidates are the leaves in
            # sibling subtrees (not the child branch we came from). For all such
            # candidates LCA(L, C) = node, so patristic_distance(L, C) reduces to
            #     dist_from_root[L] + dist_from_root[C] - 2 * dist_from_root[node].
            prev_child = current_leaf
            node = current_leaf.parent_node
            while node is not None:
                node_d_root = dist_from_root[node]
                # Lower bound on patristic distance for any leaf reachable from
                # this ancestor or higher: distance from L up to `node`.
                lb_to_node = cur_d_root - node_d_root
                # Tighter early-stop: the best possible distance from any leaf
                # at or above this ancestor cannot beat lb_to_node, and we only
                # care about beating either of the two trackers that matter.
                ceiling = best_dist
                if fallback_cap is not None and not (
                        lineage_filter_on and ascent_exclusive_effective
                ):
                    ceiling = max(ceiling, min(best_fb_dist, fallback_cap))
                if lb_to_node >= ceiling:
                    break
                if (max_patristic_distance is not None
                        and lb_to_node > max_patristic_distance):
                    break

                for child in node.child_node_iter():
                    if child is prev_child:
                        continue
                    for cand_leaf in node_leaves.get(child, ()):
                        cand_gid = cand_leaf.taxon.label.split('-')[-1]
                        if cand_gid == cur_gid:
                            continue
                        cand_date = gid_to_date.get(cand_gid)
                        if cand_date is None or pd.isna(cand_date) or cand_date >= cur_date:
                            continue
                        if self._is_identical_seq(cand_gid, cur_gid):
                            n_skipped_identical += 1
                            continue
                        pdist = (cur_d_root + dist_from_root[cand_leaf]
                                 - 2.0 * node_d_root)
                        if (max_patristic_distance is not None
                                and pdist > max_patristic_distance):
                            n_skipped_max_dist += 1
                            continue

                        if lineage_filter_on:
                            cand_lin = gid_to_lineage.get(cand_gid)
                            lin_ok = nearest_leaf_lineage_predecessor_ok(
                                cand_lin,
                                cur_lin,
                                lineage_match=lineage_match_effective,
                                lineage_min_shared_depth=lineage_min_shared_depth,
                                recombinant_parents=recomb_parents,
                                alias_key=alias_key,
                                focal_ancestors_compressed=focal_compressed_ancestors,
                                pangonet_obj=pangonet_for_nearest,
                            )
                            if lin_ok:
                                if (pdist < best_dist
                                        or (pdist == best_dist
                                            and (best_gid is None or cand_gid < best_gid))):
                                    best_dist = pdist
                                    best_gid = cand_gid
                            else:
                                n_skipped_lineage += 1
                                # Track for distance-fallback escape hatch.
                                if (fallback_cap is not None
                                        and pdist <= fallback_cap
                                        and (pdist < best_fb_dist
                                             or (pdist == best_fb_dist
                                                 and (best_fb_gid is None
                                                      or cand_gid < best_fb_gid)))):
                                    best_fb_dist = pdist
                                    best_fb_gid = cand_gid
                        else:
                            if (pdist < best_dist
                                    or (pdist == best_dist
                                        and (best_gid is None or cand_gid < best_gid))):
                                best_dist = pdist
                                best_gid = cand_gid

                prev_child = node
                node = node.parent_node

            if best_gid is not None:
                self.candidate_predecessors[cur_gid] = best_gid
            elif best_fb_gid is not None:
                self.candidate_predecessors[cur_gid] = best_fb_gid
                n_used_fallback += 1
            else:
                n_no_candidate += 1

        elapsed = time.time() - start
        logging.info(
            "by_nearestleaf done in %.1fs: %d links | "
            "no-date=%d, no-candidate=%d, skipped-identical-seq=%d, "
            "skipped-incompatible-lineage=%d, skipped-over-max-dist=%d, "
            "used-distance-fallback=%d",
            elapsed,
            len(self.candidate_predecessors),
            n_no_date, n_no_candidate, n_skipped_identical,
            n_skipped_lineage, n_skipped_max_dist, n_used_fallback,
        )
        return self.candidate_predecessors

    def by_nearest_ancestor(
            self,
            max_patristic_distance: Optional[float] = None,
            min_patristic_distance: Optional[float] = None,
            predecessor_rank: int = 1,
            lineage_col: Optional[str] = None,
            lineage_min_shared_depth: int = 2,
            lineage_aliases_path: Optional[str] = None,
            lineage_recombinant_parents: Optional[Dict[str, List[str]]] = None,
            log_every: int = 1000,
            lineage_match: str = "ancestor",
            n_jobs: int = 1,
            predecessor_sampling: Optional[Dict[str, Any]] = None,
            reversion_penalty: Optional[Dict[str, Any]] = None,
    ):
        """Nearest earlier leaf in the focal's nested ancestor clade (stem-child search).

        This method **is** ``candidate_scope='nested'`` from the design discussion:
        at each ancestor on the focal→root path only the stem-child subtree
        (toward the root) is searched — never parallel sibling subtrees
        (contrast with :meth:`by_nearestleaf`, ``candidate_scope='sibling'``).

        Candidates are drawn from stem-child subtrees at each step on the
        focal→root path (indexed once, O(depth) lookups per focal).

        Among eligible candidates ``C`` the predecessor minimizes patristic
        distance subject to:

            t_C  <  t_L
            dist_root[C]  <  dist_root[L]   (rootward on the same trunk)
            seq(C) != seq(L) when ``seq_lookup`` is set
            optional ``lineage_match`` when ``lineage_col`` is set
            optional ``max_patristic_distance`` / ``min_patristic_distance`` caps
            ``predecessor_rank`` — pick the k-th nearest when
            ``predecessor_sampling.selection`` is ``nearest`` (default)

        predecessor_sampling
            Optional JSON/YAML object controlling predecessor selection among
            eligible nested-clade candidates.  When ``selection`` is ``sample``,
            draws one predecessor with truncated patristic softmax weights
            (see ``ancestor_sampling.AncestorSamplingConfig``).  Settings are
            stored in the output JSON ``_meta`` block.

        reversion_penalty
            Optional asymmetric branch / non-reversion score (see
            ``ancestor_reversion``): candidates are ranked by
            ``d_eff = b_F + alpha * b_P + beta * n_rev`` instead of the patristic distance
            ``b_F + b_P`` (``b_P`` = candidate's private terminal branch, ``n_rev`` = number of S1
            columns where the child restores the parent-lineage consensus / root residue).
            ``null`` or ``enabled: false`` keeps the legacy scoring.

        n_jobs
            Parallel workers (``1`` = serial, ``-1`` = all CPUs), same as
            :meth:`by_nearest_stem`.

        Recommended ``lineage_match``: ``ancestor`` (strict Pango prefix).
        """
        if self.phylotree is None:
            raise ValueError("Phylogenetic tree is required for by_nearest_ancestor.")
        if self.timestamp_field not in self.df_labels.columns:
            raise ValueError(
                f"timestamp_field '{self.timestamp_field}' missing from df_labels."
            )
        if predecessor_rank < 1:
            raise ValueError(
                f"predecessor_rank must be >= 1, got {predecessor_rank!r}"
            )
        if (min_patristic_distance is not None
                and max_patristic_distance is not None
                and min_patristic_distance > max_patristic_distance):
            raise ValueError(
                "min_patristic_distance cannot exceed max_patristic_distance."
            )

        gid_to_lineage: Optional[Dict[str, str]] = None
        alias_key: Optional[Dict[str, str]] = None
        recomb_parents: Optional[Dict[str, List[str]]] = None
        lineage_filter_on = lineage_col is not None
        if lineage_filter_on:
            if lineage_col not in self.df_labels.columns:
                raise ValueError(
                    f"lineage_col='{lineage_col}' not in df_labels columns."
                )
            alias_key, _auto_recomb = maybe_load_aliases(lineage_aliases_path)
            recomb_parents = lineage_recombinant_parents
            if recomb_parents is None and _auto_recomb:
                recomb_parents = _auto_recomb

        match_raw = str(lineage_match or "ancestor").strip().lower()
        if match_raw not in ("compat", "ancestor", "pangonet"):
            raise ValueError(
                "lineage_match must be one of {'compat','ancestor','pangonet'}; "
                f"got {lineage_match!r}"
            )
        lineage_match_effective = match_raw

        pangonet_for_nearest: Any = None
        if lineage_filter_on and lineage_match_effective == "pangonet":
            try:
                from pangonet.pangonet import PangoNet  # type: ignore[import-untyped]

                pangonet_for_nearest = PangoNet().build()
                logging.info("by_nearest_ancestor: loaded pangonet for lineage_match=pangonet.")
            except Exception as exc:  # ImportError, or a pangonet data/build error (e.g. KeyError)
                logging.warning("pangonet unavailable (%r); see fallback message below.", exc)
                logging.warning(
                    "lineage_match='pangonet' but pangonet is not installed or failed to build; "
                    "falling back to lineage_match='ancestor'. "
                    'Install pangonet via: pip install "pangonet @ git+https://github.com/phac-nml/pangonet.git"'
                )
                lineage_match_effective = "ancestor"

        ancestry_memo: Dict[str, frozenset] = {}
        sampling_cfg = AncestorSamplingConfig.from_mapping(predecessor_sampling)
        reversion_cfg = ReversionPenaltyConfig.from_mapping(reversion_penalty)
        if reversion_cfg is not None:
            if reversion_cfg.reference != "root" and not lineage_filter_on:
                raise ValueError(
                    "reversion_penalty.reference in {'consensus','both'} needs lineage labels: "
                    "set lineage_col (e.g. 'lineage') or use reference: root."
                )
            logging.info("by_nearest_ancestor: reversion_penalty=%s", reversion_cfg.to_json_dict())

        logging.info(
            "by_nearest_ancestor: candidate_scope=nested (stem-child only), "
            "max_patristic_distance=%s, "
            "min_patristic_distance=%s, predecessor_rank=%s, lineage_filter=%s, "
            "lineage_min_shared_depth=%s, lineage_match=%s, seq_lookup=%s, n_jobs=%s, "
            "predecessor_sampling=%s",
            max_patristic_distance,
            min_patristic_distance,
            predecessor_rank,
            lineage_col if lineage_filter_on else "OFF",
            lineage_min_shared_depth if lineage_filter_on else "n/a",
            lineage_match_effective if lineage_filter_on else "n/a",
            "ON" if self.seq_lookup else "OFF",
            n_jobs,
            sampling_cfg.to_json_dict(),
        )
        start = time.time()

        gids = self.df_labels['genome_id'].astype(str)
        dates = pd.to_datetime(self.df_labels[self.timestamp_field], errors='coerce')
        gid_to_date: Dict[str, pd.Timestamp] = dict(zip(gids, dates))
        gid_to_date_ns: Dict[str, int] = {
            g: int(pd.Timestamp(d).value)
            for g, d in gid_to_date.items()
            if d is not None and not pd.isna(d)
        }

        gid_to_lineage_map: Dict[str, str] = {}
        if lineage_filter_on:
            gid_to_lineage_map = dict(zip(
                gids,
                self.df_labels[lineage_col].astype(str),
            ))

        dist_from_root = build_dist_from_root(
            self.phylotree,
            log_every=log_every,
        )
        node_to_id, pickle_tree = build_stem_pickle_tree(self.phylotree)
        n_leaves_total = len(self.phylotree.leaf_nodes())

        (
            gid_to_d_root,
            gid_to_path_ids,
            gid_to_path_set,
            node_id_dist,
        ) = precompute_ancestor_search_index(
            self.phylotree,
            node_to_id,
            dist_from_root,
            gid_to_date_ns,
            log_every=log_every,
        )
        focal_gids = list(gid_to_d_root.keys())

        lineage_to_focal_ancestors: Dict[str, frozenset] = {}
        gid_to_lineage_normalized: Dict[str, str] = {}
        if lineage_filter_on and lineage_match_effective == "pangonet":
            unique_lineages = set(gid_to_lineage_map.values())
            logging.info(
                "by_nearest_ancestor: caching pangonet ancestry for %d unique lineages ...",
                len(unique_lineages),
            )
            for lin in unique_lineages:
                lineage_to_focal_ancestors[lin] = focal_ancestor_normalized_set_cached(
                    lin,
                    match_mode_resolved="pangonet",
                    pangonet_obj=pangonet_for_nearest,
                    alias_key=alias_key,
                    memo_by_focal_key=ancestry_memo,
                )
            for gid, lin in gid_to_lineage_map.items():
                gid_to_lineage_normalized[gid] = normalize_lineage_string(
                    lin,
                    pango=pangonet_for_nearest,
                    alias_key=alias_key,
                )

        reversion_index = None
        if reversion_cfg is not None:
            reversion_index = build_reversion_index(
                self.seq_lookup,
                gid_to_lineage_map,
                reversion_cfg,
            )

        ctx = AncestorSearchContext(
            gid_to_d_root=gid_to_d_root,
            gid_to_date_ns=gid_to_date_ns,
            gid_to_path_ids=gid_to_path_ids,
            gid_to_path_set=gid_to_path_set,
            node_id_dist=node_id_dist,
            pickle_tree=pickle_tree,
            seq_lookup=dict(self.seq_lookup),
            max_patristic_distance=max_patristic_distance,
            min_patristic_distance=min_patristic_distance,
            predecessor_rank=predecessor_rank,
            lineage_filter_on=lineage_filter_on,
            lineage_match=lineage_match_effective,
            lineage_min_shared_depth=lineage_min_shared_depth,
            gid_to_lineage=gid_to_lineage_map,
            lineage_to_focal_ancestors=lineage_to_focal_ancestors,
            gid_to_lineage_normalized=gid_to_lineage_normalized,
            alias_key=alias_key,
            recombinant_parents=recomb_parents,
            sampling=sampling_cfg,
            reversion=reversion_cfg,
            reversion_index=reversion_index,
        )

        search_result = run_ancestor_predecessor_search(
            focal_gids,
            ctx,
            n_jobs=n_jobs,
            log_every=log_every,
        )
        self.candidate_predecessors = search_result.links
        self.predecessors_metadata = {
            "method": "by_nearest_ancestor",
            "predecessor_rank": int(predecessor_rank),
            "lineage_match": lineage_match_effective if lineage_filter_on else None,
            "predecessor_sampling": sampling_cfg.to_json_dict(),
        }
        if reversion_cfg is not None:
            self.predecessors_metadata["reversion_penalty"] = reversion_cfg.to_json_dict()

        st = search_result.stats
        if reversion_cfg is not None and st.picked:
            n = float(st.picked)
            logging.info(
                "by_nearest_ancestor[reversion_penalty]: chosen parents n=%d | mean n_rev=%.3f "
                "(n_rev==0: %.1f%%) | mean n_fwd=%.3f | mean b_parent=%.3f (b_parent==0: %.1f%%) | "
                "mean b_child=%.3f (b_child<min: %.1f%%) | n_fwd<min: %.1f%% | "
                "not the patristic-nearest shortlist member: %.1f%% | shortlist members without sequence=%d",
                st.picked,
                st.picked_n_rev_sum / n,
                100.0 * st.picked_n_rev_zero / n,
                st.picked_n_fwd_sum / n,
                st.picked_b_parent_sum / n,
                100.0 * st.picked_b_parent_zero / n,
                st.picked_b_child_sum / n,
                100.0 * st.picked_b_child_below_min / n,
                100.0 * st.picked_n_fwd_below_min / n,
                100.0 * st.picked_not_nearest_patristic / n,
                st.shortlist_no_sequence,
            )
        elapsed = time.time() - start
        logging.info(
            "by_nearest_ancestor done in %.1fs: %d links | "
            "no-date=%d, no-candidate=%d, skipped-identical-seq=%d, "
            "skipped-incompatible-lineage=%d, skipped-over-max-dist=%d, "
            "skipped-under-min-dist=%d, skipped-not-rootward=%d, "
            "skipped-not-earlier=%d",
            elapsed,
            len(self.candidate_predecessors),
            st.no_date, st.no_candidate, st.skipped_identical,
            st.skipped_lineage, st.skipped_max_dist, st.skipped_min_dist,
            st.skipped_not_rootward, st.skipped_not_earlier,
        )
        return self.candidate_predecessors

    def by_nearest_stem(
            self,
            max_patristic_distance: Optional[float] = None,
            lineage_col: Optional[str] = None,
            lineage_min_shared_depth: int = 2,
            lineage_aliases_path: Optional[str] = None,
            lineage_recombinant_parents: Optional[Dict[str, List[str]]] = None,
            log_every: int = 1000,
            lineage_match: str = "pangonet",
            n_jobs: int = 1,
    ):
        """Nearest earlier leaf on the focal's stem (path toward root).

        Unlike :meth:`by_nearestleaf`, which searches **sibling subtrees** at
        each ancestor (cousins in parallel clades), this method only considers
        leaves on **side branches along the focal->root corridor**: siblings of
        the focal leaf and side branches off internal nodes on that path.
        Parallel clades at the tree root or siblings of entire focal-lineage
        branches at higher ancestors are excluded.

        Among eligible candidates the predecessor is the one with minimum
        patristic distance to the focal leaf, subject to:

            t_C  <  t_L
            seq(C) != seq(L) when ``seq_lookup`` is set
            optional ``lineage_match`` rules when ``lineage_col`` is set
            optional ``max_patristic_distance`` cap

        n_jobs
            Parallel workers for the per-leaf search (``1`` = serial,
            ``-1`` = all CPUs). Tree precomputation always runs in the
            parent process.
        """
        if self.phylotree is None:
            raise ValueError("Phylogenetic tree is required for by_nearest_stem.")
        if self.timestamp_field not in self.df_labels.columns:
            raise ValueError(
                f"timestamp_field '{self.timestamp_field}' missing from df_labels."
            )

        gid_to_lineage: Optional[Dict[str, str]] = None
        alias_key: Optional[Dict[str, str]] = None
        recomb_parents: Optional[Dict[str, List[str]]] = None
        lineage_filter_on = lineage_col is not None
        if lineage_filter_on:
            if lineage_col not in self.df_labels.columns:
                raise ValueError(
                    f"lineage_col='{lineage_col}' not in df_labels columns."
                )
            alias_key, _auto_recomb = maybe_load_aliases(lineage_aliases_path)
            recomb_parents = lineage_recombinant_parents
            if recomb_parents is None and _auto_recomb:
                recomb_parents = _auto_recomb

        match_raw = str(lineage_match or "pangonet").strip().lower()
        if match_raw not in ("compat", "ancestor", "pangonet"):
            raise ValueError(
                "lineage_match must be one of {'compat','ancestor','pangonet'}; "
                f"got {lineage_match!r}"
            )
        lineage_match_effective = match_raw

        pangonet_for_nearest: Any = None
        if lineage_filter_on and lineage_match_effective == "pangonet":
            try:
                from pangonet.pangonet import PangoNet  # type: ignore[import-untyped]

                pangonet_for_nearest = PangoNet().build()
                logging.info("by_nearest_stem: loaded pangonet for lineage_match=pangonet.")
            except Exception as exc:  # ImportError, or a pangonet data/build error (e.g. KeyError)
                logging.warning("pangonet unavailable (%r); see fallback message below.", exc)
                logging.warning(
                    "lineage_match='pangonet' but pangonet is not installed or failed to build; "
                    "falling back to lineage_match='ancestor' (prefix ancestry). "
                    'Install pangonet via: pip install "pangonet @ git+https://github.com/phac-nml/pangonet.git"'
                )
                lineage_match_effective = "ancestor"

        ancestry_memo: Dict[str, frozenset] = {}

        logging.info(
            "by_nearest_stem: max_patristic_distance=%s, lineage_filter=%s, "
            "lineage_min_shared_depth=%s, lineage_match=%s, seq_lookup=%s, n_jobs=%s",
            max_patristic_distance,
            lineage_col if lineage_filter_on else "OFF",
            lineage_min_shared_depth if lineage_filter_on else "n/a",
            lineage_match_effective if lineage_filter_on else "n/a",
            "ON" if self.seq_lookup else "OFF",
            n_jobs,
        )
        start = time.time()

        gids = self.df_labels['genome_id'].astype(str)
        dates = pd.to_datetime(self.df_labels[self.timestamp_field], errors='coerce')
        gid_to_date: Dict[str, pd.Timestamp] = dict(zip(gids, dates))
        gid_to_date_ns: Dict[str, int] = {
            g: int(pd.Timestamp(d).value)
            for g, d in gid_to_date.items()
            if d is not None and not pd.isna(d)
        }
        gid_to_lineage_map: Dict[str, str] = {}
        if lineage_filter_on:
            gid_to_lineage_map = dict(zip(
                gids,
                self.df_labels[lineage_col].astype(str),
            ))

        dist_from_root = build_dist_from_root(
            self.phylotree,
            log_every=log_every,
        )
        node_to_id, pickle_tree = build_stem_pickle_tree(self.phylotree)
        seed_node = self.phylotree.seed_node
        n_leaves_total = len(self.phylotree.leaf_nodes())
        logging.info(
            "by_nearest_stem: tree index built in %.1fs (leaves=%d, nodes=%d)",
            time.time() - start,
            n_leaves_total,
            len(pickle_tree.children) + len(pickle_tree.leaf_gids),
        )

        t_pre = time.time()
        gid_to_d_root, focal_side_refs = precompute_stem_side_ref_index(
            self.phylotree,
            dist_from_root,
            node_to_id,
            seed_node=seed_node,
            log_every=log_every,
        )
        focal_gids = list(gid_to_d_root.keys())
        logging.info(
            "by_nearest_stem: stem side-ref index phase finished in %.1fs",
            time.time() - t_pre,
        )

        t_lineage = time.time()
        lineage_to_focal_ancestors: Dict[str, frozenset] = {}
        gid_to_lineage_normalized: Dict[str, str] = {}
        if lineage_filter_on and lineage_match_effective == "pangonet":
            unique_lineages = set(gid_to_lineage_map.values())
            logging.info(
                "by_nearest_stem: caching pangonet ancestry for %d unique lineages ...",
                len(unique_lineages),
            )
            for lin in unique_lineages:
                lineage_to_focal_ancestors[lin] = focal_ancestor_normalized_set_cached(
                    lin,
                    match_mode_resolved="pangonet",
                    pangonet_obj=pangonet_for_nearest,
                    alias_key=alias_key,
                    memo_by_focal_key=ancestry_memo,
                )
            for gid, lin in gid_to_lineage_map.items():
                gid_to_lineage_normalized[gid] = normalize_lineage_string(
                    lin,
                    pango=pangonet_for_nearest,
                    alias_key=alias_key,
                )
            logging.info(
                "by_nearest_stem: pangonet lineage cache done in %.1fs (%d lineages)",
                time.time() - t_lineage,
                len(lineage_to_focal_ancestors),
            )

        logging.info(
            "by_nearest_stem: assembling search context (%d metadata dates, "
            "%d seq_lookup entries) ...",
            len(gid_to_date_ns),
            len(self.seq_lookup),
        )

        ctx = StemSearchContext(
            tree=pickle_tree,
            focal_side_refs=focal_side_refs,
            gid_to_d_root=gid_to_d_root,
            gid_to_date_ns=gid_to_date_ns,
            seq_lookup=dict(self.seq_lookup),
            max_patristic_distance=max_patristic_distance,
            lineage_filter_on=lineage_filter_on,
            lineage_match=lineage_match_effective,
            lineage_min_shared_depth=lineage_min_shared_depth,
            gid_to_lineage=gid_to_lineage_map,
            lineage_to_focal_ancestors=lineage_to_focal_ancestors,
            gid_to_lineage_normalized=gid_to_lineage_normalized,
            alias_key=alias_key,
            recombinant_parents=recomb_parents,
        )

        t_search = time.time()
        search_result = run_stem_predecessor_search(
            focal_gids,
            ctx,
            n_jobs=n_jobs,
            log_every=log_every,
        )
        self.candidate_predecessors = search_result.links

        elapsed = time.time() - start
        st = search_result.stats
        logging.info(
            "by_nearest_stem done in %.1fs (search %.1fs): %d links | "
            "no-date=%d, no-candidate=%d, skipped-identical-seq=%d, "
            "skipped-incompatible-lineage=%d, skipped-over-max-dist=%d",
            elapsed,
            time.time() - t_search,
            len(self.candidate_predecessors),
            st.no_date, st.no_candidate, st.skipped_identical,
            st.skipped_lineage, st.skipped_max_dist,
        )
        return self.candidate_predecessors

    def by_pango_anchors(
            self,
            lineage_col: str = "lineage",
            lineage_aliases_path: Optional[str] = None,
            max_hamming_aa_per_edge: Optional[int] = 12,
            max_patristic_per_edge: Optional[float] = None,
            recomb_pick: str = "lex_min",
            interpolate_depth: int = 48,
            *,
            gid_filter_only_seq_lookup: bool = True,
            log_every: int = 5000,
    ):
        """PANGO milestone anchors + deterministic sequence/patristic interpolation.

        For each genome with lineage + sequence, build the canonical anchored
        path root→leaf (see ``anchored_genome_path_for_focal``), merge edges into
        a global ``{{child : parent}}`` map. Deterministic lineage canonical
        representatives (earliest-dated genome per milestone) plus midpoint
        bisection yields stable predecessors across leaves.

        ``pangonet`` is imported when present; otherwise dotted ``alias_key``
        prefix ancestry backs off from full DAG awareness.

        Parameters
        ----------
        lineage_col
            Column in ``df_labels`` holding Pango-like lineage strings.
        lineage_aliases_path
            Path to ``alias_key.json`` (recommended whenever labels use aliases).
        max_hamming_aa_per_edge
            Insert intermediate genomes so adjacent Spike Hamming (gap-aware)
            gaps stay ≤ this cap. ``None`` disables sequence-level splitting.
        max_patristic_per_edge
            When a phylogeny is attached to the data manager (``phylotree``),
            also requires both split segments to satisfy this patristic ceiling.
            ``None`` skips patristic caps.
        recomb_pick
            ``first`` picks the catalogued first listed parent lineage;
            ``lex_min`` resolves ties lexicographically.
        interpolate_depth
            Recursion safeguard for midpoint insertion.
        gid_filter_only_seq_lookup
            If True, consider only genome ids appearing in ``seq_lookup``.
        log_every
            Progress heartbeat when iterating genomes.

        """

        logging.info(
            "by_pango_anchors: lineage_col=%s, max_aa=%s, max_pat=%s, "
            "seq_lookup=%d rows, recombinant_policy=%s",
            lineage_col,
            max_hamming_aa_per_edge,
            max_patristic_per_edge,
            len(self.seq_lookup),
            recomb_pick,
        )
        if lineage_col not in self.df_labels.columns:
            raise ValueError(f"{lineage_col!r} not in df_labels columns.")

        alias_key_simple, _recomb_maybe = maybe_load_aliases(lineage_aliases_path)
        pango = None
        try:
            from pangonet.pangonet import PangoNet  # type: ignore[import-untyped]

            pango = PangoNet().build()
            logging.info("by_pango_anchors: pangonet lineage graph loaded.")
        except Exception as exc:  # ImportError, or a pangonet data/build error (e.g. KeyError)
            logging.warning(
                "pangonet unavailable (%r) — falling back to dotted alias ancestry only. "
                "Install pangonet for nomenclature-accurate DAG parents.", exc,
            )
        gid_filter: Optional[set] = None
        if gid_filter_only_seq_lookup and self.seq_lookup:
            gid_filter = set(self.seq_lookup.keys())

        lineage_buckets, gid_by_date_full = build_lineage_date_buckets(
            self.df_labels,
            genome_col="genome_id",
            lineage_col=lineage_col,
            timestamp_field=self.timestamp_field,
            gid_filter=gid_filter,
            pango=pango,
            alias_key=alias_key_simple,
        )

        gid_to_lin = dict(
            zip(
                self.df_labels["genome_id"].astype(str),
                self.df_labels[lineage_col].astype(str),
            )
        )

        pool_sorted = chronological_pool_sorted(gid_by_date_full)

        pat_fn: Optional[Callable[[str, str], Optional[float]]] = None
        if self.phylotree is not None and max_patristic_per_edge is not None:
            try:
                pat_fn = PatristicIndex(self.phylotree).__call__
            except Exception as e:
                logging.warning(
                    "Could not build patristic index from phylotree (%s); "
                    "Hamming interpolation only.",
                    e,
                )
        elif max_patristic_per_edge is not None and self.phylotree is None:
            logging.warning(
                "max_patristic_per_edge set but no phylotree on DataManager; "
                "ignoring patristic caps.",
            )
            max_patristic_per_edge = None

        if pat_fn is None:
            max_patristic_per_edge = None

        gids_series: List[str] = sorted(gid_by_date_full.keys())
        if log_every <= 0:
            log_every = 10 ** 18

        start = time.time()
        focal_paths: Dict[str, List[str]] = {}
        skipped_no_lin = 0
        skipped_no_seq = 0
        skipped_trivial_path = 0

        idx = 0
        for gid in gids_series:
            idx += 1
            if idx % log_every == 0:
                logging.info(
                    "by_pango_anchors: genome %d / %d (paths=%d)...",
                    idx,
                    len(gids_series),
                    len(focal_paths),
                )

            raw_lin = gid_to_lin.get(str(gid), "")
            if not raw_lin or str(raw_lin).strip().lower() in {
                    "", "n/a", "na", "nan", "none", "unassigned", "unknown", "?",
            }:
                skipped_no_lin += 1
                continue

            if self.seq_lookup and str(gid) not in self.seq_lookup:
                skipped_no_seq += 1
                continue

            raw_lin_clean = raw_lin.strip()
            path_try = anchored_genome_path_for_focal(
                str(gid),
                raw_lin_clean,
                pango=pango,
                alias_key=alias_key_simple,
                lineage_buckets=lineage_buckets,
                seq_lookup=self.seq_lookup if self.seq_lookup else {},
                gid_to_date=gid_by_date_full,
                chronological_pool_sorted_arg=pool_sorted,
                max_h_edge=max_hamming_aa_per_edge,
                max_pat_edge=max_patristic_per_edge,
                pat_fn=pat_fn,
                recomb_pick=recomb_pick,
                interpolate_depth=interpolate_depth,
            )

            if len(path_try) < 2:
                skipped_trivial_path += 1
                continue

            focal_paths[str(gid)] = path_try

        self.candidate_predecessors = accumulate_predecessor_edges(focal_paths)
        elapsed = time.time() - start
        logging.info(
            "by_pango_anchors finished in %.1fs: predecessors=%d, paths=%d, "
            "skipped(no-lin=%d,no-seq=%d,trivial-path=%d)",
            elapsed,
            len(self.candidate_predecessors),
            len(focal_paths),
            skipped_no_lin,
            skipped_no_seq,
            skipped_trivial_path,
        )
        return self.candidate_predecessors

    @staticmethod
    def _compute_pseudotime_from_velocity_graph(adata, n_dcs=5):
        """Compute diffusion pseudotime from the velocity graph.

        Uses shortest-path distances through the velocity graph as a
        lightweight approximation when the full evolocity pseudotime
        pipeline is not available.

        The root node is chosen as the observation with the strongest
        net outgoing velocity flow (sum of outgoing minus incoming scores).
        """
        from scipy.sparse.csgraph import shortest_path as sp_shortest_path
        from scipy.sparse import csr_matrix

        vgraph = adata.uns['velocity_graph']
        n = vgraph.shape[0]

        vg = csr_matrix(vgraph.copy())
        vg.data = np.where(vg.data > 0, vg.data, 0)
        vg.eliminate_zeros()

        dist = vg.copy()
        dist.data = 1.0 / dist.data

        out_flow = np.array(vg.sum(axis=1)).ravel()
        in_flow = np.array(vg.sum(axis=0)).ravel()
        net_flow = out_flow - in_flow
        root_idx = int(np.argmax(net_flow))

        logging.info(f"Pseudotime root index: {root_idx} (net_flow={net_flow[root_idx]:.3f})")

        dists = sp_shortest_path(dist, directed=True, indices=root_idx)
        finite_mask = np.isfinite(dists)
        if not np.any(finite_mask):
            raise ValueError("No reachable nodes from root in velocity graph.")
        max_finite = dists[finite_mask].max()
        dists[~finite_mask] = max_finite

        pt_min, pt_max = dists.min(), dists.max()
        if pt_max - pt_min > 0:
            pseudotime = (dists - pt_min) / (pt_max - pt_min)
        else:
            pseudotime = np.zeros(n)

        return pseudotime

    def by_pseudotime(self, adata_path, pseudotime_key="velocity_pseudotime"):
        """Select predecessors using the velocity graph filtered by pseudotime.

        For each genome j the predecessor is the velocity-graph neighbour i
        with the highest pseudotime among those whose pseudotime is strictly
        less than that of j (i.e. the closest earlier neighbour in pseudotime).

        If ``pseudotime_key`` is not found in ``adata.obs``, pseudotime is
        computed on the fly from the velocity graph using shortest-path
        distances from an automatically selected root node.

        Parameters
        ----------
        adata_path : str
            Path to an AnnData ``.h5ad`` file.
        pseudotime_key : str, optional
            Column in ``adata.obs`` that stores pseudotime values.
        """
        try:
            import anndata
        except ImportError:
            raise ImportError("anndata is required for pseudotime-based predecessors. "
                              "Install it with: pip install anndata")

        logging.info(f"Loading AnnData from '{adata_path}' ...")
        adata = anndata.read_h5ad(adata_path)

        if 'velocity_graph' not in adata.uns:
            raise ValueError("AnnData does not contain 'velocity_graph' in .uns. "
                             "Run evolocity velocity_graph first.")

        if pseudotime_key in adata.obs.columns:
            logging.info(f"Using precomputed pseudotime from adata.obs['{pseudotime_key}'].")
            pseudotimes = adata.obs[pseudotime_key].values.astype(float)
        else:
            logging.info(
                f"'{pseudotime_key}' not found in adata.obs. "
                f"Computing pseudotime from velocity graph ..."
            )
            pseudotimes = self._compute_pseudotime_from_velocity_graph(adata)

        vgraph = adata.uns['velocity_graph']
        genome_ids = adata.obs['genome_id'].values
        n_obs = len(genome_ids)

        logging.info(f"Velocity graph shape: {vgraph.shape}, genome_ids: {n_obs}")

        vgraph_csc = vgraph.tocsc()

        start = time.time()
        self.candidate_predecessors = {}
        n_skipped_no_pt = 0
        n_skipped_no_pred = 0
        n_skipped_identical = 0

        for j in range(n_obs):
            cur_pt = pseudotimes[j]
            if np.isnan(cur_pt):
                n_skipped_no_pt += 1
                continue

            col = vgraph_csc.getcol(j)
            if col.nnz == 0:
                n_skipped_no_pred += 1
                continue

            row_indices = col.indices
            cur_gid = genome_ids[j]

            best_pt = -np.inf
            best_idx = -1

            for row_i in row_indices:
                if row_i == j:
                    continue
                pred_pt = pseudotimes[row_i]
                if np.isnan(pred_pt) or pred_pt >= cur_pt:
                    continue
                if self._is_identical_seq(genome_ids[row_i], cur_gid):
                    n_skipped_identical += 1
                    continue
                if pred_pt > best_pt:
                    best_pt = pred_pt
                    best_idx = row_i

            if best_idx >= 0:
                self.candidate_predecessors[cur_gid] = genome_ids[best_idx]
            else:
                n_skipped_no_pred += 1

        elapsed = time.time() - start
        logging.info(
            f"Pseudotime predecessors computed in {elapsed:.1f}s: "
            f"{len(self.candidate_predecessors)} links, "
            f"{n_skipped_no_pt} skipped (no pseudotime), "
            f"{n_skipped_no_pred} skipped (no valid predecessor), "
            f"{n_skipped_identical} skipped (identical seq)"
        )

        return self.candidate_predecessors

    def by_evolocities(self, adata_path, min_score=0.0):
        """Extract predecessor links from a precomputed evolocity velocity graph.

        For each genome j the predecessor is the genome i with the highest
        positive velocity score toward j, subject to temporal ordering
        (predecessor must be earlier in time).

        Parameters
        ----------
        adata_path : str
            Path to an AnnData ``.h5ad`` file containing
            ``adata.uns['velocity_graph']`` (CSR sparse matrix) and
            ``adata.obs['genome_id']``.
        min_score : float, optional
            Minimum velocity score to accept a predecessor link.
        """
        try:
            import anndata
        except ImportError:
            raise ImportError("anndata is required for evolocity-based predecessors. "
                              "Install it with: pip install anndata")

        logging.info(f"Loading AnnData from '{adata_path}' ...")
        adata = anndata.read_h5ad(adata_path)

        if 'velocity_graph' not in adata.uns:
            raise ValueError("AnnData does not contain 'velocity_graph' in .uns. "
                             "Run evolocity velocity_graph first.")

        vgraph = adata.uns['velocity_graph']  # CSR (n_obs, n_obs)
        genome_ids = adata.obs['genome_id'].values
        n_obs = len(genome_ids)

        logging.info(f"Velocity graph shape: {vgraph.shape}, "
                     f"genome_ids: {n_obs}")

        # Resolve collection dates: prefer adata.obs, fall back to df_labels
        if 'collection_date' in adata.obs.columns:
            dates = pd.to_datetime(adata.obs['collection_date'], errors='coerce')
        else:
            logging.info("collection_date not in adata.obs, joining from df_labels.")
            gid_to_date = self.df_labels.set_index('genome_id')[self.timestamp_field]
            dates = pd.to_datetime(
                pd.Series(genome_ids).map(gid_to_date).values,
                errors='coerce'
            )

        # Convert to CSC for efficient column access (we need argmax per column)
        vgraph_csc = vgraph.tocsc()

        start = time.time()
        self.candidate_predecessors = {}
        n_skipped_no_date = 0
        n_skipped_no_pred = 0
        n_skipped_identical = 0

        for j in range(n_obs):
            cur_date = dates.iloc[j] if hasattr(dates, 'iloc') else dates[j]
            if pd.isna(cur_date):
                n_skipped_no_date += 1
                continue

            col = vgraph_csc.getcol(j)
            if col.nnz == 0:
                n_skipped_no_pred += 1
                continue

            row_indices = col.indices
            scores = col.data
            cur_gid = genome_ids[j]

            best_score = -1.0
            best_idx = -1

            for k, row_i in enumerate(row_indices):
                if scores[k] <= min_score:
                    continue
                if row_i == j:
                    continue
                pred_date = dates.iloc[row_i] if hasattr(dates, 'iloc') else dates[row_i]
                if pd.isna(pred_date) or pred_date >= cur_date:
                    continue
                if self._is_identical_seq(genome_ids[row_i], cur_gid):
                    n_skipped_identical += 1
                    continue
                if scores[k] > best_score:
                    best_score = scores[k]
                    best_idx = row_i

            if best_idx >= 0:
                self.candidate_predecessors[cur_gid] = genome_ids[best_idx]
            else:
                n_skipped_no_pred += 1

        elapsed = time.time() - start
        logging.info(
            f"Evolocity predecessors computed in {elapsed:.1f}s: "
            f"{len(self.candidate_predecessors)} links, "
            f"{n_skipped_no_date} skipped (no date), "
            f"{n_skipped_no_pred} skipped (no valid predecessor), "
            f"{n_skipped_identical} skipped (identical seq)"
        )

        return self.candidate_predecessors

    def audit_cycles(self, max_walk: int = 256) -> dict:
        """Detect cycles in ``candidate_predecessors`` and report a summary.

        For each leaf ``L`` we walk ``pred[L], pred[pred[L]], ...`` keeping a
        per-walk ``visited`` set; a cycle is recorded when the walk returns
        to a node that already appears in the current walk. Self-loops
        (``pred[L] == L``) and 2-cycles (``pred[A] == B`` and
        ``pred[B] == A``) are counted separately because they are the
        common failure modes of date-blind predecessor methods.

        Walks are capped at ``max_walk`` hops to keep the audit O(N * max_walk)
        even for pathologically tangled maps.

        Returns
        -------
        dict
            Keys: ``n_links``, ``n_self_loops``, ``n_two_cycles``,
            ``n_long_cycles``, ``leaves_in_cycle``, ``examples`` (up to 5
            short cycles for diagnostics).
        """
        pred = self.candidate_predecessors or {}
        n_links = len(pred)
        n_self_loops = 0
        n_two_cycles = 0
        n_long_cycles = 0
        leaves_in_cycle = 0
        examples: list[list[str]] = []

        for leaf in pred:
            if pred.get(leaf) == leaf:
                n_self_loops += 1
                leaves_in_cycle += 1
                if len(examples) < 5:
                    examples.append([leaf, leaf])
                continue
            cur = leaf
            visited: dict[str, int] = {cur: 0}
            for hop in range(1, max_walk + 1):
                nxt = pred.get(cur)
                if nxt is None:
                    break
                if nxt in visited:
                    cycle_len = hop - visited[nxt]
                    if cycle_len == 2:
                        n_two_cycles += 1
                    else:
                        n_long_cycles += 1
                    leaves_in_cycle += 1
                    if len(examples) < 5:
                        # Reconstruct the smallest illustrative cycle path.
                        path = [n for n, h in sorted(visited.items(), key=lambda kv: kv[1])
                                if h >= visited[nxt]]
                        path.append(nxt)
                        examples.append(path)
                    break
                visited[nxt] = hop
                cur = nxt

        report = {
            "n_links": n_links,
            "n_self_loops": n_self_loops,
            "n_two_cycles": n_two_cycles,
            "n_long_cycles": n_long_cycles,
            "leaves_in_cycle": leaves_in_cycle,
            "examples": examples,
        }
        log = logging.warning if leaves_in_cycle > 0 else logging.info
        log(
            "Predecessor cycle audit: %d links, %d self-loops, %d 2-cycles, "
            "%d longer cycles; %d leaves touch a cycle (sample paths: %s).",
            n_links, n_self_loops, n_two_cycles, n_long_cycles,
            leaves_in_cycle, examples,
        )
        return report

    def write_predecessors_dict(self, filepath):
        """Saves the predecessor map to JSON, after auditing for cycles.

        Cycles in this map produce alternating ``[A, B, A, B, ...]`` series
        downstream (see :meth:`SeriesFactory.get_k_series`'s visited-set
        guard), so we run :meth:`audit_cycles` before writing and surface
        the counts in the log. The file is still written either way; the
        downstream visited-set guard will drop the affected series.

        When :attr:`predecessors_metadata` is set (e.g. by
        :meth:`by_nearest_ancestor` with sampling options), the file uses
        v2 format with ``_meta`` + ``links``; otherwise a legacy flat map.
        """
        self.audit_cycles()
        write_predecessors_json(
            filepath,
            self.candidate_predecessors,
            meta=self.predecessors_metadata,
        )

    def load_predecessors_dict(self, filepath):
        """Loads predecessor links from legacy or v2 JSON."""
        meta, links = load_predecessors_json(filepath)
        self.candidate_predecessors = links
        self.predecessors_metadata = meta

