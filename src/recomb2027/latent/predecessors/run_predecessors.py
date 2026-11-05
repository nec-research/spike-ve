#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     run_predecessors.py
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
"""
Reconstruct predecessor links between genome sequences.

Usage
-----
    python -m recomb2027.latent.predecessors.run_predecessors --cfg <path-to-yaml>

Supported methods (set ``predecessor_method`` in the config):
    by_evolocity   – evolocity velocity graph (highest velocity score, collection-date ordering)
    by_pseudotime  – evolocity velocity graph (closest earlier neighbour in pseudotime)
    by_timestep    – distance-matrix nearest neighbour with minimum time gap
    by_timeslice   – distance-matrix nearest neighbour across discrete time slices
    by_phylotree   – phylogenetic-tree traversal (legacy node-representative heuristic)
    by_nearestleaf – nearest-earlier-leaf inside smallest enclosing subtree
                     with PANGO lineage match modes / ascent-exclusive pruning.
    by_nearest_stem – nearest-earlier-leaf on side branches along focal->root
                      stem (excludes parallel-clade cousins used by nearestleaf).
    by_nearest_ancestor – nearest earlier leaf in nested ancestor clade only
                      (stem-child subtrees toward root; no parallel cousins).
    by_pango_anchors – lineage milestones via pangonet (or dotted prefixes) plus
                       Hamming- and optionally patristic-capped midpoint interpolation
"""

import argparse
import json
import sys
from pathlib import Path
from recomb2027.core.paths import repo_root
from typing import Any, Dict, List, Optional

REPO_ROOT = repo_root(__file__)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omegaconf import OmegaConf
from recomb2027.config_io import load_config

from recomb2027.latent.data_series.predecessor_finder import PredecessorFinder
from recomb2027.latent.data_series.slicer import DatasetSlicer
from recomb2027.latent.io_utils import setup_logging
from recomb2027.data import DataManager

logger = setup_logging()

VALID_METHODS = {
    "by_evolocity", "by_pseudotime", "by_timestep", "by_timeslice",
    "by_phylotree", "by_nearestleaf", "by_nearest_stem", "by_nearest_ancestor",
    "by_pango_anchors",
}

#DEFAULT_CONFIG_PATH = "configs/latent/predecessors/predecessors.yaml"
DEFAULT_CONFIG_PATH = "configs/latent/predecessors/predecessors_nearestleaf.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="Reconstruct predecessor links.")
    parser.add_argument(
        "--cfg",
        type=str,
        required=False,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the predecessors YAML config file.",
    )
    return parser.parse_args()


def _load_data_manager(cfg, need_tree: bool) -> DataManager:
    """Initialise DataManager with MSA, metadata, and optionally a phylotree.

    Loads ``collection_date`` plus any extra metadata columns referenced by
    method-specific config (typically ``lineage_col`` for ``by_nearestleaf``,
    ``by_nearest_stem``, or ``by_pango_anchors``).

    Column collision handling
    ------------------------
    When both the FASTA headers and the external metadata CSV contain the same
    column (e.g. ``collection_date``), the config key
    ``metadata_collision_source`` controls which values win:

    * ``"fasta"``  – keep the values parsed from the FASTA headers;
                     drop the column from the CSV before merging.
    * ``"csv"``    – keep the values from the external CSV (default);
                     drop the column from df_labels before merging.
    """
    import pandas as pd

    collision_source: str = getattr(cfg, "metadata_collision_source", "fasta")
    if collision_source not in ("fasta", "csv"):
        raise ValueError(
            f"metadata_collision_source must be 'fasta' or 'csv', "
            f"got {collision_source!r}"
        )

    dmg = DataManager()
    dmg.load_msa_from_file(cfg.msa_path)

    cols_to_extract = ["collection_date"]
    lineage_col = getattr(cfg, "lineage_col", None)
    if lineage_col and lineage_col not in cols_to_extract:
        cols_to_extract.append(lineage_col)
    # Detect columns present in both df_labels (from FASTA) and the external CSV.
    # pandas would otherwise suffix both copies with _x/_y, making neither
    # accessible under the plain column name.
    try:
        csv_header = pd.read_csv(cfg.metadata_msa_path, dtype=str, nrows=0).columns.tolist()
    except Exception:
        csv_header = []
    overlap = [c for c in cols_to_extract if c in dmg.df_labels.columns and c in csv_header]

    if overlap:
        if collision_source == "csv":
            # Remove the FASTA-derived values so the CSV values come through.
            logger.info(
                "Column collision on %s — keeping CSV values "
                "(set metadata_collision_source: fasta to prefer FASTA headers).",
                overlap,
            )
            dmg.df_labels = dmg.df_labels.drop(columns=overlap)
        else:  # "fasta"
            # Remove the colliding columns from cols_to_extract so
            # add_metadata_from_csv will not bring them in from the CSV.
            logger.info(
                "Column collision on %s — keeping FASTA-header values "
                "(set metadata_collision_source: csv to prefer the metadata CSV).",
                overlap,
            )
            cols_to_extract = [c for c in cols_to_extract if c not in overlap]

    dmg.add_metadata_from_csv(
        cfg.metadata_msa_path,
        cols_to_extract=cols_to_extract,
    )
    if need_tree:
        dmg.add_phylogenetic_tree(
            cfg.tree_path,
            schema=getattr(cfg, "tree_schema", "newick"),
        )
    return dmg


def _ensure_distance_matrix(pred_finder, cfg, dmg):
    """Load or compute the pairwise distance matrix required by
    ``by_timestep`` and ``by_timeslice``."""
    dist_path = getattr(cfg, "distance_matrix_path", None)
    if dist_path:
        logger.info(f"Loading distance matrix from '{dist_path}' ...")
        pred_finder.distances.load(dist_path)
    else:
        if dmg.phylotree is None:
            dmg.add_phylogenetic_tree(
                cfg.tree_path,
                schema=getattr(cfg, "tree_schema", "newick"),
            )
        logger.info("Computing distance matrix from phylogenetic tree ...")
        pred_finder.distances.calc_from_phylotree(dmg.phylotree)


def _resolve_recombinant_parents(value) -> Optional[Dict[str, List[str]]]:
    """Accept either an inline mapping, a path to a JSON file, or None.

    Examples in YAML::

        # inline:
        lineage_recombinant_parents:
          XBB: [BJ.1, BM.1.1.1]
          XAY: [B.1.617.2, BA.2]

        # external JSON file:
        lineage_recombinant_parents: /path/to/recombinant_parents.json
    """
    if value is None:
        return None
    if isinstance(value, str):
        with open(value, "r") as f:
            data = json.load(f)
    else:
        data = OmegaConf.to_container(value, resolve=True)
    if not isinstance(data, dict):
        raise ValueError(
            f"lineage_recombinant_parents must be a mapping, got {type(data).__name__}"
        )
    out: Dict[str, List[str]] = {}
    for k, v in data.items():
        if not isinstance(v, (list, tuple)):
            raise ValueError(
                f"lineage_recombinant_parents['{k}'] must be a list of parent "
                f"lineage strings, got {type(v).__name__}"
            )
        out[str(k)] = [str(p) for p in v]
    return out


def _resolve_predecessor_sampling(cfg) -> Optional[Dict[str, Any]]:
    """Return ``predecessor_sampling`` object for ``by_nearest_ancestor`` (or None)."""
    raw = OmegaConf.select(cfg, "predecessor_sampling", default=None)
    if raw is None:
        selection = OmegaConf.select(cfg, "predecessor_selection", default=None)
        if selection is None:
            return None
        return {"selection": str(selection)}
    if OmegaConf.is_config(raw):
        data = OmegaConf.to_container(raw, resolve=True)
    else:
        data = raw
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError(
            f"predecessor_sampling must be a mapping, got {type(data).__name__}"
        )
    return {str(k): v for k, v in data.items()}


def _resolve_reversion_penalty(cfg) -> Optional[Dict[str, Any]]:
    """Return ``reversion_penalty`` mapping for ``by_nearest_ancestor`` (or None => legacy scoring)."""
    raw = OmegaConf.select(cfg, "reversion_penalty", default=None)
    if raw is None:
        return None
    data = OmegaConf.to_container(raw, resolve=True) if OmegaConf.is_config(raw) else raw
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError(f"reversion_penalty must be a mapping, got {type(data).__name__}")
    return {str(k): v for k, v in data.items()}


def _ensure_timeslices(pred_finder, cfg):
    """Add a ``Timeslice`` column to ``pred_finder.df_labels`` via
    ``DatasetSlicer``."""
    timeslice_step = getattr(cfg, "timeslice_step", "30 days")
    slicer = DatasetSlicer(pred_finder.df_labels, pred_finder.timestamp_field)
    pred_finder.df_labels = slicer.slice_dataset(time_step=timeslice_step)
    logger.info(
        f"Time-sliced dataset into "
        f"{pred_finder.df_labels['Timeslice'].nunique()} slices "
        f"(step={timeslice_step})."
    )


def run(cfg) -> None:
    method = cfg.predecessor_method
    if method not in VALID_METHODS:
        raise ValueError(
            f"Unknown predecessor_method '{method}'. "
            f"Choose from: {sorted(VALID_METHODS)}"
        )

    logger.info(f"Predecessor method: {method}")

    tree_path_any = OmegaConf.select(cfg, "tree_path", default=None)
    need_tree = method in (
        "by_phylotree", "by_nearestleaf", "by_nearest_stem", "by_nearest_ancestor",
        "by_timeslice", "by_timestep",
    ) or (
        method == "by_pango_anchors" and tree_path_any
    )

    dmg = _load_data_manager(cfg, need_tree=need_tree)
    pred_finder = PredecessorFinder(dmg)

    if method == "by_evolocity":
        pred_finder.by_evolocities(
            adata_path=cfg.adata_path,
            min_score=getattr(cfg, "evolocity_min_score", 0.0),
        )

    elif method == "by_pseudotime":
        pred_finder.by_pseudotime(
            adata_path=cfg.adata_path,
            pseudotime_key=getattr(cfg, "pseudotime_key", "velocity_pseudotime"),
        )

    elif method == "by_timestep":
        _ensure_distance_matrix(pred_finder, cfg, dmg)
        time_step = getattr(cfg, "time_step", "10 days")
        pred_finder.by_timestep(time_step=time_step)

    elif method == "by_timeslice":
        _ensure_distance_matrix(pred_finder, cfg, dmg)
        _ensure_timeslices(pred_finder, cfg)
        final_ts = getattr(cfg, "final_timeslice", None)
        pred_finder.by_timeslice(final_timeslice=final_ts)

    elif method == "by_phylotree":
        _ensure_timeslices(pred_finder, cfg)
        pred_finder.by_phylotree(
            closest_by_distance=getattr(cfg, "closest_by_distance", False),
        )

    elif method == "by_nearestleaf":
        recomb_cfg = getattr(cfg, "lineage_recombinant_parents", None)
        recomb_parents = _resolve_recombinant_parents(recomb_cfg)
        pred_finder.by_nearestleaf(
            max_patristic_distance=getattr(cfg, "max_patristic_distance", None),
            lineage_col=getattr(cfg, "lineage_col", None),
            lineage_min_shared_depth=getattr(cfg, "lineage_min_shared_depth", 2),
            lineage_aliases_path=getattr(cfg, "lineage_aliases_path", None),
            lineage_recombinant_parents=recomb_parents,
            distance_fallback_factor=getattr(cfg, "distance_fallback_factor", None),
            log_every=getattr(cfg, "log_every", 1000),
            lineage_match=str(getattr(cfg, "lineage_match", "compat")),
            lineage_ascent_exclusive=bool(getattr(cfg, "lineage_ascent_exclusive", False)),
        )

    elif method == "by_nearest_stem":
        recomb_cfg = getattr(cfg, "lineage_recombinant_parents", None)
        recomb_parents = _resolve_recombinant_parents(recomb_cfg)
        pred_finder.by_nearest_stem(
            max_patristic_distance=getattr(cfg, "max_patristic_distance", None),
            lineage_col=getattr(cfg, "lineage_col", None),
            lineage_min_shared_depth=getattr(cfg, "lineage_min_shared_depth", 2),
            lineage_aliases_path=getattr(cfg, "lineage_aliases_path", None),
            lineage_recombinant_parents=recomb_parents,
            log_every=getattr(cfg, "log_every", 1000),
            lineage_match=str(getattr(cfg, "lineage_match", "pangonet")),
            n_jobs=int(getattr(cfg, "n_jobs", 1)),
        )

    elif method == "by_nearest_ancestor":
        recomb_cfg = getattr(cfg, "lineage_recombinant_parents", None)
        recomb_parents = _resolve_recombinant_parents(recomb_cfg)
        pred_finder.by_nearest_ancestor(
            max_patristic_distance=getattr(cfg, "max_patristic_distance", None),
            min_patristic_distance=getattr(cfg, "min_patristic_distance", None),
            predecessor_rank=int(getattr(cfg, "predecessor_rank", 1)),
            lineage_col=getattr(cfg, "lineage_col", None),
            lineage_min_shared_depth=getattr(cfg, "lineage_min_shared_depth", 2),
            lineage_aliases_path=getattr(cfg, "lineage_aliases_path", None),
            lineage_recombinant_parents=recomb_parents,
            log_every=getattr(cfg, "log_every", 1000),
            lineage_match=str(getattr(cfg, "lineage_match", "ancestor")),
            n_jobs=int(getattr(cfg, "n_jobs", 1)),
            predecessor_sampling=_resolve_predecessor_sampling(cfg),
            reversion_penalty=_resolve_reversion_penalty(cfg),
        )

    elif method == "by_pango_anchors":
        mpc_ev = OmegaConf.select(cfg, "max_patristic_per_edge", default=None)
        if mpc_ev is None:
            mpc: Optional[float] = None
        else:
            try:
                mpc = float(mpc_ev)
            except (TypeError, ValueError):
                mpc = None

        mha_raw = OmegaConf.select(cfg, "max_hamming_aa_per_edge", default=12)
        if mha_raw is None:
            mh_max: Optional[int] = None
        else:
            try:
                mh_max = int(mha_raw)
            except (TypeError, ValueError):
                mh_max = 12

        pred_finder.by_pango_anchors(
            lineage_col=str(getattr(cfg, "lineage_col", "lineage")),
            lineage_aliases_path=getattr(cfg, "lineage_aliases_path", None),
            max_hamming_aa_per_edge=mh_max,
            max_patristic_per_edge=mpc,
            recomb_pick=str(getattr(cfg, "pango_recomb_pick", "lex_min")),
            interpolate_depth=int(getattr(cfg, "interpolate_depth", 48)),
            gid_filter_only_seq_lookup=bool(getattr(cfg, "pango_gid_filter_seq_lookup_only", True)),
            log_every=int(getattr(cfg, "log_every", 5000)),
        )

    n_links = len(pred_finder.candidate_predecessors)
    logger.info(f"Predecessor links found: {n_links}")

    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_finder.write_predecessors_dict(str(output_path))
    logger.info(f"Predecessors saved to '{output_path}'")


def main():
    args = parse_args()

    cfg = load_config(args.cfg)
    run(cfg)


if __name__ == "__main__":
    main()
