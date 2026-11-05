#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     series_factory.py
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
 Module to create and maintain k-series.
"""

import os
import pickle
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd
from tqdm import tqdm

from .slicer import DatasetSlicer
from .encodings import EncodingMatcher

import logging
logger = logging.getLogger(__name__)


def resolve_hop_strides(
    k: int,
    stride: Optional[int] = 1,
    hop_strides: Optional[Sequence[int]] = None,
) -> List[int]:
    """Number of predecessor links spanned by each of the ``k-1`` series hops (oldest -> newest).

    * ``hop_strides`` (explicit list of length ``k-1``) wins; entry ``i`` is the number of predecessor links
      between ``series[i]`` and ``series[i+1]``.  Example (k=5): ``[1, 1, 1, 2]`` keeps three one-link hops and
      makes the LAST hop (the prediction target) a 2-link hop.
    * otherwise ``stride`` (int, default 1) is used for every hop.

    ``stride=1`` / ``hop_strides=None`` reproduces the legacy behaviour exactly.
    """
    k = int(k)
    if k < 2:
        raise ValueError(f"series length k must be >= 2, got {k}")
    if hop_strides is not None:
        hs = [int(s) for s in hop_strides]
        if len(hs) != k - 1:
            raise ValueError(
                f"hop_strides must have k-1={k - 1} entries (oldest hop first), got {len(hs)}: {hs}"
            )
    else:
        s = 1 if stride is None else int(stride)
        hs = [s] * (k - 1)
    if any(s < 1 for s in hs):
        raise ValueError(f"all strides must be >= 1, got {hs}")
    return hs


def stride_kwargs_from_cfg(cfg: Any) -> Dict[str, Any]:
    """Read ``series_stride`` (int, default 1) and ``series_hop_strides`` (list, default null) from a config.

    Works with dicts and OmegaConf nodes; result is ``get_k_series(**kwargs)``-ready.
    """
    getter = cfg.get if hasattr(cfg, "get") else (lambda key, default=None: getattr(cfg, key, default))
    stride = getter("series_stride", 1)
    hops = getter("series_hop_strides", None)
    return {
        "stride": 1 if stride is None else int(stride),
        "hop_strides": None if hops is None else [int(s) for s in hops],
    }


def walk_back_series(
    predecessors: Mapping[str, str],
    genome_id: str,
    hop_strides: Sequence[int],
):
    """Walk ``sum(hop_strides)`` predecessor links back from ``genome_id`` and keep every stride-th node.

    Returns ``(series_oldest_first | None, status, path)`` with ``status`` in ``{"ok", "short", "cycle"}``.
    The visited-set guard covers ALL walked nodes (also the skipped intermediate ones), so a loop anywhere on the
    path discards the series.  ``path`` is the newest-first list of walked nodes (for diagnostics).
    """
    cur = genome_id
    path = [cur]
    visited = {cur}
    kept = [cur]
    for stride in reversed(list(hop_strides)):          # newest hop first
        for _ in range(int(stride)):
            if cur not in predecessors:
                return None, "short", path
            nxt = predecessors[cur]
            if nxt in visited:
                return None, "cycle", path + [nxt]
            path.append(nxt)
            visited.add(nxt)
            cur = nxt
        kept.append(cur)
    return kept[::-1], "ok", path


class SeriesFactory:
    def __init__(self, data_manager):
        self.dmg = data_manager
        self.dataset = DatasetSlicer(data_manager.df_labels, data_manager.timestamp_field)
        self.encodings = EncodingMatcher(data_manager)
        self.predecessors_dict = None  # after distance_matrix calculation
        self.predecessors_metadata: Optional[Dict[str, Any]] = None
        self.lst_k_series = None

    def load_predecessors_dict(self, filepath):
        """Loads predecessor links from legacy flat JSON or v2 ``_meta`` + ``links``."""
        from .predecessors_json import load_predecessors_json

        meta, links = load_predecessors_json(filepath)
        self.predecessors_dict = links
        self.predecessors_metadata = meta

    def get_k_series(self, k=3, stride: Optional[int] = 1, hop_strides: Optional[Sequence[int]] = None):
        """Generate length-``k`` series by walking the predecessors chain.

        **Strides.** By default every series hop is ONE predecessor link.  ``stride=s`` makes every hop span
        ``s`` links (the ``s-1`` intermediate genomes are walked but NOT part of the series), e.g. ``stride=2``
        with ``k=5`` walks 8 links and keeps ``x_{-8}, x_{-6}, x_{-4}, x_{-2}, x_0``.  ``hop_strides`` gives a
        per-hop list of length ``k-1`` (oldest hop first), e.g. ``[1, 1, 1, 2]`` = only the last (target) hop
        spans two links.  A longer hop means a larger latent step with less accumulated private noise of the
        stride-1 chain, but needs ``sum(hop_strides)`` links of history per series.

        For every leaf in ``predecessors_dict`` we walk ``sum(hop_strides)`` links
        backwards. Series are aborted early in two cases so the resulting
        list is loop-free and trustworthy for downstream training:

        * **Visited-set guard** — if walking back would re-introduce a
          ``genome_id`` already present in the current series (a 2-cycle or
          longer loop in ``predecessors_dict``), the series is discarded
          rather than producing an alternating ``[A, B, A, B, ...]`` sequence
          that would silently teach the model a non-monotone history.
        * **Length guard** — if the chain hits the global root before
          collecting ``k`` ids the series is dropped (existing behaviour).

        The number of cycle-truncated and short-chain series is logged so
        regressions in upstream predecessor methods are easy to spot.
        """
        if self.predecessors_dict is None:
            raise ValueError("Calculate predecessors first.")

        hops = resolve_hop_strides(k, stride, hop_strides)
        logging.info(
            "Reconstruct full series (k=%d, hop strides oldest->newest=%s, %d links of history per series)...",
            k, hops, sum(hops),
        )

        n_short = 0
        n_cycle = 0
        sample_cycles: list[tuple[str, ...]] = []

        self.lst_k_series = []
        for genome_id in self.predecessors_dict:
            series, status, path = walk_back_series(self.predecessors_dict, genome_id, hops)
            if status == "cycle":
                n_cycle += 1
                if len(sample_cycles) < 5:
                    sample_cycles.append(tuple(path))
                continue
            if status == "short":
                n_short += 1
                continue
            self.lst_k_series.append(series)

        logging.info(
            "Constructed %d series (k=%d, hop strides=%s). Dropped: %d short chains, "
            "%d cycle-truncated chains.",
            len(self.lst_k_series), k, hops, n_short, n_cycle,
        )
        if n_cycle > 0:
            logging.warning(
                "Predecessors map contains cycles touching %d focal leaves. "
                "First %d truncated walks: %s. Run "
                "PredecessorFinder.audit_cycles() and/or rebuild predecessors "
                "with a date-monotone method (by_nearestleaf, by_timestep, "
                "by_pseudotime).",
                n_cycle, len(sample_cycles), sample_cycles,
            )

        return self.lst_k_series

    def get_k_series_for_timeslice(
        self, k, timeslices, stride: Optional[int] = 1, hop_strides: Optional[Sequence[int]] = None,
    ):
        """Return length-``k`` series ending in the listed timeslices.

        Same visited-set and length guards (and ``stride`` / ``hop_strides``) as :meth:`get_k_series`.
        """
        if 'Timeslice' not in list(self.dataset.df_labels.columns):
            raise ValueError("Do the time slicing first.")

        if self.predecessors_dict is None:
            raise ValueError("Calculate predecessors first.")

        hops = resolve_hop_strides(k, stride, hop_strides)
        logging.info("Reconstruct full series (k=%d, hop strides=%s)...", k, hops)

        n_short = 0
        n_cycle = 0

        self.lst_k_series = []
        genome_ids = self.dataset.df_labels[
            self.dataset.df_labels['Timeslice'].isin(timeslices)]['genome_id'].tolist()
        for genome_id in genome_ids:
            series, status, _ = walk_back_series(self.predecessors_dict, genome_id, hops)
            if status == "cycle":
                n_cycle += 1
            elif status == "short":
                n_short += 1
            else:
                self.lst_k_series.append(series)

        logging.info(
            "Constructed %d series (k=%d, hop strides=%s). Dropped: %d short chains, "
            "%d cycle-truncated chains.",
            len(self.lst_k_series), k, hops, n_short, n_cycle,
        )
        return self.lst_k_series

    @staticmethod
    def scan_available_genome_ids(encodings_dir: str) -> set[str]:
        """Return genome_ids inferred from ``.pt`` filenames in *encodings_dir*."""
        _GENOME_ID_PREFIX = "genome_id_"
        available_ids: set[str] = set()
        n_files = 0
        import re
        try:
            import torch
        except Exception:
            torch = None

        with os.scandir(encodings_dir) as it:
            for entry in it:
                if not entry.name.endswith(".pt") or not entry.is_file():
                    continue
                n_files += 1
                stem = entry.name[:-3]
                found = False
                # Fast path: filename fields produced by label_to_filename usually
                # contain a "genome_id_<value>" field among '|' delimited parts.
                for field in stem.split("|"):
                    if field.startswith(_GENOME_ID_PREFIX):
                        available_ids.add(field[len(_GENOME_ID_PREFIX):])
                        found = True
                        break
                if found:
                    continue

                # Try to infer common accession patterns (e.g. EPI_ISL_12345) or any
                # token that looks like an identifier (letters + digits).
                candidate = None
                for field in stem.split("|"):
                    m = re.search(r'(EPI[_-]ISL[_-]?\d+)', field, flags=re.IGNORECASE)
                    if m:
                        candidate = m.group(1)
                        break
                    if re.search(r'[A-Za-z].*?\d', field):
                        candidate = field
                        break
                if candidate:
                    available_ids.add(candidate)
                    continue

                # Last resort: open the .pt file and read the embedded label to parse genome_id.
                if torch is not None:
                    try:
                        data = torch.load(entry.path, map_location="cpu")
                        label = data.get("label") if isinstance(data, dict) else None
                        if label and isinstance(label, str):
                            lab = label.replace("'", "")
                            # try to parse pipe-delimited fields like "genome_id:VALUE"
                            for p in lab.split("|"):
                                if p.lower().startswith("genome_id"):
                                    if ":" in p:
                                        val = p.split(":", 1)[1].strip()
                                    elif "_" in p:
                                        val = p.split("_", 1)[1].strip()
                                    else:
                                        val = p
                                    if val:
                                        available_ids.add(val)
                                        found = True
                                        break
                            if found:
                                continue
                            # regex fallback in the full label text
                            m2 = re.search(r'(EPI[_-]ISL[_-]?\d+)', lab, flags=re.IGNORECASE)
                            if m2:
                                available_ids.add(m2.group(1))
                                continue
                    except Exception:
                        # ignore files we can't load
                        pass
        logger.info(
            "Found %d unique genome_ids across %d embedding files in '%s'.",
            len(available_ids),
            n_files,
            encodings_dir,
        )
        return available_ids

    @staticmethod
    def filter_series_list_by_encodings(
        series_list: List[List[str]],
        encodings_dir: str,
        *,
        available_ids: Optional[set[str]] = None,
    ) -> List[List[str]]:
        """Drop series with any genome_id missing from *encodings_dir*."""
        if available_ids is None:
            available_ids = SeriesFactory.scan_available_genome_ids(encodings_dir)
        before = len(series_list)
        filtered = [
            series
            for series in series_list
            if all(gid in available_ids for gid in series)
        ]
        logger.info(
            "Dropped %d series with missing embeddings. %d series remaining.",
            before - len(filtered),
            len(filtered),
        )
        return filtered

    def filter_k_series_by_encodings(self, encodings_dir: str) -> None:
        """Filter out series for which at least one genome_id lacks an embedding.

        Availability is determined by scanning the *file names* of ``.pt``
        files in *encodings_dir* — no embedding tensors are loaded into
        memory.

        Naming convention (produced by ``encode_proteins_from_fasta.label_to_filename``):
        each file is saved as ``<sanitised_label>.pt`` where the FASTA field
        separator ``'|'`` is preserved and ``':'`` (the key/value separator)
        is replaced by ``'_'``.  A label such as::

            genome_id:EPI_ISL_123456|protein:spike|...

        therefore yields the file name::

            genome_id_EPI_ISL_123456|protein_spike|....pt

        The genome_id is extracted as the value after the ``genome_id_``
        prefix in the first matching pipe-delimited field of the stem.

        Note: files whose labels carry multiple comma-separated genome_ids
        (e.g. ``genome_id:A,B``) cannot be unambiguously decomposed from
        the file name alone (``','`` is also replaced by ``'_'``).  Such
        files are skipped; any series that relies exclusively on those
        genome_ids will be conservatively dropped.
        """
        self.lst_k_series = self.filter_series_list_by_encodings(
            self.lst_k_series,
            encodings_dir,
        )

    def filter_k_series_by_last_element(self, inplace=False):
        """ Make sure that the last element of series is not seen in any other series.

        Returns
        -------

        """
        logger.info(
            f"Filter out series, where last element seen in other series."
            f"Analyse {len(self.lst_k_series)} series."
        )
        # create search terms list
        search_terms = [series[-1] for series in self.lst_k_series]
        logger.info(f"Number of search terms: {len(search_terms)}")

        term_dict = Counter()
        series_idx = 0
        for series in tqdm(self.lst_k_series, desc="Processing series"):
            # all preceding positions (incl. first element) excluding the last
            unique_terms = set(series[:-1]) ## set(series[1:-1])
            # only keep those that are in search_terms
            for term in unique_terms.intersection(search_terms):
                term_dict[term] += 1
            series_idx += 1

        term_dict = dict(term_dict)

        df_terms = pd.DataFrame.from_dict(term_dict, orient='index', columns=['value']).reset_index()
        df_terms.columns = ['term', 'value']
        terms_to_excl = df_terms['term'].values     # terms to exclude

        filtered_kseries_lst = [series for series in self.lst_k_series if series[-1] not in terms_to_excl]
        logger.info(f"Number of series left: {len(filtered_kseries_lst)}")

        if inplace:
            self.lst_k_series = filtered_kseries_lst

        return filtered_kseries_lst

    def write_series_dict(self, series_dict, filepath):
        with open(filepath, "w") as f:
            for key, values in series_dict.items():
                f.write(f"{key}:")
                for val in values:
                    f.write(f" \t{val}")
                f.write(f"\n")

        # Save into files
    def write_series(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.lst_k_series, f)
