#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     substitution_dataset.py
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
"""Dataset wrapper attaching B1 substitution features to pca2d series items."""
from __future__ import annotations

from typing import Any, Optional

from torch.utils.data import ConcatDataset, Dataset

from recomb2027.latent.data_series.substitution_sidecar_store import (
    SubstitutionSidecarStore,
    resolve_substitution_sidecar_path,
)


class SubstitutionFeatureDataset(Dataset):
    """Append prefix substitution features (and optional target masks) per sample.

    Must wrap *outside* edge-phylo / weight wrappers so collate can peel
    substitution tensors from the item tail, and so leaf discovery still
    reaches the underlying series dataset.
    """

    def __init__(
        self,
        base: Dataset,
        store: SubstitutionSidecarStore,
        *,
        return_target_activation: bool = False,
        sample_offset: int = 0,
    ):
        self.base = base
        self.store = store
        self.return_target_activation = bool(return_target_activation)
        self.sample_offset = int(sample_offset)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Any:
        item = self.base[idx]
        si = self.sample_offset + int(idx)
        n_steps = self.store.n_transition_steps(si)
        # Need one prefix feature per supervised transition (y-step).
        n_feat = max(int(n_steps), 1)
        active = self._active_timeline_indices() if n_steps > 0 else None
        if active is not None:
            # Epoch subsampling dropped interior timepoints: the base dataset returns
            # ``len(active) - 1`` transitions ``active[t] -> active[t+1]``. The sidecar is
            # stored on the full timeline, so remap prefix / target steps accordingly.
            if len(active) < 2 or max(active) > n_steps or min(active) < 0:
                raise ValueError(
                    f"active timeline {active} incompatible with sidecar sample {si} "
                    f"(n_transition_steps={n_steps})"
                )
            prefix_steps = [int(a) for a in active[:-1]]
            subst_steps = [self.store.prefix_features(si, a) for a in prefix_steps]
            if self.return_target_activation:
                target_steps = [
                    self.store.target_activation_mask_span(
                        si, int(active[t]), int(active[t + 1])
                    )
                    for t in range(len(active) - 1)
                ]
                return (*self._normalize_item(item), subst_steps, target_steps)
            return (*self._normalize_item(item), subst_steps)
        subst_steps = [self.store.prefix_features(si, t) for t in range(n_feat)]
        if self.return_target_activation:
            target_steps = [
                self.store.target_activation_mask(si, t) for t in range(n_feat)
            ]
            return (*self._normalize_item(item), subst_steps, target_steps)
        return (*self._normalize_item(item), subst_steps)

    def _active_timeline_indices(self) -> Optional[list[int]]:
        """Active (full-timeline) genome indices set by epoch subsampling, else ``None``."""
        for leaf in _iter_leaf_datasets_local(self.base):
            active = getattr(leaf, "_timeline_indices", None)
            if active is not None:
                return [int(a) for a in active]
            return None
        return None

    @staticmethod
    def _normalize_item(item: Any) -> tuple:
        if isinstance(item, (list, tuple)):
            return tuple(item)
        return (item,)


def _wrap_concat_parts(
    ds: ConcatDataset,
    store: SubstitutionSidecarStore,
    *,
    return_target_activation: bool,
) -> Dataset:
    """Wrap each concat part as outermost SubstitutionFeatureDataset.

    Sample indices restart at 0 per part (each part is expected to share the
    same sidecar layout only when parts are alternate views of the same split;
    multi-horizon unions should not enable substitution).
    """
    parts: list[Dataset] = []
    for sub in ds.datasets:
        if isinstance(sub, ConcatDataset):
            parts.append(
                _wrap_concat_parts(
                    sub, store, return_target_activation=return_target_activation
                )
            )
        else:
            parts.append(
                SubstitutionFeatureDataset(
                    sub,
                    store,
                    return_target_activation=return_target_activation,
                    sample_offset=0,
                )
            )
    return ConcatDataset(parts)


def _sidecar_split_key(leaf_split: str) -> str:
    if leaf_split == "val":
        return "val"
    if leaf_split == "test":
        return "test"
    return leaf_split


def attach_substitution_sidecars(
    datasets: list[Optional[Dataset]],
    cfg,
    data_cfg,
    *,
    active_splits: Optional[set[str]] = None,
) -> list[Optional[Dataset]]:
    """Attach B1 prefix substitution features as the outermost dataset wrapper."""
    if not bool(cfg.get("append_msa_substitution", False)):
        return list(datasets)
    return_target = bool(cfg.get("substitution_target_activation", False))
    override = cfg.get("substitution_sidecar_path")
    wrapped_out: list[Optional[Dataset]] = []

    for ds in datasets:
        if ds is None:
            wrapped_out.append(None)
            continue
        leaves = _iter_leaf_datasets_local(ds)
        split = str(getattr(leaves[0], "split", "train")) if leaves else "train"
        split_key = _sidecar_split_key(split)
        eval_split = "cal" if split_key == "val" else split_key
        if active_splits is not None and eval_split not in active_splits:
            wrapped_out.append(ds)
            continue
        path = resolve_substitution_sidecar_path(
            data_cfg, split=split_key, override=override
        )
        if not path.is_file():
            alt = (
                "Run: .venv/bin/python3 -m recomb2027.latent.predictors.build_msa_substitution_sidecar "
                f'--config configs/latent/diagnostics/b1_substitution_sidecar.yaml '
                f'--splits {split_key}'
            )
            raise FileNotFoundError(
                f"substitution sidecar missing for split={split_key!r}: {path}\n{alt}"
            )
        store = SubstitutionSidecarStore.from_path(path)
        if isinstance(ds, ConcatDataset):
            wrapped_out.append(
                _wrap_concat_parts(
                    ds, store, return_target_activation=return_target
                )
            )
        else:
            wrapped_out.append(
                SubstitutionFeatureDataset(
                    ds,
                    store,
                    return_target_activation=return_target,
                    sample_offset=0,
                )
            )
    return wrapped_out


def _iter_leaf_datasets_local(ds: Dataset) -> list[Dataset]:
    if isinstance(ds, ConcatDataset):
        out: list[Dataset] = []
        for sub in ds.datasets:
            out.extend(_iter_leaf_datasets_local(sub))
        return out
    if isinstance(ds, SubstitutionFeatureDataset):
        return _iter_leaf_datasets_local(ds.base)
    if hasattr(ds, "base") and isinstance(getattr(ds, "base"), Dataset):
        return _iter_leaf_datasets_local(ds.base)
    return [ds]
