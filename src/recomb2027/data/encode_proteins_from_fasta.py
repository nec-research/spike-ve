#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     encode_proteins_from_fasta.py
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
Encode protein sequences from a FASTA file using an ESM language model.

Disk space quick reference (ESM1B, hidden=1280, spike ~1273 AA, 3 layers):
  2D token-level  : ~18.6 MB/sequence  →  ~12.5 TB for 670k sequences
  1D pooled       : ~15 KB/sequence    →  ~10 GB   for 670k sequences  (×1000 smaller)

By default only pooled 1D embeddings are written to disk.  Pass --save-2d to also
keep the full token-level tensors (use only for small datasets or targeted analysis).

Output layout:
  <output-dir>/<pooling_strategy>/<layer>/<label>.pt   ← pooled 1D (default)
  <output-dir>/<label>.pt                              ← 2D token-level (--save-2d)

Example – recommended invocation for a large dataset (single 10 GB GPU):

    python -m recomb2027.data.encode_proteins_from_fasta \\
        --input /mnt/container-nle-virusevolution/phylotree_matched/filtered_msa.fasta \\
        --output-dir /mnt/container-nle-virusevolution/encodings/ESM1B/filtered_msa \\
        --model ESM1B \\
        --layers 12 33 34 \\
        --gpu 0 \\
        --token-budget 16384 \\
        --aggregate-pooling average_token cls_token \\
        --skip-existing

Example – multi-GPU with DataParallel (3 GPUs, budget scaled ×3):

    python -m recomb2027.data.encode_proteins_from_fasta \\
        --input filtered_msa.fasta \\
        --output-dir ./encodings \\
        --gpu 0 1 2 \\
        --token-budget 49152 \\
        --aggregate-pooling average_token cls_token \\
        --skip-existing

Example – manual sharding across 4 GPUs (run each line in a separate terminal):

    python -m recomb2027.data.encode_proteins_from_fasta --gpu 0 --batch-offset 0    --max-batches 500 ...
    python -m recomb2027.data.encode_proteins_from_fasta --gpu 1 --batch-offset 500  --max-batches 500 ...
    python -m recomb2027.data.encode_proteins_from_fasta --gpu 2 --batch-offset 1000 --max-batches 500 ...
    python -m recomb2027.data.encode_proteins_from_fasta --gpu 3 --batch-offset 1500 --max-batches 500 ...

Example – also retain 2D embeddings (small dataset / per-position analysis):

    python -m recomb2027.data.encode_proteins_from_fasta \\
        --input sequences.fasta \\
        --output-dir ./encodings \\
        --token-budget 16384 \\
        --aggregate-pooling average_token stats_summary \\
        --save-2d

``stats_summary`` writes ``[hidden_dim, 7]`` tensors (mean, std, min, max, median,
first argmax/argmin position per dimension) under ``<output-dir>/stats_summary/<layer>/``.

Example – mean pooling plus selected extra stats (recommended for richer features):

    python -m recomb2027.data.encode_proteins_from_fasta \\
        --input filtered_msa.fasta \\
        --output-dir ./encodings/ESM1B/filtered_msa \\
        --layers 33 \\
        --aggregate-pooling average_token \\
        --extra-pooling-stats std median argmax_pos argmin_pos \\
        --token-budget 16384 \\
        --skip-existing

Extra stats are written to ``<output-dir>/<stat>/<layer>/<label>.pt`` as ``[hidden_dim]``
vectors. Position stats are normalized to ``[0, 1]`` by default (``--no-normalize-position-stats``
to keep raw residue indices).

Example – domain block pooling (Phase 2; same pooling as global mean):

    python -m recomb2027.data.encode_proteins_from_fasta \\
        --input filtered_msa.fasta \\
        --output-dir ./encodings/ESM1B/filtered_msa_stats \\
        --layers 33 \\
        --aggregate-pooling average_token \\
        --block-boundaries configs/latent/shared/spike_block_boundaries.yaml \\
        --block-pooling average_token \\
        --token-budget 16384 \\
        --skip-existing

Example – LoRA fine-tuned ESM-1b (``finetune_esm.py`` checkpoint):

    python -m recomb2027.data.encode_proteins_from_fasta \\
        --input /path/to/filtered_msa_S1.fasta \\
        --output-dir ./encodings/finetuned_esm1b/filtered_msa_S1 \\
        --model ESM1B \\
        --adapter-path /path/to/finetuned_esm/esm1b/best \\
        --layers 12 33 \\
        --aggregate-pooling average_token \\
        --token-budget 16384 \\
        --gpu 0 \\
        --skip-existing

Block vectors are written to
``<output-dir>/block_<name>/<pooling>/<layer>/<label>.pt`` (e.g. ``block_RBD_a/average_token/33/``).

Gap columns (``-`` in the MSA; ``.`` normalised to ``-`` before tokenisation) are
**excluded** from ``average_token``, block means, and extra stats by default.
Use ``--include-gap-columns`` to restore the old behaviour (mean over all columns).

For pooling **already-saved** 2D ``.pt`` files without loading ESM, use
``python -m recomb2027.data.aggregate_2d_encodings`` instead
(see that module's docstring for ``--include-gap-columns`` / block options).
"""

from __future__ import annotations

import gc
import logging
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from recomb2027.core.paths import repo_root
REPO_ROOT = repo_root(__file__)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import esm
from tqdm import tqdm

from recomb2027.data.aggregate_2d_encodings import aggregate_encodings
from recomb2027.data.esm_pooling import (
    EXTRA_STAT_NAMES,
    mean_pool_aa_tokens,
    mean_pool_sequence_dim,
    msa_non_gap_mask,
    pool_aa_tokens_stats,
    first_non_gap_row,
    pool_token_repr_single_stat,
    pool_token_repr_slice,
    pool_token_repr_stats,
    prepare_msa_seq_for_esm,
)
from recomb2027.data.hf_esm_lora import (
    DEFAULT_LAYERS_HF_ESM1B,
    load_hf_esm_lora_model,
)
from recomb2027.data.spike_blocks import (
    block_output_dirname,
    clamp_block_slice,
    load_spike_block_boundaries,
    max_block_msa_end,
    validate_block_boundaries,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_MODELS = ["ESM1B", "ESM2"]
SUPPORTED_POOLING = [
    "cls_token",
    "first_token",
    "average_token",
    "average_sequence",
    "stats_summary",
]
DEFAULT_LAYERS_ESM1B = [12, 33, 34]
DEFAULT_LAYERS_ESM2B = [12, 33]
DEFAULT_SEQ_LENGTH = 1300  # BV-BRC filtered MSA width; match spike_block_boundaries.yaml
MAX_FILENAME_LEN = 100
# Token budget controls how many tokens are packed into one batch.
# ESM's dataloader fills batches until this limit is hit, so longer sequences
# automatically result in fewer sequences per batch.
# Rule of thumb for ESM1B (float32):
#   GPU mem  |  safe budget  |  spike seqs/batch
#   ---------|---------------|------------------
#    10 GB   |  16000-24000  |  12-18
#    24 GB   |  40000-65000  |  30-50
#    40 GB   |  80000        |  60+
MEAN_POOLING_STRATEGIES = ("average_token", "average_sequence")
DEFAULT_TOKEN_BUDGET = 4096   # conservative; override with --token-budget


def _dedupe_preserve_order(names: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def label_to_filename(label: str, max_len: int = MAX_FILENAME_LEN) -> str:
    """Sanitise a sequence label for use as a filename."""
    return (
        label
        .replace("/", "-")
        .replace(":", "_")
        .replace(" ", "")
        .replace("'", "")
        .replace(",", "_")
    )[:max_len]


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

class ModelFactory:
    @staticmethod
    def load_model(
        model_type: str,
        *,
        adapter_path: str | Path | None = None,
        base_model: str | None = None,
        merge_lora: bool = False,
    ):
        if adapter_path is not None:
            if model_type != "ESM1B":
                raise ValueError(
                    f"--adapter-path is only supported with --model ESM1B "
                    f"(got {model_type!r})."
                )
            return load_hf_esm_lora_model(
                adapter_path,
                base_model=base_model,
                merge_lora=merge_lora,
            )
        if model_type == "ESM1B":
            model, alphabet = esm.pretrained.esm1_t34_670M_UR100()
        elif model_type == "ESM2":
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        else:
            raise ValueError(
                f"Unsupported model type: '{model_type}'. "
                f"Choose from {SUPPORTED_MODELS}."
            )
        return model, alphabet


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class ESMProteinEncoder:
    """Encodes protein sequences using a pre-trained ESM model."""

    def __init__(self, model, truncate_seq_length: int):
        self.model = model
        self.seq_length = truncate_seq_length
        self.repr_layers: list = []
        self.representations: list = []
        self._normalize_position_stats: bool = True
        self._block_clamp_warned: set[str] = set()
        self._exclude_gap_columns: bool = True
        self._gap_mask_fallback_warned: bool = False

    def _gap_mask_for(self, aligned_seq: str, truncate_len: int) -> torch.Tensor | None:
        if not self._exclude_gap_columns:
            return None
        return msa_non_gap_mask(aligned_seq, truncate_len)

    def load_data(self, dataset, alphabet, token_budget: int = DEFAULT_TOKEN_BUDGET):
        self.batches = dataset.get_batch_indices(token_budget, extra_toks_per_seq=1)
        self.data_loader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=alphabet.get_batch_converter(self.seq_length),
            batch_sampler=self.batches,
        )

    def _apply_pooling(
        self,
        enc_2d,
        pooling_strategy,
        *,
        msa_slice: tuple[int, int] | None = None,
        truncate_len: int | None = None,
        non_gap_mask: torch.Tensor | None = None,
    ):
        """Pool an already-extracted 2D [seq+1, hidden] tensor to a 1D vector.

        Assumes enc_2d includes the CLS token at row 0, matching the layout
        produced by _pool(..., pooling_strategy=None).

        When ``msa_slice=(start, end)`` is set, pool only MSA columns ``[start, end)``.

        All branches return a *contiguous* 1D clone so that torch.save writes
        only the pooled vector (≈ hidden_dim × 4 bytes) rather than the full
        underlying storage of enc_2d.
        """
        if msa_slice is not None:
            return pool_token_repr_slice(
                enc_2d,
                msa_slice[0],
                msa_slice[1],
                pooling_strategy,
                normalize_positions=self._normalize_position_stats,
                non_gap_mask=non_gap_mask,
            ).clone()
        tlen = truncate_len if truncate_len is not None else enc_2d.shape[0] - 1
        aa = enc_2d[1 : tlen + 1, :]
        mask = non_gap_mask[:tlen] if non_gap_mask is not None else None
        if pooling_strategy == "cls_token":
            # Row 0 is the CLS token; clone() breaks the view to avoid saving
            # the full [seq_len, hidden] storage (would inflate files ~1000×).
            return enc_2d[0, :].clone()
        elif pooling_strategy == "first_token":
            return first_non_gap_row(aa, mask).clone()
        elif pooling_strategy == "average_sequence":
            return mean_pool_sequence_dim(aa).clone()
        elif pooling_strategy == "average_token":
            return mean_pool_aa_tokens(aa, mask)
        elif pooling_strategy == "stats_summary":
            return pool_aa_tokens_stats(aa, non_gap_mask=mask).clone()
        elif pooling_strategy in EXTRA_STAT_NAMES:
            return pool_token_repr_single_stat(
                enc_2d,
                pooling_strategy,
                truncate_len=tlen,
                normalize_positions=self._normalize_position_stats,
                non_gap_mask=non_gap_mask,
            ).clone()
        else:
            raise ValueError(
                f"Unknown pooling strategy: '{pooling_strategy}'. "
                f"Choose from {SUPPORTED_POOLING}."
            )

    def _pool(self, token_repr, pooling_strategy, truncate_len, *, non_gap_mask=None):
        """Apply pooling to a single sequence's token representations.

        token_repr: 2D tensor [seq_len+2, hidden_dim] for one sequence and one
                    layer, as returned by ESM (CLS at row 0, EOS at the last row).
        truncate_len: number of amino-acid tokens to keep (excludes CLS / EOS).
        non_gap_mask: optional ``[truncate_len]`` bool; True = non-gap MSA column.
        """
        aa = token_repr[1 : truncate_len + 1]
        mask = non_gap_mask[:truncate_len] if non_gap_mask is not None else None
        if pooling_strategy == "cls_token":
            return token_repr[0, :]
        elif pooling_strategy == "first_token":
            return first_non_gap_row(aa, mask)
        elif pooling_strategy == "average_sequence":
            return mean_pool_sequence_dim(aa)
        elif pooling_strategy == "average_token":
            return mean_pool_aa_tokens(aa, mask)
        elif pooling_strategy == "stats_summary":
            return pool_token_repr_stats(
                token_repr, truncate_len, non_gap_mask=non_gap_mask
            )
        elif pooling_strategy is None:
            return token_repr[0 : truncate_len + 1]
        else:
            raise ValueError(
                f"Unknown pooling strategy: '{pooling_strategy}'. "
                f"Choose from {SUPPORTED_POOLING} or omit for 2D embeddings."
            )

    def _pooled_output_path(
        self,
        output_dir: Path,
        top_dir: str,
        layer: int,
        file_id: str,
        *,
        pool_name: str | None = None,
    ) -> Path:
        if pool_name is not None:
            return output_dir / top_dir / pool_name / str(layer) / f"{file_id}.pt"
        return output_dir / top_dir / str(layer) / f"{file_id}.pt"

    def _write_pooled_file(
        self,
        output_dir: Path,
        top_dir: str,
        layer: int,
        file_id: str,
        label: str,
        pooled: torch.Tensor,
        skip_existing: bool,
        *,
        pool_name: str | None = None,
    ) -> None:
        pool_file = self._pooled_output_path(
            output_dir, top_dir, layer, file_id, pool_name=pool_name
        )
        if skip_existing and pool_file.exists():
            return
        pool_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"label": label, "encodings": pooled}, pool_file)

    def _write_block_poolings(
        self,
        enc_2d: torch.Tensor,
        truncate_len: int,
        gap_mask: torch.Tensor | None,
        layer: int,
        file_id: str,
        label: str,
        output_dir: Path,
        block_specs: list[dict],
        block_pooling: str,
        skip_existing: bool,
    ) -> None:
        for block in block_specs:
            name = str(block["name"])
            msa_slice = clamp_block_slice(
                int(block["msa_start"]),
                int(block["msa_end"]),
                truncate_len,
                block_name=name,
                _warned=self._block_clamp_warned,
            )
            if msa_slice is None:
                logging.error(
                    "Skipping block %s for %s: msa_start=%s >= truncate_len=%s",
                    name,
                    file_id,
                    block["msa_start"],
                    truncate_len,
                )
                continue
            try:
                pooled = self._apply_pooling(
                    enc_2d,
                    block_pooling,
                    msa_slice=msa_slice,
                    truncate_len=truncate_len,
                    non_gap_mask=gap_mask,
                )
            except ValueError as exc:
                logging.error(
                    "Skipping block %s for %s: %s",
                    name,
                    file_id,
                    exc,
                )
                continue
            self._write_pooled_file(
                output_dir,
                block_output_dirname(name),
                layer,
                file_id,
                label,
                pooled,
                skip_existing,
                pool_name=block_pooling,
            )

    def _write_extra_stats(
        self,
        enc_2d: torch.Tensor,
        truncate_len: int,
        gap_mask: torch.Tensor | None,
        layer: int,
        file_id: str,
        label: str,
        output_dir: Path,
        extra_pooling_stats: list,
        skip_existing: bool,
        normalize_position_stats: bool,
    ) -> None:
        for stat in extra_pooling_stats:
            pooled = pool_token_repr_single_stat(
                enc_2d,
                stat,
                truncate_len,
                normalize_positions=normalize_position_stats,
                non_gap_mask=gap_mask,
            ).clone()
            self._write_pooled_file(
                output_dir, stat, layer, file_id, label, pooled, skip_existing
            )

    def _batch_all_exist(
        self,
        labels: list,
        output_dir: Path,
        pooling_strategy,
        inline_pooling_strategies,
        repr_layers: list,
        save_2d: bool,
        extra_pooling_stats: list | None = None,
        block_specs: list | None = None,
        block_pooling: str | None = None,
    ) -> bool:
        """Return True only when every output file for every label in the batch
        already exists on disk.  Used to skip the forward pass entirely.

        Returns False when no output artifacts are configured (vacuous success
        would incorrectly skip encoding).
        """
        block_specs = block_specs or []
        checks_configured = bool(
            (inline_pooling_strategies and pooling_strategy is None)
            or extra_pooling_stats
            or (block_specs and block_pooling)
            or (save_2d and pooling_strategy is None)
            or pooling_strategy is not None
        )
        if not checks_configured:
            return False

        for label in labels:
            file_id = label_to_filename(label)
            if inline_pooling_strategies and pooling_strategy is None:
                for strategy in inline_pooling_strategies:
                    for layer in repr_layers:
                        if not self._pooled_output_path(
                            output_dir, str(strategy), layer, file_id
                        ).exists():
                            return False
            if extra_pooling_stats:
                for stat in extra_pooling_stats:
                    for layer in repr_layers:
                        if not self._pooled_output_path(
                            output_dir, str(stat), layer, file_id
                        ).exists():
                            return False
            if block_specs and block_pooling:
                for block in block_specs:
                    top = block_output_dirname(str(block["name"]))
                    for layer in repr_layers:
                        if not self._pooled_output_path(
                            output_dir, top, layer, file_id, pool_name=block_pooling
                        ).exists():
                            return False
            if save_2d and pooling_strategy is None:
                if not (output_dir / f"{file_id}.pt").exists():
                    return False
            if pooling_strategy is not None:
                if extra_pooling_stats and pooling_strategy in MEAN_POOLING_STRATEGIES:
                    for layer in repr_layers:
                        if not (
                            output_dir / str(pooling_strategy) / str(layer) / f"{file_id}.pt"
                        ).exists():
                            return False
                elif not (
                    output_dir / f"{file_id}.pt"
                ).exists():
                    return False
        return True

    def encode(
        self,
        repr_layers: list,
        gpus: list = None,
        pooling_strategy=None,
        output_dir: Path = None,
        skip_existing: bool = False,
        max_batches: int = None,
        batch_offset: int = 0,
        inline_pooling_strategies: list = None,
        extra_pooling_stats: list = None,
        block_specs: list = None,
        block_pooling: str | None = None,
        normalize_position_stats: bool = True,
        exclude_gap_columns: bool = True,
        save_2d: bool = False,
    ):
        """Run inference and (optionally) save embeddings to disk.

        Args:
            repr_layers:               Transformer layers to extract.
            gpus:                      List of GPU device indices to use.
                                       - Single GPU [0]: standard inference.
                                       - Multiple GPUs [0, 1, 2]: DataParallel splits
                                         each batch across all listed GPUs, results
                                         gathered on gpus[0]. Scale --token-budget by
                                         the number of GPUs to fill them all.
                                       - Empty list or None: CPU inference.
            pooling_strategy:          Pooling applied per sequence. None → full 2D
                                       token-level layout used internally; whether it
                                       is written to disk is controlled by save_2d.
            output_dir:                Directory to write embedding files.
            skip_existing:             If True, skip sequences whose output file exists.
            max_batches:               Stop after this many batches. None means all.
                                       Useful for dry-runs and throughput estimation.
            batch_offset:              Skip the first N batches before processing.
                                       Used together with max_batches to shard the
                                       dataset across independent processes/GPUs
                                       without pre-splitting the FASTA file.
            inline_pooling_strategies: Pool the 2D tensors with these strategies while
                                       they are already in CPU memory (right after each
                                       batch), avoiding a separate disk-read pass.
                                       Files are written to
                                       ``output_dir/<strategy>/<layer>/<file_id>.pt``.
                                       Only active when pooling_strategy is None.
            extra_pooling_stats:     Additional per-residue stats (``std``, ``median``,
                                       ``argmax_pos``, …) written alongside mean pooling
                                       to ``output_dir/<stat>/<layer>/<file_id>.pt``.
            block_specs:             Domain block definitions (``name``, ``msa_start``,
                                       ``msa_end``) for per-block pooled outputs under
                                       ``output_dir/block_<name>/<block_pooling>/<layer>/``.
            block_pooling:           Pooling strategy applied within each block slice.
            normalize_position_stats: Divide ``argmax_pos`` / ``argmin_pos`` by sequence
                                       length when writing extra stat files.
            save_2d:                   Write the full 2D token-level .pt files to
                                       output_dir. Default False — at large scale
                                       (670k sequences, ESM1B) these files total ~12 TB.
                                       Set True only for small datasets or when
                                       per-position analysis is needed.
        """
        self.repr_layers = repr_layers
        self.model.eval()
        self._normalize_position_stats = normalize_position_stats
        self._exclude_gap_columns = exclude_gap_columns

        gpus = [g for g in (gpus or []) if g >= 0]  # -1 means CPU
        use_cuda = len(gpus) > 0 and torch.cuda.is_available()
        primary_device = f"cuda:{gpus[0]}" if use_cuda else "cpu"

        if use_cuda:
            self.model = self.model.to(primary_device)
            if len(gpus) > 1:
                logging.info(f"DataParallel across GPUs {gpus}.")
                self.model = torch.nn.DataParallel(self.model, device_ids=gpus)
            else:
                logging.info(f"Running inference on GPU {gpus[0]}.")
        else:
            logging.info("Running inference on CPU.")

        extra_pooling_stats = _dedupe_preserve_order(extra_pooling_stats or [])
        block_specs = list(block_specs or [])

        # Pre-create subdirectories for inline pooling outputs
        if output_dir is not None and save_2d:
            for layer in repr_layers:
                (output_dir / "2d" / str(layer)).mkdir(parents=True, exist_ok=True)
        if output_dir is not None and pooling_strategy is None:
            strategies = list(inline_pooling_strategies or [])
            for strategy in strategies:
                for layer in repr_layers:
                    (output_dir / str(strategy) / str(layer)).mkdir(parents=True, exist_ok=True)
            for stat in extra_pooling_stats:
                for layer in repr_layers:
                    (output_dir / str(stat) / str(layer)).mkdir(parents=True, exist_ok=True)
            if block_specs and block_pooling:
                for block in block_specs:
                    for layer in repr_layers:
                        (
                            output_dir
                            / block_output_dirname(str(block["name"]))
                            / str(block_pooling)
                            / str(layer)
                        ).mkdir(parents=True, exist_ok=True)        
        elif output_dir is not None and extra_pooling_stats:
            for stat in extra_pooling_stats:
                for layer in repr_layers:
                    (output_dir / str(stat) / str(layer)).mkdir(parents=True, exist_ok=True)
            if pooling_strategy in MEAN_POOLING_STRATEGIES:
                for layer in repr_layers:
                    (output_dir / str(pooling_strategy) / str(layer)).mkdir(
                        parents=True, exist_ok=True
                    )
        elif output_dir is not None and block_specs and block_pooling:
            for block in block_specs:
                for layer in repr_layers:
                    (
                        output_dir
                        / block_output_dirname(str(block["name"]))
                        / str(block_pooling)
                        / str(layer)
                    ).mkdir(parents=True, exist_ok=True)
            if pooling_strategy in MEAN_POOLING_STRATEGIES:
                for layer in repr_layers:
                    (output_dir / str(pooling_strategy) / str(layer)).mkdir(
                        parents=True, exist_ok=True
                    )

        self.representations = []
        start_time = time.time()

        n_batches_total = len(self.data_loader)
        # effective window after skipping batch_offset batches
        available = n_batches_total - batch_offset
        n_batches_run = (
            min(max_batches, available) if max_batches is not None else available
        )
        if batch_offset:
            logging.info(f"Skipping first {batch_offset} batches (batch_offset).")
        if max_batches is not None:
            logging.info(
                f"max_batches={max_batches}: will process "
                f"{n_batches_run}/{n_batches_total} batches."
            )

        with torch.no_grad():
            for batch_idx, (labels, strs, toks) in enumerate(
                tqdm(self.data_loader, desc="Encoding batches", total=n_batches_run)
            ):
                # skip batches before the requested offset
                if batch_idx < batch_offset:
                    continue

                if max_batches is not None and (batch_idx - batch_offset) >= max_batches:
                    logging.info(f"Reached max_batches={max_batches}, stopping.")
                    break

                # Skip inference entirely when every output file for this batch
                # already exists.  Checking labels (available before the forward
                # pass) avoids wasting GPU time on sequences already encoded.
                if skip_existing and output_dir is not None:
                    if self._batch_all_exist(
                        labels, output_dir, pooling_strategy,
                        inline_pooling_strategies, repr_layers, save_2d,
                        extra_pooling_stats,
                        block_specs=block_specs,
                        block_pooling=block_pooling,
                    ):
                        logging.info(
                            f"Batch {batch_idx + 1}/{n_batches_total}: "
                            f"all {len(labels)} output files exist — skipping inference."
                        )
                        continue

                logging.info(
                    f"Batch {batch_idx + 1}/{n_batches_total} "
                    f"({len(labels)} sequences)"
                )

                if use_cuda:
                    toks = toks.to(device=primary_device, non_blocking=True)

                iter_start = time.time()
                out = self.model(toks, repr_layers=repr_layers, return_contacts=False)

                # Move all layer tensors to CPU immediately to free GPU memory.
                # With DataParallel the gather already happened on primary_device,
                # so .to("cpu") works the same as the single-GPU case.
                representations = {
                    layer: t.to(device="cpu")
                    for layer, t in out["representations"].items()
                }

                for i, label in enumerate(labels):
                    truncate_len = min(self.seq_length, len(strs[i]))
                    aligned_seq = strs[i][:truncate_len]
                    gap_mask = self._gap_mask_for(aligned_seq, truncate_len)

                    encodings = {
                        layer: self._pool(
                            t[i], pooling_strategy, truncate_len, non_gap_mask=gap_mask
                        ).clone()
                        for layer, t in representations.items()
                    }
                    encodings_2d = None
                    need_2d = bool(
                        extra_pooling_stats
                        or block_specs
                        or inline_pooling_strategies
                        or (save_2d and pooling_strategy is None)
                    )
                    if need_2d:
                        encodings_2d = {
                            layer: self._pool(t[i], None, truncate_len).clone()
                            for layer, t in representations.items()
                        }

                    result = {
                        "label": label,
                        "encodings": encodings,
                        "pooling_strategy": pooling_strategy,
                    }

                    if output_dir is not None:
                        file_id = label_to_filename(label)

                        # 2D token-level file — opt-in only (can be ~18 MB/sequence)
                        if save_2d and pooling_strategy is None:
                            out_file = output_dir / "2d" / str(layer) / f"{file_id}.pt"
                            if not (skip_existing and out_file.exists()):
                                result_2d = {
                                    "label": label,
                                    "encodings": encodings_2d or encodings,
                                    "pooling_strategy": None,
                                    "aligned_seq": aligned_seq,
                                }
                                torch.save(result_2d, out_file)
                                logging.debug(f"Saved 2D: {out_file.name}")

                        # Inline pooling: tensors are already in CPU memory, so pool
                        # and write now rather than doing a second disk-read pass later.
                        if inline_pooling_strategies and pooling_strategy is None:
                            for strategy in inline_pooling_strategies:
                                for layer, enc_2d in (encodings_2d or encodings).items():
                                    pooled = self._apply_pooling(
                                        enc_2d,
                                        strategy,
                                        truncate_len=truncate_len,
                                        non_gap_mask=gap_mask,
                                    )
                                    self._write_pooled_file(
                                        output_dir,
                                        strategy,
                                        layer,
                                        file_id,
                                        label,
                                        pooled,
                                        skip_existing,
                                    )

                        if extra_pooling_stats and encodings_2d is not None:
                            for layer, enc_2d in encodings_2d.items():
                                self._write_extra_stats(
                                    enc_2d,
                                    truncate_len,
                                    gap_mask,
                                    layer,
                                    file_id,
                                    label,
                                    output_dir,
                                    extra_pooling_stats,
                                    skip_existing,
                                    normalize_position_stats,
                                )

                        if block_specs and block_pooling and encodings_2d is not None:
                            for layer, enc_2d in encodings_2d.items():
                                self._write_block_poolings(
                                    enc_2d,
                                    truncate_len,
                                    gap_mask,
                                    layer,
                                    file_id,
                                    label,
                                    output_dir,
                                    block_specs,
                                    block_pooling,
                                    skip_existing,
                                )

                        if pooling_strategy is not None:
                            if extra_pooling_stats and pooling_strategy in MEAN_POOLING_STRATEGIES:
                                for layer, pooled in encodings.items():
                                    self._write_pooled_file(
                                        output_dir,
                                        pooling_strategy,
                                        layer,
                                        file_id,
                                        label,
                                        pooled,
                                        skip_existing,
                                    )
                            elif not extra_pooling_stats:
                                out_file = output_dir / f"{file_id}.pt"
                                if not (skip_existing and out_file.exists()):
                                    torch.save(result, out_file)
                                    logging.debug(f"Saved pooled: {out_file.name}")
                            else:
                                out_file = output_dir / f"{file_id}.pt"
                                if not (skip_existing and out_file.exists()):
                                    torch.save(result, out_file)
                                    logging.debug(f"Saved pooled: {out_file.name}")

                        del encodings, encodings_2d, result
                    else:
                        # No output_dir: caller wants in-memory results (small runs only)
                        self.representations.append(result)

                iter_elapsed = time.time() - iter_start
                logging.info(f"Batch {batch_idx + 1} done in {iter_elapsed:.1f}s")

                # Free intermediate tensors
                del representations, out, toks
                gc.collect()
                if use_cuda:
                    torch.cuda.empty_cache()

        total_elapsed = time.time() - start_time
        logging.info(f"Encoding complete in {total_elapsed:.1f}s.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Encode protein sequences from a FASTA file with an ESM model "
            "and save per-sequence embeddings as .pt files."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input / output
    parser.add_argument(
        "--input", required=True, metavar="FASTA",
        help="Path to the input FASTA file.",
    )
    parser.add_argument(
        "--output-dir", required=True, metavar="DIR",
        help="Directory where per-sequence .pt embedding files are saved.",
    )

    # Model configuration
    parser.add_argument(
        "--model", default="ESM1B", choices=SUPPORTED_MODELS,
        help="ESM model variant.",
    )
    parser.add_argument(
        "--adapter-path", default=None, metavar="DIR",
        help=(
            "Directory with a PEFT LoRA adapter (e.g. finetune_esm.py ``.../best``). "
            "Loads HuggingFace ESM-1b instead of fair-esm; requires "
            "``pip install transformers peft`` (see requirements.txt). Only valid with --model ESM1B."
        ),
    )
    parser.add_argument(
        "--base-model", default=None, metavar="NAME",
        help=(
            "HuggingFace base checkpoint for --adapter-path "
            "(default: read from adapter_config.json or facebook/esm1b_t33_650M_UR50S)."
        ),
    )
    parser.add_argument(
        "--merge-lora", action="store_true",
        help=(
            "Merge LoRA weights into the base model at load time (faster inference)."
        ),
    )
    parser.add_argument(
        "--layers", nargs="+", type=int, default=None, metavar="L",
        help=(
            "Transformer layers to extract. "
            f"Default: {DEFAULT_LAYERS_ESM1B} (ESM1B) / "
            f"{DEFAULT_LAYERS_ESM2B} (ESM2B)."
        ),
    )
    parser.add_argument(
        "--seq-length", type=int, default=DEFAULT_SEQ_LENGTH, metavar="N",
        help="Maximum sequence length (longer sequences are truncated).",
    )
    parser.add_argument(
        "--gpu", nargs="+", type=int, default=[0], metavar="ID",
        help=(
            "GPU device index/indices. "
            "Single value (default: 0): standard single-GPU inference. "
            "Multiple values (e.g. --gpu 0 1 2): DataParallel across all listed GPUs "
            "— each batch is split evenly; scale --token-budget by the number of GPUs. "
            "Pass -1 to force CPU (e.g. --gpu -1)."
        ),
    )
    parser.add_argument(
        "--token-budget", type=int, default=DEFAULT_TOKEN_BUDGET, metavar="N",
        help=(
            "Maximum total tokens packed into one batch. "
            "Higher values increase GPU utilisation and throughput but use more memory. "
            "For spike proteins (~1274 tokens) on ESM1B: "
            "4096→~3 seqs/batch (~3.6 GB), "
            "16384→~12 seqs/batch (~5.2 GB), "
            "32768→~25 seqs/batch (~7.5 GB). "
            "With N GPUs via DataParallel, multiply the budget by N. "
            f"Default: {DEFAULT_TOKEN_BUDGET} (conservative)."
        ),
    )
    parser.add_argument(
        "--batch-offset", type=int, default=0, metavar="N",
        help=(
            "Skip the first N batches before starting to encode. "
            "Use together with --max-batches to shard one FASTA across "
            "independent processes without pre-splitting the file. "
            "Example — 4-GPU manual sharding (run in parallel): "
            "  --batch-offset 0  --max-batches 500 --gpu 0  "
            "  --batch-offset 500 --max-batches 500 --gpu 1  "
            "  --batch-offset 1000 --max-batches 500 --gpu 2  "
            "  --batch-offset 1500 --max-batches 500 --gpu 3"
        ),
    )

    # Pooling
    parser.add_argument(
        "--pooling-strategy", default=None, choices=SUPPORTED_POOLING,
        metavar="STRATEGY",
        help=(
            "Pooling applied during encoding. Omit to save full 2D "
            "(token-level) embeddings. "
            f"Choices: {SUPPORTED_POOLING}."
        ),
    )
    parser.add_argument(
        "--aggregate-pooling", nargs="+", default=None,
        choices=SUPPORTED_POOLING, metavar="STRATEGY",
        help=(
            "Pool the saved 2D embeddings with these strategies. "
            "By default this is done inline during encoding (no extra disk read). "
            "Use --no-inline-pooling to run it as a separate pass instead. "
            f"Choices: {SUPPORTED_POOLING}."
        ),
    )
    parser.add_argument(
        "--no-inline-pooling", action="store_true",
        help=(
            "When --aggregate-pooling is set, run pooling as a separate pass "
            "over saved 2D files instead of inline during encoding. "
            "Useful when adding pooling to an already-completed encoding run."
        ),
    )
    parser.add_argument(
        "--extra-pooling-stats",
        nargs="+",
        default=None,
        metavar="STAT",
        choices=list(EXTRA_STAT_NAMES),
        help=(
            "Additional per-residue stats to write alongside mean pooling "
            f"(one ``[hidden_dim]`` file per stat under ``<output-dir>/<stat>/<layer>/``). "
            f"Choices: {', '.join(EXTRA_STAT_NAMES)}. "
            "Use with ``--aggregate-pooling average_token`` or "
            "``--pooling-strategy average_token``."
        ),
    )
    parser.add_argument(
        "--no-normalize-position-stats",
        action="store_true",
        help=(
            "When writing ``argmax_pos`` / ``argmin_pos`` extra stats, keep raw "
            "0-based residue indices instead of dividing by sequence length."
        ),
    )
    parser.add_argument(
        "--block-boundaries",
        default=None,
        metavar="YAML",
        help=(
            "YAML file with Spike domain block definitions (``blocks`` list with "
            "``name``, ``msa_start``, ``msa_end``). Writes per-block pooled vectors "
            "under ``<output-dir>/block_<name>/<block-pooling>/<layer>/``."
        ),
    )
    parser.add_argument(
        "--block-pooling",
        default="average_token",
        choices=SUPPORTED_POOLING,
        metavar="STRATEGY",
        help=(
            "Pooling applied within each domain block slice (default: average_token, "
            "same as global mean pooling)."
        ),
    )
    parser.add_argument(
        "--include-gap-columns",
        action="store_true",
        help=(
            "Include alignment gap columns (``-``) when mean/stat pooling. "
            "Default: gaps are excluded from average_token, block means, and extra stats."
        ),
    )
    parser.add_argument(
        "--save-2d", action="store_true",
        help=(
            "Also write full 2D token-level embedding files to <output-dir>. "
            "WARNING: at ESM1B scale these are ~18.6 MB per sequence "
            "(~12.5 TB for 670k sequences). "
            "Omit this flag for large datasets and use --aggregate-pooling instead."
        ),
    )

    # Misc
    parser.add_argument(
        "--max-batches", type=int, default=None, metavar="N",
        help=(
            "Stop after processing N batches. Omit to process the full dataset. "
            "Useful for dry-runs and throughput estimation before a full run."
        ),
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip sequences whose output .pt file already exists (resume mode).",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )

    return parser.parse_args()


def setup_logging(output_dir: Path, level: str):
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"encode_{timestamp}.log"

    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(str(log_file)),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logging.info(f"Log file: {log_file}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(output_dir, args.log_level)

    if not input_path.exists():
        logging.error(f"Input file not found: {input_path}")
        raise FileNotFoundError(input_path)

    using_hf_lora = bool(args.adapter_path)
    layers = args.layers or (
        DEFAULT_LAYERS_HF_ESM1B
        if using_hf_lora
        else (
            DEFAULT_LAYERS_ESM1B
            if args.model == "ESM1B"
            else DEFAULT_LAYERS_ESM2B
        )
    )

    logging.info("=" * 60)
    logging.info("Protein embedding pipeline")
    logging.info(f"  Input FASTA  : {input_path}")
    logging.info(f"  Output dir   : {output_dir}")
    logging.info(f"  Model        : {args.model}")
    if using_hf_lora:
        logging.info(f"  Adapter path : {args.adapter_path}")
        if args.base_model:
            logging.info(f"  Base model   : {args.base_model}")
        logging.info(f"  Merge LoRA   : {args.merge_lora}")
    logging.info(f"  Layers       : {layers}")
    logging.info(f"  Seq length   : {args.seq_length}")
    logging.info(f"  GPU(s)       : {args.gpu if args.gpu != [-1] else 'CPU'}")
    logging.info(f"  Token budget : {args.token_budget}")
    logging.info(f"  Batch offset : {args.batch_offset}")
    logging.info(f"  Pooling      : {args.pooling_strategy}")
    logging.info(f"  Agg. pooling : {args.aggregate_pooling or 'none'}")
    logging.info(f"  Inline pool  : {not args.no_inline_pooling}")
    logging.info(f"  Save 2D      : {args.save_2d}")
    logging.info(f"  Max batches  : {args.max_batches or 'all'}")
    extra_stats = _dedupe_preserve_order(args.extra_pooling_stats or [])
    has_mean_inline = args.aggregate_pooling and any(
        s in MEAN_POOLING_STRATEGIES for s in args.aggregate_pooling
    )
    has_mean_direct = args.pooling_strategy in MEAN_POOLING_STRATEGIES
    if extra_stats and not (has_mean_inline or has_mean_direct):
        logging.warning(
            "--extra-pooling-stats is set but no mean pooling strategy "
            "(--aggregate-pooling average_token or --pooling-strategy average_token). "
            "Extra stats will still be written when 2D tensors are available."
        )

    logging.info(f"  Extra stats    : {extra_stats or 'none'}")
    logging.info(
        f"  Norm positions : {not args.no_normalize_position_stats}"
    )
    logging.info(
        f"  Exclude gaps   : {not args.include_gap_columns}"
    )
    block_cfg = None
    block_specs: list[dict] = []
    if args.block_boundaries:
        block_cfg = load_spike_block_boundaries(args.block_boundaries)
        block_specs = list(block_cfg.get("blocks") or [])
        msa_len = int(block_cfg.get("msa_length") or 0)
        if msa_len:
            validate_block_boundaries(block_cfg, msa_length=msa_len)
        else:
            validate_block_boundaries(block_cfg)
        max_end = max_block_msa_end(block_cfg)
        if max_end and args.seq_length < max_end:
            logging.warning(
                "--seq-length %s is smaller than max block msa_end=%s; "
                "tail block slices will be clamped. Use --seq-length >= %s.",
                args.seq_length,
                max_end,
                max_end,
            )
        logging.info(f"  Block YAML     : {args.block_boundaries}")
        logging.info(
            f"  Block pooling  : {args.block_pooling} "
            f"({len(block_specs)} blocks: "
            f"{', '.join(str(b['name']) for b in block_specs)})"
        )
    logging.info("=" * 60)

    # Load dataset
    logging.info("Loading FASTA dataset...")
    dataset = esm.data.FastaBatchedDataset.from_file(str(input_path))
    logging.info(f"  {len(dataset)} sequences loaded.")

    # Normalize MSA strings for ESM (uppercase, '.' → '-', '*' → 'X').
    n_star = sum(seq.count("*") for seq in dataset.sequence_strs)
    dataset.sequence_strs = [
        prepare_msa_seq_for_esm(seq) for seq in dataset.sequence_strs
    ]
    logging.info(
        "  Sequences normalized for ESM ('.' → '-', '*' → 'X', uppercased)."
    )
    if n_star:
        logging.info("  Replaced %d stop/invalid '*' characters with 'X'.", n_star)

    # Load model
    if using_hf_lora:
        logging.info(
            f"Loading fine-tuned ESM-1b from adapter {args.adapter_path}..."
        )
    else:
        logging.info(f"Loading {args.model} model...")
    model, alphabet = ModelFactory.load_model(
        args.model,
        adapter_path=args.adapter_path,
        base_model=args.base_model,
        merge_lora=args.merge_lora,
    )

    # Encode and save
    encoder = ESMProteinEncoder(model, truncate_seq_length=args.seq_length)
    encoder.load_data(dataset, alphabet, token_budget=args.token_budget)
    inline_pooling = (
        args.aggregate_pooling
        if args.aggregate_pooling and not args.no_inline_pooling
        else None
    )
    aggregate_pooling_for_pass = list(args.aggregate_pooling or [])
    save_2d = bool(args.save_2d)

    # Separate aggregate pass needs flat 2D ``.pt`` files on disk.
    if args.no_inline_pooling and aggregate_pooling_for_pass and not save_2d:
        logging.warning(
            "--no-inline-pooling with --aggregate-pooling requires saved 2D "
            "embeddings; enabling --save-2d."
        )
        save_2d = True

    if block_specs and not (has_mean_inline or has_mean_direct):
        logging.warning(
            "--block-boundaries without global mean pooling: auto-enabling "
            "average_token alongside block outputs (GA-compatible global mean)."
        )
        if args.no_inline_pooling:
            if "average_token" not in aggregate_pooling_for_pass:
                aggregate_pooling_for_pass = [
                    "average_token",
                    *aggregate_pooling_for_pass,
                ]
            if not save_2d:
                logging.warning(
                    "--no-inline-pooling with block pooling requires saved 2D "
                    "embeddings; enabling --save-2d."
                )
                save_2d = True
        else:
            inline_pooling = list(inline_pooling or [])
            if "average_token" not in inline_pooling:
                inline_pooling.append("average_token")

    encoder.encode(
        repr_layers=layers,
        gpus=args.gpu,
        pooling_strategy=args.pooling_strategy,
        output_dir=output_dir,
        skip_existing=args.skip_existing,
        max_batches=args.max_batches,
        batch_offset=args.batch_offset,
        inline_pooling_strategies=inline_pooling,
        extra_pooling_stats=extra_stats,
        block_specs=block_specs,
        block_pooling=args.block_pooling if block_specs else None,
        normalize_position_stats=not args.no_normalize_position_stats,
        exclude_gap_columns=not args.include_gap_columns,
        save_2d=save_2d,
    )

    # Separate pooling pass over saved 2D files (--no-inline-pooling).
    if aggregate_pooling_for_pass and args.no_inline_pooling:
        logging.info(
            f"Running separate pooling pass with strategies: {aggregate_pooling_for_pass}"
        )
        aggregate_encodings(
            input_dir=output_dir,
            output_dir=output_dir,
            pooling_strategies=aggregate_pooling_for_pass,
            skip_existing=args.skip_existing,
            extra_pooling_stats=extra_stats,
            block_specs=block_specs,
            block_pooling=args.block_pooling if block_specs else None,
            normalize_position_stats=not args.no_normalize_position_stats,
            exclude_gap_columns=not args.include_gap_columns,
            layers=layers,
        )

    logging.info("All done.")


if __name__ == "__main__":
    main()
