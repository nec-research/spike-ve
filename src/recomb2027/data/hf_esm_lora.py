#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     hf_esm_lora.py
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
"""Load HuggingFace ESM-1b checkpoints with optional PEFT LoRA adapters.

Fine-tuned checkpoints produced by ``recomb2027.pipelines.finetune_esm``
use ``transformers`` + ``peft`` (base: ``facebook/esm1b_t33_650M_UR50S``).  This
module exposes a fair-esm-compatible ``forward(toks, repr_layers=...)`` wrapper
so ``encode_proteins_from_fasta`` can reuse the same pooling / batching logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch
import torch.nn as nn

DEFAULT_HF_ESM1B_BASE = "facebook/esm1b_t33_650M_UR50S"
DEFAULT_LAYERS_HF_ESM1B = [12, 33]
HF_ESM1B_NUM_LAYERS = 33


def _read_base_model_name(adapter_path: Path) -> str:
    cfg_path = adapter_path / "adapter_config.json"
    if cfg_path.is_file():
        with open(cfg_path) as f:
            cfg = json.load(f)
        base = cfg.get("base_model_name_or_path")
        if base:
            return str(base)
    return DEFAULT_HF_ESM1B_BASE


class HFEsmAlphabet:
    """Minimal fair-esm ``Alphabet`` shim backed by a HF ESM tokenizer."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self.padding_idx = int(tokenizer.pad_token_id)
        self.mask_idx = getattr(tokenizer, "mask_token_id", None)
        self.cls_idx = getattr(tokenizer, "cls_token_id", None)
        self.eos_idx = getattr(tokenizer, "eos_token_id", None)
        self.all_toks = list(getattr(tokenizer, "all_tokens", []) or [])

    def get_idx(self, token: str) -> int:
        return int(self._tokenizer.convert_tokens_to_ids(token))

    def get_batch_converter(self, max_len: int | None = None) -> Callable:
        tokenizer = self._tokenizer
        pad_id = self.padding_idx

        def convert(batch: Sequence[tuple[str, str]]):
            labels = [item[0] for item in batch]
            strs = [item[1] for item in batch]
            encoded = tokenizer(
                strs,
                padding=True,
                truncation=max_len is not None,
                max_length=max_len,
                return_tensors="pt",
            )
            toks = encoded["input_ids"]
            if toks.dtype != torch.int64:
                toks = toks.to(dtype=torch.int64)
            return labels, strs, toks

        return convert


class HFEsmLoRAModel(nn.Module):
    """fair-esm-compatible wrapper around a HF ESM MLM (+ optional LoRA)."""

    def __init__(self, hf_model: nn.Module, padding_idx: int):
        super().__init__()
        self.hf_model = hf_model
        self.padding_idx = int(padding_idx)
        self.num_layers = HF_ESM1B_NUM_LAYERS

    def forward(
        self,
        toks: torch.Tensor,
        repr_layers: Iterable[int] | None = None,
        return_contacts: bool = False,
    ):
        if return_contacts:
            raise NotImplementedError(
                "Contact maps are not supported for HuggingFace ESM models."
            )
        requested = sorted({int(layer) for layer in (repr_layers or [])})
        for layer in requested:
            if layer < 1 or layer > self.num_layers:
                raise ValueError(
                    f"Layer {layer} is out of range for HF ESM-1b "
                    f"(valid: 1..{self.num_layers})."
                )

        attention_mask = toks.ne(self.padding_idx)
        outputs = self.hf_model(
            input_ids=toks,
            attention_mask=attention_mask,
            output_hidden_states=bool(requested),
        )
        hidden_states = outputs.hidden_states
        representations = {
            layer: hidden_states[layer] for layer in requested
        }
        return {"representations": representations}


def load_hf_esm_lora_model(
    adapter_path: str | Path,
    *,
    base_model: str | None = None,
    merge_lora: bool = False,
    device: str | torch.device = "cpu",
):
    """Load HF ESM-1b with an optional PEFT LoRA adapter directory.

    Args:
        adapter_path: Directory with ``adapter_config.json`` (e.g. ``.../best``).
        base_model: HF hub id or local path for the base checkpoint.  When
            omitted, read from ``adapter_config.json`` or fall back to
            ``facebook/esm1b_t33_650M_UR50S``.
        merge_lora: If True, merge LoRA weights into the base model for
            inference (slightly faster; irreversible for the returned object).
        device: Initial device for weight loading.

    Returns:
        ``(model, alphabet)`` where ``model`` implements fair-esm ``forward``.
    """
    try:
        from peft import PeftModel
        from transformers import AutoModelForMaskedLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "HuggingFace ESM LoRA loading requires optional dependencies: "
            "pip install transformers peft (see requirements.txt)."
        ) from exc

    adapter_path = Path(adapter_path)
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

    base_name = base_model or _read_base_model_name(adapter_path)
    tok_dir = adapter_path if (adapter_path / "tokenizer_config.json").is_file() else base_name
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    alphabet = HFEsmAlphabet(tokenizer)

    base = AutoModelForMaskedLM.from_pretrained(base_name)
    if (adapter_path / "adapter_config.json").is_file():
        hf_model = PeftModel.from_pretrained(base, str(adapter_path))
        if merge_lora:
            hf_model = hf_model.merge_and_unload()
    else:
        hf_model = base

    hf_model = hf_model.to(device)
    hf_model.eval()
    wrapper = HFEsmLoRAModel(hf_model, padding_idx=alphabet.padding_idx)
    return wrapper, alphabet
