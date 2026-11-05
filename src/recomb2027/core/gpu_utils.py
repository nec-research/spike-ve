#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     gpu_utils.py
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
"""GPU and memory management utilities.

This module provides utilities for GPU selection, memory management,
and batch size estimation for memory-intensive operations like ESM inference.
"""

import gc
import logging
import os
import subprocess
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def flush_cuda_cache(device: torch.device) -> None:
    """Clean CUDA cache aggressively.
    
    Args:
        device: PyTorch device to clean cache for
    """
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def get_gpu_free_memory(device: torch.device) -> int:
    """Get free GPU memory in MiB.
    
    Args:
        device: PyTorch CUDA device
        
    Returns:
        Free memory in MiB, or 0 if not a CUDA device
    """
    if device.type != 'cuda':
        return 0
    try:
        free, total = torch.cuda.mem_get_info(device)
        return free // (1024 * 1024)  # Convert to MiB
    except Exception:
        return 4000  # Fallback estimate


def estimate_esm_batch_size(
    seq_length: int,
    free_memory_mib: int,
    model_memory_mib: int = 5000,
    max_batch_size: int = 8,
) -> int:
    """Estimate safe batch size for ESM model based on sequence length and memory.
    
    ESM attention uses O(batch_size × seq_len² × hidden_dim) memory.
    For spike protein (~1273 aa), this can be massive.
    
    Args:
        seq_length: Length of sequences to process
        free_memory_mib: Available GPU memory in MiB
        model_memory_mib: Memory used by ESM model itself (default 5000 MiB)
        max_batch_size: Maximum batch size to return
    
    Returns:
        Safe batch size (at least 1, at most max_batch_size)
    """
    # Available memory for batching (leave 1GB buffer)
    available_mib = max(free_memory_mib - model_memory_mib - 1000, 500)
    
    # Empirical: ESM-1b uses roughly (seq_len^2 * 0.0015) MiB per sequence
    # For seq_len=1273: ~2.4 GiB per sequence in attention
    per_seq_memory_mib = (seq_length ** 2) * 0.0015
    
    # Calculate batch size with safety margin
    batch_size = max(1, int(available_mib / per_seq_memory_mib / 1.5))
    
    # Clamp to reasonable range
    return min(batch_size, max_batch_size)


def select_best_gpu() -> Tuple[Optional[str], int]:
    """Select the GPU with the most free memory.
    
    Uses nvidia-smi to query available GPUs and their free memory,
    then returns the ID of the GPU with the most free memory.
    
    Returns:
        Tuple of (gpu_id, free_memory_mib) or (None, 0) if no GPU found
    """
    try:
        smi_query = (
            "nvidia-smi --query-gpu=index,uuid,memory.free --format=csv,noheader,nounits"
        )
        result = subprocess.check_output(smi_query, shell=True)
        lines = [ln for ln in result.decode('utf-8').strip().split('\n') if ln.strip()]

        if not lines:
            logger.info("No GPUs found by nvidia-smi.")
            return None, 0

        # (physical index, uuid, free MiB) as reported by nvidia-smi
        physical = []
        for ln in lines:
            idx, uuid, free = [p.strip() for p in ln.split(',')]
            physical.append((int(idx), uuid, int(free)))

        # nvidia-smi always reports *physical* indices, but torch addresses devices by
        # their position in CUDA_VISIBLE_DEVICES. Restrict to the visible GPUs and return
        # the logical index (position in that list), otherwise e.g.
        # CUDA_VISIBLE_DEVICES=0 + "best GPU = physical #1" -> cuda:1 -> "invalid device ordinal".
        visible_env = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible_env is None:
            candidates = [(p[0], p[0], p[2]) for p in physical]  # (logical, physical, free)
        else:
            candidates = []
            for logical, token in enumerate(
                t.strip() for t in visible_env.split(',') if t.strip()
            ):
                match = None
                for idx, uuid, free in physical:
                    if token == str(idx) or (
                        token.startswith(("GPU-", "MIG-")) and uuid.startswith(token)
                    ):
                        match = (logical, idx, free)
                        break
                if match is not None:
                    candidates.append(match)
            if not candidates:
                logger.info("CUDA_VISIBLE_DEVICES=%r exposes no GPU known to nvidia-smi.",
                            visible_env)
                return None, 0

        best_logical, best_physical, free_mem = max(candidates, key=lambda c: c[2])
        note = (f" (physical #{best_physical}, CUDA_VISIBLE_DEVICES={visible_env})"
                if visible_env is not None else "")
        logger.info(f"GPUs found: {len(candidates)}. "
                   f"Auto-selecting GPU #{best_logical}{note} with {free_mem} MiB free.")
        print(f"GPUs found: {len(candidates)}. "
              f"Auto-selecting GPU #{best_logical}{note} with {free_mem} MiB free.")

        return str(best_logical), free_mem
        
    except Exception as e:
        logger.warning(f"Could not automatically select GPU: {e}. "
                      "Defaulting to PyTorch's choice.")
        print(f"Could not automatically select GPU: {e}. "
              "Defaulting to PyTorch's choice.")
        return None, 0


def estimate_decoder_batch_size(
    seq_length: int,
    free_memory_mib: int,
    *,
    d_model: int = 256,
    n_layers: int = 4,
    model_memory_mib: int = 256,
    max_batch_size: int = 256,
) -> int:
    """Estimate safe batch size for autoregressive sequence-decoder inference.

    Memory scales roughly with ``batch * seq_len * d_model * n_layers`` across
    autoregressive steps (much lighter than ESM, so batches can be larger).
    """
    available_mib = max(free_memory_mib - model_memory_mib - 512, 256)
    per_seq_mib = max(
        (seq_length * d_model * n_layers * 48) / (1024 * 1024),
        8.0,
    )
    batch_size = max(1, int(available_mib / per_seq_mib / 1.25))
    return min(batch_size, max_batch_size)


def parse_gpu_ids(
    *,
    gpus: Optional[str] = None,
    device: Optional[str] = None,
) -> list[int]:
    """Parse comma-separated GPU ids, or derive a single id from ``device``."""
    if gpus:
        ids = [int(x.strip()) for x in str(gpus).split(",") if x.strip()]
        if not ids:
            raise ValueError("--gpus was provided but no GPU ids were parsed.")
        return ids

    if device is None:
        if torch.cuda.is_available():
            return [int(torch.cuda.current_device())]
        return []

    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if dev.type != "cuda":
        return []
    if dev.index is None:
        return [int(torch.cuda.current_device())]
    return [int(dev.index)]


def validate_gpu_ids(gpu_ids: list[int]) -> None:
    if not gpu_ids:
        return
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA GPUs {gpu_ids} requested but CUDA is unavailable.")
    n = torch.cuda.device_count()
    bad = [g for g in gpu_ids if g < 0 or g >= n]
    if bad:
        raise RuntimeError(f"Invalid GPU id(s) {bad}; this machine has {n} GPU(s).")


def get_device(gpu_id: Optional[str] = None) -> torch.device:
    """Get PyTorch device, optionally specifying GPU ID.
    
    Args:
        gpu_id: Specific GPU ID to use, or None for auto-selection
        
    Returns:
        PyTorch device (CUDA if available, else CPU)
    """
    if gpu_id is not None and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Sets seeds for Python random, NumPy, and PyTorch (including CUDA).
    
    Args:
        seed: Random seed value
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    logger.info(f"Random seed set to {seed} for reproducibility")
