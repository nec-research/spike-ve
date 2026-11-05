#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     analyse_branching_sequences.py
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
"""Analyse whether the small embedding-space branch divergences correspond
to meaningful sequence-level changes.

1. Parse MSA to build genome_id -> protein sequence mapping.
2. For branching groups (shared prefixes), compute:
   - Hamming distance (amino-acid mutations) between branch target sequences
   - Embedding L2 distance for the same pairs
3. Global calibration: embedding L2 vs Hamming distance on random pairs.
"""

import re
import sys
from pathlib import Path
from recomb2027.core.paths import repo_root
from collections import defaultdict

import numpy as np

REPO_ROOT = repo_root(__file__)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recomb2027.data.series import SeriesEncodingsDataset


def parse_msa_fasta(fasta_path):
    """Parse MSA FASTA, returning {genome_id: sequence} dict.

    Handles multi-ID headers like:
        genome_id:2697049.123', '2697049.456|...
    """
    gid_to_seq = {}
    current_ids = []
    current_seq_parts = []

    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_ids and current_seq_parts:
                    seq = "".join(current_seq_parts)
                    for gid in current_ids:
                        gid_to_seq[gid] = seq

                header = line[1:]
                m = re.search(r"genome_id:([^|]+)", header)
                if m:
                    raw_ids = m.group(1)
                    ids = re.findall(r"(\d+\.\d+)", raw_ids)
                    current_ids = ids
                else:
                    current_ids = []
                current_seq_parts = []
            else:
                current_seq_parts.append(line)

    if current_ids and current_seq_parts:
        seq = "".join(current_seq_parts)
        for gid in current_ids:
            gid_to_seq[gid] = seq

    return gid_to_seq


def hamming_distance(seq1, seq2):
    """Hamming distance between two aligned sequences, ignoring gap-gap columns."""
    mismatches = 0
    comparable = 0
    for a, b in zip(seq1, seq2):
        if a == "-" and b == "-":
            continue
        comparable += 1
        if a != b:
            mismatches += 1
    return mismatches, comparable


def mutation_positions(seq1, seq2):
    """Return list of (position, aa1, aa2) for mismatches, ignoring gap-gap."""
    muts = []
    for i, (a, b) in enumerate(zip(seq1, seq2)):
        if a == "-" and b == "-":
            continue
        if a != b:
            muts.append((i, a, b))
    return muts


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None,
                        help="Path to train_dataset.npz (default: sampled_tau4 train series of the paper)")
    parser.add_argument("--label", default=None,
                        help="Label for this run (printed in output header)")
    args = parser.parse_args()

    msa_path = "demo_dataset/sars_cov2_spike/msa_s1.fasta"
    dataset_path = args.dataset or ""
    label = args.label or dataset_path
    print(f"{'=' * 70}")
    print(f"DATASET: {label}")
    print(f"{'=' * 70}\n")

    print("Parsing MSA...")
    gid_to_seq = parse_msa_fasta(msa_path)
    print(f"  {len(gid_to_seq)} genome IDs with sequences\n")

    print("Loading dataset...")
    ds = SeriesEncodingsDataset.load_from_file(dataset_path)
    gids = ds.sample_gids
    encs = ds.sample_encodings
    N, K, D = encs.shape
    print(f"  {N} series, length {K}, dim {D}")

    # Check coverage
    all_gids_in_dataset = set(gids.flatten())
    matched = all_gids_in_dataset & set(gid_to_seq.keys())
    print(f"  Genome IDs in dataset: {len(all_gids_in_dataset)}, "
          f"matched in MSA: {len(matched)} "
          f"({100 * len(matched) / len(all_gids_in_dataset):.1f}%)\n")

    # =========================================================
    # 1. CALIBRATION: embedding L2 vs Hamming distance
    # =========================================================
    print("=" * 70)
    print("CALIBRATION: Embedding L2 vs Hamming distance (random pairs)")
    print("=" * 70)

    rng = np.random.default_rng(42)
    n_cal_pairs = 5000
    cal_l2, cal_hamming = [], []

    flat_gids = gids.reshape(-1)
    flat_encs = encs.reshape(-1, D)

    attempts = 0
    while len(cal_l2) < n_cal_pairs and attempts < n_cal_pairs * 10:
        i, j = rng.choice(len(flat_gids), size=2, replace=False)
        g1, g2 = flat_gids[i], flat_gids[j]
        if g1 not in gid_to_seq or g2 not in gid_to_seq or g1 == g2:
            attempts += 1
            continue
        l2 = np.linalg.norm(flat_encs[i] - flat_encs[j])
        hd, _ = hamming_distance(gid_to_seq[g1], gid_to_seq[g2])
        cal_l2.append(l2)
        cal_hamming.append(hd)
        attempts += 1

    cal_l2 = np.array(cal_l2)
    cal_hamming = np.array(cal_hamming)

    # Bin by L2 distance
    bins = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5),
            (0.5, 0.8), (0.8, 1.2), (1.2, 2.0), (2.0, 5.0)]
    print(f"\n{'L2 range':>12s} | {'n_pairs':>8s} | {'mean mut':>8s} | "
          f"{'median mut':>10s} | {'max mut':>8s}")
    print("-" * 65)
    for lo, hi in bins:
        mask = (cal_l2 >= lo) & (cal_l2 < hi)
        if mask.sum() == 0:
            continue
        h = cal_hamming[mask]
        print(f"  [{lo:.1f}, {hi:.1f})"
              f" | {mask.sum():>8d}"
              f" | {h.mean():>8.1f}"
              f" | {np.median(h):>10.0f}"
              f" | {h.max():>8d}")

    # =========================================================
    # 2. BRANCH TARGET SEQUENCE ANALYSIS
    # =========================================================
    print("\n" + "=" * 70)
    print("BRANCH TARGET SEQUENCE ANALYSIS")
    print("=" * 70)

    for prefix_len in range(1, K):
        groups = defaultdict(list)
        for i in range(N):
            key = tuple(gids[i][:prefix_len])
            groups[key].append(i)
        branching_groups = {k: v for k, v in groups.items() if len(v) > 1}
        if not branching_groups:
            continue

        target_pos = prefix_len
        all_hamming = []
        all_l2 = []
        all_n_muts_detail = []

        for grp_indices in branching_groups.values():
            G = len(grp_indices)
            for ii in range(G):
                for jj in range(ii + 1, G):
                    idx_a, idx_b = grp_indices[ii], grp_indices[jj]
                    gid_a = gids[idx_a][target_pos]
                    gid_b = gids[idx_b][target_pos]

                    if gid_a == gid_b:
                        continue
                    if gid_a not in gid_to_seq or gid_b not in gid_to_seq:
                        continue

                    hd, comp = hamming_distance(gid_to_seq[gid_a], gid_to_seq[gid_b])
                    l2 = np.linalg.norm(encs[idx_a, target_pos] - encs[idx_b, target_pos])
                    all_hamming.append(hd)
                    all_l2.append(l2)

        if not all_hamming:
            continue

        h_arr = np.array(all_hamming)
        l2_arr = np.array(all_l2)

        print(f"\nPrefix len {prefix_len} (divergent target pairs: {len(h_arr)}):")
        print(f"  Mutations:  mean={h_arr.mean():.1f}, median={np.median(h_arr):.0f}, "
              f"std={h_arr.std():.1f}, max={h_arr.max()}")
        print(f"  Embed. L2:  mean={l2_arr.mean():.4f}, median={np.median(l2_arr):.4f}")

        # Distribution of mutation counts
        mut_bins = [0, 1, 2, 3, 5, 10, 20, 50, 100, 500]
        print(f"  Mutation distribution:")
        for bi in range(len(mut_bins) - 1):
            lo, hi = mut_bins[bi], mut_bins[bi + 1]
            count = ((h_arr >= lo) & (h_arr < hi)).sum()
            pct = 100 * count / len(h_arr)
            if count > 0:
                print(f"    {lo:>3d}-{hi:<3d} mutations: {count:>6d} pairs ({pct:>5.1f}%)")
        count = (h_arr >= mut_bins[-1]).sum()
        if count > 0:
            pct = 100 * count / len(h_arr)
            print(f"    {mut_bins[-1]:>3d}+   mutations: {count:>6d} pairs ({pct:>5.1f}%)")

    # =========================================================
    # 3. DETAILED EXAMPLES: biggest branching groups at prefix_len = K-1
    # =========================================================
    detail_prefix = K - 1
    detail_target = detail_prefix  # the diverging position
    print("\n" + "=" * 70)
    print(f"DETAILED EXAMPLES: Largest branching groups (prefix_len={detail_prefix})")
    print("=" * 70)

    groups = defaultdict(list)
    for i in range(N):
        key = tuple(gids[i][:detail_prefix])
        groups[key].append(i)
    branching_groups = {k: v for k, v in groups.items() if len(v) > 1}

    sorted_groups = sorted(branching_groups.items(), key=lambda x: -len(x[1]))

    for grp_key, grp_indices in sorted_groups[:5]:
        print(f"\n  Group prefix: ...{grp_key[-1][-8:]}")
        print(f"  Size: {len(grp_indices)} series")

        target_gids = [gids[idx][detail_target] for idx in grp_indices]
        unique_targets = list(set(target_gids))
        print(f"  Unique target genome IDs: {len(unique_targets)}")

        if len(unique_targets) <= 1:
            print("  (all targets identical)")
            continue

        for ii in range(min(len(unique_targets), 3)):
            for jj in range(ii + 1, min(len(unique_targets), 3)):
                g1, g2 = unique_targets[ii], unique_targets[jj]
                if g1 not in gid_to_seq or g2 not in gid_to_seq:
                    continue
                hd, comp = hamming_distance(gid_to_seq[g1], gid_to_seq[g2])
                muts = mutation_positions(gid_to_seq[g1], gid_to_seq[g2])

                idx1 = next(idx for idx in grp_indices if gids[idx][detail_target] == g1)
                idx2 = next(idx for idx in grp_indices if gids[idx][detail_target] == g2)
                l2 = np.linalg.norm(encs[idx1, detail_target] - encs[idx2, detail_target])

                print(f"\n    {g1[-8:]} vs {g2[-8:]}: "
                      f"{hd} mutations, L2={l2:.4f}")
                if muts:
                    for pos, a, b in muts[:15]:
                        print(f"      pos {pos:4d}: {a} -> {b}")
                    if len(muts) > 15:
                        print(f"      ... and {len(muts) - 15} more")


if __name__ == "__main__":
    main()
