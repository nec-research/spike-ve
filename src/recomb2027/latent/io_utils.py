#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     io_utils.py
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
""" Input/ Output utils for experiments.
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from typing import TypedDict, Iterator, Dict, Any, Union, List
from numpy.typing import NDArray

import logging
logger = logging.getLogger(__name__)


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            #logging.FileHandler(log_dir / 'app.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    custom_logger = logging.getLogger(__name__)

    return custom_logger


def add_filehandler_to_logger(logger, filepath):
    """ Add file to write logs.
    """
    file_handler = logging.FileHandler(filepath)
    file_handler.setLevel(logging.INFO)            # optional
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    ))
    logger.addHandler(file_handler)


def write_preds_to_csv(pred, dataset, filepath):
    """
    Write predictions for each seris to CSV.
    """
    # Assume that ordering in the pred matches the order in the dataset
    x_gid, _ = dataset.get_xy_gids()

    assert x_gid.shape[0] == pred.shape[0]

    records_lst = []
    for idx in range(x_gid.shape[0]):
        logger.info(f"Prediciton index: {idx}.")
        record_dict = {
            'series_gids': list(x_gid[idx]),
            'predicted_encoding': pred[idx]
        }
        records_lst.append(record_dict)

    pd.DataFrame(records_lst).to_csv(filepath, index=False)


def write_preds_and_gt_to_csv(pred: NDArray, dataset, filepath):
    """
    Write predictions & ground truth for each seris to CSV.
    """
    # Assume that ordering in the pred matches the order in the dataset
    #x_gid, y_gid = dataset.get_xy_gids()
    x_gid, y_gid = dataset.get_xy_gids()
    y_true = dataset.get_y()

    #assert x_gid.shape[0] == pred.shape[0]
    assert len(x_gid) == pred.shape[0]

    records_lst = []
    #for idx in range(x_gid.shape[0]):
    for idx in range(len(x_gid)):
        #logger.info(f"Prediciton index: {idx}.")
        record_dict = {
            'series_gids': x_gid[idx],
            'gt_gid': y_gid[idx][-1],
            'gt_encoding': y_true[idx].tolist(),
            'predicted_encoding': pred[idx].tolist()
        }
        records_lst.append(record_dict)

    pd.DataFrame(records_lst).to_csv(filepath, index=False)
    logger.info(f"Output saved to: '{filepath}'")


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return super().default(obj)


def write_preds_and_gt_to_ndjson(
        pred: NDArray, dataset, filepath: Union[str, Path], compress=True
) -> None:
    """
    Write predictions & ground truth for each series to newline-delimited JSON.
    Each line is an object:
      {
        "series_gids": [...],
        "gt_gid": "...",
        "predicted_encoding": [...]
      }
    """
    #x_gid, y_gid = dataset.get_xy_gids()
    x_gid, y_gid = dataset.get_xy_gids()
    y_true = dataset.get_y()

    #assert x_gid.shape[0] == pred.shape[0]
    assert len(x_gid) == pred.shape[0]

    if compress:
        filepath = str(filepath)+".gz"
        opener = lambda p: gzip.open(p, "wt", encoding="utf-8")
    else:
        opener = lambda p: open(p, "w", encoding="utf-8")

    with opener(filepath) as f:
        for idx in range(len(x_gid)):
            obj = {
                "series_gids": x_gid[idx],
                "gt_gid": y_gid[idx][-1],  # or dataset-specific
                "gt_encoding": y_true[idx],
                "predicted_encoding": pred[idx]
            }
            f.write(json.dumps(obj, separators=(",", ":"), cls=NumpyEncoder))
            f.write("\n")

    logger.info(f"Output saved to: '{filepath}'")


def read_preds_and_gt_from_ndjson(
        filepath: Union[str, Path]
) -> List[Dict[str, Any]]:
    """
    Load full NDJSON file into memory.
    Returns a list of dicts, each matching write_preds_and_gt_to_ndjson.
    """
    path = Path(filepath)
    out: List[Dict[str, Any]] = []

    if path.suffix == ".gz":
        opener = lambda p: gzip.open(p, "rt", encoding="utf-8")
    else:
        opener = lambda p: open(p, "r", encoding="utf-8")

    with opener(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))

    return out