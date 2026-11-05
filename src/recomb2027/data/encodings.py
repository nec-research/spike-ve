#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     encodings.py
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
import pandas as pd
import logging

from recomb2027.core import constants
from recomb2027.data.manager import DataManager


class EncodingMatcher:
    def __init__(self, data_manager):
        self.dmg = data_manager
        self.df_encodings = None

    def match_from_dir(self, encodings_dir, encoding_key="encodings", key_fields=None):
        """ Load encodings from a directory and match them to the uploaded sequences by labels.
                Output the full dataset as AnnData object."""
        if key_fields is None:
            key_fields = ['genome_id', 'protein']
        if key_fields is None:
            #key_fields = self.default_keys
            print(f"No key fields specified.")
            return

            # load encoding files from directory
        print("Read encodings from directory")
        labels, encodings = self.dmg.read_seq_encodings_from_dir(
            encodings_dir,
            encoding_key=encoding_key
        )
        # parse labels
        print("Parse labels")
        data_dicts = []
        for label, enc in list(zip(labels, encodings)):
            label = label.replace("'", "")
            label_fields = label.split(constants.FASTA_FILE_FIELD_SEP)
            data_dict = dict(
                map(
                    str.strip,
                    field.split(constants.FASTA_FILE_VALUE_SEP, 1)
                ) for field in label_fields if constants.FASTA_FILE_VALUE_SEP in field
            )
            # Ensure genome_id exists by extracting from label when missing.
            if constants.FASTA_FILE_FIELD_GENOME_ID not in data_dict:
                import re

                m = re.search(r'(?:genome_id|sequence_id|seqid|id|accession)[:=]\s*([^\|\s,;]+)', label, flags=re.IGNORECASE)
                if m:
                    data_dict[constants.FASTA_FILE_FIELD_GENOME_ID] = m.group(1)
                else:
                    first = re.split(r'[\|\s,;]+', label.strip())[0]
                    if constants.FASTA_FILE_VALUE_SEP in first:
                        parts = first.split(constants.FASTA_FILE_VALUE_SEP, 1)
                        data_dict[constants.FASTA_FILE_FIELD_GENOME_ID] = parts[1].strip()
                    else:
                        data_dict[constants.FASTA_FILE_FIELD_GENOME_ID] = first

            # attach encodings and split multi-valued genome_id fields
            data_dict[encoding_key] = enc  # add encodings
            genome_ids = str(data_dict[constants.FASTA_FILE_FIELD_GENOME_ID]).split(constants.FASTA_FILE_MULTIVALUED_SEP)
            if len(genome_ids) > 1:
                # split multiple genome ids into separate rows
                data_dict[constants.FASTA_FILE_FIELD_GENOME_ID] = genome_ids[0].strip()
                for genome_id in genome_ids[1:]:
                    data_dict_ = data_dict.copy()
                    data_dict_[constants.FASTA_FILE_FIELD_GENOME_ID] = genome_id.strip()
                    data_dicts.append(data_dict_)
            data_dicts.append(data_dict)
        # convert into dataframe
        self.df_encodings = pd.DataFrame(data_dicts)
        logging.info(f"Samples in total: {len(self.dmg.df_labels)}")
        # match with current labels
        self.df_encodings = self.dmg.df_labels[key_fields].drop_duplicates().merge(
            self.df_encodings[key_fields + [encoding_key]].drop_duplicates(subset=key_fields),
            on=key_fields,
            how='inner'
        )
        logging.info(f"Samples with encodings: {len(self.df_encodings)}")

    @staticmethod
    def read_from_dir(encodings_dir, encoding_key="encodings"):
        """ Read encodings from a directory and return as a dataframe."""

        logging.info(f"Load encodings from directory: '{encodings_dir}'")
        labels, encodings = DataManager.read_seq_encodings_from_dir(
            encodings_dir,
            encoding_key=encoding_key,
            verbose=False
        )

        logging.info("Parse encoding records.")

        data_dicts = []
        for label, enc in list(zip(labels, encodings)):
            label = label.replace("'", "")
            label_fields = label.split(constants.FASTA_FILE_FIELD_SEP)
            data_dict = dict(
                map(
                    str.strip,
                    field.split(constants.FASTA_FILE_VALUE_SEP, 1)
                ) for field in label_fields if constants.FASTA_FILE_VALUE_SEP in field
            )
            # Ensure genome_id exists by extracting from label when missing.
            if constants.FASTA_FILE_FIELD_GENOME_ID not in data_dict:
                import re

                m = re.search(r'(?:genome_id|sequence_id|seqid|id|accession)[:=]\s*([^\|\s,;]+)', label, flags=re.IGNORECASE)
                if m:
                    data_dict[constants.FASTA_FILE_FIELD_GENOME_ID] = m.group(1)
                else:
                    first = re.split(r'[\|\s,;]+', label.strip())[0]
                    if constants.FASTA_FILE_VALUE_SEP in first:
                        parts = first.split(constants.FASTA_FILE_VALUE_SEP, 1)
                        data_dict[constants.FASTA_FILE_FIELD_GENOME_ID] = parts[1].strip()
                    else:
                        data_dict[constants.FASTA_FILE_FIELD_GENOME_ID] = first

            data_dict[encoding_key] = enc  # add encodings
            genome_ids = str(data_dict[constants.FASTA_FILE_FIELD_GENOME_ID]).split(constants.FASTA_FILE_MULTIVALUED_SEP)
            if len(genome_ids) > 1:
                # split multiple genome ids into separate rows
                data_dict[constants.FASTA_FILE_FIELD_GENOME_ID] = genome_ids[0].strip()
                for genome_id in genome_ids[1:]:
                    data_dict_ = data_dict.copy()
                    data_dict_[constants.FASTA_FILE_FIELD_GENOME_ID] = genome_id.strip()
                    data_dicts.append(data_dict_)
            data_dicts.append(data_dict)

        # Convert into dataframe
        df_encodings = pd.DataFrame(data_dicts)
        # Ensure there is a usable genome_id column. Some .pt files may store the id
        # under a different key (e.g. "id", "seqid", "accession", or only in "description").
        # Provide robust fallbacks so downstream callers can reliably call .set_index('genome_id').
        if constants.FASTA_FILE_FIELD_GENOME_ID not in df_encodings.columns:
            cols = list(df_encodings.columns)
            # look for common alternative column names
            alt_names = ['id', 'ID', 'seqid', 'sequence_id', 'accession', 'name', 'identifier', 'index', 'genome']
            candidate = None
            for c in cols:
                if c in alt_names or c.lower() in alt_names:
                    candidate = c
                    break
            # also prefer any column containing "id" as a substring
            if candidate is None:
                for c in cols:
                    if 'id' in c.lower() and c.lower() != 'description':
                        candidate = c
                        break
            if candidate is not None:
                df_encodings = df_encodings.rename(columns={candidate: constants.FASTA_FILE_FIELD_GENOME_ID})
            else:
                # try to extract genome_id from description field if present
                if 'description' in df_encodings.columns:
                    import re

                    def _extract_id(desc: str):
                        if not isinstance(desc, str):
                            return None
                        # look for explicit key patterns like genome_id:XYZ or id:XYZ
                        m = re.search(r'(?:genome_id|sequence_id|seqid|id|accession)[:=]\s*([^\|\s,;]+)', desc, flags=re.IGNORECASE)
                        if m:
                            return m.group(1)
                        # fallback: take first token before whitespace or pipe
                        token = re.split(r'[\|\s,;]+', desc.strip())[0]
                        return token if token != '' else None

                    df_encodings[constants.FASTA_FILE_FIELD_GENOME_ID] = df_encodings['description'].apply(_extract_id)
                else:
                    # As a last resort, create genome_id from the first available column
                    if len(cols) > 0:
                        df_encodings[constants.FASTA_FILE_FIELD_GENOME_ID] = df_encodings[cols[0]].astype(str)

        logging.info(f"Encodings for {len(df_encodings)} samples loaded.")
        return df_encodings