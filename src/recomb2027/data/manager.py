#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     manager.py
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
Module that contains data management class.
"""

import os
import re
from pathlib import Path
import collections.abc

import numpy as np
import pandas as pd
import torch

import dendropy
from tqdm import tqdm

import recomb2027.core.constants as constants

import logging
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
from Bio.Phylo.TreeConstruction import DistanceCalculator

class DataManager(torch.utils.data.Dataset):
    """ Class contains data input-output and basic preprocessing utils.
    """

    # ─────────────────────────────
    # Class methods
    # ─────────────────────────────

    @classmethod
    def read_seq_encodings_from_dir(cls, input_dir, encoding_key="encodings", max_samples=None, verbose=False):
        input_dir = Path(input_dir)
        file_list = [f for f in os.listdir(input_dir) if f.endswith(".pt")]
        file_list.sort()  # ensures consistent order

        if max_samples is not None:
            file_list = file_list[:max_samples]

        if verbose:
            print(f"Reading {len(file_list)} encodings from folder: {input_dir}")

        labels = []
        encodings = []

        # Load files in a single pass
        #for idx, fname in enumerate(file_list, 1):
        for idx, fname in enumerate(tqdm(file_list, desc="Reading files", unit="file", ncols=100), 1):
            if verbose:
                print(f"Reading file {idx}/{len(file_list)}: {fname}")
            try:
                data = torch.load(input_dir / fname, map_location="cpu")  # load once
                labels.append(data["label"])
                enc = data[encoding_key].numpy().T
                encodings.append(enc)
            except:
                continue

        # Preallocate final array
        max_len = max(map(len, encodings))
        encodings_ = np.zeros((len(encodings), max_len), dtype=encodings[0].dtype)
        for i, arr in enumerate(encodings):
            encodings_[i, :len(arr)] = arr

        logging.info(f"{len(encodings_)} encodings loaded (shape: {encodings_.shape}).")
        return labels, encodings_
    

    def __init__(
            self,
            data_dir=None,
            val_sep=constants.FASTA_FILE_VALUE_SEP,
            field_sep=constants.FASTA_FILE_FIELD_SEP,
            seq_col=constants.FASTA_FILE_FIELD_SEQ,
            timestamp_field='collection_date'
    ):
        super().__init__()
        # variable to read
        self.data_dir = data_dir
        self.val_sep = val_sep
        self.field_sep = field_sep
        self.seq_col = seq_col
        self.timestamp_field = timestamp_field
        # other variables with default values
        self.num_sequences = 0
        self.alignment_length = 0
        self.seq_path = None
        self.sequence_labels = None
        self.sequence_label_dicts = None
        self.sequence_strs = None
        self.class_labels = None
        self.df_labels = None
        self.phylotree = None
        self.is_msa = True
        self.ncbi_format = False

    # ─────────────────────────────
    # Public interface
    # ─────────────────────────────

    def load_fasta_file(self, fasta_file, remove_duplicates=True, ncbi_format=False):
        if self.data_dir:
            self.seq_path = os.path.join(self.data_dir, fasta_file)
        else:
            self.seq_path = fasta_file

        self.ncbi_format = ncbi_format
        self.sequence_labels, self.sequence_strs = [], []

        logging.info(f"Process file '{fasta_file}'...")

        cur_seq_label = None
        buf = []
        with open(self.seq_path, "r") as infile:
            for line_idx, line in enumerate(infile):
                if line.startswith(">"):  # label line
                    if cur_seq_label is not None:
                        self.sequence_labels.append(cur_seq_label)
                        self.sequence_strs.append("".join(buf))
                        cur_seq_label = None
                        buf = []
                    line = line[1:].strip()
                    if len(line) > 0:
                        cur_seq_label = line
                    else:
                        cur_seq_label = f"seqnum{line_idx:09d}"
                else:  # sequence line
                    buf.append(line.strip())
        if cur_seq_label is not None:
            self.sequence_labels.append(cur_seq_label)
            self.sequence_strs.append("".join(buf))

        self._parse_labels()

        if remove_duplicates:
            seen_labels = set()
            duplicate_ids = []
            for idx, item in enumerate(self.sequence_labels):
                if item in seen_labels:
                    print(f"Duplicate item removed: {item}")
                    duplicate_ids.append(idx)
                else:
                    seen_labels.add(item)
            # remove duplicates by index from lists
            duplicate_ids = sorted(duplicate_ids, reverse=True)
            for idx in duplicate_ids:
                if 0 <= idx <= len(self.sequence_labels):
                    del self.sequence_labels[idx]
                    del self.sequence_strs[idx]
                    del self.sequence_label_dicts[idx]

            assert len(set(self.sequence_labels)) == len(self.sequence_labels), \
                "Found duplicate sequence labels"

        assert len(self.sequence_labels) == len(self.sequence_label_dicts), \
            "Some labels couldn't be parsed"

        self.num_sequences = len(self.sequence_strs)
        logging.info(f"{self.num_sequences} sequences loaded.")

        return list(zip(self.sequence_labels, self.sequence_strs))

    def load_msa_from_file(self, msa_fasta_file):
        """ Load dataset from fasta file. Protein sequences should be aligned into MSA.
        """
        self.load_fasta_file(msa_fasta_file)
        if self.num_sequences < 2:
            raise ValueError("Alignment contains less than 2 sequences.")

        self.alignment_length = len(self.sequence_strs[0])
        logging.info(f"{self.num_sequences} sequences of length {self.alignment_length} loaded from MSA.")

        self._labels_to_dataframe()

    def add_metadata_from_csv(self, metadata_file, on_key=('genome_id', 'protein'), cols_to_extract=None, sep=",", dtype=str):
        """ Add extra metadata from a CSV file to the uploaded sequences.
        """
        df_metadata = pd.read_csv(metadata_file, sep=sep, dtype=dtype)

        if cols_to_extract is not None and isinstance(cols_to_extract, collections.abc.Iterable):
            df_metadata = df_metadata[list(set(cols_to_extract) | (set(on_key) & set(df_metadata.columns)))]

        self.df_labels = self.df_labels.merge(
            df_metadata, how="left", on=list(set(on_key) & set(df_metadata.columns)))

    def add_phylogenetic_tree(self, tree_file, schema="newick"):
        """ Load phylogenetic tree.
        """
        self.phylotree = dendropy.Tree.get_from_path(src=tree_file, schema=schema)
        logging.info(f"Phylogenetic tree loaded from: '{tree_file}'.")

    def get_phylotree_metadata(self, taxon_col='genome_id', cols_to_return=None):
        """ Align sequence labels with phylogenetic tree leafs and return as a dataframe. """
        if self.phylotree is None:
            raise ValueError("Phylogenetic tree not found.")

        # Extract leaf labels
        leaf_nodes = self.phylotree.leaf_nodes()
        if len(leaf_nodes) < 1:
            return None

        leaf_prefix = leaf_nodes[0].taxon.label.split("-")[0]
        leaf_labels = {leaf.taxon.label.split("-")[1] for leaf in leaf_nodes}
        # Filter the dataframe
        filtered_df = self.df_labels[self.df_labels[taxon_col].isin(leaf_labels)].copy()
        #filtered_df.insert(0, 'taxon_label', leaf_prefix + filtered_df[taxon_col].astype(str))
        filtered_df.insert(0, 'taxon_label', taxon_col+'-'+filtered_df[taxon_col].astype(str))

        logging.info(f"{len(leaf_labels)} leaf in a tree.")
        logging.info(f"{len(filtered_df)} rows with metadata.")
        if len(leaf_labels) > len(filtered_df):
            missing_ids = list(set(leaf_labels) - set(filtered_df[taxon_col].tolist()))
            logging.info(f"No metadata for leafs: {missing_ids}")
        if len(leaf_labels) < len(filtered_df):
            missing_ids = list(set(filtered_df[taxon_col].tolist()) - set(leaf_labels))
            logging.info(f"No leafs for rows: {missing_ids}")

        if cols_to_return is not None:
            return filtered_df[['taxon_label']+cols_to_return]

        return filtered_df

    def load_multiple_fasta_from_dir(self, fasta_dir):
        """ Load dataset from multiple fasta files stored in a directory.
        We will need to do a pairwise alignment every time before calculating evolutionary distances
        between each pair sequences in this case."""
        self.is_msa = False
        # TBD: implement method
        pass

    def get_labels_and_sequences(self):
        """ Return labels and sequences as two independent lists.
        """
        return self.sequence_labels, self.sequence_strs

    def set_class_labels(self, class_labels):
        """ Assign class labels.
        """
        if class_labels is None:
            print("Error: No labels to assign!")
            return
        self.class_labels = class_labels
        assert len(self.sequence_labels) == len(self.class_labels), \
            "Size of class labels list do not match number of samples"

    def get_dataset(self):
        """ Return (label, sequence) list of tuples."
        """
        if self.class_labels is not None:
            data = zip(self.class_labels, self.sequence_strs)
        else:
            data = zip(self.sequence_labels, self.sequence_strs)

        return list(data)

    def to_dataframe(self, split_by_genome_id=False, keep_seq_col=True):
        """ Convert data to dataframe.

        :param split_by_genome_id: splits records with multiple genome_id into separate rows in the dataframe
        :keep_seq_col: assign sequences into a separate column of the returned dataframe
        """
        self._labels_to_dataframe(
            split_by_genome_id=split_by_genome_id,
            keep_seq_col=keep_seq_col,
            ncbi_format=self.ncbi_format
        )

        return self.df_labels

    def from_dataframe(self, df, seq_col='seq', val_sep=':'):
        """ Load label and sequence data from dataframe.
        """
        self.sequence_labels, self.sequence_strs = [], []
        self.sequence_label_dicts = None
        label_col = list(df.columns)
        label_col.remove(seq_col)
        for index, row in df.iterrows():
            label_text = ""
            for col in label_col:
                label_text += f"{col}{val_sep}{row[col]}|"
            if len(label_text) > 1:
                label_text = label_text[:-1]
            self.sequence_labels.append(label_text)
            self.sequence_strs.append(row[seq_col])

        return self.sequence_labels, self.sequence_strs

    def get_batch_indices(self, toks_per_batch, extra_toks_per_seq=0):
        """ Helper function for some pyTorch model usage.
        """
        sizes = [(len(s), i) for i, s in enumerate(self.sequence_strs)]
        sizes.sort()
        batches = []
        buf = []
        max_len = 0

        def _flush_current_buf():
            nonlocal max_len, buf
            if len(buf) == 0:
                return
            batches.append(buf)
            buf = []
            max_len = 0

        for sz, i in sizes:
            sz += extra_toks_per_seq
            if max(sz, max_len) * (len(buf) + 1) > toks_per_batch:
                _flush_current_buf()
            max_len = max(max_len, sz)
            buf.append(i)

        _flush_current_buf()

        return batches

    def write_fasta(self, fasta_file):
        if (self.sequence_labels is None) or (self.sequence_strs is None):
            print("Nothing to write.")
            return

        assert len(self.sequence_strs) == len(self.sequence_labels), \
            "Sizes of label array and sequence array do not match"
        # write into FASTA file
        with open(fasta_file, 'w') as fout:
            print(f"Writing to file:\n '{fasta_file}'.")
            for idx, label in enumerate(self.sequence_labels):
                # write
                fout.write(f">{self.sequence_labels[idx]}\n{self.sequence_strs[idx]}\n")
                fout.flush()
                if idx % 101 == 0:
                    print(f"Record {idx} processed")
            print(f"Record {len(self.sequence_labels)-1} processed")
            print(f"In total {len(self.sequence_labels)} sequences written.")

    # ─────────────────────────────
    # Private helpers
    # ─────────────────────────────

    def __len__(self):
        return len(self.sequence_labels)

    def __getitem__(self, idx):
        if self.class_labels is not None:
            return self.class_labels[idx], self.sequence_strs[idx]
        else:
            return None

    def _parse_labels(self, ncbi_format=False):
        """ Trim each field in the labels and join again into a single string.
        """
        self.sequence_label_dicts = []
        for seq_label in self.sequence_labels:
            label_dict = {}
            if not ncbi_format:
                # the default header looks like "field1_name:field1_value | field2_name:field2_value | ..."
                fields = seq_label.split("|")
                for field in fields:
                    filed_list = field.split(":")
                    key, value = filed_list[0], "".join(filed_list[1:])
                    label_dict[key] = value.strip().replace("'", "")
            else:
                # in case of ncbi standard, genome_id is everything before the first space
                fields = re.split(r"[ \t]", seq_label)
                label_dict[constants.FASTA_FILE_FIELD_GENOME_ID] = fields[0]
                if len(fields) > 1:
                    label_dict[constants.FASTA_FILE_FIELD_DESCRIPTION] = "".join(fields[1:]).strip().replace("'", "")

            self.sequence_label_dicts.append(label_dict)

    def _labels_to_dataframe(self, split_by_genome_id=True, keep_seq_col=False, ncbi_format=False):
        """ Convert labels to dataframe, check that the timestamp field exists.
        If split_by_genome_id is True, parse rows with multiple genome ids into single entries with unique genome ids.
        """
        # convert label fields and sequences to a list of dictionaries
        data_dicts = []
        for label, seq in list(zip(self.sequence_labels, self.sequence_strs)):
            label = label.replace("'", "")
            # parse labels field into separate fields
            if ncbi_format:
                label_fields = re.split(r"[ \t]", label)
                data_dict = {}
                data_dict[constants.FASTA_FILE_FIELD_GENOME_ID] = label_fields[0]
                if len(label_fields) > 1:
                    data_dict[constants.FASTA_FILE_FIELD_DESCRIPTION] = "".join(label_fields[1:]).strip()
            else:
                label_fields = label.split(constants.FASTA_FILE_FIELD_SEP)
                data_dict = dict(
                    map(
                        str.strip,
                        field.split(constants.FASTA_FILE_VALUE_SEP, 1)
                    ) for field in label_fields if constants.FASTA_FILE_VALUE_SEP in field)
                # add description field if description not in label
                if 'description' not in data_dict:
                    data_dict['description'] = label

            # add sequence column when needed
            if keep_seq_col:
                data_dict[self.seq_col] = seq

            # special case when genome_id contains multiple values
            if constants.FASTA_FILE_FIELD_GENOME_ID in data_dict:
                genome_ids = str(data_dict[constants.FASTA_FILE_FIELD_GENOME_ID]).split(
                        constants.FASTA_FILE_MULTIVALUED_SEP)
                if split_by_genome_id and len(genome_ids) > 1:
                    data_dict[constants.FASTA_FILE_FIELD_GENOME_ID] = genome_ids[0].strip()
                    for genome_id in genome_ids[1:]:
                        data_dict_ = data_dict.copy()
                        data_dict_[constants.FASTA_FILE_FIELD_GENOME_ID] = genome_id.strip()
                        data_dicts.append(data_dict_)
            data_dicts.append(data_dict)

        self.df_labels = pd.DataFrame(data_dicts)
        # Ensure there is a usable genome_id column. Some FASTA headers may
        # not include an explicit 'genome_id' field; provide fallbacks so other
        # code can reliably merge on 'genome_id'.
        if constants.FASTA_FILE_FIELD_GENOME_ID not in self.df_labels.columns:
            cols = list(self.df_labels.columns)
            alt_names = ['id', 'ID', 'seqid', 'sequence_id', 'accession', 'name', 'identifier', 'index', 'genome']
            candidate = None
            for c in cols:
                if c in alt_names or c.lower() in alt_names:
                    candidate = c
                    break
            if candidate is not None:
                self.df_labels = self.df_labels.rename(columns={candidate: constants.FASTA_FILE_FIELD_GENOME_ID})
            else:
                # try to extract from description if available
                if 'description' in self.df_labels.columns:
                    def _extract_id(desc: str):
                        if not isinstance(desc, str):
                            return None
                        m = re.search(r'(?:genome_id|sequence_id|seqid|id|accession)[:=]\s*([^\|\s,;]+)', desc, flags=re.IGNORECASE)
                        if m:
                            return m.group(1)
                        token = re.split(r'[\|\s,;]+', desc.strip())[0]
                        return token if token != '' else None

                    self.df_labels[constants.FASTA_FILE_FIELD_GENOME_ID] = self.df_labels['description'].apply(_extract_id)
                else:
                    # last resort: use row index as a string (guarantees column exists)
                    self.df_labels[constants.FASTA_FILE_FIELD_GENOME_ID] = self.df_labels.index.astype(str)


def load_fasta_by_header_field(
        fasta_path,
        key_field: str,
) -> dict:
    """Load a FASTA file and index sequences by a pipe-delimited header field."""
    dmg = DataManager()
    dmg.load_fasta_file(str(fasta_path), remove_duplicates=False)

    records = {}
    for label_dict, seq in zip(dmg.sequence_label_dicts, dmg.sequence_strs):
        if key_field not in label_dict:
            description = label_dict.get("description", "")[:120]
            raise ValueError(
                f"FASTA record in '{fasta_path}' is missing '{key_field}' "
                f"in header: {description}"
            )
        key = str(label_dict[key_field]).strip()
        records[key] = seq
    return records


def load_fasta_by_genome_id_field(
        fasta_path,
        genome_id_field: str,
) -> dict:
    """Load a FASTA file and index sequences by a genome-id header field."""
    return load_fasta_by_header_field(fasta_path, genome_id_field)


def _bio_validation_fasta_index(row, source_sequence_index: int) -> str:
    """Map a validation CSV row to the corresponding FASTA ``index`` header value."""
    row_index = row.get("index")
    if row_index is None or (isinstance(row_index, float) and pd.isna(row_index)):
        return str(source_sequence_index)
    return str(int(row_index) - 1)


def load_bio_validation_pair(config) -> tuple:
    """Load source/target aligned sequences for bio-validation experiments.

    Genome IDs and lineages come from ``validation_dataset_*`` CSV columns
    ``in_seqid``, ``out_seqid``, ``Input``, and ``Output``. Sequences are
    resolved from separate source and target alignment FASTA files.
    """
    df = pd.read_csv(config.dataset_filepath)
    df.columns = [col.lstrip("\ufeff") for col in df.columns]

    required_columns = {"in_seqid", "out_seqid", "Input", "Output"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Bio-validation CSV must contain columns: {required_columns}"
        )

    if config.source_fasta_filepath is None or config.target_fasta_filepath is None:
        raise ValueError(
            "Bio-validation format requires paths.source_fasta_filepath and "
            "paths.target_fasta_filepath."
        )

    row = df.iloc[config.source_sequence_index]
    source_id = str(row["in_seqid"]).strip()
    target_id = str(row["out_seqid"]).strip()
    source_lineage = row["Input"]
    target_lineage = row.get("Output")
    fasta_index = _bio_validation_fasta_index(row, config.source_sequence_index)

    source_index = load_fasta_by_header_field(
        config.source_fasta_filepath,
        "index",
    )
    target_index = load_fasta_by_header_field(
        config.target_fasta_filepath,
        "index",
    )

    if fasta_index not in source_index:
        raise ValueError(
            f"FASTA index '{fasta_index}' not found in "
            f"{config.source_fasta_filepath}"
        )
    if fasta_index not in target_index:
        raise ValueError(
            f"FASTA index '{fasta_index}' not found in "
            f"{config.target_fasta_filepath}"
        )

    source_sequence = source_index[fasta_index]
    target_sequence = target_index[fasta_index]

    logging.info(
        "Loaded bio-validation pair index=%s source=%s (%s, len=%d) "
        "target=%s (%s, len=%d)",
        config.source_sequence_index,
        source_id,
        source_lineage,
        len(source_sequence),
        target_id,
        target_lineage,
        len(target_sequence),
    )

    return (
        source_id,
        source_sequence,
        source_lineage,
        target_id,
        target_sequence,
        None,
    )


def get_sequence_from_fasta(fasta_file: str, sequence_id: str) -> str:
    """ Find sequence by given sequence_id. Sequence header must be in the NCBI standard.
    """
    fasta_path = Path(fasta_file)
    if not fasta_path.is_file():
        print(f"Error: The file '{fasta_file}' was not found.")
        return None

    print(f"  > Searching for ID '{sequence_id}' in '{fasta_file}'...")

    try:
        for record in SeqIO.parse(fasta_path, "fasta"):
            if record.id == sequence_id:
                print(f"✓ Found direct match for ID '{sequence_id}'.")
                return str(record.seq)
            header_parts = record.description.split('|')
            for part in header_parts:
                if part.startswith("genome_id:"):
                    extracted_id = part.split(':', 1)[1]
                    if extracted_id == sequence_id:
                        print(f"✓ Found embedded match for ID '{sequence_id}'.")
                        return str(record.seq)
    except Exception as e:
        print(f"Error: An issue occurred while parsing the FASTA file: {e}")
        return None

    print(f"Error: Sequence ID '{sequence_id}' was not found in the file.")
    return None


def calc_pairwise_phylo_distance(seq1, seq2):
    """ Return distance between the two given sequences.
    """
    seq1_record = SeqRecord(Seq(seq1), id='seq1')
    seq2_record = SeqRecord(Seq(seq2), id='seq2')
    align = MultipleSeqAlignment([seq1_record, seq2_record])
    calculator = DistanceCalculator('blosum62')
    dm_pairwise = calculator.get_distance(align)


    return dm_pairwise.matrix[1][0]