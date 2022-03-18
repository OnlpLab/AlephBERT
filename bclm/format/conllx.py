from copy import copy
import logging
import pandas as pd
from .format_utils import split_sentences, lattice_fields

# SPMRL Lattice format is described in "Input Formats" section of the SPMRL14 shared task description:
# http://dokufarm.phil.hhu.de/spmrl2014/doku.php?id=shared_task_description
# CPOSTAG = Coarse Part of Speech Tag (Canonical representation of the POS tag shared across all languages)
# FPOSTAG = Fine grained Part of Speech tag (language specific)
LATTICE_COLUMN_NAMES = ['START', 'END', 'FORM', 'LEMMA', 'CPOSTAG', 'FPOSTAG', 'FEATS', 'TOKEN_ID']


# We transform the SPMRL lattice format into an enriched normalized format including sentence id, token and gold flag
# ['sent_id', 'from_node_id', 'to_node_id', 'form', 'lemma', 'tag', 'feats', 'token_id', 'token', 'is_gold']
def _build_conllx_sample_rows(sent_id, lattice, tokens, is_gold):
    rows = []
    for morpheme in lattice:
        sample_row = copy(morpheme)
        sample_row[0] = int(sample_row[0])  # Start
        sample_row[1] = int(sample_row[1])  # End
        del sample_row[5]  # Remove FPOSTAG, keep CPOSTAG
        sample_row[-1] = int(sample_row[-1])  # TokenId
        sample_row.append(tokens[sample_row[-1]])  # Token
        sample_row.append(is_gold)  # Gold Flag
        sample_row.insert(0, sent_id)  # SentenceId
        rows.append(sample_row)
    return rows


def _get_conllx_partition_df(lattice_sentences, token_sentences, is_gold):
    partition = []
    for i, (lattice, tokens) in enumerate(zip(lattice_sentences, token_sentences)):
        sent_id = i + 1
        tokens = {j + 1: t for j, t in enumerate(tokens)}
        lattice = [line.split() for line in lattice]
        partition.extend(_build_conllx_sample_rows(sent_id, lattice, tokens, is_gold))
    return pd.DataFrame(partition, columns=lattice_fields)


def load_conllx(tb_root_path, partition, tb_name, ma_name):
    treebank = {}
    for partition_type in partition:
        file_name = f'{partition_type}_{tb_name}'.lower()
        logging.info(f'loading {file_name} dataset')
        if ma_name is not None:
            lattices_path = tb_root_path / tb_name / f'{file_name}.lattices'
        else:
            lattices_path = tb_root_path / tb_name / f'{file_name}-gold.lattices'
        tokens_path = tb_root_path / tb_name / f'{file_name}.tokens'
        lattice_sentences = split_sentences(lattices_path)
        token_sentences = split_sentences(tokens_path)
        lattices_df = _get_conllx_partition_df(lattice_sentences, token_sentences, ma_name is None)
        # lattices = lattices_df.groupby(lattices_df.sent_id)
        # lattices = [lattices.get_group(x) for x in lattices.groups]
        # logging.info(f'{file_name} lattices: {len(lattices)}')
        treebank[partition_type] = lattices_df
    return treebank
