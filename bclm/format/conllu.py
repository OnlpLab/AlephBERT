from copy import copy
import pandas as pd
import unicodedata
from .format_utils import split_sentences, lattice_fields


CONLLU_COLUMN_NAMES = ['ID', 'FORM', 'LEMMA', 'UPOS', 'XPOS', 'FEATS', 'HEAD', 'DEPREL', 'DEPS', 'MISC']


# https://en.wikipedia.org/wiki/Unicode_character_property
# https://stackoverflow.com/questions/48496869/python3-remove-arabic-punctuation
def _normalize_unicode(s):
    return ''.join(c for c in s if not unicodedata.category(c).startswith('M'))


def _normalize_lattice(lattice):
    return [[_normalize_unicode(part) for part in morpheme] for morpheme in lattice]


def _build_conllu_sample_rows(sent_id, ud_lattice, is_gold):
    tokens = {}
    rows = []
    for morpheme in ud_lattice:
        sample_row = copy(morpheme)
        if len(sample_row[0].split('-')) == 2:
            from_node_id, to_node_id = (int(v) for v in sample_row[0].split('-'))
            token = sample_row[1]
            tokens[to_node_id] = token
        else:
            sample_row[0] = int(sample_row[0])
            sample_row.insert(0, sample_row[0] - 1)
            del sample_row[7]
            del sample_row[7]
            morpheme_token_node_id = sample_row[1]
            token = sample_row[2]
            token_node_id = 0
            token_id = 0
            for i, node_id in enumerate(tokens):
                if morpheme_token_node_id <= node_id:
                    token_id = i + 1
                    token_node_id = node_id
                    break
            if token_node_id == 0:
                token_node_id = morpheme_token_node_id
                token_id = len(tokens) + 1
                tokens[token_node_id] = token
            del sample_row[5]
            sample_row[6] = token_id
            sample_row[7] = tokens[token_node_id]
            sample_row.append(is_gold)
            sample_row.insert(0, sent_id)
            rows.append(sample_row)
    return rows


def _load_conllu_partition_df(lattice_sentences, is_gold):
    partition = []
    for i, lattice in enumerate(lattice_sentences):
        sent_id = i + 1
        # Bug fix - invalid lines (missing '_') found in the Hebrew treebank
        lattice = [line.replace("\t\t", "\t_\t").replace("\t\t", "\t_\t").split('\t') for line in lattice if line[0] != '#']
        # Bug fix - clean unicode characters
        lattice = _normalize_lattice(lattice)
        partition.extend(_build_conllu_sample_rows(sent_id, lattice, is_gold))
    return pd.DataFrame(partition, columns=lattice_fields)


def load_conllu(tb_path, partition, lang, la_name, tb_name, ma_name=None):
    treebank = {}
    for partition_type in partition:
        file_name = f'{la_name}_{tb_name}-ud-{partition_type}'.lower()
        if ma_name is not None:
            lattices_path = tb_path / f'conllul/UL_{lang}-{tb_name}' / f'{file_name}.{ma_name}.conllul'
        else:
            lattices_path = tb_path / f'UD_{lang}-{tb_name}' / f'{file_name}.conllu'
        lattice_sentences = split_sentences(lattices_path)
        lattices_df = _load_conllu_partition_df(lattice_sentences, ma_name is None)
        treebank[partition_type] = lattices_df
    return treebank
