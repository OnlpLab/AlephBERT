from transformers import BertTokenizerFast
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import fasttext_emb as ft
from pathlib import Path


def _insert_morph_id_column(df: pd.DataFrame) -> pd.DataFrame:
    sentences = sorted(df.groupby(df.sent_id))
    sent_morph_ids = [list(sent_df.reset_index(drop=True).index + 1) for sent_id, sent_df in sentences]
    morph_ids = [mid for l in sent_morph_ids for mid in l]
    df.insert(3, 'morph_id', morph_ids)
    return df


def _get_morph_df(raw_lattice_df: pd.DataFrame) -> pd.DataFrame:
    morph_df = raw_lattice_df[['sent_id', 'token_id', 'token', 'form', 'lemma', 'tag', 'feats']]
    morph_df = _insert_morph_id_column(morph_df)
    return morph_df


def add_char_column(df: pd.DataFrame, field_name) -> pd.DataFrame:
    field = df.columns.get_loc(field_name)
    rows = [[list(row[1:]) + [c] for c in row[1:][field]] for row in df.itertuples()]
    rows = [char_row for word_rows in rows for char_row in word_rows]
    return pd.DataFrame(rows, columns=list(df.columns) + ['char'])


def _create_xtoken_df(morph_df: pd.DataFrame, xtokenizer: BertTokenizerFast, sos, eos) -> pd.DataFrame:
    token_df = morph_df[['sent_id', 'token_id', 'token']].drop_duplicates()
    sent_groups = sorted(token_df.groupby([token_df.sent_id]))
    num_sentences = len(sent_groups)
    tq = tqdm(total=num_sentences, desc="Sentence")
    data_rows = []
    for sent_id, sent_df in sent_groups:
        xtokens = [(tid, t, xt) for tid, t in zip(sent_df.token_id, sent_df.token) for xt in xtokenizer.tokenize(t)]
        sent_token_indices = [0] + [tid for tid, t, xt in xtokens] + [sent_df.token_id.max() + 1]
        sent_tokens = [sos] + [t for tid, t, xt in xtokens] + [eos]
        sent_xtokens = [xtokenizer.cls_token] + [xt for tid, t, xt in xtokens] + [xtokenizer.sep_token]
        sent_index = [sent_id] * len(sent_xtokens)
        data_rows.extend(list(zip(sent_index, sent_token_indices, sent_tokens, sent_xtokens)))
        tq.update(1)
    tq.close()
    return pd.DataFrame(data_rows, columns=['sent_id', 'token_id', 'token', 'xtoken'])


def _create_token_char_df(morph_df: pd.DataFrame) -> pd.DataFrame:
    char_df = morph_df[['sent_id', 'token_id', 'token']].drop_duplicates()
    return add_char_column(char_df, 'token')


def _collate_xtokens(xtoken_df: pd.DataFrame, xtokenizer: BertTokenizerFast, pad) -> pd.DataFrame:
    sent_groups = xtoken_df.groupby(xtoken_df.sent_id)
    num_sentences = len(sent_groups)
    max_sent_len = max([len(sent_df) for sent_id, sent_df in sent_groups])
    data_rows = []
    tq = tqdm(total=num_sentences, desc="Sentence")
    for sent_id, sent_df in sent_groups:
        sent_index = list(sent_df.sent_id)
        sent_token_index = list(sent_df.token_id)
        sent_tokens = list(sent_df.token)
        sent_xtokens = list(sent_df.xtoken)
        sent_xtoken_ids = xtokenizer.convert_tokens_to_ids(sent_xtokens)
        pad_len = max_sent_len - len(sent_index)
        sent_index.extend(sent_index[-1:] * pad_len)
        sent_tokens.extend([pad] * pad_len)
        sent_token_index.extend([-1] * pad_len)
        sent_xtokens.extend([xtokenizer.pad_token] * pad_len)
        sent_xtoken_ids.extend([xtokenizer.pad_token_id] * pad_len)
        data_rows.extend(list(row)
                         for row in zip(sent_index, sent_token_index, sent_tokens, sent_xtokens, sent_xtoken_ids))
        tq.update(1)
    tq.close()
    return pd.DataFrame(data_rows, columns=['sent_idx', 'token_idx', 'token', 'xtoken', 'xtoken_id'])


def _collate_token_chars(token_char_df: pd.DataFrame, char2id: dict, pad) -> pd.DataFrame:
    sent_groups = token_char_df.groupby(token_char_df.sent_id)
    num_sentences = len(sent_groups)
    # max_sent_len = max([len(sent_df) for sent_id, sent_df in sent_groups])
    token_groups = token_char_df.groupby([token_char_df.sent_id, token_char_df.token_id])
    max_num_chars = max([len(token_df) for _,  token_df in token_groups])
    data_sent_indices, data_token_indices, data_tokens = [], [], []
    data_chars, data_char_ids = [], []
    tq = tqdm(total=num_sentences, desc="Sentence")
    cur_sent_id = None
    for (sent_id, token_id), token_df in sorted(token_groups):
        if cur_sent_id != sent_id:
            if cur_sent_id is not None:
                tq.update(1)
            cur_sent_id = sent_id
        sent_index = list(token_df.sent_id)
        token_index = list(token_df.token_id)
        tokens = list(token_df.token)
        chars = list(token_df.char)
        char_ids = [char2id[c] for c in chars]
        pad_len = max_num_chars - len(chars)
        sent_index.extend(sent_index[-1:] * pad_len)
        token_index.extend(token_index[-1:] * pad_len)
        tokens.extend(tokens[-1:] * pad_len)
        chars.extend([pad] * pad_len)
        char_ids.extend([char2id[pad]] * pad_len)
        data_sent_indices.extend(sent_index)
        data_token_indices.extend(token_index)
        data_tokens.extend(tokens)
        data_chars.extend(chars)
        data_char_ids.extend(char_ids)
    tq.update(1)
    tq.close()
    data_features = [data_sent_indices, data_token_indices, data_tokens, data_chars, data_char_ids]
    data_rows = list(zip(*data_features))
    data_column_names = ['sent_idx', 'token_idx', 'token', 'char', 'char_id']
    return pd.DataFrame(data_rows, columns=data_column_names)


def save_char_vocab(data_path: Path, ft_root_path: Path, raw_partition: dict, pad, sep, sos, eos):
    logging.info(f'saving char embedding')
    tokens = set(token for part in raw_partition for token in raw_partition[part].token)
    forms = set(token for part in raw_partition for token in raw_partition[part].form)
    lemmas = set(token for part in raw_partition for token in raw_partition[part].lemma)
    # chars = set(c.lower() for word in list(tokens) + list(forms) + list(lemmas) for c in word)
    chars = set(c for word in list(tokens) + list(forms) + list(lemmas) for c in word)
    chars = [pad, sep, sos, eos] + sorted(list(chars))
    char_vectors, char2id = ft.get_word_vectors('he', ft_root_path / 'models/cc.he.300.bin', chars)
    ft.save_word_vectors(data_path / 'ft_char.vec.txt', char_vectors, char2id)


def load_char_vocab(data_path: Path) -> (np.array, dict):
    logging.info(f'Loading char embedding')
    char_vectors, char2id = ft.load_word_vectors(data_path / 'ft_char.vec.txt')
    id2char = {char2id[c]: c for c in char2id}
    char_vocab = {'char2id': char2id, 'id2char': id2char}
    return char_vectors, char_vocab


def get_morph_data(data_path: Path, raw_partition: dict) -> dict:
    morph_partition = {}
    for part in raw_partition:
        morph_file = data_path / f'{part}_morph.csv'
        if not morph_file.exists():
            logging.info(f'preprocessing {part} morphemes')
            morph_df = _get_morph_df(raw_partition[part])
            logging.info(f'saving {morph_file}')
            morph_df.to_csv(str(morph_file))
        else:
            logging.info(f'loading {morph_file}')
            morph_df = pd.read_csv(str(morph_file), index_col=0)
        morph_partition[part] = morph_df
    return morph_partition


def get_token_char_data(data_path: Path, morph_partition: dict) -> dict:
    token_char_partition = {}
    for part in morph_partition:
        token_char_file = data_path / f'{part}_token_char.csv'
        if not token_char_file.exists():
            logging.info(f'preprocessing {part} token chars')
            token_char_df = _create_token_char_df(morph_partition[part])
            logging.info(f'saving {token_char_file}')
            token_char_df.to_csv(str(token_char_file))
        else:
            logging.info(f'loading {token_char_file}')
            token_char_df = pd.read_csv(str(token_char_file), index_col=0)
        token_char_partition[part] = token_char_df
    return token_char_partition


def get_xtoken_data(data_path: Path, morph_partition: dict, xtokenizer: BertTokenizerFast, sos, eos) -> dict:
    xtoken_partition = {}
    for part in morph_partition:
        xtoken_file = data_path / f'{part}_xtoken.csv'
        if not xtoken_file.exists():
            logging.info(f'preprocessing {part} xtokens')
            xtoken_df = _create_xtoken_df(morph_partition[part], xtokenizer, sos=sos, eos=eos)
            logging.info(f'saving {xtoken_file}')
            xtoken_df.to_csv(str(xtoken_file))
        else:
            logging.info(f'loading {xtoken_file}')
            xtoken_df = pd.read_csv(str(xtoken_file), index_col=0)
        xtoken_partition[part] = xtoken_df
    return xtoken_partition


def save_xtoken_data_samples(data_path: Path, xtoken_partition: dict, xtokenizer: BertTokenizerFast, pad):
    for part in xtoken_partition:
        xtoken_samples_file = data_path / f'{part}_xtoken_data_samples.csv'
        logging.info(f'preprocessing {part} xtoken data samples')
        samples_df = _collate_xtokens(xtoken_partition[part], xtokenizer, pad=pad)
        logging.info(f'saving {xtoken_samples_file}')
        samples_df.to_csv(str(xtoken_samples_file))


def load_xtoken_data_samples(data_path: Path, partition: list) -> dict:
    xtoken_samples_partition = {}
    for part in partition:
        xtoken_samples_file = data_path / f'{part}_xtoken_data_samples.csv'
        logging.info(f'loading {xtoken_samples_file}')
        samples_df = pd.read_csv(str(xtoken_samples_file), index_col=0)
        xtoken_samples_partition[part] = samples_df
    return xtoken_samples_partition


def save_token_char_data_samples(data_path: Path, token_partition: dict, char2id: dict, pad):
    for part in token_partition:
        token_char_samples_file = data_path / f'{part}_token_char_data_samples.csv'
        logging.info(f'preprocessing {part} token char data samples')
        samples_df = _collate_token_chars(token_partition[part], char2id, pad=pad)
        logging.info(f'saving {token_char_samples_file}')
        samples_df.to_csv(str(token_char_samples_file))


def _load_token_char_data_samples(data_path: Path, partition: list) -> dict:
    token_char_samples_partition = {}
    for part in partition:
        token_char_samples_file = data_path / f'{part}_token_char_data_samples.csv'
        logging.info(f'loading {token_char_samples_file}')
        samples_df = pd.read_csv(str(token_char_samples_file), index_col=0)
        token_char_samples_partition[part] = samples_df
    return token_char_samples_partition


def to_sub_token_seq(token_data_samples: dict, field_names: list) -> dict:
    data_arrs = {}
    for part in token_data_samples:
        token_df = token_data_samples[part]
        data_column_names = ['sent_idx', 'token_idx'] + [f'{fname}_id' for fname in field_names]
        sub_token_field_data = token_df[data_column_names]
        sent_groups = sub_token_field_data.groupby('sent_idx')
        num_sentences = len(sent_groups)
        max_num_tokens = sub_token_field_data.token_idx.max()
        token_groups = sub_token_field_data.groupby(['sent_idx', 'token_idx'])
        token_lengths = set([len(token_df) for _, token_df in token_groups])
        if len(token_lengths) != 1:
            raise Exception(f'malformed token data samples: len(token_lengths) != 1 ({len(token_lengths)})')
        token_len = list(token_lengths)[0]
        tq = tqdm(total=num_sentences, desc="Sentence")
        sent_arrs = []
        for sent_id, sent_df in sorted(sent_groups):
            sent_arr = sent_df.to_numpy().reshape(-1, token_len, len(data_column_names))
            sent_num_tokens = sent_df.token_idx.max()
            pad_len = max_num_tokens - sent_num_tokens
            if pad_len > 0:
                pad_data = [sent_id, -1] + [0] * len(field_names)
                pad_arr = np.array([[pad_data] * token_len] * pad_len)
                sent_arr = np.concatenate([sent_arr, pad_arr])
            sent_arrs.append(sent_arr)
            tq.update(1)
        tq.close()
        data_arrs[part] = np.stack(sent_arrs, axis=0)
    return data_arrs


def load_xtoken_data(data_path: Path, partition: list) -> dict:
    data_samples = load_xtoken_data_samples(data_path, partition)
    arr_data = {}
    for part in partition:
        xtoken_df = data_samples[part]
        token_data = xtoken_df[['sent_idx', 'token_idx', 'xtoken_id']]
        token_data_groups = token_data.groupby('sent_idx')
        arr_data[part] = np.stack([sent_df.to_numpy() for sent_id, sent_df in sorted(token_data_groups)], axis=0)
    return arr_data


def load_token_char_data(data_path: Path, partition: list) -> dict:
    data_samples = _load_token_char_data_samples(data_path, partition)
    return to_sub_token_seq(data_samples, ['char'])
