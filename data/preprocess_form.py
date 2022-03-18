from .preprocess_base import *


def _create_form_char_df(morph_df: pd.DataFrame, sep, eos) -> pd.DataFrame:
    morph_form_char_df = morph_df.copy()
    morph_form_char_df = add_char_column(morph_form_char_df, 'form')
    sent_groups = morph_form_char_df.groupby([morph_form_char_df.sent_id])
    num_sentences = len(sent_groups)
    morphemes = sorted(morph_form_char_df.groupby([morph_form_char_df.sent_id, morph_form_char_df.morph_id]))
    data_sent_indices, data_token_indices, data_tokens, data_morph_ids = [], [], [], []
    data_forms, data_chars = [], []
    prev_sent_id, prev_token_id = 1, 1
    tq = tqdm(total=num_sentences, desc="Sentence")
    for (sent_id, morph_id), morph_df in morphemes:
        sent_index = list(morph_df.sent_id) + [sent_id]
        token_index = list(morph_df.token_id)
        token_index += token_index[:1]
        tokens = list(morph_df.token)
        tokens += tokens[:1]
        morph_ids = list(morph_df.morph_id) + [morph_id]
        forms = list(morph_df.form)
        forms += forms[:1]
        cur_token_id = token_index[-1]
        if sent_id != prev_sent_id or cur_token_id != prev_token_id:
            data_chars[-1] = eos
            prev_token_id = cur_token_id
            if sent_id != prev_sent_id:
                tq.update(1)
                prev_token_id = 1
                prev_sent_id = sent_id
        chars = list(morph_df.char) + [sep]
        data_sent_indices.extend(sent_index)
        data_token_indices.extend(token_index)
        data_tokens.extend(tokens)
        data_morph_ids.extend(morph_ids)
        data_forms.extend(forms)
        data_chars.extend(chars)
    data_chars[-1] = eos
    tq.update(1)
    tq.close()
    data_fields = [data_sent_indices, data_token_indices, data_tokens, data_morph_ids, data_forms, data_chars]
    data_rows = list(zip(*data_fields))
    data_column_names = ['sent_id', 'token_id', 'token', 'morph_id', 'form', 'char']
    return pd.DataFrame(data_rows, columns=data_column_names)


def _collate_form_chars(morph_form_char_df: pd.DataFrame, char2id: dict, pad) -> pd.DataFrame:
    sent_groups = morph_form_char_df.groupby([morph_form_char_df.sent_id])
    num_sentences = len(sent_groups)
    token_groups = morph_form_char_df.groupby([morph_form_char_df.sent_id, morph_form_char_df.token_id])
    max_num_chars = max([len(token_df) for _,  token_df in token_groups])
    data_sent_indices, data_token_indices, data_tokens, data_morph_indices = [], [], [], []
    data_forms, data_chars, data_char_ids = [], [], []
    cur_sent_id = None
    tq = tqdm(total=num_sentences, desc="Sentence")
    for (sent_id, token_id), token_df in sorted(token_groups):
        if cur_sent_id != sent_id:
            if cur_sent_id is not None:
                tq.update(1)
            cur_sent_id = sent_id
        sent_index = list(token_df.sent_id)
        token_index = list(token_df.token_id)
        tokens = list(token_df.token)
        morph_index = list(token_df.morph_id)
        forms = list(token_df.form)
        chars = list(token_df.char)
        char_ids = [char2id[c] for c in chars]
        pad_len = max_num_chars - len(chars)
        sent_index.extend(sent_index[-1:] * pad_len)
        token_index.extend(token_index[-1:] * pad_len)
        tokens.extend(tokens[-1:] * pad_len)
        morph_index.extend([-1] * pad_len)
        forms.extend([pad] * pad_len)
        chars.extend([pad] * pad_len)
        char_ids.extend([char2id[pad]] * pad_len)
        data_sent_indices.extend(sent_index)
        data_token_indices.extend(token_index)
        data_tokens.extend(tokens)
        data_morph_indices.extend(morph_index)
        data_forms.extend(forms)
        data_chars.extend(chars)
        data_char_ids.extend(char_ids)
    tq.update(1)
    tq.close()
    data_fields = [data_sent_indices, data_token_indices, data_tokens, data_morph_indices, data_forms, data_chars,
                     data_char_ids]
    data_rows = list(zip(*data_fields))
    data_column_names = ['sent_idx', 'token_idx', 'token', 'morph_idx', 'form', 'char', 'char_id']
    return pd.DataFrame(data_rows, columns=data_column_names)


def get_form_char_data(data_path: Path, morph_partition: dict, sep, eos) -> dict:
    morph_form_char_partition = {}
    for part in morph_partition:
        morph_form_char_file = data_path / f'{part}_form_char.csv'
        if not morph_form_char_file.exists():
            logging.info(f'preprocessing {part} form chars')
            morph_form_char_df = _create_form_char_df(morph_partition[part], sep=sep, eos=eos)
            logging.info(f'saving {morph_form_char_file}')
            morph_form_char_df.to_csv(str(morph_form_char_file))
        else:
            logging.info(f'loading {morph_form_char_file}')
            morph_form_char_df = pd.read_csv(str(morph_form_char_file), index_col=0)
        morph_form_char_partition[part] = morph_form_char_df
    return morph_form_char_partition


def save_form_char_data_samples(data_path: Path, morph_form_char_partition: dict, char2id: dict, pad):
    for part in morph_form_char_partition:
        form_char_samples_file = data_path / f'{part}_form_char_data_samples.csv'
        logging.info(f'preprocessing {part} form char data samples')
        samples_df = _collate_form_chars(morph_form_char_partition[part], char2id, pad=pad)
        logging.info(f'saving {form_char_samples_file}')
        samples_df.to_csv(str(form_char_samples_file))


def _load_form_char_data_samples(data_path: Path, partition: list) -> dict:
    form_char_samples_partition = {}
    for part in partition:
        form_char_samples_file = data_path / f'{part}_form_char_data_samples.csv'
        logging.info(f'loading {form_char_samples_file}')
        samples_df = pd.read_csv(str(form_char_samples_file), index_col=0)
        form_char_samples_partition[part] = samples_df
    return form_char_samples_partition


def load_form_data(data_path: Path, partition: list) -> dict:
    data_samples = _load_form_char_data_samples(data_path, partition)
    return to_sub_token_seq(data_samples, ['char'])
