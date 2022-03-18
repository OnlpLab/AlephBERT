from collections import defaultdict

from .preprocess_base import *


def _collate_labels(morph_df: pd.DataFrame, labels2id: dict, pad, eos):
    sent_groups = morph_df.groupby([morph_df.sent_id])
    num_sentences = len(sent_groups)
    token_groups = sorted(morph_df.groupby([morph_df.sent_id, morph_df.token_id]))
    max_num_morphemes = max([len(token_df) for _, token_df in token_groups])
    if eos is not None:
        max_num_morphemes += 1
    data_sent_indices, data_token_indices, data_tokens, data_morph_indices = [], [], [], []
    data_forms = []
    data_labels = {l: [] for l in labels2id}
    data_label_ids = {l: [] for l in labels2id}
    feat_names = labels2id.keys()
    cur_sent_id = None
    tq = tqdm(total=num_sentences, desc="Sentence")
    for (sent_id, token_id), token_df in token_groups:
        if cur_sent_id != sent_id:
            if cur_sent_id is not None:
                tq.update(1)
            cur_sent_id = sent_id
        sent_index = list(token_df.sent_id)
        token_index = list(token_df.token_id)
        tokens = list(token_df.token)
        morph_index = list(token_df.morph_id)
        forms = list(token_df.form)
        labels = {feat_name: [] for feat_name in feat_names}
        for morph_feats in token_df.feats:
            feats = {feat_name: '_' for feat_name in feat_names}
            if morph_feats != '_':
                morph_feats = morph_feats.split('|')
                for f in morph_feats:
                    [feat_name, feat_value] = f.split('=')
                    feats[feat_name] = feat_value
            for feat_name in feats:
                labels[feat_name].append(feats[feat_name])
        labels['tag'] = list(token_df.tag)
        labels = {l: labels[l] for l in labels2id}
        label_ids = {l: [labels2id[l][v] for v in labels[l]] for l in labels}
        if eos is not None:
            sent_index.append(sent_index[-1])
            token_index.append(token_index[-1])
            tokens.append(tokens[-1])
            morph_index.append(morph_index[-1])
            forms.append(forms[-1])
            for l in labels2id:
                labels[l].append(eos)
                label_ids[l].append(labels2id[l][eos])
        pad_len = max_num_morphemes - len(morph_index)
        sent_index.extend(sent_index[-1:] * pad_len)
        token_index.extend(token_index[-1:] * pad_len)
        tokens.extend(tokens[-1:] * pad_len)
        morph_index.extend([-1] * pad_len)
        forms.extend([pad] * pad_len)
        for l in labels2id:
            labels[l].extend([pad] * pad_len)
            label_ids[l].extend([labels2id[l][pad]] * pad_len)
        data_sent_indices.extend(sent_index)
        data_token_indices.extend(token_index)
        data_tokens.extend(tokens)
        data_morph_indices.extend(morph_index)
        data_forms.extend(forms)
        for l in labels:
            data_labels[l].extend(labels[l])
            data_label_ids[l].extend(label_ids[l])
    tq.update(1)
    tq.close()
    data_column_names = ['sent_idx', 'token_idx', 'token', 'morph_idx', 'form']
    data_fields = [data_sent_indices, data_token_indices, data_tokens, data_morph_indices, data_forms]
    for l in data_labels:
        data_column_names.append(l)
        data_column_names.append(f'{l}_id')
        data_fields.append(data_labels[l])
        data_fields.append(data_label_ids[l])
    data_rows = list(zip(*data_fields))
    return pd.DataFrame(data_rows, columns=data_column_names)


def load_label_vocab(data_path: Path, partition: iter, pad, sos=None, eos=None) -> dict:
    logging.info(f'Loading morph vocab')
    # char_vectors, char_vocab = load_char_vocab(data_path)
    tags = set()
    feats = defaultdict(set)
    for part in partition:
        morph_file = data_path / f'{part}_morph.csv'
        morph_data = pd.read_csv(str(morph_file), index_col=0)
        tags |= set(morph_data.tag)
        for morph_feats in morph_data.feats:
            morph_feats = morph_feats.split('|')
            for f in morph_feats:
                if f != '_':
                    [feat_name, feat_value] = f.split('=')
                    feats[feat_name].add(feat_value)
    # tags.add('_')
    for f in feats:
        if f != 'biose_layer0':
            feats[f].add('_')
    special_labels = [pad]
    if sos is not None:
        special_labels += [sos]
    if eos is not None:
        special_labels += [eos]
    tags = special_labels + sorted(list(tags))
    labels = {fname: special_labels + sorted(list(feats[fname])) for fname in feats}
    labels['tag'] = tags
    labels2id = {lname: {l: i for i, l in enumerate(labels[lname])} for lname in labels}
    id2labels = {lname: {v: k for k, v in labels2id[lname].items()} for lname in labels2id}
    label_vocab = {'labels2id': labels2id, 'id2labels': id2labels}
    return label_vocab


def save_labeled_data_samples(data_path: Path, morph_partition: dict, labels2id: dict, pad, eos=None):
    for part in morph_partition:
        label_samples_file = data_path / f'{part}_label_data_samples.csv'
        logging.info(f'preprocessing {part} labeled data samples')
        samples_df = _collate_labels(morph_partition[part], labels2id, pad=pad, eos=eos)
        logging.info(f'saving {label_samples_file}')
        samples_df.to_csv(str(label_samples_file))


def _load_labeled_data_samples(data_path: Path, partition: list) -> dict:
    tag_samples_partition = {}
    for part in partition:
        label_samples_file = data_path / f'{part}_label_data_samples.csv'
        logging.info(f'loading {label_samples_file}')
        samples_df = pd.read_csv(str(label_samples_file), index_col=0)
        tag_samples_partition[part] = samples_df
    return tag_samples_partition


# def load_morph_data(data_path: Path, partition: list, pad, eos):
#     data_samples = _load_morph_data_samples(data_path, partition)
#     if eos is None:
#         for part in data_samples:
#             mask = data_samples[part].tag_id == 2
#             data_samples[part].loc[mask, 'tag_id'] = 0
#             data_samples[part].loc[mask, 'tag'] = pad
#             data_samples[part].loc[mask, 'form'] = pad
#             data_samples[part].loc[mask, 'morph_idx'] = -1
#     return to_sub_token_seq(data_samples, 'tag_id')
def load_labeled_data(data_path: Path, partition: list, label_names: list) -> dict:
    data_samples = _load_labeled_data_samples(data_path, partition)
    return to_sub_token_seq(data_samples, label_names)


def get_label_names(data_path: Path, partition: list):
    data_samples = _load_labeled_data_samples(data_path, partition)
    non_label_column_names = ['sent_idx', 'token_idx', 'morph_idx', 'token', 'form', 'tag']
    label_names = set()
    for part in partition:
        label_names |= set([name for name in data_samples[part].columns
                            if name not in non_label_column_names and name[-3:] != '_id'])
    return ['tag'] + list(sorted(label_names))
