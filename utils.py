import json
import torch
import pandas as pd
from itertools import zip_longest
from collections import Counter
from bclm import treebank as tb


def to_sent_tokens(token_chars, id2char: dict) -> list:
    tokens = []
    for chars in token_chars:
        token = ''.join([id2char[c] for c in chars[chars > 0].tolist()])
        tokens.append(token)
    return tokens


def to_token_morph_segments(chars, id2char: dict, eos, sep) -> list:
    tokens = []
    token_mask = torch.nonzero(torch.eq(chars, eos))
    token_mask_map = {m[0].item(): m[1].item() for m in token_mask}
    for i, token_chars in enumerate(chars):
        token_len = token_mask_map[i] if i in token_mask_map else chars.shape[1]
        token_chars = token_chars[:token_len]
        form_mask = torch.nonzero(torch.eq(token_chars, sep))
        forms = []
        start_pos = 0
        for to_pos in form_mask:
            form_chars = token_chars[start_pos:to_pos.item()]
            form = ''.join([id2char[c.item()] for c in form_chars])
            forms.append(form)
            start_pos = to_pos.item() + 1
        form_chars = token_chars[start_pos:]
        form = ''.join([id2char[c.item()] for c in form_chars])
        forms.append(form)
        tokens.append(forms)
    return tokens


def to_token_morph_labels(labels, label_names, id2labels: dict, pads: list) -> list:
    tokens = []
    for i, (feat_labels, feat_name) in enumerate(zip(labels, label_names)):
        morph_labels = []
        for token_feat_labels in feat_labels:
            token_feat_labels = token_feat_labels[token_feat_labels != pads[i]]
            token_feat_labels = [id2labels[feat_name][t.item()] for t in token_feat_labels]
            morph_labels.append(token_feat_labels)
        tokens.append(morph_labels)
    return list(map(list, zip(*tokens)))


def _to_feats_strs(labels: dict) -> list:
    feats_strs = []
    feature_names = sorted(labels)
    feature_values = [labels[feat_name] for feat_name in feature_names]
    feature_values = [f for f in zip_longest(*feature_values)]
    for fvalues in feature_values:
        fstrs = [f'{feature_names[j]}={fvalues[j]}' for j in range(len(feature_names)) if fvalues[j] != '_']
        feats_str = '|'.join(fstrs) if len(fstrs) > 0 else '_'
        feats_strs.append(feats_str)
    return feats_strs


def _to_sent_token_lattice_rows(sent_id, tokens, token_segments, token_labels, label_names: list) -> list:
    rows = []
    node_id = 0
    for token_id, (token, forms, labels) in enumerate(zip_longest(tokens, token_segments, token_labels, fillvalue=[])):
        labels = {label_names[j]: labels[j] for j in range(len(label_names))}
        # tags = labels['tag'] if 'tag' in labels else ['_' for _ in range(len(forms))]
        tags = labels['tag'] if 'tag' in labels else []
        feats_strs = _to_feats_strs({k: v for k, v in labels.items() if k != 'tag'})
        for form, tag, feat in zip_longest(forms, tags, feats_strs, fillvalue='_'):
            row = [sent_id, node_id, node_id+1, form, '_', tag, feat, token_id+1, token, True]
            rows.append(row)
            node_id += 1
    return rows


def get_lattice_data(sent_token_seg_tag_rows, label_names: list) -> pd.DataFrame:
    lattice_rows = []
    for row in sent_token_seg_tag_rows:
        lattice_rows.extend(_to_sent_token_lattice_rows(*row, label_names))
    columns = ['sent_id', 'from_node_id', 'to_node_id', 'form', 'lemma', 'tag', 'feats', 'token_id', 'token', 'is_gold']
    return pd.DataFrame(lattice_rows, columns=columns)


def morph_eval(decoded_sent_tokens, target_sent_tokens) -> (tuple, tuple):
    aligned_decoded_counts, aligned_target_counts, aligned_intersection_counts = 0, 0, 0
    mset_decoded_counts, mset_target_counts, mset_intersection_counts = 0, 0, 0
    for decoded_tokens, target_tokens in zip(decoded_sent_tokens, target_sent_tokens):
        for decoded_segments, target_segments in zip(decoded_tokens, target_tokens):
            decoded_segment_counts, target_segment_counts = Counter(decoded_segments), Counter(target_segments)
            intersection_segment_counts = decoded_segment_counts & target_segment_counts
            mset_decoded_counts += sum(decoded_segment_counts.values())
            mset_target_counts += sum(target_segment_counts.values())
            mset_intersection_counts += sum(intersection_segment_counts.values())
            aligned_segments = [d for d, t in zip(decoded_segments, target_segments) if d == t]
            aligned_decoded_counts += len(decoded_segments)
            aligned_target_counts += len(target_segments)
            aligned_intersection_counts += len(aligned_segments)
    precision = aligned_intersection_counts / aligned_decoded_counts if aligned_decoded_counts else 0.0
    recall = aligned_intersection_counts / aligned_target_counts if aligned_target_counts else 0.0
    f1 = 2.0 * (precision * recall) / (precision + recall) if precision + recall else 0.0
    aligned_scores = precision, recall, f1
    precision = mset_intersection_counts / mset_decoded_counts if mset_decoded_counts else 0.0
    recall = mset_intersection_counts / mset_target_counts if mset_target_counts else 0.0
    f1 = 2.0 * (precision * recall) / (precision + recall) if precision + recall else 0.0
    mset_scores = precision, recall, f1
    return aligned_scores, mset_scores


def print_eval_scores(decoded_df, truth_df, fields, phase, step):
    aligned_scores, mset_scores = tb.morph_eval(pred_df=decoded_df, gold_df=truth_df, fields=fields)
    for fs in mset_scores:
        p, r, f = aligned_scores[fs]
        print(f'{phase} step {step} aligned {fs} eval scores: [P: {p}, R: {r}, F: {f}]')
        p, r, f = mset_scores[fs]
        print(f'{phase} step {step} mset {fs} eval scores   : [P: {p}, R: {r}, F: {f}]')


# 0	1	גנן	גנן	NN	NN	gen=M|num=S	1
# 1	2	גידל	גידל	VB	VB	gen=M|num=S|per=3|tense=PAST	2
# 2	3	דגן	דגן	NN	NN	gen=M|num=S	3
# 3	4	ב	ב	PREPOSITION	PREPOSITION	_	4
# 4	5	ה	ה	DEF	DEF	_	4
# 5	6	גן	גן	NN	NN	gen=M|num=S	4
# 6	7	.	_	yyDOT	yyDOT	_	5
def save_lattice(df, out_file_path):
    gb = df.groupby('sent_id')
    with open(out_file_path, 'w') as f:
        for sid, group in gb:
            # for row in group[['from_node', 'to_node', 'form', 'lemma', 'tag', 'feats', 'token_id']].itertuples():
            for row in group.iterrows():
                lattice_line = '\t'.join([str(v) for v in row[1][['from_node_id', 'to_node_id', 'form', 'lemma', 'tag', 'tag', 'feats', 'token_id']].tolist()])
                f.write(f'{lattice_line}\n')
            f.write('\n')


# Save bmes file used by the ner evaluation script
def save_ner(df, out_file_path, ner_feat_name):
    gb = df.groupby('sent_id')
    with open(out_file_path, 'w') as f:
        for sid, group in gb:
            for row in group[['form', 'feats']].itertuples():
                if row.feats == '_':
                    feats = {ner_feat_name: 'O'}
                else:
                    feats = {feat.split('=')[0]: feat.split('=')[1] for feat in row.feats.split('|')}
                f.write(f"{row.form} {feats[ner_feat_name]}\n")
            f.write('\n')


# Attempt to create a CSV to use as input to the ner_run.py script
# The CSV option doesn't work though in the script, use the json format instead
def save_token_classification_finetune_ner_csv(df, out_file_path):
    gb = df.groupby('sent_id')
    for i, (sid, group) in enumerate(gb):
        g = group[['form', 'tag']]
        g.columns = ['tokens', 'ner_tags']
        if i == 0:
            g.to_csv(out_file_path, index=False)
        else:
            g.to_csv(out_file_path, index=False, mode='a', header=False)
        with open(out_file_path, 'a') as f:
            f.write('\n')


# JSON to use as input to the ner_run.py script
def save_token_classification_finetune_ner_json(df, out_file_path):
    with open(out_file_path, 'w') as f:
        gb = df.groupby('sent_id')
        for sid, group in gb:
            words = list(group.form)
            ner = list(group.tag)
            j = {'words': words, 'ner': ner}
            json.dump(j, f)
            f.write('\n')
