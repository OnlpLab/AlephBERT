from collections import defaultdict
from bclm import treebank as tb
from pathlib import Path
import pickle

vocab_file_path = Path('vocab.pickle')
if vocab_file_path.exists():
    with open(str(vocab_file_path), 'rb') as f:
        (token2id, form2id, lemma2id, char2id, tag2id, feats2id) = pickle.load(f)
        id2token = {v: k for k, v in token2id.items()}
        id2form = {v: k for k, v in form2id.items()}
        id2lemma = {v: k for k, v in lemma2id.items()}
        id2char = {v: k for k, v in char2id.items()}
        id2tag = {v: k for k, v in tag2id.items()}
        id2feats = {}
        for feat_type in feats2id:
            id2feats[feat_type] = {v: k for k, v in feats2id[feat_type].items()}
else:
    properties = ['token', 'form', 'lemma', 'tag', 'feats']

    token2id, form2id, lemma2id, char2id, tag2id = ({'<pad>': 0, '<s>': 1, '</s>': 2, '_': 3} for _ in range(5))
    id2token, id2form, id2lemma, id2char, id2tag = ({0: '<pad>', 1: '<s>', 2: '</s>', 3: '_'} for _ in range(5))
    feats2id = {}
    id2feats = {}
    prop2ids = [token2id, form2id, lemma2id, tag2id, feats2id]
    id2props = [id2token, id2form, id2lemma, id2tag, id2feats]
    partition = tb.spmrl(None)
    for part in partition:
        for sample in partition[part]:
            for prop, prop2id, id2prop in zip(properties, prop2ids, id2props):
                for value in sample[prop]:
                    if prop == 'feats':
                        if value == '_':
                            continue
                        feats = defaultdict(set)
                        for feat_key_value in value.split("|"):
                            feat_key, feat_value = feat_key_value.split("=")
                            feats[feat_key].add(feat_value)
                        for feat_key in feats:
                            if feat_key not in prop2id:
                                prop2id[feat_key] = {'<pad>': 0, '<s>': 1, '</s>': 2, '_': 3}
                                id2prop[feat_key] = {0: '<pad>', 1: '<s>', 2: '</s>', 3: '_'}
                            feat_value = ','.join(feats[feat_key])
                            if feat_value not in prop2id[feat_key]:
                                next_id = len(id2prop[feat_key])
                                id2prop[feat_key][next_id] = feat_value
                                prop2id[feat_key][feat_value] = next_id
                    else:
                        if value not in prop2id:
                            next_id = len(id2prop)
                            id2prop[next_id] = value
                            prop2id[value] = next_id
                        if prop in ['token', 'form', 'lemma']:
                            for c in value:
                                if c not in char2id:
                                    next_id = len(id2char)
                                    id2char[next_id] = c
                                    char2id[c] = next_id
    with open(str(vocab_file_path), 'wb') as f:
        pickle.dump((token2id, form2id, lemma2id, char2id, tag2id, feats2id), f)

print(f'{len(id2token)} tokens')
print(f'{len(id2form)} forms')
print(f'{len(id2lemma)} lemmas')
print(f'{len(id2char)} chars')
print(f'{len(id2tag)} tags')
print(f'{len(id2feats)} feats')
