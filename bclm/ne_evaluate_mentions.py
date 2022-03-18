from collections import defaultdict
from itertools import islice
import pandas as pd
import sys


def fix_multi_biose(tag, multi_delim='^'):
    parts = [x[0] for x in tag.split('^')]
    cat = ''
    
    if '-' in tag:
        cat = '-' + tag.split('-')[1][:3]
        
    bio = 'O'
    if 'S' in parts:
        bio = 'S'
    elif 'B' in parts and 'E' in parts:
        bio='S'
    elif 'E' in parts:
        bio = 'E'
    elif 'B' in parts:
        bio = 'B'
    elif 'I' in parts:
        bio = 'I'
        
    return bio+cat


def read_file_sents(path, comment_prefix='#', field_delim=' ', multi_delim='^', fix_multi_tag=True, sent_id_shift=0):
    sents = []
    for i, sent in enumerate(open(path, 'r', encoding='utf8').read().split('\n\n')):
        if len(sent)>0:
            cur = []
            for line in sent.split('\n'):
                if not line.startswith(comment_prefix):
                    ls = line.split(field_delim)
                    tok, tag = ls[0], ls[-1]
                    if fix_multi_tag and multi_delim in tag:
                        tag = fix_multi_biose(tag, multi_delim=multi_delim)
                    cur.append((tok, tag))
            sents.append((cur, i+sent_id_shift))
    idx, values = zip(*sents)
    sents = pd.Series(idx, values)
    return sents


def evaluate_files(gold_path, pred_path, fix_multi_tag_pred=True, truncate=None, ignore_cat=False, str_join_char=''):
    gold_sents = read_file_sents(gold_path)
    pred_sents = read_file_sents(pred_path)
    gold_mentions = sents_to_mentions(gold_sents, truncate=truncate, ignore_cat=ignore_cat, str_join_char=str_join_char)
    pred_mentions = sents_to_mentions(pred_sents, truncate=truncate, ignore_cat=ignore_cat, str_join_char=str_join_char)
    return evaluate_mentions(gold_mentions, pred_mentions, verbose=True)


def evaluate_mentions(true_ments, pred_ments, examples=5, verbose=True, return_tpc=False):
    t, p = set(true_ments), set(pred_ments)
    correct = p.intersection(t)
    
    if len(p)==0:
        prec=-1
    else:
        prec = len(correct) / len(p)
    
    if len(t)==0:
        recall=-1
    else:
        recall = len(correct) / len(t)

    if prec+recall==0:
        f1=-1
    else:
        f1 = 2*prec*recall/(prec+recall)
    if verbose:
        print(len(t), 'mentions,', len(p), 'found,', len(correct), 'correct.')
        print('Precision:', round(prec, 2))
        print('Recall:   ', round(recall, 2))
        print('F1:       ', round(f1, 2))
        print('FP ex.:', [e[1] for e in list(p-t)[:examples]])
        print('FN ex.:', [e[1] for e in list(t-p)[:examples]])
    if return_tpc:
        return prec, recall, f1, len(t), len(p), len(correct)
    else:
        return prec, recall, f1
        
        
def sent_to_mentions_dict(sent, sent_id, truncate=80, ignore_cat=False, str_join_char=' '):
    mentions = defaultdict(lambda: 0)
    current_mention= None
    current_cat = None
    if truncate is not None:
        it = islice(sent, truncate)
    else:
        it = sent
        
    for tok, bio, cat in it:
        if ignore_cat:
            cat = 'NAN'
        if bio=='S':
            mentions[(sent_id, tok, cat)]+=1
            current_mention= None
            current_cat = None
        if bio=='B':
            current_mention = [tok]
            current_cat = cat
        if bio=='I' and current_mention is not None:
            current_mention.append(tok)
        if bio=='E' and current_mention is not None:
            current_mention.append(tok)
            mentions[(sent_id, str_join_char.join(current_mention), current_cat)]+=1
            current_mention= None
            current_cat = None
        if bio=='O':
            current_mention = None
            current_cat = None
    return mentions


def get_ment_set(ments):
    ment_set = []
    for ment in ments:
        for k, val in ment.items():
            for i in range(val):
                ment_set.append((k[0], k[1], k[2], i+1))
    return ment_set

def get_sents_fixed(sents):
    sf = []
    for sent in sents:
        new_sent = []
        for tok, biose in sent:
            tag = biose.split('-')
            biose = tag[0]
            if len(tag)>1:
                cat = tag[1]
            else:
                cat = '_'
            new_sent.append((tok, biose, cat))
        sf.append(new_sent)
    sf = list(zip(list(sents.index), sf))
    return sf

def sents_to_mentions(sents, truncate=80, ignore_cat=False, str_join_char=' '):
    sents_fixed = get_sents_fixed(sents)
    ments = [sent_to_mentions_dict(sent, sent_id, truncate, ignore_cat=ignore_cat, str_join_char=str_join_char) for sent_id, sent in sents_fixed]
    ment_set = get_ment_set(ments)
    return ment_set


def get_sents_with_pred_tags(splits, preds, truncate=80):
    sents_preds = []
    for split, pred in zip(splits, preds):
        spl_preds = []
        test_sents = split[3]
        
        i=0
        for sent in test_sents:
            new_sent = []
            for tok, bio, cat in islice(sent, truncate):
                pred_tag = pred[i].split('-')
                pred_bio = pred_tag[0]
                if len(pred_tag)>1:
                    pred_cat = pred_tag[1]
                else:
                    pred_cat = '_'
                new_sent.append((tok, pred_bio, pred_cat))
                i+=1
            spl_preds.append(new_sent)
        spl_preds = pd.Series(spl_preds, index=test_sents.index)
        sents_preds.append(spl_preds)
    return sents_preds


if __name__ == '__main__':
    evaluate_files(sys.argv[1], sys.argv[2])
