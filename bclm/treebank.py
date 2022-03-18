from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd
import logging
import itertools
from bclm.format import conllx, conllu
from bclm import ne_evaluate_mentions


# gen=F|gen=M -> gen=FM, num=P|num=D -> num=DP
def _unify_multi_value_feats(feats_str: str) -> str:
    features = defaultdict(set)
    for feat in feats_str.split('|'):
        parts = feat.split('=')
        if len(parts) == 2:
            features[parts[0]].add(parts[1])
    feats = []
    for fname in sorted(features):
        fvalue = ','.join(sorted(features[fname]))
        feat_str = f'{fname}={fvalue}'
        feats.append(feat_str)
    if len(feats) == 0:
        return '_'
    return '|'.join(feats)


def _fix_spmrl_lattice_multivalue_feats(lattice_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    feats_column_index = lattice_df.columns.get_loc('feats')
    for values in lattice_df.itertuples():
        feats = _unify_multi_value_feats(values.feats)
        new_values = list(values[1:])
        new_values[feats_column_index] = feats
        rows.append(new_values)
    return pd.DataFrame(rows, columns=lattice_df.columns)


# switch tag and ner values
def _switch_tag_and_feature(lattice_df: pd.DataFrame, feat_name: str) -> pd.DataFrame:
    tags = lattice_df.tag
    feats = [{ff.split('=')[0]: ff.split('=')[1] for ff in f.split('|')} for f in lattice_df['feats'].tolist()]
    new_tags = [f[feat_name] for f in feats]
    for f, t in zip(feats, tags):
        f['tag'] = t
        f.pop(feat_name, None)
    new_feats = ['|'.join([f'{k}={f[k]}' for k in f]) for f in feats]
    lattice_df['tag'] = new_tags
    lattice_df['feats'] = new_feats
    return lattice_df


def spmrl_ner_conllu(data_root_path, tb_name, tb_root_path=None, ma_name=None):
    logging.info('SPMRL NER conllu')
    partition = {'train': None, 'dev': None, 'test': None}
    ma_type = ma_name if ma_name is not None else 'gold'
    data_tb_path = Path(data_root_path) / tb_name / ma_type
    if tb_root_path is not None:
        data_tb_path.mkdir(parents=True, exist_ok=True)
        logging.info(f'Loading treebank: {tb_root_path}')
        partition = conllu.load_conllu(tb_root_path, partition, 'Hebrew', 'he', tb_name, ma_name)
        for part in partition:
            lattice_file_path = data_tb_path / f'{part}_{tb_name}-{ma_type}.lattices.csv'
            logging.info(f'Saving: {lattice_file_path}')
            partition[part].to_csv(lattice_file_path)
    else:
        for part in partition:
            lattice_file_path = data_tb_path / f'{part}_{tb_name}-{ma_type}.lattices.csv'
            logging.info(f'Loading: {lattice_file_path}')
            partition[part] = pd.read_csv(lattice_file_path, index_col=0)
    return partition


def spmrl_conllu(data_root_path, tb_name, tb_root_path=None, ma_name=None):
    logging.info('SPMRL conllu')
    partition = {'train': None, 'dev': None, 'test': None}
    ma_type = ma_name if ma_name is not None else 'gold'
    data_tb_path = Path(data_root_path) / tb_name / ma_type
    if tb_root_path is not None:
        data_tb_path.mkdir(parents=True, exist_ok=True)
        logging.info(f'Loading treebank: {tb_root_path}')
        partition = conllu.load_conllu(tb_root_path, partition, 'Hebrew', 'he', tb_name, ma_name)
        for part in partition:
            partition[part] = _fix_spmrl_lattice_multivalue_feats(partition[part])
        for part in partition:
            lattice_file_path = data_tb_path / f'{part}_{tb_name}-{ma_type}.lattices.csv'
            logging.info(f'Saving: {lattice_file_path}')
            partition[part].to_csv(lattice_file_path)
    else:
        for part in partition:
            lattice_file_path = data_tb_path / f'{part}_{tb_name}-{ma_type}.lattices.csv'
            logging.info(f'Loading: {lattice_file_path}')
            partition[part] = pd.read_csv(lattice_file_path, index_col=0)
    return partition


def spmrl(data_root_path, tb_name, tb_root_path=None, ma_name=None):
    logging.info('SPMRL lattices')
    partition = {'train': None, 'dev': None, 'test': None}
    ma_type = ma_name if ma_name is not None else 'gold'
    data_tb_path = Path(data_root_path) / tb_name / ma_type
    if tb_root_path is not None:
        data_tb_path.mkdir(parents=True, exist_ok=True)
        logging.info(f'Loading treebank: {tb_root_path}')
        partition = conllx.load_conllx(tb_root_path, partition, tb_name, ma_name)
        for part in partition:
            logging.info(f'Unify {part} multivalued SPMRL features')
            partition[part] = _fix_spmrl_lattice_multivalue_feats(partition[part])
        for part in partition:
            lattice_file_path = data_tb_path / f'{part}_{tb_name}-{ma_type}.lattices.csv'
            logging.info(f'Saving: {lattice_file_path}')
            partition[part].to_csv(lattice_file_path)
    else:
        for part in partition:
            lattice_file_path = data_tb_path / f'{part}_{tb_name}-{ma_type}.lattices.csv'
            logging.info(f'Loading: {lattice_file_path}')
            partition[part] = pd.read_csv(lattice_file_path, index_col=0)
    return partition


def ud(data_root_path, tb_name, tb_root_path=None, ma_name=None):
    logging.info('UD lattices')
    partition = {'train': None, 'dev': None, 'test': None}
    ma_type = ma_name if ma_name is not None else 'gold'
    data_tb_path = Path(data_root_path) / tb_name / ma_type
    if tb_root_path is not None:
        data_tb_path.mkdir(parents=True, exist_ok=True)
        logging.info(f'Loading treebank: {tb_root_path}')
        partition = conllu.load_conllu(tb_root_path, partition, 'Hebrew', 'he', tb_name, ma_name)
        for part in partition:
            lattice_file_path = data_tb_path / f'{part}_{tb_name}-{ma_type}.lattices.csv'
            logging.info(f'Saving: {lattice_file_path}')
            partition[part].to_csv(lattice_file_path)
    else:
        for part in partition:
            lattice_file_path = data_tb_path / f'{part}_{tb_name}-{ma_type}.lattices.csv'
            logging.info(f'Loading: {lattice_file_path}')
            partition[part] = pd.read_csv(lattice_file_path, index_col=0)
    return partition


def get_subsets(s, n):
    return list(itertools.combinations(s, n))


def morph_eval(pred_df, gold_df, fields):
    gold_gb = gold_df.groupby([gold_df.sent_id, gold_df.token_id])
    pred_gb = pred_df.groupby([pred_df.sent_id, pred_df.token_id])
    aligned_gold_counts, aligned_pred_counts, aligned_intersection_counts = defaultdict(int), defaultdict(int), defaultdict(int)
    mset_gold_counts, mset_pred_counts, mset_intersection_counts = defaultdict(int), defaultdict(int), defaultdict(int)
    for (sent_id, token_id), gold in sorted(gold_gb):
        for n in range(1, len(fields) + 1):
            fsets = get_subsets(fields, n)
            for fs in fsets:
                gold_values = [tuple(row[1].values) for row in gold[list(fs)].iterrows()]
                if (sent_id, token_id) not in pred_gb.groups:
                    pred_values = []
                else:
                    pred = pred_gb.get_group((sent_id, token_id))
                    pred_values = [tuple(row[1].values) for row in pred[list(fs)].iterrows()]
                # mset
                gold_count, pred_count = Counter(gold_values), Counter(pred_values)
                intersection_count = gold_count & pred_count
                mset_gold_counts[fs] += sum(gold_count.values())
                mset_pred_counts[fs] += sum(pred_count.values())
                mset_intersection_counts[fs] += sum(intersection_count.values())
                # aligned
                intersection_values = [p for g, p in zip(gold_values, pred_values) if p == g]
                aligned_gold_counts[fs] += len(gold_values)
                aligned_pred_counts[fs] += len(pred_values)
                aligned_intersection_counts[fs] += len(intersection_values)
    aligned_scores, mset_scores = {}, {}
    for fs in aligned_gold_counts:
        precision = aligned_intersection_counts[fs] / aligned_pred_counts[fs] if aligned_pred_counts[fs] else 0.0
        recall = aligned_intersection_counts[fs] / aligned_gold_counts[fs] if aligned_gold_counts[fs] else 0.0
        f1 = 2.0 * (precision * recall) / (precision + recall) if precision + recall else 0.0
        aligned_scores[fs] = precision, recall, f1
    for fs in mset_gold_counts:
        precision = mset_intersection_counts[fs] / mset_pred_counts[fs] if mset_pred_counts[fs] else 0.0
        recall = mset_intersection_counts[fs] / mset_gold_counts[fs] if mset_gold_counts[fs] else 0.0
        f1 = 2.0 * (precision * recall) / (precision + recall) if precision + recall else 0.0
        mset_scores[fs] = precision, recall, f1
    return aligned_scores, mset_scores


def ner_eval(ner_file_path, truth_file_path, with_type=True):
    ne_evaluate_mentions.evaluate_files(truth_file_path, ner_file_path, ignore_cat=with_type)
