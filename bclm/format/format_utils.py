import sys
import uuid
from collections import defaultdict
from copy import deepcopy
import pandas as pd

lattice_fields = ['sent_id', 'from_node_id', 'to_node_id', 'form', 'lemma', 'tag', 'feats', 'token_id', 'token', 'is_gold']


def split_sentences(file_path):
    with open(str(file_path)) as f:
        lines = [line.strip() for line in f.readlines()]
    sent_sep_pos = [i for i in range(len(lines)) if len(lines[i]) == 0]
    sent_sep = [(0, sent_sep_pos[0])] + [(sent_sep_pos[i]+1, sent_sep_pos[i+1]) for i in range(len(sent_sep_pos) - 1)]
    return [lines[sep[0]:sep[1]] for sep in sent_sep]


# This is a trick to enable using nested functions with multiprocessing
# The inner function can be either annotated by @globalized or defined for example: func = globalize(lambda x: x)
# https://gist.github.com/EdwinChan/3c13d3a746bb3ec5082f
def globalize(func):
    def result(*args, **kwargs):
        return func(*args, **kwargs)

    result.__name__ = result.__qualname__ = uuid.uuid4().hex
    setattr(sys.modules[result.__module__], result.__name__, result)
    return result


def _dfs(edges, cur_node_id, next_node_id, analysis_in, analyses):
    node = edges[cur_node_id]
    edge = node[next_node_id]
    analysis = deepcopy(analysis_in)
    analysis.append(tuple(edge))
    if edge.to_node_id not in edges:
        analyses.append(analysis)
        return
    next_node = edges[edge.to_node_id]
    for i in range(len(next_node)):
        _dfs(edges, edge.to_node_id, i, analysis, analyses)


def _parse_sent_analyses(df, column_names):
    token_analyses = {}
    token_edges = defaultdict(lambda: defaultdict(list))
    for row in df.itertuples():
        token_edges[row.token_id][row.from_node_id].append(row)
    for token_id in token_edges:
        analyses = []
        token_lattice_start_node_id = min(token_edges[token_id].keys())
        token_lattice_start_node = token_edges[token_id][token_lattice_start_node_id]
        for j in range(len(token_lattice_start_node)):
            _dfs(token_edges[token_id], token_lattice_start_node_id, j, [], analyses)
        token_analyses[token_id] = analyses
    return _lattice_to_dataframe(token_analyses, column_names)


def _lattice_to_dataframe(lattice, column_names):
    rows = []
    for token_id in lattice:
        for i, analyses in enumerate(lattice[token_id]):
            for j, morpheme in enumerate(analyses):
                row = [*morpheme[1:], i, j]
                rows.append(row)
    # return pd.merge(lattice_df, token_df, on='token_id')
    return pd.DataFrame(rows, columns=column_names)


def _to_data_lattices(treebank):
    dataset = {}
    column_names = lattice_fields + ['analysis_id', 'morpheme_id']
    for partition_type in treebank:
        lattices = [_parse_sent_analyses(df, column_names) for df in treebank[partition_type]]
        dataset[partition_type] = lattices
    return dataset
