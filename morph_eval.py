import logging
import re
from pathlib import Path
import pandas as pd

# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)


def read_file(path):
    with open(str(path)) as f:
        return [line.strip() for line in f.readlines()]


def parse_scores(lines: list) -> list:
    regexp = re.compile(r'^[ ]*"\(0\.[0-9]+, 0\.[0-9]+, 0\.[0-9]+\)\\n"')
    parsed_eval_scores, parsed_ner_scores = [], []
    eval_scores_added = False
    add_ner_scores = False
    for line in lines:
        if 'eval scores' in line:
            eval_scores = parse_eval_score_line(line)
            parsed_eval_scores.append(eval_scores)
            eval_scores_added = True
        elif eval_scores_added:
            ner_scores = {'partition': parsed_eval_scores[-1]['partition'], 'eval_type': 'aligned',
                          'morph_type': "('biose',)", 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
            parsed_eval_scores.append(ner_scores)
            parsed_ner_scores.append(ner_scores)
            ner_scores = {'partition': parsed_eval_scores[-1]['partition'], 'eval_type': 'mset',
                          'morph_type': "('biose',)", 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
            parsed_eval_scores.append(ner_scores)
            parsed_ner_scores.append(ner_scores)
            eval_scores_added = False
            add_ner_scores = True
        if regexp.search(line):
            if add_ner_scores:
                [precision, recall, f1] = parse_ner_score_line(line)
                if parsed_ner_scores[-3]['precision'] == 0.0:
                    parsed_ner_scores[-3]['precision'] = precision
                else:
                    parsed_ner_scores[-1]['precision'] = precision
                if parsed_ner_scores[-3]['recall'] == 0.0:
                    parsed_ner_scores[-3]['recall'] = recall
                else:
                    parsed_ner_scores[-1]['recall'] = recall
                if parsed_ner_scores[-3]['f1'] == 0.0:
                    parsed_ner_scores[-3]['f1'] = f1
                else:
                    parsed_ner_scores[-1]['f1'] = f1
                add_ner_scores = False
            else:
                add_ner_scores = True
    return parsed_eval_scores


def parse_ner_score_line(line: str) -> list:
    return line[2:-5].split(", ")


def parse_eval_score_line(line: str) -> dict:
    parts = line.split(':')
    info_parts = parts[0].split()
    score_parts = line.split()[-6:]
    data_partition_name = info_parts[0][1:]
    eval_type = info_parts[3]
    morph_type = ''.join([info_parts[i] for i in range(4, len(info_parts)-2)])
    morph_type = [t[1:-1] for t in tuple(map(str, morph_type.split(',')))]
    morph_type[0] = morph_type[0][1:]
    morph_type[-1] = morph_type[-1][:-1]
    if len(morph_type[-1]) == 0:
        morph_type = morph_type[:-1]
    precision = float(score_parts[-5][:-1])
    recall = float(score_parts[-3][:-1])
    f1 = float(score_parts[-1][:-5])
    return {'partition': data_partition_name, 'eval_type': eval_type, 'morph_type': tuple(morph_type),
            'precision': precision, 'recall': recall, 'f1': f1}


def parse_input_file(input_file_path: Path) -> pd.DataFrame:
    lines = read_file(input_file_path)
    parsed_eval_scores = [parse_eval_score_line(line) for line in lines if 'eval scores' in line]
    return pd.DataFrame(parsed_eval_scores)


def parse_input_ner_file(input_file_path: Path) -> pd.DataFrame:
    lines = read_file(input_file_path)
    parsed_scores = parse_scores(lines)
    return pd.DataFrame(parsed_scores)


def insert_eval_info(df, crf, task, tb, epochs, vocab, corpus, tokenizer, model_size, model_name):
    num_morph_types = len(df['morph_type'].unique())
    df.insert(0, 'iter', [int(i // (4 * num_morph_types)) + 1 for i in range(len(df))])
    df.insert(0, 'crf', crf)
    df.insert(0, 'task', task)
    df.insert(0, 'tb', tb.upper())
    df.insert(0, 'epochs', epochs)
    df.insert(0, 'vocab', vocab)
    df.insert(0, 'corpus', corpus)
    df.insert(0, 'tokenizer', tokenizer)
    df.insert(0, 'model_size', model_size)
    df.insert(0, 'model_name', model_name)


def fix_partition(df):
    num_morph_types = len(df['morph_type'].unique())
    rows = [j for i, j in enumerate(list(range(0, len(df), 2 * num_morph_types))) if (i % 2) == 1]
    for j in range(2 * num_morph_types):
        for i in [row + j for row in rows]:
            df.at[i, 'partition'] = 'test'


def get_task_eval_data(experiments_path, task, bert_size, tokenizer_type, bert_name, corpus_name, vocab_size, num_epochs) -> list:
    if bert_name == 'bert':
        bert_info = f'{bert_name}-{bert_size}-{tokenizer_type}-{corpus_name}-{vocab_size}-{num_epochs}'
    else:
        bert_info = bert_name
    tb_dataframes = []
    for schema in ['spmrl', 'ud']:
        if schema == 'spmrl':
            tb_name = 'hebtb'
            if 'ner' in task:
                tb_type = 'for_amit_spmrl'
            else:
                tb_type = 'HebrewTreebank'
        elif 'ner' in task:
            continue
        else:
            tb_name = 'HTB'
            tb_type = 'UD_Hebrew'
        task_path = experiments_path / 'morph' / f'morph_{task.replace("-", "_")}' / 'bert' / bert_size / 'wordpiece'
        if not task_path.exists():
            # print(f"{task_path} task path doesn't exist")
            continue
        bert_path = task_path / bert_info
        if not bert_path.exists():
            # print(f"{bert_path} bert path doesn't exist")
            continue
        tb_path = bert_path / tb_type / tb_name
        nb_file_path = tb_path / f'{bert_info}-{schema}-{task}.ipynb'
        if 'ner' in task:
            eval_scores_df = parse_input_ner_file(nb_file_path)
            insert_eval_info(eval_scores_df, False, task, schema, num_epochs, vocab_size, corpus_name, tokenizer_type,
                             bert_size, 'bert')
            tb_dataframes.append(eval_scores_df)
            nb_file_path = tb_path / 'crf' / f'{bert_info}-{schema}-{task}-crf.ipynb'
            if not nb_file_path.exists():
                # print(f"{nb_file_path} crf path doesn't exist")
                continue
            eval_scores_df = parse_input_ner_file(nb_file_path)
            insert_eval_info(eval_scores_df, True, task, schema, num_epochs, vocab_size, corpus_name, tokenizer_type,
                             bert_size, 'bert')
            tb_dataframes.append(eval_scores_df)
        else:
            eval_scores_df = parse_input_file(nb_file_path)
            insert_eval_info(eval_scores_df, False, task, schema, num_epochs, vocab_size, corpus_name, tokenizer_type,
                             bert_size, 'bert')
            tb_dataframes.append(eval_scores_df)
            if bert_size == 'small' and corpus_name == 'oscar' and vocab_size == 32000:
                fix_partition(eval_scores_df)
                # continue
        # if 'ner' in task:
        #     nb_file_path = tb_path / 'crf' / f'{bert_info}-{schema}-{task}-crf.ipynb'
        #     if not nb_file_path.exists():
        #         # print(f"{nb_file_path} crf path doesn't exist")
        #         continue
        #     eval_scores_df = parse_input_file(nb_file_path)
        #     insert_eval_info(eval_scores_df, True, task, schema, num_epochs, vocab_size, corpus_name, tokenizer_type, bert_size, 'bert')
        #     tb_dataframes.append(eval_scores_df)
    return tb_dataframes


dataframes = []
p = Path('experiments')
for s in ['seg-only', 'seg-ner', 'seg-tag', 'seg-tag-ner', 'seg-tag-feats']:
    for b in ['small', 'basic']:
        for t in ['wordpiece', 'wordpiece_roots']:
            for c in ['oscar', 'owt']:
                for v in ['10000', '32000', '52000', '52000-1e-5', '52000-5e-5', '104000', '104000-5e-5']:
                    for n in ['05', '10', '15']:
                        df = get_task_eval_data(p, s, b, t, 'bert', c, v, n)
                        dataframes.extend(df)
            if b == 'basic' and t == 'wordpiece':
                df = get_task_eval_data(p, s, b, t, 'hebert', c, v, n)
                dataframes.extend(df)
                df = get_task_eval_data(p, s, b, t, 'mbert', c, v, n)
                dataframes.extend(df)
pd.concat(dataframes).to_csv('morph-eval.csv', index=False)
