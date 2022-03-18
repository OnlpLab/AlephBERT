import gzip
from pathlib import Path
import fasttext
import numpy as np
import logging


ft_models = {}
ft_pad_vector = np.zeros(300, dtype=np.float)


def _save_to(path: Path, lines: list):
    if path.suffix == ".gz":
        with gzip.open(str(path), "wt") as f:
            for line in lines:
                f.write(line)
                f.write('\n')
    else:
        with open(str(path), 'w') as f:
            for line in lines:
                f.write(line)
                f.write('\n')


def _load_from(path: Path) -> list:
    if path.suffix == ".gz":
        with gzip.open(str(path), "rt") as f:
            lines = f.readlines()
    else:
        with open(str(path), 'r') as f:
            lines = f.readlines()
    return lines


def _save_vec_file(path: Path, words: list, vectors: list):
    lines = []
    for word, vector in zip(words, vectors):
        vector_str = ' '.join([str(num) for num in vector])
        line = f"{word} {vector_str}"
        lines.append(line)
    _save_to(path, lines)


def _load_vec_file(path: Path) -> (dict, list):
    word_indices = {}
    vectors = []
    lines = _load_from(path)
    for i, line in enumerate(lines):
        parts = line.split()
        word = parts[0]
        vector = [float(v) for v in parts[1:]]
        vectors.append(vector)
        word_indices[word] = i
    return word_indices, vectors


def load_word_vectors(vec_file_path: Path) -> (np.array, dict):
    logging.info(f'Loading FastText vectors from {vec_file_path}')
    word2index, vectors = _load_vec_file(vec_file_path)
    word_vectors = np.array([vectors[word2index[word]] for word in word2index], dtype=np.float)
    return word_vectors, word2index


def get_word_vectors(lang, model_path, words: list):
    global ft_models
    if lang not in ft_models:
        logging.info(f'Loading FastText model from {model_path}')
        ft_models[lang] = fasttext.load_model(f'{model_path}')
    word2index = {word: i + 1 for i, word in enumerate(words)}
    word_vectors = np.stack([ft_models[lang].get_word_vector(word) if word != '<pad>' else ft_pad_vector
                             for word in words], axis=0)
    return word_vectors, word2index


def save_word_vectors(vec_file_path: Path, word_vectors, word2index):
    index2word = {word2index[word]: word for word in word2index}
    with open(str(vec_file_path), 'w') as f:
        for ind, vec in enumerate(word_vectors):
            word = index2word[ind+1]
            line = ' '.join([word] + [str(v) for v in vec.tolist()])
            f.write(line)
            f.write('\n')
