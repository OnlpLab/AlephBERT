from data.preprocess_form import *
from data.preprocess_labels import *
from bclm import treebank as tb
from hebrew_root_tokenizer import AlefBERTRootTokenizer


pad, sos, eos, sep = '<pad>', '<s>', '</s>', '<sep>'


if __name__ == '__main__':
    # Setup logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    # tokenizer_type = 'wordpiece'
    tokenizer_type = 'wordpiece_roots'
    transformer_type = 'bert'
    # transformer_type = 'mBERT'
    # transformer_type = 'heBERT'
    vocab_size = 10000
    corpus_name = 'oscar'
    tokenizer_version = f'{tokenizer_type}-{corpus_name}-{vocab_size}'
    if transformer_type == 'bert':
        transformer_type = f'{transformer_type}-{tokenizer_version}'
    else:
        tokenizer_version = transformer_type

    dev_root_path = Path('/home/amit/dev')
    tb_root_path = dev_root_path / 'onlplab'

    # tb_root_path = tb_root_path / 'UniversalDependencies'
    tb_root_path = tb_root_path / 'HebrewResources/for_amit_spmrl'
    # tb_root_path = tb_root_path / 'HebrewResources/HebrewTreebank'

    # raw_root_path = Path('data/raw/UD_Hebrew')
    raw_root_path = Path('data/raw/for_amit_spmrl')
    # raw_root_path = Path('data/raw/HebrewTreebank')

    # preprocessed_root_path = Path(f'data/preprocessed/UD_Hebrew/HTB/{transformer_type}')
    preprocessed_root_path = Path(f'data/preprocessed/for_amit_spmrl/hebtb/{transformer_type}')
    # preprocessed_root_path = Path(f'data/preprocessed/HebrewTreebank/hebtb/{transformer_type}')
    preprocessed_root_path.mkdir(parents=True, exist_ok=True)

    if not raw_root_path.exists():
        # raw_partition = tb.ud(raw_root_path, 'HTB', tb_root_path)
        raw_partition = tb.spmrl_ner_conllu(raw_root_path, 'hebtb', tb_root_path)
        # raw_partition = tb.spmrl(raw_root_path, 'hebtb', tb_root_path)
    else:
        # raw_partition = tb.ud(raw_root_path, 'HTB')
        raw_partition = tb.spmrl_ner_conllu(raw_root_path, 'hebtb')
        # raw_partition = tb.spmrl(raw_root_path, 'hebtb')

    bert_root_path = Path(f'./experiments/tokenizers/{tokenizer_type}/{tokenizer_version}')
    if tokenizer_type == 'wordpiece_roots':
        bert_root_path = Path(f'./experiments/tokenizers/wordpiece/{tokenizer_version}')
        bert_tokenizer = AlefBERTRootTokenizer(str(bert_root_path / 'vocab.txt'))
    elif tokenizer_type == 'mBERT':
        bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')
    elif tokenizer_type == 'heBERT':
        bert_tokenizer = BertTokenizerFast.from_pretrained(f'avichr/{tokenizer_type}')
    else:
        bert_tokenizer = BertTokenizerFast.from_pretrained(str(bert_root_path))

    morph_data = get_morph_data(preprocessed_root_path, raw_partition)
    morph_form_char_data = get_form_char_data(preprocessed_root_path, morph_data, sep=sep, eos=eos)
    token_char_data = get_token_char_data(preprocessed_root_path, morph_data)
    xtoken_df = get_xtoken_data(preprocessed_root_path, morph_data, bert_tokenizer, sos=sos, eos=eos)

    ft_root_path = dev_root_path / 'facebookresearch' / 'fastText'
    save_char_vocab(preprocessed_root_path, ft_root_path, raw_partition, pad=pad, sep=sep, sos=sos, eos=eos)
    char_vectors, char_vocab = load_char_vocab(preprocessed_root_path)
    label_vocab = load_label_vocab(preprocessed_root_path, morph_data, pad=pad)

    save_xtoken_data_samples(preprocessed_root_path, xtoken_df, bert_tokenizer, pad=pad)
    save_token_char_data_samples(preprocessed_root_path, token_char_data, char_vocab['char2id'], pad=pad)
    save_form_char_data_samples(preprocessed_root_path, morph_form_char_data, char_vocab['char2id'], pad=pad)
    save_labeled_data_samples(preprocessed_root_path, morph_data, label_vocab['labels2id'], pad=pad)
