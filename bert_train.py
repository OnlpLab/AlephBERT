import logging
from pathlib import Path
from datasets import load_dataset
from transformers import BertConfig, TrainingArguments, set_seed
from transformers.trainer import Trainer
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from transformers.models.bert.modeling_bert import BertForMaskedLM
from hebrew_root_tokenizer import AlefBERTRootTokenizer


def run_on_ram():
    import tempfile
    print(tempfile.tempdir)
    tempfile.tempdir = '/dev/shm'
    print(tempfile.tempdir)


def get_config():
    return BertConfig(
        vocab_size=vocab_size,
        max_position_embeddings=512,
        num_attention_heads=12,
        num_hidden_layers=num_hidden_layers,
        type_vocab_size=1,
    )


def get_tokenizer():
    tokenizer_root_path = Path('experiments/tokenizers/bert') / tokenizer_type
    pretrained_tokenizer_path = tokenizer_root_path / f'bert-{tokenizer_type}-{data_source_name}-{vocab_size}'
    if 'roots' in tokenizer_type:
        logger.info(f'loading AlefBERTRootTokenizer from {pretrained_tokenizer_path}')
        return AlefBERTRootTokenizer(pretrained_tokenizer_path / 'vocab.txt')
    else:
        logger.info(f'loading BertTokenizerFast from {pretrained_tokenizer_path}')
        return BertTokenizerFast.from_pretrained(str(pretrained_tokenizer_path), max_len=512)


def get_model(model_path=None):
    if model_path is None:
        config = get_config()
        return BertForMaskedLM(config=config)
    logging.info('Loading pre-trained AlephBERT')
    bert = BertForMaskedLM.from_pretrained(str(model_path))
    bert_tokenizer = BertTokenizerFast.from_pretrained(str(model_path))
    return bert, bert_tokenizer


def get_train_data(max_length, min_length=0):
    # paths = [str(x) for x in Path("/dev/shm/amitse").glob("*.*")]
    # paths = ['data/raw/oscar/he_dedup.txt', 'data/raw/wikipedia/wikipedia.raw',
    #          'data/raw/twitter/hebrew_tweets_text_clean_full.txt']
    paths = ['data/raw/oscar/he_dedup.txt']
    logger.info(f'loading training data from: {paths}')
    # ds = load_dataset('text', data_files=[str(p)], cache_dir='/dev/shm/amitse/.cache')
    ds = load_dataset('text', data_files=paths)

    def tokenize_function(examples):
        examples["text"] = [line for line in examples["text"] if len(line.split()) > 1]
        return tokenizer(examples["text"], add_special_tokens=True, return_special_tokens_mask=False,
                         return_length=True, return_token_type_ids=False, return_attention_mask=False)
    return ds.map(
        tokenize_function,
        batched=True,
        num_proc=8,
    ).filter(lambda e: min_length < e['length'] < max_length)


def get_data_collator():
    return DataCollatorForLanguageModeling(tokenizer=tokenizer)


def get_train_args(lr=1e-4):
    train_root_path = Path('experiments/transformers/bert') / bert_model_size_type / tokenizer_type
    p = train_root_path / f'bert-{bert_model_size_type}-{tokenizer_type}-{data_source_name}-{vocab_size}-05-64'
    # p = train_root_path / f'bert-{bert_model_size_type}-{tokenizer_type}-{data_source_name}-{vocab_size}-05-128'
    # p = train_root_path / f'bert-{bert_model_size_type}-{tokenizer_type}-{data_source_name}-{vocab_size}-05'
    p.mkdir(parents=True, exist_ok=True)
    return TrainingArguments(
        output_dir=str(p),
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=48,
        gradient_accumulation_steps=5,
        save_total_limit=0,
        save_steps=0,
        learning_rate=lr,
        # fp16=True,
        dataloader_num_workers=8
    )


# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

# data_source_name = 'owt'
data_source_name = 'oscar'
# tokenizer_type = 'wordpiece'
tokenizer_type = 'wordpiece_roots'
vocab_size = 10000
num_hidden_layers = 6
bert_model_size_type = 'small' if num_hidden_layers == 6 else 'basic'

# model_root_path = Path('experiments/transformers/bert') / bert_model_size_type / tokenizer_type
# model_path = model_root_path / f'bert-{bert_model_size_type}-{tokenizer_type}-{data_source_name}-{vocab_size}-05-64'
# model, tokenizer = get_model(model_path)
tokenizer = get_tokenizer()
model = get_model()

data_collator = get_data_collator()
train_dataset = get_train_data(64)
# train_dataset = get_train_data(128, 64)
# train_dataset = get_train_data(512, 128)
print(train_dataset['train'])

training_args = get_train_args()
length_series = train_dataset['train'].data[1].to_pandas()
print(f'num samples: {len(length_series)}')
print(f'avg sample length: {length_series.mean(axis=0)}')
print(f'sample length stddev: {length_series.std(axis=0)}')

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset['train'],
)

set_seed(42)
trainer.train()
trainer.save_model()
tokenizer.save_pretrained(training_args.output_dir)
