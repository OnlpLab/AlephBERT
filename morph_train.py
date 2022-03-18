import random
import logging
from pathlib import Path
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
from transformers import BertModel, BertTokenizerFast
from data import preprocess_form, preprocess_labels
from morph_model import BertTokenEmbeddingModel, SegmentDecoder, MorphSequenceModel, MorphPipelineModel
from bclm import treebank as tb, ne_evaluate_mentions
from hebrew_root_tokenizer import AlefBERTRootTokenizer
import utils

# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

# Config
tb_schema = "UD"
# tb_schema = "SPMRL"
# tb_data_src = "UD_Hebrew"
tb_data_src = "for_amit_spmrl"
# tb_data_src = "HebrewTreebank"
# tb_name = "HTB"
tb_name = "hebtb"

# bert_tokenizer_type = 'wordpiece'
bert_tokenizer_type = 'wordpiece_roots'
bert_vocab_size = 10000
epochs = 10
bert_corpus_name = 'oscar'
bert_model_size_type = 'small'
bert_model_name = 'bert'
# bert_model_name = 'heBERT'
# bert_model_name = 'mBERT'
bert_version = f'{bert_model_name}-{bert_model_size_type}-{bert_tokenizer_type}-{bert_corpus_name}-{bert_vocab_size}-{epochs}'
tokenizer_version = f'{bert_model_name}-{bert_tokenizer_type}-{bert_corpus_name}-{bert_vocab_size}'
# tokenizer_version = f'{bert_tokenizer_type}-{bert_corpus_name}-{bert_vocab_size}'
# bert_version = f'{bert_model_name}'
# tokenizer_version = f'{bert_model_name}'

md_strategry = "morph-pipeline"
# md_strategry = "morph-sequence"
# md_strategry = "segment-only"

# Data
raw_root_path = Path(f'data/raw/{tb_data_src}')
if tb_name == 'HTB':
    partition = tb.ud(raw_root_path, tb_name)
elif tb_name == 'hebtb':
    if tb_schema == "UD":
        partition = tb.spmrl_conllu(raw_root_path, tb_name)
    else:
        partition = tb.spmrl(raw_root_path, tb_name)
else:
    partition = {'train': None, 'dev': None, 'test': None}
preprocessed_data_root_path = Path(f'data/preprocessed/{tb_data_src}/{tb_name}/{tokenizer_version}')


def load_preprocessed_data_samples(data_root_path, partition, label_names) -> dict:
    logging.info(f'Loading preprocesssed {tb_schema} form tag data samples')
    xtoken_data = preprocess_form.load_xtoken_data(data_root_path, partition)
    token_char_data = preprocess_form.load_token_char_data(data_root_path, partition)
    form_char_data = preprocess_form.load_form_data(data_root_path, partition)
    label_data = preprocess_labels.load_labeled_data(data_root_path, partition, label_names=label_names)
    datasets = {}
    for part in partition:
        xtoken_tensor = torch.tensor(xtoken_data[part][:, :, 1:], dtype=torch.long)
        token_char_tensor = torch.tensor(token_char_data[part][:, :, :, 1:], dtype=torch.long)
        form_char_tensor = torch.tensor(form_char_data[part], dtype=torch.long)
        label_tensor = torch.tensor(label_data[part], dtype=torch.long)
        datasets[part] = TensorDataset(xtoken_tensor, token_char_tensor, form_char_tensor, label_tensor)
    return datasets


datasets = {}
# label_names = ['tag']
label_names = ['biose_layer0']
# label_names = ['tag', 'biose_layer0']
# label_names = ['tag', 'Gender', 'Number', 'Person', 'Tense']
# label_names = []
# label_names = None
if label_names is None:
    label_names = preprocess_labels.get_label_names(preprocessed_data_root_path, partition)

# Output folder path
out_morph_type = 'morph_seg'
if len(label_names) == 0:
    pass
elif len(label_names) == 1:
    out_morph_type = f'{out_morph_type}_{label_names[0]}'
elif len(label_names) == 2:
    out_morph_type = f'{out_morph_type}_{label_names[0]}_{label_names[1]}'
else:
    if 'tag' in label_names:
        out_morph_type = f'{out_morph_type}_tag_feats'
    else:
        out_morph_type = f'{out_morph_type}_feats'
out_base = Path(f'experiments/{out_morph_type}/{bert_model_name}')
# out_path = out_base / bert_model_type / bert_tokenizer_type / bert_version / tb_data_src / tb_name
out_path = out_base / bert_model_size_type / bert_version / tb_data_src / tb_name
out_path.mkdir(parents=True, exist_ok=True)

data_samples_file_paths = {part: preprocessed_data_root_path / f'{part}_{out_morph_type}_data_samples.pt'
                           for part in partition}
if all([data_samples_file_paths[part].exists() for part in data_samples_file_paths]):
    for part in partition:
        file_path = data_samples_file_paths[part]
        logging.info(f'Loading {tb_schema} {out_morph_type} tensor dataset from {file_path}')
        datasets[part] = torch.load(file_path)
else:
    datasets = load_preprocessed_data_samples(preprocessed_data_root_path, partition, label_names)
    for part in datasets:
        file_path = data_samples_file_paths[part]
        logging.info(f'Saving {tb_schema} {out_morph_type} tensor dataset to {file_path}')
        torch.save(datasets[part], file_path)
# datasets['train'] = TensorDataset(*[t[:100] for t in datasets['train'].tensors])
# datasets['dev'] = TensorDataset(*[t[:100] for t in datasets['dev'].tensors])
# datasets['test'] = TensorDataset(*[t[:100] for t in datasets['test'].tensors])
train_dataloader = DataLoader(datasets['train'], batch_size=1, shuffle=False)
dev_dataloader = DataLoader(datasets['dev'], batch_size=100)
test_dataloader = DataLoader(datasets['test'], batch_size=100)

# Language Model
bert_folder_path = Path(f'./experiments/transformers/{bert_model_name}/{bert_model_size_type}/{bert_tokenizer_type}/{bert_version}')
if bert_tokenizer_type == 'wordpiece_roots':
    bert_folder_path = Path(f'./experiments/transformers/{bert_model_name}/{bert_model_size_type}/wordpiece/{bert_version}')
    logging.info(f'Loading roots tokenizer BERT from: {str(bert_folder_path)}')
    bert_tokenizer = AlefBERTRootTokenizer(str(bert_folder_path / 'vocab.txt'))
    bert = BertModel.from_pretrained(str(bert_folder_path))
elif bert_model_name == 'mBERT':
    logging.info(f'Loading {bert_model_name}')
    bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual')
    bert = BertModel.from_pretrained('bert-base-multilingual')
elif bert_model_name == 'heBERT':
    logging.info(f'Loading {bert_model_name}')
    bert_tokenizer = BertTokenizerFast.from_pretrained(f'avichr/{bert_model_name}')
    bert = BertModel.from_pretrained(f'avichr/{bert_model_name}')
else:
    logging.info(f'Loading BERT from: {str(bert_folder_path)}')
    bert_tokenizer = BertTokenizerFast.from_pretrained(str(bert_folder_path))
    bert = BertModel.from_pretrained(str(bert_folder_path))
logging.info('BERT model and tokenizer loaded')


# Vocabs
pad, sos, eos, sep = '<pad>', '<s>', '</s>', '<sep>'
char_vectors, char_vocab = preprocess_labels.load_char_vocab(preprocessed_data_root_path)
label_vocab = preprocess_labels.load_label_vocab(preprocessed_data_root_path, partition, pad=pad)

# Special symbols
char_sos = torch.tensor([char_vocab['char2id'][sos]], dtype=torch.long)
char_eos = torch.tensor([char_vocab['char2id'][eos]], dtype=torch.long)
char_sep = torch.tensor([char_vocab['char2id'][sep]], dtype=torch.long)
char_pad = torch.tensor([char_vocab['char2id'][pad]], dtype=torch.long)
label_pads = [torch.tensor([label_vocab['labels2id'][l][pad]], dtype=torch.long) for l in label_names]

# MD Model
char_tensors = torch.tensor(char_vectors, dtype=torch.float)
char_emb_pad_id = char_vocab['char2id'][pad]
char_emb = nn.Embedding.from_pretrained(char_tensors, freeze=False, padding_idx=char_emb_pad_id)
# num_labels = {len(label_vocab['labels2id'][name]) for name in label_names}
num_layers = 2
hidden_size = bert.config.hidden_size // num_layers
dropout = 0.1
num_chars = len(char_vocab['char2id'])
out_dropout = 0.5
xtoken_emb = BertTokenEmbeddingModel(bert, bert_tokenizer)
label_classifier_configs = []
for name in label_names:
    config = {'id2label': label_vocab['id2labels'][name]}
    # if name == 'biose_layer0':
    #     config['crf_trans_type'] = 'BIOSE'
    label_classifier_configs.append(config)

if md_strategry == "morph-pipeline":
    segmentor = SegmentDecoder(char_emb, hidden_size, num_layers, dropout, out_dropout, num_chars)
    md_model = MorphPipelineModel(xtoken_emb, segmentor, hidden_size, num_layers, dropout, out_dropout,
                                  label_classifier_configs)
elif md_strategry == "morph-sequence":
    segmentor = SegmentDecoder(char_emb, hidden_size, num_layers, dropout, out_dropout, num_chars,
                               label_classifier_configs)
    md_model = MorphSequenceModel(xtoken_emb, segmentor)
else:
    segmentor = SegmentDecoder(char_emb, hidden_size, num_layers, dropout, out_dropout, num_chars)
    md_model = MorphSequenceModel(xtoken_emb, segmentor)
device = 1
char_special_symbols = {sos: char_sos.to(device), eos: char_eos.to(device),
                        sep: char_sep.to(device), pad: char_pad.to(device)}
if device is not None:
    md_model.to(device)
print(md_model)


# Training and evaluation routine
def process(model: MorphSequenceModel, data: DataLoader, criterion: nn.CrossEntropyLoss, epoch, phase, print_every,
            teacher_forcing_ratio=0.0, optimizer: optim.AdamW = None, max_grad_norm=None):
    print_form_loss, total_form_loss = 0, 0
    print_label_losses, total_label_losses = [0 for _ in range(len(label_names))], [0 for _ in range(len(label_names))]
    print_target_forms, total_target_forms = [], []
    print_target_labels, total_target_labels = [], []
    print_decoded_forms, total_decoded_forms = [], []
    print_decoded_labels, total_decoded_labels = [], []
    print_decoded_lattice_rows, total_decoded_lattice_rows = [], []

    for i, batch in enumerate(data):
        batch = tuple(t.to(device) for t in batch)
        batch_form_scores, batch_label_scores, batch_form_targets, batch_label_targets = [], [], [], []
        batch_token_chars, batch_sent_ids, batch_num_tokens = [], [], []
        for sent_xtoken, sent_token_chars, sent_form_chars, sent_labels in zip(*batch):
            input_token_chars = sent_token_chars[:, :, -1]
            num_tokens = len(sent_token_chars[sent_token_chars[:, 0, 1] > 0])
            target_token_form_chars = sent_form_chars[:, :, -1]
            max_form_len = target_token_form_chars.shape[1]
            target_token_labels = sent_labels[:, :, 2:]
            max_num_labels = target_token_labels.shape[1]
            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
            form_scores, _, label_scores = model(sent_xtoken, input_token_chars, char_special_symbols, num_tokens,
                                                 max_form_len, max_num_labels,
                                                 target_token_form_chars if use_teacher_forcing else None)
            batch_form_scores.append(form_scores)
            batch_label_scores.append(label_scores)
            batch_form_targets.append(target_token_form_chars[:num_tokens])
            batch_label_targets.append(target_token_labels[:num_tokens])
            batch_token_chars.append(input_token_chars[:num_tokens])
            batch_sent_ids.append(sent_form_chars[:, :, 0].unique().item())
            batch_num_tokens.append(num_tokens)

        # Decode
        batch_form_scores = nn.utils.rnn.pad_sequence(batch_form_scores, batch_first=True)
        batch_label_scores = [nn.utils.rnn.pad_sequence(label_scores, batch_first=True)
                              for label_scores in list(map(list, zip(*batch_label_scores)))]
        with torch.no_grad():
            batch_decoded_chars, batch_decoded_labels = model.decode(batch_form_scores, batch_label_scores)

        # Form Loss
        batch_form_targets = nn.utils.rnn.pad_sequence(batch_form_targets, batch_first=True)
        form_loss = model.form_loss(batch_form_scores, batch_form_targets, criterion)
        print_form_loss += form_loss.item()

        # Label Losses
        batch_label_targets = [[t[:, :, j] for j in range(t.shape[-1])] for t in batch_label_targets]
        batch_label_targets = [nn.utils.rnn.pad_sequence(label_targets, batch_first=True)
                               for label_targets in list(map(list, zip(*batch_label_targets)))]
        label_losses = model.labels_losses(batch_label_scores, batch_label_targets, criterion)
        for j in range(len(label_losses)):
            print_label_losses[j] += label_losses[j].item()

        # Optimization Step
        if optimizer is not None:
            form_loss.backward(retain_graph=len(label_losses) > 0)
            for j in range(len(label_losses)):
                label_losses[j].backward(retain_graph=(j < len(label_losses)-1))
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        # To Lattice
        for j in range(len(batch_sent_ids)):
            sent_id = batch_sent_ids[j]
            input_chars = batch_token_chars[j]
            target_form_chars = batch_form_targets[j]
            target_labels = [label_targets[j] for label_targets in batch_label_targets]
            decoded_form_chars = batch_decoded_chars[j]
            decoded_labels = [decoded_labels[j] for decoded_labels in batch_decoded_labels]
            num_tokens = batch_num_tokens[j]
            input_chars = input_chars.to('cpu')
            target_form_chars = target_form_chars[:num_tokens].to('cpu')
            decoded_form_chars = decoded_form_chars[:num_tokens].to('cpu')
            target_labels = [labels[:num_tokens].to('cpu') for labels in target_labels]
            decoded_labels = [labels[:num_tokens].to('cpu') for labels in decoded_labels]
            input_tokens = utils.to_sent_tokens(input_chars, char_vocab['id2char'])
            target_morph_segments = utils.to_token_morph_segments(target_form_chars,
                                                                        char_vocab['id2char'],
                                                                        char_eos, char_sep)
            decoded_morph_segments = utils.to_token_morph_segments(decoded_form_chars,
                                                                         char_vocab['id2char'],
                                                                         char_eos, char_sep)
            target_morph_labels = utils.to_token_morph_labels(target_labels, label_names,
                                                                    label_vocab['id2labels'],
                                                                    label_pads)
            decoded_morph_labels = utils.to_token_morph_labels(decoded_labels, label_names,
                                                                     label_vocab['id2labels'],
                                                                     label_pads)

            decoded_token_lattice_rows = (sent_id, input_tokens, decoded_morph_segments, decoded_morph_labels)
            print_decoded_lattice_rows.append(decoded_token_lattice_rows)
            print_target_forms.append(target_morph_segments)
            print_target_labels.append(target_morph_labels)
            print_decoded_forms.append(decoded_morph_segments)
            print_decoded_labels.append(decoded_morph_labels)

        # Log Print Eval
        if (i + 1) % print_every == 0:
            sent_id, input_tokens, decoded_segments, decoded_labels = print_decoded_lattice_rows[-1]
            target_segments = print_target_forms[-1]
            target_labels = print_target_labels[-1]
            decoded_segments = print_decoded_forms[-1]
            decoded_labels = print_decoded_labels[-1]

            print(f'epoch {epoch} {phase}, batch {i + 1} form char loss: {print_form_loss / print_every}')
            for j in range(len(label_names)):
                print(f'epoch {epoch} {phase}, batch {i + 1} {label_names[j]} loss: {print_label_losses[j] / print_every}')
            print(f'epoch {epoch} {phase}, batch {i + 1} sent #{sent_id} input tokens  : {input_tokens}')
            print(f'epoch {epoch} {phase}, batch {i + 1} sent #{sent_id} target forms  : {list(reversed(target_segments))}')
            print(f'epoch {epoch} {phase}, batch {i + 1} sent #{sent_id} decoded forms : {list(reversed(decoded_segments))}')
            for j in range(len(label_names)):
                target_values = [labels[j] for labels in target_labels]
                print(f'epoch {epoch} {phase}, batch {i + 1} sent #{sent_id} target {label_names[j]} labels  : {list(reversed([target_values]))}')
                decoded_values = [labels[j] for labels in decoded_labels]
                print(f'epoch {epoch} {phase}, batch {i + 1} sent #{sent_id} decoded {label_names[j]} labels : {list(reversed([decoded_values]))}')
            total_form_loss += print_form_loss
            for j, label_loss in enumerate(print_label_losses):
                total_label_losses[j] += label_loss
            print_form_loss = 0
            print_label_losses = [0 for _ in range(len(label_names))]

            total_decoded_forms.extend(print_decoded_forms)
            total_decoded_labels.extend(print_decoded_labels)
            total_target_forms.extend(print_target_forms)
            total_target_labels.extend(print_target_labels)
            total_decoded_lattice_rows.extend(print_decoded_lattice_rows)

            aligned_scores, mset_scores = utils.morph_eval(print_decoded_forms, print_target_forms)
            # print(f'epoch {epoch} {phase}, batch {i + 1} form aligned scores: {aligned_scores}')
            print(f'epoch {epoch} {phase}, batch {i + 1} form mset scores: {mset_scores}')

            for j in range(len(label_names)):
                if label_names[j][:3].lower() in ['tag', 'bio', 'gen', 'num', 'per', 'ten']:
                    decoded_values = [labels[j] for sent_labels in print_decoded_labels for labels in sent_labels]
                    target_values = [labels[j] for sent_labels in print_target_labels for labels in sent_labels]
                    aligned_scores, mset_scores = utils.morph_eval(decoded_values, target_values)
                    # print(f'epoch {epoch} {phase}, batch {i + 1} {label_names[j]} aligned scores: {aligned_scores}')
                    print(f'epoch {epoch} {phase}, batch {i + 1} {label_names[j]} mset scores: {mset_scores}')

            print_target_forms = []
            print_target_labels = []
            print_decoded_forms = []
            print_decoded_labels = []
            print_decoded_lattice_rows = []

    # Log Total Eval
    if print_form_loss > 0:
        total_form_loss += print_form_loss
        for j, label_loss in enumerate(print_label_losses):
            total_label_losses[j] += label_loss
        total_decoded_forms.extend(print_decoded_forms)
        total_decoded_labels.extend(print_decoded_labels)
        total_target_forms.extend(print_target_forms)
        total_target_labels.extend(print_target_labels)
        total_decoded_lattice_rows.extend(print_decoded_lattice_rows)

    print(f'epoch {epoch} {phase}, total form char loss: {total_form_loss / len(data)}')
    for j in range(len(label_names)):
        print(f'epoch {epoch} {phase}, total {label_names[j]} loss: {total_label_losses[j] / len(data)}')

    for j in range(len(label_names)):
        if label_names[j][:3].lower() in ['tag', 'bio', 'gen', 'num', 'per', 'ten']:
            decoded_values = [labels[j] for sent_labels in total_decoded_labels for labels in sent_labels]
            target_values = [labels[j] for sent_labels in total_target_labels for labels in sent_labels]
            aligned_scores, mset_scores = utils.morph_eval(decoded_values, target_values)
            # print(f'epoch {epoch} {phase}, total {label_names[j]} aligned scores: {aligned_scores}')
            print(f'epoch {epoch} {phase}, total {label_names[j]} mset scores: {mset_scores}')

    return utils.get_lattice_data(total_decoded_lattice_rows, label_names)


eval_fields = ['form']
if 'tag' in label_names:
    eval_fields.append('tag')
if len([name for name in label_names if name not in ['biose_layer0', 'tag']]) > 0:
    eval_fields.append('feats')

# Optimizer
epochs = 3
max_grad_norm = 1.0
lr = 1e-3
# freeze bert
for param in bert.parameters():
    param.requires_grad = False
parameters = list(filter(lambda p: p.requires_grad, md_model.parameters()))
# parameters = morph_tagger_model.parameters()
adam = optim.AdamW(parameters, lr=lr)
loss_fct = nn.CrossEntropyLoss(ignore_index=0)
teacher_forcing_ratio = 1.0

# Training epochs
for i in trange(epochs, desc="Epoch"):
    epoch = i + 1
    md_model.train()
    process(md_model, train_dataloader, loss_fct, epoch, 'train', 10, teacher_forcing_ratio, adam, max_grad_norm)
    md_model.eval()
    with torch.no_grad():
        dev_samples = process(md_model, dev_dataloader, loss_fct, epoch, 'dev', 1)
        dev_samples.to_csv(out_path / 'dev_samples.csv')
        utils.print_eval_scores(decoded_df=dev_samples, truth_df=partition['dev'], phase='dev', step=epoch,
                                      fields=eval_fields)
        test_samples = process(md_model, test_dataloader, loss_fct, epoch, 'test', 1)
        test_samples.to_csv(out_path / 'test_samples.csv')
        utils.print_eval_scores(decoded_df=test_samples, truth_df=partition['test'], phase='test', step=epoch,
                                      fields=eval_fields)

        if 'biose_layer0' in label_names:
            utils.save_ner(dev_samples, out_path / 'morph_label_dev.bmes', 'biose_layer0')
            dev_gold_file_path = Path(f'data/raw/{tb_data_src}/{tb_name}/gold/morph_gold_dev.bmes')
            dev_pred_file_path = out_path / 'morph_label_dev.bmes'
            print(ne_evaluate_mentions.evaluate_files(dev_gold_file_path, dev_pred_file_path))
            print(ne_evaluate_mentions.evaluate_files(dev_gold_file_path, dev_pred_file_path, ignore_cat=True))

            utils.save_ner(test_samples, out_path / 'morph_label_test.bmes', 'biose_layer0')
            test_gold_file_path = Path(f'data/raw/{tb_data_src}/{tb_name}/gold/morph_gold_test.bmes')
            test_pred_file_path = out_path / 'morph_label_test.bmes'
            print(ne_evaluate_mentions.evaluate_files(test_gold_file_path, test_pred_file_path))
            print(ne_evaluate_mentions.evaluate_files(test_gold_file_path, test_pred_file_path, ignore_cat=True))
