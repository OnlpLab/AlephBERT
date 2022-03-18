import random
import logging
from pathlib import Path
import torch
import torch.optim as optim
import torch.nn as nn
from conditional_random_field import ConditionalRandomField, allowed_transitions
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
from transformers import BertModel
from data import preprocess_morph_tag
from collections import Counter
import pandas as pd
from bclm import treebank as tb


# Logging setup
from model_tag import TokenTagsDecoder, TaggerModel

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

schema = "UD"
# schema = "SPMRL"
data_src = "UD_Hebrew"
# data_src = "HebrewTreebank"
# data_src = "for_amit_spmrl"
tb_name = "HTB"
# tb_name = "hebtb"
raw_root_path = Path(f'data/raw/{data_src}')

# Data
if tb_name == 'HTB':
    partition = tb.ud(raw_root_path, tb_name)
elif tb_name == 'hebtb':
    if schema == "UD":
        partition = tb.spmrl_conllu(raw_root_path, tb_name)
    else:
        partition = tb.spmrl(raw_root_path, tb_name)
else:
    partition = {'train': None, 'dev': None, 'test': None}
tokenizer_type = 'wordpiece'
vocab_size = 52000
corpus_name = 'oscar'
bert_model_size_type = 'small'
# bert_version = 'mbert'
bert_version = f'bert-{bert_model_size_type}-{tokenizer_type}-{corpus_name}-{vocab_size}'
preprocessed_data_root_path = Path(f'data/preprocessed/{data_src}/{tb_name}/{bert_version}')


def load_preprocessed_data_samples(data_root_path, partition):
    logging.info(f'Loading preprocesssed {schema} tag data samples')
    xtoken_data = preprocess_morph_tag.load_xtoken_data(data_root_path, partition)
    token_char_data = preprocess_morph_tag.load_token_char_data(data_root_path, partition)
    tag_data = preprocess_morph_tag.load_morph_tag_data(data_root_path, partition)
    datasets = {}
    for part in partition:
        xtoken_tensor = torch.tensor(xtoken_data[part][:, :, 1:], dtype=torch.long)
        token_char_tensor = torch.tensor(token_char_data[part][:, :, :, 1:], dtype=torch.long)
        morph_tag_tensor = torch.tensor(tag_data[part], dtype=torch.long)
        datasets[part] = TensorDataset(xtoken_tensor, token_char_tensor, morph_tag_tensor)
    return datasets


datasets = {}
data_samples_file_paths = {part: preprocessed_data_root_path / f'{part}_tag_data_samples.pt' for part in partition}
if all([data_samples_file_paths[part].exists() for part in data_samples_file_paths]):
    for part in partition:
        file_path = data_samples_file_paths[part]
        logging.info(f'Loading {schema} tag tensor dataset to file {file_path}')
        datasets[part] = torch.load(file_path)
else:
    datasets = load_preprocessed_data_samples(preprocessed_data_root_path, partition)
    for part in datasets:
        file_path = data_samples_file_paths[part]
        logging.info(f'Saving {schema} tag tensor dataset to file {file_path}')
        torch.save(datasets[part], file_path)
train_dataloader = DataLoader(datasets['train'], batch_size=1, shuffle=False)
dev_dataloader = DataLoader(datasets['dev'], batch_size=1)
test_dataloader = DataLoader(datasets['test'], batch_size=1)

# Language Model
bert_folder_path = Path(f'./experiments/transformers/bert/{bert_model_size_type}/{tokenizer_type}/{bert_version}')
logging.info(f'BERT folder path: {str(bert_folder_path)}')
bert = BertModel.from_pretrained(str(bert_folder_path))
# bert_tokenizer = BertTokenizerFast.from_pretrained(str(bert_folder_path))
logging.info('BERT model and tokenizer loaded')

# Morph Segmentation Model
char_vectors, char_vocab, tag_vocab, _ = preprocess_morph_tag.load_morph_vocab(preprocessed_data_root_path, partition,
                                                                               include_eos=True)
char_emb = nn.Embedding.from_pretrained(torch.tensor(char_vectors, dtype=torch.float), freeze=False,
                                        padding_idx=char_vocab['char2id']['<pad>'])
num_tags = len(tag_vocab['tag2id'])
tag_emb = nn.Embedding(num_embeddings=num_tags, embedding_dim=50, padding_idx=0)
num_layers = 2
hidden_size = bert.config.hidden_size // num_layers
dropout = 0.1
out_dropout = 0.5
tag_dec = TokenTagsDecoder(tag_emb, hidden_size, num_layers, dropout, num_tags, out_dropout)
if data_src == "for_amit_spmrl":
    crf = ConditionalRandomField(num_tags=num_tags, constraints=allowed_transitions(constraint_type="BIOSE",
                                                                                    labels=tag_vocab['id2tag']))
else:
    crf = ConditionalRandomField(num_tags=num_tags)
tagger_model = TaggerModel(bert, tag_dec, crf)
device = None
if device is not None:
    tagger_model.to(device)
print(tagger_model)

# Special symbols
sos = torch.tensor([tag_vocab['tag2id']['<s>']], dtype=torch.long)
eos = torch.tensor([tag_vocab['tag2id']['</s>']], dtype=torch.long)
pad = torch.tensor([tag_vocab['tag2id']['<pad>']], dtype=torch.long)
special_symbols = {'<s>': sos.to(device), '</s>': eos.to(device), '<pad>': pad.to(device)}


def to_sent_tokens(token_chars):
    tokens = []
    for chars in token_chars:
        token = ''.join([char_vocab['id2char'][c] for c in chars[chars > 0].tolist()])
        tokens.append(token)
    return tokens


def to_sent_token_tags(sent_token_tags):
    tokens = []
    token_mask = torch.nonzero(torch.eq(sent_token_tags, eos))
    token_mask_map = {m[0].item(): m[1].item() for m in token_mask}
    for i, token_tags in enumerate(sent_token_tags):
        token_len = token_mask_map[i] if i in token_mask_map else sent_token_tags.shape[1]
        token_tags = token_tags[:token_len]
        tags = [tag_vocab['id2tag'][t.item()] for t in token_tags]
        tokens.append(tags)
    return tokens


def to_sent_token_tag_lattice_rows(sent_id, tokens, token_tags):
    rows = []
    node_id = 0
    for token_id, (token, tags) in enumerate(zip(tokens, token_tags)):
        for tag in tags:
            row = [sent_id, node_id, node_id+1, '_', '_', tag, '_', token_id+1, token, True]
            rows.append(row)
            node_id += 1
    return rows


def get_morph_tag_lattice_data(sent_token_seg_rows):
    lattice_rows = []
    for row in sent_token_seg_rows:
        lattice_rows.extend(to_sent_token_tag_lattice_rows(*row))
    return pd.DataFrame(lattice_rows,
                        columns=['sent_id', 'from_node_id', 'to_node_id', 'form', 'lemma', 'tag', 'feats', 'token_id',
                                 'token', 'is_gold'])


def morph_eval(decoded_sent_tokens, target_sent_tokens):
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


def print_eval_scores(decoded_df, truth_df, step):
    aligned_scores, mset_scores = tb.morph_eval(pred_df=decoded_df, gold_df=truth_df, fields=['tag'])
    for fs in aligned_scores:
        p, r, f = aligned_scores[fs]
        print(f'eval {step} aligned {fs}: [P: {p}, R: {r}, F: {f}]')
        p, r, f = mset_scores[fs]
        print(f'eval {step} mset    {fs}: [P: {p}, R: {r}, F: {f}]')


# Training and evaluation routine
def process(model: TaggerModel, data: DataLoader, criterion: nn.CrossEntropyLoss, epoch, phase, print_every,
            teacher_forcing_ratio=0.0, optimizer=None, max_grad_norm=None):
    print_loss, total_loss = 0, 0
    print_target_tags, total_target_tags = [], []
    print_decoded_tags, total_decoded_tags = [], []
    print_decoded_lattice_rows, total_decoded_lattice_rows = [], []

    for i, batch in enumerate(data):
        batch = tuple(t.to(device) for t in batch)
        batch_scores, batch_targets, batch_token_chars = [], [], []
        batch_sent_ids, batch_num_tokens, batch_target_masks = [], [], []
        for sent_token_ctx, sent_token_chars, sent_tags in zip(model.embed_xtokens(batch[0]), batch[1], batch[2]):
            input_token_chars = sent_token_chars[:, :, -1]
            num_tokens = len(sent_token_chars[sent_token_chars[:, 0, 1] > 0])
            target_token_tags = sent_tags[:, :, -1]
            max_tag_len = target_token_tags.shape[-1]
            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
            sent_scores = model(sent_token_ctx, special_symbols, num_tokens, max_tag_len,
                                target_token_tags if use_teacher_forcing else None)
            batch_scores.append(sent_scores)
            batch_targets.append(target_token_tags[:num_tokens])
            batch_token_chars.append(input_token_chars[:num_tokens])
            batch_sent_ids.append(sent_tags[:, :, 0].unique().item())
            batch_num_tokens.append(num_tokens)
            batch_target_mask = torch.eq(target_token_tags[:num_tokens], special_symbols['</s>'])
            batch_target_mask = ~torch.cumsum(batch_target_mask, dim=-1).bool()
            batch_target_masks.append(batch_target_mask)

        # Decode
        batch_scores = nn.utils.rnn.pad_sequence(batch_scores, batch_first=True)
        with torch.no_grad():
            batch_decoded_tags = model.decode(batch_scores)
            if model.crf is not None:
                crf_batch_masks = torch.eq(batch_decoded_tags, special_symbols['</s>'])
                crf_batch_masks = ~torch.cumsum(crf_batch_masks, dim=2).bool()
                crf_batch_masked_scores = [scores[mask] for scores, mask in zip(batch_scores, crf_batch_masks)]
                crf_batch_scores = nn.utils.rnn.pad_sequence(crf_batch_masked_scores, batch_first=True)
                # crf_batch_token_masks = [torch.ones_like(scores[:, 0], dtype=torch.bool)
                #                          for scores in crf_batch_masked_scores]
                crf_batch_token_masks = [mask[mask] for mask in crf_batch_masks]
                crf_batch_token_masks = nn.utils.rnn.pad_sequence(crf_batch_token_masks, batch_first=True)
                crf_batch_decoded_tags = model.crf.viterbi_tags(logits=crf_batch_scores, mask=crf_batch_token_masks)
                for decoded_tags, crf_decoded_tags, crf_mask in zip(batch_decoded_tags, crf_batch_decoded_tags, crf_batch_masks):
                    idxs_vals = [torch.unique_consecutive(mask, return_counts=True) for mask in crf_mask]
                    idxs = torch.cat([idx for idx, _ in idxs_vals])
                    vals = torch.cat([val for _, val in idxs_vals])
                    decoded_token_tags = torch.split_with_sizes(torch.tensor(crf_decoded_tags[0]), tuple(vals[idxs]))
                    for idx, token_tags in enumerate(decoded_token_tags):
                        decoded_tags[idx, :len(token_tags)] = token_tags

        # Loss
        batch_targets = nn.utils.rnn.pad_sequence(batch_targets, batch_first=True)
        batch_target_masks = nn.utils.rnn.pad_sequence(batch_target_masks, batch_first=True)
        loss_batch_targets = batch_targets.view(-1)
        loss_batch_scores = batch_scores.view(-1, batch_scores.shape[-1])
        loss = criterion(loss_batch_scores, loss_batch_targets)
        crf_loss = None
        if model.crf is not None:
            crf_batch_masked_targets = [target[mask] for target, mask in zip(batch_targets, batch_target_masks)]
            crf_batch_masked_scores = [scores[mask] for scores, mask in zip(batch_scores, batch_target_masks)]
            crf_loss_batch_targets = nn.utils.rnn.pad_sequence(crf_batch_masked_targets, batch_first=True)
            crf_loss_batch_scores = nn.utils.rnn.pad_sequence(crf_batch_masked_scores, batch_first=True)
            crf_loss_batch_mask = torch.ne(crf_loss_batch_targets, 0)
            crf_log_likelihood = model.crf(inputs=crf_loss_batch_scores, tags=crf_loss_batch_targets,
                                           mask=crf_loss_batch_mask)
            crf_log_likelihood /= torch.sum(crf_loss_batch_mask)
            crf_loss = -crf_log_likelihood
        print_loss += loss.item()
        if optimizer is not None:
            loss.backward(retain_graph=crf_loss is not None)
            if crf_loss is not None:
                crf_loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        # To Lattice
        for sent_id, input_chars, target_tags, decoded_tags, num_tokens in zip(batch_sent_ids, batch_token_chars,
                                                                               batch_targets, batch_decoded_tags,
                                                                               batch_num_tokens):
            input_chars = input_chars.to('cpu')
            target_tags = target_tags[:num_tokens].to('cpu')
            decoded_tags = decoded_tags[:num_tokens].to('cpu')
            input_tokens = to_sent_tokens(input_chars)
            target_tags = to_sent_token_tags(target_tags)
            decoded_tags = to_sent_token_tags(decoded_tags)
            decoded_token_lattice_rows = (sent_id, input_tokens, decoded_tags)
            print_target_tags.append(target_tags)
            print_decoded_tags.append(decoded_tags)
            print_decoded_lattice_rows.append(decoded_token_lattice_rows)

        # Log Print Eval
        if (i + 1) % print_every == 0:
            sent_id, input_tokens, decoded_tags = print_decoded_lattice_rows[-1]
            target_tags = print_target_tags[-1]
            print(f'epoch {epoch} {phase}, step {i + 1} tag loss: {print_loss / print_every}')
            print(f'sent #{sent_id} input tokens  : {input_tokens}')
            print(f'sent #{sent_id} target tags  : {list(reversed(target_tags))}')
            print(f'sent #{sent_id} decoded tags : {list(reversed(decoded_tags))}')
            total_loss += print_loss
            print_loss = 0

            aligned_scores, mset_scores = morph_eval(print_decoded_tags, print_target_tags)
            print(aligned_scores)
            print(mset_scores)
            total_decoded_tags.extend(print_decoded_tags)
            total_target_tags.extend(print_target_tags)
            total_decoded_lattice_rows.extend(print_decoded_lattice_rows)
            print_target_tags = []
            print_decoded_tags = []
            print_decoded_lattice_rows = []

    # Log Total Eval
    if print_loss > 0:
        total_loss += print_loss
        total_target_tags.extend(print_target_tags)
        total_decoded_tags.extend(print_decoded_tags)
        total_decoded_lattice_rows.extend(print_decoded_lattice_rows)
    print(f'epoch {epoch} {phase}, total tag loss: {total_loss / len(data)}')
    aligned_scores, mset_scores = morph_eval(total_decoded_tags, total_target_tags)
    print(aligned_scores)
    print(mset_scores)
    return get_morph_tag_lattice_data(total_decoded_lattice_rows)


# Optimization
epochs = 3
max_grad_norm = 1.0
lr = 1e-3
# freeze bert
for param in bert.parameters():
    param.requires_grad = False
parameters = list(filter(lambda p: p.requires_grad, tagger_model.parameters()))
# parameters = morph_model.parameters()
adam = optim.AdamW(parameters, lr=lr)
loss_fct = nn.CrossEntropyLoss(ignore_index=0)
teacher_forcing_ratio = 1.0


# Training epochs
for i in trange(epochs, desc="Epoch"):
    epoch = i + 1
    tagger_model.train()
    process(tagger_model, train_dataloader, loss_fct, epoch, 'train', 100, teacher_forcing_ratio, adam, max_grad_norm)
    tagger_model.eval()
    with torch.no_grad():
        dev_samples = process(tagger_model, dev_dataloader, loss_fct, epoch, 'dev', 100)
        print_eval_scores(decoded_df=dev_samples, truth_df=partition['dev'], step=epoch)
        test_samples = process(tagger_model, test_dataloader, loss_fct, epoch, 'test', 100)
        print_eval_scores(decoded_df=test_samples, truth_df=partition['test'], step=epoch)
# dev_samples.to_csv(out_path / 'dev.lattices.csv', index=False)
# test_samples.to_csv(out_path / 'test.lattices.csv', index=False)
# torch.save(morph_model.state_dict(), out_path / 'tagger-model-state.pt')
# torch.save(morph_model, out_path / 'tagger-model.pt')


def test():
    out_path = Path(f'experiments/morph-seg/bert/small/wordpiece/{bert_version}/UD_Hebrew/HTB')
    m = torch.load(out_path / 'tagger-model.pt', map_location=torch.device('cpu'))
    m.eval()
    with torch.no_grad():
        dev_samples = process(m, dev_dataloader, loss_fct, epoch, 'dev', 10)
        print_eval_scores(decoded_df=dev_samples, truth_df=partition['dev'], step=epoch)
        test_samples = process(m, test_dataloader, loss_fct, epoch, 'test', 10)
        print_eval_scores(decoded_df=test_samples, truth_df=partition['test'], step=epoch)
