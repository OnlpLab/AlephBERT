import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from conditional_random_field import ConditionalRandomField, allowed_transitions


def compute_loss(scores, targets, criterion: nn.CrossEntropyLoss):
    loss_target = targets.view(-1)
    loss_input = scores.view(-1, scores.shape[-1])
    return criterion(loss_input, loss_target)


# Mask starting from the position of the first mask_value occurrence
# Done on the last dimension [batch, token, morph_labels]
# E.g. [[[28, 0, 0, 0, 0], [5, 7, 0, 0, 0], [6, 0, 0, 0, 0], [13, 13, 0, 0, 0], [8, 0, 0, 0, 0]]]
# The reason for masking from the first occurrence onward (as opposed to just using torch.ne(labels, mask_value)
# is that the mask_value might be predicted more than once (e.g. if the mask_value is the </s> (3) value):
# [[[12, 3, 3, 5, 0], [3, 1, 2, 5, 5], [5, 5, 3, 0, 0]]]
def first_occurence_mask(labels: torch.Tensor, mask_value) -> torch.BoolTensor:
    masks = torch.eq(labels, mask_value)
    return ~torch.cumsum(masks, dim=-1).bool()


def crf_prepare(scores, labels) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    masks = torch.ne(labels, 0)
    masked_scores = [scores[mask] for scores, mask in zip(scores, masks)]
    masked_labels = [label[mask] for label, mask in zip(labels, masks)]
    logits = nn.utils.rnn.pad_sequence(masked_scores, batch_first=True)
    tags = nn.utils.rnn.pad_sequence(masked_labels, batch_first=True)
    return logits, tags, masks


class BertTokenEmbeddingModel(nn.Module):

    def __init__(self, bert: BertModel, bert_tokenizer: BertTokenizer):
        super(BertTokenEmbeddingModel, self).__init__()
        self.bert = bert
        self.bert_tokenizer = bert_tokenizer

    @property
    def embedding_dim(self):
        return self.bert.config.hidden_size

    def forward(self, token_seq):
        mask = torch.ne(token_seq[:, :, 1], self.bert_tokenizer.pad_token_id)
        bert_output = self.bert(token_seq[:, :, 1], attention_mask=mask)
        bert_emb_tokens = bert_output.last_hidden_state
        emb_tokens = []
        for i in range(len(token_seq)):
            # # groupby token_id
            # mask = torch.ne(input_xtokens[i, :, 1], 0)
            idxs, vals = torch.unique_consecutive(token_seq[i, :, 0][mask[i]], return_counts=True)
            token_emb_xtoken_split = torch.split_with_sizes(bert_emb_tokens[i][mask[i]], tuple(vals))
            # token_xcontext = {k.item(): v for k, v in zip(idxs, [torch.mean(t, dim=0) for t in token_emb_xtokens])}
            emb_tokens.append(torch.stack([torch.mean(t, dim=0) for t in token_emb_xtoken_split], dim=0))
        return emb_tokens


class LabelClassifier(nn.Module):

    def __init__(self, char_emb_size, config: dict):
        super(LabelClassifier, self).__init__()
        self.config = config
        self.num_labels = len(config['id2label'])
        self.ff = nn.Linear(in_features=char_emb_size, out_features=self.num_labels)
        self.crf = None
        if 'crf_trans_type' in config:
            constraint_type = config['crf_trans_type']
            labels = config['id2label']
            transitions = allowed_transitions(constraint_type=constraint_type, labels=labels)
            self.crf = ConditionalRandomField(num_tags=self.num_labels, constraints=transitions)

    def forward(self, dec_chars):
        return self.ff(dec_chars)

    def loss(self, scores, targets, criterion: nn.CrossEntropyLoss):
        if self.crf is None:
            loss_value = compute_loss(scores, targets, criterion)
        else:
            crf_scores, crf_tags, _ = crf_prepare(scores, targets)
            crf_masks = torch.ne(crf_tags, 0).bool()
            crf_log_likelihood = self.crf(inputs=crf_scores, tags=crf_tags, mask=crf_masks)
            crf_log_likelihood /= torch.sum(crf_masks)
            loss_value = -crf_log_likelihood
        return loss_value

    def decode(self, scores) -> torch.Tensor:
        decoded_labels = torch.argmax(scores, dim=-1)
        if self.crf is not None:
            crf_scores, crf_tags, token_masks = crf_prepare(scores, decoded_labels)
            crf_masks = torch.ne(crf_tags, 0).bool()
            crf_decoded_labels = self.crf.viterbi_tags(logits=crf_scores, mask=crf_masks)
            for labels, crf_labels, token_mask in zip(decoded_labels, crf_decoded_labels, token_masks):
                idxs_vals = [torch.unique_consecutive(mask, return_counts=True) for mask in token_mask]
                idxs = torch.cat([idx for idx, _ in idxs_vals])
                vals = torch.cat([val for _, val in idxs_vals])
                decoded_token_tags = torch.split_with_sizes(torch.tensor(crf_labels[0]), tuple(vals[idxs]))
                # TODO: this doesn't do the right thing if a token is decoded as all pads (0)
                # In such a case the first mask is all False and the above split doesn't indicate that this token
                # should be skipped
                for idx, token_tags in enumerate(decoded_token_tags):
                    labels[idx, :len(token_tags)] = token_tags
        return decoded_labels


class SegmentDecoder(nn.Module):

    def __init__(self, char_emb: nn.Embedding, hidden_size, num_layers, dropout, char_dropout, char_out_size,
                 labels_configs: list = None):
        super(SegmentDecoder, self).__init__()
        if labels_configs is None:
            labels_configs = []
        self.char_emb = char_emb
        self.char_encoder = nn.GRU(input_size=char_emb.embedding_dim,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   bidirectional=False,
                                   batch_first=False,
                                   dropout=dropout)
        self.char_decoder = nn.GRU(input_size=char_emb.embedding_dim,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   bidirectional=False,
                                   batch_first=False,
                                   dropout=dropout)
        self.char_dropout = nn.Dropout(char_dropout)
        self.char_out = nn.Linear(in_features=self.char_decoder.hidden_size, out_features=char_out_size)
        self.classifiers = nn.ModuleList([LabelClassifier(char_out_size, config) for config in labels_configs])

    @property
    def enc_num_layers(self):
        return self.char_encoder.num_layers

    @property
    def dec_num_layers(self):
        return self.char_decoder.num_layers

    def forward(self, char_seq, enc_state, special_symbols, max_out_char_seq_len, target_char_seq, max_num_labels):
        char_scores, char_states, label_scores = [], [], []
        enc_output, dec_char_state = self._forward_encode(char_seq, enc_state)
        sos, eos, sep = special_symbols['<s>'], special_symbols['</s>'], special_symbols['<sep>']
        dec_char = sos
        for _ in self.classifiers:
            label_scores.append([])
        while (len(char_scores) < max_out_char_seq_len and
               (max_num_labels is None or all([len(scores) < max_num_labels for scores in label_scores]))):
            dec_output = self._forward_decoder_step(dec_char, dec_char_state, target_char_seq, len(char_scores), sep)
            dec_char, dec_char_output, dec_char_state, dec_labels_output = dec_output
            char_scores.append(dec_char_output)
            char_states.append(dec_char_state.view(1, -1, self.dec_num_layers * dec_char_state.shape[2]))
            if dec_labels_output:
                for output, scores in zip(dec_labels_output, label_scores):
                    scores.append(output)
            if torch.all(torch.eq(dec_char, eos)):
                break
        dec_labels_output = self._labels_decode(char_scores[-1])
        if dec_labels_output:
            for output, scores in zip(dec_labels_output, label_scores):
                scores.append(output)
        char_fill_len = max_out_char_seq_len - len(char_scores)
        char_scores = torch.cat(char_scores, dim=1)
        char_scores_out = F.pad(char_scores, (0, 0, 0, char_fill_len))
        char_states = torch.cat(char_states, dim=1)
        char_states_out = F.pad(char_states, (0, 0, 0, char_fill_len))
        label_scores_out = []
        for scores in label_scores:
            label_fill_len = max_num_labels - len(scores)
            scores = torch.cat(scores, dim=1)
            label_scores_out.append(F.pad(scores, (0, 0, 0, label_fill_len)))
        return char_scores_out, char_states_out, label_scores_out

    def form_loss(self, form_scores, form_targets, criterion: nn.CrossEntropyLoss):
        return compute_loss(form_scores, form_targets, criterion)

    def labels_losses(self, labels_scores, labels_targets, criterion: nn.CrossEntropyLoss):
        return [classifier.loss(scores, targets, criterion)
                for scores, targets, classifier in zip(labels_scores, labels_targets, self.classifiers)]

    def _forward_encode(self, char_seq, enc_state):
        mask = torch.ne(char_seq, 0)
        emb_chars = self.char_emb(char_seq[mask]).unsqueeze(1)
        enc_state = enc_state.view(1, 1, -1)
        enc_state = torch.split(enc_state, enc_state.shape[2] // self.enc_num_layers, dim=2)
        enc_state = torch.cat(enc_state, dim=0)
        enc_output, enc_state = self.char_encoder(emb_chars, enc_state)
        return enc_output, enc_state

    def _forward_decoder_step(self, cur_dec_char, dec_char_state, target_char_seq, num_scores, sep):
        emb_dec_char = self.char_emb(cur_dec_char).unsqueeze(1)
        dec_char_output, dec_char_state = self.char_decoder(emb_dec_char, dec_char_state)
        dec_char_output = self.char_dropout(dec_char_output)
        dec_char_output = self.char_out(dec_char_output)
        if target_char_seq is not None:
            next_dec_char = target_char_seq[num_scores].unsqueeze(0)
        else:
            next_dec_char = self._form_decode(dec_char_output).squeeze(0)
        if torch.eq(next_dec_char, sep):
            dec_labels_output = self._labels_decode(dec_char_output)
        else:
            dec_labels_output = None
        return next_dec_char, dec_char_output, dec_char_state, dec_labels_output

    def _labels_decode(self, dec_char_output) -> list:
        return [classifier(dec_char_output) for classifier in self.classifiers]

    def _form_decode(self, scores):
        return torch.argmax(scores, dim=-1)

    def decode(self, form_scores, label_scores) -> (torch.Tensor, torch.Tensor):
        dec_forms = self._form_decode(form_scores)
        dec_labels = [classifier.decode(scores) for scores, classifier in zip(label_scores, self.classifiers)]
        return dec_forms, dec_labels


class MorphSequenceModel(nn.Module):

    def __init__(self, xtoken_emb: BertTokenEmbeddingModel, segment_decoder: SegmentDecoder):
        super(MorphSequenceModel, self).__init__()
        self.xtoken_emb = xtoken_emb
        self.segment_decoder = segment_decoder

    @property
    def embedding_dim(self):
        return self.xtoken_emb.embedding_dim

    def forward(self, xtoken_seq, char_seq, special_symbols, num_tokens, max_form_len, max_num_labels,
                target_chars=None):
        token_ctx = self.xtoken_emb(xtoken_seq.unsqueeze(dim=0))[0]
        out_char_scores, out_char_states = [], []
        out_label_scores = []
        for _ in self.segment_decoder.classifiers:
            out_label_scores.append([])
        for cur_token_idx in range(num_tokens):
            cur_token_state = token_ctx[cur_token_idx + 1]
            cur_input_chars = char_seq[cur_token_idx]
            cur_target_chars = None
            if target_chars is not None:
                cur_target_chars = target_chars[cur_token_idx]
            seg_output = self.segment_decoder(cur_input_chars, cur_token_state, special_symbols, max_form_len,
                                              cur_target_chars, max_num_labels)
            cur_token_segment_scores, cur_token_segment_states, cur_token_label_scores = seg_output
            out_char_scores.append(cur_token_segment_scores)
            out_char_states.append(cur_token_segment_states)
            for out_scores, seg_scores in zip(out_label_scores, cur_token_label_scores):
                out_scores.append(seg_scores)
        out_char_scores = torch.cat(out_char_scores, dim=0)
        out_char_states = torch.cat(out_char_states, dim=0)
        out_label_scores = [torch.cat(label_scores, dim=0) for label_scores in out_label_scores]
        return out_char_scores, out_char_states, out_label_scores

    def decode(self, morph_seg_scores, label_scores: list):
        return self.segment_decoder.decode(morph_seg_scores, label_scores)

    def form_loss(self, form_scores, form_targets, criterion: nn.CrossEntropyLoss):
        return self.segment_decoder.form_loss(form_scores, form_targets, criterion)

    def labels_losses(self, labels_scores, labels_targets, criterion: nn.CrossEntropyLoss):
        return self.segment_decoder.labels_losses(labels_scores, labels_targets, criterion)


class MorphPipelineModel(MorphSequenceModel):

    def __init__(self, xtoken_emb: BertTokenEmbeddingModel, segment_decoder: SegmentDecoder, hidden_size, num_layers,
                 dropout, seg_dropout, labels_configs: list = None):
        super(MorphPipelineModel, self).__init__(xtoken_emb, segment_decoder)
        if labels_configs is None:
            labels_configs = []
        self.encoder = nn.LSTM(input_size=xtoken_emb.embedding_dim,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               bidirectional=True,
                               batch_first=False,
                               dropout=dropout)
        self.seg_dropout = nn.Dropout(seg_dropout)
        self.classifiers = nn.ModuleList([LabelClassifier(hidden_size*2, config) for config in labels_configs])

    def forward(self, xtoken_seq, char_seq, special_symbols, num_tokens, max_form_len, max_num_labels,
                target_chars=None):
        morph_scores, morph_states, _ = super().forward(xtoken_seq, char_seq, special_symbols, num_tokens,
                                                        max_form_len, max_num_labels, target_chars)
        if target_chars is not None:
            morph_chars = target_chars
        else:
            morph_chars, _ = self.decode(morph_scores, [])
            morph_chars = morph_chars.squeeze(0)
        eos, sep = special_symbols['</s>'], special_symbols['<sep>']
        eos_mask = torch.eq(morph_chars[:num_tokens], eos)
        eos_mask[:, -1] = True
        eos_mask = torch.bitwise_and(torch.eq(torch.cumsum(eos_mask, dim=1), 1), eos_mask)

        sep_mask = torch.eq(morph_chars[:num_tokens], sep)
        sep_mask = torch.bitwise_and(torch.eq(torch.cumsum(eos_mask, dim=1), 0), sep_mask)

        seg_state_mask = torch.bitwise_or(eos_mask, sep_mask)
        seg_states = morph_states[seg_state_mask]
        enc_seg_scores, _ = self.encoder(seg_states.unsqueeze(dim=1))
        enc_seg_scores = self.seg_dropout(enc_seg_scores)
        label_scores = []
        seg_sizes = torch.sum(seg_state_mask, dim=1)
        for classifier in self.classifiers:
            scores = classifier(enc_seg_scores)
            scores = torch.split_with_sizes(scores.squeeze(dim=1), tuple(seg_sizes))
            scores = nn.utils.rnn.pad_sequence(scores, batch_first=True)
            fill_len = max_num_labels - scores.shape[1]
            label_scores.append(F.pad(scores, (0, 0, 0, fill_len)))
        return morph_scores, morph_states, label_scores

    def decode(self, morph_seg_scores, label_scores: list) -> (torch.Tensor, torch.Tensor):
        dec_forms, _ = self.segment_decoder.decode(morph_seg_scores, label_scores)
        dec_labels = [classifier.decode(scores) for scores, classifier in zip(label_scores, self.classifiers)]
        return dec_forms, dec_labels

    def form_loss(self, form_scores, form_targets, criterion: nn.CrossEntropyLoss):
        return self.segment_decoder.form_loss(form_scores, form_targets, criterion)

    def labels_losses(self, labels_scores, labels_targets, criterion: nn.CrossEntropyLoss):
        return [classifier.loss(scores, targets, criterion)
                for scores, targets, classifier in zip(labels_scores, labels_targets, self.classifiers)]
