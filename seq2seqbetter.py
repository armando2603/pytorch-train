import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)
        self.linear = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

    def forward(self, input, input_lens):
        # input : [max_len, batch_size]
        # input_len : [batch_size]
        embedded = self.dropout(self.embedding(input))
        # embedded : [max_len, batch_size, emb_dim]
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lens, enforce_sorted=True)
        packed_outputs, state = self.rnn(packed_embedded)
        # state : [n_layers * directions, batch_size, hid_dim]
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        # outputs : [max_len, batch_size, directions * hid_dim]
        # state is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        state = torch.tanh(self.linear(torch.cat((state[-2, :, :], state[-1, :, :]), dim=1)))
        state = state.unsqueeze(0)
        # state  : [1, batch_size, hiden_dim]
        return outputs, state


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()

        self.vocab_size = vocab_size

        # attention
        self.w = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)
        # decoder
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, vocab_size)

    def forward(self, input, prev_state, enc_outputs, mask):
        # input :[batch_size]
        # mask: [batch_size, max_len]
        # prev_state :[1, batch_size, hid_dim]
        embedded = self.dropout(self.embedding(input)).unsqueeze(0)
        # embedded : [1, batch_size, emb_size]
        alignment = self.attention(prev_state, enc_outputs, mask)
        # alignment : [batch_size, max_len]
        alignment = alignment.unsqueeze(1)
        # alignment :[batch_size, 1, max_len]
        enc_outputs = enc_outputs.permute([1, 0, 2])
        # enc_outputs :[batch_size, max_len, hiden_dim * 2]
        context = torch.bmm(alignment, enc_outputs)
        # context :[batch_size, 1, hiden_dim * 2]
        context = context.permute([1, 0, 2])
        # context :[1, batch_size, hiden_dim * 2]
        rnn_input = torch.cat((embedded, context), dim=2)
        # rnn_input :[1, batch_size, hid_dim * 2 + emb_dim]
        output, state = self.rnn(rnn_input, prev_state)
        # output = [seq len, batch size, dec hid dim * n directions]
        # state = [n layers * n directions, batch size, dec hid dim]
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        context = context.squeeze(0)

        prediction = self.linear(torch.cat((embedded, output, context), dim=1))
        # prediction : [batch_size, vocab_size]
        return prediction, state, alignment.squeeze(1)

    def attention(self, state, encoder_outputs, mask):
        # hidden = [1 ,batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]
        # mask: [batch_size, max_len]

        max_len = encoder_outputs.shape[0]
        state = state.permute([1, 0, 2]).repeat(1, max_len, 1)
        # state : [batch_size, max_len, hid_dim]
        encoder_outputs = encoder_outputs.permute([1, 0, 2])
        # enc_outputs : [batch_size, max_len, hid_dim * 2]
        energy = torch.tanh(self.w(torch.cat((state, encoder_outputs), dim=2)))
        # energy :[batch_size, max_len, hid_dim]
        attention = self.v(energy).squeeze(2)
        # attention :[batch_size, max_len]
        attention = attention.masked_fill(mask == 0, -1e10)
        return F.softmax(attention, dim=1)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_ids, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.pad_ids = src_pad_ids

    def create_mask(self, src):
        mask = (src != self.pad_ids).permute(1, 0)
        return mask

    def forward(self, src, src_len, trg, teacher_forcing_ratio=0.5):
        # src = [src len, batch size]
        # src_len = [batch size]
        # trg = [trg len, batch size]

        batch_size = src.shape[1]
        max_len = trg.shape[0]
        vocab_size = self.decoder.vocab_size

        outputs = torch.zeros(max_len, batch_size, vocab_size).to(self.device)

        encoder_outputs, state = self.encoder(src, src_len)

        input = trg[0, :]

        mask = self.create_mask(src)

        for t in range(1, max_len):
            output, state, _ = self.decoder(input, state, encoder_outputs, mask)
            # output : [batch_size, vocab_size]
            # state = [1, batch size, dec hid dim]
            outputs[t] = output
            top1 = output.argmax(1)
            teacher_force = random.random() < teacher_forcing_ratio
            input = trg[t] if teacher_force else top1
        return outputs
