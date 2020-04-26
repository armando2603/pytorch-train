
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru_rnn = nn.GRU(embed_size, hidden_size, dropout=0, bidirectional=True)

    def forward(self, input):
        # input : [max_len, batch_size]
        embedded = self.embedding(input)
        # embedded : [max_len, batch_size, embed_size]
        outputs, state = self.gru_rnn(embedded)
        state = state.view(1, state.shape[1], -1)
        # outputs = [sen_len, batch_size, hid_dim * n_directions]
        # state = [n_layers * n_direction, batch_size, hid_dim]
        return outputs, state


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, dropout):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru_rnn = nn.GRU(embed_size, hidden_size*2, dropout=dropout, bidirectional=False)
        self.linear = nn.Linear(hidden_size * 4, vocab_size)

    def forward(self, target, prev_state, enc_outputs):
        target = target.unsqueeze(0)
        # target : [1, batch_size]
        embedded = self.embedding(target)
        # embedded :[1, batch_size, embed_size]
        # max_len = enc_outputs.shape[0]
        # batch_size = prev_state.shape[1]
        # hid_dim = prev_state.shape[2]

        output, state = self.gru_rnn(embedded, prev_state)
        # outputs = [sen_len, batch_size, hid_dim * n_directions]
        # state = [n_layers * n_direction, batch_size, hid_dim]

        alignment = self.attention(enc_outputs, state)

        attn_weights = F.softmax(alignment, dim=0)
        attn_weights = attn_weights.permute([0, 2, 1])
        enc_outputs = enc_outputs.permute([1, 0, 2])
        context = torch.bmm(attn_weights, enc_outputs)
        # context : [batch, 1, hidden]
        
        output = output.permute([1, 0, 2])
        concat = torch.cat((output, context), dim=2)
        concat = concat.squeeze(1)
        
        dec_output = self.linear(concat)

        return dec_output, state

    def attention(self, enc_outputs, decoder_state):
        '''
        enc_outputs : [len_max, batch, hidden]
        decoder_state : [1, batch, hidden]
        '''
        enc_outputs = enc_outputs.permute([1, 0, 2])
        decoder_state = decoder_state.permute([1, 2, 0])
        allignment = torch.bmm(enc_outputs, decoder_state)
        # allignment : [batch, len_max, 1]
        return allignment


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input, target, teacher_forcing_ratio):

        enc_outputs, prev_state = self.encoder(input)

        batch_size = input.shape[1]
        max_len = target.shape[0]
        vocab_size = self.decoder.vocab_size

        outputs = torch.zeros([max_len, batch_size, vocab_size])

        input_dec = target[0, :]

        for t in range(1, max_len):
            output, prev_state = self.decoder(input_dec, prev_state, enc_outputs)
            outputs[t] = output
            top1 = output.argmax(1)
            # decide if we are going to use teacher forcing or not.
            teacher_force = random.random() < teacher_forcing_ratio
            input_dec = target[t] if teacher_force else top1

        return outputs
