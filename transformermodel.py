import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        hid_dim,
        n_layers,
        n_heads,
        pf_dim,
        dropout,
        device,
        max_length=100
    ):
        super(Encoder, self).__init__()

        self.device = device

        self.tok_embedding = nn.Embedding(vocab_size, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([EncoderLayer(
            hid_dim,
            n_heads,
            pf_dim,
            dropout,
            device) for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, src_mask):
        # src : [batch_size, max_len]
        # src_mask : [batch_size, 1, 1, max_len]

        batch_size = src.shape[0]
        max_len = src.shape[1]

        pos = torch.arange(0, max_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # pos : [batch_size, max_len]

        src = self.dropout(self.tok_embedding(src) * self.scale + self.pos_embedding(pos))

        # src : [batch_size, max_len, hid_dim]

        for layer in self.layers:
            src = layer(src, src_mask)

        # src = [batch_size, max_len, hid_dim]

        return src


class EncoderLayer(nn.Module):
    def __init__(
        self,
        hid_dim,
        n_heads,
        pf_dim,
        dropout,
        device
    ):
        super(EncoderLayer, self).__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(
            hid_dim,
            n_heads,
            dropout,
            device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(
            hid_dim,
            pf_dim,
            dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # src : [batch_size, max_len, hid_dim]
        # src_mask : [batch_size, 1, 1, max_len]

        _src, _ = self.self_attention(src, src, src, src_mask)

        # dropout, residual_connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))

        # src : [batch_size, max_len, hid_dim]

        # positionwise feedforward
        _src = self.positionwise_feedforward(src)

        # dropout residual layernorm
        src = self.ff_layer_norm(src + self.dropout(_src))

        # src = [batch_size, max_len, hid_dim]

        return src


class MultiHeadAttentionLayer(nn.Module):
    def __init__(
        self,
        hid_dim,
        n_heads,
        dropout,
        device
    ):
        super(MultiHeadAttentionLayer, self).__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.linear_q = nn.Linear(hid_dim, hid_dim)
        self.linear_k = nn.Linear(hid_dim, hid_dim)
        self.linear_v = nn.Linear(hid_dim, hid_dim)

        self.linear_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):

        batch_size = query.shape[0]

        # query/key/value : [batch_size, max_len, hid_dim]

        Q = self.linear_q(query)
        K = self.linear_k(key)
        V = self.linear_v(value)

        # Q/K/V : [batch_size, max_len, hid_dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute([0, 2, 1, 3])
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute([0, 2, 1, 3])
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute([0, 2, 1, 3])

        # Q/K/V : [batch_size, n_heads, max_len, head_dim]

        # energy is the attention non normalized

        energy = torch.matmul(Q, K.permute([0, 1, 3, 2])) / self.scale

        # energy : [batch_size, n_heads, max_len_query, max_len_keys]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)

        # attention : [batch_size, n_heads, max_len_query, max_len_keys]

        x = torch.matmul(self.dropout(attention), V)

        # x : [batch_size, n_heads, max_len_query, head_dim]

        x = x.permute([0, 2, 1, 3]).contiguous()

        # x : [batch_size, max_len_query, n_heads, head_dim]

        x = x.view(batch_size, -1, self.hid_dim)

        # x : [batch_size, max_len, hid_dim]

        x = self.linear_o(x)

        # x : [batch_size, max_len, hid_dim]

        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(
        self,
        hid_dim,
        pf_dim,
        dropout
    ):
        super(PositionwiseFeedforwardLayer, self).__init__()

        self.linear_1 = nn.Linear(hid_dim, pf_dim)
        self.linear_2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x :[batch_sie, max_len, hid_dim]

        # most of the model uses F.Gelu instead of relu

        # x = self.dropout(torch.relu(self.linear_1(x)))
        x = self.dropout(F.gelu(self.linear_1(x)))

        # x : [batch_size, max_len, pf_dim]

        x = self.linear_2(x)

        # x : [batch_size, max_len, hid_dim]

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        hid_dim,
        n_layers,
        n_heads,
        pf_dim,
        dropout,
        device,
        max_len=100,
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(vocab_size, hid_dim)
        self.pos_embedding = nn.Embedding(max_len, hid_dim)
        self.layers = nn.ModuleList(
            [DecoderLayer(
                hid_dim,
                n_heads,
                pf_dim,
                dropout,
                device,
                ) for _ in range(n_layers)]
            )
        self.linear_output = nn.Linear(hid_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg/mask :[batch_size, 1 , max_len_trg, max_len_trg]
        # src_mask :[batch_size, 1, 1, max_len_src]
        # enc_src : [batch_size, max_len_src, hid_dim]

        batch_size = trg.shape[0]
        max_len_trg = trg.shape[1]

        pos = torch.arange(0, max_len_trg).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # pos : [batch_size, max_len_trg]

        pos_embedded = self.pos_embedding(pos)
        tok_embedded = self.tok_embedding(trg)

        trg = self.dropout((tok_embedded * self.scale) + pos_embedded)

        # trg : [batch_size, max_len_trg, hid_dim]

        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

            # trg : [batch_size, max_len. hid_dim]
            # attention : [batch_size, n_heads, max_len_trg, max_len_src]

        output = self.linear_output(trg)

        # output :[batch_size, max_len, vocab_size]

        return output, attention


class DecoderLayer(nn.Module):
    def __init__(
        self,
        hid_dim,
        n_heads,
        pf_dim,
        dropout,
        device
    ):
        super(DecoderLayer, self).__init__()

        # Unlike batch normalization that applies scalar scale and bias
        # for each entire channel/plane, layer normalization
        # applies per-element scale and bias

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg : [batch_size, max_len_trg, hid_dim]
        # enc_src : [batch_size, max_len_src, hid_dim]
        # trg_mask : [batch_size, 1, max_len_trg, max_len_trg]
        # src_mask : [batch_size, 1, 1, max_len_src]

        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)

        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))

        # trg : [batch_size, max_len_trg, hid_dim]

        # encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))

        # trg : same as before

        _trg = self.positionwise_feedforward(trg)

        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        # trg : [batch_size, max_len_trg, hid_dim]
        # attention : [batch_size, n_heads, max_len_trg, max_len_src]

        return trg, attention


class Seq2Seq(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        src_pad_ids,
        trg_pad_ids,
        device
    ):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_ids = src_pad_ids
        self.trg_pad_ids = trg_pad_ids
        self.device = device

    def make_src_mask(self, src):
        # src : [batch_size, max_len_src]

        src_mask = (src != self.src_pad_ids).unsqueeze(1).unsqueeze(2)

        # what I want is a mask for the encoder that have dimensions
        # batch, n_heads, max_len_query, max_len_keys

        # src_mask : [batch_size, 1, 1, max_len_src]

        return src_mask

    def make_trg_mask(self, trg):

        # trg : [batch_size, max_len_trg]

        trg_pad_mask = (trg != self.trg_pad_ids).unsqueeze(1).unsqueeze(2)

        # what I want is a mask for the decoder that have dimensions
        # batch, n_heads, max_len_query, max_len_keys

        # trg_pad_mask : [batch_size, 1, 1, max_len_trg]

        max_len_trg = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((max_len_trg, max_len_trg), device=self.device)).bool()

        # trg_sub_mask :[max_len_trg, max_len_trg]

        trg_mask = trg_pad_mask & trg_sub_mask

        # trg_mask : [batch_size, 1, max_len_trg, max_len_trg]

        return trg_mask

    def forward(self, src, trg):

        # src : [batch_size, max_len_src]
        # trg : [batch_size, max_len_trg]

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        # src_mask : [batch_size, 1, 1, max_len_src]
        # trg_mask : [batch_size, 1, max_len_trg, max_len_trg]

        enc_src = self.encoder(src, src_mask)

        # enc_src : [batch_size, max_len_src, hid_dim]

        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        # output : [batch_size, max_len_trg, output_dim]

        # attention : [batch_size, n_heads, max_len_trg, max_len_src]

        return output, attention
