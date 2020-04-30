import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim,
        emb_dim,
        hid_dim,
        n_layers,
        kernel_size,
        dropout,
        device,
        max_len=100,
    ):
        super().__init__()

        assert kernel_size % 2 == 1, 'kernel must be odd'
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.tok_embedding = nn.Embedding(input_dim, emb_dim)
        self.pos_embedding = nn.Embedding(max_len, emb_dim)
        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(hid_dim, emb_dim)
        self.convs = nn.ModuleList([nn.Conv1d(
            in_channels=hid_dim,
            out_channels=hid_dim * 2,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            ) for _ in range(n_layers)
        ])
        self.droput = nn.Dropout(dropout)

    def forward(self, src):
        # src : [batch_size, max_len]

        batch_size = src.shape[0]
        max_len = src.shape[1]

        # create position tensor

        pos = torch.arange(0, max_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # pos : [batch_size, max_len]

        tok_embedded = self.tod_embedding(src)
        pos_embedded = self.pos_embedding(pos)

        # pos/tok embedding : [batch_size, max_len, embedding]

        embedded = nn.dropout(tok_embedded + pos_embedded)

        # same as pos/tiok emb

        conv_input = nn.emb2hid(embedded)

        # conv_input : [batch_size, max_len, hid_dim]

        conv_input = conv_input.permute([1, 2, 0])

        # conv_input : [batch_size, hid_dim, max_len]

        for i, conv in enumerate(self.convs):
            conved = conv(self.dropout(conv_input))

            # conved : [batch_size, hid_dim * 2, max_len]

            conved = F.glu(conved, dim=1)

            # conved : [batch_size, hid_dim, max_len]

            conved = (conved + embedded) * self.scale

            # same as before

            conv_input = conved

        conved = conved.permute([0, 2, 1])

        # conved : [batch_size, max_len, hid_dim]

        conved = self.hid2emb(conved)

        # conved : [batch_size, max_len, emb_dim]

        combined = (conved + embedded) * self.scale

        # same as conved

        return conved, combined


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim,
        hid_dim,
        n_layers,
        kernel_size,
        dropout,
        trg_pad_idx,
        device,
        max_len=100,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)

        self.tok_embedding = nn.Embedding(vocab_size, emb_dim)
        self.pos_embedding = nn.Embedding(max_len, emb_dim)

        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(hid_dim, emb_dim)

        self.attn_hid2emb = nn.Linear(hid_dim, emb_dim)
        self.attn_emb2hid = nn.Linear(emb_dim, hid_dim)

        self.output = nn.Linear(emb_dim, vocab_size)

        self.convs = nn.ModuleList([nn.Conv1d(
            in_channels=hid_dim,
            out_channels=2 * hid_dim,
            kernel_size=kernel_size
            )] for _ in range(n_layers))

        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, encoder_conved, encoder_combined):

        # trg : [batch_size, max_len]
        # encoder_conved = encoder_combined : [batch_size, max_len, emb_dim]

        batch_size = trg.shape[0]
        max_len = encoder_combined.shape[1]

        pos = torch.arange(0, max_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # pos : [batch_size, max_len]

        tok_embedded = self.tok_embedding(trg)
        pos_embedded = self.pos_embedding(pos)

        # tok/pos embedded : [batch_size, max_len, emb_size]

        embedded = (tok_embedded + pos_embedded) * self.scale

        # same as before

        conv_input = self.emb2hid(embedded)

        # conv_input :[batch_size, max_len, hid_dim]

        # permute for convolutional layer

        conv_input = conv_input.permute(0, 2, 1)

        # conv_input : [batch_size, hid_dim, max_len]

        hid_dim = conv_input.shape[1]

        for i, conv in enumerate(self.convs):

            conved = self.dropout(conv_input)

            # need to pad so decoder can't cheat
            padding = torch.zeros(
                batch_size,
                hid_dim,
                self.kernel_size - 1).fill_(self.trg_pad_idx).to(self.device)

            padded_conv_input = torch.cat((padding, conv_input), dim=2)

            # padded_conv_input : [batch_size, hid_dim, max_len + kernel - 1]

            conved = conv(padded_conv_input)

            # conved : [batch_size, 2 * hid_dim, max_len]

            conved = F.glu(conved, dim=1)

            # conved : [batch_size, hid_dim, max_len]

            attention, conved = self.calculate_attention(
                embedded,
                conved,
                encoder_conved,
                encoder_combined,
                )
            # attention : [batch_size, max_len_trg, max_len_src]

            conved = (conved + conv_input) * self.scale

            # same as conved before

            conv_input = conved

        conved = self.hid2emb(conved.permute(0, 2, 1))

        # conved : [batch_size, max_len, emb_dim]

        output = self.output(self.dropout(conved))

        # output : [batch_size, max_len, vocab_size]

        return output, attention

    def calculate_attention(
        self,
        embedded,
        conved,
        encoder_conved,
        encoder_combined,
    ):

        # embedded :[batch_size, max_len_trg, emb_size]
        # conved : [batch_size, hid_dim, max_len_trg]
        # encoder_conved/combined : [batch_size, max_len_src, emb_dim]

        conved_emb = self.attn_hid2emb(conved.permute([0, 2, 1]))

        # conved_emb : [batch_size, max_len_trg, emb_dim]

        combined = (conved_emb + embedded) * self.scale

        energy = torch.matmul(combined, encoder_conved.permute([0, 2, 1]))

        # energy :[batch_size, max_len_trg, max_len_src]

        attention = F.softmax(energy, dim=2)

        # same as energy

        attended_encoding = torch.matmul(attention, encoder_combined)

        # attended_encoding : [batch_size, max_len_trg, emb_dim]

        attended_encoding = self.attn_emb2hid(attended_encoding)

        # attended_encoding : [batch_size, max_len_trg, hid_dim]

        attended_combined = (conved, attended_encoding.permute(0, 2, 1)) * self.scale

        return attention, attended_combined


class Seq2Seq(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):
        # src : [batch_size, max_len_src]
        # trg : [batch_size, max_len_trg - 1] (<eos> token sliced off the end)

        encoder_conved, encoder_combined = self.encoder(src)

        # encoder_conved/combined : [batch_size, max_len_src, emb_dim]

        output, attention = self.decoder(src, trg)

        # output : [batch_size, max_len_trg - 1, vocab_size]
        # attention : [batch_size, max_len_trg - 1, max_len_src]

        return output, attention
