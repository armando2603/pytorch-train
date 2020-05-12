import math
import time

import convmodel
# import original_transformer as transformermodel
import seq2seqbetter
import torch
import torch.nn as nn
import torch.optim as optim
import transformermodel
from tokenizers import (BertWordPieceTokenizer, ByteLevelBPETokenizer,
                        CharBPETokenizer, SentencePieceBPETokenizer)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

src_lang = 'fr'
trg_lang = 'en'

tokenizer_src = CharBPETokenizer(lowercase=True)
tokenizer_src.train(
    ['Translation_dataset/train.' + src_lang, 'Translation_dataset/train.' + trg_lang],
    vocab_size=5000,
    special_tokens=['<pad>', '<sos>', '<eos>'],
    )

src_vocab = tokenizer_src.get_vocab()
tokenizer_src.enable_padding(pad_id=0, pad_token='<pad>')

tokenizer_trg = tokenizer_src
trg_vocab = src_vocab

# tokenizer_trg = CharBPETokenizer(lowercase=True)
# tokenizer_trg.train(
#     ['Translation_dataset/train.en'],
#     vocab_size=5001,
#     special_tokens=['<pad>', '<sos>', '<eos>']
#     )

# trg_vocab = tokenizer_trg.get_vocab()
# tokenizer_trg.enable_padding(pad_id=0, pad_token='<pad>')


# class SSTDataset(Dataset):

#     def __init__(self, filename, maxlen=95):

#         # Store the contents of the file in a pandas dataframe
#         self.input = []
#         with open('Translation_dataset/train.de') as src_file:
#             for line in src_file.readlines():
#                 self.input.append('<sos> ' + line[:-1] + ' <eos>')
#         self.gold_reference = []
#         with open('Translation_dataset/train.en') as trg_file:
#             for line in trg_file.readlines():
#                 self.gold_reference.append('<sos> ' + line[:-1] + ' <eos>')

#         self.maxlen = maxlen

#         self.tokenizer = CharBPETokenizer()
#         tokenizer_src.train(
#             ['Translation_dataset/train.de'],
#             vocab_size=5000,
#             special_tokens=['<pad>', '<sos>', '<eos>']
#             )

#         tokenizer_src.enable_padding(pad_id=0, pad_token='<pad>')

#     def __len__(self):
#         return len(self.input)

#     def __getitem__(self, index):

#         # Selecting the sentence and label at the specified index in the data frame
#         sentence = self.input[index]
#         label = self.gold_reference[index]

#         # tokenize the sentence
#         encoded_sent = self.tokenizer.encode(sentence)

#         # padding the sentence to the max length
#         if len(encoded_sent) > self.maxlen:
#             encoded_sent = encoded_sent[:self.maxlen - 2]
#         padded_sent = encoded_sent + [0 for _ in range(self.maxlen - len(encoded_sent))]
#         # Converting the list to a pytorch tensor
#         tokens_ids_tensor = torch.tensor(padded_sent)  # Converting the list to a pytorch tensor
#         # Obtaining the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
#         attn_mask = (tokens_ids_tensor != 0).long()

#         return tokens_ids_tensor, attn_mask, label

# output = tokenizer.encode_batch(['mi chiamo giuseppe', '<sos> io sono io <eos>'])
# output = tokenizer.post_process(output[1])
# print(output.ids, output.tokens)
# print(tokenizer.decode(output.ids))
train_input = []
with open('Translation_dataset/train.' + src_lang) as train_src_file:
    for line in train_src_file.readlines():
        train_input.append('<sos> ' + line[:-1] + ' <eos>')
train_gold = []
with open('Translation_dataset/train.' + trg_lang) as train_trg_file:
    for line in train_trg_file.readlines():
        train_gold.append('<sos> ' + line[:-1] + ' <eos>')

# with open('Translation_dataset/train_en_fr_char_ids.txt', mode='w') as train_file:
#     for sample in train:
#         train_file.write(' '.join(map(str, sample)) + '\n')

val_input = []
val_gold = []
with open('Translation_dataset/val.' + src_lang) as val_src_file:
    for line in val_src_file.readlines():
        val_input.append('<sos> ' + line[:-1] + ' <eos>')

with open('Translation_dataset/val.' + trg_lang) as val_trg_file:
    for line in val_trg_file.readlines():
        val_gold.append('<sos> ' + line[:-1] + ' <eos>')

# with open('Translation_dataset/val_en_fr_char_ids.txt', mode='w') as val_file:
#     for sample in val:
#         val_file.write(' '.join(map(str, sample)) + '\n')

test_input = []
test_gold = []
with open('Translation_dataset/test.' + src_lang) as test_src_file:
    for line in test_src_file.readlines():
        test_input.append('<sos> ' + line[:-1] + ' <eos>')

with open('Translation_dataset/test.' + trg_lang) as test_trg_file:
    for line in test_trg_file.readlines():
        test_gold.append('<sos> ' + line[:-1] + ' <eos>')

# with open('Translation_dataset/test_en_fr_char_ids.txt', mode='w') as test_file:
#     for sample in test:
#         test_file.write(' '.join(map(str, sample)) + '\n')


batch_size = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 10
clip = 1
pad_ids = 0
input_dim = tokenizer_src.get_vocab_size()
output_dim = tokenizer_trg.get_vocab_size()
model_type = 'TRANSF'
if model_type == 'RNN':
    emb_dim = 300
    hid_dim = 200
    attn_dim = 40
    drop = 0.5

if model_type == 'CONV':
    emb_dim = 300
    hid_dim = 512  # each conv. layer has 2 * hid_dim filters
    layers = 10  # number of conv. blocks in encoder
    kernel_size = 3
    conv_dropout = 0.25

if model_type == 'TRANSF':
    hid_dim = 256
    enc_layers = 3
    dec_layers = 3
    enc_heads = 8
    dec_heads = 8
    enc_pf_dim = 512
    dec_pf_dim = 512
    enc_dropout = 0.1
    dec_dropout = 0.1

if model_type == 'TRANSF':
    encoder = transformermodel.Encoder(
        input_dim,
        hid_dim,
        enc_layers,
        enc_heads,
        enc_pf_dim,
        enc_dropout,
        device,
    )
    decoder = transformermodel.Decoder(
        output_dim,
        hid_dim,
        dec_layers,
        dec_heads,
        dec_pf_dim,
        dec_dropout,
        device,
    )
    model = transformermodel.Seq2Seq(
        encoder,
        decoder,
        pad_ids,
        pad_ids,
        device,
    ).to(device)

if model_type == 'RNN':
    encoder = seq2seqbetter.Encoder(input_dim, emb_dim, hid_dim, hid_dim, drop)
    decoder = seq2seqbetter.Decoder(output_dim, emb_dim, hid_dim, hid_dim, drop)
    model = seq2seqbetter.Seq2Seq(encoder, decoder, pad_ids, device).to(device=device)
if model_type == 'CONV':
    encoder = convmodel.Encoder(input_dim, emb_dim, hid_dim, layers, kernel_size, conv_dropout, device)
    decoder = convmodel.Decoder(
        output_dim,
        emb_dim,
        hid_dim,
        layers,
        kernel_size,
        conv_dropout,
        pad_ids,
        device)
    model = convmodel.Seq2Seq(encoder, decoder).to(device=device)

train_loader = DataLoader(list(zip(train_input, train_gold)), batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(list(zip(val_input, val_gold)), batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(list(zip(test_input, test_gold)), batch_size=batch_size, shuffle=True, drop_last=True)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')


if model_type == 'TRANSF':
    def init_weights(m):
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.xavier_uniform_(m.weight.data)
else:
    def init_weights(m: nn.Module):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param.data)
            else:
                nn.init.constant_(param.data, 0)


model.apply(init_weights)

optimizer = optim.Adam(model.parameters(), lr=0.0005)


loss_function = nn.CrossEntropyLoss(ignore_index=pad_ids)


def train(model, iterator, optimizer, loss_function, clip):

    epoch_loss = 0

    for input_batch, target_batch in iterator:
        model.train()
        if model_type == 'RNN':
            lens_input = list(map(len, [tokenizer_src.encode(x).ids for x in input_batch]))
            batch_sorted = list(zip(lens_input, input_batch, target_batch))
            batch_sorted.sort(reverse=True)
            lens_input, input_batch, target_batch = zip(*(batch_sorted))
            lens_input = torch.tensor(lens_input).to(device=device)

        print(input_batch[0])
        print(target_batch[0])

        input_batch = tokenizer_src.encode_batch(list(input_batch))
        target_batch = tokenizer_trg.encode_batch(list(target_batch))

        input_batch = [tokenizer_src.post_process(item).ids for item in input_batch]
        input_batch = torch.tensor(input_batch).to(device).permute([1, 0])

        target_batch = [tokenizer_trg.post_process(item).ids for item in target_batch]
        target_batch = torch.tensor(target_batch).to(device).permute([1, 0]).contiguous()

        optimizer.zero_grad()

        if model_type == 'RNN':

            outputs = model(input_batch, lens_input, target_batch, 1)

            output_dim = outputs.shape[-1]

            logits = outputs[1:].view(-1, output_dim)
            target_batch = target_batch[1:].view(-1)

        if model_type == 'CONV' or model_type == 'TRANSF':
            input_batch = input_batch.permute([1, 0])
            target_batch = target_batch.permute([1, 0])
            output, _ = model(input_batch, target_batch[:, :-1])

            # output = [batch size, trg len - 1, output dim]
            # trg = [batch size, trg len]

            outputs = output.permute([1, 0, 2])

            output_dim = output.shape[-1]

            logits = output.contiguous().view(-1, output_dim)
            target_batch = target_batch[:, 1:].contiguous().view(-1)

            # output = [batch size * trg len - 1, output dim]
            # trg = [batch size * trg len - 1]

        loss = loss_function(logits, target_batch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        model.eval()

        my_string = outputs.argmax(2)

        print(tokenizer_trg.decode(list(my_string[1:, 0])))

        print(loss.item())

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for input_batch, target_batch in iterator:
            if model_type == 'RNN':
                lens_input = list(map(len, [tokenizer_src.encode(x).ids for x in input_batch]))
                batch_sorted = list(zip(lens_input, input_batch, target_batch))
                batch_sorted.sort(reverse=True)
                lens_input, input_batch, target_batch = zip(*(batch_sorted))
                lens_input = torch.tensor(lens_input).to(device=device)

            # print(input_batch[0])
            # print(target_batch[0])

            input_batch = tokenizer_src.encode_batch(list(input_batch))
            target_batch = tokenizer_trg.encode_batch(list(target_batch))

            input_batch = [tokenizer_src.post_process(item).ids for item in input_batch]
            input_batch = torch.tensor(input_batch).to(device).permute([1, 0]).contiguous()

            target_batch = [tokenizer_trg.post_process(item).ids for item in target_batch]
            target_batch = torch.tensor(target_batch).to(device).permute([1, 0]).contiguous()
            if model_type == 'RNN':
                output = model(input_batch, lens_input, target_batch, 0)  # turn off teacher forcing
                # trg = [trg len, batch size]
                # output = [trg len, batch size, output dim]

                output_dim = output.shape[-1]
                output = output[1:].view(-1, output_dim)
                target_batch = target_batch[1:].view(-1)

                # trg = [(trg len - 1) * batch size]
                # output = [(trg len - 1) * batch size, output dim]
            if model_type == 'CONV' or model_type == 'TRANSF':
                input_batch = input_batch.permute([1, 0])
                target_batch = target_batch.permute([1, 0])
                output, _ = model(input_batch, target_batch[:, :-1])

                # output = [batch size, trg len - 1, output dim]
                # trg = [batch size, trg len]

                output_dim = output.shape[-1]

                output = output.contiguous().view(-1, output_dim)
                target_batch = target_batch[:, 1:].contiguous().view(-1)

                # output = [batch size * trg len - 1, output dim]
                # trg = [batch size * trg len - 1]

            loss = criterion(output, target_batch)

            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


best_valid_loss = float('inf')

for epoch in range(epochs):

    start_time = time.time()

    train_loss = train(model, train_loader, optimizer, loss_function, clip)
    valid_loss = evaluate(model, val_loader, loss_function)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut4-model.pt')

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

model.load_state_dict(torch.load('tut4-model.pt'))

test_loss = evaluate(model, test_loader, loss_function)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
