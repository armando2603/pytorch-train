import torch
import torch.nn as nn
import torch.optim as optim
from seq2seq import Attention, Decoder, Encoder, Seq2Seq
from tokenizers import (BertWordPieceTokenizer, ByteLevelBPETokenizer,
                        CharBPETokenizer, SentencePieceBPETokenizer)
from torch.utils.data import DataLoader
from tqdm import tqdm

tokenizer = CharBPETokenizer()
tokenizer.train(
    ['Translation_dataset/train.en', 'Translation_dataset/train.fr'],
    vocab_size=5000,
    special_tokens=['<pad>', '<sos>', '<eos>']
    )
tokenizer.save(directory='.', name='vocab')
vocab = tokenizer.get_vocab()
tokenizer.enable_padding(pad_id=0, pad_token='<pad>')

output = tokenizer.encode_batch(['mi chiamo giuseppe', '<sos> io sono io <eos>'])
output = tokenizer.post_process(output[1])
print(output.ids, output.tokens)
print(tokenizer.decode(output.ids))
train_input = []
with open('Translation_dataset/train.en') as train_en_file:
    for line in train_en_file.readlines():
        train_input.append('<sos> ' + line + ' <eos>')
train_gold = []
with open('Translation_dataset/train.fr') as train_fr_file:
    for line in train_fr_file.readlines():
        train_gold.append(line + ' <eos>')

# with open('Translation_dataset/train_en_fr_char_ids.txt', mode='w') as train_file:
#     for sample in train:
#         train_file.write(' '.join(map(str, sample)) + '\n')

val = []
with open('Translation_dataset/val.en') as val_en_file:
    for line in val_en_file.readlines():
        val.append('<sos> ' + line + ' <eos>')

with open('Translation_dataset/val.fr') as val_fr_file:
    for line in val_fr_file.readlines():
        val.append('<sos> ' + line + ' <eos>')

# with open('Translation_dataset/val_en_fr_char_ids.txt', mode='w') as val_file:
#     for sample in val:
#         val_file.write(' '.join(map(str, sample)) + '\n')

test = []
with open('Translation_dataset/test.en') as test_en_file:
    for line in test_en_file.readlines():
        test.append('<sos> ' + line + ' <eos>')

with open('Translation_dataset/test.fr') as test_fr_file:
    for line in test_fr_file.readlines():
        test.append('<sos> ' + line + ' <eos>')

# with open('Translation_dataset/test_en_fr_char_ids.txt', mode='w') as test_file:
#     for sample in test:
#         test_file.write(' '.join(map(str, sample)) + '\n')
train_loader = DataLoader(list(zip(train_input, train_gold)), batch_size=32, shuffle=True)

device = torch.device('cpu')
epochs = 5
input_dim = tokenizer.get_vocab_size()
output_dim = input_dim
emb_dim = 100
hid_dim = 50
attn_dim = 40
drop = 0.9
attention = Attention(hid_dim, hid_dim, attn_dim)
encoder = Encoder(input_dim, emb_dim, hid_dim, hid_dim, drop)
decoder = Decoder(output_dim, emb_dim, hid_dim, hid_dim, drop, attention)
model = Seq2Seq(encoder, decoder, device)

def init_weights(m: nn.Module):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

loss_function = nn.CrossEntropyLoss(ignore_index=0)
opt = optim.Adam(model.parameters(), lr=0.001)
model.apply(init_weights)
opt.zero_grad()
for epoch in range(epochs):
    model.train()
    for x, y in tqdm(train_loader):
        w = tokenizer.encode_batch(list(x) + list(y))
        x = w[:32]
        y = w[32:]
        x = [tokenizer.post_process(item).ids for item in x]
        print(tokenizer.decode(x[0]))
        x = torch.LongTensor(x).permute([1, 0])

        y = [tokenizer.post_process(item).ids for item in y]
        print(tokenizer.decode(y[0]))
        y = torch.LongTensor(y).permute([1, 0])

        outputs = model(x, y, teacher_forcing_ratio=1)
        # print(tokenizer.decode(list(x[0])))
        # print(tokenizer.decode(list(y[0])))
        outputs = outputs.permute([1, 2, 0])

        # print(tokenizer.decode(list(outputs[0, :].int()))
        
        loss = loss_function(outputs[:, :, : -1], y.permute([1, 0])[:, 1:])
        my_trasl = torch.max(outputs, dim=1)[1]
        print(my_trasl[0])
        print(tokenizer.decode(list(my_trasl[0])))
        print(loss.item())


        loss.backward()
        opt.step()
