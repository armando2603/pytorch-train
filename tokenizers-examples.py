import torch
import torch.nn as nn
import torch.optim as optim
from seq2seq import Decoder, Encoder, Seq2Seq
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
# print(output.ids, output.tokens)
# print(tokenizer.decode(output.ids))
train_input = []
with open('Translation_dataset/train.en') as train_en_file:
    for line in train_en_file.readlines():
        train_input.append('<sos> ' + line[:-1] + ' <eos>')
train_gold = []
with open('Translation_dataset/train.fr') as train_fr_file:
    for line in train_fr_file.readlines():
        train_gold.append('<sos> ' + line[:-1] + ' <eos>')

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
train_loader = DataLoader(list(zip(train_input, train_gold)), batch_size=64, shuffle=True, drop_last=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 5
input_dim = tokenizer.get_vocab_size()
output_dim = input_dim
emb_dim = 200
hid_dim = 100
attn_dim = 40
drop = 0
clip = 1
encoder = Encoder(input_dim, emb_dim, hid_dim, drop)
decoder = Decoder(output_dim, emb_dim, hid_dim, drop)
model = Seq2Seq(encoder, decoder, device).to(device=device)


def init_weights(m: nn.Module):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


model.apply(init_weights)

optimizer = optim.Adagrad(model.parameters(), lr=0.1)

loss_function = nn.CrossEntropyLoss(ignore_index=0)


def train(model, iterator, optimizer, loss_function, clip):

    model.train()

    epoch_loss = 0

    for input_batch, target_batch in iterator:
        print(target_batch[0])
        batch = tokenizer.encode_batch(list(input_batch) + list(target_batch))
        input_batch = batch[:64]
        target_batch = batch[64:]

        input_batch = [tokenizer.post_process(item).ids for item in input_batch]
        input_batch = torch.tensor(input_batch).to(device).permute([1, 0]).clone()

        target_batch = [tokenizer.post_process(item).ids for item in target_batch]
        target_batch = torch.tensor(target_batch).to(device).permute([1, 0])
        target_batch = target_batch.clone()

        optimizer.zero_grad()

        outputs = model(input_batch, target_batch, 1)

        

        logits = outputs[1:].view(-1, output_dim)
        target_batch = target_batch[1:].view(-1)

        loss = loss_function(logits, target_batch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        model.eval()

        my_string = outputs.argmax(2)

        print(tokenizer.decode(list(my_string[1:, 0])))

        print(loss.item())


train(model, train_loader, optimizer, loss_function, clip)
