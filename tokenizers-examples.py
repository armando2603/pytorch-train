import math
import time

import torch
import torch.nn as nn
import torch.optim as optim
from seq2seqbetter import Decoder, Encoder, Seq2Seq
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

val_input = []
val_gold = []
with open('Translation_dataset/val.en') as val_en_file:
    for line in val_en_file.readlines():
        val_input.append('<sos> ' + line[:-1] + ' <eos>')

with open('Translation_dataset/val.fr') as val_fr_file:
    for line in val_fr_file.readlines():
        val_gold.append('<sos> ' + line[:-1] + ' <eos>')

# with open('Translation_dataset/val_en_fr_char_ids.txt', mode='w') as val_file:
#     for sample in val:
#         val_file.write(' '.join(map(str, sample)) + '\n')

test_input = []
test_gold = []
with open('Translation_dataset/test.en') as test_en_file:
    for line in test_en_file.readlines():
        test_input.append('<sos> ' + line[:-1] + ' <eos>')

with open('Translation_dataset/test.fr') as test_fr_file:
    for line in test_fr_file.readlines():
        test_gold.append('<sos> ' + line[:-1] + ' <eos>')

# with open('Translation_dataset/test_en_fr_char_ids.txt', mode='w') as test_file:
#     for sample in test:
#         test_file.write(' '.join(map(str, sample)) + '\n')

batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 5
input_dim = tokenizer.get_vocab_size()
output_dim = input_dim
emb_dim = 200
hid_dim = 100
attn_dim = 40
drop = 0.5
clip = 1
pad_ids = 0
encoder = Encoder(input_dim, emb_dim, hid_dim, hid_dim, drop)
decoder = Decoder(output_dim, emb_dim, hid_dim, hid_dim, drop)
model = Seq2Seq(encoder, decoder, pad_ids, device).to(device=device)

train_loader = DataLoader(list(zip(train_input, train_gold)), batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(list(zip(val_input, val_gold)), batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(list(zip(test_input, test_gold)), batch_size=batch_size, shuffle=True, drop_last=True)


def init_weights(m: nn.Module):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.kaiming_normal_(param.data)
        else:
            nn.init.constant_(param.data, 0)


model.apply(init_weights)

optimizer = optim.Adagrad(model.parameters(), lr=0.01)


loss_function = nn.CrossEntropyLoss(ignore_index=pad_ids)


def train(model, iterator, optimizer, loss_function, clip):

    model.train()

    epoch_loss = 0

    for input_batch, target_batch in iterator:

        lens_input = list(map(len, [tokenizer.encode(x).ids for x in input_batch]))
        batch_sorted = list(zip(lens_input, input_batch, target_batch))
        batch_sorted.sort(reverse=True)
        lens_input, input_batch, target_batch = zip(*(batch_sorted))
        lens_input = torch.tensor(lens_input).to(device=device)

        print(input_batch[0])
        print(target_batch[0])

        input_batch = tokenizer.encode_batch(list(input_batch))
        target_batch = tokenizer.encode_batch(list(target_batch))

        input_batch = [tokenizer.post_process(item).ids for item in input_batch]
        input_batch = torch.tensor(input_batch).to(device).permute([1, 0]).clone()

        target_batch = [tokenizer.post_process(item).ids for item in target_batch]
        target_batch = torch.tensor(target_batch).to(device).permute([1, 0])
        target_batch = target_batch.clone()

        optimizer.zero_grad()

        outputs = model(input_batch, lens_input, target_batch, 1)

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

        epoch_loss += loss.item()

        return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for input_batch, target_batch in iterator:

            lens_input = list(map(len, [tokenizer.encode(x).ids for x in input_batch]))
            batch_sorted = list(zip(lens_input, input_batch, target_batch))
            batch_sorted.sort(reverse=True)
            lens_input, input_batch, target_batch = zip(*(batch_sorted))
            lens_input = torch.tensor(lens_input).to(device=device)

            print(input_batch[0])
            print(target_batch[0])

            input_batch = tokenizer.encode_batch(list(input_batch))
            target_batch = tokenizer.encode_batch(list(target_batch))

            input_batch = [tokenizer.post_process(item).ids for item in input_batch]
            input_batch = torch.tensor(input_batch).to(device).permute([1, 0]).clone()

            target_batch = [tokenizer.post_process(item).ids for item in target_batch]
            target_batch = torch.tensor(target_batch).to(device).permute([1, 0])
            target_batch = target_batch.clone()

            output = model(input_batch, lens_input, target_batch, 0)  # turn off teacher forcing
            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            target_batch = target_batch[1:].view(-1)

            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]

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
