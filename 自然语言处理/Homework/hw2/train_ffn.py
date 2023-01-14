import os
import time

import torch
import random
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.optim import AdamW
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from model import FeedforwardNN
import corpus

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

parser = ArgumentParser()
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=2048)
parser.add_argument("--seq_len", type=int, default=10)
parser.add_argument("--embeds_size", type=int, default=256)
parser.add_argument("--hidden_size", type=int, default=512)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--max_norm", type=float, default=1)
parser.add_argument("--lr", type=float, default=0.01)
args = parser.parse_args()


def create_batch(d, batch_size, seq_len, device):
    x = [d[i - seq_len:i] for i in range(seq_len, len(d))]
    y = d[seq_len:]

    x = torch.tensor(x, dtype=torch.long).to(device)
    y = torch.tensor(y, dtype=torch.long).to(device)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader


if not (os.path.isfile("./data/dict.json") and os.path.isfile("./data/train_input_ids") and os.path.isfile(
        "./data/valid_input_ids") and os.path.isfile("./data/test_input_ids")):
    my_corpus = corpus.Corpus(dic_path=None)
    my_corpus.construct_dict("./data/data.txt")
    with open("./data/data.txt", "r", encoding="utf-8") as f:
        data = f.readlines()
    train_data, valid_test_data = train_test_split(data, test_size=0.2, random_state=42)
    valid_data, test_data = train_test_split(valid_test_data, test_size=0.5, random_state=42)
    with open("./data/train_data.txt", "w", encoding="utf-8") as f:
        f.writelines(train_data)
    with open("./data/valid_data.txt", "w", encoding="utf-8") as f:
        f.writelines(valid_data)
    with open("./data/test_data.txt", "w", encoding="utf-8") as f:
        f.writelines(test_data)

    train_input_ids_flatten, valid_input_ids_flatten, test_input_ids_flatten = my_corpus.create_input_ids("./data")
else:
    my_corpus = corpus.Corpus(dic_path="./data/dict.json")
    train_input_ids_flatten, valid_input_ids_flatten, test_input_ids_flatten = torch.load(
        "./data/train_input_ids_flatten"), torch.load("./data/valid_input_ids_flatten"), torch.load(
        "./data/test_input_ids_flatten")

epochs = args.epochs
vocab_size = len(my_corpus.dictionary.word2idx)
batch_size = args.batch_size
seq_len = args.seq_len
embeds_size = args.embeds_size
hidden_size = args.hidden_size
dropout = args.dropout
max_norm = args.max_norm
lr = args.lr
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

train_dataloader = create_batch(train_input_ids_flatten, batch_size=batch_size, seq_len=seq_len, device=device)
valid_dataloader = create_batch(valid_input_ids_flatten, batch_size=batch_size, seq_len=seq_len, device=device)
test_dataloader = create_batch(test_input_ids_flatten, batch_size=batch_size, seq_len=seq_len, device=device)
model = FeedforwardNN(vocab_size=vocab_size, num_history=seq_len, embeds_size=embeds_size, hidden_size=hidden_size,
                      dropout=dropout)
model.to(device)
optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
criterion = nn.CrossEntropyLoss()

optimizer_param_name = [{'params': [n for n, p in model.named_parameters()]}]
optimizer_param_num = sum([p.nelement() for n, p in model.named_parameters()])
print(optimizer_param_num)
print(optimizer_param_name)

best_val_loss = float('inf')
val_cnt = 0
train_loss_list = []
for epoch in range(epochs):
    total_loss = 0.0
    step = 0
    step_time = 0
    with tqdm(train_dataloader, desc="Training epoch {}".format(epoch + 1), colour='green', mininterval=30) as pbar:
        for batch in pbar:
            start = time.time()
            model.train()
            x, y = batch
            output = model(x)
            loss = criterion(output, y)
            loss = loss.mean()
            loss.backward()
            train_loss_list.append(loss.item())
            total_loss += loss.item()
            step += 1
            pbar.set_postfix({'loss': total_loss / step, 'ppl': np.exp(total_loss / step),
                              'lr': optimizer.state_dict()['param_groups'][0]['lr'], 'step time': step_time},
                             refresh=False)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            end = time.time()
            step_time = ((end - start) + step_time * (step - 1)) / step
            if step % 1000 == 0:
                model.eval()
                valid_loss = 0.0
                valid_step = 0
                with torch.no_grad():
                    for batch_valid in valid_dataloader:
                        x_valid, y_valid = batch_valid
                        output_valid = model(x_valid)
                        loss = criterion(output_valid, y_valid)
                        loss = loss.mean()
                        valid_loss += loss.item()
                        valid_step += 1
                average_loss = round(valid_loss / valid_step, 3)
                if average_loss < best_val_loss:
                    best_val_loss = average_loss
                else:
                    val_cnt += 1
                    if val_cnt == 2:
                        print("\nvalid loss doesn't decline for 3 valid steps, training stops!")
                ppl_valid = round(np.exp(average_loss), 3)
                print("\nmodel at step {}, nll loss:{}, ppl: {}".format(step, average_loss, ppl_valid))
            if val_cnt == 2:
                break
plt.figure()
plt.plot(range(len(train_loss_list)), train_loss_list)
plt.xlabel("training step")
plt.ylabel("nll loss")
plt.savefig('./ffn_nll_loss.jpg')
plt.show()

plt.figure()
plt.plot(range(len(train_loss_list)), np.exp(train_loss_list))
plt.xlabel("training step")
plt.ylabel("ppl")
plt.savefig('./ffn_ppl.jpg')
plt.show()

model.eval()
test_loss = 0.0
test_step = 0
with torch.no_grad():
    for batch_test in test_dataloader:
        x_test, y_test = batch_test
        output_test = model(x_test)
        loss = criterion(output_test, y_test)
        loss = loss.mean()
        test_loss += loss.item()
        test_step += 1
    average_loss = round(test_loss / test_step, 3)
    ppl_test = round(np.exp(average_loss), 3)
    print("\ntest nll loss:{}, ppl: {}".format(average_loss, ppl_test))
