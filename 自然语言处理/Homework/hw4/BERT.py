import itertools
import os
from collections import Counter, defaultdict
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report

from transformers import BertForSequenceClassification, BertTokenizerFast, AdamW, get_linear_schedule_with_warmup

punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~“”？，！【】（）、。：；’‘……￥·"""


def get_data(data_path):
    stopwords = set([x.strip() for x in open('./data/stopwords.txt', encoding='utf-8').readlines()])
    with open(data_path, 'r', encoding='utf-8') as f:
        data = [x.strip() for x in f.readlines()]
    label_list = [x.split('\t')[0] for x in data]
    data_list = [[y for y in x.split('\t')[1].split(' ') if (y not in stopwords and y not in punctuation)] for x in
                 data]
    new_data_list = [''.join(x) for x in data_list]

    # word_list = list(itertools.chain.from_iterable(data_list))
    # tf_dict = Counter(word_list)
    #
    # idf_dict = defaultdict(int)
    # for d in data_list:
    #     for w in set(d):
    #         idf_dict[w] += 1
    #
    # tf_idf_dict = {}
    #
    # for word in tf_dict.keys():
    #     tf_idf_dict.setdefault(word, tf_dict[word] * np.log(len(data_list) / idf_dict[word]))
    #
    # feature_words = sorted(tf_idf_dict.items(), key=lambda x: x[1], reverse=True)[:2000]
    # feature_words = set([x[0] for x in feature_words])
    #
    # new_data_list = [[y for y in x if y in feature_words] for x in data_list]
    # new_data_list = [''.join(x) for x in new_data_list]

    return new_data_list, label_list


class MyDataset(Dataset):
    def __init__(self, data_list, label_list, tokenizer, device):
        assert len(data_list) == len(label_list)
        self.encoded_data = tokenizer(data_list, return_tensors='pt', truncation=True, padding=True, max_length=512)
        self.labels = label_list
        self.device = device

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: value[idx].to(self.device) for key, value in self.encoded_data.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long).to(self.device)
        return item


def train():
    model.train()
    best_val_loss = float('inf')
    val_cnt = 0
    total_loss = 0.
    step = 0

    for epoch in range(total_epochs):
        pbar = tqdm(train_loader, desc='Epoch {}'.format(epoch + 1))
        for batch in pbar:
            step += 1
            outputs = model(**batch)

            loss = outputs.loss

            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)

            pbar.set_postfix({'current loss': loss.item(),
                              'average loss': total_loss / step,
                              'lr': optimizer.state_dict()['param_groups'][0]['lr']}, refresh=True)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if step % 100 == 0:
                print("Starting validating for step {}".format(step))
                model.eval()
                with torch.no_grad():
                    pred_list = []
                    label_list = []
                    valid_total_loss = 0.
                    valid_step = 0
                    correct = 0
                    for valid_batch in valid_loader:
                        valid_step += 1
                        valid_outputs = model(**valid_batch)

                        valid_loss = valid_outputs.loss.item()
                        valid_total_loss += valid_loss

                        logits = valid_outputs.logits.detach().cpu().numpy()
                        pred = logits.argmax(axis=1).tolist()
                        label_ids = valid_batch['labels'].cpu().numpy().tolist()
                        correct += np.sum(np.array(pred) == np.array(label_ids))

                        pred_list.extend(pred)
                        label_list.extend(label_ids)

                    valid_average_loss = valid_total_loss / valid_step
                    accuracy = correct / len(valid_loader.dataset)
                    print("average loss: {}, accuracy: {}".format(valid_average_loss, accuracy))
                    print(classification_report(label_list, pred_list, digits=4))

                if valid_average_loss < best_val_loss:
                    val_cnt = 0
                    best_val_loss = valid_average_loss
                    save_path = 'bert_step{}'.format(step)
                    model.save_pretrained(save_path)
                else:
                    val_cnt += 1
            if val_cnt == 5:
                print("Valid loss doesn't decline for 10 valid steps, training stops!")
                break
        if val_cnt == 5:
            break


def test(model_path, test_loader):
    model = BertForSequenceClassification.from_pretrained(model_path)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()
    with torch.no_grad():
        pred_list = []
        label_list = []
        for test_batch in test_loader:
            test_outputs = model(**test_batch)
            logits = test_outputs.logits.detach().cpu().numpy()
            pred = logits.argmax(axis=1).tolist()
            label_ids = test_batch['labels'].cpu().numpy().tolist()

            pred_list.extend(pred)
            label_list.extend(label_ids)

    return pred_list, label_list


if __name__ == "__main__":
    total_epochs = 5
    tokenizer = BertTokenizerFast.from_pretrained('./pretrained_model/bert-base-chinese')
    model = BertForSequenceClassification.from_pretrained('./pretrained_model/bert-base-chinese', num_labels=5)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    train_data_list, train_label_list = get_data('./data/traindata.txt')
    valid_data_list, valid_label_list = get_data('./data/devdata.txt')
    test_data_list, test_label_list = get_data('./data/testdata.txt')

    label2idx = {}
    for i, ele in enumerate(set(train_label_list)):
        label2idx[ele] = i
    train_label_list = [label2idx[label] for label in train_label_list]
    valid_label_list = [label2idx[label] for label in valid_label_list]
    test_label_list = [label2idx[label] for label in test_label_list]

    train_dataset = MyDataset(train_data_list, train_label_list, tokenizer, device)
    valid_dataset = MyDataset(valid_data_list, valid_label_list, tokenizer, device)
    test_dataset = MyDataset(test_data_list, test_label_list, tokenizer, device)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=True)
    total_steps = len(train_loader) * total_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)
    train()
