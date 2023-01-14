import json
import os
import torch
from tqdm import tqdm


class Dictionary:
    def __init__(self, dic_path=None):
        self.word2idx = {}
        self.idx2word = []

        if dic_path:
            with open(dic_path, "r", encoding="utf-8") as f:
                dic = json.load(f)
            self.idx2word = list(dic.values())
            self.word2idx = {v: int(k) for k, v in dic.items()}

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus:
    def __init__(self, dic_path=None):
        self.dictionary = Dictionary(dic_path)

    def construct_dict(self, path):
        word_set = set()
        with open(path, 'r', encoding="utf-8") as f:
            for line in f:
                word_set.update(["<bos>"] + list(line.strip()) + ["<eos>"])
            f.close()
        for word in word_set:
            self.dictionary.add_word(word)
        dic = dict(zip(range(len(self.dictionary.idx2word)), self.dictionary.idx2word))
        with open("./data/dict.json", "w", encoding="utf-8") as f:
            json.dump(dic, f, ensure_ascii=False)

    def tokenize(self, path):
        input_ids = []
        input_ids_flatten = []
        root = os.path.dirname(path)
        filename = os.path.basename(path)
        file_type = filename.split("_")[0]
        with open(path, 'r', encoding="utf-8") as f:
            lines = f.readlines()
            for line in tqdm(lines, desc="Tokenizing " + filename):
                words = ["<bos>"] + list(line.strip()) + ["<eos>"]
                words_ids = [self.dictionary.word2idx[word] for word in words]
                input_ids.append(words_ids)
                input_ids_flatten.extend(words_ids)
        save_path = os.path.join(root, file_type + "_input_ids")
        save_path_flatten = os.path.join(root, file_type + "_input_ids_flatten")
        torch.save(input_ids, save_path)
        torch.save(input_ids_flatten, save_path_flatten)

        return input_ids_flatten

    def encode(self, input_text):
        if isinstance(input_text, str):
            input_text = list(input_text)
        input_ids = [self.dictionary.word2idx[word] for word in input_text]
        return input_ids

    def decode(self, input_ids):
        if isinstance(input_ids, int):
            input_ids = [input_ids]
        input_text = [self.dictionary.idx2word[ids] for ids in input_ids]
        input_text = "".join(input_text)
        return input_text

    def create_input_ids(self, root_path):
        train = self.tokenize(os.path.join(root_path, 'train_data.txt'))
        valid = self.tokenize(os.path.join(root_path, 'valid_data.txt'))
        test = self.tokenize(os.path.join(root_path, 'test_data.txt'))

        return train, valid, test
