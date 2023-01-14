import json
from sklearn.model_selection import train_test_split


class Tokenizer:
    def __init__(self, dic_path=None, word_list=None):
        if dic_path:
            with open(dic_path, 'r') as f:
                dic = json.load(f)
            self.idx2word = dic
            self.word2idx = {v: int(k) for k, v in dic.items()}

            if word_list:
                self.construct_dic(word_list)

        elif word_list:
            self.word2idx = dict()
            self.idx2word = dict()
            self.construct_dic(word_list)

        else:
            self.word2idx = dict()
            self.idx2word = dict()

    def construct_dic(self, word_list):
        for word in word_list:
            self.add_word(word)
        return self.idx2word, self.word2idx

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.setdefault(len(self.idx2word), word)
            self.word2idx.setdefault(word, len(self.word2idx))
        return self.word2idx.get(word)

    def encode(self, input_word):
        unk_idx = self.word2idx.get('<unk>')
        if not isinstance(input_word, list):
            input_idx = self.word2idx.get(input_word, unk_idx)
        else:
            input_idx = [self.word2idx.get(word, unk_idx) for word in input_word]
        return input_idx

    def batch_encode(self, input_lists):
        return [self.encode(x) for x in input_lists]

    def decode(self, input_idx):
        if not isinstance(input_idx, list):
            input_word = self.idx2word.get(input_idx)
        else:
            input_word = [self.idx2word.get(idx) for idx in input_idx]
        return input_word

    def batch_decode(self, input_lists):
        return [self.decode(x) for x in input_lists]

    def __len__(self):
        return len(self.word2idx)


def build(filename, dic_path=None):
    with open(filename, 'r', encoding='gbk') as f:
        data = [x.strip() for x in f.readlines()]
        data = [x for x in data if len(x) != 0]

    sample_list = []
    word_list = []
    label_list = []
    for sen in data:
        if len(sen) == 0:
            continue
        sen_seg = sen.split('  ')[1:]
        sen_seg = [x.split('/')[0] for x in sen_seg]
        sample_list.append(list("".join(sen_seg)) + ["<eos>"])
        word_list.extend(list("".join(sen_seg)))
        label_list.append(get_label(sen_seg) + ["<eos>"])

    tokenizer = Tokenizer(dic_path=dic_path, word_list=word_list)
    label_dict = Tokenizer(word_list=['S', 'B', 'M', 'E'])
    for special_token in ['<bos>', '<eos>', '<pad>', '<unk>']:
        tokenizer.add_word(special_token)
        if special_token != '<unk>':
            label_dict.add_word(special_token)

    test_data = sample_list[:len(sample_list) // 10]
    test_label = label_list[:len(label_list) // 10]

    sample_list = sample_list[len(sample_list) // 10:]
    label_list = label_list[len(label_list) // 10:]

    train_data, valid_data, train_label, valid_label = train_test_split(sample_list, label_list,
                                                                        test_size=0.1, random_state=42)

    return train_data, train_label, valid_data, valid_label, test_data, test_label, tokenizer, label_dict


def get_label(sen_seg):
    label = []
    for token in sen_seg:
        if len(token) == 1:
            label.append('S')
        elif len(token) == 2:
            label.extend('BE')
        else:
            label.extend('B' + 'M' * (len(token) - 2) + 'E')
    return label
