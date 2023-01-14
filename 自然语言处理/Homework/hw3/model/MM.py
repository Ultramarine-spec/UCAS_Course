import json
import os.path

from sklearn.metrics import classification_report
from tqdm import tqdm


def create_dic(data_path):
    if not os.path.isfile('../dic.json'):
        dic = {}
        with open(data_path, 'r', encoding='gbk') as f:
            data = [x.strip() for x in f.readlines()]
            data = [x for x in data if len(x) != 0]
            data = data[len(data) // 10:]
            f.close()
        word_set = set()
        for sen in tqdm(data):
            sen = sen.split('  ')[1:]
            sen = [x.split('/')[0] for x in sen]
            word_set.update(sen)

        for word, idx in enumerate(word_set):
            dic[word] = idx

        with open('../dic.json', 'w') as f:
            json.dump(dic, f, ensure_ascii=False)
            f.close()
    else:
        with open('../dic.json', 'r', encoding='gbk') as f:
            dic = json.load(f)
            f.close()
    return dic


# Maximum Matching
class MaxMatching:
    def __init__(self, dic, max_length):
        self.dic = dic
        self.max_length = max_length

    def FMM(self, sens):
        pred = []
        for sen in tqdm(sens):
            seg_fmm = []
            flag = 0
            while flag < len(sen):
                match = False
                k = min(self.max_length, len(sen) - flag)
                while k > 0:
                    words = sen[flag:flag + k]
                    if words in self.dic.values():
                        seg_fmm.append(words)
                        match = True
                        break
                    if words.isdigit() or words.encode('utf-8').isalpha():
                        seg_fmm.append(words)
                        match = True
                        break
                    k -= 1
                if not match:
                    k = 1
                    seg_fmm.append(sen[flag])
                flag += k
            pred.append(seg_fmm)
        return pred

    def BMM(self, sens):
        pred = []
        for sen in tqdm(sens):
            seg_bmm = []
            flag = len(sen)
            while flag > 0:
                match = False
                k = min(self.max_length, flag)
                while k > 0:
                    words = sen[flag - k:flag]
                    if words in self.dic.values():
                        seg_bmm.append(words)
                        match = True
                        break
                    if words.isdigit() or words.encode('utf-8').isalpha():
                        seg_bmm.append(words)
                        match = True
                        break
                    k -= 1
                if not match:
                    k = 1
                    seg_bmm.append(sen[flag - 1])
                flag -= k
            seg_bmm.reverse()
            pred.append(seg_bmm)
        return pred

    def MM(self, sens, fmm_out=None, bmm_out=None):
        pred = []
        if fmm_out is None or bmm_out is None:
            fmm_out = self.FMM(sens)
            bmm_out = self.BMM(sens)

        for f, b in zip(fmm_out, bmm_out):
            if len(f) == len(b):
                single_f = [x for x in f if len(x) == 1]
                single_b = [x for x in b if len(x) == 1]
                pred.append(f if len(single_f) < len(single_b) else b)
            else:
                pred.append(f if len(f) < len(b) else b)

        return pred


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


if __name__ == '__main__':
    data_path = '../ChineseCorpus199801.txt'
    word_dict = create_dic(data_path)
    MM_seg = MaxMatching(word_dict, 5)

    with open('../ChineseCorpus199801.txt', 'r', encoding='gbk') as f:
        test_data = [x.strip() for x in f.readlines()]
        test_data = [x for x in test_data if len(x) != 0]
        test_data = test_data[:len(test_data) // 10]
        f.close()
    sen_seg = [x.split('  ')[1:] for x in test_data]
    sen_list = [''.join([x.split('/')[0] for x in sen]) for sen in sen_seg]
    sen_seg = [[x.split('/')[0] for x in sen] for sen in sen_seg]
    gold = [get_label(x) for x in sen_seg]

    fmm_output = MM_seg.FMM(sen_list)
    fmm_pred = [get_label(x) for x in fmm_output]

    bmm_output = MM_seg.BMM(sen_list)
    bmm_pred = [get_label(x) for x in bmm_output]

    mm_output = MM_seg.MM(sen_list, fmm_output, bmm_output)
    mm_pred = [get_label(x) for x in mm_output]

    print('FMM')
    print(classification_report(sum(gold, []), sum(fmm_pred, [])))
    print('BMM')
    print(classification_report(sum(gold, []), sum(bmm_pred, [])))
    print('MM')
    print(classification_report(sum(gold, []), sum(mm_pred, [])))

    # sen = '我爱北京天安门。'
    # print(MM_seg.FMM(sen))
    # print(MM_seg.BMM(sen))
