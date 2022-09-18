import re
import math
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from collections import Counter


def cal_h(dic, n):
    h = 0
    for v in dic.values():
        h += - v / n * math.log(v / n, 2)
    return h



ch_path = "./ch_data.xlsx"
ch_file = pd.read_excel(ch_path)
ch_txt_list = ch_file["text"]

ch_txt_list_processed = []

for txt in tqdm(ch_txt_list):
    if pd.isna(txt):
        continue
    txt_rm = re.sub(r'[^\u4e00-\u9fa5]+', '', txt)
    ch_txt_list_processed.append(txt_rm)
ch_N = len("".join(ch_txt_list_processed))

ch_scale = []
ch_H = []
for num in tqdm(range(350000, ch_N + 1, 350000)):
    random.shuffle(ch_txt_list_processed)
    txt_data = "".join(ch_txt_list_processed)[:num]
    d = dict(Counter(txt_data))
    H_Chinese = cal_h(d, num)
    ch_scale.append(num)
    ch_H.append(H_Chinese)

plt.figure(1)
plt.plot(ch_scale, ch_H, label="Entropy_Chinese")
plt.xlabel("Corpus scale")
plt.ylabel("Entropy")
plt.legend()
plt.show()

en_path = "./en_data.xlsx"
en_file = pd.read_excel(en_path)
en_txt_list = en_file["text"]

en_txt_list_processed = []

for txt in tqdm(en_txt_list):
    if pd.isna(txt):
        continue
    try:
        txt_rm = re.sub(r"[^\u0041-\u005a\u0061-\u007a]+", "", txt)
        txt_rm = txt_rm.lower()
        en_txt_list_processed.append(txt_rm)
    except:
        print(txt)
en_N = len("".join(en_txt_list_processed))

en_scale = []
en_H = []

for num in tqdm(range(1000000, en_N, 1000000)):
    random.shuffle(en_txt_list_processed)
    txt_data = "".join(en_txt_list_processed)[:num]
    d = dict(Counter(txt_data))
    H_English = cal_h(d, num)
    en_scale.append(num)
    en_H.append(H_English)

plt.figure(2)
plt.plot(en_scale, en_H, label="Entropy_English")
plt.xlabel("Corpus scale")
plt.ylabel("Entropy")
plt.legend()
plt.show()

