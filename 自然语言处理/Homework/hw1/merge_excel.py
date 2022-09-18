import os
import pandas as pd
from tqdm import tqdm

ch_path = "./ch_data"
ch_urls = []
ch_texts = []
for filename in tqdm(os.listdir(ch_path)):
    file = pd.read_excel(os.path.join(ch_path, filename))
    ch_urls.extend(list(file['url']))
    ch_texts.extend(list(file['text']))
ch_result = pd.DataFrame()
ch_result["url"] = ch_urls
ch_result["text"] = ch_texts
ch_result.to_excel("./ch_data.xlsx")

en_path = "./en_data"
en_urls = []
en_texts = []
for filename in tqdm(os.listdir(en_path)):
    file = pd.read_excel(os.path.join(en_path, filename))
    en_urls.extend(list(file['url']))
    en_texts.extend(list(file['text']))
en_result = pd.DataFrame()
en_result["url"] = en_urls
en_result["text"] = en_texts
en_result.to_excel("./en_data.xlsx")