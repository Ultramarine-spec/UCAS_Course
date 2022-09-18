import os
from argparse import ArgumentParser
import pandas as pd
import requests
from bs4 import BeautifulSoup

from tqdm import tqdm

headers = {
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.164 Safari/537.36"
}
proxies = {
    "http": "http://127.0.0.1:7890",
    "https": "http://127.0.0.1:7890",
}


def get_text_ch(url, num, save_path, save_feq=10):
    os.makedirs(save_path, exist_ok=True)
    dic_ch = {}
    for idx in tqdm(range(1, num + 1, 1)):
        try:
            r = requests.get(url + str(idx) + '.htm', headers=headers, proxies=proxies)
            # r.encoding = r.apparent_encoding
            soup = BeautifulSoup(r.text, 'html.parser')
            text = ''
            for x in soup.find_all('div', {'class': 'para'}):
                text += x.text
            dic_ch[url + str(idx) + '.htm'] = text
            r.close()
        except:
            pass

        if idx % save_feq == 0:
            u = list(dic_ch.keys())
            t = list(dic_ch.values())
            result = pd.DataFrame()
            result["url"] = u
            result["text"] = t
            result.to_excel(os.path.join(save_path, "./data{}.xlsx".format(idx)))
            dic_ch = {}


def get_text_en(url, num, keyword, save_path, save_feq=10):
    os.makedirs(save_path, exist_ok=True)
    mem = set()
    mem.add("/wiki/" + keyword)
    dic_en = {}
    for idx in tqdm(range(1, num + 1, 1)):
        key = mem.pop()
        try:
            r = requests.get(url + key, headers=headers, proxies=proxies)
            # r.encoding = r.apparent_encoding
            soup = BeautifulSoup(r.text, 'html.parser')
            text = ''
            for x in soup.find_all('p'):
                text += x.text
            cnt = 0
            link = soup.find_all('a')
            # random.shuffle(link)
            for x in link:
                tmp = x.get('href')
                if tmp and tmp.startswith('/wiki/') and tmp not in mem:
                    mem.add(tmp)
                    cnt += 1
                    if cnt == 3:
                        break
            dic_en[url + key] = text
            r.close()
        except:
            pass
        if idx % save_feq == 0:
            u = list(dic_en.keys())
            t = list(dic_en.values())
            result = pd.DataFrame()
            result["url"] = u
            result["text"] = t
            result.to_excel(os.path.join(save_path, "./data{}.xlsx".format(idx)))
            dic_en = {}


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--N_ch", type=int, default=100)
    parser.add_argument("--N_en", type=int, default=100)
    args = parser.parse_args()
    url_cn = "https://baike.baidu.com/view/"
    get_text_ch(url_cn, args.N_ch, "./cn_data")
    url_en = "https://en.wikipedia.org/"
    get_text_en(url_en, args.N_en, 'country', "./en_data")
