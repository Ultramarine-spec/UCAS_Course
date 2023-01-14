from collections import defaultdict


def compress(sens):
    word_dict = defaultdict(int)
    for word in sum(sens, []):
        if len(word) > 1:
            for i in range(len(word) - 1):
                word_dict[word[i:i + 2]] += 1
    max_word = max(word_dict, key=word_dict.get)
    print('Replace {} with @'.format(max_word))
    replaced = sens.copy()
    for sen in replaced:
        for i in range(len(sen)):
            sen[i] = sen[i].replace(max_word, '@')
    return replaced


with open('./ChineseCorpus199801.txt', 'r', encoding='gbk') as f:
    test_data = [x.strip() for x in f.readlines()]
    test_data = [x for x in test_data if len(x) != 0]
    test_data = test_data[:5]
    f.close()
sen_seg = [x.split('  ')[1:] for x in test_data]
sen_seg = [[x.split('/')[0] for x in sen] for sen in sen_seg]

compress_1 = compress(sen_seg)
print([' / '.join(x) for x in compress_1])
compress_2 = compress(compress_1)
print([' / '.join(x) for x in compress_2])

