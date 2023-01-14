import os
import itertools
from collections import Counter
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report


def preprocess(path):
    stopwords = set([x.strip() for x in open('./stopwords.txt', encoding='utf-8').readlines()])
    train_data_path = os.path.join(path, 'traindata.txt')
    with open(train_data_path, 'r', encoding='utf-8') as f:
        train_data = [x.strip() for x in f.readlines()]

    test_data_path = os.path.join(path, 'testdata.txt')
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = [x.strip() for x in f.readlines()]

    train_label_list = [x.split('\t')[0] for x in train_data]
    train_data_list = [[y for y in x.split('\t')[1].split(' ') if y not in stopwords] for x in train_data]

    test_label_list = [x.split('\t')[0] for x in test_data]
    test_data_list = [[y for y in x.split('\t')[1].split(' ') if y not in stopwords] for x in test_data]

    word_list = list(itertools.chain.from_iterable(train_data_list))
    tf_dict = Counter(word_list)

    feature_words = sorted(tf_dict.items(), key=lambda x: x[1], reverse=True)[:5000]
    feature_words = [x[0] for x in feature_words]

    return feature_words, train_data_list, train_label_list, test_data_list, test_label_list


def create_feature(text, feature):
    f = set(feature)
    text_feature = [1 if word in text else 0 for word in f]
    return text_feature


data_path = './data'
feature_num = 1000
delete_n = 10

feature_words, train_data_list, train_label_list, test_data_list, test_label_list = preprocess(data_path)
filtered_feature = []
for i in range(delete_n, len(feature_words)):
    if len(filtered_feature) == feature_num:
        break
    if not feature_words[i].isdigit() and not feature_words[i].encode('utf-8').isalpha() and 1 < len(
            feature_words[i]) < 5:
        filtered_feature.append(feature_words[i])

train_feature_list = [create_feature(text, filtered_feature) for text in train_data_list]
test_feature_list = [create_feature(text, filtered_feature) for text in test_data_list]

classifier = BernoulliNB()
classifier.fit(train_feature_list, train_label_list)

predict_label = classifier.predict(test_feature_list)

print(classification_report(test_label_list, predict_label, digits=4))
