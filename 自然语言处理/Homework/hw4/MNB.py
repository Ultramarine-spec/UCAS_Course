import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB

data_path = './data'
stopwords = set([x.strip() for x in open('./stopwords.txt', encoding='utf-8').readlines()])
train_data_path = os.path.join(data_path, 'traindata.txt')
with open(train_data_path, 'r', encoding='utf-8') as f:
    train_data = [x.strip() for x in f.readlines()]

test_data_path = os.path.join(data_path, 'testdata.txt')
with open(test_data_path, 'r', encoding='utf-8') as f:
    test_data = [x.strip() for x in f.readlines()]

train_label_list = [x.split('\t')[0] for x in train_data]
train_data_list = [x.split('\t')[1] for x in train_data]
test_label_list = [x.split('\t')[0] for x in test_data]
test_data_list = [x.split('\t')[1] for x in test_data]

tf = TfidfVectorizer(stop_words=stopwords)
train_features = tf.fit_transform(train_data_list)
test_features = tf.transform(test_data_list)

clf = MultinomialNB(alpha=0.001)
clf.fit(train_features, train_label_list)
predict_label = clf.predict(test_features)

print(classification_report(test_label_list, predict_label, digits=4))
