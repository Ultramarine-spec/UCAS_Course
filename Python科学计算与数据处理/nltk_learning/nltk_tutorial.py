import nltk
from nltk.book import *
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

"""Tokenizing"""
example_string = "Python is a high-level, general-purpose programming language. " \
                 "Its design philosophy emphasizes code readability with the use of significant indentation."
sent_list = sent_tokenize(example_string)
word_list = word_tokenize(example_string)

"""Filtering Stop Words"""
stop_words = set(stopwords.words("english"))
filtered_list = [word for word in word_list if word.casefold() not in stop_words]

"""Stemming"""
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in word_list]

"""Tagging Parts of Speech"""
pos_tag = nltk.pos_tag(word_list)
nltk.help.upenn_tagset()

"""Lemmatizing"""
lemmatizer = WordNetLemmatizer()
lemmatizer.lemmatize("scarves")
lemmatizer.lemmatize("worst", pos='a')
string_for_lemmatizing = "The friends of DeSoto love scarves."
lemmatized_words = [lemmatizer.lemmatize(word) for word in word_tokenize(string_for_lemmatizing)]
print(lemmatized_words)

"""Chunking"""
lotr_quote = "It's a dangerous business, Frodo, going out your door."
lotr_pos_tags = nltk.pos_tag(word_tokenize(lotr_quote))
grammar = "NP: {<DT>?<JJ>*<NN>}"
chunk_parser = nltk.RegexpParser(grammar)
tree = chunk_parser.parse(lotr_pos_tags)
tree.draw()

"""Chinking"""
grammar = "Chunk: {<.*>+}\n}<JJ>{"
chunk_parser = nltk.RegexpParser(grammar)
tree = chunk_parser.parse(lotr_pos_tags)
tree.draw()

"""Using Named Entity Recognition"""
tree = nltk.ne_chunk(lotr_pos_tags)
tree.draw()
tree = nltk.ne_chunk(lotr_pos_tags, binary=True)
tree.draw()


def extract_ne(text):
    tags = nltk.pos_tag(word_tokenize(text))
    tree = nltk.ne_chunk(tags)
    return [(" ".join(i[0] for i in t), t.label()) for t in tree if hasattr(t, "label")]


text = """
Men like Schiaparelli watched the red planet—it is odd, by-the-bye, that
for countless centuries Mars has been the star of war—but failed to
interpret the fluctuating appearances of the markings they mapped so well.
All that time the Martians must have been getting ready.

During the opposition of 1894 a great light was seen on the illuminated
part of the disk, first at the Lick Observatory, then by Perrotin of Nice,
and then by other observers. English readers heard of it first in the
issue of Nature dated August 2."""

print(extract_ne(text))


"""Getting Text to Analyze"""
text8.concordance("man")
text8.concordance("woman")


"""Making a Dispersion Plot"""
text8.dispersion_plot(
    ["woman", "lady", "girl", "gal", "man", "gentleman", "boy", "guy"]
)
text2.dispersion_plot(["Allenham", "Whitwell", "Cleveland", "Combe"])


"""Making a Frequency Distribution"""
frequency_distribution = FreqDist(text8)
print(frequency_distribution.most_common(20))

meaningful_words = [
    word for word in text8 if word.casefold() not in stop_words
]
frequency_distribution = FreqDist(meaningful_words)
print(frequency_distribution.most_common(20))
frequency_distribution.plot(20, cumulative=True)


"""Finding Collocations"""
text8.collocations()
lemmatized_words = [lemmatizer.lemmatize(word) for word in text8]
new_text = nltk.Text(lemmatized_words)
new_text.collocations()