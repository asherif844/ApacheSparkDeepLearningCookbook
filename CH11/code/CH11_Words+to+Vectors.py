
# coding: utf-8

# In[1]:


pwd


# In[2]:


cd '/Users/Chanti/Desktop/USF'


# In[3]:


from __future__ import absolute_import, division, print_function


# In[4]:


import codecs
import glob
import logging
import multiprocessing
import os
import pprint
import re


# In[5]:


import nltk
import gensim.models.word2vec as w2v
import sklearn.manifold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[6]:


get_ipython().magic('pylab inline')


# In[7]:


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# In[8]:


nltk.download("punkt")
nltk.download("stopwords")


# In[9]:


book_names = sorted(glob.glob("./*.txt"))


# In[10]:


print("Found books:")
book_names


# In[11]:


corpus = u''
for book_name in book_names:
    print("Reading '{0}'...".format(book_name))
    with codecs.open(book_name,"r","Latin1") as book_file:
        corpus += book_file.read()
    print("Corpus is now {0} characters long".format(len(corpus)))
    print()


# In[12]:


#Load the English pickle tokenizer from punkt
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


# In[13]:


#Tokenize the corpus into sentences
raw_sentences = tokenizer.tokenize(corpus)


# In[14]:


#Convert sentences into list of words
#remove unecessary characters, split into words, remove hyphens and special characters
def sentence_to_wordlist(raw):
    clean = re.sub("[^a-zA-Z]"," ", raw)
    words = clean.split()
    return words


# In[15]:


#for each sentence, sentences where each word is tokenized
sentences = []
for raw_sentence in raw_sentences:
    if len(raw_sentence) > 0:
        sentences.append(sentence_to_wordlist(raw_sentence))


# In[16]:


print(raw_sentences[50])
print(sentence_to_wordlist(raw_sentences[50]))


# In[17]:


#count tokens, each one being a sentence
token_count = sum([len(sentence) for sentence in sentences])
print("The book corpus contains {0:,} tokens".format(token_count))


# In[18]:


#Define hyperparameters

# Dimensionality of the resulting word vectors.
num_features = 300

# Minimum word count threshold.
min_word_count = 3

# Number of threads to run in parallel.
num_workers = multiprocessing.cpu_count()

# Context window length.
context_size = 7

# Downsample setting for frequent words.
downsampling = 1e-3

# Seed for the RNG, to make the results reproducible.
seed = 1


# In[19]:


got2vec = w2v.Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)


# In[20]:


got2vec.build_vocab(sentences,progress_per=10000, keep_raw_vocab=False, trim_rule=None)


# In[21]:


#train model on sentences
got2vec.train(sentences, total_examples=got2vec.corpus_count, 
              total_words=None, epochs=got2vec.iter, 
              start_alpha=None, end_alpha=None, word_count=0, 
              queue_factor=2, report_delay=1.0, compute_loss=False)


# In[22]:


#save model
if not os.path.exists("trained"):
    os.makedirs("trained")


# In[23]:


got2vec.wv.save(os.path.join("trained", "got2vec.w2v"), ignore=[])


# In[24]:


#load model
got2vec = w2v.KeyedVectors.load(os.path.join("trained", "got2vec.w2v"))


# In[25]:


#Squash dimensionality to 2
tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)


# In[26]:


#Put all the word vectors into one big matrix
all_word_vectors_matrix = got2vec.wv.syn0


# In[27]:


print (all_word_vectors_matrix)


# In[28]:


#train tsne
all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)


# In[29]:


#plot point in 2d space
points = pd.DataFrame(
    [
        (word, coords[0], coords[1])
        for word, coords in [
            (word, all_word_vectors_matrix_2d[got2vec.vocab[word].index])
            for word in got2vec.vocab
        ]
    ],
    columns=["word", "x", "y"]
)


# In[30]:


points.head(20)


# In[31]:


# Plotting using the seaborn library
sns.set_context("poster")


# In[32]:


points.plot.scatter("x", "y", s=10, figsize=(10, 10))


# In[33]:


def plot_region(x_bounds, y_bounds):
    slice = points[
        (x_bounds[0] <= points.x) &
        (points.x <= x_bounds[1]) & 
        (y_bounds[0] <= points.y) &
        (points.y <= y_bounds[1])
    ]
    
    ax = slice.plot.scatter("x", "y", s=35, figsize=(10, 8))
    for i, point in slice.iterrows():
        ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=11)


# In[34]:


plot_region(x_bounds=(20.0, 25.0), y_bounds=(15.5, 20.0))


# In[35]:


plot_region(x_bounds=(4, 41), y_bounds=(-0.5, -0.1))


# In[36]:


plot_region(x_bounds=(10, 15), y_bounds=(5, 10))


# In[37]:


got2vec.most_similar("Stark")


# In[38]:


got2vec.most_similar("Lannister")


# In[39]:


got2vec.most_similar("Jon")


# In[40]:


#distance, similarity, and ranking
def nearest_similarity_cosmul(start1, end1, end2):
    similarities = got2vec.most_similar_cosmul(
        positive=[end2, start1],
        negative=[end1]
    )
    start2 = similarities[0][0]
    print("{start1} is related to {end1}, as {start2} is related to {end2}".format(**locals()))
    return start2


# In[41]:


nearest_similarity_cosmul("Stark", "Winterfell", "Riverrun")
nearest_similarity_cosmul("Jaime", "sword", "wine")
nearest_similarity_cosmul("Arya", "Nymeria", "dragons")

