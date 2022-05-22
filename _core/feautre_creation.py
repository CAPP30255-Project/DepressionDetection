"""
Feature creation helper.
Authors: Jacob Jameson, Deniz Tokmakoglu
"""

from collections import Counter
from torchtext.vocab import vocab as v 
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from torch.autograd import Variable
from tqdm import tqdm
import mmap
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer


LABEL_MAPPINGS = {"suicide": 1, "non-suicide": 0}
DEVICE = 'cuda' # change if working on a CPU

### Bag of Words ###

def bow_classifier(data):
    counter = Counter()
    for (line, label) in data:
        counter.update(line)
    global vocab_words
    vocab_words = v(counter, specials = ['<unk>'], special_first = True, min_freq = 1000)
    vocab_words.set_default_index(0)
    return vocab_words, counter

def collate_into_bow(batch, device = DEVICE):
    labels = [0] * len(batch)
    vectors = torch.zeros(len(batch), len(vocab_words))
    for index, (words, label) in enumerate(batch):
        labels[index] = LABEL_MAPPINGS[label]
        for word in words:
            index_word = vocab_words[word]    
            vectors[index, int(index_word)] += 1 / len(words)     
    labels = torch.tensor(labels)
    return labels.to(device), vectors.to(device)

def data_loader_bow(data, vocab_words, batch_size, shuffle = False, data_object = None):
    if data_object:
        vocab_words = bow_classifier(data_object.all_data)
    print("Vocab Size = ", len(vocab_words))
    dataloader = DataLoader(data, 
                            batch_size=batch_size, 
                            shuffle=shuffle, 
                            collate_fn=collate_into_bow)
    return dataloader


    
### TF - IDF ###

def create_tf_idf(data):

    assert type(data) == type(pd.DataFrame()), "Please use a pandas dataframe"
    text = data["text"].to_list()
    labels = data["class"].map(LABEL_MAPPINGS).to_list()
    countvectorizer = CountVectorizer(analyzer= 'word', stop_words='english', min_df=1000)
    tfidfvectorizer = TfidfVectorizer(analyzer='word',stop_words= 'english',min_df=1000)

    # convert th documents into a matrix
    count_wm = countvectorizer.fit_transform(text)
    tfidf_wm = tfidfvectorizer.fit_transform(text)

    count_tokens = countvectorizer.get_feature_names()
    tfidf_tokens = tfidfvectorizer.get_feature_names()
    params = tfidfvectorizer.get_params()

    tfidf_tokens_mapping = {token: index for index, token in enumerate(tfidf_tokens)}

    return  torch.tensor(tfidf_wm.toarray()), torch.tensor(labels), params