"""
Feature creation helper.
Authors: Jacob Jameson, Deniz Tokmakoglu
"""

from collections import Counter
from torchtext.vocab import Vocab
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
    global vocab 
    vocab = Vocab(counter)
    return vocab, counter

def collate_into_bow(batch, device = DEVICE):
    labels = [0] * len(batch)
    vectors = torch.zeros(len(batch), len(vocab))
    for index, (words, label) in enumerate(batch):
        labels[index] = LABEL_MAPPINGS[label]
        for word in words:
            index_word = vocab[word]    
            vectors[index, int(index_word)] += 1 / len(words)     
    labels = torch.tensor(labels)
    return labels.to(device), vectors.to(device)

def data_loader_bow(data, batch_size, shuffle = False):
    vocab = bow_classifier(data)
    dataloader = DataLoader(data, 
                            batch_size=batch_size, 
                            shuffle=shuffle, 
                            collate_fn=collate_into_bow)
    return dataloader

    
    
### TF - IDF ###

def create_tf_idf(data):

    assert type(data) == type(pd.DataFrame()), "Please use a pandas dataframe"
    text = data["text"].to_list()
    countvectorizer = CountVectorizer(analyzer= 'word', stop_words='english')
    tfidfvectorizer = TfidfVectorizer(analyzer='word',stop_words= 'english')

    # convert th documents into a matrix
    count_wm = countvectorizer.fit_transform(text)
    tfidf_wm = tfidfvectorizer.fit_transform(text)

    count_tokens = countvectorizer.get_feature_names()
    tfidf_tokens = tfidfvectorizer.get_feature_names()

    return  torch.tensor(tfidf_wm.toarray())

## Glove Vectors (Reference gao paper) ## This is not working for now, come back to it later

def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

def get_word2idx_idx2word(vocab):
    """
    :param vocab: a set of strings: vocabulary
    :return: word2idx: string to an int
             idx2word: int to a string
    """
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    idx2word = {0: "<PAD>", 1: "<UNK>"}
    for (word, label)  in vocab:
        for w in word:
            assigned_index = len(word2idx)
            word2idx[w] = assigned_index
            idx2word[assigned_index] = w
    return word2idx, idx2word

def get_embedding_matrix(glove_path, word2idx, idx2word, normalization=False):
    """
    assume padding index is 0
    :param word2idx: a dictionary: string --> int, includes <PAD> and <UNK>
    :param idx2word: a dictionary: int --> string, includes <PAD> and <UNK>
    :param normalization:
    :return: an embedding matrix: a nn.Embeddings
    """
    # Load the GloVe vectors into a dictionary, keeping only words in vocab
    embedding_dim = 300
    # glove_path = "../glove/glove840B300d.txt" # don't hardcode path
    glove_vectors = {}
    with open(glove_path) as glove_file:
        for line in tqdm(glove_file, total=get_num_lines(glove_path)):
            split_line = line.rstrip().split()
            word = split_line[0]
            if len(split_line) != (embedding_dim + 1) or word not in word2idx:
                continue
            assert (len(split_line) == embedding_dim + 1)
            vector = np.array([float(x) for x in split_line[1:]], dtype="float32")
            if normalization:
                vector = vector / np.linalg.norm(vector)
            assert len(vector) == embedding_dim
            glove_vectors[word] = vector

    print("Number of pre-trained word vectors loaded: ", len(glove_vectors))

    # Calculate mean and stdev of embeddings
    all_embeddings = np.array(list(glove_vectors.values()))
    embeddings_mean = float(np.mean(all_embeddings))
    embeddings_stdev = float(np.std(all_embeddings))
    print("Embeddings mean: ", embeddings_mean)
    print("Embeddings stdev: ", embeddings_stdev)

    # Randomly initialize an embedding matrix of (vocab_size, embedding_dim) shape
    # with a similar distribution as the pretrained embeddings for words in vocab.
    vocab_size = len(word2idx)
    embedding_matrix = torch.FloatTensor(vocab_size, embedding_dim).normal_(embeddings_mean, embeddings_stdev)
    # Go through the embedding matrix and replace the random vector with a
    # pretrained one if available. Start iteration at 2 since 0, 1 are PAD, UNK
    for i in range(2, vocab_size):
        word = idx2word[i]
        if word in glove_vectors:
            embedding_matrix[i] = torch.FloatTensor(glove_vectors[word])
    if normalization:
        for i in range(vocab_size):
            embedding_matrix[i] = embedding_matrix[i] / float(np.linalg.norm(embedding_matrix[i]))
    embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
    embeddings.weight = nn.Parameter(embedding_matrix)
    return embeddings

def embed_glove_words(words, word2idx, glove_embeddings):
    """
    Assume that word2idx has 1 mapped to UNK
    Assume that word2idx maps well implicitly with glove_embeddings
    i.e. the idx for each word is the row number for its corresponding embedding
    :param sequence: a single string: a sentence with space
    :param word2idx: a dictionary: string --> int
    :param glove_embeddings: a nn.Embedding with padding idx 0
    :param elmo_embeddings: a h5py file
                    each group_key is a string: a sentence
                    each inside group is an np array (seq_len, 1024 elmo)
    :param suffix_embeddings: a nn.Embedding without padding idx
    :return: a np.array (seq_len, embed_dim=glove+elmo+suffix)
    """

    # 1. embed the sequence by glove vector
    # Replace words with tokens, and 1 (UNK index) if words not indexed.
    indexed_sequence = []
    for x in words:
        if x != "i":
            indexed_sequence.append(word2idx.get(x, 1))
        else:
            indexed_sequence.append(1)
   
    # glove_part has shape: (seq_len, glove_dim)
    
    variable = Variable(torch.LongTensor(indexed_sequence))
    print(variable)
    for i, word in enumerate(indexed_sequence):
        print(words[i])
        print("*-*-")
        print(word)
        glove_part = glove_embeddings(Variable(torch.LongTensor(word)))
    
    # concatenate three parts: glove+elmo+suffix along axis 1
    # glove_part and suffix_part are Variables, so we need to use .data
    # otherwise, throws weird ValueError: incorrect dimension, zero-dimension, etc..
    
    return glove_part

##################################################################

## Glove with Torchtext (Bag of Words)

def collate_into_cbow_glove(object,  device = DEVICE):
    batch, glove = object
    labels = [0] * len(batch)
    vectors = torch.zeros(len(batch), len(vocab))
    for index, (words, label) in enumerate(batch):
        labels[index] = LABEL_MAPPINGS[label]
        glove_embedding = glove.get_vecs_by_tokens(words)
        vectors[index] = glove_embedding
    labels = torch.tensor(labels)
    return labels.to(device), vectors.to(device)
    

def data_loader_bow_glove(object, batch_size, shuffle = False):
    dataloader = DataLoader(object, 
                            batch_size=batch_size, 
                            shuffle=shuffle, 
                            collate_fn=collate_into_cbow_glove)
    return dataloader