"""
Feature creation helper.
Authors: Jacob Jameson, Deniz Tokmakoglu
"""

from collections import Counter
from torchtext.vocab import Vocab
import torch
from torch.utils.data import DataLoader

LABEL_MAPPINGS = {"suicide": 1, "non-suicide": 0}
DEVICE = 'cuda' # change if working on a CPU

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

    
    
