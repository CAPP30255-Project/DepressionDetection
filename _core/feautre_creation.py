"""
Feature creation helper.
Authors: Jacob Jameson, Deniz Tokmakoglu
"""

from collections import Counter
from torchtext.vocab import vocab
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
    vocab = vocab(counter, min_freq=1000)
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

def data_loader_bow(data, vocab, batch_size, shuffle = False, data_object = None):
    if data_object:
        vocab = bow_classifier(data_object.all_data)
    print("Vocab Size = ", len(vocab))
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

def get_embedding_matrix(glove_path, word2idx, normalization=False):
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
    all_embeddings = np.array(list(glove_vectors.values()))
    embeddings_mean = float(np.mean(all_embeddings))
    embeddings_stdev = float(np.std(all_embeddings))
    glove_vectors["<UNK>"] = np.random.normal(embeddings_mean, embeddings_stdev, embedding_dim)
    return glove_vectors

def embed_glove_words(words, glove_embeddings):
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
    #indexed_sequence = []
    #for x in words:
    #    if x != "i":
    #        indexed_sequence.append(word2idx.get(x, 1))
    #    else:
    #        indexed_sequence.append(1)
   
    # glove_part has shape: (seq_len, glove_dim)
    embeddings = []
    for i, word in enumerate(words):
        embedding = glove_embeddings.get(word, glove_embeddings["<UNK>"])
        embeddings.append(embedding) 
    return torch.from_numpy(np.array(embeddings))

def embed_data(data, glove):
    embedded = []
    for words, label in data:
        embeddings = embed_glove_words(words, glove)
        embedded.append([embeddings, label])
    return embedded


## Glove with Torchtext (Bag of Words)

def collate_into_cbow_glove(object, embedding_dim = 300, device = DEVICE):
    batch, glove_embeddings, vocab = object
    vocab = vocab[0]
    labels = [0] * len(batch)
    vocab_size = len(vocab)
    vectors = torch.zeros(len(batch), vocab_size, embedding_dim)
    for index, (word, label) in enumerate(batch):
        labels[index] = LABEL_MAPPINGS[label]
        for w in word:
            index_word = vocab[w]    
            vectors[index, int(index_word)]= glove_embeddings.get(w, glove_embeddings["<UNK>"])
    labels = torch.tensor(labels)
    return labels.to(device), vectors.to(device)
    

def data_loader_bow_glove(object, batch_size, shuffle = False):
    data, glove = object
    vocab = bow_classifier(data)

    object2 = data, glove, vocab
    
    dataloader = DataLoader(object2, 
                            batch_size=batch_size, 
                            shuffle=shuffle, 
                            collate_fn=collate_into_cbow_glove)
    return dataloader

class TextWithGlove():
    def __init__(self, embedded_text, labels, max_sequence_length=100):
        """
        :param embedded_text:
        :param labels: a list of ints
        :param max_sequence_length: an int
        """
        if len(embedded_text) != len(labels):
            raise ValueError("Differing number of sentences and labels!")
        # A list of numpy arrays, where each inner numpy arrays is sequence_length * embed_dim
        # embedding for each word is : glove + elmo + suffix
        self.embedded_text = embedded_text
        # A list of ints, where each int is a label of the sentence at the corresponding index.
        self.labels = labels
        # Truncate examples that are longer than max_sequence_length.
        # Long sequences are expensive and might blow up GPU memory usage.
        self.max_sequence_length = max_sequence_length
    def __getitem__(self, idx):
        """
        Return the Dataset example at index `idx`.
        Returns
        -------
        example_text: numpy array
        length: int
            The length of the (possibly truncated) example_text.
        example_label: int 0 or 1
            The label of the example.
        """
        example_text = self.embedded_text[idx]
        example_label = self.labels[idx]
        # Truncate the sequence if necessary
        example_text = example_text[:self.max_sequence_length]
        example_length = example_text.shape[0]

        return example_text, example_length, example_label

    def __len__(self):
        """
        Return the number of examples in the Dataset.
        """
        return len(self.labels)

    @staticmethod
    def collate_fn(batch):
        """
        Given a list of examples (each from __getitem__),
        combine them to form a single batch by padding.
        Returns:
        -------
        batch_padded_example_text: LongTensor
          LongTensor of shape (batch_size, longest_sequence_length) with the
          padded text for each example in the batch.
        length: LongTensor
          LongTensor of shape (batch_size,) with the unpadded length of the example.
        example_label: LongTensor
          LongTensor of shape (batch_size,) with the label of the example.
        """
        batch_padded_example_text = []
        batch_lengths = []
        batch_labels = []

        # Get the length of the longest sequence in the batch
        max_length = max(batch, key=lambda example: example[1])[1]

        # Iterate over each example in the batch
        for text, length, label in batch:
            # Unpack the example (returned from __getitem__)

            # Amount to pad is length of longest example - length of this example.
            amount_to_pad = max_length - length
            # Tensor of shape (amount_to_pad,), converted to LongTensor
            pad_tensor = torch.zeros(amount_to_pad, text.shape[1])

            # Append the pad_tensor to the example_text tensor.
            # Shape of padded_example_text: (padded_length, embeding_dim)
            # top part is the original text numpy,
            # and the bottom part is the 0 padded tensors

            # text from the batch is a np array, but cat requires the argument to be the same type
            # turn the text into a torch.FloatTenser, which is the same type as pad_tensor
            text = torch.Tensor(text)
            padded_example_text = torch.cat((text, pad_tensor), dim=0)

            # Add the padded example to our batch
            batch_padded_example_text.append(padded_example_text)
            batch_lengths.append(length)
            batch_labels.append(label)

        # Stack the list of LongTensors into a single LongTensor
        return (torch.stack(batch_padded_example_text),
                torch.LongTensor(batch_lengths),
                torch.LongTensor(batch_labels))
