import pandas as pd
import csv
from nltk.tokenize import WordPunctTokenizer, word_tokenize
import string
import numpy as np
import random
from _core.feautre_creation import *
from sklearn.utils import shuffle
from torchtext.vocab import GloVe
from sklearn.model_selection import train_test_split


GLOVE_PATH = "data/glove.840B.300d.txt"

def read_experiment_file(filename, pandas = False, tokenizer = None):
    """
    Static method to read file
    Input: filename
    Output: if pandas: pd DataFrame, else: list of strings
    """
    enc = 'utf-8'

    if pandas:
        data = pd.read_csv(filename, encoding=enc)
        data.drop(columns = [data.columns[0]], inplace = True)
        return data

    data = []
    with open(filename, encoding=enc) as f:
        lines = csv.reader(f)
        next(lines)
        for line in lines:
            line = line[1:]
            if tokenizer == "split":
                line[0] = line[0].split()
                line[0] = [word.lower() for word in line[0] if word not in string.punctuation]
            elif tokenizer == "NLTK":
                line[0] = word_tokenize(line[0])
                line[0] = [word.lower() for word in line[0] if word not in string.punctuation]
            data.append(line)
    f.close()
    return data
    
class dep_data():
    def __init__(self, filename, pandas = False, tokenizer = "split", r_seed = 123,
                        train_rate = 0.6, test_rate = 0.2, val_rate = 0.2, batch_size = 16):

        self.all_data = read_experiment_file(filename, pandas, tokenizer)
        self.train = None
        self.test = None
        self.val = None
        self.vocab, self.counter = bow_classifier(self.all_data)

        self.bow_train_dl = None
        self.bow_test_dl = None
        self.bow_val_dl = None

        self.tf_idf_train = None
        self.tf_idf_test = None
        self.tf_idf_val = None

        self.bow_train_glove = None
        self.bow_test_glove = None
        self.bow_val_glove = None




    def split_data(self, r_seed, train_rate = 0.6, test_rate = 0.2, pandas = False):

        if pandas:
            self.all_data = shuffle(self.all_data, random_state = r_seed)
            
            X_train, X_test, y_train, y_test = train_test_split(self.all_data["text"].to_list(), 
                                                                self.all_data["class"].to_list(),
                                                                test_size=test_rate, random_state=r_seed)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size= 1 - train_rate - test_rate, 
                                                            random_state=r_seed)
            self.train = pd.DataFrame(data = list(zip(X_train, y_train)), columns = ["text", "class"])                                             
            self.test = pd.DataFrame(data = list(zip(X_test, y_test)), columns = ["text", "class"])
            self.val = pd.DataFrame(data = list(zip(X_val, y_val)), columns = ["text", "class"])

        else:
            random.seed(r_seed)
            random.shuffle(self.all_data)
            self.train, rest = np.split(self.all_data, [int(train_rate*len(self.all_data))])
            self.train, rest = list(self.train), list(rest)
            self.test, self.val = np.split(rest, [int(train_rate*len(rest))])
            self.test, self.val = list(self.test), list(self.val)


    def prep_bow_dataloaders(self, batch_size):

        self.bow_train_dl = data_loader_bow(self.train, self.vocab, batch_size, shuffle = False)
        self.bow_test_dl = data_loader_bow(self.test, self.vocab, batch_size, shuffle = False)
        self.bow_val_dl = data_loader_bow(self.val, self.vocab, batch_size, shuffle = False)

    def split_label_text(self, data):

        labels = []
        texts = []
        for text, label in data:
            labels.append(label)
            texts.append(text)

        return labels, texts

    def load_glove_embeddings(self, batch_size = 32):
        
        word2idx, idx2word = get_word2idx_idx2word(self.all_data)
        self.vocab_size = len(word2idx)
        glove_embeddings = get_embedding_matrix(GLOVE_PATH, word2idx)
        
        self.bow_train_glove = data_loader_bow_glove([self.train, glove_embeddings], batch_size, shuffle = False)
        self.bow_test_glove = data_loader_bow_glove([self.test, glove_embeddings],batch_size, shuffle = False)
        self.bow_val_glove = data_loader_bow_glove([self.val, glove_embeddings], batch_size,shuffle = False)





    