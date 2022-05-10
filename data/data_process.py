import pandas as pd
import csv
from nltk.tokenize import WordPunctTokenizer, word_tokenize
import string
import numpy as np
import random
from _core.feautre_creation import *

def read_experiment_file(filename, pandas = False, tokenizer = None):
        """
        Static method to read file
        Input: filename
        Output: if pandas: pd DataFrame, else: list of strings
        """
        enc = 'utf-8'

        if pandas:
            data = pd.read_csv(filename, encoding=enc)
            return data, len(data)

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

        self.bow_train_dl = None
        self.bow_test_dl = None
        self.bow_val_dl = None

    def split_data(self, r_seed, train_rate = 0.6, test_rate = 0.2):

        random.seed(r_seed)
        random.shuffle(self.all_data)
        self.train, rest = np.split(self.all_data, [int(train_rate*len(self.all_data))])
        self.train, rest = list(self.train), list(rest)
        self.test, self.val = np.split(rest, [int(train_rate*len(rest))])
        self.test, self.val = list(self.test), list(self.val)

    def prep_bow_dataloaders(self, batch_size):

        self.bow_train_dl = data_loader_bow(self.train, batch_size, shuffle = False)
        self.bow_test_dl = data_loader_bow(self.test, batch_size, shuffle = False)
        self.bow_val_dl = data_loader_bow(self.val, batch_size, shuffle = False)




    