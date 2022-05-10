import pandas as pd
import csv
from nltk.tokenize import WordPunctTokenizer, word_tokenize
import string

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
        return data, len(data)
    