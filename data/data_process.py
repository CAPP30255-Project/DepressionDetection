import pandas as pd
import csv
from nltk.tokenize import WordPunctTokenizer
import spacy

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
                    line[0] = [word.lower() for word in line[0]]
                elif tokenizer == "NLTK":
                    tk_object = WordPunctTokenizer()
                    line[0] = tk_object.tokenize(line[0])
                    line[0] = [word.lower() for word in line[0]]
                elif tokenizer == "spacy":
                    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])
                    line[0] = nlp(line[0])
                    line[0] = [word.lower() for word in line[0]]
                data.append(line)
        f.close()
        return data, len(data)
    