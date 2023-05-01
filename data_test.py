from TextAnalyze import TextAnalyze
from collections import Counter
import json
import numpy as np
import nltk
import spacy
from gensim.models import LdaModel
from parameter_experiments import prepare_training_data


with open("params_test_data.json", "r", encoding='utf-8') as f:
    data = json.load(f)
    q_data = data['q_data']
    a_data = data['a_data']
    f.close()

with open("stack_data/CoreLanguageStackData2023-03-16T13-50-43.json", "r", encoding='utf-8') as f:
    raw_data = json.load(f)
    f.close()


if __name__ == "__main__":
    """
    a_list = []
    for i in raw_data:
        a_list.append([ans['content'] for ans in i['answers']])

    tokens = []
    for d in a_data:
        for ans in d:
            tokens += ans

    print(len(tokens))
    unique = Counter(tokens)
    print(len(unique))
    """
    q = "Why am I getting an UnboundLocalError when the variable has a value?"
    model = LdaModel.load("model/questions/1/model")
    corpus, dictionary = prepare_training_data(a_data)
