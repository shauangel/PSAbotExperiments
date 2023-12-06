from collections import Counter
from TextAnalyze import TextAnalyze
import json
import logging
import config
import time
import numpy as np
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
from gensim.models.callbacks import PerplexityMetric, CoherenceMetric
from gensim.models.coherencemodel import CoherenceModel
import random

post = 7

# Logging config
logging.basicConfig(level=config.LOG_MODE,
                    filename="logs/posts/post_num"+str(post)+".log",
                    filemode='w',
                    format=config.FORMAT,
                    datefmt=config.DATE_FORMAT)


# Preparing data & Pre-process progress
def data_preprocess(stack_data):
    # question_datasets = []
    # answer_blocks_datasets = []
    q_corpus = []
    ans_corpus = []
    analyzer = TextAnalyze()
    for i in stack_data:
        print(i['question']["title"])
        # question_datasets.append(i['question']['content'])
        terms, doc = analyzer.content_pre_process(i['question']['content'])
        print("Question: " + str(terms))
        q_corpus.append(terms)

        # log information
        logging.info("-" * 10 + i['question']["title"] + "-" * 10)
        logging.info("Question: " + str(terms))

        # answer blocks
        # answer_blocks_datasets.append([ans['content'] for ans in i['answers']])
        temp = []
        for ans in i['answers']:
            terms, doc = analyzer.content_pre_process(ans['content'])
            print(terms)
            temp.append(terms)
            logging.info(str(terms))
        ans_corpus.append(temp)

    return q_corpus, ans_corpus


# Train LDA model
def train_lda_model(my_corpus, id2word):
    # evaluation loggers
    u_mass_logger = CoherenceMetric(
        corpus=my_corpus,
        dictionary=id2word,
        coherence='u_mass',
        topn=10,
        logger='shell'
    )
    perplexity_logger = PerplexityMetric(corpus=my_corpus, logger='shell')

    # LDA model settings
    lda_model = LdaModel(
        corpus=my_corpus,
        id2word=id2word,
        num_topics=config.TOPIC_NUM,
        chunksize=config.CHUNKSIZE,
        update_every=1,
        alpha=config.ALPHA,
        eta=config.ETA,
        iterations=config.ITERATION,
        per_word_topics=True,
        eval_every=1,
        passes=config.PASSES,
        callbacks=[perplexity_logger, u_mass_logger]
    )

    cm = CoherenceModel(model=lda_model, corpus=my_corpus, coherence='u_mass')
    coherence = cm.get_coherence()
    print("u_mass: " + str(coherence))
    return lda_model


def prepare_training_data(a):
    training_data = [np.concatenate(ans) for ans in a if len(ans) > 0]
    dictionary = Dictionary(training_data)
    corpus = [dictionary.doc2bow(text) for text in training_data]
    return corpus, dictionary


if __name__ == "__main__":
    # Test Question:
    # "Why am I getting an UnboundLocalError when the variable has a value?"
    # Load searched data
    """
    with open("stack_data/CoreLanguageStackData2023-03-16T13-50-43.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        f.close()
    q_data, a_data = data_preprocess(data)
    """
    with open("params_test_data.json", "r", encoding='utf-8') as f:
        # json.dump({"q_data": q_data, "a_data": a_data}, f)
        temp = json.load(f)
        q_data = temp["q_data"]
        a_data = temp["a_data"][:7]
        f.close()

    # test = [a_token for a_doc in a_data for a_list in a_doc for a_token in a_list]
    # print(len(Counter(test)))

    for i in range(10):

        logging.info("="*60)
        logging.info("Starts Training Model")
        logging.info("=" * 60)

        start_t = time.time()
        corpus, dictionary = prepare_training_data(a_data)
        model = train_lda_model(my_corpus=corpus, id2word=dictionary)
        end_t = time.time()

        logging.info("<<Training Time>>: " + str(end_t-start_t))

        for i in model.print_topics():
            print(i)

        model.save("model/post_num/model")
