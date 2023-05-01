# basic packages
import os
import json
import logging
import config
import re
import math
import numpy as np
from collections import Counter
# Custom
from TextAnalyze import TextAnalyze
from parameter_experiments import data_preprocess, train_lda_model, prepare_training_data
# word embeddings
import tensorflow as tf
import tensorflow_hub as hub

# Custom analyzer
analyzer = TextAnalyze()
# Logging config
logging.getLogger('').handlers = []


def min_max_norm(z, l):
    return (z-min(l))/(max(l)-min(l))


def get_filename():
    with open("testQ.json", "r", encoding='utf-8') as f:
        q = json.load(f)
        q = q['CoreLanguage']
    filelist = []
    for i in os.listdir('stack_data'):
        if "CoreLanguage" in i:
            filelist.append(i)
    return q, sorted(filelist)


def save_json(filename, dumpling):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(dumpling, f)
        f.close()


if __name__ == "__main__":
    # Step 1. get user question
    # Step 2. outer search
    questions, responses = get_filename()

    for idx in range(len(questions)):
    # for idx in range(10):
        # Set loggers
        logging.basicConfig(level=config.LOG_MODE, force=True,
                            filename="logs/elda/" + str(idx+1) + "/compare.log", filemode='w',
                            format=config.FORMAT, datefmt=config.DATE_FORMAT)

        # Step 3. parse posts
        with open("stack_data/" + responses[idx], "r", encoding="utf-8") as raw_file:
            data = json.load(raw_file)
            titles = [i['question']['title'] for i in data]
            raw_file.close()

        print(questions[idx])
        logging.info("-"*20)
        logging.info(questions[idx])
        logging.info("-"*20)

        # Step 4. pre-process
        try:
            # check if clean data existed
            with open('model/questions/' + str(idx+1) + "/processed.json", 'r', encoding='utf-8') as file:
                processed_data = json.load(file)
                q_data = processed_data['q_data']
                a_data = processed_data['a_data']
                file.close()
            print("exist")
            logging.info("exist")
        except Exception:
            q_data, a_data = data_preprocess(data)
            print("processed")
            logging.info("processed")
            with open('model/questions/' + str(idx+1) + "/processed.json", 'w', encoding='utf-8') as file:
                json.dump({"q_data": q_data, "a_data": a_data}, file)
                file.close()
            print("saved")
            logging.info("saved")

    # Step 5. train lda model
        # model = TextAnalyze.train_lda_model(my_corpus=corpus, id2word=dictionary)
        corpus, dictionary = prepare_training_data(a_data)
        training_data = [np.concatenate(ans) for ans in a_data if len(ans) > 0]
        model = TextAnalyze.train_elda_model(training_data, 5, 5)
        model.save("model/elda/" + str(idx+1) + "/model")

        #for i in model.print_topics():
        topics = []
        for i in model.print_topics(config.TOPIC_TERM_NUM):
            topics.append(i)
            # print(i)

        # Step 6. apply word embedding
        embed = hub.KerasLayer("embeds/Wiki-words-250_2",
                               input_shape=[],
                               dtype=tf.string,
                               trainable=True,
                               name="Word_Embedding_Layer")
        word_pattern = r'\*"(.*?)"'
        # topics = [model.print_topics(idx, config.TOPIC_TERM_NUM) for idx in range(config.TOPIC_NUM)]
        user_q_vector = embed([questions[idx]])
        q_title_vectors = embed(titles)
        a_vector = [embed([' '.join(ans) for ans in q]) for q in a_data]
        # t_vector = [embed([" ".join(re.findall(word_pattern, t))]) for t in topics] # lda
        t_vector = [embed([" ".join(re.findall(word_pattern, t[1]))]) for t in topics]

    # Step 7. calculate similarity between questions
        question_sim = [tf.losses.CosineSimilarity()(user_q_vector[0], t_vec).numpy() for t_vec in q_title_vectors]
        abs_question_sim = [abs(1+sim) for sim in question_sim]
        # print(question_sim)
        print(abs_question_sim)
        logging.info("Question Similarity: " + str(abs_question_sim))

    # Step 8. predict the topic distribution of user question & blocks
        user_q_topic_dist = model[dictionary.doc2bow(analyzer.content_pre_process(questions[idx])[0])][0]
        user_q_topic_dist = {t[0]: t[1] for t in user_q_topic_dist}
        ans_blocks_dist = []
        for ans in a_data:
            a_corpus = [dictionary.doc2bow(terms) for terms in ans]
            ans_blocks_dist.append([model[block][0] for block in a_corpus])

    # Step 9. calculate the probability of the user question and block belongs to same topic
        block_prob = []
        block_terms = []
        block_count = 0
        for d in range(len(data)):
            temp = []
            ans_score = [a['score'] for a in data[d]['answers']]
            for a_idx in range(len(data[d]['answers'])):
                block_count += len(data[d]['answers'])
                # print(data[d]['answers'][a_idx]['id'])
                # log because the result is too small
                prob = math.log(sum([user_q_topic_dist[t[0]]*t[1] for t in ans_blocks_dist[d][a_idx]])/3)
                if ans_score[a_idx] >= 0 and abs_question_sim[d] >= 0.1:
                    block_prob.append({"id": data[d]['answers'][a_idx]['id'],
                                       "prob": prob,
                                       "q_sim": str(abs_question_sim[d]),
                                       "b_score": str(prob*abs_question_sim[d]),
                                       "so_score": str(ans_score[a_idx])})
                # prepare terms for keyword extraction
                temp.append(Counter(a_data[d][a_idx]))
            block_terms.append(temp)
            # print("-"*20)
        save_json("model/elda/" + str(idx+1) + "/blocks_ranking.json", block_prob)
        print("total block count: " + str(block_count))
        print("reduced block: " + str(len(block_prob)))
    # Step 10. Rank block
        # print(sorted(block_prob, key=lambda x: x['b_score']))
        for block in sorted(block_prob, key=lambda x: x['b_score']):
            logging.info(str(block["id"]) + ": " + str(block['b_score']))

    # Step 11. Extract Keywords
        keywords = {}
        for b in ans_blocks_dist:
            for dist in b:
                terms = [i[0] for i in block_terms[ans_blocks_dist.index(b)][b.index(dist)].most_common(15)]
                term_vectors = embed(terms)
                max_dist = max(dist, key=lambda x: x[1])
                term_sims = [tf.losses.CosineSimilarity()(t_vector[max_dist[0]], t_vec).numpy()
                             for t_vec in term_vectors]
                norm_sims = dict(sorted({terms[k]: term_sims[k]
                                         for k in range(len(terms))}.items(),
                                        key=lambda item: item[1]))
                # the maximum keyword amount of one block is 5
                add_count = 0
                for k, v in norm_sims.items():
                    if add_count >= 5:
                        break
                    if k in keywords.keys():
                        if keywords[k][0] >= 3:
                            continue
                        else:
                            keywords[k][0] += 1
                            if keywords[k][1] > v:
                                keywords[k][1] = v
                    else:
                        keywords[k] = [1, v]
        # sort by count
        keywords = dict(sorted(keywords.items(), key=lambda x: x[1][1]))
        # sort by score
        print(keywords)
        logging.info("Recommended Keywords")
        for k, v in keywords.items():
            logging.info(k + " score: " + str(v[1]))
        logging.info("="*60)
