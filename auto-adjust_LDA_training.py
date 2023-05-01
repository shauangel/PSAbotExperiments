# !-- Created time: 2023/04/24 Mon. --!
# !-- Purpose: A LDA training Process that can automatically adjust parameters --!
# !-- Method: We considered two evaluation metrics, Perplexity and Coherence, to decide the best model. --!
# Custom models
from TextAnalyze import TextAnalyze
# Tool models
import config
import re
import json
import numpy as np
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # hide tensorflow warnings


def get_log(file_name):
    per = []
    coh = []
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            if "Perplexity" in line:
                p = re.compile("Perplexity estimate: (\d+\.\d+)")
                match = p.findall(line)
                per.append(match[0])
            if "Coherence" in line:
                u_mass = re.compile("Coherence estimate: (-*\d+\.\d+)")
                match = u_mass.findall(line)
                coh.append(match[0])

    print("Final Perplexity: " + str(per[len(per) - 1]))
    print("Final Coherence: " + str(coh[len(coh) - 1]))
    return {"per": float(per[len(per) - 1]), "coh": float(coh[len(coh) - 1])}


if __name__ == "__main__":
    analyzer = TextAnalyze()
    # load experiment data 1~3 (pre-processed)
    for num in range(1, 11):
        evaluation = []
        f_name = "model/questions/" + str(num) + "/processed.json"
        print(f_name)
        with open(f_name, 'r', encoding='utf-8') as f:
            a_data = json.load(f)['a_data']
            f.close()

    # train lda
        model_lists = []
        training_data = [np.concatenate(ans) for ans in a_data if len(ans) > 0]
        for t_num in config.TOPIC_NUM_LIST:
            # switch logger to each model's coherence logger
            log_name = "auto-adjust_models/logs/model_" + str(num) + "_t" + str(t_num) + ".log"
            logging.basicConfig(level=config.LOG_MODE,
                                filename=log_name,
                                filemode='w',
                                format=config.FORMAT,
                                datefmt=config.DATE_FORMAT,
                                force=True)
            logging.info("---Start Analyzing---")
            logging.info("---Train for " + str(t_num) + " Topics---")
            print("---Train for " + str(t_num) + " Topics---")
            model= analyzer.train_lda_model(training_data, t_num)
            model_lists.append(model)
            model_name = "auto-adjust_models/models/" + str(num) + "/model_t" + str(t_num)
            model.save(model_name)

            # record evaluation scores
            evaluation.append(get_log(log_name))

        # decide best model
        avg_per = sum([e['per'] for e in evaluation])/len(config.TOPIC_NUM_LIST)
        avg_coh = sum([e['coh'] for e in evaluation])/len(config.TOPIC_NUM_LIST)
        valid_per = [e['per'] < avg_per for e in evaluation]
        valid_coh = [e['coh'] < avg_coh for e in evaluation]
        valid_model = [valid_coh[i] & valid_per[i] for i in range(len(valid_per))]
        print(evaluation)
        print(valid_model)
















