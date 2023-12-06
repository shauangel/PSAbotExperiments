from TextAnalyze import TextAnalyze
from OuterSearch import outer_search
from stack_overflow_parser import StackData
import json
import logging
import config

q_type = "CoreLanguage"

# Logging config
logging.basicConfig(level=config.LOG_MODE,
                    filename="logs/stack_data_" + q_type + ".log",
                    filemode='w',
                    format=config.FORMAT,
                    datefmt=config.DATE_FORMAT)

if __name__ == '__main__':
    # load python.org FAQs
    with open("testQ.json", "r", encoding='utf-8') as f:
        allQ = json.load(f)
        f.close()

    # parser
    analyzer = TextAnalyze()

    for k in [q_type]:
        logging.info("<<" + k + ">>")
        for q in allQ[k]:
            # load question
            print(q)
            logging.info(q)

            # search with Google
            links = outer_search(q.split(' ') + ['python'], 10, 0)
            print(links)
            logging.info(str(links))

            # scrap stack overflow website
            parser = StackData(links)
            parser.get_results()
            parser.save_results(file='stack_data/' + k)
            print("-" * 20)
            logging.info("-" * 20)
