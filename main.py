from TextAnalyze import TextAnalyze
from OuterSearch import outer_search
from stack_overflow_parser import StackData
import json
import logging
import config

# Logging config
logging.basicConfig(level=config.LOG_MODE,
                    filename="logs/stack_data1.log",
                    filemode='w',
                    format=config.FORMAT,
                    datefmt=config.DATE_FORMAT)

if __name__ == '__main__':
    with open("testQ.json", "r", encoding='utf-8') as f:
        allQ = json.load(f)
        f.close()

    # Parse data
    analyzer = TextAnalyze()
    # for k in list(allQ.keys()):
    for k in ['Objects', 'Modules']:
        logging.info("<<" + k + ">>")
        for q in allQ[k]:
            print(q)
            logging.info(q)
            keywords = ['python'] + analyzer.content_pre_process(q)[0]
            print(keywords)
            logging.info(str(keywords))
            links = outer_search(keywords, 10, 0)
            print(links)
            logging.info(str(links))
            parser = StackData(links)
            parser.get_results()
            parser.save_results(file='stack_data/' + k)
            print("-" * 20)
            logging.info("-" * 20)

    """
        for q in core_q[16:]:
        print(q)
        keywords = analyzer.content_pre_process(q)[0]
        print(keywords)
        links = outer_search(keywords, 10, 0)
        print(links)
        parser = StackData(links)
        parser.get_results()
        parser.save_results()
        print("-" * 20)

    # analyze
    with open("stack_data/StackData2023-03-14T01-54-13.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        f.close()
    for d in data:
        print("="*20)
        print(d['question']['id'])
        print("="*20)
        for ans in d['answers']:
            print(ans['id'])
            print(ans['content'])
            print('-'*20)
    # mylist = [f for f in glob.glob("*.json")]
    # print(mylist)
    """
