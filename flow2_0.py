import json
import os


def get_filename():
    with open("testQ.json", "r", encoding='utf-8') as f:
        q = json.load(f)
        q = q['CoreLanguage']
    filelist = []
    for i in os.listdir('stack_data'):
        if "CoreLanguage" in i:
            filelist.append(i)
    return q, sorted(filelist)


if __name__ == "__main__":
    titles, data = get_filename()
    for t in titles:
        print(t)
