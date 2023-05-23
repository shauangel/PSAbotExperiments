import json
import os
import csv
import statistics


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
    titles, files = get_filename()
    for i in range(len(titles)):
        print(titles[i])
        with open("stack_data/" + files[i], 'r', encoding='utf-8') as f:
            data = json.load(f)
            f.close()
        scores = []
        for q in data:
            temp = []
            for ans in q['answers']:
                record = {'question_id': q['question']['id'],
                          'answer_id' : ans['id'],
                          'origin' : ans['score']}
                temp.append(ans['score'])
                scores.append(record)

            # statistical analysis
            # 3 kinds of normalization methods
            for r in scores:
                try:
                    r['min_max'] = ( r['origin'] - min(temp) ) / ( max(temp) - min(temp) )
                    r['mean'] = ( r['origin'] - statistics.mean(temp) ) / ( max(temp) - min(temp) )
                    r['z-score'] = ( r['origin'] - statistics.mean(temp) ) / statistics.stdev(temp)
                    print(r)
                except Exception:
                    r['min_max'] = 0
                    r['mean'] = 0
                    r["z-score"] = 0
                    continue

            # save as csv file
            attrs = ['question_id', 'answer_id', 'origin', 'min_max', 'mean', 'z-score']
            with open('model/questions/'+str(i+1)+"/ratings.csv", 'w', encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=attrs)
                writer.writeheader()
                writer.writerows(scores)
                csvfile.close()


