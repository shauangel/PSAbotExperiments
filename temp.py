import json
import os
import _db
from datetime import datetime


def get_filename(q_type):
    with open("testQ.json", "r", encoding='utf-8') as f:
        q = json.load(f)
        q = q[q_type]
    filelist = []
    for i in os.listdir('stack_data'):
        if q_type in i:
            filelist.append(i)
    return q, sorted(filelist)


# 將測試資料加入資料庫（模擬實際資料庫運作情形）
if __name__ == "__main__":
    # FAQ types
    cat = ['GeneralQuestions', 'CoreLanguage', 'NumbersAndstrings',
           'Performance', 'Sequences_TuplesLists', 'Objects', 'Modules']

    for c in cat:
        q, file = get_filename(c)
        # _db.TEST_DATA.delete_many({})
        # _db.TEST_OUTER_DATA.delete_many({})

        for i in range(len(q)):
            f = {"question": q[i]}
            _db.TEST_DATA.update_one(f, {"$set": {'category': c} })
            print(_db.TEST_DATA.find_one(f))
            """
            with open("stack_data/" + file[i], 'r', encoding='utf-8') as f:
                data = json.load(f)
                f.close()
            print(q[i])
            if _db.TEST_DATA.find_one({"question": q[i]}):
                print("question exists")
            else:
                posts_id = [i['question']['id'] for i in data]
                _db.TEST_DATA.insert_one({"question": q[i],
                                          "posts": posts_id})
            for d in data:
                # 檢查資料庫是否有該筆貼文
                if _db.TEST_OUTER_DATA.find_one({"question.id": d['question']['id']}):
                    print("post id: " + str(d['question']['id']) + " already exists")
                # 沒有貼文就新增
                else:
                    d['create'] = datetime.now()
                    _db.TEST_OUTER_DATA.insert_one(d)
            """
