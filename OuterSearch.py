#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 00:31:47 2021

@author: linxiangling
"""
from googlesearch import search


# 輸入參數關鍵字string array、一次需回傳筆數、第幾頁
def outer_search(keywords, result_num, page_num):

    separator = " "
    # to search
    query = separator.join(keywords) + " site:stackoverflow.com"
    # Test
    # for i in search(query, tld = "com", num = resultNum, start = resultNum * pageNum,stop = resultNum, pause = 0.1):
    #    print(i)
    return [i for i in search(query, tld="com", num=result_num, start=result_num * page_num, stop=result_num, pause=0.1)]
    
# pause (float) – Lapse to wait between HTTP requests. A lapse too long will make the search slow, but a lapse too short may cause Google to block your IP. Your mileage may vary!


if __name__ == "__main__":
    # Test
    result = outer_search(['flask', 'CORS', 'error'], 5, 0)
    
    for i in result:
        print(i)
    
