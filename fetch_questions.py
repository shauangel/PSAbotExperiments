from bs4 import BeautifulSoup
import requests
import json

if __name__ == "__main__":
    website = "https://docs.python.org/3/faq/programming.html"
    webpage = requests.get(website)
    soup = BeautifulSoup(webpage.text, 'html.parser')
    content = soup.find(id="contents")
    li = content.find("ul", "simple").find("li").find("ul").find_all("li", recursive=False)
    a = {}
    for i in li:
        category = i.find("p").find("a").text
        questions = i.find("ul").find_all("a")
        a[category] = [q.text for q in questions]

    for k in a.keys():
        print(k)
        print(a[k])
        print("-"*20)

    with open("testQ.json", "w", encoding="utf-8") as f:
        json.dump(a, f)
