from sys import argv
from bs4 import BeautifulSoup
from datetime import datetime as dt
from pickle import dump
import sys

def date(x):
    x = x.replace(",", "").encode('utf-8')
    # encode('utf-8')
    return dt.strptime(x, '%b %d %Y %I:%M%p')


def thread_parse(file_path):
    with open(file_path) as f:
        html = f.read()

    # Create a BS object
    soup = BeautifulSoup(html, 'lxml')

    t = soup.findAll('div', {'class': '_3-96 _2let'})
    texts = []
    for i in t:
        texts.append(i.div.findAll("div")[1].text)
    print texts

    users = []
    u = soup.findAll('div', {'class': '_3-96 _2pio _2lek _2lel'})
    for h in u:
        x = h.text.split()
        m = x[-1] + " " + x[0]
        users.append(m)
    print users

    first_time = soup.find("div", {"class": "_3-94 _2lem"})
    first_time.extract()

    times = []
    t = soup.findAll('div', {'class': '_3-94 _2lem'})
    for h in t:
        if t != "":
            times.append(date(h.text))
    print times

    master = [{'sndr': users[i],
               'time': times[i],
               'text': texts[i]}
              for i in range(len(texts))]

    with open('./input/group_chat.pkl', 'w') as f:
        dump(master, f)

    return master


if __name__ == '__main__':

    thread_parse("../message.html")
