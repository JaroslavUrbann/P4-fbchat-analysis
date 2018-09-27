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
    texts = []
    users = []
    times = []
    first_time = soup.find("div", {"class": "pam _3-95 _2pi0 _2lej uiBoxWhite noborder"})
    first_time.extract()
    all = soup.findAll("div", {"class": "pam _3-95 _2pi0 _2lej uiBoxWhite noborder"})
    for a in all:
        tx = a.findAll('div', {'class': '_3-96 _2let'})
        u = a.findAll('div', {'class': '_3-96 _2pio _2lek _2lel'})
        t = a.findAll('div', {'class': '_3-94 _2lem'})
        if u:
            texts.append(tx[0].div.findAll("div")[1].text)
            users.append(u[0].text.split()[-1] + " " + u[0].text.split()[0])
            times.append(date(t[0].text))
        else:
            texts.append(tx[0].div.findAll("div")[1].text)
            users.append("Vrba Jirka")
            times.append(date(t[0].text))

    print(len(texts))
    print(len(users))
    print(len(times))
    # tx = soup.findAll('div', {'class': '_3-96 _2let'})
    # texts = []
    # for i in tx:
    #     texts.append(i.div.findAll("div")[1].text)
    # print "texts done"
    # print len(tx)
    #
    # users = []
    # u = soup.findAll('div', {'class': '_3-96 _2pio _2lek _2lel'})
    # for h in u:
    #     x = h.text.split()
    #     m = x[-1] + " " + x[0]
    #     if m:
    #         users.append(m)
    # print "users done"
    # print len(u)
    #
    # first_time = soup.find("div", {"class": "_3-94 _2lem"})
    # first_time.extract()
    #
    # times = []
    # t = soup.findAll('div', {'class': '_3-94 _2lem'})
    # for h in t:
    #     if t != "":
    #         times.append(date(h.text))
    # print "times done"
    # print len(t)

    master = [{'sndr': users[i],
               'time': times[i],
               'text': texts[i]}
              for i in range(len(users))]

    with open('./input/group_chat.pkl', 'w') as f:
        dump(master, f)

    return master


if __name__ == '__main__':

    thread_parse("../message.html")
