# Trisha Mandal
# Hidden Markov Models
import json
import sys
import math
from collections import defaultdict

# path = r"/Users/trishamandal/PycharmProjects/csci544hmm/it_isdt_train_tagged.txt"
path = sys.argv[1]
input = open(path, encoding='UTF-8').readlines()
s, e = "<BEGIN>", "<END>"
countoftags = {s: len(input), e: len(input)}
# print(input)
transition, emission, previoustag = defaultdict(lambda: 1), defaultdict(lambda: 1), ""
for line in input:
    previoustag = s
    for words in line.split():
        word, tag = words.rsplit("/", 1)
        if tag in countoftags:
            countoftags[tag] = countoftags[tag] + 1
        elif tag not in countoftags:
            countoftags[tag] = 1
        # emission matrix
        if word in emission:
            if tag in emission[word]:
                emission[word][tag] = emission[word][tag] + 1
            else:
                emission[word][tag] = 1
        elif word not in emission:
            emission[word] = {}
            emission[word][tag] = 1
        # transition matrix
        if previoustag in transition:
            if tag in transition[previoustag]:
                transition[previoustag][tag] = transition[previoustag][tag] + 1
            else:
                transition[previoustag][tag] = 1
        elif previoustag not in transition:
            transition[previoustag] = {}
            transition[previoustag][tag] = 1
        previoustag = tag
    # handling end cases
    if previoustag in transition:
        if e in transition[previoustag]:
            transition[previoustag][e] = transition[previoustag][e] + 1
        elif e not in transition[previoustag]:
            transition[previoustag][e] = 1
    elif previoustag not in transition:
        transition[previoustag] = {}
        transition[previoustag][e] = 1
# print(emission)
# Log Probabilities
for wor in transition:
    tagval = 0
    for tag in countoftags:
        if tag in transition[wor]:
            continue
        if tag not in transition[wor]:
            transition[wor][tag] = 0
        transition[wor][tag] = transition[wor][tag] + 1

    tot = sum(transition[wor].values())

    for tag in transition[wor]:
        t1 = transition[wor][tag] / tot
        m1 = math.log(t1)
        transition[wor][tag] = m1

for word in emission:
    for tag in emission[word]:
        t2 = emission[word][tag] / countoftags[tag]
        m2 = math.log(t2)
        emission[word][tag] = m2

ta, em, tr = " Tags for Words ", " Emission Probabilities ", " Transition Probabilities "
with open('hmmmodel.txt', 'w') as file:
    file.write(json.dumps([ta, countoftags, em, emission, tr, transition]))
