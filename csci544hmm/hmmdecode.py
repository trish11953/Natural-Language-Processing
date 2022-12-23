# Trisha Mandal
# Hidden Markov Models
import sys
import json

s, e, negmax= "<BEGIN>", "<END>", -sys.maxsize
def dataread(file, path):
    lst = json.loads(open(file, encoding='UTF-8').read())
    ta, countoftags, em, emissionprobs, tr, transitionprobs = lst[0], lst[1], lst[2], lst[3], lst[4], lst[5]
    input = open(path, encoding='UTF-8').readlines()
    return input, countoftags, emissionprobs, transitionprobs


def Viterbi(line, countoftags, emissionprobs, transitionprobs):
    viterbi_prob, viterbi_bp, words, wordsleng, emmprob = [{}], [{}], line.split(), len(line.split()), 0
    emmkeys = emissionprobs.keys()
    if words[0] not in emmkeys:
        wordtag = countoftags
    elif words[0] in emmkeys:
        wordtag = emissionprobs[words[0]]

    for tag in wordtag.keys():
        if tag == s and words[0] not in emmkeys:
            continue
        elif tag != s and words[0] in emmkeys:
            emmprob = emissionprobs[words[0]][tag]

        viterbi_prob[0][tag] = {}
        viterbi_bp[0][tag] = {}
        viterbi_bp[0][tag]['back_pointer'] = s
        viterbi_prob[0][tag]['max_probability'] = transitionprobs[s][tag] + emmprob

    for i in range(1, wordsleng):
        word = words[i]
        viterbi_prob.append({})
        viterbi_bp.append({})
        if word in emmkeys:
            wordtag = emissionprobs[word]
        elif word not in emmkeys:
            wordtag = countoftags
        for tag in wordtag.keys():
            if tag != s:
                if tag != e:
                    if word in emmkeys:
                        emmprob = emissionprobs[word][tag]
            else:
                continue
            if word not in emmkeys:
                emmprob = 0
            maxprobability = {'p': negmax, 'bp': ''}
            probkeys = viterbi_prob[i - 1].keys()
            for previous in probkeys:
                if previous == s or previous == e:
                    continue
                p1 = viterbi_prob[i - 1][previous]['max_probability']
                p2 = transitionprobs[previous][tag] + emmprob
                if maxprobability['p'] < p1 + p2:
                    maxprobability['bp'] = previous
                    maxprobability['p'] = p1 + p2

            viterbi_prob[i][tag] = {}
            viterbi_bp[i][tag] = {}
            viterbi_prob[i][tag]['max_probability'] = maxprobability['p']
            viterbi_bp[i][tag]['back_pointer'] = maxprobability['bp']

    wordtag = viterbi_prob[-1].keys()
    viterbi_prob.append({})
    viterbi_bp.append({})
    maxprobability = {'p': negmax, 'bp': ''}
    for tag in wordtag:
        if tag != e:
            pa1 = viterbi_prob[wordsleng - 1][tag]['max_probability']
            pa2 = transitionprobs[tag][e]
            proba = pa1 + pa2
        if proba > maxprobability['p']:
            maxprobability['p'], maxprobability['bp'] = proba, tag

    viterbi_prob[-1][e] = {}
    viterbi_bp[-1][e] = {}
    viterbi_prob[-1][e]['max_probability'] = maxprobability['p']
    viterbi_bp[-1][e]['back_pointer'] = maxprobability['bp']

    tag, POStagging, le = e, " ", wordsleng - 1
    for i in range(le, -1, -1):
        pt1 = words[i] + "/"
        pt2 = viterbi_bp[i + 1][tag]['back_pointer']
        pt3 = " " + POStagging
        POStagging = pt1 + pt2 + pt3
        tag = viterbi_bp[i + 1][tag]['back_pointer']
    return POStagging


def filewrite(prediction):
    writeFile = open('hmmoutput.txt', mode='w', encoding='UTF-8')
    for s in prediction:
        writeFile.write(s + "\n")


input, countoftags, emissionprobs, transitionprobs = dataread("hmmmodel.txt", sys.argv[1])
prediction = [Viterbi(line, countoftags, emissionprobs, transitionprobs) for line in input]
filewrite(prediction)
