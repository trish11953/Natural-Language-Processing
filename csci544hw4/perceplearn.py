import re
import json
import random
import sys


finput = open(sys.argv[1], "r")
file_data = finput.read()
lines = file_data.splitlines()
finput.close()

allfiles = []
for f in lines:
    words = f.split(" ")
    class1, class2 = words[1], words[2]
    sentence = ""
    for i in range(3, len(words)):
        sentence = sentence + words[i] + " "
    if class1 == 'Fake' and class2 == 'Pos':
        tup = (sentence, words[0], 'positive', 'deceptive')
    elif class1 == 'Fake' and class2 == 'Neg':
        tup = (sentence, words[0], 'negative', 'deceptive')
    elif class1 == 'True' and class2 == 'Pos':
        tup = (sentence, words[0], 'positive', 'truthful')
    elif class1 == 'True' and class2 == 'Neg':
        tup = (sentence, words[0], 'negative', 'truthful')
    allfiles.append(tup)


# print(allfileList[0])

def datacleaning(text):
    text = re.sub('[^a-z\s]+', ' ', text)
    text = re.sub('(\s+)', ' ', text)
    #text = text.replace('\n', '')
    text = text.replace('\t', '')
    return text


def pre_processing(text, currentfeats):
    cleanedtext = datacleaning(text)
    stopwords = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'it', 'hers', 'between', 'yourself', 'but', 'the',
                  'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an',
                  'be', 'some', 'for', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'him',
                  'each', 'the', 'themselves', 'until', 'below', 'are', 'his', 'through', 'don', 'nor', 'me', 'were',
                  'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'up', 'to', 'ours', 'before',
                  'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that',
                  'because', 'what', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has',
                  'just', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my',
                  'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than', 'hotel', 'stay', 'i',
                  'we', 'these', 'your', 'while', 'above', 'both', 'where', 'too', 'only', 'had', 'she', 'all', 'do',
                  'its', 'yours', 'such', 'chicago', 'day', 'ourselves', 'no', 'when', 'at', 'any', 'who', 'as', 'from',
                  'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 'u', 'v', 'w', 'x',
                  'y', 'z'}

    for i in cleanedtext.split():
        if i in stopwords:
            continue
        if not(i in stopwords):
            if i in currentfeats:
                currentfeats[i] = currentfeats[i] + 1
            else:
                currentfeats[i] = 1
    return currentfeats


def train_perceptron():
    posnegweights, trudecweights = {'<bias>': 0}, {'<bias>': 0}
    avgposnegweights, avgtrudecweights = {'<bias>': 0}, {'<bias>': 0}
    uavgposnegweights, uavgtrudecweights = {'<bias>': 0}, {'<bias>': 0}
    count, maxiterations, labels = 1, 50, {'positive': 1, 'negative': -1, 'truthful': 1, 'deceptive': -1}
    for i in range(maxiterations):
        random.shuffle(allfiles)
        for j in range(len(allfiles)):
            file = allfiles[j]
            features = pre_processing(file[0], {})
            scale = len(features)
            for k in features.keys():

                if not (k in posnegweights):
                    posnegweights[k], avgposnegweights[k], uavgposnegweights[k] = 0, 0, 0
                if not (k in trudecweights):
                    trudecweights[k], avgtrudecweights[k], uavgtrudecweights[k] = 0, 0, 0

            posnegfunc, trudecfunc, avgposnegfunc, avgtrudecfunc = posnegweights['<bias>'], trudecweights['<bias>'], avgposnegweights['<bias>'], avgtrudecweights['<bias>']

            for k in features:
                posnegfunc, trudecfunc = posnegfunc + (features[k] * posnegweights[k]), trudecfunc + (features[k] * trudecweights[k])
                avgposnegfunc, avgtrudecfunc = avgposnegfunc + (features[k] * avgposnegweights[k]), avgtrudecfunc + (features[k] * avgtrudecweights[k])

            if 0 >= (posnegfunc * labels[file[2]]):
                for k in features:
                    posnegweights[k], posnegweights['<bias>'] = posnegweights[k] + labels[file[2]] * features[k] * scale, \
                                                                    posnegweights['<bias>'] + labels[file[2]]
            if 0 >= (trudecfunc * labels[file[3]]):
                for k in features:
                    trudecweights[k], trudecweights['<bias>'] = trudecweights[k] + labels[file[3]] * features[k] * scale, \
                                                                    trudecweights['<bias>'] + labels[file[3]]

            if 0 >= (avgposnegfunc * labels[file[2]]):
                for k in features:
                    avgposnegweights[k], uavgposnegweights[k] = avgposnegweights[k] + (labels[file[2]] * features[k]), \
                                                                uavgposnegweights[k] + (
                                                                        labels[file[2]] * count * features[k] * scale)
                    avgposnegweights['<bias>'], uavgposnegweights['<bias>'] = avgposnegweights['<bias>'] + labels[
                        file[2]], uavgposnegweights['<bias>'] + (labels[file[2]] * count * scale)

            if 0 >= (avgtrudecfunc * labels[file[3]]):
                for k in features:
                    avgtrudecweights[k], uavgtrudecweights[k] = avgtrudecweights[k] + (labels[file[3]] * features[k]), \
                                                                uavgtrudecweights[k] + (
                                                                        labels[file[3]] * count * features[k] * scale)
                    avgtrudecweights['<bias>'], uavgtrudecweights['<bias>'] = avgtrudecweights['<bias>'] + labels[
                        file[3]], uavgtrudecweights['<bias>'] + (labels[file[3]] * count * scale)

            count = count + 1

    ctfac = (count * 1.0)
    for k in avgposnegweights:
        if k in uavgposnegweights:
            avgposnegweights[k] = avgposnegweights[k] - (uavgposnegweights[k] / ctfac)
    for k in avgtrudecweights:
        if k in uavgtrudecweights:
            avgtrudecweights[k] = avgtrudecweights[k] - (uavgtrudecweights[k] / ctfac)

    vanilla = {'posneg': posnegweights, 'trudec': trudecweights}
    fhandle = open('vanillamodel.txt', 'w')
    fhandle.write(json.dumps(vanilla, indent=4))
    fhandle.close()
    averaged = {'posneg': avgposnegweights, 'trudec': avgtrudecweights}
    fhandle = open('averagedmodel.txt', 'w')
    fhandle.write(json.dumps(averaged, indent=4))
    fhandle.close()


train_perceptron()

