import re
import json
import sys

input = open(sys.argv[2], "r")
file_data = input.read()
allfiles = file_data.splitlines()
input.close()


def datacleaning(text):
    text = re.sub('[^a-z\s]+', ' ', text)
    text = re.sub('(\s+)', ' ', text)
    text = text.replace('\n', '')
    text = text.replace('\t', '')
    return text


def pre_processing(text, currentfeats):
    cleanedtext = datacleaning(text)

    stopwords = {'it', 'hers', 'between', 'yourself', 'but', 'the', 'no', 'when', 'at', 'any', 'who', 'as', 'from',
                 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an',
                 'be', 'some', 'for', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'him',
                 'each', 'the', 'themselves', 'until', 'below', 'are', 'his', 'through', 'don', 'nor', 'me', 'were',
                 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'up', 'to', 'ours', 'before',
                 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that',
                 'because', 'what', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has',
                 'just', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my',
                 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than', 'hotel', 'stay', 'i',
                 'we', 'these', 'your', 'while', 'above', 'both', 'where', 'too', 'only', 'had', 'she', 'all', 'do',
                 'its', 'yours', 'such', 'chicago', 'day', 'ourselves', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k',
                 'l', 'm', 'n', 'o', 'p', 'q', 'r', 'u', 'v', 'w', 'x', 'y', 'z', '1', '2', '3', '4', '5', '6', '7',
                 '8', '9', '0'}

    final = ""

    for word in cleanedtext.split():
        if word in stopwords:
            continue
        elif not(word in stopwords):
            final = final + " " + word
            if word in currentfeats:
                currentfeats[word] = currentfeats[word] + 1
            elif not(word in currentfeats):
                currentfeats[word] = 1
    return currentfeats


def classify():
    fout, model = open('percepoutput.txt', 'w'), open(sys.argv[1]).read()
    vanillaoravgmodel = json.loads(model)
    for f in allfiles:
        words = f.split(" ")
        sentence = ""
        for i in range(1, len(words)):
            sentence = sentence + words[i] + " "
        features = pre_processing(sentence, {})
        posnegfunc, trudecfunc = vanillaoravgmodel['posneg']['<bias>'], vanillaoravgmodel['trudec']['<bias>']
        for word in features:
            if word in vanillaoravgmodel['posneg']:
                posnegfunc = posnegfunc + vanillaoravgmodel['posneg'][word] * features[word]
            if word in vanillaoravgmodel['trudec']:
                trudecfunc = trudecfunc + vanillaoravgmodel['trudec'][word] * features[word]

        output = words[0] + " "
        if trudecfunc >= 0:
            output = output + "True"
        else:
            output = output + "Fake"
        output = output + " "
        if posnegfunc >= 0:
            output = output + "Pos"
        else:
            output = output + "Neg"
        output = output + "\n"
        fout.write(output)
    fout.close()


classify()
