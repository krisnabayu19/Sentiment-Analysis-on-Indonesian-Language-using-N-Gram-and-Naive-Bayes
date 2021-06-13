import numpy as np
import texttable as tt
import csv
from pymongo import MongoClient
import pymongo
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re
import string
import emoji
import functools
import operator

with open('data-training.csv', newline='') as csvfile:
    TRAINING_DATA = list(csv.reader(csvfile))


class NaiveBayes:

    def __init__(self, data, vocab):
        self._displayHelper = DisplayHelper(data,vocab)
        self._vocab = vocab

        # LabelArray
        labelArray = []
        for i in range(1, len(data)):
            labelArray.append(data[i][1])
        self._label = np.array(labelArray)

        # docArray
        docArray = []
        for i in range(1, len(data)):
            docArray.append(self.map_doc_to_vocab(data[i][0].split()))
        self._doc = np.array(docArray)
        self.calc_prior_prob().calc_cond_probs()


    def calc_prior_prob(self):
        sum = 0

        # Laplacian Smoothing
        for i in self._label:
            if ("-".__eq__(i)) : sum += 1;
        self._priorProb = sum / len(self._label)
        self._displayHelper.set_priors(sum, len(self._label))
        return self

    def calc_cond_probs(self):
        pProbNum = np.ones(len(self._doc[0])); nProbNum = np.ones(len(self._doc[0]))
        pProbDenom = len(self._vocab); nProbDenom = len(self._vocab)
        for i in range(len(self._doc)):
            if "-".__eq__(self._label[i]):
                nProbNum += self._doc[i]
                nProbDenom += sum(self._doc[i])
            else:
                pProbNum += self._doc[i]
                pProbDenom += sum(self._doc[i])
        self._negProb = np.log(nProbNum / nProbDenom)
        self._posProb = np.log(pProbNum / pProbDenom)
        self._displayHelper.display_calc_cond_probs(nProbNum, pProbNum, nProbDenom, pProbDenom)
        return self

    # Function classify label
    def classify(self, doc):
        sentiment = "-"
        nLogSums = doc @ self._negProb + np.log(self._priorProb)
        pLogSums = doc @ self._posProb + np.log(1.0 - self._priorProb)
        self._displayHelper.display_classify(doc, pLogSums, nLogSums)
        if pLogSums > nLogSums:
            sentiment = "Positive"

        if pLogSums < nLogSums:
            sentiment = "Negative"

        if pLogSums == nLogSums:
            sentiment = "Neutral"
        return "Text Classified as ("+ sentiment+ ") label"

    def map_doc_to_vocab(self, doc):
        mappedDoc = [0] * len(self._vocab)
        for d in doc:
            counter = 0
            for v in self._vocab:
                if (d.__eq__(v)): mappedDoc[counter] += 1
                counter += 1
        return mappedDoc



# Class display
class DisplayHelper:
    def __init__(self, data, vocab):
        self._vocab = vocab
        self.print_training_data(data)

    # print training data table
    def print_training_data(self, data):
        table = tt.Texttable()
        table.header(data[0])
        for i in range(1, data.__len__()): table.add_row(data[i])

        # Print table data training
        # print(table.draw().__str__())

    def set_priors(self, priorNum, priorDenom):
        self._priorNum = priorNum
        self._priorDenom = priorDenom


    def display_classify(self, sentiment, posProb, negProb):

        # N-Gram Feature
        # Happy Label
        temp = "N-Gram Data Training Happy Emotion Label = ("+ \
               (self._priorDenom - self._priorNum).__str__()+ "/"+ self._priorDenom.__str__()+ ")"
        for i in range(0, len(sentiment)):
            if sentiment[i] == 1 :
                temp = temp
        print(temp)


        # Unhappy Label
        temp = "N-Gram Data Training Unhappy Emotion Label = ("+ self._priorNum.__str__()\
                                    + "/"+ self._priorDenom.__str__()+ ")"
        for i in range(0, len(sentiment)):
            if sentiment[i] == 1:
                temp = temp
        print(temp)

        # Probabilitas sentiment Naive Bayes Method
        print("Probabilitas of (Happy Emotion) = ", np.exp(posProb))
        print("Probabilitas of (Unhappy Emotion) = ", np.exp(negProb))


    # function to display calculation word probability
    def display_calc_cond_probs(self, nProbNum, pProbNum, nProbDenom, pProbDenom):

        # Array Calculation Unhappy Emotion
        nProb = []
        nProb.append("P(w|Unhappy Emotion)")
        for i in range (0, len(self._vocab)):
            nProb.append((int)(nProbNum[i]).__str__()+"/"+nProbDenom.__str__())

        # Array Calculation Happy Emotion
        pProb = []
        pProb.append("P(w|Happy Emotion)")
        for i in range (0, len(self._vocab)):
             pProb.append((int)(pProbNum[i]).__str__()+ "/" + pProbDenom.__str__())

        tempVocab = []
        tempVocab.append("")
        for i in range(0, len(self._vocab)) : tempVocab.append(self._vocab[i])

        # Limit row table
        table = tt.Texttable(1000000)
        table.header(tempVocab)
        table.add_row(pProb)
        table.add_row(nProb)

        # print table calculation Frequency data training
        print(table.draw().__str__())

        self._nProbNum = nProbNum; self._pProbNum = pProbNum
        self._nProbDenom = nProbDenom; self._pProbDenom = pProbDenom


# Function to Clean
def clean_tweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|(_[A-Za-z0-9]+)|(\w+:\/\/\S+)|(\d+)|"
                           "(\s([@#][\w_-]+)|(#\\S+))|((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))", " ", tweet).replace(",","").replace(".","").replace("?","").replace("!","").replace("/","").replace("&","").replace(":","").replace("_","").replace("@","").replace("#","").split())

# Stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# array stopword list
stopwords = []

# array stop words list text input
stopwords_list = []


if __name__ == '__main__':

    # Function input command line
    def handle_command_line(nb):

        # open stop word list and append to array
        with open('list-stopwords.csv', 'r') as file:
            stopwords = []
            for line in file:
                clear_line = line.replace("\n", '').strip()
                stopwords.append(clear_line)
            print("Stop Words List : ",stopwords)

        flag = True
        while (flag):

            print("===================================================================================================")
            dataInput = input("> Input Text : ")
            arrrayInput = []
            arrrayInput.append(dataInput)
            data = arrrayInput

            for index, entryData in enumerate(data):
                text = entryData
                final_words = []
                after_stopwords = []

                print("Text Tweet :", text)

                # cleaning process
                gas = text.strip()
                blob = clean_tweet(gas)
                print("Text After Cleaning Process :", blob)

                # split text and emoticon
                em_split_emoji = emoji.get_emoji_regexp().split(blob)
                em_split_whitespace = [substr.split() for substr in em_split_emoji]
                em_split = functools.reduce(operator.concat, em_split_whitespace)
                strSplit = ' '.join(em_split)
                # print("Text Split Emoticon and Text :", strSplit)

                # lowering case process
                lower_case = strSplit.lower()
                print("Text After Lower Case Process :", lower_case)

                # convert emoticon process
                punctuationText = lower_case.translate(str.maketrans('', '', string.punctuation))
                tokenized_words = punctuationText.split()
                for tokenized_words_emoticon in tokenized_words:
                    arrayTokenizingEmoticon = []
                    arrayTokenizingEmoticon.append(tokenized_words_emoticon)

                    with open('list-konversi-emoticon.csv', 'r', encoding='utf-8') as fileEmoticon:
                        for lineEmoticon in fileEmoticon:
                            clear_line_emoticon = lineEmoticon.replace("\n", '').strip()
                            emoticon, convert = clear_line_emoticon.split(',')
                            if emoticon in arrayTokenizingEmoticon:
                                # emoticon_detection.append(emoticon)
                                tokenized_words.append(convert)
                                print("Emoticon Convert Process :", emoticon, "to", convert)
                strEmoticonConvert = ' '.join(tokenized_words)
                # print("Text Emoticon Convert :", strEmoticonConvert)

                # stemming process
                hasilStemmer = stemmer.stem(strEmoticonConvert)
                print("Text After Stemming Process :", hasilStemmer)

                # stop words process
                punctuationText2 = hasilStemmer.translate(str.maketrans('', '', string.punctuation))
                tokenized_words2 = punctuationText2.split()
                for tokenized_words3 in tokenized_words2:
                    if tokenized_words3 not in stopwords:
                        stopwords_list.append(stopwords)
                        after_stopwords.append(tokenized_words3)

                strTextFix = ' '.join(after_stopwords)
                print("Text After Stop Words Process : ", strTextFix)
                entryClean = strTextFix
                
                if (entryClean!= "exit"):
                    print(nb.classify(np.array(nb.map_doc_to_vocab(entryClean.lower().split()))))
                else:
                    flag = False

    # Prepare data training to lower case
    def prepare_data () :
        data = []
        for i in range (0, len(TRAINING_DATA)):
            data.append([TRAINING_DATA[i][0].lower(), TRAINING_DATA [i][1]])
        return data

    # Split data training beetwen text and label
    def prepare_vocab(data) :
        vocabSet = set([])
        for i in range(1, len(data)):
            for word in data [i][0].split(): vocabSet.add(word)
        return list(vocabSet)

    # Calling Function to Run
    data = prepare_data()
    handle_command_line(NaiveBayes(data, prepare_vocab(data)))