import os
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter

str_red = WordNetLemmatizer()
path = "data/dictionary_processing/"
hamPath = "data/nonspam-train/"
spamPath = "data/spam-train/"
hamPath_tst = "data/nonspam-test/"
spamPath_tst = "data/spam-test/"


# Create Dictionary of words with count (Lexicon) in all training, testing data


def create_dict():
    dictionary = []
    files = os.listdir(path)

    for file in files:
        with open(path + file, 'r') as f:
            contents = f.read()
            words = word_tokenize(contents)
            dictionary += words

    dictionary = [str_red.lemmatize(i) for i in dictionary]
    dictionary = Counter(dictionary)
    lexicon = []
    for w in dictionary:
        if dictionary[w] > 14:
            lexicon.append(w)
    print(len(lexicon))
    return lexicon


# Create featureset for training data


def create_featureset(dictCount):
    spam_f = os.listdir(spamPath)
    ham_f = os.listdir(hamPath)
    spam_f_tst = os.listdir(spamPath_tst)
    ham_f_tst = os.listdir(hamPath_tst)
    featureSet_train = []
    featureSet_test = []

    # Creating features for Training - spam
    for file in spam_f:
        with open(spamPath + file, 'r') as f:
            contents = f.read()
            words = word_tokenize(contents)
            words = [str_red.lemmatize(i) for i in words]
            ft = np.zeros(len(dictCount))
            for word in words:
                if word in dictCount:
                    idx = dictCount.index(word)
                    ft[idx] += 1
        featureSet_train.append([list(ft), [0, 1]])
    # Creating features for Training - ham
    for file in ham_f:
        with open(hamPath + file, 'r') as f:
            contents = f.read()
            words = word_tokenize(contents)
            words = [str_red.lemmatize(i) for i in words]
            ft = np.zeros(len(dictCount))
            for word in words:
                if word in dictCount:
                    idx = dictCount.index(word)
                    ft[idx] += 1
        featureSet_train.append([list(ft), [1, 0]])
    print("Training feature set done")
    # Creating features for Testing - spam
    for file in spam_f_tst:
        with open(spamPath_tst + file, 'r') as f:
            contents = f.read()
            words = word_tokenize(contents)
            words = [str_red.lemmatize(i) for i in words]
            ft = np.zeros(len(dictCount))
            for word in words:
                if word in dictCount:
                    idx = dictCount.index(word)
                    ft[idx] += 1
        featureSet_test.append([list(ft), [0, 1]])
    # Creating features for Testing - ham
    for file in ham_f_tst:
        with open(hamPath_tst + file, 'r') as f:
            contents = f.read()
            words = word_tokenize(contents)
            words = [str_red.lemmatize(i) for i in words]
            ft = np.zeros(len(dictCount))
            for word in words:
                if word in dictCount:
                    idx = dictCount.index(word)
                    ft[idx] += 1
        featureSet_test.append([list(ft), [1, 0]])
    print("Testing feature set done")
    random.shuffle(featureSet_train)
    featureSet_train = np.array(featureSet_train)
    random.shuffle(featureSet_test)
    featureSet_test = np.array(featureSet_test)

    train_data = list(featureSet_train[:, 0])
    train_label = list(featureSet_train[:, 1])
    test_data = list(featureSet_test[:, 0])
    test_label = list(featureSet_test[:, 1])
    return train_data, train_label, test_data, test_label


if __name__ == '__main__':
    dictWords = create_dict()
    tr_data, tr_label, tst_data, tst_label = create_featureset(dictWords)
    with open('data/data.pickle', 'wb') as f:
        pickle.dump([tr_data, tr_label, tst_data, tst_label], f)
