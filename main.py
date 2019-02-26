import numpy as np
import pandas as pd
import os.path
import re
import string

def get_counts(filename):
    df = pd.read_csv(filename, delimiter=',', encoding = "ISO-8859-1", names = ["y", "garb", "time", "topic", "user", "tweet"])
    pos_dict = Counter()
    neg_dict = Counter()

    for index, row in df.iterrows():
        words = row["tweet"].lower().translate(string.punctuation).split()
        if(row["y"] == 4):
            for word in words:
                pos_dict[word] += 1
        else:
            for word in words:
                neg_dict[word] += 1

    return pos_dict, neg_dict

def learn_distributions(pos_dict, neg_dict):
    vocabulary = (set(pos_dict.keys()) | set(neg_dict.keys()))
    total_count = len(vocabulary)

    p_d = {}
    n_d = {}

    total_pos_count = sum(pos_dict.values())
    total_neg_count = sum(neg_dict.values())

    for w in vocabulary:
        p_d[w] = (pos_dict[w] + 1) / (total_pos_count + total_count)
        n_d[w] = (neg_dict[w] + 1) / (total_neg_count + total_count)

    return p_d, n_d

def classify(s, p_d, n_d):
    new_words = s.lower().translate(string.punctuation).split()
    vocabulary = (set(p_d.keys()) | set(n_d.keys()))
    pos_score = 0
    neg_score = 0

    for word in new_words:
        if word in vocabulary:
            pos_score += np.log(p_d[word])
            neg_score += np.log(n_d[word])

    return "positive" if pos_score > neg_score else "negative"

# Default dictionary
class Counter(dict):
    def __missing__(self, key):
        return 0

def calculate_error(filename, p_d, n_d):
    df = pd.read_csv(filename, delimiter=',', encoding = "ISO-8859-1", names = ["y", "garb", "time", "topic", "user", "tweet"])

    pos_incorrect = 0
    neg_incorrect = 0
    pos_total = 0
    neg_total = 0

    for index, row in df.iterrows():
        prediction = classify(row["tweet"], p_d, n_d)
        if(row["y"] == 4): # positive
            pos_total += 1
            if prediction == "negative":
                pos_incorrect += 1
        elif(row["y"] == 0): # negative
            neg_total += 1
            if prediction == "positive":
                neg_incorrect += 1

    return pos_incorrect/pos_total, neg_incorrect/neg_total

if __name__ == '__main__':
    pos_dict, neg_dict = get_counts(os.path.basename('train.csv'))
    p_d, n_d = learn_distributions(pos_dict, neg_dict)
    pos_rate, neg_rate = calculate_error(os.path.basename('test.csv'), p_d, n_d)

    print("Failure rate for positive samples: ", pos_rate)
    print("Failure rate for negative samples: ", neg_rate)

    while 1:
        s = input("Enter a tweet: ")
        pred = classify(s, p_d, n_d)
        print("Prediction: " + pred)

    # print(classify("I love twitter so much it's great!", p_d, n_d))
    # print(classify("Twitter is a terrible app!", p_d, n_d))
    # print(classify("what a horrible turn of events", p_d, n_d))
    # print(classify("with a bit of luck i'll get to see my wife next month", p_d, n_d))
    # print(classify("i'm pretty worried about this upcoming school year", p_d, n_d))
