import json, re
from nltk.corpus import stopwords
import numpy as np


#Creates positive and negative ratios and returns an array of 1(positive) and 0(negative).    
def ratio_weight(data):
    pos_words = [line.strip() for line in open('positive-words.txt', 'r')]
    neg_words = [line.strip() for line in open('negative-words.txt', 'r')]
    ret_set = []
    for r in data:
        pos_count = 0
        neg_count = 0
        for word in remove_stopwords_inner(r['text'].split(),stopwords.words('english')):
            if word in pos_words:
                pos_count += 1
            elif word in neg_words:
                neg_count += 1
        if pos_count > neg_count:
            ret_set.append(1)
        else:
            ret_set.append(0)                
    return np.array(ret_set)

#Loads yelp's json data
def load_json_data(file, data_amount):
    data = []    
    with open(file) as f:
        count = 0
        for line in f:
            count+= 1
            if count == data_amount:
                break
            data.append(json.loads(line))
    return data

def remove_stopwords_inner(tokens, stopwords):   
    stopwords = set(stopwords)
    tokens = [word for word in tokens if word not in stopwords]
    return tokens

if __name__ == '__main__':
    data = load_json_data('yelp_academic_dataset_review.json', 10000)
    print(ratio_weight(data))