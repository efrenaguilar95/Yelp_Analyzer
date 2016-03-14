import json, re
from nltk import FreqDist
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import math
import nltk

from nltk.classify import NaiveBayesClassifier
#from nltk.classify import SklearnClassifier
#from sklearn.naive_bayes import BernoulliNB
#from sklearn.svm import SVC
#from nltk.classify import maxent
#from nltk.corpus import subjectivity
#from nltk.sentiment import SentimentAnalyzer
#from nltk.sentiment.util import *
import collections
from nltk.metrics import precision, recall, f_measure

def tokenize_simple(text):
    """
    Tokenizes a string and returns it as a bag of words

    Parameters
    ----------
    text : str
        A string of raw text to be tokenized
    
    Returns
    -------
    list of strs
        One string for each token in the document, in the same order as the original
    """
    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    replace_punctuation = str.maketrans(punctuation, " "*len(punctuation))
    text = text.lower()
    text = text.translate(replace_punctuation)
    text = re.sub(r" {2,}", " ", text)
    text = text.strip()
    return text.split()

def remove_stopwords_inner(tokens, stopwords):   
    """
    Removes stopwords within a bag of words
    
    Parameters
    ----------
    tokens : list of strs
        The bag of words to be edited
    stopwords : list of strs
        The list of stopwords to be removed from the bag of words
    
    Returns
    -------
    list of strs
        Returns the new bag of words, with all of the stopwords removed
    """
    stopwords = set(stopwords)
    tokens = [word for word in tokens if word not in stopwords]
    return tokens
    
def load_json_data(file, data_amount):
    """
    Loads the json data from a file and returns it as a dictionary
    
    Parameters
    ----------
    file : str
        The location of the json file to be loaded
    data_amount : int
        The amount of data entries to be loaded from the json file
    
    Returns
    -------
    list of dicts
        Returns the json data, represented by a list of dictionaries
    """
    data = []    
    with open(file) as f:
        count = 0
        for line in f:
            count+= 1
            if count == data_amount:
                break
            data.append(json.loads(line))
    return data

def split_by_rating(data, cap):
    """
    Splits json data based on if a review is positive or negative with a max
    number of positive and negative reviews
    
    Parameters
    ---------
    data : list of dicts
        The yelp data to be split
    cap : int
        The max amount of positive or negative reviews to be in a list
    
    Returns
    -------
    two lists of json dicts
        Returns two lists of json data. One for positive data, one for negative
        data
    """
    positive = []
    negative = []
    for review in data:
        if review['stars'] >= 4 and len(positive) != cap:
            positive.append(review)
        elif review['stars'] <= 2 and len(negative) != cap:
            negative.append(review)
    return positive, negative

def word_feats(words):
    data = [];
    for word in words:
        data.append(tokenize_simple(word))
    data = remove_stopwords_inner(words, stopwords = stopwords.words('english'))
    return dict([(word, True) for word in set(data)])
    
if __name__ == '__main__':
    
    data = load_json_data('yelp_academic_dataset_review.json', 40000)
    posids, negids = split_by_rating(data[:900000], 6000)
#    print(len(posids))
#    print(len(negids))
    negfeats = [((word_feats(f['text'].split(' ')), 'neg')) for f in negids]
    posfeats = [((word_feats(f['text'].split(' ')), 'pos')) for f in posids]
    train_data = posids[:4000] + negids[:4000]    
    test_data = posids[4000:] + negids[4000:]
    trainfeats = negfeats[:4000] + posfeats[:4000]
    testfeats = negfeats[4000:] + posfeats[4000:]
    print ('train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats))) 
    classifier = NaiveBayesClassifier.train(trainfeats)
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)
    for i, (feats, label) in enumerate(testfeats):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)
        
    #cross validation  3-fold   
    feats = negfeats + posfeats
    M = math.floor(len(feats)/ 3)
    result = []
    for n in range(3):
        val_set = feats[n*M:][:M]
        train_set = feats[(n+1)*M:] + feats[:n*M]
        classifier = nltk.NaiveBayesClassifier.train(train_set)
        result.append("{:.4f}".format(round(nltk.classify.accuracy(classifier, val_set)*100,4)))
    
    print('cross_validation:', result)
 
 
    print ('pos precision:', precision(refsets['pos'], testsets['pos']))
    print ('pos recall:', recall(refsets['pos'], testsets['pos']))
    print ('pos F-measure:', f_measure(refsets['pos'], testsets['pos']))
    print ('neg precision:', precision(refsets['neg'], testsets['neg']))
    print ('neg recall:', recall(refsets['neg'], testsets['neg']))
    print ('neg F-measure:', f_measure(refsets['neg'], testsets['neg']))
    classifier.show_most_informative_features()