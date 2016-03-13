import json, re
from nltk import FreqDist
from nltk.probability import ConditionalFreqDist
 
from nltk.corpus import stopwords
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.naive_bayes import BernoulliNB
#from sklearn.linear_model import LogisticRegression
import nltk
from operator import itemgetter
from nltk.classify import NaiveBayesClassifier
import collections
from nltk.metrics import precision, recall, f_measure
import itertools
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
        
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

def best_word_feats(words):
    return dict([(word, True) for word in words if word in bestwords])
 
def bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])

def best_bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    d = dict([(bigram, True) for bigram in bigrams])
    d.update(best_word_feats(words))
    return d
    
def evaluate_classifier(featx):
    negfeats = [((featx(f['text'].split(' ')), 'neg')) for f in negids]
    posfeats = [((featx(f['text'].split(' ')), 'pos')) for f in posids]
#    train_data = posids[:4000] + negids[:4000]    
#    test_data = posids[4000:] + negids[4000:]
    trainfeats = negfeats[:8000] + posfeats[:8000]
    testfeats = negfeats[8000:] + posfeats[8000:]
    print ('train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats))) 
    classifier = NaiveBayesClassifier.train(trainfeats)
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)
    for i, (feats, label) in enumerate(testfeats):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)
    print ('pos precision:', precision(refsets['pos'], testsets['pos']))
    print ('pos recall:', recall(refsets['pos'], testsets['pos']))
    print ('pos F-measure:', f_measure(refsets['pos'], testsets['pos']))
    print ('neg precision:', precision(refsets['neg'], testsets['neg']))
    print ('neg recall:', recall(refsets['neg'], testsets['neg']))
    print ('neg F-measure:', f_measure(refsets['neg'], testsets['neg']))
    print ('accuracy:', nltk.classify.util.accuracy(classifier, testfeats))
    classifier.show_most_informative_features() 
    
### MAIN FUNCTION    
    ###HUGE NOTE, MAKE SURE TO REMOVE SET WORDS FROM THE FREQDISTS
if __name__ == '__main__':
    pass
    """Need to try implementing text classifier as follows:
        group text into 2 categories, positive and negative
        create a transform/classifer to test and train on overall dataset
        using these two categories
        Run tests on this classifier
        Find ways to improve it"""
    data = load_json_data('yelp_academic_dataset_review.json', 1500000)
    posids, negids = split_by_rating(data[250000:1500000], 10000)
    print ('evaluating single word features')
    cap = 8000
    pos = 0
    neg = 0
    evaluate_classifier(word_feats)
    word_fd = FreqDist()
    label_word_fd = ConditionalFreqDist()
    for review in posids:
        pos += 1
        if (pos == 8000):
            break
        word_fd.update(review['text'].split(' '))
        label_word_fd['pos'].update(f for f in review['text'].split(' ') if f not in remove_set_words)
 
    for review in negids:
        neg += 1
        if (neg == 8000):
            break
        word_fd.update(review['text'].split(' '))
        label_word_fd['neg'].update(f for f in review['text'].split(' ') if f not in remove_set_words)
    
    pos_word_count = label_word_fd['pos'].N()
    neg_word_count = label_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count

    word_scores = {}
    for word, freq in word_fd.items():
        pos_score = BigramAssocMeasures.chi_sq(label_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(label_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score
    best = sorted(word_scores.items(), key=itemgetter(1), reverse=True)[:10000]
    bestwords = set([w for w, s in best])
    word_scores[word] = pos_score + neg_score
    print ('evaluating best word features')
    evaluate_classifier(best_word_feats)
    print ('evaluating best words + bigram chi_sq word features')
    evaluate_classifier(best_bigram_word_feats) 
#    print(len(posids))
#    print(len(negids))