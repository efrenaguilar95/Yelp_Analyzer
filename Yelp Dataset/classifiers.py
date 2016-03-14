#Get data
import json
import token_helpers

#FreqDist maybe remove
from nltk.probability import FreqDist, ConditionalFreqDist

#Gets index from dict and other dict stuff
from operator import itemgetter
import collections
import itertools

#NLTK stuff
import nltk
from nltk.metrics import precision, recall
from nltk.classify import NaiveBayesClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.corpus import stopwords

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
        if len(positive)+len(negative) == cap*2:
            return positive, negative
        if review['stars'] >= 4 and len(positive) != cap:
            positive.append(review)
        elif review['stars'] <= 2 and len(negative) != cap:
            negative.append(review)
    return positive, negative

def remove_reviews(data, min_len, max_len):
    """
    Takes json data and removes any entries with review lengths above or
    below set limits
    
    Parameters
    ----------
    data : a list of dicts
        The yelp data to be analyzed
    
    min_len : int
        The minimum cut off point. If a review is less than this, it is removed
    
    max_len : int
        The maximum cut off point. If a review is more than this, it is removed
    
    Returns
    -------
    list of dicts
        Returns the json data with review with lengths above or below parameters
        removed
    
    """
    copy = []
    for review in data:
        if min_len < len(review['text']) < max_len:
            copy.append(review)
    return copy
    
def precision_and_recall(classifier, testfeats):
    #Finds precision and recall on that big booty classifier.
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)
    
    
    #Feats is the dictionary of words
    #label is the label, pos or neg
    for i, (feats, label) in enumerate(testfeats):
        
        #a mapping of which entries are pos and negative
        #ex refsets[pos] = {1,2,3,4,6,7,11,78}
        refsets[label].add(i)
        
        #Classifies something as pos or neg given its feats
        observed = classifier.classify(feats)
        
        #a mapping of entries and their classifications
        #ex testsets[pos] = {1,2,3,4,5,8,11}
        testsets[observed].add(i)
        
        prec = {}
        rec = {}
        
    for label in classifier.labels():
        prec[label] = precision(refsets[label], testsets[label])
        rec[label] = recall(refsets[label], testsets[label])
    
    return prec, rec

def high_words(posids, negids, cutoff, score_fn=BigramAssocMeasures.chi_sq, min_score=5):

    word_fd = FreqDist()
    label_word_fd = ConditionalFreqDist()
    pos = 0
    neg = 0
    for review in posids:
        pos += 1
        if (pos != cutoff):
            for word in review['text'].split(' '):
                word_fd.update(token_helpers.tokenize_simple(word))
                label_word_fd['pos'].update(token_helpers.tokenize_simple(word))
 
    for review in negids:
        neg += 1
        if (neg != cutoff):
            for word in review['text'].split(' '):
                word_fd.update(token_helpers.tokenize_simple(word))
                label_word_fd['neg'].update(token_helpers.tokenize_simple(word))
    
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
    return bestwords
    
    """
def create_train_and_test_set(data, cap, split=0.75):
    {
    
        return train_set, test_set
    }
    """

def create_train_and_test_sets(posData, negData, bow_func, bestWords = [], bestOnly = False, split=0.75):
    """
    Takes positive and negative data and splits them into a training and testing
    set, split by a given percentage.
    
    Parameters
    ----------
    posData : a list of dicts
        A list of positive yelp review data
    
    negData : a list of dicts
        A list of negative yelp review data
    
    bestWords : an iterable of strings
        The tokens to keep in the data sets
        Default : an empty list. bestWords should only be assigned if bestOnly
        is assigned as True
    
    bestonly : bool
        True if only the words in bestWords should be kept
        False if all words should be kept
        Default : False
    
    split : float
        The percentage of data to use for training, the rest will be used for
        testing
    
    Returns
    -------
    two lists : train_set, test_set
        Returns a list of training and testing data in the format
        [[list of words in document], label]
    
    """
    if (bestOnly):
        negfeats = [((bow_func(f["text"], bestWords), "neg")) for f in negData]
        posfeats = [((bow_func(f["text"], bestWords), "pos")) for f in posData]            
    else:
        negfeats = [((bow_func(f["text"]), "neg")) for f in negData]
        posfeats = [((bow_func(f["text"]), "pos")) for f in posData]
    negcutoff = int(len(negfeats)*split)
    poscutoff = int(len(posfeats)*split)
    train_set = negfeats[:negcutoff] + posfeats[:poscutoff]
    test_set = negfeats[negcutoff:] + posfeats[poscutoff:]
    return train_set,test_set
            
            

def evaluate_classifier(trainfeats, testfeats, split=0.75):
    #featx the bag of words function to use
    #This function will do many things listed here.
    # 1) It will split the categories AKA the negative and positive reviews
    # 2) It will add a cutoff so that 75% is training data, 25% is testing
    # 3) It will then train the classifier and print out whatever is needed.
    print ('train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats))) 
    mb_classifier = SklearnClassifier(MultinomialNB()).train(trainfeats)
    lr_classifier = SklearnClassifier(LogisticRegression()).train(trainfeats)
    lsvm_classifier = SklearnClassifier(LinearSVC()).train(trainfeats)
    nsvm_classifier = SklearnClassifier(NuSVC()).train(trainfeats)
    classifier_list = [(mb_classifier, 'Multinomial Naive Bayes'),
                       (lr_classifier, 'Logistic Regression'),
                       (lsvm_classifier, 'Linear Support Vector Machine'),
                       (nsvm_classifier, 'Nu Support Vector Machine')]
                       
    #PRINTS OUT ACCURACY, RECALL, AND PRECISION FOR CLASSIFIERS
    for classifier, name in classifier_list:
        prec, rec = precision_and_recall(classifier, testfeats)
        print ('NAME OF CLASSIFIER: ', name)
        print ('accuracy:', nltk.classify.util.accuracy(classifier, testfeats))
        print ('pos precision: ', prec['pos'])
        print ('pos recall: ', rec['pos'])
        print ('neg precision:', prec['neg'])
        print ('neg recall:', rec['neg'])
        if name != classifier_list[-1][1]:
            print('\n --- Next Classifier ---\n')
    
    
### MAIN FUNCTION    
if __name__ == '__main__':
    """Need to try implementing text classifier as follows:
        group text into 2 categories, positive and negative
        create a transform/classifer to test and train on overall dataset
        using these two categories
        Run tests on this classifier
        Find ways to improve it"""
    data = load_json_data('yelp_academic_dataset_review.json', 700000)
    cap = 5000
    posids, negids = split_by_rating(data[250000:1500000], cap)
    bestwords = high_words(posids, negids, int(cap*2*0.75))
    #train_data, test_data = create_train_and_test_sets(posids, negids, token_helpers.bag_of_bestwords, bestwords, True)
    train_data, test_data = create_train_and_test_sets(posids, negids, token_helpers.bag_of_bestwords_and_bigrams, bestwords, True)
    evaluate_classifier(train_data, test_data)
    k = token_helpers.bag_of_bestwords_and_bigrams(posids[1]["text"], bestwords)
    