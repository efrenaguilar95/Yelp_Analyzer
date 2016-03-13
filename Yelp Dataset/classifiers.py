#Get data
import json, re

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
    punctuation = '!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~'
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

def bag_of_words(words):
    #Simple, tokenized, bag of words!    
    data = [];
    for word in words:
        for w in tokenize_simple(word):
            data.append(w)
    return dict([(word, True) for word in set(data)])
    
def bag_of_bestwords(words):
    data = [];
    for word in words:
        for w in tokenize_simple(word):
            data.append(w)
    return dict([(word, True) for word in set(data) if word in bestwords])
    
def bag_of_words_remove(words, badwords):
    #Removes words we do not want in our bag of words.
    return bag_of_words(set(words) - set(badwords))
    
def bag_of_words_good(words, goodwords):
    return bag_of_words(set(words) & set(goodwords))
    
def bag_of_stopwords(words, stopfile='english'):
    #Removes stopwords. Can be manually set.
    badwords = stopwords.words(stopfile)
    return bag_of_words_remove(words, badwords)
    
def bag_of_bigrams(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    #Returns not only a bag of words, but a bag of bigrams holy mother dam daniel dude so op wtf.
    #Maximum number of bigrams is 'n' AKA n=200
    finder = BigramCollocationFinder.from_words(words)
    bigrams = finder.nbest(score_fn, n)
    return bag_of_words(words + bigrams)
    
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
        
        #Is this indentation correct???
        #Wouldn't it just need to run once???
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
                word_fd.update(tokenize_simple(word))
                label_word_fd['pos'].update(tokenize_simple(word))
 
    for review in negids:
        neg += 1
        if (neg != cutoff):
            for word in review['text'].split(' '):
                word_fd.update(tokenize_simple(word))
                label_word_fd['neg'].update(tokenize_simple(word))
    
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
    
def evaluate_classifier(featx, split=0.75):
    #featx the bag of words function to use
    #This function will do many things listed here.
    # 1) It will split the categories AKA the negative and positive reviews
    # 2) It will add a cutoff so that 75% is training data, 25% is testing
    # 3) It will then train the classifier and print out whatever is needed.
    negfeats = [((featx(f['text'].split(' ')), 'neg')) for f in negids]
    posfeats = [((featx(f['text'].split(' ')), 'pos')) for f in posids]
    negcutoff = int(len(negfeats)*split)
    poscutoff = int(len(posfeats)*split)
    trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
    testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
    print ('train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats))) 
    mb_classifier = SklearnClassifier(MultinomialNB()).train(trainfeats)
    lr_classifier = SklearnClassifier(LogisticRegression()).train(trainfeats)
    lsvm_classifier = SklearnClassifier(LinearSVC()).train(trainfeats)
    nsvm_classifier = SklearnClassifier(NuSVC()).train(trainfeats)
    classifier_list = [(mb_classifier, 'Multinomial Naive Bayes'),
                       (lr_classifier, 'Logistic Regression'),
                       (lsvm_classifier, 'Linear Support Vector Machine'),
                       (nsvm_classifier, 'Nu Support Vector Machine')]
#    refsets = collections.defaultdict(set)
#    testsets = collections.defaultdict(set)
#    for i, (feats, label) in enumerate(testfeats):
#        refsets[label].add(i)
#        observed = classifier.classify(feats)
#        testsets[observed].add(i)
    
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
    evaluate_classifier(bag_of_bestwords)    

    