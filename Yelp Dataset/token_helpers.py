import re
from nltk.metrics import BigramAssocMeasures
from nltk.collocations import BigramCollocationFinder
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

def bag_of_words(words):
    #Simple, tokenized, bag of words!    
    data = [];
    for word in words:
        for w in tokenize_simple(word):
            data.append(w)
    return dict([(word, True) for word in set(data)])

def bag_of_bestwords(words, bestwords):
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
    
if __name__ == '__main__':
    pass