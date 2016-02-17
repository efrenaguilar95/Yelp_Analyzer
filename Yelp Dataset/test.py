# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 11:19:17 2016

@author: Monami
"""
import re
import nltk 
from typing import Iterable, List, Tuple
from collections import Counter
import matplotlib.pyplot as plt
from nltk.corpus import gutenberg
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import FreqDist
import numpy as np
import math

def load_hamlet():
    """
    Loads the contents of the play Hamlet into a string.

    Returns
    -------
    str
        The one big, raw, unprocessed string.

    Example
    -------
    >>> document = load_hamlet()
    >>> document[:80]
    '[The Tragedie of Hamlet by William Shakespeare 1599]\n\n\nActus Primus. Scoena Prim'
    """
    return gutenberg.raw('shakespeare-hamlet.txt')

def tokenize_simple(text):
    #punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    #replace_punctuation = str.maketrans(punctuation, " "*len(punctuation))
    text = text.lower()
    #text = text.translate(replace_punctuation)
    text = re.sub(r" {2,}", " ", text)
    text = text.strip()
    return text.split()

def remove_stopwords_nltk(tokens):
    """
    Remove the NLTK English stopwords from the given tokens using remove_stopwords_inner.

    Parameters
    ----------
    tokens : Iterable[str]
        The tokens from which to remove stopwords.

    Returns
    -------
    List[str]
        The input tokens with NLTK's English stopwords removed.

    Example
    -------
    >>> document = load_hamlet()
    >>> tokens = tokenize_nltk(document)
    >>> tokens = remove_stopwords_nltk(tokens)
    >>> tokens[:10]
    ['[', 'tragedie', 'hamlet', 'william', 'shakespeare', '1599', ']', 'actus', 'primus', '.']
    """
    from nltk.corpus import stopwords
    return remove_stopwords_inner(tokens, stopwords=stopwords.words('english'))

def find_adjective(tokens):
    tagged_tokens = nltk.pos_tag(tokens, tagset = 'universal')
    result = []
    
    previous = None
    for index, t in enumerate(tagged_tokens):
        print(t)
        if t[1] == 'ADJ' and index != 0 :
            previous = tagged_tokens[index - 1]
            #print(t)
            #print(previous)
            result.append(previous[0] + ' ' + t[0])        
    
    return result
    
def split_two_word():
    result = []
    
    return result

if __name__ == '__main__':
    document = load_hamlet()
    tokens = tokenize_simple(document)[:50]
    remove_stopwords_nltk(tokens)
    find_adjective(tokens)