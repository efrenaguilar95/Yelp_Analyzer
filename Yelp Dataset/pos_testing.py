# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 11:34:05 2016

@author: efrenaguilar
"""

import nltk

def pos_tag(text, tag_set):
    return nltk.pos_tag(text, tagset = tag_set)

def get_tag(tagged_token):
    return nltk.pos_tag(tagged_token[1])

def get_pos_freq_dist(tagged_token_list):
    pos_list = []
    for token_list in tagged_token_list:
        pos_list += [tag for (word, tag) in token_list]
    return nltk.FreqDist(pos_list)
    
def get_pos_list(tagged_tokens, pos):
    return [word for (word, tag) in tagged_text if pos == tag]



text = nltk.tokenize.word_tokenize("And now for something completely different")
tagged_text = nltk.pos_tag(text, tagset = "universal")
tagged_token = nltk.tag.str2tuple("beautiful/JJ")
token_tag = nltk.help.upenn_tagset(tagged_token[1])
tag_fd = nltk.FreqDist(tag for (word, tag) in tagged_text)