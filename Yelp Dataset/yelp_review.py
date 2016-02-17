import json, re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import FreqDist
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import math

#tokenizes text and makes bag of words
def tokenize_simple(text):
    #punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    #replace_punctuation = str.maketrans(punctuation, " "*len(punctuation))
    text = text.lower()
    #text = text.translate(replace_punctuation)
    text = re.sub(r" {2,}", " ", text)
    text = text.strip()
    return text.split()

#Removes stopwords in bag of words
def remove_stopwords_inner(tokens, stopwords):   
    stopwords = set(stopwords)
    tokens = [word for word in tokens if word not in stopwords]
    return tokens

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

#Splits the data into positive and negative reviews, caps the length of list
def split_by_rating(data, cap):
    positive = []
    negative = []
    for review in data:
        if review['stars'] >= 4 and len(positive) != cap:
            positive.append(review)
        elif review['stars'] <= 2 and len(negative) != cap:
            negative.append(review)
    return positive, negative

#Gets the freq dist of bag of words 'DASH IS TEMPORARY FIX'
def get_freq_dist(data):
    tokens = []
    for review in data:
        tokens += tokenize_simple(review['text'])
    tokens = remove_stopwords_inner(tokens, stopwords = stopwords.words('english') + ['time', 'would', 'got', 'i\'m', '-', 'food', 'like', 'really', 'service'])
    return FreqDist(tokens)

#wtf
def get_count_vect(data):
    tokens = []
    count_vect = CountVectorizer(stop_words = stopwords.words("english"))
    for review in data:
        tokens += tokenize_simple(review['text'])
    return count_vect.fit_transform(tokens)
    
#Takes list of data.most_common yelp and a word limit to process a bar graph of # of most common words
def plot_common_words(data, word_limit):
    freq = get_freq_dist(data).most_common(word_limit)
    fig, axes = plt.subplots()    
    fig.suptitle('Most Common Words' )
    axes.set_xlabel('Word')
    axes.set_ylabel('Number of Tokens')
    maxWord = word_limit
    x = np.array(range(0, word_limit+1))
    histList = []
    amountWords = []
    counter = 0
    while len(histList) < maxWord:
        histList.append(freq[counter][0])
        amountWords.append(freq[counter][1])
        counter += 1
    axes.bar(range(len(histList)), amountWords, width = .5)
    plt.xticks(x, histList)
    return (fig, axes)

#Takes list of yelp data and splits data, converts to list of length for reviews and charts it.
def plot_review_length(data, n_fold):
    fig, axes = plt.subplots()
    fig.suptitle('Length of Reviews Bar Graph')
    axes.set_xlabel('Length of Review')
    axes.set_ylabel('Number of Reviews')
    len_data = FreqDist(len(review['text']) for review in data)
    fold = math.floor(len(len_data) / n_fold)
    range_list = []    
    amountReviews = []
    x = np.array(range(0, n_fold+1))
    for num in range(n_fold):
        range_list.append(str((fold*(num+1)) - fold + 1) + '-' + str(fold*(num+1)))
        counter = 0
        for number in range((fold*num+1) - fold + 1, fold*(num+1)):
            counter += len_data[number]
        amountReviews.append(counter)
    print(amountReviews[0])
    axes.bar(range(len(range_list)), amountReviews, width = .5)
    plt.xticks(x, range_list)
    return (fig, axes)

#Takes the data and removes reviews that do not go past the min or exceed the max. Returns back list of dict.
def remove_reviews(data, min_len, max_len):
    copy = []
    for review in data:
        if min_len < len(review['text']) < max_len:
            copy.append(review)
    return copy

### MAIN FUNCTION    
if __name__ == '__main__':
    pass
    """Need to try implementing text classifier as follows:
        group text into 2 categories, positive and negative
        create a transform/classifer to test and train on overall dataset
        using these two categories
        Run tests on this classifier
        Find ways to improve it"""
    data = load_json_data('yelp_academic_dataset_review.json', 40000)
    train_target = []
    positive_train, negative_train = split_by_rating(data[:14000], 2500)
    plot_common_words(positive_train, 20)
    plot_common_words(negative_train, 20)
#    plot_review_length(positive_train, 10)
    train_target = np.array([1]*len(positive_train) + [0]*len(negative_train))
    train_data = positive_train+negative_train
    positive_test, negative_test = split_by_rating(data[14000:], 2500)
    test_target = np.array([1]*len(positive_test) + [0]*len(negative_test))
    test_data = positive_test + negative_test
    count_vect = CountVectorizer(stop_words = stopwords.words("english"))
    train_tokens = []
    test_tokens = []
    for review in train_data:
        train_tokens += [review['text']]
    for review in test_data:
        test_tokens += [review['text']]
    X_train_counts = count_vect.fit_transform(train_tokens)
    X_test_counts = count_vect.transform(test_tokens)
    tf_transformer = TfidfTransformer()
    X_train_tfidf = tf_transformer.fit_transform(X_train_counts)
    X_test_tfidf = tf_transformer.transform(X_test_counts)
    clf = MultinomialNB()
    clf.fit(X_train_tfidf, train_target)
    predicted_multNB = clf.predict(X_test_tfidf)
    clf = BernoulliNB()
    clf.fit(X_train_tfidf, train_target)
    predicted_bernoulliNB = clf.predict(X_test_tfidf)
    
    clf = LogisticRegression().fit(X_train_tfidf, train_target)
    predicted_LR = clf.predict(X_test_tfidf)    
    print('Accuracy with multinomial naive Bayes: ', '%.4f'
          % np.mean(predicted_multNB == test_target) )
    print('Accuracy with Bernoulli naive Bayes: ', '%.4f'
          % np.mean(predicted_bernoulliNB == test_target) )    
    print('Accuracy with Logistic Regression: ', '%.4f'
          % np.mean(predicted_LR == test_target))
#    pos_dist = get_freq_dist(positive)
#    neg_dist = get_freq_dist(negative)
#    pos_vect = get_count_vect(positive)
#    #neg_vect = get_count_vect(negative)
#    
#    tf_transformer = TfidfTransformer();
#    pos_transform = tf_transformer.fit_transform(pos_vect)
#    #neg_transform = tf_transformer.transform(neg_vect)
#    
#    
#    
##    print(pos_dist.most_common(69))
##    print ('\n\n')
##    print(neg_dist.most_common(69))            
