import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import FreqDist

def remove_stopwords_inner(tokens, stopwords):
    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'    
    stopwords = set(stopwords)
    for x in punctuation:
        stopwords.add(x)
    tokens = [word for word in tokens if word not in stopwords]
    return tokens
    
data = []
with open('yelp_academic_dataset_review.json') as f:
    count = 0
    for line in f:
        count += 1
        if count == 420000:
            break
        data.append(json.loads(line))
positive = []
negative = []
for review in data:
    if review['stars'] >= 4:
        positive.append(review)
    elif review['stars'] <= 2:
        negative.append(review)

#positive_words = []
#for review in positive:
#    for word in review['text'].split(' '):
#        positive_words.append(word.lower())
#positive_words = remove_stopwords_inner(positive_words, stopwords = stopwords.words('english'))
#pos = FreqDist(positive_words)
#print(pos.most_common(69))
#print ('\n\n')

negative_words = []
for review in negative:
    for word in review['text'].split(' '):
        negative_words.append(word.lower())
negative_words = remove_stopwords_inner(negative_words, stopwords = stopwords.words('english'))
neg = FreqDist(negative_words)
print(neg.most_common(69))
        
