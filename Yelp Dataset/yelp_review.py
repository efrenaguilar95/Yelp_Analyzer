import json, re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import FreqDist

def tokenize_simple(text):
    #punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    #replace_punctuation = str.maketrans(punctuation, " "*len(punctuation))
    text = text.lower()
    #text = text.translate(replace_punctuation)
    text = re.sub(r" {2,}", " ", text)
    text = text.strip()
    return text.split()

def remove_stopwords_inner(tokens, stopwords):   
    stopwords = set(stopwords)
    tokens = [word for word in tokens if word not in stopwords]
    return tokens

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

def split_by_rating(data):
    positive = []
    negative = []
    for review in data:
        if review['stars'] >= 4:
            positive.append(review)
        elif review['stars'] <= 2:
            negative.append(review)
    return positive, negative
    
def get_freq_dist(data):
    tokens = []
    for review in data:
        tokens += tokenize_simple(review['text'])
    tokens = remove_stopwords_inner(tokens, stopwords = stopwords.words('english'))
    return FreqDist(tokens)
    
if __name__ == '__main__':
    data = load_json_data('yelp_academic_dataset_review.json', 420000)
    positive, negative = split_by_rating(data)
    pos_dist = get_freq_dist(positive)
    neg_dist = get_freq_dist(negative)
    
    
    print(pos_dist.most_common(69))
    print ('\n\n')
    print(neg_dist.most_common(69))
            
