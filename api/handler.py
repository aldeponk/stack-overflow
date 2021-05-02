import json


import pickle
import nltk
import nltk.data
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.snowball import EnglishStemmer
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 

import gensim
from gensim import corpora, models
from pprint import pprint
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

from xgboost import XGBClassifier


"""
---------------------------------------------------------------------------------------------------
tokenizer function
---------------------------------------------------------------------------------------------------
"""

def myTokenizer(text):
    '''
    Create tokens from text (English words > 3 letters)
    '''
    def stem_tokens(tokens, stemmer):
        '''
        Stem words in tokens.
        and suppress word < 3 characters
        '''
        stemmed = []
        for item in tokens:
            if re.match('[a-zA-Z0-9]{3,}',item):
                stemmed.append(stemmer.stem(item))
        return stemmed

    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, EnglishStemmer())
    return stems

"""
---------------------------------------------------------------------------------------------------
main prediction function
---------------------------------------------------------------------------------------------------
"""

def predict(post, vectorizer, model):
    pattern = re.compile('[^A-Za-z +]')
    intermediate = re.sub(pattern, ' ', post)
    intermediate = post.lower()

    stop_words = set(stopwords.words('english')) 
    #print(text)
    word_tokens = word_tokenize(intermediate) 
    filtered_text = ' '.join(w for w in word_tokens if not w in stop_words)

    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    stemmer = EnglishStemmer()
    stemmed = myTokenizer(filtered_text)
    x_input = vectorizer.transform(stemmed).astype('float64')
    tags = model.predict(x_input)
    return tags


"""
---------------------------------------------------------------------------------------------------
dummy test function
---------------------------------------------------------------------------------------------------
"""
def prediction(post):
    
    xgb_model = pickle.load(open('./../models/xgb.obj', 'rb'))
    vectorizer = pickle.load(open('./../models/vectorizer.obj', 'rb'))
    
    tags = predict(post, vectorizer, xgb_model)
    set_tags = set(tags)
    set_tags    
    output_tags = ""
    for t in set_tags:
        output_tags += t + ' '    
        
    return output_tags

