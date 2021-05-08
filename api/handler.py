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
import pandas as pd

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
main supervised prediction implementation function
---------------------------------------------------------------------------------------------------
"""

def supervised_predict(post, vectorizer, model):
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
supervised prediction function
---------------------------------------------------------------------------------------------------
"""
def supervised_prediction(post):
    
    xgb_model = pickle.load(open('./../models/xgb.obj', 'rb'))
    vectorizer = pickle.load(open('./../models/vectorizer.obj', 'rb'))
    
    tags = supervised_predict(post, vectorizer, xgb_model)
    set_tags = set(tags)
    set_tags    
    output_tags = ""
    for t in set_tags:
        output_tags += t + ' '    
        
    return output_tags



"""
---------------------------------------------------------------------------------------------------
main unsupervised prediction implementation function
---------------------------------------------------------------------------------------------------
"""

def unsupervised_predict(post, lda_tags_df_scaled, dictionary, lda):
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    stemmer = EnglishStemmer()
    stemmed = ' '.join(stemmer.stem(WordNetLemmatizer().lemmatize(w, pos='v')) for w in w_tokenizer.tokenize(post))

    pattern = re.compile('[^A-Za-z +]')
    normalized = re.sub(pattern, ' ', stemmed)

    result = []
    for token in gensim.utils.simple_preprocess(normalized):
        if token not in gensim.parsing.preprocessing.STOPWORDS:
            result.append(token)
    this_dictionary = []
    this_dictionary.append(result)
    
    other_corpus = [dictionary.doc2bow(text) for text in this_dictionary]
    unseen_doc = other_corpus[0]
    vector = lda[unseen_doc]

    topic = vector[0][0][0]
    perc = vector[0][0][1]
    tags = lda_tags_df_scaled[int(topic)]
    tags_output = tags.sort_values(ascending=False).head(5)    
    
    return tags_output

"""
---------------------------------------------------------------------------------------------------
unsupervised prediction function
---------------------------------------------------------------------------------------------------
"""
def unsupervised_prediction(post):
    
    lda_tags_df_scaled = pickle.load(open('./../models/lda_tags_df_scaled.obj', 'rb'))
    lda = gensim.models.LdaMulticore.load('./../models/lda_model_tfidf_optimized_2')    
    dictionary = pickle.load(open('./../models/dictionary.obj', 'rb'))   
    
        
    tags_output = unsupervised_predict(post, lda_tags_df_scaled, dictionary, lda)
    output_tags = ""
    for index in tags_output.index:
        output_tags += index + ' '
        
    return output_tags
