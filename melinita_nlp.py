import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('vader_lexicon')

from gensim.models import LdaModel


import pandas as pd

from gensim import models
from gensim import corpora


from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

import pyLDAvis
import pyLDAvis.gensim_models as gensimvis



stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def get_stop_words():
    return stop_words

def get_lemmatizer():
    return lemmatizer

def preprocess_text(text):
    tokens = preprocess_text_as_tokens(text)  # Remove stopwords
    return ' '.join(tokens)

def preprocess_text_as_tokens(text):
    tokens = word_tokenize(text.lower())  # Tokenization and lowercase
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum()]  # Lemmatization
    return [token for token in tokens if token not in stop_words]  # Remove stopwords

def preprocess_series_of_texts(series):
    preprocessed_series = series.apply(preprocess_text)
    return preprocessed_series

def preprocess_series_of_texts_as_tokens(series):
    preprocessed_series = series.apply(preprocess_text_as_tokens)
    return preprocessed_series
    

def count_words_in_list_of_texts(list_of_texts):
    dict_count= {}
    for text in list_of_texts: 
        splitted_text = text.split()
        for word in splitted_text:
            if word in dict_count.keys():
                dict_count[word] = dict_count[word] + 1
            else:
                dict_count[word] = 1
                
    return dict_count


def do_nmf_vectorization_model(preprocessed_text_as_tokens_series, num_topics):
    texts = [' '.join(text) for text in preprocessed_text_as_tokens_series]

    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(texts)

    nmf_model = NMF(n_components=num_topics, random_state=1)
    nmf_W = nmf_model.fit_transform(tfidf)
    nmf_H = nmf_model.components_

    feature_names = tfidf_vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(nmf_H):
        print(f"Topic {topic_idx}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-11:-1]]))
                
        
def do_lda_model_with_graphs(preprocessed_text_as_tokens_series, num_topics, subreddit = "general"):

    dictionary = corpora.Dictionary(preprocessed_text_as_tokens_series.tolist())
    corpus = [dictionary.doc2bow(text) for text in preprocessed_text_as_tokens_series]

    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

    for idx, topic in lda_model.print_topics(num_topics=num_topics, num_words=10):
        print(f"Topic {idx}: {topic}")

    # Prepare the visualization data
    vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis_data, "lda_vis\\" + subreddit + '_lda_vis.html')
    
def remove_this_words_from_token_list(words_to_remove, preprocessed_text_as_tokens_list):
    result = []

    for token_list in preprocessed_text_as_tokens_list:
        new_token_list = []
        for token in token_list:
            if token not in words_to_remove:
                new_token_list.append(token)
                
        result.append(new_token_list)
    
    return result