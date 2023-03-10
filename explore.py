import unicodedata
import re
import json
import os
from requests import get
from bs4 import BeautifulSoup

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer

def tts(df, stratify=None):
    '''
    removing your test data from the data
    '''
    train_validate, test=train_test_split(df, 
                                 train_size=.7, 
                                 random_state=8675309,
                                 stratify=None)
    '''
    splitting the remaining data into the train and validate groups
    '''            
    train, validate =train_test_split(train_validate, 
                                      test_size=.3, 
                                      random_state=8675309,
                                      stratify=None)
    return train, validate, test

ADDITIONAL_STOPWORDS = ['covid19', 'coronavirus']

def clean(text):
    'A simple function to cleanup text data'
    wnl = nltk.stem.WordNetLemmatizer()
    stopwords = nltk.corpus.stopwords.words('english') + ADDITIONAL_STOPWORDS
    text = (unicodedata.normalize('NFKD', text)
             .encode('ascii', 'ignore')
             .decode('utf-8', 'ignore')
             .lower())
    words = re.sub(r'[^\w\s]', '', text).split()
    return [wnl.lemmatize(word) for word in words if word not in stopwords]



def ex_q1(df):
    '''
    returns top languages
    '''
    return df.language.value_counts()



def explore_q2(train):
    '''
    this function will print out the dataframe for words in each language 
    '''
    
    other_words = clean(' '.join(train[train.language == 'others'].lemmatized))
    js_words = clean(' '.join(train[train.language == 'JavaScript'].lemmatized))
    py_words = clean(' '.join(train[train.language == 'Python'].lemmatized))
    jn_words = clean(' '.join(train[train.language == 'Jupyter Notebook'].lemmatized))
    html_words = clean(' '.join(train[train.language == 'HTML'].lemmatized))
    r_words = clean(' '.join(train[train.language == 'R'].lemmatized))
    all_words = clean(' '.join(train.lemmatized))
    
    other_freq = pd.Series(other_words).value_counts()
    js_freq = pd.Series(js_words).value_counts()
    py_freq = pd.Series(py_words).value_counts()
    jn_freq = pd.Series(jn_words).value_counts()
    html_freq = pd.Series(html_words).value_counts()
    r_freq = pd.Series(r_words).value_counts()
    all_freq = pd.Series(all_words).value_counts()
    
    word_counts = (pd.concat([all_freq, other_freq, js_freq, py_freq, jn_freq, html_freq, r_freq], axis=1, sort=True)
                .set_axis(['all', 'other', 'js', 'py', 'jn', 'html', 'r'], axis=1, inplace=False)
                .fillna(0)
                .apply(lambda s: s.astype(int)))

    return word_counts.sort_values(by='all', ascending=False).head(20)




def words(train):
    other_words = clean(' '.join(train[train.language == 'others'].lemmatized))
    js_words = clean(' '.join(train[train.language == 'JavaScript'].lemmatized))
    py_words = clean(' '.join(train[train.language == 'Python'].lemmatized))
    jn_words = clean(' '.join(train[train.language == 'Jupyter Notebook'].lemmatized))
    html_words = clean(' '.join(train[train.language == 'HTML'].lemmatized))
    r_words = clean(' '.join(train[train.language == 'R'].lemmatized))

    all_words = clean(' '.join(train.lemmatized))
    
    
    other_freq = pd.Series(other_words).value_counts()
    js_freq = pd.Series(js_words).value_counts()
    py_freq = pd.Series(py_words).value_counts()
    jn_freq = pd.Series(jn_words).value_counts()
    html_freq = pd.Series(html_words).value_counts()
    r_freq = pd.Series(r_words).value_counts()
    all_freq = pd.Series(all_words).value_counts()
    
    
    word_counts = (pd.concat([all_freq, other_freq, js_freq, py_freq, jn_freq, html_freq, r_freq], axis=1, sort=True)
                .set_axis(['all', 'other', 'js', 'py', 'jn', 'html', 'r'], axis=1, inplace=False)
                .fillna(0)
                .apply(lambda s: s.astype(int)))
    return word_counts


def explore_plots(train):
    '''
    This function plots the necessary plots to visualize in exploration
    '''
    word_counts=words(train)
    
    other_pop=word_counts.sort_values(by='other', ascending=False).head(10)
    js_pop=word_counts.sort_values(by='js', ascending=False).head(10)
    py_pop=word_counts.sort_values(by='py', ascending=False).head(10)
    jn_pop=word_counts.sort_values(by='jn', ascending=False).head(10)
    html_pop=word_counts.sort_values(by='html', ascending=False).head(10)
    r_pop=word_counts.sort_values(by='r', ascending=False).head(10)
    
    plt.figure(figsize=(15,5))
    plt.subplot(321)
    plt.bar(height=other_pop['other'],x=other_pop.index,label='Other')
    plt.title('Common Words for "Other"')
    plt.xlabel('Words')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.xticks(rotation=30)
    plt.legend()
    

    plt.subplot(322)
    plt.bar(height=js_pop['js'],x=js_pop.index,label='JavaScript', color='rebeccapurple')
    plt.title('Common Words for JavaScript')
    plt.xlabel('Words')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.xticks(rotation=30)
    plt.legend()
    
    
    plt.subplot(323)
    plt.bar(height=py_pop['py'],x=js_pop.index,label='Python', color='seagreen')
    plt.title('Common Words for Python')
    plt.xlabel('Words')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.xticks(rotation=30)
    plt.legend()
    
    
    plt.subplot(324)
    plt.bar(height=jn_pop['jn'],x=js_pop.index,label='Jupyter Notebook', color='peru')
    plt.title('Common Words for Jupyter Notebook')
    plt.xlabel('Words')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.xticks(rotation=30)
    plt.legend()
    
    
    plt.subplot(325)
    plt.bar(height=html_pop['html'],x=js_pop.index,label='HTML', color='darkkhaki')
    plt.title('Common Words for HTML')
    plt.xlabel('Words')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.xticks(rotation=30)
    plt.legend()
    
    
    plt.subplot(326)
    plt.bar(height=r_pop['r'],x=js_pop.index,label='R', color= 'darkred')
    plt.title('Common Words for R')
    plt.xlabel('Words')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.xticks(rotation=30)
    plt.legend()
    
    
    
    
    
    plt.subplots_adjust(left=0.1,
                            bottom=-0.1,
                            right=0.9,
                            top=2,
                            wspace=0.4,
                            hspace=0.4)
    plt.show()
    
    

def get_list_unique_words(df, column):
    all_text = ' '.join(df[column].tolist())
    all_words = pd.Series(all_text.split()).unique().tolist()
    return all_words

def get_series_all_words(df,column):
    # Split the words into a list
    test_df = pd.DataFrame()
    test_df['words'] = df[column].str.split()
    
    # Explode the list to create a series of individual words
    word_series = test_df['words'].explode()
    word_series = word_series.reset_index(drop=True)
    
    return word_series

def idf(unique_words, corpus):
    idf_scores = {}
    n_docs = len(corpus)
    
    for word in unique_words:
        n_occurrences = sum([1 for doc in corpus if word in doc])
        idf_scores[word] = n_docs / n_occurrences
        
    idf_df = pd.DataFrame(list(idf_scores.items()), columns=['word', 'idf_score'])
    return idf_df

def get_top_idf_words(train, n):
    # Combine all rows into a giant string
    all_text = ' '.join(train['lemmatized'].tolist())

    # Create a TfidfVectorizer object
    vectorizer = TfidfVectorizer()

    # Fit the vectorizer on the combined text
    vectorizer.fit_transform([all_text])

    # Get the feature names and their idf values
    feature_names = vectorizer.get_feature_names()
    idf_values = vectorizer.idf_

    # Create a dictionary of feature names and their idf values
    idf_dict = dict(zip(feature_names, idf_values))

    # Sort the dictionary by idf values and get the top n words with highest idf
    top_n_words = sorted(idf_dict.items(), key=lambda x: x[1], reverse=True)[:n]

    # Create a dataframe with the top n words and their idf scores
    df = pd.DataFrame(top_n_words, columns=['word', 'idf_score'])

    return df
