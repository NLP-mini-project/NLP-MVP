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




def ex_q1(df):
    '''
    returns top languages
    '''
    return df.language.value_counts()



def explore_q2(train):
    '''
    this function will print out the dataframe for words in each language 
    '''
    
    other_words = clean(' '.join(train[train.language == 'others'].readme_contents))
    js_words = clean(' '.join(train[train.language == 'JavaScript'].readme_contents))
    py_words = clean(' '.join(train[train.language == 'Python'].readme_contents))
    jn_words = clean(' '.join(train[train.language == 'Jupyter Notebook'].readme_contents))
    html_words = clean(' '.join(train[train.language == 'HTML'].readme_contents))
    r_words = clean(' '.join(train[train.language == 'R'].readme_contents))
    all_words = clean(' '.join(train.readme_contents))
    
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

    return word_counts['all'].sort_values(by='all', ascending=False).head(20)


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