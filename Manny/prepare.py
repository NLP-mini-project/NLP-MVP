import unicodedata
import re
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

import pandas as pd

def basic_clean(text):
    '''
        This function accepts a string of text, cleans,
        and then returns the cleaned text.
    
        params:
        ------
        string: Input text to clean

        return:
        ------
        string: cleaned text
    '''
    
    # lowercase
    text = text.lower()
    
    # normalize unicode characters
    text = unicodedata.normalize('NFKD', text)\
    .encode('ascii', 'ignore')\
    .decode('utf-8', 'ignore')
    
    # only alphanumeric, apostrophe, & Spaces
    text = re.sub(r"[^a-z0-9'\s]", '', text)
    
    return text

def tokenize(text):
    '''
        this function accepts a string of text, tokenizes,
        and then returns the tokenized text.
    
        params:
        ------
        string: input text to tokenize

        return:
        ------
        string: tokenized text
    '''
    
    #create tokenizer object
    tokenizer = nltk.tokenize.ToktokTokenizer()

    #use the tokenizer
    text = tokenizer.tokenize(text, return_str = True)

    return text

def stem(text):
    '''
        this function accepts a string of text, stems,
        and then returns the stemmed string.
    
        params:
        ------
        string: input string to stem

        return:
        ------
        string: stemmed string
    '''
    
    #create stemmer object
    ps = nltk.porter.PorterStemmer()
    
    #use the stem, split text using each word
    stems = [ps.stem(word) for word in text.split()]
    
    #join stem word to text
    text = ' '.join(stems)

    return text

def lemmatize(text):
    '''
        this function accepts a string of text, lemmatizes,
        and then returns the lemmatized string.
    
        params:
        ------
        text:
            string: input string to lemmatize

        return:
        ------
        text:
            string: lemmatized string
    '''
    
    #  create lemmatizer object
    wnl = nltk.stem.WordNetLemmatizer()
    
    # split text string into words
    lemmas = [wnl.lemmatize(word) for word in text.split()]
    
    #join lemmatized words into article
    text = ' '.join(lemmas)

    return text

def remove_stopwords(text,extra_words=None,exclude_words=None):
    '''
        this function accepts a string of text, removes stopwords,
        and then returns the transformed string.
    
        params:
        ------
        text:
            string: input string to transform

        return:
        ------
        text:
            string: transformed string
    '''

    #create stopword list
    stopword_list = stopwords.words('english')
    
    #remove excluded words from list
    stopword_list = set(stopword_list) - set(exclude_words)
    
    #add the extra words to the list
    stopword_list = stopword_list.union(set(extra_words))
    
    #split the string into different words
    words = text.split()
    
    #create a list of words that are not in the list
    filtered_words = [word for word in words if word not in stopword_list]
    
    #join the words that are not stopwords (filtered words) back into the string
    text = ' '.join(filtered_words)
    
    return text

def prep_repo_data(df, column, extra_words=[], exclude_words=[]):
    '''
        This function take in a DataFrame formatted like so:
        "title 	content 	category"

        and the string name for a text column with 
        option to pass lists for extra_words and exclude_words and
        returns a df with the text article title, original text, stemmed text,
        lemmatized text, cleaned, tokenized, & lemmatized text with stopwords removed.
    '''

    #original text from content column
    df['original'] = df['readme_contents']
    
    #chain together clean, tokenize, remove stopwords
    df['clean'] = df[column].apply(basic_clean)\
                            .apply(tokenize)\
                            .apply(remove_stopwords, 
                                   extra_words=extra_words, 
                                   exclude_words=exclude_words)
    
    #chain clean, tokenize, stem, remove stopwords
    df['stemmed'] = df['clean'].apply(stem)
    
    #clean clean, tokenize, lemmatize, remove stopwords
    df['lemmatized'] = df['clean'].apply(lemmatize)
    
    return df[['language','repo', 'original', 'clean', 'stemmed', 'lemmatized']]
