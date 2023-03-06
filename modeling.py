import pandas as pd
import numpy as np 

from pprint import pprint
import unicodedata
import re
import json

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords

#function to split data for Count models including bi/tri-grams models
def split_cv_models(df, stem_or_lem, ngram_range = (1,1)): 
    
    random_seed = 42
    
    X = df[['stemmed', 'lemmatized']] 
    y = df.language
          
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = .3, random_state = random_seed)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = .5, random_state = random_seed)
    
    cv = CountVectorizer(ngram_range = ngram_range)
    X_train = cv.fit_transform(X_train[stem_or_lem])
    X_val = cv.transform(X_val[stem_or_lem])
    X_test = cv.transform(X_test[stem_or_lem])
    y_train = y_train 

    return X_train, y_train, X_val, y_val, X_test, y_test


def shape_split_data():
    
    X_train, y_train, X_val, y_val, X_test, y_test = split_cv_models(df,'lemmatized')
    print(X_train.shape, X_val.shape, X_test.shape) 
    print(y_train.shape[0], y_val.shape[0], y_test.shape[0])
    
    
def baseline_model(df, stem_or_lem, ngram_range = (1,1)):
    
    random_seed = 42
    
    X_train, y_train, X_val, y_val, X_test, y_test = split_cv_models(df, stem_or_lem, ngram_range = (1,1))
    
    baseline = (y_train =='others').mean()
    
    print(f'The baseline accuracy is {baseline:.2%}')
    
def cv_model(df, stem_or_lem, ngram_range = (1,1)):
    
    X_train, y_train, X_val, y_val, X_test, y_test = split_cv_models(df, stem_or_lem, ngram_range = (1,1))
    
    # Count Vectorizer
    bwtree = DecisionTreeClassifier(max_depth=12, random_state=123)
    bwtree.fit(X_train, y_train)
    print(f'Accuracy Score: {bwtree.score(X_val, y_val) * 100:.2f}%')    
    
    
def bigram_model(df, stem_or_lem, ngram_range = (1,1)):
    
    X2_train, y2_train, X2_val, y2_val, X2_test, y2_test = split_cv_models(df, stem_or_lem,
                                                                           ngram_range = ngram_range)
    
    # Bigram Count Vectorizer
    bitree = DecisionTreeClassifier(max_depth=16, random_state=13)
    bitree.fit(X2_train, y2_train)

    print(f'Accuracy Score: {bitree.score(X2_val, y2_val) * 100:.2f}%')  
    

def trigram_model(df, stem_or_lem, ngram_range = (1,1)):
    
    X3_train, y3_train, X3_val, y3_val, X3_test, y3_test = split_cv_models(df, stem_or_lem,
                                                                           ngram_range = ngram_range)
    
    # trigram Count Vectorizer
    titree = DecisionTreeClassifier(max_depth=12, random_state=123)
    titree.fit(X3_train, y3_train)

    print(f'Accuracy Score: {titree.score(X3_val, y3_val) * 100:.2f}%')  
    
    
def split_tf_idf_data(df, stem_or_lem): 
    
    random_seed = 42
    
    X = df[['stemmed', 'lemmatized']] 
    y = df.language
          
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = .3, random_state = random_seed)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = .5, random_state = random_seed)
    
    tfidf = TfidfVectorizer()
    X_train = tfidf.fit_transform(X_train[stem_or_lem])
    X_val = tfidf.transform(X_val[stem_or_lem])
    X_test = tfidf.transform(X_test[stem_or_lem])

    return X_train, y_train, X_val, y_val, X_test, y_test


def tf_idf_model(df, stem_or_lem):
    
    X_train, y_train, X_val, y_val, X_test, y_test = split_tf_idf_data(df, stem_or_lem)
    
    # Count Vectorizer
    tftree = DecisionTreeClassifier(max_depth=17, random_state=13)
    tftree.fit(X_train, y_train)

    print(f'Accuracy Score: {tftree.score(X_val, y_val) * 100:.2f}%')
    
    
def models(df, stem_or_lem):
    
    random_state = 42
    
    X_train, y_train, X_val, y_val, X_test, y_test = split_tf_idf_data(df, stem_or_lem)
    
    #Tree model
    tftree = DecisionTreeClassifier(max_depth = 2, random_state=random_state)
    tftree.fit(X_train, y_train)

    in_sample_accuracy = tftree.score(X_train, y_train)
    out_of_sample_accuracy = tftree.score(X_val, y_val)

    # KNN model
    knn = KNeighborsClassifier(n_neighbors = 7)
    knn = knn.fit(X_train, y_train)

    accuracy_train = knn.score(X_train, y_train)
    accuracy_val = knn.score(X_val, y_val)
    
    #Logistic Regression
    logit = LogisticRegression(random_state = random_state)
    logit.fit(X_train, y_train)

    acc_train = logit.score(X_train, y_train)
    acc_val = logit.score(X_val, y_val)
    
    #Random Forest
    
    rf = RandomForestClassifier(max_depth = 2, min_samples_leaf = 9,
                                random_state = random_state, n_estimators = 200)
    rf = rf.fit(X_train, y_train)
    
    in_accuracy = rf.score(X_train, y_train)
    out_accuracy = rf.score(X_val, y_val)
    
    
    #Baseline
    baseline_model(df, stem_or_lem)
    baseline = (y_train =='others').mean()
    
    dff = pd.DataFrame({'model': ['Decision Tree', 'KNN', 'Logistic Regression', 
                                  'Random Forest', 'Baseline'],
                      'train_accuracy': [in_sample_accuracy, accuracy_train,
                                         acc_train, in_accuracy ,baseline],
                      'validate_accuracy': [out_of_sample_accuracy, accuracy_val,
                                            acc_val, out_accuracy, baseline]})

    return dff.sort_values('validate_accuracy', ascending = False)


def best_model(df, stem_or_lem):
    
    random_state = 42
    
    X_train, y_train, X_val, y_val, X_test, y_test = split_tf_idf_data(df, stem_or_lem)
    
    knn = KNeighborsClassifier(n_neighbors = 7)
    knn = knn.fit(X_train, y_train)

    accuracy_train = knn.score(X_train, y_train)
    accuracy_val = knn.score(X_val, y_val)
    accuracy_test = knn.score(X_test, y_test)
    
    df = pd.DataFrame({'model': ['KNN','baseline'],
                      'train_accuracy': [accuracy_train, baseline],
                      'validate_accuracy': [accuracy_val, baseline],
                      'test_accuracy': [accuracy_test, baseline]})

    
    return df
    