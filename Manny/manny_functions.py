import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

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

