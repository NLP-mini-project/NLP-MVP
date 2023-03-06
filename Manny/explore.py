from sklearn.model_selection import train_test_split

def split_data(df,random_state=42):
    '''
    This function take in a dataframe performs a train, validate, test split

    100 = 50 + 30 + 20 = 100

    Returns train, validate, test
    '''
    
    # create train_validate, and test datasets
    train, test = train_test_split(df, train_size = 0.8, random_state = random_state)
    
    # create train and validate datasets
    # 62.5% of 80% = 50% (Leaving 30% for Validate)
    train, validate = train_test_split(train, train_size = 0.625, random_state = random_state)

    return train, validate, test

def get_wordcounts_for_language(df, column, language):
    # filter dataframe to rows with specified language
    language_df = df[df['language'] == language]

    # create a new series by splitting the text in the specified column into individual words
    words_series = language_df[column].str.split(expand=True).stack()

    # group the series by word and count the number of occurrences
    words_counts = words_series.groupby(words_series).count()
    
    # sort value counts in descending order
    words_counts = words_counts.sort_values(ascending=False) 
    
    return words_counts

def get_unique_words_for_language(df, column, language):
    '''
    This function is the same as get_wordcounts_for_language(),
    but without the counts, and just the words
    '''
    # filter dataframe by rows with the specified language
    language_df = df[df['language'] == language]

    # create a new series by splitting the text in the specified column into individual words
    # using Pandas explode method
    words_series = language_df[column].str.split(expand=True).stack()

    # get a series holding each unique word for the specified language
    unique_words_series = pd.Series(words_series.unique()).reset_index(drop=True)

    return unique_words_series

def get_words_for_language(df, column, language):
    '''
    This is the same as get_unique_words_for_language()
    '''
    # filter dataframe by rows with the specified language
    language_df = df[df['language'] == language]

    # create a new series by splitting the text in the specified column into individual words
    # using Pandas explode method
    words_series = language_df[column].str.split(expand=True).stack()
    
    # reset index to remove double index caused by expand=True
    words_series = words_series.reset_index(drop=True)
    
    return words_series

def get_top_words_by_language(df, column_name, n_top_words):
    # group the dataframe by language
    grouped_df = df.groupby('language')
    
    # initiialize an empty dictionary to store the results
    results = {}
    
    # loop over each group
    for language, group in grouped_df:
        # get the value counts for the specified column
        word_counts = group[column_name].str.split(expand=True).stack().value_counts()
        # sort the value counts in descending order
        word_counts_sorted = word_counts.sort_values(ascending=False)
        # get the top n words
        top_words = word_counts_sorted.iloc[:n_top_words]
        
        # add the language and top words to the results dictionary
        results[language] = top_words
        
    # get the value counts for the specified column for the ungrouped dataframe
    total_word_counts = df[column_name].str.split(expand=True).stack().value_counts()
    # sort the value counts in descending order
    total_word_counts_sorted = total_word_counts.sort_values(ascending=False)
    # get the top n words
    top_total_words = total_word_counts_sorted.iloc[:n_top_words]
    
    # add the total top words to the results dictionary
    results['total'] = top_total_words
    
    # create a dataframe from the results dictionary
    df_results = pd.DataFrame(results)
    
    # replace NaN values with 0
    df_results.fillna(0, inplace=True)
    
    # sort the columns by the total values
    df_results = df_results.sort_values(by='total',ascending=False)[:n_top_words]
    
    return df_results

def language_breakdown(df,column):
    '''
    This function is like value_counts but for a specified column
    and adds a proportion column as well
    '''
    # Group by "language" column and sum the word count for each group
    word_count_by_language = df.groupby("language")[column].apply(lambda x: x.str.split().str.len().sum()).reset_index(name="word_count")

    # Calculate the total word count for all languages
    total_word_count = word_count_by_language["word_count"].sum()

    # Calculate the proportion of total words in each language
    word_count_by_language["percentage"] = word_count_by_language["word_count"] / total_word_count

    # format percentage column
    word_count_by_language["percentage"] = word_count_by_language["percentage"].map("{:.2%}".format)
    
    # sort and fix index
    word_count_by_language = word_count_by_language.sort_values(by='word_count',ascending=False).reset_index(drop=Tr)
    
    return word_count_by_language
