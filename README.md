# NLP-MVP

# Project Description
Github is an online version control system that stores your code history. Repsositories are made to store different files that are written in various programming languages and named; usually based on what is inside the repository. The format for the repository titles are user_name/title. A github search query was made for COVID-19 and repository file titles were scraped and stored in a .json file. Natural Language Processing was applied to the README files in each repository and words were analyzed. A machine learning algoritm was applied to the information obtained from the natural language processing statifying on programming language in order to predict primary programming language of repository. 

# Project Goals
* Identify popular words that are associated with certain programming languages using natural language processing
* Develop a machine learning model to predict programming langauge of repositories

# Initial Thoughts
There are similarities in READMEs with the same programming language. We will be able to develop model with fairly high accuracy that uses words in the README to predict the 

# The Plan
* Search COVID-19 on github website
* Scrape username/title of repos using BeautifulSoup
* Export README contents of repos into .json file
* Explore data in search of common and unique words
    * Answer the following intial questions:
        * What are the top 20 words in all of the READMEs
        * What words are more/less common in READMEs across multiple programming langauges?
        * How often do common words occur in each programming language?
        * WHat words have the highest Inverse Documen Frequency (IDF)?
        * What are the top languages of READMEs from repos that focus on COVID-19
* Develop a machine learning model that will predict programming language
* Experiment with words used in model to improve accuracy:
   * Evaluate models on train and validate data
   * Select the best model based on accuracy
   * Evaluate the best model on test data
