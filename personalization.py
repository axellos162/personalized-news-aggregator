'''This program uses collaborative filtering to recommend articles personalized to a specific user's interests based on their reading behavior. 
The program first loads the processed articles and user behavior data into pandas dataframes and merges them on the article ID. 
It then calculates the user-item matrix, which represents each user's rating (or lack thereof) for each article. 
The program then calculates the cosine similarity between users based on their reading behavior and defines a 
function to recommend articles to a user based on their reading behavior.

The recommend_articles function takes a user ID and an optional number of 
recommendations as input and returns a list of recommended articles along with their topics. 
The function first calculates the user's average rating for all articles and the cosine similarity 
between the user and all other users. It then calculates the weighted average rating for all articles 
based on user similarity and removes articles that the user has already rated. Finally, it sorts the 
unrated articles by their weighted rating and recommends the top N.'''

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Load the NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the list of stop words
stop_words = set(stopwords.words('english'))

# Load the processed articles into a pandas dataframe
articles_df = pd.read_csv('processed_articles.csv')

# Extract the headlines from the articles
headlines = articles_df['headline'].tolist()

# Tokenize the headlines into words
tokenized_headlines = []
for headline in headlines:
    words = word_tokenize(headline)
    # Remove stop words from the list of words
    words = [word.lower() for word in words if word.lower() not in stop_words]
    tokenized_headlines.append(' '.join(words))

# Create a bag-of-words representation of the headlines
vectorizer = CountVectorizer(max_features=1000, stop_words='english')
bow_headlines = vectorizer.fit_transform(tokenized_headlines)

# Perform LDA topic modeling on the bag-of-words representation
num_topics = 5
lda = LatentDirichletAllocation(n_components=num_topics, random_state=0)
lda.fit(bow_headlines)

# Identify the main topics covered in the headlines
topic_words = []
for topic_idx, topic in enumerate(lda.components_):
    top_words_idxs = topic.argsort()[:-6:-1]
    topic_words.append([vectorizer.get_feature_names_out()[i] for i in top_words_idxs])

# Add a new row to the dataframe with the identified topics for each article
topics_df = pd.DataFrame(topic_words, columns=[f"Topic {i}" for i in range(1, num_topics+1)])
articles_df = pd.concat([articles_df, topics_df], axis=1)

# Save the updated dataframe to the processed_articles.csv file
articles_df.to_csv('processed_articles.csv', index=False)
