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

