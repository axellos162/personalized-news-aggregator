from flask import Flask, render_template
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import random

# Load the processed articles into a pandas dataframe
articles_df = pd.read_csv('processed_articles.csv')

# Load the user's interests from a file or database
user_interests = ['politics', 'sports', 'technology']

# Use content analysis to identify the main topics covered in the articles
vectorizer = CountVectorizer(max_features=1000, stop_words='english')
bow_articles = vectorizer.fit_transform(articles_df['headline'])
num_topics = 5
lda = LatentDirichletAllocation(n_components=num_topics, random_state=0)
lda.fit(bow_articles)
topic_words = []
for topic_idx, topic in enumerate(lda.components_):
    top_words_idxs = topic.argsort()[:-6:-1]
    topic_words.append([vectorizer.get_feature_names_out()[i] for i in top_words_idxs])

# Use collaborative filtering to recommend articles based on the user's interests and reading behavior
def get_recommendations(user_interests, articles_df):
    # Create a user profile based on their reading history
    user_profile = np.zeros((num_topics,))
    user_articles = articles_df[articles_df['category'].isin(user_interests)]
    for article in user_articles['headline']:
        article_bow = vectorizer.transform([article])
        article_topics = lda.transform(article_bow)
        user_profile += article_topics[0]
    user_profile /= len(user_articles)

    # Find articles similar to the user's interests
    similarities = []
    for idx, row in articles_df.iterrows():
        article_bow = bow_articles[idx]
        article_topics = lda.transform(article_bow)
        similarity = np.dot(user_profile, article_topics.T)
        similarities.append(similarity)
    articles_df['similarity'] = similarities
    recommendations = articles_df.sort_values('similarity', ascending=False).head(10)
    return recommendations

app = Flask(__name__)

@app.route('/')
def index():
    # Get personalized article recommendations
    recommendations = get_recommendations(user_interests, articles_df)
    # Convert the recommendations to a list of dictionaries
    articles = []
    for idx, row in recommendations.iterrows():
        article = {}
        article['headline'] = row['headline']
        article['category'] = row['category']
        article['publisher'] = row['publisher']
        article['date'] = row['date']
        article['url'] = row['url']
        articles.append(article)
    # Randomize the order of the articles
    random.shuffle(articles)
    # Render the recommendations in an HTML template
    return render_template('index.html', articles=articles)

if __name__ == '__main__':
    app.run(debug=True)
