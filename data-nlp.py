'''This program loads the NLTK resources and defines a function called process_text that performs various 
NLP tasks on a given text. The function tokenizes the text into sentences and words, removes stop words, 
performs part-of-speech tagging, and performs named entity recognition. It returns a dictionary containing 
the processed text, including the sentences, words, part-of-speech tags, and named entities.

The program then reads the articles from the articles.csv file and uses the process_text function to 
perform NLP on the article content. It prints the results, including the news source, article URL, 
headline, and named entities extracted from the article content. Note that you can customize the NLP 
tasks and output format to suit your specific requirements.'''

import csv
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag, map_tag
from nltk.chunk import ne_chunk

# Load the NLTK resources
nltk.download('universal_tagset')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Load the list of stop words
stop_words = set(stopwords.words('english'))

# Define a function to perform NLP on the given text
def process_text(text):
    # Tokenize the text into sentences and words
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    # Remove stop words from the list of words
    words = [word.lower() for word in words if word.lower() not in stop_words]
    # Perform part-of-speech tagging on the list of words
    pos_tags = pos_tag(words, tagset='universal')
    # Map the part-of-speech tags to a simpler tag set
    simple_pos_tags = map_tag('en-ptb', 'universal', tuple(pos_tags))
    # Perform named entity recognition on the list of words
    named_entities = ne_chunk(pos_tags)
    # Extract the named entities from the named entity tree
    entities = []
    for subtree in named_entities.subtrees():
        if subtree.label() == 'NE':
            entity = ' '.join([word for word, tag in subtree])
            entities.append((entity, subtree.label()))
    # Return the processed text
    return {
        'sentences': sentences,
        'words': words,
        'pos_tags': simple_pos_tags,
        'entities': entities
    }


# Open the articles CSV file and read the articles
with open('articles.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip the header row
    rows = []
    for row in reader:
        source = row[0]
        url = row[1]
        headline = row[2]
        content = row[3]
        # Perform NLP on the article content
        processed_content = process_text(content)
        # Add the processed content to the rows list
        rows.append({
            'source': source,
            'url': url,
            'headline': headline,
            'sentences': processed_content['sentences'],
            'words': processed_content['words'],
            'pos_tags': processed_content['pos_tags'],
            'entities': processed_content['entities']
        })
        
# Create a data frame from the rows list
df = pd.DataFrame(rows)

# Save the data frame to a CSV file
df.to_csv('processed_articles.csv', index=False)

'''This code reads the articles from an input CSV file named articles.csv, 
performs NLP on the article content using the process_text() function, 
and stores the processed data into a data frame with columns for source, 
url, headline, sentences, words, pos_tags, and entities. 
It then saves the data frame into a CSV file named processed_articles.csv. 
The index=False argument tells pandas not to include the row index in the output CSV file.'''