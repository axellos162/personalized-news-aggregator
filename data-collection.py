import csv
import requests
from bs4 import BeautifulSoup

# List of news sources to scrape
news_sources = [
    'https://www.bbc.com/news',
    'https://www.reuters.com/',
    'https://www.nytimes.com/',
    # Add more sources here
]

# Loop through each news source and scrape the articles
for source in news_sources:
    # Send a request to the news source's website
    response = requests.get(source)
    # Parse the HTML content of the response using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    # Find all the links to articles on the website
    article_links = soup.find_all('a', href=True)
    # Loop through each link and check if it is a news article
    for link in article_links:
        url = link['href']
        # Check if the link is a news article
        if 'article' in url:
            # Send a request to the article's URL
            article_response = requests.get(url)
            # Parse the HTML content of the article using BeautifulSoup
            article_soup = BeautifulSoup(article_response.content, 'html.parser')
            # Extract the headline and content of the article
            headline = article_soup.find('h1')
            content = article_soup.find('div', {'class': 'article-body'})
            # Check if the headline and content elements exist
            if headline and content:
                headline_text = headline.text
                content_text = content.text
                # Store the article in a CSV file
                with open('articles.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([source, url, headline_text, content_text])
