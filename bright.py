#!/usr/bin/env python

import sys
import urllib.request
import ssl

print('If you get error "ImportError: No module named \'six\'" install six:\n$ sudo pip install six\n\n')

# Bright Data Access Credentials
brd_user = 'hl_05950eb8'
brd_zone = 'serp_api1'
brd_passwd = ''
brd_superproxy = 'brd.superproxy.io:33335'  # Updated to port 33335
brd_connectStr = 'brd-customer-' + brd_user + '-zone-' + brd_zone + ':' + brd_passwd + '@' + brd_superproxy

brd_test_url = 'https://www.google.com/search?q=pizza'

# Path to the SSL certificate (update this to your actual certificate path)
ca_cert_path = '/mnt/c/Users/abhay/Desktop/scaper/brightdata_proxy_ca/New SSL certifcate - MUST BE USED WITH PORT 33335/BrightData SSL certificate (port 33335).crt'
context = ssl.create_default_context(cafile=ca_cert_path)

# Python 2 and Python 3 compatibility handling
if sys.version_info[0] == 2:
    print("Running in Python 2.x environment")
    import six
    from six.moves.urllib import request
    opener = request.build_opener(
        request.ProxyHandler(
            {'http': 'http://' + brd_connectStr,
             'https': 'https://' + brd_connectStr}),
        request.HTTPSHandler(context=context)
    )
    print("Attempting to open URL via proxy...")
    print(opener.open(brd_test_url).read())
    
elif sys.version_info[0] == 3:
    print("Running in Python 3.x environment")
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler(
            {'http': 'http://' + brd_connectStr,
             'https': 'https://' + brd_connectStr}),
        urllib.request.HTTPSHandler(context=context)
    )
    print("Attempting to open URL via proxy...")
    print(opener.open(brd_test_url).read().decode())

# Script to scrape IMDb reviews
import requests
from bs4 import BeautifulSoup
import csv
import os

def scrapeReviews(soup, ImdbId):
    print(f"Scraping reviews for IMDb ID: {ImdbId}")
    try:
        reviews = soup.find_all('div', {'class': 'imdb-user-review'})
        print(f"Found {len(reviews)} reviews on this page.")
    except:
        reviews = []
        print(f"No reviews found for IMDb ID: {ImdbId}")

    reviews_text = []
    for review in reviews:
        review_imdb = {}

        # Extracting reviewer name
        try:
            review_imdb['reviewer_name'] = review.find('span', {'class': 'display-name-link'}).find('a').string.strip()
        except:
            review_imdb['reviewer_name'] = ""
        print(f"Reviewer Name: {review_imdb['reviewer_name']}")

        # Extracting reviewer URL
        try:
            review_imdb['reviewer_url'] = review.find('span', {'class': 'display-name-link'}).find('a')['href']
        except:
            review_imdb['reviewer_url'] = ""
        print(f"Reviewer URL: {review_imdb['reviewer_url']}")

        # Extracting review ID
        try:
            review_imdb['data_review_id'] = review['data-review-id']
        except:
            review_imdb['data_review_id'] = ""
        print(f"Review ID: {review_imdb['data_review_id']}")

        # Extracting short review
        try:
            short_review = review.find('a', {'class': 'title'})
            review_imdb['short_review'] = short_review.string.strip()
        except:
            review_imdb['short_review'] = ""
        print(f"Short Review: {review_imdb['short_review']}")

        # Extracting full review
        try:
            full_review = review.find('div', {'class': 'show-more__control'})
            review_imdb['full_review'] = full_review.string.strip()
        except:
            review_imdb['full_review'] = ""
        print(f"Full Review: {review_imdb['full_review']}")

        # Extracting review date
        try:
            review_date = review.find('span', {'class': 'review-date'})
            review_imdb['review_date'] = review_date.string.strip()
        except:
            review_imdb['review_date'] = ""
        print(f"Review Date: {review_imdb['review_date']}")

        # Extracting rating value
        try:
            ratings_span = review.find('span', {'class': 'rating-other-user-rating'})
            review_imdb['rating_value'] = ratings_span.find('span').string.strip()
        except:
            review_imdb['rating_value'] = ""
        print(f"Rating Value: {review_imdb['rating_value']}")

        reviews_text.append(review_imdb)

    return reviews_text

def scrap(movie_url, ImdbId, all_data):
    print(f"Scraping URL: {movie_url}")
    r = requests.get(headers={'User-Agent': 'Mozilla/5.0'}, url=movie_url)
    soup = BeautifulSoup(r.text, 'html.parser')

    reviews_data = scrapeReviews(soup, ImdbId)
    all_data.extend(reviews_data)

    try:
        pagination_key = soup.find('div', {'class': 'load-more-data'})['data-key']
        print(f"Found pagination key: {pagination_key}. Loading next page...")
        movie_url = "https://www.imdb.com/title/" + ImdbId + "/reviews/_ajax?&paginationKey=" + pagination_key
        scrap(movie_url, ImdbId, all_data)
    except Exception as e:
        print(f"Pagination not found or error: {e}")
        print(f"Scraping completed for IMDb ID: {ImdbId}")
        return all_data

def start_scraping(ImdbId):
    print(f"Starting to scrape IMDb ID: {ImdbId}")
    movie_url = "https://www.imdb.com/title/" + ImdbId + "/reviews/_ajax?"
    all_data = []
    scrap(movie_url, ImdbId, all_data)
    return all_data

def save_to_csv(ImdbId, data):
    print(f"Saving reviews to CSV for IMDb ID: {ImdbId}")
    os.makedirs("reviews", exist_ok=True)
    csv_file_path = 'reviews/' + "reviews_" + ImdbId + '.csv'

    # Define the CSV columns
    fieldnames = ['reviewer_name', 'reviewer_url', 'data_review_id', 'short_review', 'full_review', 'review_date', 'rating_value']

    # Write data to CSV
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        # Write header
        writer.writeheader()

        # Write review data rows
        for review in data:
            writer.writerow(review)

    print(f"Data successfully saved to {csv_file_path}")

def start(ImdbId):
    print(f"Starting review scraping for IMDb ID: {ImdbId}")
    data = start_scraping(ImdbId)
    save_to_csv(ImdbId, data)
    print(f"Scraping and saving completed for IMDb ID: {ImdbId}")

# Start scraping for IMDb ID "tt0903747"
start(ImdbId="tt0903747")
