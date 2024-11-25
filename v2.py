# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
import csv
import os
import pandas as pd
import time
import threading
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

# For proxy setup
import pprint

# Function to check if we are rate-limited or blocked
def is_blocked(response):
    if response.status_code != 200:
        print(f"Received status code {response.status_code}")
        return True
    # Additional checks can be implemented here based on response content
    return False

# Function to scrape reviews for a given IMDb ID
def scrapeReviews(soup, ImdbId):
    try:
        reviews = soup.find_all('div', {'class': 'imdb-user-review'})
    except Exception as e:
        print(f"Error finding reviews: {e}")
        return []
    
    reviews_text = []
    for review in reviews:
        review_imdb = {}

        # Extracting reviewer name
        try:
            review_imdb['reviewer_name'] = review.find('span', {'class': 'display-name-link'}).find('a').string.strip()
        except:
            review_imdb['reviewer_name'] = ""

        # Extracting reviewer URL
        try:
            review_imdb['reviewer_url'] = review.find('span', {'class': 'display-name-link'}).find('a')['href']
        except:
            review_imdb['reviewer_url'] = ""

        # Extracting review ID
        try:
            review_imdb['data_review_id'] = review['data-review-id']
        except:
            review_imdb['data_review_id'] = ""

        # Extracting short review
        try:
            short_review = review.find('a', {'class': 'title'})
            review_imdb['short_review'] = short_review.string.strip()
        except:
            review_imdb['short_review'] = ""

        # Extracting full review
        try:
            full_review = review.find('div', {'class': 'show-more__control'})
            review_imdb['full_review'] = full_review.get_text(strip=True)
        except:
            review_imdb['full_review'] = ""

        # Extracting review date
        try:
            review_date = review.find('span', {'class': 'review-date'})
            review_imdb['review_date'] = review_date.string.strip()
        except:
            review_imdb['review_date'] = ""

        # Extracting rating value
        try:
            ratings_span = review.find('span', {'class': 'rating-other-user-rating'})
            review_imdb['rating_value'] = ratings_span.find('span').string.strip()
        except:
            review_imdb['rating_value'] = ""

        reviews_text.append(review_imdb)

    return reviews_text

# Function to recursively scrape all reviews (including pagination)
def scrap(movie_url, ImdbId, all_data, proxies, ca_cert_path):
    print(f"Scraping reviews for IMDb ID: {ImdbId} | URL: {movie_url}")
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(
            url=movie_url,
            headers=headers,
            proxies=proxies,
            timeout=10,
            verify=ca_cert_path
        )
        if is_blocked(response):
            print("Blocked or rate-limited. Retrying after delay...")
            time.sleep(60)  # Wait for 60 seconds before retrying
            return scrap(movie_url, ImdbId, all_data, proxies, ca_cert_path)
    except Exception as e:
        print(f"Exception during request: {e}")
        print("Retrying after delay...")
        time.sleep(60)
        return scrap(movie_url, ImdbId, all_data, proxies, ca_cert_path)

    soup = BeautifulSoup(response.text, 'html.parser')

    reviews_data = scrapeReviews(soup, ImdbId)
    all_data.extend(reviews_data)

    try:
        pagination_key = soup.find('div', {'class': 'load-more-data'})['data-key']
        movie_url = f"https://www.imdb.com/title/{ImdbId}/reviews/_ajax?&paginationKey={pagination_key}"
        # time.sleep(2)  # Delay between requests to avoid rate-limiting
        scrap(movie_url, ImdbId, all_data, proxies, ca_cert_path)
    except Exception as e:
        print("No more pages to load or error encountered.")
        return all_data

# Function to scrape all reviews for a movie using its IMDb ID
def start_scraping(ImdbId, proxy, ca_cert_path):
    all_data = []
    initial_url = f"https://www.imdb.com/title/{ImdbId}/reviews/_ajax"
    scrap(initial_url, ImdbId, all_data, proxy, ca_cert_path)
    return all_data

# Function to save reviews to a CSV file
def save_to_csv(ImdbId, data):
    os.makedirs("reviews", exist_ok=True)
    csv_file_path = f'reviewsb/reviews_{ImdbId}.csv'

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

# Function to read pending IMDb IDs from a file
def read_pending_ids(pending_file):
    if os.path.exists(pending_file):
        with open(pending_file, 'r') as f:
            pending_ids = f.read().splitlines()
    else:
        pending_ids = []
    return pending_ids

# Function to write pending IMDb IDs to a file
def write_pending_ids(pending_file, pending_ids):
    with open(pending_file, 'w') as f:
        for imdb_id in pending_ids:
            f.write(f"{imdb_id}\n")

# Function to get a random proxy from the list
def get_random_proxy(proxy_list):
    return random.choice(proxy_list)

# Function to prepare the proxy list
def prepare_proxy_list():
    # Replace with your actual proxy credentials and setup
    proxy_list = []

    # Example of adding multiple proxies to the list
    # You need to replace this with actual proxies from your proxy provider
    for i in range(11):  # Assuming you have 10 proxies
        host = 'brd.superproxy.io'
        port = 33335
# 'http': 'http://brd-customer-hl_75908a63-zone-residential_proxy1:43j1chzo8dgj@brd.superproxy.io:22225'
        username = f'brd-customer-hl_05950eb8-zone-residential_proxy1' #ninehz
        username = f'brd-customer-hl_75908a63-zone-residential_proxy1' #kaluraam
        username = f'brd-customer-hl_75908a63-zone-datacenter_proxy1' # kaluraam 5$remaining
        username = f'brd-customer-hl_d4c9e22d-zone-datacenter_proxy1' #qrixzs
        username = f'brd-customer-hl_d4a3822b-zone-datacenter_proxy1' #shiva
        
        password = ''
        password = ''
        password = ''
        password = ''
        password = ''
        
        
        
        

        proxy_url = f'http://{username}:{password}@{host}:{port}'

        proxies = {
            'http': proxy_url,
            'https': proxy_url
        }
        proxy_list.append(proxies)

    return proxy_list

# Main function to scrape reviews for all movies listed in the CSV file
def scrape_reviews_from_csv(csv_file_path):
    # Path to store pending IMDb IDs
    pending_file = 'pending_ids.txt'

    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Convert IMDb IDs to strings and prepend 'tt'
    df['imdbId'] = df['imdbId'].apply(lambda x: 'tt' + str(x).zfill(7))

    # Read pending IDs
    pending_ids = read_pending_ids(pending_file)

    if not pending_ids:
        # If pending IDs are empty, initialize with all IMDb IDs from the CSV
        pending_ids = df['imdbId'].tolist()
        write_pending_ids(pending_file, pending_ids)

    print(f"Pending IMDb IDs: {pending_ids}")

    # Prepare proxy list
    proxy_list = prepare_proxy_list()

    # Path to your CA certificate file
    ca_cert_path = '/mnt/c/Users/abhay/Desktop/scaper/brightdata_proxy_ca/New SSL certifcate - MUST BE USED WITH PORT 33335/BrightData SSL certificate (port 33335).crt'  # Update with your CA certificate path

    # Use ThreadPoolExecutor for parallel processing
    max_workers = 250  # Adjust this number based on your system capabilities
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_imdb = {}
        # pending_ids.reverse()
        for imdb_id in pending_ids:
            proxy = get_random_proxy(proxy_list)
            future = executor.submit(process_imdb_id, imdb_id, proxy, ca_cert_path)
            future_to_imdb[future] = imdb_id

        for future in as_completed(future_to_imdb):
            imdb_id = future_to_imdb[future]
            try:
                result = future.result()
                # Remove the IMDb ID from pending IDs and update the pending file
                pending_ids.remove(imdb_id)
                write_pending_ids(pending_file, pending_ids)
            except Exception as e:
                print(f"An error occurred while processing IMDb ID {imdb_id}: {e}")
                print("Will retry this IMDb ID later.")

# Function to process a single IMDb ID
def process_imdb_id(imdb_id, proxy, ca_cert_path):
    try:
        reviews_data = start_scraping(imdb_id, proxy, ca_cert_path)
        if reviews_data:
            save_to_csv(imdb_id, reviews_data)
        else:
            print(f"No reviews found for IMDb ID: {imdb_id}")
    except Exception as e:
        print(f"An error occurred while processing IMDb ID {imdb_id}: {e}")
        raise e  # Raise exception to be caught by as_completed

# Path to your CSV file containing the list of movies
csv_file_path = '/mnt/c/Users/abhay/Desktop/scaper/ml-32m/remaining_linksb.csv'

# Start scraping reviews for all movies in the CSV
scrape_reviews_from_csv(csv_file_path)
