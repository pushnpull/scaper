# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
import csv
import os
import pandas as pd
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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
def scrap(movie_url, ImdbId, all_data, proxies_list):
    headers = {'User-Agent': 'Mozilla/5.0'}
    max_retries = 5
    for attempt in range(max_retries):
        proxy = get_random_proxy(proxies_list)
        print(f"Attempt {attempt+1}: Using proxy {proxy}")
        try:
            response = requests.get(
                url=movie_url,
                headers=headers,
                proxies=proxy,
                timeout=10,
                verify=False  # Disable SSL verification for proxies
            )
            if is_blocked(response):
                print(f"Blocked or rate-limited with proxy {proxy}. Retrying...")
                time.sleep(5)
                continue
            else:
                break  # Successful request
        except Exception as e:
            print(f"Request failed with proxy {proxy}: {e}")
            time.sleep(5)
            continue
    else:
        print(f"Failed to retrieve {movie_url} after {max_retries} attempts.")
        return

    soup = BeautifulSoup(response.text, 'html.parser')

    reviews_data = scrapeReviews(soup, ImdbId)
    all_data.extend(reviews_data)

    try:
        pagination_key = soup.find('div', {'class': 'load-more-data'})['data-key']
        movie_url = f"https://www.imdb.com/title/{ImdbId}/reviews/_ajax?&paginationKey={pagination_key}"
        time.sleep(2)  # Delay between requests to avoid rate-limiting
        scrap(movie_url, ImdbId, all_data, proxies_list)
    except Exception as e:
        print("No more pages to load or error encountered.")
        return all_data

# Function to scrape all reviews for a movie using its IMDb ID
def start_scraping(ImdbId, proxies_list):
    all_data = []
    initial_url = f"https://www.imdb.com/title/{ImdbId}/reviews/_ajax"
    scrap(initial_url, ImdbId, all_data, proxies_list)
    return all_data

# Function to save reviews to a CSV file
def save_to_csv(ImdbId, data):
    os.makedirs("reviews", exist_ok=True)
    csv_file_path = f'reviewsfree/reviews_{ImdbId}.csv'

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

# Function to scrape proxies from spys.me and free-proxy-list.net
def scrape_proxies():
    proxies = set()

    # Scrape from spys.me
    print("Scraping proxies from spys.me...")
    try:
        regex = r"[0-9]+(?:\.[0-9]+){3}:[0-9]+"
        c = requests.get("https://spys.me/proxy.txt")
        proxies.update(re.findall(regex, c.text))
    except Exception as e:
        print(f"Error scraping spys.me: {e}")

    # Scrape from free-proxy-list.net
    print("Scraping proxies from free-proxy-list.net...")
    try:
        d = requests.get("https://free-proxy-list.net/")
        soup = BeautifulSoup(d.content, 'html.parser')
        table = soup.find('table', {'id': 'proxylisttable'})
        rows = table.tbody.find_all('tr')

        for row in rows:
            cols = row.find_all('td')
            if cols[6].text.strip() == 'yes':  # HTTPS support
                ip = cols[0].text.strip()
                port = cols[1].text.strip()
                proxy = f"{ip}:{port}"
                proxies.add(proxy)
    except Exception as e:
        print(f"Error scraping free-proxy-list.net: {e}")

    # Save proxies to file
    with open("proxies_list.txt", "w") as file:
        for proxy in proxies:
            file.write(f"{proxy}\n")

    print(f"Total proxies scraped: {len(proxies)}")
    return list(proxies)

# Function to validate proxies
def validate_proxy(proxy):
    test_url = 'https://httpbin.org/ip'
    try:
        response = requests.get(test_url, proxies=proxy, timeout=5, verify=False)
        if response.status_code == 200:
            return True
    except:
        return False
    return False

# Function to prepare the proxy list from the scraped proxies
def prepare_proxy_list():
    proxy_list = []
    with open('proxies_list.txt', 'r') as file:
        for line in file:
            proxy = line.strip()
            proxies = {
                'http': f'http://{proxy}',
                'https': f'http://{proxy}',
            }
            # Optionally validate proxies
            # if validate_proxy(proxies):
            proxy_list.append(proxies)
    print(f"Total proxies prepared: {len(proxy_list)}")
    return proxy_list

# Function to scrape reviews from CSV
def scrape_reviews_from_csv(csv_file_path):
    # Path to store pending IMDb IDs
    pending_file = 'pending_ids.txt'

    # Scrape proxies first
    scrape_proxies()

    # Prepare proxy list
    proxy_list = prepare_proxy_list()

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

    # Use ThreadPoolExecutor for parallel processing
    max_workers = 100  # Adjust based on your system capabilities
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_imdb = {}
        for imdb_id in pending_ids:
            future = executor.submit(process_imdb_id, imdb_id, proxy_list)
            future_to_imdb[future] = imdb_id

        for future in as_completed(future_to_imdb):
            imdb_id = future_to_imdb[future]
            try:
                future.result()
                # Remove the IMDb ID from pending IDs and update the pending file
                pending_ids.remove(imdb_id)
                write_pending_ids(pending_file, pending_ids)
            except Exception as e:
                print(f"An error occurred while processing IMDb ID {imdb_id}: {e}")
                print("Will retry this IMDb ID later.")

# Function to process a single IMDb ID
def process_imdb_id(imdb_id, proxy_list):
    try:
        reviews_data = start_scraping(imdb_id, proxy_list)
        if reviews_data:
            save_to_csv(imdb_id, reviews_data)
        else:
            print(f"No reviews found for IMDb ID: {imdb_id}")
    except Exception as e:
        print(f"An error occurred while processing IMDb ID {imdb_id}: {e}")
        raise e  # Raise exception to be caught by as_completed

# Main execution
if __name__ == "__main__":
    # Path to your CSV file containing the list of movies
    csv_file_path = 'ml-32m/remaining_linksb.csv'  # Update with your CSV file path

    # Start scraping reviews for all movies in the CSV
    scrape_reviews_from_csv(csv_file_path)
