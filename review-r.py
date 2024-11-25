# # -*- coding: utf-8 -*-

# import requests
# from bs4 import BeautifulSoup
# from flask import  jsonify
# import json,os


# #movie_url = "https://www.imdb.com/title/"+ImdbId+"/reviews/_ajax?"+"sort="+sort+"&dir="+dir+"&ratingFilter="+ratingFilter

# #movie_url = "https://www.imdb.com/title/tt0944947/reviews/_ajax?"
 
 
# def scrapeReviews(soup,ImdbId) :  
    
#     try :
#         reviews = soup.find_all('div',{'class' : 'imdb-user-review'})
#     except :
#         pass   
    

#     data = {}
#     data['ImdbId'] = ImdbId
#     reviews_text =[]
#     for review in reviews :
#         review_imdb={}
        
#         ################
#         try :
#             review_imdb['reviewer_name']=review.find('span',{'class':'display-name-link'}).find('a').string.strip()
#         except :
#             review_imdb['reviewer_name']=""    
#         ###############
#         try :
#             review_imdb['reviewer_url']=review.find('span',{'class':'display-name-link'}).find('a')['href']
#         except:
#             review_imdb['reviewer_url']=""
#         ############
#         try :
#             review_imdb['data-review-id']=review['data-review-id']
#         except :
#             review_imdb['data-review-id']=""
            
#         #############
#         try:
#             short_review =review.find('a',{'class': 'title'})
#             text=short_review.string.strip()
#             review_imdb['short_review']=text
#         except :
#             review_imdb['short_review']=""
    
#         ######################
#         try :
#             full_review = review.find('div',{'class' : 'show-more__control'})
#             text = full_review.string.strip()
#             review_imdb['full_review']=text
#         except :
#              review_imdb['full_review']=""
#         #############
#         try :
#             review_date = review.find('span',{'class' : 'review-date'})
#             text=review_date.string.strip()
#             review_imdb['review_date'] =text    
#         except :
#              review_imdb['review_date']  ="" 
#         #######
#         try :
#             ratings_span = review.find('span',{'class' : 'rating-other-user-rating'})
#             text=ratings_span.find('span').string.strip()
#             review_imdb['rating_value']  = text      
#         except :
#             review_imdb['rating_value']  = "" 
#         ##########
#         reviews_text.append(review_imdb)    
    
#     data['reviews']=reviews_text
#     return data



# def scrap(movie_url,ImdbId,all_data) :
#     print(movie_url)
#     r = requests.get(headers={'User-Agent': 'Mozilla/5.0'},url=movie_url)
#     soup = BeautifulSoup(r.text, 'html.parser')
     
#     data =scrapeReviews(soup,ImdbId) 
#     all_data.append(data)
#     try :
#         pagination_key =soup.find('div',{'class' : 'load-more-data'})['data-key']
#         movie_url = "https://www.imdb.com/title/"+ImdbId+"/reviews/_ajax?&paginationKey="+pagination_key
# #        print(movie_url) 
#         scrap(movie_url,ImdbId,all_data)
#     except Exception as e:             
#         print(e,"scraping done successfully")
#         return all_data
        

    
# def start_scraping(ImdbId) :
#      movie_url = "https://www.imdb.com/title/"+ImdbId+"/reviews/_ajax?"
#      all_data=[]
#      scrap(movie_url,ImdbId,all_data)
#      reviews= {}
#      reviews['ImdbId']=ImdbId
#      reviews['reviews']=all_data
#      return reviews



# def start(ImdbId) :
#     data = start_scraping(ImdbId)
#     os.makedirs("reviews", exist_ok=True)
#     with open('reviews/'+"reviews_"+ImdbId+'.json', 'w') as json_file:
#         json.dump(data, json_file)

# start(ImdbId="tt0903747")



# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
import csv
import os

def scrapeReviews(soup, ImdbId):
    try:
        reviews = soup.find_all('div', {'class': 'imdb-user-review'})
    except:
        pass

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
            review_imdb['full_review'] = full_review.string.strip()
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

def scrap(movie_url, ImdbId, all_data):
    print(movie_url)
    r = requests.get(headers={'User-Agent': 'Mozilla/5.0'}, url=movie_url)
    soup = BeautifulSoup(r.text, 'html.parser')

    reviews_data = scrapeReviews(soup, ImdbId)
    all_data.extend(reviews_data)

    try:
        pagination_key = soup.find('div', {'class': 'load-more-data'})['data-key']
        movie_url = "https://www.imdb.com/title/" + ImdbId + "/reviews/_ajax?&paginationKey=" + pagination_key
        scrap(movie_url, ImdbId, all_data)
    except Exception as e:
        print(e, "scraping done successfully")
        return all_data

def start_scraping(ImdbId):
    movie_url = "https://www.imdb.com/title/" + ImdbId + "/reviews/_ajax?"
    all_data = []
    scrap(movie_url, ImdbId, all_data)
    return all_data

def save_to_csv(ImdbId, data):
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
    data = start_scraping(ImdbId)
    save_to_csv(ImdbId, data)

start(ImdbId="tt0903747")
