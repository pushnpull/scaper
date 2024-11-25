import pandas as pd

# Load the MovieLens dataset
movielens_data = pd.read_csv("ml-32m/ratings.csv")  # Replace with the actual path to your MovieLens data file

# Load the MovieLens to IMDb mapping dataset
movie_mapping = pd.read_csv("ml-32m/links.csv")  # Replace with the actual path to your movie mapping file

# Merge MovieLens data with the mapping on the 'movieId' column
movielens_data_with_imdb = movielens_data.merge(movie_mapping[['movieId', 'imdbId']], on='movieId', how='left')

# Drop the original movieId column if only imdbId is needed
movielens_data_with_imdb = movielens_data_with_imdb.drop(columns=['movieId'])

# Save the updated dataset if needed
movielens_data_with_imdb.to_csv("movielens_data_with_imdb.csv", index=False)

print("Conversion complete. Updated data with IMDb IDs is saved as 'movielens_data_with_imdb.csv'.")
