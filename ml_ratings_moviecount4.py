import pandas as pd

# Load your MovieLens data (replace 'movielens_data.csv' with the actual file path)
# Note: If you have a large dataset, use the 'chunksize' parameter in read_csv to load data in parts.
movielens_data = pd.read_csv('movielens_ratings_with_imdb_r_100.csv')

# Step 1: Calculate the number of ratings for each movie (imdbId)
movie_rating_counts = movielens_data['imdbId'].value_counts()

# Step 2: Filter movies that have been rated at least 4 times
movies_with_enough_ratings = movie_rating_counts[movie_rating_counts >= 4].index

# Step 3: Filter the original data to only include ratings for these movies
filtered_data = movielens_data[movielens_data['imdbId'].isin(movies_with_enough_ratings)]

# Check the filtered data
print(filtered_data)

# Optionally, save the filtered data to a new CSV for future use
filtered_data.to_csv('movielens_ratings_with_imdb_m_4_r_100.csv', index=False)
