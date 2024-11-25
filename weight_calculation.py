import pandas as pd
import numpy as np

# Load the filtered MovieLens data (after filtering for movies rated at least four times)
movielens_data = pd.read_csv('movielens_ratings_with_imdb_4ormore.csv')

# Step 1: Calculate the support (number of ratings) for each movie (imdbId)
movie_support_counts = movielens_data['imdbId'].value_counts()

# Step 2: Compute the weight for each movie based on its support count
movie_weights = 1 / np.log(movie_support_counts)

# Step 3: Convert to DataFrame for easier saving
weights_df = movie_weights.reset_index()
weights_df.columns = ['imdbId', 'weight']  # Rename columns for clarity

# Step 4: Save the weights to a CSV file
weights_df.to_csv('ml_movie_weights.csv', index=False)

print("Movie weights calculated and saved to 'movie_weights.csv'.")
