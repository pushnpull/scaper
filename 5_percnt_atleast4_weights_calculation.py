import pandas as pd
import numpy as np

# Load the MovieLens data (replace with your actual file path)
movielens_data = pd.read_csv('movielens_ratings_with_imdb_r_200.csv')

# Step 1: Sample 5% of unique users
unique_users = movielens_data['userId'].unique()
sample_size = int(0.05 * len(unique_users))  # Calculate 5% of unique users
sampled_users = pd.Series(unique_users).sample(n=sample_size, random_state=42).values

# Filter the dataset to only include ratings by the sampled users
sampled_data = movielens_data[movielens_data['userId'].isin(sampled_users)]

# Step 2: Filter movies with at least 4 ratings
movie_rating_counts = sampled_data['imdbId'].value_counts()
movies_with_enough_ratings = movie_rating_counts[movie_rating_counts >= 4].index
filtered_data = sampled_data[sampled_data['imdbId'].isin(movies_with_enough_ratings)]

# Step 3: Calculate weights based on support count for each movie
movie_support_counts = filtered_data['imdbId'].value_counts()
movie_weights = 1 / np.log(movie_support_counts)

# Convert weights to DataFrame for easier saving
weights_df = movie_weights.reset_index()
weights_df.columns = ['imdbId', 'weight']  # Rename columns for clarity

# Save the final filtered data and weights to CSV files
filtered_data.to_csv('5_alteast4_movielens_data.csv', index=False)
weights_df.to_csv('5_movie_weights.csv', index=False)




# Print summary
print(f"Number of unique users in sampled data: {filtered_data['userId'].nunique()}")
print(f"Number of movies with at least 4 ratings: {len(movies_with_enough_ratings)}")
print(" '5_alteast4_movielens_data.csv' and '5_movie_weights.csv'.")




adversary_data = pd.read_csv('combined_data_with_userid_r_200.csv')  # Load the adversary data

# Step 4: Filter adversary data to only include rows with movies in the filtered 5% dataset
subset_movies = filtered_data['imdbId'].unique()  # Get unique movie IDs from filtered 5% data
intersected_adversary_data = adversary_data[adversary_data['imdbId'].isin(subset_movies)]

intersected_adversary_data.to_csv('5_percent_mldataset_intersected_adversary_data_for.csv', index=False)
print(f"Number of rows in filtered adversary data: {len(intersected_adversary_data)}")
print("and intersected adversary data have been saved to CSV files.")