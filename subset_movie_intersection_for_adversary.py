import pandas as pd

# Load the 5% user dataset and the adversary dataset
user_data = pd.read_csv('sampled_movielens_data_5percent_users.csv')
adversary_data = pd.read_csv('combined_reviews_epoch.csv')

# Step 1: Extract the unique movie IDs from the 5% dataset
subset_movies = user_data['imdbId'].unique()

# Step 2: Filter the adversary data to only include rows with movies in the subset
# Assume that 'movie_title' in adversary_data corresponds to 'imdbId' in user_data
intersected_adversary_data = adversary_data[adversary_data['movie_title'].isin(subset_movies)]

# Optional: Save the intersected data to a new CSV for analysis
intersected_adversary_data.to_csv('intersected_adversary_data_for_5_percent_mldataset.csv', index=False)

# Display result for verification
print(intersected_adversary_data)
