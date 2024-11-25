import pandas as pd

# Load the MovieLens data (replace 'movielens_data.csv' with your actual file path)
movielens_data = pd.read_csv('movielens_ratings_with_imdb_4ormore.csv')

# Step 1: Get a list of unique user IDs
unique_users = movielens_data['userId'].unique()

# Step 2: Randomly sample 5% of the users
sample_size = int(0.05 * len(unique_users))  # 5% of unique users
sampled_users = pd.Series(unique_users).sample(n=sample_size, random_state=42).values

# Step 3: Filter the dataset to include only the ratings by the sampled users
sampled_data = movielens_data[movielens_data['userId'].isin(sampled_users)]

# Optional: Save the sampled data to a new CSV file
sampled_data.to_csv('sampled_movielens_data_5percent_users.csv', index=False)

# Check the number of unique users and rows in the sampled data
print(f"Number of unique users in sampled data: {sampled_data['userId'].nunique()}")
print(f"Number of ratings in sampled data: {len(sampled_data)}")
