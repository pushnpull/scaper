import pandas as pd
import matplotlib.pyplot as plt
# Load the CSV into a DataFrame (assuming your CSV is named 'ratings.csv')
df = pd.read_csv('combined_data_with_userid.csv')



# Count the number of ratings per user
user_rating_counts = df['userId'].value_counts()

# Create a list of users who have rated more than 100 times
users_with_many_ratings = user_rating_counts[user_rating_counts > 200].index

# Filter out rows for those users
filtered_df = df[~df['userId'].isin(users_with_many_ratings)]

# Reindexing the DataFrame
filtered_df.reset_index(drop=True, inplace=True)

# Display the filtered DataFrame
print(filtered_df)

# If you want to save it to a new CSV (optional)
filtered_df.to_csv("combined_data_with_userid_r_200.csv", index=False)


# Load the CSV into a DataFrame (assuming your CSV is named 'ratings.csv')
df = pd.read_csv('movielens_ratings_with_imdb.csv')





# Count the number of ratings per user
user_rating_counts = df['userId'].value_counts()

# Create a list of users who have rated more than 100 times
users_with_many_ratings = user_rating_counts[user_rating_counts > 200].index

# Filter out rows for those users
filtered_df = df[~df['userId'].isin(users_with_many_ratings)]

# Reindexing the DataFrame
filtered_df.reset_index(drop=True, inplace=True)

# Display the filtered DataFrame
print(filtered_df)

# If you want to save it to a new CSV (optional)
filtered_df.to_csv("movielens_ratings_with_imdb_r_200.csv", index=False)