import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset (replace with your data loading step)
# Assuming 'movie_id' is the ID of the movie and 'user_id' is the ID of the user who rated the movie
df = pd.read_csv('5_alteast4_movielens_data.csv')

# Count number of ratings per movie to determine popularity rank
movie_counts = df['imdbId'].value_counts().reset_index()
movie_counts.columns = ['imdbId', 'num_ratings']
movie_counts = movie_counts.sort_values(by='num_ratings', ascending=False).reset_index(drop=True)
movie_counts['rank'] = movie_counts.index + 1  # Rank movies by their popularity (number of ratings)

# Calculate entropy for each movie
def calculate_entropy(num_ratings):
    if num_ratings == 0:
        return 0
    p = 1 / num_ratings  # Probability for a uniform distribution (since we don't have user-specific probabilities)
    entropy = - p * np.log2(p)  # Shannon entropy
    return entropy

movie_counts['entropy'] = movie_counts['num_ratings'].apply(calculate_entropy)

# Plot the entropy by movie rank
plt.figure(figsize=(10, 6))
plt.plot(movie_counts['rank'], movie_counts['entropy'], marker='o', linestyle='-', markersize=4, label='Entropy by Movie Rank')
plt.xlabel('Movie Rank (by number of ratings)')
plt.ylabel('Entropy (bits)')
plt.title('Entropy of Movie by Rank')
plt.xscale('log')  # Optional: Use log scale for better visualization if data is skewed
# plt.yscale('log')  # Optional: Use log scale for entropy if desired
plt.grid(True)
plt.legend()
plt.show()
