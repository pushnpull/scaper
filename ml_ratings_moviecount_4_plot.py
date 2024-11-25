import pandas as pd
import matplotlib.pyplot as plt

# Load the filtered MovieLens data
filtered_data = pd.read_csv('movielens_ratings_with_imdb_4ormore.csv')

# Step 1: Count the number of ratings for each movie
movie_rating_counts = filtered_data['imdbId'].value_counts()

# Step 2: Count the frequency of each rating count (e.g., how many movies are rated exactly 4 times, 5 times, etc.)
rating_count_frequencies = movie_rating_counts.value_counts().sort_index()

# Step 3: Plot the distribution on a log scale
plt.figure(figsize=(10, 6))
plt.plot(rating_count_frequencies.index, rating_count_frequencies.values, marker='o', linestyle='-')
plt.xscale('log')  # Set x-axis to log scale
plt.yscale('log')  # Set y-axis to log scale
plt.xlabel('Number of Ratings (log scale)')
plt.ylabel('Number of Movies (log scale)')
plt.title('Distribution of Movies by Rating Counts on Log Scale')
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
output_plot_path = '/mnt/c/Users/abhay/Desktop/scaper/movielens_ratings_with_imdb_4ormore.png'
plt.savefig(output_plot_path)
plt.show()
