import pandas as pd
import matplotlib.pyplot as plt

# Path to the combined reviews file
combined_reviews_file = '/mnt/c/Users/abhay/Desktop/scaper/combined_reviews.csv'

# Read the combined reviews CSV
df = pd.read_csv(combined_reviews_file)

# Group by reviewer_name and count how many unique movies each reviewer reviewed
reviewer_movie_counts = df.groupby('reviewer_name')['movie_title'].nunique()

# Count how many reviewers reviewed 1 movie, 2 movies, etc.
reviewer_count_by_movie_count = reviewer_movie_counts.value_counts().sort_index()[:100]

# Print the counts for each group
print(reviewer_count_by_movie_count)

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(reviewer_count_by_movie_count.index, reviewer_count_by_movie_count.values, color='skyblue',log=True)

# Add labels and title
plt.xlabel('Number of Movies Reviewed', fontsize=14)
plt.ylabel('Number of Reviewers', fontsize=14)
plt.title('Distribution of Reviewers by Number of Movies Reviewed', fontsize=16)

# Show gridlines for better visualization
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.savefig('reviewer_frequency_plot_log.png')
plt.show()

# Save the reviewer count by movie count data for further analysis if needed
reviewer_count_by_movie_count.to_csv('/mnt/c/Users/abhay/Desktop/scaper/reviewer_movie_count_distribution.csv')
