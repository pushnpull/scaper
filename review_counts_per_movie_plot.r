import pandas as pd
import matplotlib.pyplot as plt

# Path to the review counts file
review_counts_file = '/mnt/c/Users/abhay/Desktop/scaper/review_counts_per_movie.csv'

# Read the CSV file containing the number of reviews per movie
df = pd.read_csv(review_counts_file)

# Plotting the distribution of review counts per movie
plt.figure(figsize=(10, 6))
plt.hist(df['review_count'], bins=50, color='skyblue', edgecolor='black')

# Add labels and title
plt.xlabel('Number of Reviews per Movie', fontsize=14)
plt.ylabel('Number of Movies', fontsize=14)
plt.title('Distribution of Review Counts per Movie', fontsize=16)

# Show gridlines for better visualization
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Save the plot as an image file
output_plot_path = '/mnt/c/Users/abhay/Desktop/scaper/review_counts_per_movie_distribution.png'
plt.tight_layout()
plt.savefig(output_plot_path)

# Show the plot
plt.show()

# Print the path where the plot is saved
print(f"Plot saved to {output_plot_path}")
