import os
import pandas as pd

# Path to the folder containing review files
reviews_folder = "/mnt/c/Users/abhay/Desktop/scaper/reviewsb/"
# Path to the combined output file
output_file = '/mnt/c/Users/abhay/Desktop/scaper/combined_reviews.csv'
# Path to save the review counts per movie
review_counts_file = '/mnt/c/Users/abhay/Desktop/scaper/review_counts_per_movie.csv'

# Initialize the CSV file with headers if it doesn't exist
if not os.path.exists(output_file):
    with open(output_file, 'w') as f:
        f.write('reviewer_name,movie_title,review_date,rating_value\n')

# Function to get the movie title from IMDb ID (you can modify to match to your linksb.csv if needed)
def get_movie_title_from_imdb_id(imdb_id):
    return imdb_id  # As an example, returning IMDb ID. Replace this with actual movie title lookup if needed.

# Dictionary to store the count of reviews per movie title
review_counts = {}

# Movie counter
movie_count = 0

# Iterate over all review files in the folder
for file in os.listdir(reviews_folder):
    if file.startswith("reviews_tt") and file.endswith(".csv"):
        try:
            # Extract IMDb ID from the filename
            imdb_id = file.split('_tt')[1].split('.csv')[0]
            movie_title = get_movie_title_from_imdb_id(imdb_id)
            
            # Read the review file
            file_path = os.path.join(reviews_folder, file)
            df = pd.read_csv(file_path, usecols=['reviewer_name', 'review_date', 'rating_value'])
            
            # Add movie title to the DataFrame
            df['movie_title'] = movie_title

            # Reorder columns
            df = df[['reviewer_name', 'movie_title', 'review_date', 'rating_value']]
            
            # Append the data to the combined CSV file
            df.to_csv(output_file, mode='a', header=False, index=False)
            
            # Count the number of reviews for this movie
            review_count = len(df)
            review_counts[movie_title] = review_count
            
            # Increment movie count
            movie_count += 1
            
            # Print progress
            print(f"Processed movie {movie_count}: {movie_title} with {review_count} reviews")

        except Exception as e:
            print(f"Error processing file {file}: {e}")

# Save the review counts dictionary to a CSV file for plotting later
review_counts_df = pd.DataFrame(list(review_counts.items()), columns=['movie_title', 'review_count'])
review_counts_df.to_csv(review_counts_file, index=False)

print(f"Processing complete. Total movies processed: {movie_count}")
print(f"Review counts per movie saved to: {review_counts_file}")
