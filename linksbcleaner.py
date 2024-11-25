import os
import pandas as pd

# Path to the folder containing review files
reviews_folder = "/mnt/c/Users/abhay/Desktop/scaper/reviewsb/"

# List all review files in the folder
review_files = [f for f in os.listdir(reviews_folder) if f.startswith("reviews_tt") and f.endswith(".csv")]

# Extract the IMDb IDs from the filenames
completed_imdb_ids = [int(f.split('_tt')[1].split('.csv')[0]) for f in review_files]

# Load the linksb.csv file
linksb_path = '/mnt/c/Users/abhay/Desktop/scaper/ml-32m/linksb.csv'
linksb_df = pd.read_csv(linksb_path)

# Filter out the completed movies by matching the imdbId
remaining_links_df = linksb_df[~linksb_df['imdbId'].isin(completed_imdb_ids)]

# Save the filtered DataFrame back to a CSV file
remaining_links_path = '/mnt/c/Users/abhay/Desktop/scaper/ml-32m/remaining_linksb.csv'
remaining_links_df.to_csv(remaining_links_path, index=False)

print(f"Remaining links saved to {remaining_links_path}")
