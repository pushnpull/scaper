import pandas as pd
from datetime import datetime

# Path to the combined reviews CSV file
input_file = '/mnt/c/Users/abhay/Desktop/scaper/combined_reviews.csv'
output_file_with_epoch = 'combined_reviews_epoch.csv'

# Function to convert date string to epoch time (12:00 PM as time)
def convert_to_epoch(date_str):
    try:
        # Parse the date string and assume time is 12:00 PM (noon)
        date_obj = datetime.strptime(date_str, '%d %B %Y')
        date_obj = date_obj.replace(hour=12, minute=0, second=0)  # Set time to 12:00 PM
        return int(date_obj.timestamp())  # Return epoch time
    except:
        return None  # Return None if date is invalid or missing

# Read the input CSV file
df = pd.read_csv(input_file)

# Convert the review_date column to epoch time
df['review_date_epoch'] = df['review_date'].apply(convert_to_epoch)

# Save the DataFrame with the epoch time to a new CSV file
df.to_csv(output_file_with_epoch, index=False)

print("New file with epoch times saved successfully.")
