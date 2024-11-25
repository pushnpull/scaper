import pandas as pd
import numpy as np
from tqdm import tqdm  # For progress bar

# ---------------------- Configuration ----------------------

# Paths to your CSV files
DATABASE_PATH = '5_alteast4_movielens_data.csv'
AUX_DATA_PATH = '5_percent_mldataset_intersected_adversary_data_for.csv'
WEIGHTS_PATH = '5_movie_weights.csv'

# Output path for results
OUTPUT_DIR = 'deanonymization_results'  # Ensure this directory exists or create it

# Batch size for processing ml users
BATCH_SIZE = 1000

# ---------------------- Load Datasets ----------------------

print("Loading datasets...")
database = pd.read_csv(DATABASE_PATH)
aux_data = pd.read_csv(AUX_DATA_PATH)
weights_df = pd.read_csv(WEIGHTS_PATH)

# ---------------------- Preprocessing ----------------------

# Convert 'imdbId' and 'movie_title' to strings to ensure consistency
database['imdbId'] = database['imdbId'].astype(str)
aux_data['imdbId'] = aux_data['imdbId'].astype(str)  # Assuming 'movie_title' has been renamed to 'imdbId'

# Rename 'userId' columns to avoid confusion
database = database.rename(columns={'userId': 'ml_userId'})
aux_data = aux_data.rename(columns={'userId': 'aux_userId'})

# Convert weights to a dictionary for fast lookup
movie_weights = dict(zip(weights_df['imdbId'].astype(str), weights_df['weight']))

# ---------------------- Create imdbId to aux_userIds Mapping ----------------------

print("Creating imdbId to aux_userIds mapping...")
# Group aux_data by 'imdbId' and aggregate 'aux_userId' into sets for fast lookup
imdb_to_aux_users = aux_data.groupby('imdbId')['aux_userId'].apply(set).to_dict()
print(f"Total unique movies in aux data: {len(imdb_to_aux_users)}")

# ---------------------- Get Unique ml_userIds ----------------------

ml_user_ids = database['ml_userId'].unique()
total_ml_users = len(ml_user_ids)
print(f"Total unique MovieLens users: {total_ml_users}")

# ---------------------- Ensure Output Directory Exists ----------------------

import os

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created output directory at '{OUTPUT_DIR}'.")

# ---------------------- Processing ml Users in Batches ----------------------

print("Starting batch processing of MovieLens users...")

for i in range(0, len(ml_user_ids), BATCH_SIZE):
    batch_ml_user_ids = ml_user_ids[i:i+BATCH_SIZE]
    batch_start = i
    batch_end = i + len(batch_ml_user_ids) - 1
    print(f"\nProcessing MovieLens users {batch_start} to {batch_end} (Batch size: {len(batch_ml_user_ids)})")
    
    # Filter database for current batch of ml users
    ml_batch = database[database['ml_userId'].isin(batch_ml_user_ids)]
    
    # Group by 'ml_userId' and aggregate 'imdbId' into sets
    ml_user_imdbs = ml_batch.groupby('ml_userId')['imdbId'].apply(set).to_dict()
    
    # Initialize results list
    results = []
    
    # Iterate over each ml user in the batch
    for ml_user_id, ml_imdb_set in tqdm(ml_user_imdbs.items(), desc="Processing ml users in batch"):
        # Initialize a set to collect aux_userIds who have rated at least one common movie
        aux_users_set = set()
        
        # Iterate over each imdbId rated by the ml user
        for imdb_id in ml_imdb_set:
            # Get aux_userIds who have rated this imdbId
            aux_user_ids = imdb_to_aux_users.get(imdb_id, set())
            aux_users_set.update(aux_user_ids)
        
        # Convert aux_users_set to a sorted list for consistency
        aux_users_list = sorted(aux_users_set)
        
        # Store the result
        results.append({
            'ml_user_id': ml_user_id,
            'aux_users': ','.join(map(str, aux_users_list))  # Convert list to comma-separated string
        })
    
    # Create a DataFrame from the results
    batch_results_df = pd.DataFrame(results)
    
    # Define the output filename
    output_filename = f'batch_{batch_start}_to_{batch_end}_results.csv'
    output_filepath = os.path.join(OUTPUT_DIR, output_filename)
    
    # Save the batch results to CSV
    batch_results_df.to_csv(output_filepath, index=False)
    print(f"Saved results for MovieLens users {batch_start} to {batch_end} to '{output_filepath}'")
    
    # Clear results to free memory
    del results, batch_results_df, ml_batch, ml_user_imdbs

print("\nBatch processing completed successfully.")



