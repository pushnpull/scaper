For the initial steps, such as preprocessing, cleaning, and organizing the data, we will use the DataFrames

and for computationally intensive tasks, especially where the data is sparse (e.g., scoring, matching, and similarity computations in your de-anonymization algorithm), transitioning to sparse matrices is recommended for optimal performance. 





we have done till this point


import pandas as pd
import numpy as np
from tqdm import tqdm  # For progress bar

# ---------------------- Configuration ----------------------

# Paths to your CSV files
DATABASE_PATH = '5_alteast4_movielens_data.csv'
AUX_DATA_PATH = '5_percent_mldataset_intersected_adversary_data_for.csv'
WEIGHTS_PATH = '5_movie_weights.csv'



# ---------------------- Load Datasets ----------------------

print("Loading datasets...")
database = pd.read_csv(DATABASE_PATH)
aux_data = pd.read_csv(AUX_DATA_PATH)
weights_df = pd.read_csv(WEIGHTS_PATH)

# ---------------------- Preprocessing ----------------------

# Convert 'imdbId' and 'movie_title' to strings to ensure consistency
database['imdbId'] = database['imdbId'].astype(str)
aux_data['imdbId'] = aux_data['imdbId'].astype(str)  # Assuming 'movie_title' has been renamed to 'imdbId'

# Convert weights to a dictionary for fast lookup
movie_weights = dict(zip(weights_df['imdbId'].astype(str), weights_df['weight']))




movie lense data looks like this:
userId,rating,timestamp,imdbId
5,4.0,840768638,113189
5,3.0,840768897,114369
5,4.0,840768763,112573
5,3.0,840763914,112384
5,3.0,840764018,112462
5,4.0,840764183,112740
5,4.0,840764017,112864
5,3.0,840768638,113957

adversary data look like this


imdbId,review_date,rating_value,review_date_epoch,userId
10,4 March 2005,8.0,1109917800,43
10,2 January 2003,8.0,1041489000,44
10,24 April 2007,8.0,1177396200,45
10,19 October 2004,,1098167400,46
10,22 February 2022,7.0,1645511400,47
10,12 March 2012,7.0,1331533800,48
10,27 February 2003,10.0,1046327400,49
10,2 October 2022,6.0,1664692200,50
10,22 November 2002,10.0,1037946600,51
10,15 August 1999,7.0,934698600,37
10,6 September 2013,2.0,1378449000,8


Note: we will use epoch time for the calculations

i have precalculated the weights required for our algorithm(movie_weights). (1/(log(supp(i))))

that look like this

imdbId,weight
111161,0.11705146344208521
109830,0.11739854858715228
110912,0.11781951351471258
133093,0.11806526100287247
102926,0.11896843401434963
76759,0.11947972562629464


I have already gathered the intersection of mapping of 
mluser : [all_adversary_id]

using this code.

# Convert 'imdbId' to ensure consistency
database['imdbId'] = database['imdbId'].astype(str).str.lstrip('0')
aux_data['imdbId'] = aux_data['imdbId'].astype(str).str.lstrip('0')

# Rename 'userId' columns to avoid confusion
database = database.rename(columns={'userId': 'ml_userId'})
aux_data = aux_data.rename(columns={'userId': 'aux_userId'})

# ---------------------- Create imdbId to aux_userIds Mapping ----------------------

print("Creating imdbId to aux_userIds mapping...")
# Group aux_data by 'imdbId' and aggregate 'aux_userId' into sets for fast lookup
imdb_to_aux_users = aux_data.groupby('imdbId')['aux_userId'].apply(set).to_dict()
print(f"Total unique movies in aux data: {len(imdb_to_aux_users)}")

# ---------------------- Get Unique ml_userIds ----------------------

ml_user_ids = database['ml_userId'].unique()
total_ml_users = len(ml_user_ids)
print(f"Total unique MovieLens users: {total_ml_users}")

# ---------------------- Processing Function ----------------------

def process_ml_users_in_batches(database, ml_user_ids, output_dir, exclude_top_n=0):
    """
    Process MovieLens users in batches, finding auxiliary users who have rated at least
    one common movie, and save the results to CSV files.
    """
    # Optionally filter out the top N most popular movies
    if exclude_top_n > 0:
        print(f"Excluding top {exclude_top_n} most popular movies...")
        top_movies = database['imdbId'].value_counts().head(exclude_top_n).index
        database = database[~database['imdbId'].isin(top_movies)]

    # Iterate over batches of MovieLens users
    for i in range(0, len(ml_user_ids), BATCH_SIZE):
        batch_ml_user_ids = ml_user_ids[i:i + BATCH_SIZE]
        batch_start = i
        batch_end = i + len(batch_ml_user_ids) - 1
        print(f"\nProcessing MovieLens users {batch_start} to {batch_end} (Batch size: {len(batch_ml_user_ids)})")

        # Filter database for the current batch of ml users
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
        output_filepath = os.path.join(output_dir, output_filename)

        # Save the batch results to CSV
        batch_results_df.to_csv(output_filepath, index=False)
        print(f"Saved results for MovieLens users {batch_start} to {batch_end} to '{output_filepath}'")

        # Clear results to free memory
        del results, batch_results_df, ml_batch, ml_user_imdbs




i have made sure the adversay data contains only imdbID that are present in our ml subset.
so when we will make sparce matrix. for both dataset's sparse matrix where rows are users and columns are movie .(columns of both matrices corresponds to same imdbID) 



To create sparse matrices for both datasets where rows represent users and columns represent movies (`imdbId`), here's the step-by-step outline:



### Step-by-Step Outline to Create Sparse Matrices

0. {row_index: ml_userId} and {row_index: aux_userId} mappings should be used to easily convert matrix row indices back to user IDs during or after computations.

1. **Create a Set of Unique `imdbId` Values:**
   - Combine all unique `imdbId` values from both `database` and `aux_data`.
   - Ensure columns of both matrices align with this combined set of `imdbId` values.

2. **Map `imdbId` Values to Column Indices:**
   - Create a mapping (dictionary) where each unique `imdbId` is assigned a column index.

3. **Prepare Data for the `database` Sparse Matrix:**
   - Initialize a sparse matrix where rows correspond to unique `ml_userId` values and columns correspond to `imdbId`.
   - Populate the matrix with user ratings from `database` using the column index mapping.

4. **Prepare Data for the `aux_data` Sparse Matrix:**
   - Similarly, create a sparse matrix where rows represent `aux_userId` and columns are the same as the `database` matrix.
   - Populate the matrix using the adversary's known ratings data.

Consider initializing sparse matrices using scipy.sparse.csr_matrix or coo_matrix directly, which allows efficient construction and modification.


Sparse Matrices Setup

    Ratings Matrices:
        Rows: Users, Columns: Movies (imdbId)
        Values: User's ratings for each movie.

    Timestamps Matrices:
        Rows: Users, Columns: Movies (imdbId)
        Values: Timestamps (in epoch time) when each movie was rated.

Matrix Summary:

    database: Ratings matrix & Timestamps matrix.
    aux_data: Ratings matrix & Timestamps matrix.


so we can do vectorized operations for scoreboardRH algorithm