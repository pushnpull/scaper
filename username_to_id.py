import pandas as pd
import os

def create_reviewer_userid_mapping(aux_data_path, mapping_output_path, modified_aux_output_path):
    """
    Creates a unique userId mapping for each reviewer_name in the auxiliary dataset
    and replaces the reviewer_name with userId in the auxiliary dataset.

    Parameters:
    - aux_data_path (str): Path to the original auxiliary data CSV file.
    - mapping_output_path (str): Path to save the reviewer_name to userId mapping CSV.
    - modified_aux_output_path (str): Path to save the modified auxiliary data with userId.
    """
    
    # ---------------------- Load Auxiliary Data ----------------------
    print("Loading auxiliary data...")
    try:
        aux_data = pd.read_csv(aux_data_path)
    except FileNotFoundError:
        print(f"Error: The file {aux_data_path} was not found.")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: The file {aux_data_path} is empty.")
        return
    except Exception as e:
        print(f"An error occurred while reading {aux_data_path}: {e}")
        return
    
    # ---------------------- Validate Columns ----------------------
    required_columns = {'reviewer_name', 'movie_title', 'review_date', 'rating_value', 'review_date_epoch'}
    if not required_columns.issubset(aux_data.columns):
        missing = required_columns - set(aux_data.columns)
        print(f"Error: The following required columns are missing in {aux_data_path}: {missing}")
        return
    
    # ---------------------- Handle Missing Reviewer Names ----------------------
    print("Handling missing reviewer names...")
    initial_row_count = len(aux_data)
    aux_data = aux_data.dropna(subset=['reviewer_name'])
    final_row_count = len(aux_data)
    dropped_rows = initial_row_count - final_row_count
    if dropped_rows > 0:
        print(f"Dropped {dropped_rows} rows due to missing 'reviewer_name'.")
    
    # ---------------------- Create Unique userId Mapping ----------------------
    print("Creating userId mapping...")
    unique_reviewers = aux_data['reviewer_name'].unique()
    user_ids = pd.Series(range(1, len(unique_reviewers) + 1), index=unique_reviewers)
    
    # Create a DataFrame for the mapping
    mapping_df = user_ids.reset_index()
    mapping_df.columns = ['reviewer_name', 'userId']
    
    # ---------------------- Save the Mapping to CSV ----------------------
    print(f"Saving reviewer_name to userId mapping to {mapping_output_path}...")
    try:
        mapping_df.to_csv(mapping_output_path, index=False)
    except Exception as e:
        print(f"An error occurred while saving {mapping_output_path}: {e}")
        return
    
    # ---------------------- Replace reviewer_name with userId in Auxiliary Data ----------------------
    print("Replacing reviewer_name with userId in auxiliary data...")
    aux_data = aux_data.merge(mapping_df, on='reviewer_name', how='left')
    
    # Verify if any reviewer_name was not mapped
    unmapped = aux_data['userId'].isna().sum()
    if unmapped > 0:
        print(f"Warning: {unmapped} rows have reviewer_names that were not mapped to userIds.")
        # Optionally, drop these rows
        aux_data = aux_data.dropna(subset=['userId'])
        aux_data['userId'] = aux_data['userId'].astype(int)
    # ---------------------- Remove reviewer_name Column ----------------------
    print("Removing 'reviewer_name' column from auxiliary data...")
    aux_data = aux_data.drop(columns=['reviewer_name'])
    
    # ---------------------- Save the Modified Auxiliary Data to CSV ----------------------
    print(f"Saving modified auxiliary data with userId to {modified_aux_output_path}...")
    try:
        aux_data.to_csv(modified_aux_output_path, index=False)
    except Exception as e:
        print(f"An error occurred while saving {modified_aux_output_path}: {e}")
        return
    
    print("Mapping and replacement completed successfully.")

if __name__ == "__main__":
    # ---------------------- Configuration ----------------------
    
    # Define file paths
    AUX_DATA_PATH = 'combined_reviews_epoch.csv'
    MAPPING_OUTPUT_PATH = 'reviewer_name_to_userid.csv'
    MODIFIED_AUX_OUTPUT_PATH = 'combined_data_with_userid.csv'
    
    # Check if input file exists
    if not os.path.isfile(AUX_DATA_PATH):
        print(f"Error: The auxiliary data file '{AUX_DATA_PATH}' does not exist.")
    else:
        # Run the mapping function
        create_reviewer_userid_mapping(AUX_DATA_PATH, MAPPING_OUTPUT_PATH, MODIFIED_AUX_OUTPUT_PATH)
