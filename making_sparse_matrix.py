import pandas as pd
from scipy.sparse import coo_matrix, save_npz

# Load IMDb data
imdb_data = pd.read_csv("imdb_data.csv")  # Replace with your actual file path
imdb_data = imdb_data.dropna(subset=['rating_value', 'review_date_epoch'])  # Drop rows with missing ratings or dates

# Encode 'reviewer_name' and 'movie_title' as categorical for efficient mapping to integer indices
imdb_data['reviewer_name'] = imdb_data['reviewer_name'].astype('category')
imdb_data['movie_title'] = imdb_data['movie_title'].astype('category')

# Convert reviewer_name and movie_title to integer codes for row and column indices
imdb_row_indices = imdb_data['reviewer_name'].cat.codes
imdb_col_indices = imdb_data['movie_title'].cat.codes
imdb_ratings = imdb_data['rating_value'].astype(float)
imdb_timestamps = imdb_data['review_date_epoch'].astype(float)

# Create separate sparse matrices for ratings and timestamps
imdb_sparse_matrix_ratings = coo_matrix((imdb_ratings, (imdb_row_indices, imdb_col_indices)),
                                        shape=(imdb_data['reviewer_name'].nunique(), imdb_data['movie_title'].nunique()))

imdb_sparse_matrix_timestamps = coo_matrix((imdb_timestamps, (imdb_row_indices, imdb_col_indices)),
                                           shape=(imdb_data['reviewer_name'].nunique(), imdb_data['movie_title'].nunique()))

# Optionally save the sparse matrices
save_npz("imdb_sparse_matrix_ratings.npz", imdb_sparse_matrix_ratings)
save_npz("imdb_sparse_matrix_timestamps.npz", imdb_sparse_matrix_timestamps)


# Load MovieLens data
movielens_data = pd.read_csv("movielens_data.csv")  # Replace with your actual file path

# Convert userId and movieId to categorical for efficient mapping to integer indices
movielens_data['userId'] = movielens_data['userId'].astype('category')
movielens_data['movieId'] = movielens_data['movieId'].astype('category')

# Convert userId and movieId to integer codes for row and column indices
movielens_row_indices = movielens_data['userId'].cat.codes
movielens_col_indices = movielens_data['movieId'].cat.codes
movielens_ratings = movielens_data['rating'].astype(float)
movielens_timestamps = movielens_data['timestamp'].astype(float)

# Create separate sparse matrices for ratings and timestamps
movielens_sparse_matrix_ratings = coo_matrix((movielens_ratings, (movielens_row_indices, movielens_col_indices)),
                                             shape=(movielens_data['userId'].nunique(), movielens_data['movieId'].nunique()))

movielens_sparse_matrix_timestamps = coo_matrix((movielens_timestamps, (movielens_row_indices, movielens_col_indices)),
                                                shape=(movielens_data['userId'].nunique(), movielens_data['movieId'].nunique()))

# Optionally save the sparse matrices
save_npz("movielens_sparse_matrix_ratings.npz", movielens_sparse_matrix_ratings)
save_npz("movielens_sparse_matrix_timestamps.npz", movielens_sparse_matrix_timestamps)

print("Separate sparse matrices for ratings and timestamps created and saved for IMDb and MovieLens datasets.")
