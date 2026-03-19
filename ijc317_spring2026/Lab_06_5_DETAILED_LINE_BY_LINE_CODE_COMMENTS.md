# Complete Line-by-Line Code Comments
## IJC317 Week 6 Lab Worksheet - Every Line Explained

---

## CODE CELL 1: IMPORTS (Setup)

```python
# Import pandas library for data manipulation and analysis
# Used for: loading CSV files, creating dataframes, groupby, pivot, merge operations
import pandas as pd

# Import numpy for numerical computing and array operations
# Used for: mathematical operations, random numbers, dot products, array manipulation
import numpy as np

# Import train_test_split function from sklearn
# Used to: split dataset into training (80%) and testing (20%) sets
from sklearn.model_selection import train_test_split

# Import mean_squared_error function from sklearn
# Used to: calculate RMSE (Root Mean Square Error) for model evaluation
from sklearn.metrics import mean_squared_error
```

---

## CODE CELL 2: EXERCISE 1 - POPULARITY-BASED RECOMMENDER

### 2.1: Load the Data

```python
# ============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================================

# Define URL where the MovieLens 100K ratings data is hosted
# This is the ratings file: ~100,000 user-movie ratings
url = "https://files.grouplens.org/datasets/movielens/ml-100k/u.data"

# Create list of column names for the ratings file
# The actual file doesn't have a header row, so we specify column names
# user_id: which user gave the rating (range: 1-943)
# item_id: which movie was rated (range: 1-1682)
# rating: the rating value (range: 1-5 stars)
# timestamp: when the rating was given (Unix timestamp)
columns = ['user_id', 'item_id', 'rating', 'timestamp']

# Read the CSV file from the URL
# sep='\t' means the data is separated by TAB characters (not commas)
# names=columns assigns our column names to the dataframe
# This creates a pandas DataFrame with ~100,000 rows
df = pd.read_csv(url, sep='\t', names=columns)

# Define URL for the movie metadata file
# This file contains information ABOUT each movie (title, genres, etc)
items_url = "https://files.grouplens.org/datasets/movielens/ml-100k/u.item"

# Create list of column names for the movie metadata file
# item_id: the movie ID (same as in ratings file)
# movie_title: the name of the movie
# release_date: when the movie was released
# video_release_date: when it was released on video (often blank)
# IMDb_URL: link to IMDb page for the movie
# genre_0 through genre_18: 19 boolean flags (0 or 1) for each genre
#   (Drama, Action, Comedy, etc.)
item_columns = ['item_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL'] + [f'genre_{i}' for i in range(19)]

# Read the movie metadata CSV file
# sep='|' means the data is separated by pipe characters
# names=item_columns assigns our column names
# encoding='latin-1' is needed because the file uses this character encoding
#   (some special characters in movie titles need this encoding)
movies_metadata = pd.read_csv(items_url, sep='|', names=item_columns, encoding='latin-1')

# ============================================================================
# STEP 2: SPLIT DATA INTO TRAINING AND TESTING SETS
# ============================================================================

# Split the ratings dataframe into two parts:
# - train_df: 80% of the data (used to build/train the model)
# - test_df: 20% of the data (used to evaluate the model)
# test_size=0.2 means 20% goes to test, 80% goes to training
# random_state=42 ensures we get the same split every time we run the code
#   (reproducibility - important for debugging and comparison)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# ============================================================================
# SECTION 1: BUILDING THE "MODEL"
# ============================================================================
# This section builds a popularity-based recommender system
# The "model" is just: calculate average rating for each movie

# Calculate the global mean rating
# train_df['rating'] extracts the 'rating' column as a pandas Series
# .mean() calculates the average of all ratings
# Example result: ~3.72 (movies are rated between 1-5 on average)
mu = train_df['rating'].mean()

# Print the global mean to see what it is
# f-string formats the number to 4 decimal places
# \n adds a newline for readability
print(f"mu: {mu:.4f}\n")

# Define minimum vote count threshold
# We won't recommend movies that have fewer than 25 ratings
# Why? Because a movie with 1 rating of 5 stars looks "best" but unreliable
# We want at least 25 ratings to trust the average
m = 25

# ============================================================================
# SOLUTION 1: CREATE MOVIE_STATS DATAFRAME
# ============================================================================
# TODO: Create a new df called `movie_stats` that contains each movie's mean rating and rating count
# Hint: Group by 'item_id'!

# SOLUTION:
# Group the training data by item_id (each movie)
# Then aggregate to get statistics for each movie
movie_stats = train_df.groupby('item_id')['rating'].agg(['mean', 'count'])

# Rename the columns for clarity
# 'mean' → 'mean_rating' (the average rating of this movie)
# 'count' → 'vote_count' (how many people rated this movie)
movie_stats.columns = ['mean_rating', 'vote_count']

# ============================================================================
# SOLUTION 2: IMPLEMENT BAYESIAN AVERAGING
# ============================================================================

# Define a function to calculate the Bayesian weighted average
# This is the formula IMDB uses for its ratings
# Parameters:
#   x: one row of movie_stats (contains 'mean_rating' and 'vote_count')
#   m: minimum vote count threshold (default 25)
#   mu: global mean rating (default is the mu calculated above)
def weight_rating(x, m=m, mu=mu):
    # Get the number of votes for this movie
    # x is a pandas Series, x['vote_count'] extracts that value
    n_i = x['vote_count']
    
    # Get the mean rating for this movie
    # This is the average of all ratings this movie received
    mu_i = x['mean_rating']
    
    # TODO: Return the IMDB Bayesian rating
    # Formula: W_i = [n_i / (n_i + m)] * μ_i + [m / (n_i + m)] * μ
    # 
    # Explanation:
    # [n_i / (n_i + m)] = weight for the movie's actual mean
    #   - If n_i=0: weight=0 (no ratings, don't trust the mean)
    #   - If n_i=∞: weight≈1 (many ratings, fully trust the mean)
    # [m / (n_i + m)] = weight for the global mean
    #   - If n_i=0: weight=1 (no ratings, use global mean)
    #   - If n_i=∞: weight≈0 (many ratings, don't need global mean)
    #
    # This prevents overfitting: a movie with 1 five-star rating
    # won't be rated higher than movies with 100 four-star ratings
    
    # SOLUTION:
    return (n_i / (n_i + m)) * mu_i + (m / (n_i + m)) * mu

# ============================================================================
# CONTINUE: FILTER AND PREPARE RECOMMENDATIONS
# ============================================================================

# Filter movie_stats to only include movies with at least m=25 ratings
# The expression movie_stats['vote_count'] >= m returns a boolean Series
#   (True for rows where vote_count ≥ 25, False otherwise)
# Passing this to [] filters the dataframe to only keep True rows
# .copy() creates a new independent copy (good practice)
qualified_movies = movie_stats[movie_stats['vote_count'] >= m].copy()

# Create a new column 'weighted_score' for each qualified movie
# .apply(weight_rating, axis=1) applies the weight_rating function to each row
# axis=1 means apply to each row (axis=0 would be each column)
# This calculates the Bayesian average for each movie
qualified_movies['weighted_score'] = qualified_movies.apply(weight_rating, axis=1)

# Merge movie titles with our statistics
# pd.merge combines two dataframes based on a common column
# qualified_movies has: item_id, mean_rating, vote_count, weighted_score
# movies_metadata has: item_id, movie_title, release_date, etc
# on='item_id' means join where item_id matches in both dataframes
# Result: qualified_movies now also has the movie_title column
qualified_with_names = pd.merge(qualified_movies, movies_metadata[['item_id', 'movie_title']], on='item_id')

# ============================================================================
# SECTION 2: MODEL ANALYSES
# ============================================================================
# This section analyzes the model by showing the top 10 recommendations

# TODO: Sort qualified_movies by 'weighted_score' and print the Top 10

# SOLUTION:
# Sort the qualified movies by weighted_score in descending order
# .nlargest(10, 'weighted_score') gets the 10 rows with highest scores
# [['movie_title', 'weighted_score', 'vote_count']] selects only these columns
top_10 = qualified_with_names.nlargest(10, 'weighted_score')[['movie_title', 'weighted_score', 'vote_count']]

# Print the top 10 movies
# This shows us: what movies does our model recommend most?
print("Top 10 Recommended Movies:\n", top_10)

# ============================================================================
# SECTION 3: EVALUATION
# ============================================================================
# This section evaluates the model on the test set
# We check: how well does our model predict actual ratings?

# Merge test set with our model's predictions
# test_df has: user_id, item_id, rating (actual ratings from test set)
# qualified_movies has: item_id, mean_rating, weighted_score
# how='left' keeps all rows from test_df, matching with qualified_movies where possible
# If a test movie isn't in qualified_movies, the merged columns will be NaN
comparison_df = test_df.merge(qualified_movies[['mean_rating', 'weighted_score']], on='item_id', how='left')

# Fill NaN values in 'mean_rating' column with the global mean (mu)
# Why? For movies not in our qualified list, we predict the global average
# This is a fallback prediction when we have no specific information
comparison_df['mean_rating'] = comparison_df['mean_rating'].fillna(mu)

# Fill NaN values in 'weighted_score' column with the global mean (mu)
# Same reasoning as above
comparison_df['weighted_score'] = comparison_df['weighted_score'].fillna(mu)

# Calculate RMSE for the simple average approach
# mean_squared_error(actual, predicted) calculates: mean((actual - predicted)^2)
# np.sqrt(...) takes the square root to get RMSE
# comparison_df['rating'] = actual ratings from test set
# comparison_df['mean_rating'] = our predictions using simple mean
rmse_simple = np.sqrt(mean_squared_error(comparison_df['rating'], comparison_df['mean_rating']))

# Calculate RMSE for the weighted average approach
# comparison_df['weighted_score'] = our predictions using Bayesian average
# Same metric as above, but with better predictions
rmse_weighted = np.sqrt(mean_squared_error(comparison_df['rating'], comparison_df['weighted_score']))

# Print evaluation results
print(f"\n\n--- Evaluation Results ---")
print(f"Simple Average RMSE:   {rmse_simple:.4f}")    # Expected: ~0.97
print(f"Weighted Average RMSE: {rmse_weighted:.4f}")  # Expected: ~0.94
```

---

## CODE CELL 5: EXERCISE 2 - COLLABORATIVE FILTERING

```python
# ============================================================================
# STEP 1: CREATE USER-ITEM MATRIX
# ============================================================================

# TODO: Pivot the training dataframe to create a matrix
# (rows="index"=users, columns=movies, values=ratings)
# and fill missing values with 0

# SOLUTION:
# Transform from long format to wide format
# Long format: each row is (user, movie, rating)
# Wide format: rows are users, columns are movies, values are ratings
#
# train_df.pivot_table creates this transformation:
# index='user_id': rows will be user IDs
# columns='item_id': columns will be movie IDs
# values='rating': cells will contain ratings
# fill_value=0: cells with no rating (user didn't rate that movie) become 0
user_item_train = train_df.pivot_table(
    index='user_id',           # Rows: each row represents one user
    columns='item_id',         # Columns: each column represents one movie
    values='rating',           # Cell values: the ratings
    fill_value=0               # If no rating exists: use 0 (not rated)
)

# Result: a 943×1682 matrix where:
# - Rows: 943 users
# - Columns: 1682 movies
# - Values: ratings (1-5) or 0 (not rated)
# - ~99.99% of cells are 0 (very sparse data!)

# ============================================================================
# STEP 2: CALCULATE PEARSON CORRELATIONS
# ============================================================================

# TODO: Calculate Pearson Correlation between users
# Hint: Use the built-in pandas .corr() method on the transposed matrix
# use min_periods to ensure we only correlate users with more than
# a specific threshold value of 5 shared movies

# SOLUTION:
# .corr(method='pearson') calculates Pearson correlation coefficient
# This measures: how similar are two users' rating patterns?
# Correlation ranges from -1 (opposite tastes) to +1 (identical tastes)
#
# Why transpose? 
# - user_item_train is: rows=users, columns=movies
# - .corr() calculates correlation between ROWS by default
# - We want correlation between USERS (rows), but in original orientation
# - Actually, .corr() already calculates column correlations
# - So we use the original orientation: correlates each user with each other user
#
# min_periods=5 means:
# - Only calculate correlation if two users have rated at least 5 movies in common
# - If fewer than 5 common movies: correlation will be NaN (missing value)
# - This prevents spurious correlations from very few data points
user_corr_df = user_item_train.corr(method='pearson', min_periods=5)

# Result: a 943×943 correlation matrix where:
# - Rows: user IDs
# - Columns: user IDs
# - Values: correlation between that pair of users (-1 to +1)
# - Diagonal (user compared to themselves): will be 1.0 (perfect correlation)

# Set diagonal to 0
# This removes the perfect correlation of each user with themselves
# Why? User A is ALWAYS the most similar to user A (correlation=1)
# If we don't remove this:
#   - When predicting for user A, we'd always pick user A as the best neighbour
#   - But user A has no other movies rated to recommend!
#   - We need to look at OTHER users (correlation < 1)
np.fill_diagonal(user_corr_df.values, 0)

# ============================================================================
# STEP 3: PRECOMPUTE MEAN RATINGS
# ============================================================================

# Calculate each user's average rating
# train_df.groupby('user_id') groups all ratings by user
# ['rating'].mean() calculates the average rating for each user
# Result: Series with user_id as index, mean rating as value
# This captures: how harsh/lenient is each user?
#   - Some users give mostly 5 stars (mean ≈ 4.5)
#   - Some users give mostly 3 stars (mean ≈ 3.0)
user_means = train_df.groupby('user_id')['rating'].mean()

# Calculate each item's (movie's) average rating
# Same logic as above, but grouped by item_id instead
# Result: Series with item_id as index, mean rating as value
# This captures: how good is each movie?
#   - Some movies are great (mean ≈ 4.2)
#   - Some movies are bad (mean ≈ 2.1)
item_means = train_df.groupby('item_id')['rating'].mean()

# ============================================================================
# STEP 4: DEFINE PREDICTION FUNCTION WITH FALLBACKS
# ============================================================================

# Define function to predict a user's rating for a movie
# This is the core of user-user collaborative filtering
def predict_rating(user_id, item_id, k=20):
    # Function parameters:
    # user_id: which user are we predicting for
    # item_id: which movie are we predicting a rating for
    # k: how many neighbours to use (default: 20 most similar users)
    
    # ====== FALLBACK 1: Unknown user or item ======
    # Check if user_id is in the correlation matrix
    # Check if item_id is in the pivot table columns
    if user_id not in user_corr_df.index or item_id not in user_item_train.columns:
        # At least one of them is completely unknown (not in training data)
        return mu  # Return global mean rating as default
    
    # ====== FALLBACK 2: Item never rated in training set ======
    # Check if item_id appears in the item_means Series
    if item_id not in item_means:
        # No one rated this movie in training set
        # We can't compare with other users because there are no ratings
        return user_means.get(user_id, mu)  # User's typical rating, or global mean
    
    # ====== FALLBACK 3: User never rated anything in training set ======
    # Check if user_id appears in the user_means Series
    if user_id not in user_means:
        # User gave no ratings in training set (new user)
        # Can't find neighbours because we don't know their preferences
        return item_means[item_id]  # Return the item's average rating
    
    # ====== MAIN ALGORITHM: Find neighbours and predict ======
    
    # Step 1: Find all users who rated this movie in training set
    # Filter train_df to rows where item_id matches this movie
    potential_neighbours = train_df[train_df['item_id'] == item_id]
    
    # Step 2: Get similarity scores between this user and all neighbours
    # user_corr_df.loc[user_id, ...] gets the row for this user
    # [potential_neighbours['user_id']] selects columns for users who rated this item
    # .dropna() removes any NaN values (users with insufficient common ratings)
    sim_scores = user_corr_df.loc[user_id, potential_neighbours['user_id']].dropna()
    
    # Step 3: Filter to strong positive correlations
    # sim_scores[sim_scores > 0.2] keeps only users with correlation > 0.2
    # Why 0.2? This threshold filters out weak correlations (noisy neighbours)
    # .sort_values(ascending=False) orders by correlation (highest first)
    # .head(k) keeps only the top k most similar users
    top_k_sims = sim_scores[sim_scores > 0.2].sort_values(ascending=False).head(k)
    
    # ====== FALLBACK 4: No good neighbours found ======
    # If we couldn't find any neighbours (empty list)
    if top_k_sims.empty:
        # No similar users found
        return item_means[item_id]  # Return the item's average rating
    
    # ====== CALCULATE PREDICTION: Mean-Centered Offsets ======
    # This is the key insight of collaborative filtering
    
    # Step 4a: Get ratings from our neighbours for this specific item
    # potential_neighbours.set_index('user_id') makes user_id the row index
    # .loc[top_k_sims.index, 'rating'] gets ratings from our neighbours
    # Result: Series with user_id as index, rating values as data
    neighbour_ratings = potential_neighbours.set_index('user_id').loc[top_k_sims.index, 'rating']
    
    # Step 4b: Get each neighbour's average rating (their "baseline")
    # This removes individual rating scale differences
    # Example: User A always rates 1 star higher than User B
    # By subtracting their average, we normalize for this
    neighbour_means_sub = user_means.loc[top_k_sims.index]
    
    # Step 4c: Calculate how much each neighbour deviates from THEIR average
    # offsets = (actual rating) - (their average rating)
    # Positive offset: they rated this movie higher than their typical rating
    # Negative offset: they rated this movie lower than their typical rating
    offsets = neighbour_ratings - neighbour_means_sub
    
    # Step 4d: Weight these offsets by similarity and average
    # np.dot(top_k_sims, offsets) = sum of (similarity × offset) for each neighbour
    # / top_k_sims.sum() = divide by sum of similarities to normalize
    # This gives us a weighted average offset
    weighted_offset = np.dot(top_k_sims, offsets) / top_k_sims.sum()
    
    # Step 5: Calculate final prediction
    # user_means[user_id] = this user's average rating (baseline)
    # + weighted_offset = how much neighbours deviate for this movie
    # = Our prediction!
    # np.clip(..., 1, 5) ensures prediction is in valid rating range
    return np.clip(user_means[user_id] + weighted_offset, 1, 5)

# ============================================================================
# STEP 5: EVALUATE WITH DIFFERENT K VALUES
# ============================================================================

# Create a smaller test sample for speed
# test_df.sample(1000, random_state=42) randomly selects 1000 rows
# random_state=42 ensures we get the same sample every time (reproducibility)
# Why only 1000? Full evaluation on 20,000 rows would take several minutes
test_sample = test_df.sample(1000, random_state=42)

# Try different k values and see which works best
# k = number of neighbours to use
for k_val in [5, 10, 20, 40, 75, 100]:
    # Make predictions for all rows in test_sample
    # .apply(lambda x: ...) applies a function to each row
    # lambda x: predict_rating(x['user_id'], x['item_id'], k=k_val)
    #   - For each row, extract user_id and item_id
    #   - Call predict_rating with current k_val
    #   - Return prediction
    test_sample[f'pred_k{k_val}'] = test_sample.apply(
        lambda x: predict_rating(x['user_id'], x['item_id'], k=k_val), axis=1
    )
    
    # Calculate RMSE for this k value
    # test_sample['rating'] = actual ratings from test set
    # test_sample[f'pred_k{k_val}'] = our predictions for this k
    rmse = np.sqrt(mean_squared_error(test_sample['rating'], test_sample[f'pred_k{k_val}']))
    
    # Print result
    # Expected: k=20 usually has best RMSE (~0.92)
    print(f"RMSE with k={k_val}: {rmse:.4f}")
```

---

## CODE CELL 8: EXERCISE 3 - MATRIX FACTORISATION (BASIC)

```python
# ============================================================================
# STEP 1: SET HYPERPARAMETERS
# ============================================================================

# These parameters control the learning process and model complexity

# Number of latent factors (hidden dimensions)
# We decompose the user-item matrix into U (users × factors) and V (items × factors)
# K << num_items, so we're reducing dimensionality
# n_factors=30 is a good balance: not too simple, not too complex
n_factors = 30

# Learning rate (controls how big each update step is)
# alpha=0.015 means: take small steps (0.015) in gradient direction
# Too high (0.05+): steps too large, oscillation, no convergence
# Too low (0.001): convergence very slow, many epochs needed
alpha = 0.015

# Regularisation constant (prevents overfitting)
# lambda_reg=0.1 penalises large factor values
# Larger U or V matrices = larger penalty in loss function
# This prevents the model from fitting noise
lambda_reg = 0.1

# Number of epochs (passes through training data)
# We'll go through entire training dataset 25 times
# More epochs = more learning (up to a point where it stops improving)
epochs = 25

# ============================================================================
# STEP 2: INITIALISE FACTOR MATRICES
# ============================================================================

# Find the maximum user_id in the full dataset
# We need one row per user (including user ID 0 if it exists)
# df.user_id.max() gives maximum user ID (e.g., 943)
# +1 because user IDs are 1-indexed, array indices are 0-indexed
# (Array indices: 0, 1, 2, ..., 943 for 944 users total)
n_users = df.user_id.max() + 1

# Find the maximum item_id in the full dataset
# Same logic as above
# df.item_id.max() gives maximum item ID (e.g., 1682)
# +1 because array is 0-indexed
n_items = df.item_id.max() + 1

# Initialise U matrix (user factors)
# Shape: (n_users, n_factors) = (944, 30)
# Each row: one user's preferences in 30 latent dimensions
# np.random.normal(scale=0.1, size=...) generates random numbers from normal distribution
# scale=0.1 means standard deviation of 0.1 (small random values)
# Why small? Helps with convergence; starting near 0 is good for gradient descent
U = np.random.normal(scale=0.1, size=(n_users, n_factors))

# Initialise V matrix (item factors)
# Shape: (n_items, n_factors) = (1682, 30)
# Each row: one item's characteristics in 30 latent dimensions
# Same initialisation as U: small random values
V = np.random.normal(scale=0.1, size=(n_items, n_factors))

# ============================================================================
# STEP 3: TRAINING LOOP - STOCHASTIC GRADIENT DESCENT (SGD)
# ============================================================================

# Loop through each epoch (pass through the data)
for epoch in range(epochs):
    # Accumulator for total error in this epoch
    # We'll add up squared errors and calculate RMSE at end
    total_error = 0
    
    # Shuffle training data for this epoch
    # train_df[['user_id', 'item_id', 'rating']] selects these 3 columns
    # .sample(frac=1) shuffles all rows (frac=1 means 100% of data)
    # .values converts to numpy array (faster iteration)
    # Why shuffle? Different orderings can help convergence
    shuffled_train = train_df[['user_id', 'item_id', 'rating']].sample(frac=1).values
    
    # Loop through each rating in the shuffled training data
    # Each iteration: one (user, item, rating) triplet
    # "Stochastic" = update after each sample (not in batches)
    for u, i, r in shuffled_train:
        # Convert user and item IDs to integers
        # They're loaded as floats from the array, but we need them as int indices
        u, i = int(u), int(i)
        
        # ====== SOLUTION 1: CALCULATE PREDICTION ======
        # TODO: Calculate prediction (Dot product of U and V)
        
        # Prediction is the dot product of:
        # U[u] = user u's factor vector (size: 30)
        # V[i] = item i's factor vector (size: 30)
        # np.dot gives sum of element-wise products = inner product
        # Result: a single number (predicted rating, ~0 to ~5)
        prediction = np.dot(U[u], V[i])
        
        # ====== SOLUTION 2: CALCULATE ERROR ======
        # TODO: Calculate error (Actual - Prediction)
        
        # Error = actual rating - predicted rating
        # Positive error: we under-predicted (predicted too low)
        # Negative error: we over-predicted (predicted too high)
        error = r - prediction
        
        # ====== SOLUTION 3: UPDATE FACTORS USING SGD ======
        # TODO: Update U and V using alpha and lambda
        # Formula: U[u] = U[u] + alpha * (error * V[i] - lambda * U[u])
        # Formula: V[i] = V[i] + alpha * (error * U[u] - lambda * V[i])
        
        # Update user factor vector
        # Direction to move:
        #   error * V[i] = gradient direction to reduce error
        #   - lambda * U[u] = regularisation (shrink towards zero)
        # Step size: alpha
        U[u] += alpha * (error * V[i] - lambda_reg * U[u])
        
        # Update item factor vector
        # Same principle as U[u] update, but with different gradient
        V[i] += alpha * (error * U[u] - lambda_reg * V[i])
        
        # Accumulate squared error for RMSE calculation
        total_error += error ** 2
    
    # Calculate RMSE for this epoch
    # total_error / len(train_df) = mean squared error
    # np.sqrt(...) = take square root = root mean squared error
    epoch_rmse = np.sqrt(total_error / len(train_df))
    
    # Print progress
    # Students can watch RMSE decrease as model learns
    print(f"Epoch {epoch+1}: Training RMSE = {epoch_rmse:.4f}")
```

---

## CODE CELL 10: EXERCISE 3 - TEST EVALUATION (EMPTY - STUDENTS FILL IN)

```python
# ============================================================================
# TODO: Run through the test set, make predictions and calculate RMSE
# ============================================================================

# SOLUTION:
# After training is complete, we need to evaluate on the test set
# This shows how well the model generalises to unseen data

# Create a list to store predictions
test_predictions = []

# Loop through each row in the test set
for idx, row in test_df.iterrows():
    # Extract user and item IDs from this row
    u = int(row['user_id'])
    i = int(row['item_id'])
    
    # Make prediction using trained factors
    # np.dot(U[u], V[i]) = dot product of learned factors
    pred = np.dot(U[u], V[i])
    
    # Clip prediction to valid rating range [1, 5]
    # Some predictions might be 0.5 or 5.8 from the math
    # But valid ratings are 1, 2, 3, 4, 5
    test_predictions.append(np.clip(pred, 1, 5))

# Add predictions to test dataframe as a column
test_df['pred_mf'] = test_predictions

# Calculate test RMSE
# Compare actual test ratings with our predictions
rmse_mf = np.sqrt(mean_squared_error(test_df['rating'], test_df['pred_mf']))

# Print result
print(f"Test RMSE (MF-SGD): {rmse_mf:.4f}")
# Expected: ~1.02 (worse than training! overfitting)
```

---

## CODE CELL 12: EXERCISE 4 - FULL NETFLIX PRIZE MODEL (EMPTY - STUDENTS FILL IN)

```python
# ============================================================================
# TODO: Write a new version implementing the full Netflix prize-winning model
# ============================================================================

# SOLUTION:
# The full model adds user and item biases
# This allows factors to focus on interactions
# Instead of trying to capture everything in U and V

# Initialise bias vectors to zeros
# b_u[i] = bias for user i (how much they rate above/below average)
# b_i[j] = bias for item j (how good/bad the item is)
user_bias = np.zeros(n_users)
item_bias = np.zeros(n_items)

# Training loop (same structure as before)
for epoch in range(epochs):
    total_error = 0
    shuffled_train = train_df[['user_id', 'item_id', 'rating']].sample(frac=1).values
    
    for u, i, r in shuffled_train:
        u, i = int(u), int(i)
        
        # ====== FULL PREDICTION WITH BIASES ======
        # Decompose into parts:
        # μ = global mean (average rating across all users/items)
        # b_u = user bias (how much this user rates above/below average)
        # b_i = item bias (how good/bad this item is)
        # U_u · V_i = personalised interaction (user-item fit)
        prediction = mu + user_bias[u] + item_bias[i] + np.dot(U[u], V[i])
        
        # Calculate error
        error = r - prediction
        
        # ====== UPDATE BIASES FIRST ======
        # Important: update biases before factors
        # Biases capture global trends, factors focus on residuals
        
        # Update user bias
        # User bias shifts based on error and regularisation
        user_bias[u] += alpha * (error - lambda_reg * user_bias[u])
        
        # Update item bias
        # Item bias shifts based on error and regularisation
        item_bias[i] += alpha * (error - lambda_reg * item_bias[i])
        
        # ====== THEN UPDATE FACTORS ======
        # Same as before, but now error is after removing biases
        U[u] += alpha * (error * V[i] - lambda_reg * U[u])
        V[i] += alpha * (error * U[u] - lambda_reg * V[i])
        
        total_error += error ** 2
    
    epoch_rmse = np.sqrt(total_error / len(train_df))
    print(f"Epoch {epoch+1}: Training RMSE = {epoch_rmse:.4f}")
```

---

## CODE CELL 13: EVALUATE FULL MODEL

```python
# ============================================================================
# FINAL EVALUATION: Test the full Netflix prize model
# ============================================================================

# Define prediction function for the full model
def predict_final(user_id, item_id):
    """
    Full Netflix Prize prediction with biases
    
    Prediction = global_mean + user_bias + item_bias + (U · V)
    """
    # Convert to int indices
    u = int(user_id)
    i = int(item_id)
    
    # Calculate prediction with all components
    # mu: global mean rating
    # user_bias[u]: how much this user rates above/below average
    # item_bias[i]: how good/bad this movie is
    # np.dot(U[u], V[i]): personalised interaction between user and item
    pred = mu + user_bias[u] + item_bias[i] + np.dot(U[u], V[i])
    
    # Clip to valid range
    return np.clip(pred, 1, 5)

# Apply prediction function to entire test set
# .apply(lambda x: ...) applies function to each row
# lambda x: predict_final(int(x['user_id']), int(x['item_id']))
#   - Extract user_id and item_id from row
#   - Convert to int
#   - Call predict_final
# axis=1 means apply to rows (not columns)
test_df['pred_mf'] = test_df.apply(lambda x: predict_final(int(x['user_id']), int(x['item_id'])), axis=1)

# Calculate test RMSE
# Compare actual test ratings with full model predictions
rmse_mf = np.sqrt(mean_squared_error(test_df['rating'], test_df['pred_mf']))

# Print results
print(f"\n--- Model Summary ---")
print(f"Final Test RMSE (MF-SGD): {rmse_mf:.4f}")
# Expected: ~0.91 (much better! biases fixed overfitting)
```

---

## Summary of All Code Concepts

### Key Programming Concepts Used:

1. **Pandas Operations**:
   - `.read_csv()`: Load data
   - `.groupby().agg()`: Calculate statistics
   - `.pivot_table()`: Transform data
   - `.merge()`: Combine dataframes
   - `.apply()`: Apply functions to rows/columns
   - `.sample()`: Random sampling
   - `.fillna()`: Handle missing values

2. **NumPy Operations**:
   - `.dot()`: Matrix/vector multiplication
   - `.sqrt()`: Square root
   - `.fill_diagonal()`: Set diagonal to value
   - `np.random.normal()`: Random initialization
   - `np.clip()`: Constrain values to range

3. **Machine Learning Concepts**:
   - **Bayesian Averaging**: Regularise small samples
   - **Pearson Correlation**: Measure user similarity
   - **Mean-Centered Offsets**: Normalize for rating scales
   - **Stochastic Gradient Descent**: Learn by updating on each sample
   - **Regularisation**: Prevent overfitting with lambda parameter
   - **Biases**: Capture global trends separately from interactions
   - **Train-Test Split**: Evaluate generalisation

4. **Evaluation**:
   - **RMSE**: Root Mean Square Error
   - **Overfitting**: Model fits training data but not test data
   - **Hyperparameter Tuning**: Find best k, alpha, lambda values
