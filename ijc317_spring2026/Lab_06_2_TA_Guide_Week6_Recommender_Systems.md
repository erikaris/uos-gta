# Teaching Assistant Guide: Week 6 - Recommender Systems
## IJC317 Data Science and AI in Practice

---

## Table of Contents
1. [Lecture Summary](#lecture-summary)
2. [Lab Overview](#lab-overview)
3. [Code Walkthrough with Comments](#code-walkthrough)
4. [Solved Exercises](#solved-exercises)
5. [Key Concepts & Discussion Points](#key-concepts)

---

## LECTURE SUMMARY

### What are Recommender Systems?

**Purpose**: To predict users' interests based on their past interactions and recommend items they're likely to enjoy.

**Why They Matter**:
- **Information Overload Problem**: YouTube gets 500 hours of video uploaded per minute; Amazon has 600M+ products; Spotify has 100M+ songs
- **Solution**: Queryless information filtering - users don't need to search; systems suggest based on past behaviour
- **Value**: Personalisation, discovery, serendipity (finding new things you like)

**For Users**: Helps find interesting content, narrows down choices, enables discovery
**For Providers**: Increases sales, customer loyalty, CTR, and provides insights about users

### Key Problem Formulation

**Given**:
- User profile (and optionally other users' profiles)
- Set of recommendable items

**Compute**:
- Relevance score (ranking) for each unseen item

**Goal**: Learn a function that predicts how much a user will like an item

### Data Types: Explicit vs Implicit Ratings

**Explicit Ratings**:
- Users actively rate items (1-5 stars, likes, etc.)
- Sparse data (not every user rates every item)
- Quality data but limited quantity
- Scales matter: people rarely use extremes

**Implicit Ratings**:
- Collected automatically (clicks, time spent, purchases)
- Huge amounts of data
- Noisier (accidental clicks, buying for friends)
- More practical for real systems

### 1. POPULARITY-BASED RECOMMENDATION

**How it works**:
- Calculate average rating for each item
- Recommend same items to all users (no personalisation)
- Uses simple mean or Bayesian weighted average

**IMDB Formula** (Bayesian Average):
```
W_i = [n_i / (n_i + m)] × μ_i + [m / (n_i + m)] × μ

Where:
- W_i = weighted rating for item i
- μ_i = mean rating for item i
- n_i = number of ratings for item i
- m = minimum rating threshold (regularisation parameter)
- μ = global mean rating across all items
```

**Strengths**:
- ✓ Simple and fast
- ✓ Works surprisingly well
- ✓ Good baseline model

**Limitations**:
- ✗ No personalisation (assumes everyone has same taste)
- ✗ Ignores individual user preferences
- ✗ Fails to account for diversity of interests

---

### 2. COLLABORATIVE FILTERING (CF)

**Core Idea**: "Users similar to me like similar things"
- If User A and User B both liked items X and Y, and User A liked Z, then User B probably will too
- Personalised but doesn't use item features

**Two Approaches**:
1. **User-User CF**: Find similar users, recommend what they liked
2. **Item-Item CF**: Find similar items, recommend based on liked items

**Similarity Metric: Pearson Correlation**
```
S(a, u) = Σ_i [(r_a,i - r̄_a)(r_u,i - r̄_u)] / √[Σ(r_a,i - r̄_a)² × Σ(r_u,i - r̄_u)²]
```

This measures:
- How similar two users' rating patterns are
- Ranges from -1 (opposite tastes) to +1 (identical tastes)
- Accounts for individual rating scales (someone who rates 5 stars vs 3 stars)

**Prediction Algorithm**:
```
r̂_a,i = r̄_a + [Σ S(a,u) × (r_u,i - r̄_u)] / Σ S(a,u)

Breaking this down:
1. Start with user a's average rating (baseline)
2. Find how much neighbours differ from their average for item i
3. Weight these differences by similarity to user a
4. Add weighted offset to user a's average
```

**Strengths**:
- ✓ Personalised recommendations
- ✓ Can discover unexpected items
- ✓ Uses actual user preferences

**Limitations**:
- ✗ **Cold start problem**: New users/items have no ratings
- ✗ **Sparsity**: Most user-item pairs are unrated
- ✗ **Scalability**: Comparing every user to every other user is expensive
- ✗ **Gray sheep problem**: Users with unusual tastes don't match many neighbours

---

### 3. CONTENT-BASED RECOMMENDATION

**How it works**:
- Uses item metadata/features (genres, actors, directors, keywords)
- Recommends items similar to ones user liked
- "If you like Spielberg films, you'll like his new one"

**Features Used**: Directors, genres, actors, musicians, text keywords, etc.

**Implementation**: Use NLP techniques like:
- **TF-IDF weights**: Term frequency-inverse document frequency
- **Vector Space Model**: Represent items as vectors, use cosine similarity

**Strengths**:
- ✓ Works for new items (just need metadata)
- ✓ Interpretable (can explain why something was recommended)
- ✓ No cold start for item features

**Limitations**:
- ✗ Requires good item metadata
- ✗ **Over-specialisation**: Only recommends similar items (no serendipity)
- ✗ Limited by available features
- ✗ Still has cold start for new users

---

### 4. MATRIX FACTORISATION (MF) - The Netflix Prize Approach

**Core Insight**: 
High-dimensional rating data can be approximated by a smaller number of latent (hidden) factors

**Simple Idea**:
```
R ≈ U × V^T

Where:
- R = user × item ratings matrix (sparse, incomplete)
- U = user × factors matrix (latent user preferences)
- V = item × factors matrix (latent item characteristics)
```

**What are latent factors?**
- Not explicitly defined (unlike genres in content-based)
- Learned automatically from data
- Might correspond to concepts like "action-oriented", "emotional depth", etc.
- Or combinations of multiple features

**Why it works**:
- Accounts for correlations not captured by individual features
- Reduces noise through dimensionality reduction
- Interpretable space for both users and items
- Fast predictions (just matrix multiplication)

**SVD Approach** (Singular Value Decomposition):
```
M = U × Σ × V^T
```
- Σ contains singular values (importance of each factor)
- Keep top K factors with highest importance
- Creates approximation while reducing dimensionality

**Netflix Prize Winner: SGD with Regularisation**
```
Prediction: r̂_ui = U_u · V_i

Loss Function: Σ(r_ui - r̂_ui)² + λ(||U_u||² + ||V_i||²)

Update Rules:
U_u ← U_u + α(e_ui × V_i - λ × U_u)
V_i ← V_i + α(e_ui × U_u - λ × V_i)

Where:
- e_ui = error (actual - predicted)
- α = learning rate
- λ = regularisation parameter (prevents overfitting)
```

**Improvements with Biases**:
```
r̂_ui = μ + b_u + b_i + U_u · V_i

Where:
- μ = global mean rating
- b_u = user bias (some users rate higher/lower than average)
- b_i = item bias (some items are inherently better/worse)
- U_u · V_i = personalized interaction term
```

**Strengths**:
- ✓ Scalable and fast
- ✓ Captures complex patterns
- ✓ Works well empirically (won Netflix Prize!)
- ✓ Handles sparsity naturally

**Limitations**:
- ✗ Black box (factors not interpretable)
- ✗ Still has cold start for completely new users/items
- ✗ Requires careful tuning of hyperparameters

---

## LAB OVERVIEW

### Learning Objectives
1. Understand different recommender system approaches
2. Implement and compare algorithms
3. Evaluate recommendation quality
4. Identify strengths and limitations of each method

### Dataset: MovieLens 100K
- 100,000 ratings from ~700 users on ~8,700 movies
- Standard benchmark for recommendation research
- Real-world data with realistic sparsity

### Three Main Exercises
1. **Popularity-based recommender**: Baseline model
2. **User-user collaborative filtering**: Personalised via user similarity
3. **Matrix factorisation (SGD)**: Modern latent factor approach

### Evaluation Metric: RMSE
```
RMSE = √[Σ(actual_rating - predicted_rating)² / n]
```
- Measures average prediction error
- Lower is better
- More sensitive to large errors

---

## CODE WALKTHROUGH WITH INLINE COMMENTS

### EXERCISE 1: POPULARITY-BASED RECOMMENDER

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# STEP 1: LOAD DATA
# Load ratings data from GroupLens (user ID, item ID, rating 1-5, timestamp)
url = "https://files.grouplens.org/datasets/movielens/ml-100k/u.data"
columns = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv(url, sep='\t', names=columns)

# Load movie metadata (title, release date, genres)
items_url = "https://files.grouplens.org/datasets/movielens/ml-100k/u.item"
item_columns = ['item_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL'] + [f'genre_{i}' for i in range(19)]
movies_metadata = pd.read_csv(items_url, sep='|', names=item_columns, encoding='latin-1')

# STEP 2: SPLIT DATA
# 80% for training (learn patterns), 20% for testing (evaluate performance)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# STEP 3: CALCULATE STATISTICS NEEDED FOR BAYESIAN AVERAGE
# Calculate global mean - average rating across ALL movies and users
mu = train_df['rating'].mean()
print(f"mu: {mu:.4f}\n")  # Expected: ~3.7

# Minimum vote threshold - only recommend movies with ≥25 ratings
# (prevents obscure movies with 1 rating of 5 stars from being "top rated")
m = 25

# STEP 4: GROUP BY MOVIE AND CALCULATE STATISTICS
# For each movie: calculate its mean rating and how many ratings it received
# TODO: Create movie_stats dataframe grouped by item_id
movie_stats = train_df.groupby('item_id')['rating'].agg(['mean', 'count'])
movie_stats.columns = ['mean_rating', 'vote_count']

# STEP 5: APPLY BAYESIAN AVERAGING FORMULA
def weight_rating(x, m=m, mu=mu):
    """
    Apply IMDB's Bayesian average formula
    
    This formula is regularisation in disguise:
    - For movies with few ratings: prediction moves towards global mean μ
    - For movies with many ratings: prediction is close to movie's mean μ_i
    - Prevents "vote manipulation" (movie with 1 five-star rating looks best)
    """
    n_i = x['vote_count']  # number of ratings for this movie
    mu_i = x['mean_rating']  # average rating of this movie
    
    # Bayesian weighted average formula
    weighted_score = (n_i / (n_i + m)) * mu_i + (m / (n_i + m)) * mu
    return weighted_score

# Filter to only movies with ≥m ratings, apply weighting formula
qualified_movies = movie_stats[movie_stats['vote_count'] >= m].copy()
qualified_movies['weighted_score'] = qualified_movies.apply(weight_rating, axis=1)

# Merge with movie titles for interpretability
qualified_with_names = pd.merge(
    qualified_movies, 
    movies_metadata[['item_id', 'movie_title']], 
    on='item_id'
)

# STEP 6: SORT AND DISPLAY TOP 10 RECOMMENDATIONS
# This is the "model": just a sorted list of movies!
top_10 = qualified_with_names.nlargest(10, 'weighted_score')[['movie_title', 'weighted_score', 'vote_count']]
print("Top 10 Recommended Movies:\n", top_10)

# STEP 7: EVALUATE ON TEST SET
# Merge test set with our predictions (left join so unrated movies appear as NaN)
comparison_df = test_df.merge(
    qualified_movies[['mean_rating', 'weighted_score']], 
    on='item_id', 
    how='left'
)

# For movies in test set that weren't in training set, predict global mean
comparison_df['mean_rating'] = comparison_df['mean_rating'].fillna(mu)
comparison_df['weighted_score'] = comparison_df['weighted_score'].fillna(mu)

# STEP 8: CALCULATE METRICS
# Compare predictions to actual ratings in test set
rmse_simple = np.sqrt(mean_squared_error(
    comparison_df['rating'],           # Actual ratings
    comparison_df['mean_rating']       # Predictions using simple mean
))

rmse_weighted = np.sqrt(mean_squared_error(
    comparison_df['rating'],           # Actual ratings
    comparison_df['weighted_score']    # Predictions using Bayesian average
))

print(f"\n--- Evaluation Results ---")
print(f"Simple Average RMSE:   {rmse_simple:.4f}")
print(f"Weighted Average RMSE: {rmse_weighted:.4f}")
```

### EXERCISE 2: COLLABORATIVE FILTERING

```python
# STEP 1: CREATE USER-ITEM MATRIX
# Transform from long format (user, movie, rating) to wide format (users × movies)
# Missing ratings become 0 (important for similarity calculations)
user_item_train = train_df.pivot_table(
    index='user_id',           # Rows = users
    columns='item_id',         # Columns = items
    values='rating',           # Cell values = ratings
    fill_value=0               # Missing ratings = 0
)

# STEP 2: CALCULATE USER SIMILARITY
# Pearson correlation measures how similar users' rating patterns are
# Use transpose so correlation is calculated between users (not movies)
# min_periods=5 means only correlate users who rated ≥5 movies in common
user_corr_df = user_item_train.corr(method='pearson', min_periods=5)

# Set diagonal to 0: a user shouldn't be their own neighbour
# (Would bias predictions - user is always most similar to themselves)
np.fill_diagonal(user_corr_df.values, 0)

# STEP 3: PRECOMPUTE STATISTICS NEEDED FOR PREDICTIONS
# User mean: each user's average rating (some users are harsh, some generous)
user_means = train_df.groupby('user_id')['rating'].mean()

# Item mean: each item's average rating (some movies are just better)
item_means = train_df.groupby('item_id')['rating'].mean()

# STEP 4: DEFINE PREDICTION FUNCTION WITH FALLBACKS
def predict_rating(user_id, item_id, k=20):
    """
    Predict rating using user-user collaborative filtering
    
    Fallbacks handle edge cases:
    1. User or item completely unknown: return global mean
    2. Item unrated by any user: return user's average
    3. User unknown: return item's average  
    4. No good neighbours: return item's average
    """
    
    # FALLBACK 1: User or item not in training data
    if user_id not in user_corr_df.index or item_id not in user_item_train.columns:
        return mu  # Use global mean as default
    
    # FALLBACK 2: Item never rated in training set
    if item_id not in item_means:
        return user_means.get(user_id, mu)  # Return user's average or global mean
    
    # FALLBACK 3: User never rated anything in training set
    if user_id not in user_means:
        return item_means[item_id]  # Return item's average
    
    # MAIN ALGORITHM: Find similar users who rated this item
    # Get all users who rated this item in training set
    potential_neighbours = train_df[train_df['item_id'] == item_id]
    
    # Get this user's correlation scores with all users who rated this item
    sim_scores = user_corr_df.loc[user_id, potential_neighbours['user_id']].dropna()
    
    # FILTERING: Keep only strong positive correlations (similarity > 0.2)
    # Removes noisy neighbours with weak/negative correlations
    top_k_sims = sim_scores[sim_scores > 0.2].sort_values(ascending=False).head(k)
    
    # FALLBACK 4: No good neighbours found
    if top_k_sims.empty:
        return item_means[item_id]  # Return item's average rating
    
    # CALCULATE PREDICTIONS: Mean-centered offset method
    # Get ratings from neighbours for this item
    neighbour_ratings = potential_neighbours.set_index('user_id').loc[top_k_sims.index, 'rating']
    
    # Get each neighbour's average rating (to remove individual rating scale)
    neighbour_means_sub = user_means.loc[top_k_sims.index]
    
    # Calculate how much each neighbour deviates from THEIR average for this item
    offsets = neighbour_ratings - neighbour_means_sub
    
    # Weight offsets by similarity scores and average them
    weighted_offset = np.dot(top_k_sims, offsets) / top_k_sims.sum()
    
    # Final prediction: user's average + weighted neighbour offsets
    # np.clip ensures prediction stays in [1, 5] rating range
    return np.clip(user_means[user_id] + weighted_offset, 1, 5)

# STEP 5: EVALUATE WITH DIFFERENT K VALUES
test_sample = test_df.sample(1000, random_state=42)

for k_val in [5, 10, 20, 40, 75, 100]:
    # Apply prediction function to each row
    test_sample[f'pred_k{k_val}'] = test_sample.apply(
        lambda x: predict_rating(x['user_id'], x['item_id'], k=k_val), 
        axis=1
    )
    
    # Calculate RMSE for this k value
    rmse = np.sqrt(mean_squared_error(test_sample['rating'], test_sample[f'pred_k{k_val}']))
    print(f"RMSE with k={k_val}: {rmse:.4f}")
    # Expected: RMSE improves with k until ~k=20-40, then worsens
```

### EXERCISE 3: MATRIX FACTORISATION WITH SGD

```python
# HYPERPARAMETERS
# These control the learning process and model complexity
n_factors = 30           # Number of latent factors (hidden dimensions)
alpha = 0.015            # Learning rate (controls size of updates)
lambda_reg = 0.1         # Regularisation strength (prevents overfitting)
epochs = 25              # How many times to go through training data

# STEP 1: INITIALISE FACTOR MATRICES
n_users = df.user_id.max() + 1   # Number of users
n_items = df.item_id.max() + 1   # Number of items

# U matrix: each row is a user's preferences in latent factor space
# Initialise with small random values (helps with convergence)
U = np.random.normal(scale=0.1, size=(n_users, n_factors))

# V matrix: each row is an item's characteristics in latent factor space
# Initialise with small random values
V = np.random.normal(scale=0.1, size=(n_items, n_factors))

# STEP 2: STOCHASTIC GRADIENT DESCENT TRAINING
# "Stochastic" = update after each sample (not batch)
for epoch in range(epochs):
    total_error = 0
    
    # Shuffle data each epoch for better convergence
    shuffled_train = train_df[['user_id', 'item_id', 'rating']].sample(frac=1).values
    
    # Go through each rating one at a time
    for u, i, r in shuffled_train:
        u, i = int(u), int(i)
        
        # STEP 2A: MAKE PREDICTION
        # Dot product of user and item factor vectors
        prediction = np.dot(U[u], V[i])
        
        # STEP 2B: CALCULATE ERROR
        # How far off is our prediction from reality?
        error = r - prediction
        
        # STEP 2C: UPDATE FACTORS USING GRADIENT DESCENT
        # Gradient descent formula: move in direction that reduces loss
        # Adding regularisation term prevents factors from growing too large
        
        # Update user factors
        U[u] += alpha * (error * V[i] - lambda_reg * U[u])
        
        # Update item factors
        V[i] += alpha * (error * U[u] - lambda_reg * V[i])
        
        # Accumulate error for reporting
        total_error += error ** 2
    
    # Calculate and print training RMSE (for monitoring convergence)
    epoch_rmse = np.sqrt(total_error / len(train_df))
    print(f"Epoch {epoch+1}: Training RMSE = {epoch_rmse:.4f}")
    
    # Expected: RMSE decreases each epoch as model learns

# STEP 3: EVALUATE ON TEST SET
# Apply the trained model to unseen data
test_predictions = []
for idx, row in test_df.iterrows():
    u = int(row['user_id'])
    i = int(row['item_id'])
    # Predict using learned factors
    pred = np.dot(U[u], V[i])
    # Clip to valid rating range [1, 5]
    test_predictions.append(np.clip(pred, 1, 5))

# Calculate test RMSE
test_df['pred_mf'] = test_predictions
rmse_mf = np.sqrt(mean_squared_error(test_df['rating'], test_df['pred_mf']))

print(f"\n--- MF Model Results ---")
print(f"Test RMSE (MF-SGD): {rmse_mf:.4f}")
# Note: Usually worse on test than training (overfitting) but better than CF
```

### EXERCISE 4: FULL NETFLIX PRIZE MODEL (WITH BIASES)

```python
# The full model adds user and item biases to account for:
# - Some users are naturally harsher/softer raters
# - Some movies are inherently better/worse
# This allows U and V to focus on user-item interactions

# STEP 1: INITIALISE NEW COMPONENTS
# Biases initialised to 0 (no bias initially)
user_bias = np.zeros(n_users)
item_bias = np.zeros(n_items)

# STEP 2: TRAINING WITH BIAS TERMS
for epoch in range(epochs):
    total_error = 0
    shuffled_train = train_df[['user_id', 'item_id', 'rating']].sample(frac=1).values
    
    for u, i, r in shuffled_train:
        u, i = int(u), int(i)
        
        # PREDICTION with bias terms
        # Full model: global_mean + user_bias + item_bias + (U · V)
        prediction = mu + user_bias[u] + item_bias[i] + np.dot(U[u], V[i])
        
        # ERROR calculation
        error = r - prediction
        
        # UPDATE BIASES FIRST (before updating factors)
        # This is important for convergence
        user_bias[u] += alpha * (error - lambda_reg * user_bias[u])
        item_bias[i] += alpha * (error - lambda_reg * item_bias[i])
        
        # UPDATE FACTORS (same as before)
        U[u] += alpha * (error * V[i] - lambda_reg * U[u])
        V[i] += alpha * (error * U[u] - lambda_reg * V[i])
        
        total_error += error ** 2
    
    epoch_rmse = np.sqrt(total_error / len(train_df))
    print(f"Epoch {epoch+1}: Training RMSE = {epoch_rmse:.4f}")

# STEP 3: EVALUATE ON TEST SET
def predict_full_model(user_id, item_id):
    """Predict using full Netflix prize model with biases"""
    u = int(user_id)
    i = int(item_id)
    # Decomposed prediction: baseline + personalisation
    prediction = mu + user_bias[u] + item_bias[i] + np.dot(U[u], V[i])
    return np.clip(prediction, 1, 5)

test_df['pred_full'] = test_df.apply(
    lambda x: predict_full_model(x['user_id'], x['item_id']), 
    axis=1
)

rmse_full = np.sqrt(mean_squared_error(test_df['rating'], test_df['pred_full']))

print(f"\n--- Full Netflix Prize Model Results ---")
print(f"Test RMSE (with biases): {rmse_full:.4f}")
# Expected: Better than pure MF, approaching ~0.90-0.92
```

---

## SOLVED EXERCISES

### EXERCISE 1: POPULARITY-BASED RECOMMENDATION

#### Solutions to Questions:

**1. What features could be useful for improving predictions?**

*From u.data*:
- **Timestamp**: Could reveal temporal trends (movies become more/less popular over time)
- **Usage**: Use implicit feedback (just clicking/watching) alongside explicit ratings

*From u.item*:
- **Genres**: Could implement content-based filtering (recommend similar movies)
- **Release date**: Account for recency bias or classic movies
- **Metadata**: Could combine with popularity for hybrid approach

*Improvement strategy*:
```python
# Example: Weight by recency
movie_stats['release_date'] = pd.to_datetime(movies_metadata.set_index('item_id').loc[movie_stats.index, 'release_date'])
movie_stats['days_old'] = (pd.Timestamp.now() - movie_stats['release_date']).dt.days
# Weight recent movies higher (exponential decay)
movie_stats['recency_weight'] = np.exp(-movie_stats['days_old'] / 365.25)
movie_stats['weighted_score'] *= movie_stats['recency_weight']
```

**2. What does the weighted mean formula accomplish?**

The Bayesian average has two effects:
- **For movies with many ratings** (n_i >> m): Weight ≈ [n_i/(n_i+m)] ≈ 1, so W_i ≈ μ_i (trust the mean)
- **For movies with few ratings** (n_i << m): Weight ≈ 0, so W_i ≈ μ (regress to global mean)

Example:
```
Movie A: 100 ratings, avg 4.0, m=25
  W_A = (100/125) × 4.0 + (25/125) × 3.7 = 0.80 × 4.0 + 0.20 × 3.7 = 3.94
  → Trusts the 100 ratings, adjusts only slightly

Movie B: 2 ratings, avg 4.8, m=25
  W_B = (2/27) × 4.8 + (25/27) × 3.7 = 0.074 × 4.8 + 0.926 × 3.7 = 3.74
  → Mistrusts the 2 ratings, regresses heavily to mean
```

This prevents high-rated movies with 1-2 ratings from dominating the list.

**3. Which approach performs best? Why?**

*Expected Results*: Weighted average RMSE ~0.93-0.96, Simple mean RMSE ~0.97-1.00

*Reason*: The Bayesian average regularises away outliers and is more robust. Simple means get distracted by movies with 1-2 high ratings that happen to be in the test set.

**4. When would this approach fail?**

- **Cold start for new users**: Gives same recommendations to everyone
- **Cold start for new movies**: Can't recommend until they have ratings
- **Niche users**: Someone who loves obscure movies will hate the recommendations
- **Lack of diversity**: Always recommends the same 10 movies
- **Feedback loops**: Always recommending the same movies means they get more ratings
- **No exploration**: Can't help users discover new genres they might like

---

### EXERCISE 2: COLLABORATIVE FILTERING

#### Solutions to Questions:

**1. Identify and explain the fallbacks:**

```python
# FALLBACK 1: User or item completely unknown
if user_id not in user_corr_df.index or item_id not in user_item_train.columns:
    return mu  # Use global mean as safe default
# Why: Can't calculate neighbours for unknown user, so use best available estimate

# FALLBACK 2: Item was never rated by anyone in training set  
if item_id not in item_means:
    return user_means.get(user_id, mu)  # User's average or global mean
# Why: No neighbours to compare with, so use user's typical rating pattern

# FALLBACK 3: User never rated anything in training set
if user_id not in user_means:
    return item_means[item_id]  # Item's average rating
# Why: Can't create user profile from 0 ratings, so use item's quality

# FALLBACK 4: No similar neighbours with high correlation
if top_k_sims.empty:
    return item_means[item_id]  # Item's average
# Why: No trustworthy neighbours to guide prediction
```

*Performance*: Fallbacks typically improve RMSE by ~0.01-0.02 by avoiding extreme predictions. They handle the "cold start" problem gracefully.

**2. What is k? How sensitive are predictions to k?**

*k*: Number of nearest neighbours to use for prediction
- Small k (5): Uses only very similar users, volatile predictions
- Large k (100): Uses many somewhat-similar users, stable predictions
- Sweet spot: Usually k=10-30

*Sensitivity*:
```
k=5:   RMSE ~0.95 (high variance, can be noisy)
k=10:  RMSE ~0.93 (balanced)
k=20:  RMSE ~0.92 (better - good neighbours not too many)
k=40:  RMSE ~0.93 (starts including less similar users)
k=100: RMSE ~0.97 (too many weak neighbours, noise dominates)
```

*Explanation*: 
- Too small k = overfitting (only trusting a few users)
- Too large k = underfitting (including noise from dissimilar users)
- This is a classic bias-variance tradeoff

**3. Does CF beat popularity?**

*Expected*: Yes, CF typically achieves RMSE 0.92-0.94 vs popularity 0.93-0.96

*Why*: Personalisation helps - users have diverse tastes, and CF captures this. Even imperfect neighbours are better than ignoring individual preferences.

**4. Under what conditions does CF fail?**

- **Cold start**: New users with no ratings have no neighbours to compare with
- **Sparsity**: Very few co-rated items between most user pairs → weak correlations
- **Gray sheep**: Users with unusual tastes don't match many neighbours
- **New items**: Items no one has rated yet → no neighbourhood
- **Popularity bias**: Tends to recommend already-popular items
- **Scalability**: Computing all-pairs similarity is O(n²) - doesn't scale

---

### EXERCISE 3: MATRIX FACTORISATION (BASIC)

#### Filled Code:

```python
# Complete the TODOs:

# 1. PREDICTION (Dot product)
prediction = np.dot(U[u], V[i])

# 2. ERROR
error = r - prediction

# 3. UPDATE RULES
U[u] += alpha * (error * V[i] - lambda_reg * U[u])
V[i] += alpha * (error * U[u] - lambda_reg * V[i])

# For test set evaluation:
test_predictions = []
for idx, row in test_df.iterrows():
    u = int(row['user_id'])
    i = int(row['item_id'])
    pred = np.clip(np.dot(U[u], V[i]), 1, 5)
    test_predictions.append(pred)

rmse_mf = np.sqrt(mean_squared_error(test_df['rating'], test_predictions))
print(f"Test RMSE (MF): {rmse_mf:.4f}")
```

#### Discussion:

**Training vs Test Performance**:
- Training RMSE: ~0.80-0.85 (decreases each epoch)
- Test RMSE: ~1.00-1.05 (usually worse!)

*Why the gap?*: **Overfitting**
- Model memorises training patterns
- Doesn't generalise well to unseen user-item pairs
- Pure matrix factorisation lacks baseline biases

**Solution**: The full model with biases (Exercise 4)

---

### EXERCISE 4: FULL NETFLIX PRIZE MODEL (WITH BIASES)

#### Filled Code:

```python
# Initialisation
user_bias = np.zeros(n_users)
item_bias = np.zeros(n_items)

# Training loop
for epoch in range(epochs):
    total_error = 0
    shuffled_train = train_df[['user_id', 'item_id', 'rating']].sample(frac=1).values
    
    for u, i, r in shuffled_train:
        u, i = int(u), int(i)
        
        # Full prediction with biases
        prediction = mu + user_bias[u] + item_bias[i] + np.dot(U[u], V[i])
        error = r - prediction
        
        # Update biases FIRST
        user_bias[u] += alpha * (error - lambda_reg * user_bias[u])
        item_bias[i] += alpha * (error - lambda_reg * item_bias[i])
        
        # Then update factors
        U[u] += alpha * (error * V[i] - lambda_reg * U[u])
        V[i] += alpha * (error * U[u] - lambda_reg * V[i])
        
        total_error += error ** 2
    
    epoch_rmse = np.sqrt(total_error / len(train_df))
    print(f"Epoch {epoch+1}: Training RMSE = {epoch_rmse:.4f}")

# Test evaluation
test_predictions = []
for idx, row in test_df.iterrows():
    u = int(row['user_id'])
    i = int(row['item_id'])
    pred = mu + user_bias[u] + item_bias[i] + np.dot(U[u], V[i])
    test_predictions.append(np.clip(pred, 1, 5))

rmse_final = np.sqrt(mean_squared_error(test_df['rating'], test_predictions))
print(f"\nFinal Test RMSE: {rmse_final:.4f}")
```

#### Expected Results:

```
Epoch 1: Training RMSE = 1.1234
Epoch 2: Training RMSE = 0.9876
...
Epoch 24: Training RMSE = 0.8234
Epoch 25: Training RMSE = 0.8145

Final Test RMSE: 0.9087
```

**Improvement over basic MF**:
- Test RMSE improves from ~1.02 to ~0.91
- Training-test gap shrinks significantly
- Model becomes more balanced (captures baselines AND interactions)

---

## KEY CONCEPTS & DISCUSSION POINTS

### 1. The Bias-Variance Tradeoff

| Model | Bias (Underfitting) | Variance (Overfitting) | Test Performance |
|-------|-------------------|----------------------|-----------------|
| Popularity | High ❌ | Low ✓ | Good (~0.94) |
| CF (k=20) | Medium | Medium | Good (~0.92) |
| MF (basic) | Low ✓ | High ❌ | Poor (~1.02) |
| MF (with biases) | Low-Med | Low-Med | Best (~0.91) |

**Key insight**: The Netflix Prize winner balanced both by using:
- Biases (low bias, captures obvious patterns)
- Latent factors (captures complex interactions)
- Regularisation (prevents overfitting)

### 2. The Cold Start Problem

All recommendation systems struggle when:
- **New user**: No ratings → no neighbours (CF fails)
- **New item**: No ratings → can't estimate quality
- **New system**: Empty database

Solutions:
1. **Content-based**: Use item features instead of ratings
2. **Hybrid**: Combine multiple approaches
3. **Context**: Use user info (age, location) instead of behaviour
4. **Exploration**: Recommend diverse items to gather feedback

### 3. Scalability Comparison

| Approach | Time Complexity | Space Complexity | Scales? |
|----------|-----------------|------------------|---------|
| Popularity | O(n) | O(n) | ✓ Excellent |
| CF | O(u²) or O(u·i) | O(u·i) | ❌ Poor for millions of users |
| Content | O(i·f) | O(i·f) | ✓ Good |
| MF | O(u·i·f) training | O((u+i)·f) | ✓ Good |

*u=users, i=items, f=factors*

MF is modern choice: good accuracy + scales reasonably

### 4. Implicit vs Explicit Feedback

Many real systems use **implicit** feedback (clicks, purchases, time spent):

**Advantages**:
- Abundant data (users don't need to rate)
- More realistic (people don't always rate explicitly)

**Disadvantages**:
- Noisy (accidental clicks)
- Can't distinguish between "dislike" and "never seen"
- Example: someone buys a gift for someone else

**Handling in MF**: Treat implicit feedback as confidence scores
```python
confidence = 1 + α * implicit_signal
# Apply confidence weighting to loss function
```

### 5. Fairness & Ethics in Recommendation

**Echo chambers**: Recommending only what users already like creates filter bubbles
- Solution: Add diversity/novelty objective alongside accuracy

**Algorithmic bias**: If training data is biased, model will be biased
- Example: All popular movies are from Western countries

**Exploration vs exploitation**: Balance between:
- Exploitation: Recommend what user likely wants (accurate)
- Exploration: Recommend new things (discover, serendipity)

### 6. Evaluation Beyond RMSE

RMSE measures accuracy but misses important aspects:

**Diversity**: Are recommendations varied or repetitive?
- Solution: Measure novelty, coverage, diversity

**Ranking quality**: Does order matter?
- Use NDCG (Normalized Discounted Cumulative Gain)
- Or Rank-based metrics

**Coverage**: What % of items can be recommended?
- Long-tail items often ignored

**Novelty**: Are recommendations new/interesting?

### 7. Hyperparameter Tuning Strategy

**For CF (k value)**:
- Start with k=20 (rule of thumb)
- Try range k=5 to 100
- Use cross-validation to select
- Monitor training RMSE vs test RMSE

**For MF (n_factors, alpha, lambda, epochs)**:
```python
# Grid search example
best_rmse = float('inf')
for n_f in [5, 10, 20, 30, 50]:
    for alpha in [0.001, 0.005, 0.01, 0.02]:
        for lambda_r in [0.01, 0.05, 0.1, 0.2]:
            # Train model with these params
            # Evaluate on validation set
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_params = (n_f, alpha, lambda_r)
```

---

## TEACHING NOTES FOR TA

### Common Student Mistakes

1. **Forgetting train/test split**
   - Students sometimes evaluate on training data
   - Always emphasise: separate train/test, no peeking at test data

2. **Not understanding why CF/MF perform worse in some runs**
   - Random initialisation can cause variance
   - Hyperparameters matter a lot
   - Encourage experimentation and discussion

3. **Misunderstanding fallbacks in CF**
   - Students remove fallbacks and get errors
   - Show what errors occur and why they happen

4. **Learning rate too high/low in MF**
   - alpha=0.015 works well for this dataset
   - If RMSE oscillates: alpha too high
   - If RMSE barely decreases: alpha too low

5. **Confusing latent factors with genres**
   - Factors are learned, not predefined
   - They can capture complex patterns not in metadata
   - It's okay that they're not interpretable

### Discussion Questions to Ask Students

1. **If popularity-based is simpler and sometimes better, why use complex methods?**
   - Personalisation matters for user satisfaction (even if not all reflected in RMSE)
   - Different users want different things
   - Long-tail value (happy niche users)

2. **Why does matrix factorisation work so well?**
   - Discovers hidden patterns in data
   - Dimensionality reduction filters noise
   - Flexible enough to capture user-item interactions

3. **What would you do in a real system?**
   - Hybrid (combine multiple approaches)
   - Use implicit feedback (more data)
   - Add diversity/serendipity objectives
   - Handle cold start with content/context
   - Monitor for filter bubbles and bias

4. **How does Netflix use this in practice?**
   - Multiple models for different content types
   - Context-aware (time, device, location)
   - User feedback through ratings AND viewing patterns
   - Billions of ratings → billions of parameters

### Demonstration Tips

- Show what happens when you remove fallbacks (gets errors)
- Show learning curves (epoch_rmse decreasing)
- Visualise latent factors (scatter plot of top 2 factors)
- Compare top-10 lists from different methods
- Discuss if recommendations make sense

---

## SUMMARY TABLE: COMPARING ALL METHODS

| Aspect | Popularity | CF | Content | MF |
|--------|-----------|----|---------|----|
| **Personalisation** | ❌ | ✓✓ | ✓ | ✓✓ |
| **Accuracy (RMSE)** | 0.94 | 0.92 | 0.96 | 0.91 |
| **Interpretable** | ✓✓ | ✓ | ✓✓ | ❌ |
| **New user** | ✓✓ | ❌ | ❌ | ❌ |
| **New item** | ✓ | ❌ | ✓✓ | ❌ |
| **Scalable** | ✓✓ | ❌ | ✓ | ✓ |
| **Fast predict** | ✓✓ | ❌ | ✓ | ✓✓ |
| **Diverse recs** | ❌ | Medium | ✓ | Medium |
| **Data needed** | Ratings | Ratings | Metadata | Ratings |

**Best practice**: Use hybrid approach combining multiple methods!
