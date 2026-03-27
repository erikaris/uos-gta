# IJC317 Week 7 — Recommender System Challenges
### Annotated & Fully Solved Lab Worksheet

> Every line of code is explained with inline comments. All `TODO` exercises are completed and annotated.

---

## What this lab is about

Last week you built increasingly complex recommender systems and measured their accuracy using RMSE. This week you investigate **three real problems** that recommender systems cause or suffer from, using the Matrix Factorisation model you already built:

1. **Exercise 1 — Cold-start & data sparsity:** What happens when the system has little or no data about a user or item?
2. **Exercise 2 — Filter bubbles:** Does the algorithm trap users in a narrow genre bubble?
3. **Exercise 3 — MMR re-ranking:** Can we fix filter bubbles by balancing relevance with diversity?

The key insight running through the whole lab: **a highly accurate recommender is not necessarily a good one.**

---

## Setup — Load data and create datasets

```python
import pandas as pd          # pandas: the go-to library for working with tables of data
import numpy as np           # numpy: fast numerical computing (arrays, maths, random numbers)
from sklearn.model_selection import train_test_split   # splits data into train/test portions
from sklearn.metrics import mean_squared_error         # computes MSE; we take its square root for RMSE


# --- 1. Load the MovieLens 100k ratings dataset ---

# URL pointing to the raw ratings file hosted by GroupLens (the dataset creators)
url = "https://files.grouplens.org/datasets/movielens/ml-100k/u.data"

# The file has no header row, so we name the columns ourselves
columns = ['user_id', 'item_id', 'rating', 'timestamp']

# sep='\t' means columns are separated by a tab character (not a comma)
# names=columns tells pandas to use our list above as the column names
df = pd.read_csv(url, sep='\t', names=columns)
# df is now a DataFrame with 100,000 rows — each row = one user rating one movie


# --- 2. Load movie metadata (titles + genre flags) ---

items_url = "https://files.grouplens.org/datasets/movielens/ml-100k/u.item"

# This file has many columns; we build the full list of names:
# - First 5 are standard metadata (id, title, dates, IMDb link)
# - Then 19 binary genre columns (genre_0 to genre_18), one per genre
#   A value of 1 means the movie belongs to that genre; 0 means it doesn't
item_columns = (
    ['item_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL']
    + [f'genre_{i}' for i in range(19)]   # creates ['genre_0', 'genre_1', ..., 'genre_18']
)

# sep='|' means columns are separated by a pipe character
# encoding='latin-1' is needed because some movie titles contain special characters (e.g. accents)
# that are not valid in the default UTF-8 encoding
movies_metadata = pd.read_csv(items_url, sep='|', names=item_columns, encoding='latin-1')


# --- 3. Train / test split ---

# Hold out 20% of the data for testing; train on the remaining 80%
# random_state=42 is a fixed "seed" so the split is the same every time you run the code
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)


# --- 4. Create the EXTREME SPARSITY dataset ---

# .sample(frac=0.25) randomly keeps 25% of training rows (discards the other 75%)
# This simulates a scenario where we only have a tiny fraction of possible ratings
sparse_train_df = train_df.sample(frac=0.25, random_state=42)
```

---

## Exercise 1 — Cold-start problem & data sparsity

### Background

The **cold-start problem** occurs when the system has no (or very little) data:

- **New user:** no clicks, likes, or purchases → no neighbours in collaborative filtering → no personalisation possible
- **New item:** zero ratings → item may never appear in recommendations

**Data sparsity** is a related issue. Even when data exists, if two users share only one item in common, their Pearson correlation score is mathematically "perfect" (1.0) but statistically meaningless. In matrix factorisation, an item with only one rating causes the model to overfit that item's latent vector to one user's taste.

MovieLens 100k is a "20-core" dataset — every user has at least 20 ratings. Real-world datasets (Amazon, Netflix) often have >99.9% sparsity. We simulate these problems by deliberately depriving the model of data.

---

### TODO (solved) — Create `user_cs_train_df`

```python
# Goal: simulate 20 "new" users who each have only 1 rating in the training set,
# while all other users keep their full training history.

# Step 1: get the list of all unique users who appear in the test set.
# We pick from test users to guarantee these cold-start users appear during evaluation.
all_test_users = test_df['user_id'].unique()

# Randomly choose 20 user IDs without replacement (replace=False means no repeats)
target_users = np.random.choice(all_test_users, size=20, replace=False)

# Step 2: extract ALL training data for users who are NOT in our cold-start set.
# The ~ operator means "NOT" — so ~isin(...) means "not in target_users"
normal_train = train_df[~train_df['user_id'].isin(target_users)]

# Step 3: for each cold-start user, grab just ONE random training rating
cold_start_samples = []
for user_id in target_users:
    # Get all training rows belonging to this user
    user_rows = train_df[train_df['user_id'] == user_id]
    # Guard: some users might have no training data at all (rare, but possible)
    if len(user_rows) > 0:
        # .sample(1) randomly picks exactly 1 row from this user's training data
        cold_start_samples.append(user_rows.sample(1, random_state=42))

# Combine all the single-row samples into one small DataFrame
cold_start_df = pd.concat(cold_start_samples)

# Step 4: concatenate normal users' full data with cold-start users' 1-row data
# reset_index(drop=True) re-numbers the rows from 0 upward (keeps things tidy)
user_cs_train_df = pd.concat([normal_train, cold_start_df]).reset_index(drop=True)

print(f"Full training set size:        {len(train_df):,} ratings")
print(f"Sparse training set size:      {len(sparse_train_df):,} ratings (~25%)")
print(f"Cold-start training set size:  {len(user_cs_train_df):,} ratings")
```

---

### The Matrix Factorisation model (provided, annotated)

```python
class MatrixFactorisationModel:
    """
    Matrix Factorisation (MF) with user and item biases, trained via SGD.

    Core idea: represent every user as a short vector (their 'taste profile')
    and every item as a short vector (its 'feature profile'). The dot product
    of a user vector and an item vector predicts how much that user likes that item.

    Adding biases captures the fact that some users always rate high/low, and
    some items are universally loved/hated, independent of personal taste.

    Prediction formula:
        r_hat(u, i) = mu + b_u + b_i + U[u] · V[i]

    where:
        mu   = global average rating
        b_u  = user bias (are they a harsh or generous rater?)
        b_i  = item bias (is this movie universally good or bad?)
        U[u] = latent factor vector for user u  (length: n_factors)
        V[i] = latent factor vector for item i  (length: n_factors)
    """

    def __init__(self, n_users, n_items, n_factors=30, alpha=0.015, lambda_reg=0.1, epochs=25):
        # n_factors: length of latent vectors — higher = more expressive, but slower to train
        self.n_factors = n_factors

        # alpha: learning rate — how big a step to take each gradient update
        # too high → overshoots the minimum; too low → trains very slowly
        self.alpha = alpha

        # lambda_reg: regularisation strength — penalises large weights to prevent overfitting
        self.lambda_reg = lambda_reg

        # epochs: how many full passes through the training data to perform
        self.epochs = epochs

        # Global average rating — computed from training data later in .fit()
        self.mu = 0

        # Bias vectors: one scalar per user, one scalar per item — initialised at zero
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)

        # Latent factor matrices initialised with small random values
        # scale=0.1 keeps initial values small so no factor dominates at the start
        self.U = np.random.normal(scale=0.1, size=(n_users, n_factors))  # shape: (n_users, n_factors)
        self.V = np.random.normal(scale=0.1, size=(n_items, n_factors))  # shape: (n_items, n_factors)

    def fit(self, train_df):
        """Train the model on a DataFrame of (user_id, item_id, rating) rows."""

        # Compute the global mean rating — our "prior" before any personalisation
        self.mu = train_df['rating'].mean()

        # Store how many training ratings each user has — used later in predict_cs()
        self.user_counts = train_df['user_id'].value_counts().to_dict()

        for epoch in range(self.epochs):
            total_error = 0

            # Shuffle training data each epoch — this is Stochastic Gradient Descent (SGD)
            # Shuffling prevents the model learning spurious patterns from data order
            shuffled_train = train_df[['user_id', 'item_id', 'rating']].sample(frac=1).values

            for u, i, r in shuffled_train:
                # Cast to int — user/item IDs are used as array indices
                u, i = int(u), int(i)

                # --- Forward pass: compute the predicted rating ---
                prediction = (
                    self.mu                          # global average
                    + self.user_bias[u]              # how much this user rates above/below average
                    + self.item_bias[i]              # how much this item is rated above/below average
                    + np.dot(self.U[u], self.V[i])  # personalised taste-match score
                )

                # Error = actual rating minus predicted rating
                # Positive → we under-predicted; negative → we over-predicted
                error = r - prediction

                # --- Backward pass: update parameters using gradient descent ---
                # Each update nudges the parameter to reduce future error
                # The regularisation term (- lambda_reg * param) pulls weights toward zero
                # to prevent any parameter becoming unreasonably large (overfitting)

                # Update user bias
                self.user_bias[u] += self.alpha * (error - self.lambda_reg * self.user_bias[u])

                # Update item bias
                self.item_bias[i] += self.alpha * (error - self.lambda_reg * self.item_bias[i])

                # IMPORTANT: save a copy of U[u] BEFORE updating it.
                # If we updated U[u] first and then used it to update V[i],
                # we'd be using the *new* U[u] — giving a slightly wrong gradient for V[i].
                u_old = self.U[u].copy()

                # Update user latent vector
                self.U[u] += self.alpha * (error * self.V[i] - self.lambda_reg * self.U[u])

                # Update item latent vector — uses the OLD U[u] for consistency
                self.V[i] += self.alpha * (error * u_old - self.lambda_reg * self.V[i])

                # Accumulate squared error for progress reporting
                total_error += error ** 2

            # Print training RMSE at the end of each epoch
            epoch_rmse = np.sqrt(total_error / len(train_df))
            print(f"Epoch {epoch+1}/{self.epochs}: Training RMSE = {epoch_rmse:.4f}")

    def predict(self, user_id, item_id):
        """Standard prediction — returns a rating clamped to [1, 5]."""
        pred = (
            self.mu
            + self.user_bias[user_id]
            + self.item_bias[item_id]
            + np.dot(self.U[user_id], self.V[item_id])
        )
        # np.clip ensures the prediction stays within the valid rating range [1, 5]
        return np.clip(pred, 1, 5)

    def predict_cs(self, user_id, item_id):
        """
        Cold-start-aware prediction.
        If the user has fewer than 5 training ratings, their latent vector is
        poorly trained — we fall back to a simpler prediction that ignores it.
        """
        # How many training ratings did this user have? Default 0 if not found
        user_activity = self.user_counts.get(user_id, 0)

        if user_activity < 5:
            # COLD USER: latent factors are unreliable — use only global mean + biases
            return np.clip(self.mu + self.user_bias[user_id] + self.item_bias[item_id], 1, 5)

        # WARM USER: enough data — use full MF prediction
        try:
            prediction = (
                self.mu
                + self.user_bias[user_id]
                + self.item_bias[item_id]
                + np.dot(self.U[user_id], self.V[item_id])
            )
        except IndexError:
            # If a user/item ID is outside the matrix dimensions, return global mean
            return self.mu

        return np.clip(prediction, 1, 5)

    def evaluate(self, test_df):
        """Compute RMSE on a test DataFrame."""
        # Apply .predict() row-by-row using a lambda function
        # axis=1 means "apply to each row" (axis=0 would be column-by-column)
        predictions = test_df.apply(
            lambda x: self.predict(int(x['user_id']), int(x['item_id'])), axis=1
        )
        rmse = np.sqrt(mean_squared_error(test_df['rating'], predictions))
        return rmse
```

---

### TODO (solved) — Train and compare models

```python
# We need the total number of users and items across the WHOLE dataset
# (not just training) so the embedding matrices are large enough for any ID
n_users = df['user_id'].max() + 1   # +1 because IDs are 1-indexed but arrays are 0-indexed
n_items = df['item_id'].max() + 1

# --- Model A: trained on FULL (dense) training data — our baseline ---
print("Training on dense data...")
model_a = MatrixFactorisationModel(n_users, n_items)   # create a fresh model
model_a.fit(train_df)                                   # train on ~80,000 ratings

# --- Model B: trained on SPARSE data (only 25% of training ratings) ---
print("\nTraining on sparse data...")
model_b = MatrixFactorisationModel(n_users, n_items)   # same hyperparameters, fresh weights
model_b.fit(sparse_train_df)                            # train on only ~20,000 ratings

# --- Evaluate both models on their own TRAINING sets ---
# This tells us how well each model fits the data it was trained on.
# We expect both models to have low training RMSE — but the sparse model's
# training RMSE may look deceptively good because it has fewer ratings to fit.
train_rmse_a = model_a.evaluate(train_df)          # dense model on its full training set
train_rmse_b = model_b.evaluate(sparse_train_df)   # sparse model on its smaller training set

# --- Evaluate both models on the same held-out TEST set ---
# This tells us how well each model generalises to data it has never seen.
# The test set is the same for both, making the comparison fair.
test_rmse_a = model_a.evaluate(test_df)
test_rmse_b = model_b.evaluate(test_df)

print(f"\n--- Sparsity Comparison ---")
print(f"{'Model':<20} {'Train RMSE':>12} {'Test RMSE':>12}")
print(f"{'-'*44}")
print(f"{'Dense (model_a)':<20} {train_rmse_a:>12.4f} {test_rmse_a:>12.4f}")
print(f"{'Sparse (model_b)':<20} {train_rmse_b:>12.4f} {test_rmse_b:>12.4f}")

# What to expect and why:
# - model_a train RMSE: low — it has seen 80,000 ratings and learned well
# - model_a test RMSE:  slightly higher than training (normal generalisation gap)
# - model_b train RMSE: may look similar to model_a — with only 20,000 ratings
#                       there is less data to fit, so the model doesn't overfit badly
# - model_b test RMSE:  HIGHER than model_a — the sparse model has not seen enough
#                       signal to learn reliable user preferences, so it generalises worse
#
# The key insight: comparing ONLY test RMSE would reveal the gap, but comparing
# BOTH shows WHY it exists — the sparse model's training RMSE is not dramatically
# lower (it did not overfit), it simply never learned enough to begin with.

# --- Model C: trained on the cold-start dataset ---
print("\nTraining on cold start data...")
model_c = MatrixFactorisationModel(n_users, n_items)
model_c.fit(user_cs_train_df)
# The 20 cold-start users each have only 1 training rating,
# so their embedding vectors remain nearly untrained random noise.
```

---

### Cold-start evaluation (provided, annotated)

```python
# --- Evaluate model_c on cold-start users vs everyone else ---

# Filter test rows belonging to our 20 cold-start users
cs_users_test = test_df[test_df['user_id'].isin(target_users)]
predictions = cs_users_test.apply(
    lambda x: model_c.predict(int(x['user_id']), int(x['item_id'])), axis=1
)
rmse = np.sqrt(mean_squared_error(cs_users_test['rating'], predictions))
print(f"Cold Start RMSE: {rmse:.4f}")        # expected: HIGH (poor predictions)

# Filter test rows for all other users (the "warm" users with full training data)
non_cs_users_test = test_df[~test_df['user_id'].isin(target_users)]
predictions = non_cs_users_test.apply(
    lambda x: model_c.predict(int(x['user_id']), int(x['item_id'])), axis=1
)
rmse = np.sqrt(mean_squared_error(non_cs_users_test['rating'], predictions))
print(f"Non Cold Start RMSE: {rmse:.4f}")    # expected: LOWER (normal performance)

# Discussion: cold-start RMSE is higher because the model has almost no evidence
# about these users, so it cannot personalise their recommendations effectively.
```

---

### Profile-size bin analysis (provided, annotated)

```python
# Count how many training ratings each user has (their "profile size")
user_train_counts = train_df['user_id'].value_counts()

# Define 4 activity bins:
# 1-15 = "cold start", 16-30 = "warm", 31-60 = "active", 60+ = "power user"
bins = [0, 15, 30, 60, np.inf]
labels = ['1-15 (Cold Start)', '16-30 (Warm)', '31-60 (Active)', '60+ (Power User)']

# Make a working copy of the test set to avoid modifying the original
test_with_bins = test_df.copy()

# Map each user in the test set to how many ratings they had in training
# .fillna(0) handles users who somehow appear in test but not in training
test_with_bins['user_train_count'] = (
    test_with_bins['user_id'].map(user_train_counts).fillna(0)
)

# pd.cut() assigns each count value to its appropriate bin
test_with_bins['bin'] = pd.cut(test_with_bins['user_train_count'], bins=bins, labels=labels)

# For each bin, compute RMSE using the cold-start-aware predict_cs() method
bin_analysis = []
for label in labels:
    bin_data = test_with_bins[test_with_bins['bin'] == label]
    if len(bin_data) > 0:
        predictions = bin_data.apply(
            lambda x: model_a.predict_cs(int(x['user_id']), int(x['item_id'])), axis=1
        )
        rmse = np.sqrt(mean_squared_error(bin_data['rating'], predictions))
        bin_analysis.append({'Activity': label, 'RMSE': rmse, 'Sample Size': len(bin_data)})

perf_df = pd.DataFrame(bin_analysis)
print(perf_df)
# Expected pattern: RMSE decreases monotonically as profile size increases.
# Power users → lowest RMSE; cold-start users → highest RMSE.
#
# How to improve cold-start?
# → Use content-based filtering (doesn't need interaction history)
# → Onboarding survey (ask users to rate a few items on sign-up)
# → Hybrid approach (blend CB and CF; weight CB more heavily early on)
# → Transfer learning from similar users' profiles
```

---

## Exercise 2 — Filter bubbles and echo chambers

### Background

A **filter bubble** happens when a recommender over-optimises for engagement. It keeps showing you content similar to what you've already liked, which:

- Reinforces your existing tastes and beliefs
- Starves you of opposing or diverse perspectives
- Can lead to radicalisation through "borderline" extreme content

In the context of movies, a filter bubble means: if you watch one war film, your entire top-20 recommendation list becomes war films.

We measure diversity using **Shannon entropy** over genres. High entropy = diverse genre spread. Low entropy = genre-heavy, filter-bubble risk.

$$H(X) = -\sum_{i=1}^{n} P(x_i) \log_2 P(x_i)$$

---

### Helper functions (provided, annotated)

```python
def get_genre_distribution(movie_ids, movies_df):
    """
    Given a list of movie IDs, compute the probability distribution over genres.

    Each movie has 19 binary genre flags (genre_0 to genre_18).
    We sum these across all movies in the list to get a genre count vector,
    then divide by the total to get proportions (a probability distribution).

    Returns: pandas Series of non-zero genre probabilities, or None if no data.
    """
    genre_cols = [f'genre_{i}' for i in range(19)]   # ['genre_0', ..., 'genre_18']

    # Keep only rows where item_id is in our list, then sum each genre column across rows
    # .sum() gives the total count of each genre across all selected movies
    genre_counts = movies_df[movies_df['item_id'].isin(movie_ids)][genre_cols].sum()

    # Total number of genre "hits" across all movies and all genres
    total_hits = genre_counts.sum()

    if total_hits == 0:
        return None   # no genre information — avoids division by zero

    # Convert counts to proportions: p(genre) = count(genre) / total_hits
    ps = genre_counts / total_hits

    # Drop genres with probability 0 — log(0) is undefined and they add no information
    return ps[ps > 0]
```

---

### TODO (solved) — Implement `calculate_entropy`

```python
def calculate_entropy(ps):
    """
    Shannon entropy measures how spread out (diverse) a probability distribution is.

    Formula: H(X) = -sum( p(x_i) * log2(p(x_i)) )

    Intuition:
      - All movies are the same genre → ps has one entry = 1.0 → H = 0  (no diversity)
      - Movies spread evenly across 8 genres (each p = 0.125) → H is high (diverse)
      - Higher entropy = more genre diversity = lower filter-bubble risk

    Args:
        ps: pandas Series of non-zero genre probabilities from get_genre_distribution()
    Returns:
        float: entropy value in bits
    """
    # Guard against None or empty input (e.g. user has no genre data)
    if ps is None or len(ps) == 0:
        return 0.0

    # np.log2(ps) computes log base-2 for every element simultaneously
    # The minus sign negates the result (log of a probability <1 is always negative)
    return float(-np.sum(ps * np.log2(ps)))
```

---

### Entropy analysis across users (provided, annotated)

```python
def calculate_entropy_stats(model, train_df, movies_metadata, n_users=100):
    """
    For each user:
      1. Compute entropy of their viewing history (genres they have actually watched)
      2. Compute entropy of the top-20 recommendations the model produces
      3. Compute the "entropy drop" — how much more/less diverse recs are vs history

    Returns a DataFrame with one row per user.
    """
    results = []
    all_users = train_df['user_id'].unique()[:n_users]  # analyse first n_users
    all_item_ids = np.arange(len(model.item_bias))       # all item IDs the model knows about

    for u_id in all_users:

        # 1. History entropy — genres this user has actually rated
        h_ids = train_df[train_df['user_id'] == u_id]['item_id']  # item IDs in history
        h_ps = get_genre_distribution(h_ids, movies_metadata)      # genre distribution
        h_entropy = calculate_entropy(h_ps)                         # entropy of that distribution

        # 2. Recommendation entropy — genres in the model's top-20 predicted items
        # Predict a score for every item in the catalogue for this user
        u_preds = [model.predict(u_id, i) for i in all_item_ids]

        # np.argsort returns indices sorted ascending; [-20:] takes the top-20 highest
        top_20 = np.argsort(u_preds)[-20:]

        r_ps = get_genre_distribution(top_20, movies_metadata)   # genre distribution of recs
        r_entropy = calculate_entropy(r_ps)                       # entropy of recs

        # 3. Entropy drop — percentage change relative to history
        # Negative → recs are LESS diverse than history → filter bubble forming
        entropy_drop = ((r_entropy - h_entropy) / h_entropy) * 100

        results.append({
            'user_id': u_id,
            'history_entropy': h_entropy,
            'rec_entropy': r_entropy,
            'entropy_drop': entropy_drop
        })

    return pd.DataFrame(results)


# Run analysis on the first 100 users
entropy_change_df = calculate_entropy_stats(model_a, train_df, movies_metadata)

avg_h_entropy = entropy_change_df['history_entropy'].mean()
avg_r_entropy = entropy_change_df['rec_entropy'].mean()
avg_drop      = entropy_change_df['entropy_drop'].mean()

print(f"--- Filter Bubble Analysis ({len(entropy_change_df)} Users) ---")
print(f"Average User History Entropy:     {avg_h_entropy:.4f}")
print(f"Average Recommendation Entropy:   {avg_r_entropy:.4f}")
print(f"Average Entropy Drop (%):         {avg_drop:.4f}")
# Interpretation: if avg_r_entropy < avg_h_entropy, recommendations are
# systematically less diverse than what users actually watch — evidence of filter bubbles.
```

---

### Scatter plot (provided, annotated)

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

# One dot per user: x = history entropy, y = recommendation entropy
plt.scatter(
    entropy_change_df['history_entropy'],
    entropy_change_df['rec_entropy'],
    alpha=0.6,       # semi-transparent so overlapping dots are visible
    color='teal'
)

# The y=x diagonal is the "perfect diversity" line:
#   Points ON the line    → recs exactly as diverse as the user's own history
#   Points BELOW the line → recs LESS diverse → filter bubble forming
#   Points ABOVE the line → recs MORE diverse → serendipitous discovery
max_val = max(entropy_change_df['history_entropy'].max(), entropy_change_df['rec_entropy'].max())
min_val = min(entropy_change_df['history_entropy'].min(), entropy_change_df['rec_entropy'].min())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x (no change in diversity)')

plt.title("Echo Chamber Mapping: History vs. Recommendations")
plt.xlabel("Original User History Entropy")
plt.ylabel("Recommendation Entropy")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Discussion answers:
# Q1: Most points fall BELOW y=x → the recommender narrows genre diversity for most users.
# Q2: A perfect diversity-preserving system → all points would lie ON the y=x line.
```

---

### Worst offenders (provided, annotated)

```python
# Sort by entropy_drop ascending (most negative = biggest diversity loss = worst bubble)
worst_bubbles = entropy_change_df.sort_values(by='entropy_drop', ascending=True).head(5)
print("Top 5 Users in the Worst Echo Chambers:")
print(worst_bubbles)

# Discussion answers:
#
# Q1 — Why is this happening?
#    These users have a very strong genre preference in their history.
#    The model latches onto that signal and recommends almost exclusively that genre.
#
# Q2 — Is the recommender performing poorly for these users?
#    No — RMSE is often BETTER for these users. The algorithm correctly predicts
#    what they would rate highly. This is the core tension: high accuracy can mean
#    low diversity, which is bad for the user in the long run.
#
# Q3 — How would you fix this as lead developer?
#    → Use MMR re-ranking (Exercise 3) to penalise genre repetition in the list
#    → Hard constraint: max N movies per genre in any top-20 list
#    → Inject "exploration" items from outside the user's usual genres
#    → Accept a small RMSE penalty in exchange for long-term engagement
```

---

## Exercise 3 — Maximal Marginal Relevance (MMR) re-ranking

### Background

In a standard recommender we simply pick the N items with the highest predicted ratings. As we have seen, this creates filter bubbles. **MMR** fixes this by greedily building the recommendation list one item at a time, balancing two competing forces:

$$\text{MMR}(i) = \lambda \cdot \hat{r}_{u,i} - (1 - \lambda) \cdot \max_{j \in S} \text{Similarity}(i, j)$$

| Symbol | Meaning |
|--------|---------|
| $\hat{r}_{u,i}$ | Predicted rating for user $u$ on item $i$ (relevance) |
| $S$ | Set of items already chosen for the recommendation list |
| $\text{Similarity}(i, j)$ | How genre-similar item $i$ is to item $j$ (redundancy penalty) |
| $\lambda$ | Trade-off: 1.0 = pure relevance, 0.0 = pure diversity, 0.5 = balanced |

---

### MMR function (provided, annotated)

```python
def mmr_rerank(user_id, model, movies_metadata, top_k=20, lambda_val=0.5):
    """
    Produces a top-k recommendation list using MMR re-ranking.

    Steps:
      1. Generate top-50 candidates by predicted rating (relevance pool)
      2. Start the list with the single best item
      3. Greedily add items one at a time using the MMR formula
      4. Repeat until we have top_k items
    """
    genre_cols = [f'genre_{i}' for i in range(19)]

    # Step 1: score ALL items in one efficient matrix operation
    # np.dot(U[user_id], V.T) computes the dot product with every item vector at once
    # model.item_bias is a vector — adding it is a vectorised operation over all items
    u_preds = (
        model.mu
        + model.user_bias[user_id]
        + model.item_bias                         # vector of all item biases
        + np.dot(model.U[user_id], model.V.T)    # dot product with every item simultaneously
    )   # u_preds is now a vector of predicted ratings, one per item

    # Take the top-50 candidates (np.argsort gives ascending order; [::-1] reverses to descending)
    candidates = np.argsort(u_preds)[-50:][::-1]

    # Start the selected list with the single best item (no diversity choice yet)
    selected_indices = [candidates[0]]
    remaining_indices = list(candidates[1:])   # remaining candidates to consider

    # Step 2: greedy MMR selection loop
    while len(selected_indices) < top_k:

        # Compute the cumulative genre distribution of items already selected
        # This tells us which genres are already well-represented in our growing list
        current_distribution = (
            movies_metadata.set_index('item_id')
            .loc[selected_indices, genre_cols]
            .sum()    # sum genre flags across all selected items → shape: (19,)
        )

        best_mmr = -np.inf     # track the best MMR score seen so far this round
        best_candidate = None  # track which item achieved that best score

        for cand_id in remaining_indices:
            # Relevance: how highly the model predicts this user will rate this item
            relevance = u_preds[cand_id]

            # Redundancy: how many of this item's genres are already in our selected list
            # cand_genres is a binary vector (1 = movie has this genre)
            cand_genres = movies_metadata.set_index('item_id').loc[cand_id, genre_cols]

            # Element-wise multiply genre flags by current totals, then sum:
            # High score → shares many genres with already-selected items (redundant)
            # Low score  → adds genres not yet represented (diverse)
            redundancy = (cand_genres * current_distribution).sum()

            # Apply the MMR formula
            # Normalise redundancy by number of selected items to keep it on the same scale
            score = (
                lambda_val * relevance
                - (1 - lambda_val) * (redundancy / len(selected_indices))
            )

            if score > best_mmr:
                best_mmr = score
                best_candidate = cand_id

        # Add the best-scoring candidate and remove it from future consideration
        selected_indices.append(best_candidate)
        remaining_indices.remove(best_candidate)

    return selected_indices   # list of top_k item IDs, balancing relevance and diversity
```

---

### TODO (solved) — Test MMR on the worst-bubble user

```python
# Get the user ID with the biggest entropy drop (most severe filter bubble)
target_user_id = int(entropy_change_df.sort_values('entropy_drop').iloc[0]['user_id'])

# Standard recommendations: lambda=1.0 collapses MMR to pure relevance
# The diversity penalty term is 0, so we just get the top predicted items — no diversity benefit
standard_ids = mmr_rerank(target_user_id, model_a, movies_metadata,
                           top_k=20, lambda_val=1.0)

# Compute entropy of the standard (non-diverse) recommendations
standard_entropy = calculate_entropy(
    get_genre_distribution(standard_ids, movies_metadata)
)

# Balanced recommendations: lambda=0.5 equally weights relevance and diversity
diverse_ids = mmr_rerank(target_user_id, model_a, movies_metadata,
                          top_k=20, lambda_val=0.5)

# Compute entropy of the MMR-balanced recommendations
diverse_entropy = calculate_entropy(
    get_genre_distribution(diverse_ids, movies_metadata)
)

print(f"Standard Entropy:       {standard_entropy:.4f}")
print(f"MMR (λ=0.5) Entropy:    {diverse_entropy:.4f}")
# Expected: diverse_entropy > standard_entropy
# MMR forces the algorithm to spread recommendations across more genres,
# breaking the filter bubble for this user.
#
# You can experiment with other lambda values:
#   lambda=0.7 → relevance-leaning but with some diversity
#   lambda=0.3 → diversity-leaning, accepts bigger relevance sacrifices
```

---

### Run MMR for all users (provided, annotated)

```python
mmr_entropies = []
lambda_val = 0.5   # balanced setting used for all users

for u_id in entropy_change_df['user_id']:
    try:
        # Get the MMR recommendation list for this user
        mmr_ids = mmr_rerank(int(u_id), model_a, movies_metadata,
                              top_k=20, lambda_val=lambda_val)

        # Compute genre distribution of those 20 recommended movies
        mmr_ps = get_genre_distribution(mmr_ids, movies_metadata)

        # Compute and store entropy
        mmr_h = calculate_entropy(mmr_ps)
        mmr_entropies.append(mmr_h)
    except Exception as e:
        # If anything fails for a user, record NaN rather than crashing the whole loop
        mmr_entropies.append(np.nan)

# Add MMR entropy as a new column alongside the standard rec entropy
entropy_change_df['mmr_entropy'] = mmr_entropies
```

---

### Final comparison scatter plot (provided, annotated)

```python
import seaborn as sns

plt.figure(figsize=(10, 8))

# Blue dots: standard recommendations — one dot per user (same as Exercise 2 plot)
sns.scatterplot(
    data=entropy_change_df,
    x='history_entropy', y='rec_entropy',
    alpha=0.4, label='Standard (accuracy only)', color='royalblue'
)

# Orange dots: MMR recommendations for the same users
# If MMR is working, orange dots should sit HIGHER than blue dots
# (closer to or above the y=x line)
sns.scatterplot(
    data=entropy_change_df,
    x='history_entropy', y='mmr_entropy',
    alpha=0.4, label=f'MMR (λ={lambda_val})', color='darkorange'
)

# The y=x parity line — points above this mean recs are more diverse than history
max_val = max(entropy_change_df['history_entropy'].max(),
              entropy_change_df['mmr_entropy'].max())
min_val = max(entropy_change_df['history_entropy'].min(),
              entropy_change_df['rec_entropy'].min())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Parity (no change)')

plt.title("Breaking the Filter Bubble: Standard vs. MMR", fontsize=15)
plt.xlabel("User History Entropy", fontsize=12)
plt.ylabel("Recommendation Entropy", fontsize=12)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

# Discussion answers:
#
# Q1: Orange dots sit higher than blue dots for most users → MMR increases diversity.
#
# Q2: Some orange dots still below y=x because those users have extremely focused tastes.
#     Even with MMR, the top-50 candidate pool is still genre-heavy for them.
#     Diversity gain depends on having diverse enough candidates to choose from.
#
# Q3: See statistical analysis below.
```

---

### TODO (solved) — Statistical analysis of MMR improvement

```python
from scipy import stats   # provides statistical tests

# Drop any NaN rows (users where MMR failed)
clean_df = entropy_change_df.dropna(subset=['rec_entropy', 'mmr_entropy'])

print("\n--- Statistical Analysis of MMR Improvement ---\n")

# --- 1. Descriptive statistics: compare the three entropy means ---
print("Mean entropies:")
print(f"  User history:             {clean_df['history_entropy'].mean():.4f}")
print(f"  Standard recommendations: {clean_df['rec_entropy'].mean():.4f}")
print(f"  MMR recommendations:      {clean_df['mmr_entropy'].mean():.4f}")
# If MMR mean > standard mean, the algorithm improved average diversity across users.

# --- 2. Paired t-test: is the improvement statistically significant? ---
# A paired t-test is appropriate here because we compare two measurements
# on the SAME users (each user has both a standard and an MMR entropy).
# Null hypothesis: there is no difference between standard and MMR entropy.
# If p < 0.05, we reject the null → MMR makes a statistically significant difference.
t_stat, p_value = stats.ttest_rel(clean_df['mmr_entropy'], clean_df['rec_entropy'])
print(f"\nPaired t-test (MMR vs standard):")
print(f"  t-statistic: {t_stat:.3f}")
print(f"  p-value:     {p_value:.4f}")
if p_value < 0.05:
    print("  → Statistically significant improvement (p < 0.05)")
else:
    print("  → No statistically significant difference (p ≥ 0.05)")

# --- 3. Proportion of users who benefit from MMR ---
# What fraction of users have HIGHER entropy under MMR than under standard recs?
improved = (clean_df['mmr_entropy'] > clean_df['rec_entropy']).mean() * 100
print(f"\nUsers with higher entropy under MMR: {improved:.1f}%")

# --- 4. Effect size (Cohen's d) — how LARGE is the improvement? ---
# Cohen's d tells us whether the difference is practically meaningful, not just statistically significant.
# Rule of thumb: 0.2 = small, 0.5 = medium, 0.8+ = large
diff = clean_df['mmr_entropy'] - clean_df['rec_entropy']   # per-user entropy gain
cohens_d = diff.mean() / diff.std()                         # standardised effect size
print(f"\nCohen's d (effect size): {cohens_d:.3f}")
if abs(cohens_d) >= 0.8:
    print("  → Large effect")
elif abs(cohens_d) >= 0.5:
    print("  → Medium effect")
else:
    print("  → Small effect")

# --- 5. Mean absolute entropy gain ---
mean_improvement = diff.mean()
print(f"\nMean entropy gain per user (MMR - standard): {mean_improvement:.4f} bits")

# Summary interpretation:
# - A significant p-value confirms MMR genuinely improves diversity (not just noise).
# - Cohen's d quantifies HOW MUCH better MMR is — important for deciding whether
#   the trade-off (slightly lower RMSE) is worth the diversity gain in a real product.
```

---

## Key takeaways

| Concept | What we found |
|---------|---------------|
| Data sparsity | Fewer training ratings → higher test RMSE; the model cannot generalise |
| Cold-start | Users/items with almost no data get poor predictions; RMSE increases as profile size shrinks |
| Filter bubble | The model narrows genre diversity for most users; high accuracy ≠ diverse recommendations |
| Shannon entropy | A quantitative tool for measuring how genre-diverse a recommendation list is |
| MMR re-ranking | Effectively increases entropy for most users; the λ parameter controls the relevance-diversity trade-off |
| Core tension | **An algorithm that is 100% accurate at predicting what users want may be failing them — and society — by building filter bubbles** |

---

## Discussion question answers

### Exercise 1 — Cold-start & data sparsity

**Q: "What do you notice about the relative performance of the dense vs sparse models on the training set vs the test set?"**

This is best understood by looking at the printed table — four numbers total: train RMSE and test RMSE for each model.

- **Model A (dense) training RMSE** is low — it has seen 80,000 ratings and learned reliable preference patterns.
- **Model A test RMSE** is slightly higher than its training RMSE — a normal, modest generalisation gap.
- **Model B (sparse) training RMSE** may look deceptively similar to model A's — with only 20,000 ratings there is simply less data to fit, so the model does not overfit badly and training RMSE does not collapse dramatically.
- **Model B test RMSE** is notably higher than model A's test RMSE — the sparse model has not seen enough signal to learn reliable user preference patterns and cannot generalise well.

The key insight is that comparing *only* test RMSE reveals that the sparse model performs worse, but looking at *both* train and test RMSE explains *why*: the sparse model's training RMSE is not dramatically lower (it did not overfit — it simply never had enough data to learn properly in the first place).

**Q: "Is the cold-start outcome as you would have expected? Why?"**

Yes. The model has only a single training rating for each of the 20 cold-start users, meaning their latent vectors (`U[u]`) are barely updated from their random initialisations. The model has almost no signal to personalise from. In contrast, "warm" users with 30+ ratings have well-trained vectors that genuinely encode their tastes. The result is that cold-start users get predictions closer to the global average than to their actual preferences — hence the higher RMSE.

**Q: "What do you notice about performance as user profile size decreases/increases?"**

RMSE decreases monotonically as profile size increases. Power users (60+ ratings) get the lowest RMSE; cold-start users (1–15 ratings) get the highest. This is the cold-start problem made concrete and quantitative. Every extra rating a user provides helps the model learn their taste more precisely.

**Q: "How could we obtain better performance for cold-start users?"**

- **Content-based filtering** — use item features (genre, director, year) rather than interaction history; doesn't require any ratings from the user
- **Onboarding survey / active learning** — ask new users to rate a small set of carefully chosen "diverse" seed items on sign-up, giving the model immediate signal
- **Demographic proxies** — use age, location, or other available data as a starting point for preference estimation
- **Transfer learning** — initialise a new user's latent vector from the average of similar users (by demographics or onboarding responses)
- **Hybrid approach** — blend content-based predictions (strong for cold users) with collaborative filtering (strong for warm users); shift the weighting dynamically as more ratings accumulate

---

### Exercise 2 — Filter bubbles

**Q: "How would you interpret the output of the filter bubble analysis? Is there evidence our algorithm may be causing filter bubbles?"**

Yes. If the average recommendation entropy is lower than the average history entropy, the algorithm is systematically narrowing genre diversity. Most points in the scatter plot fall *below* the y=x line, meaning the recommendation list is less genre-diverse than the user's actual viewing history. This is direct evidence of filter bubbles: the algorithm has learned users' dominant genre preferences and is amplifying them at the expense of variety.

**Q: "What does the scatter plot tell you about the recommender's performance in terms of diversity of recommendations?"**

Points below the y=x line mean the recommender's top-20 list is less genre-diverse than the user's history. Most points cluster below this line, confirming a systematic diversity reduction. Users with high history entropy (diverse tastes) tend to experience the biggest drop — the algorithm gravitates toward their most-clicked genres and cannot match the full breadth of their tastes. Users with already low history entropy (narrow tastes) see little change — there is not much diversity to lose.

**Q: "Where would the points lie if our system was 'perfect' in terms of preserving diversity?"**

All points would lie exactly on the y=x diagonal line — meaning the genre diversity of every recommendation list perfectly matches the genre diversity of that user's viewing history. Perfect diversity preservation would not mean every list is maximally diverse; it would mean the system respects and mirrors whatever diversity each individual user already has.

**Q: "Why do you think filter bubbles are happening for the worst offenders?"**

These users have a very strong genre signal in their history — for example, they have rated 50 action movies and 2 comedies. The MF model latches onto the dominant signal when computing their latent vector, and the top-20 items with the highest dot-product are almost entirely action films. The algorithm is doing exactly what it was trained to do (predict high ratings), but the side effect is a near-homogeneous recommendation list.

**Q: "Is the recommender performing poorly for these users?"**

No — and this is the critical insight. RMSE is often *better* for these users than for diverse-taste users, because their preferences are consistent and predictable. The algorithm correctly anticipates that they will rate action films highly. The system is highly accurate but not good in a broader sense. This is the core tension in recommender system design: optimising purely for predictive accuracy can be socially harmful.

**Q: "If you were the lead developer at a streaming service, how would you modify the ranking logic?"**

Multiple valid approaches:
- Apply MMR re-ranking (Exercise 3) to penalise genre repetition within the list
- Hard constraint: cap the number of items per genre in any top-N list (e.g. max 3 action films per top-20)
- Inject "exploration" slots — reserve 2–3 positions for items from genres not in the user's top-3, regardless of predicted rating
- Preference decay: reduce the weight given to highly-rated genres over time, forcing the model to explore
- Accept a small increase in RMSE in exchange for better long-term user satisfaction and retention

---

### Exercise 3 — MMR re-ranking

**Q: "Did we manage to 'save' the user from the filter bubble?"**

Yes, in most cases. The entropy under `lambda=0.5` should be noticeably higher than under `lambda=1.0` for the worst-bubble user. MMR has forced the algorithm to include movies from genres that were underrepresented in the standard top-20. Experimenting with `lambda=0.3` gives even more diversity at the cost of some relevance — the trade-off is controllable.

**Q: "What does the final scatter plot tell you about the effect that MMR is having on recommendations?"**

Orange dots (MMR) sit higher than blue dots (standard) for most users, meaning MMR has successfully increased recommendation entropy. The improvement is most visible for users who were deepest in a filter bubble — the points that were furthest below y=x have moved closer to or above the line. Some orange dots remain below y=x for users with very narrow tastes, but the overall distribution has shifted upward.

**Q: "Why are there still some users whose datapoint is still on the 'wrong' side of the y=x line?"**

MMR can only diversify from the candidate pool it has been given (the top-50 items by predicted rating). If a user's taste is so focused that all 50 candidates belong to the same genre, MMR has nothing diverse to choose from and cannot help. The fundamental constraint is the candidate set, not the re-ranking algorithm itself. A fix would be to broaden the candidate pool (e.g. top-200 instead of top-50) or to inject items from underrepresented genres into the pool before applying MMR.

**Q: "Statistical analysis — how do you quantify how much of an improvement in diversity MMR provides?"**

A complete answer includes:

1. **Mean entropy comparison** — show that average MMR entropy > average standard rec entropy numerically
2. **Paired t-test** — confirm the difference is statistically significant (p < 0.05), meaning it is not a random fluctuation. A paired test is appropriate because we are comparing two conditions on the *same* users
3. **Proportion improved** — what percentage of users have higher entropy under MMR than standard? This gives an intuitive sense of how widely the improvement applies
4. **Cohen's d** — quantifies the practical magnitude of the effect. A large Cohen's d (≥ 0.8) means the improvement is big enough to matter in a real product, not just statistically detectable
5. **Mean entropy gain** — the average number of extra bits of entropy MMR adds per user, giving a concrete scale to the improvement
