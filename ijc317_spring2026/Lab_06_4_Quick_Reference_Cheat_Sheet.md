# Quick Reference: Recommender Systems Cheat Sheet
## For Teaching Assistants - IJC317 Week 6

---

## 1. THE FOUR METHODS AT A GLANCE

### Popularity-Based
```
CONCEPT: "If everyone likes it, you'll like it too"

CODE PATTERN:
  1. Calculate mean rating for each item
  2. Apply regularisation (Bayesian average)
  3. Same recommendations for all users

FORMULA: W_i = [n_i/(n_i+m)] × μ_i + [m/(n_i+m)] × μ

PROS: Simple ✓, Fast ✓✓, Works surprisingly well ✓
CONS: No personalisation ✗, Cold start for items ✗

TEST RMSE: ~0.94
```

### Collaborative Filtering (CF)
```
CONCEPT: "Show me what users similar to me liked"

CODE PATTERN:
  1. Build user × item matrix
  2. Calculate Pearson correlation between users
  3. Find k most similar users
  4. Weight their preferences by similarity

FORMULA: r̂_a,i = r̄_a + Σ[S(a,u) × (r_u,i - r̄_u)] / Σ S(a,u)

PROS: Personalised ✓, Interpretable ✓, Discovers new items ✓
CONS: Cold start ✗, Scales poorly ✗, Sparsity ✗

TEST RMSE: ~0.92
```

### Content-Based
```
CONCEPT: "If you like action movies, watch more action movies"

CODE PATTERN:
  1. Extract item features (genres, actors, keywords)
  2. Represent as TF-IDF vectors
  3. Calculate similarity between items
  4. Recommend items similar to user's liked items

FORMULA: cosine(user_profile, item_features)

PROS: Works for new items ✓, Interpretable ✓
CONS: Needs metadata ✗, Over-specialises ✗, New user cold start ✗

TEST RMSE: ~0.96
```

### Matrix Factorisation (MF)
```
CONCEPT: "Decompose user×item matrix into latent factors"

CODE PATTERN:
  1. Initialise U (user factors) and V (item factors)
  2. For each training rating:
     - Predict: r̂ = U_u · V_i
     - Calculate error: e = r - r̂
     - Update: U_u ← U_u + α(e × V_i - λU_u)
     - Update: V_i ← V_i + α(e × U_u - λV_i)
  3. Repeat for multiple epochs

FORMULA (Full): r̂_ui = μ + b_u + b_i + U_u · V_i

PROS: Scalable ✓, Accurate ✓✓, Fast predictions ✓
CONS: Black box ✗, Hyperparameters matter ✗

TEST RMSE: ~0.91
```

---

## 2. QUICK PROBLEM-SOLVING FLOWCHART

```
STUDENT SAYS...                          SOLUTION
─────────────────────────────────────────────────────────────

"My RMSE is way too high!"
├─ Check if train/test RMSE both high    → Check data loading
├─ Check if only test RMSE high          → Reduce overfitting (↓ n_factors, ↑ lambda)
└─ Check if training RMSE not decreasing → Increase alpha, more epochs

"My CF is slower than MF"
├─ Yes, that's normal!                   → O(n²) vs O(n) per prediction
├─ Can optimize with KDTree              → OK to discuss but beyond scope
└─ MF is why Netflix uses it              → Industry-standard approach

"Why does CF need fallbacks?"
├─ New user/item = no ratings            → No neighbours to compare with
├─ Sparse data = most pairs unrated      → Neighbourhood often empty
└─ Need sensible defaults                → Use item/global mean

"My factors are negative/huge!"
├─ Regularisation too weak               → Increase lambda
├─ Learning rate too high                → Decrease alpha
├─ Initialize differently               → Use smaller scale in random.normal()
└─ This is normal, factors are relative  → Don't worry, still works!

"How do I pick k in CF?"
├─ Plot RMSE vs k                        → Usually peaks at 10-30
├─ Use cross-validation                  → Try k=10, 20, 30
├─ Domain knowledge: sparse data = ?     → Smaller k
└─ Default: k=20                         → Start here

"Should I use explicit or implicit data?"
├─ Explicit: cleaner, sparser           → Better signal but less data
├─ Implicit: noisier, abundant          → More data but noisy
├─ Best: combine both!                  → Weight by confidence
└─ Real systems use implicit             → Clicks are gold

"How do I know if my model is good?"
├─ Compare to baselines                  → Must beat popularity!
├─ Compare to literature                 → Netflix prize ~0.86 RMSE
├─ RMSE is not everything                → Check diversity, coverage, novelty
└─ Offline ≠ online metrics              → A/B test in production
```

---

## 3. HYPERPARAMETER TUNING QUICK GUIDE

### For Collaborative Filtering (k parameter)

| k Value | Bias | Variance | RMSE | When to use |
|---------|------|----------|------|------------|
| 1-5 | High | High | Bad | Very sparse data, extreme filtering |
| **10-20** | **Low** | **Low** | **Good** | **Default choice** |
| 30-50 | Low | Medium | Fair | Abundant co-ratings |
| 100+ | Very Low | High | Poor | Overfitting, noise dominates |

**Rule of thumb**: Start with k=20, try range [10, 30]

### For Matrix Factorisation

| Parameter | Small | Good | Too Large |
|-----------|-------|------|-----------|
| **n_factors** | 5: underfitting (RMSE 1.1+) | 20-50: balanced | 100+: overfitting (RMSE 0.6 train, 1.0 test) |
| **alpha** (learning rate) | 0.001: slow convergence | 0.01-0.02: stable | 0.05+: oscillation, no convergence |
| **lambda** (regularisation) | 0.001: overfitting | 0.05-0.2: balanced | 1.0+: underfitting (RMSE 1.0) |
| **epochs** | 5: incomplete training | 20-30: converged | 50+: diminishing returns |

**Grid search quick version**:
```python
for n_f in [10, 20, 30, 50]:
    for alpha in [0.01, 0.02]:
        for lambda_r in [0.05, 0.1, 0.2]:
            # Train and evaluate
            # Choose best combination
```

---

## 4. COMMON CODE PATTERNS

### Pattern 1: Build Popularity-Based Model
```python
# Step 1: Group and calculate statistics
item_stats = train_df.groupby('item_id')['rating'].agg(['mean', 'count'])

# Step 2: Apply regularisation
m = 25  # minimum votes
mu = train_df['rating'].mean()
item_stats['weighted_score'] = \
    (item_stats['count'] / (item_stats['count'] + m)) * item_stats['mean'] + \
    (m / (item_stats['count'] + m)) * mu

# Step 3: Generate top-k list
top_10 = item_stats.nlargest(10, 'weighted_score')

# Step 4: Evaluate
predictions = test_df['item_id'].map(item_stats['weighted_score']).fillna(mu)
rmse = np.sqrt(mean_squared_error(test_df['rating'], predictions))
```

### Pattern 2: User-User Collaborative Filtering
```python
# Step 1: Create matrix
user_item = train_df.pivot_table(index='user_id', columns='item_id', 
                                  values='rating', fill_value=0)

# Step 2: Calculate similarities
similarities = user_item.corr(min_periods=5)
np.fill_diagonal(similarities.values, 0)

# Step 3: Make prediction
def predict(user_id, item_id, k=20):
    if user_id not in similarities.index:
        return global_mean
    
    # Find neighbours who rated this item
    raters = train_df[train_df['item_id'] == item_id]['user_id'].unique()
    sims = similarities.loc[user_id, raters].dropna()
    top_k = sims.nlargest(k)
    
    # Weight by similarity
    user_mean = train_df[train_df['user_id'] == user_id]['rating'].mean()
    neighbour_ratings = train_df[(train_df['user_id'].isin(top_k.index)) & 
                                 (train_df['item_id'] == item_id)].set_index('user_id')['rating']
    
    weighted_sum = (top_k * (neighbour_ratings - train_df.groupby('user_id')['rating'].mean())).sum()
    return user_mean + weighted_sum / top_k.sum()
```

### Pattern 3: Matrix Factorisation Training
```python
# Initialise
U = np.random.normal(scale=0.1, size=(n_users, n_factors))
V = np.random.normal(scale=0.1, size=(n_items, n_factors))
user_bias = np.zeros(n_users)
item_bias = np.zeros(n_items)

# Train
for epoch in range(epochs):
    for u, i, r in train_data:
        # Predict
        pred = mu + user_bias[u] + item_bias[i] + U[u] @ V[i]
        
        # Error
        e = r - pred
        
        # Update
        user_bias[u] += alpha * (e - lambda_reg * user_bias[u])
        item_bias[i] += alpha * (e - lambda_reg * item_bias[i])
        U[u] += alpha * (e * V[i] - lambda_reg * U[u])
        V[i] += alpha * (e * U[u] - lambda_reg * V[i])

# Evaluate
def predict(u, i):
    return np.clip(mu + user_bias[u] + item_bias[i] + U[u] @ V[i], 1, 5)
```

---

## 5. EVALUATION METRICS QUICK REFERENCE

### RMSE (Root Mean Square Error)
```
RMSE = sqrt(1/n * sum((actual - predicted)^2))

Range: 0 (perfect) to ∞ (terrible)
Interpretation:
  - RMSE 0.85:  Netflix competition level
  - RMSE 0.90:  Good system
  - RMSE 0.95:  Decent (beats popularity)
  - RMSE 1.00:  Global mean prediction
  - RMSE 1.10+: Probably broken

ADVANTAGE: Penalises large errors
DISADVANTAGE: Ignores coverage, diversity, novelty
```

### Other Metrics to Discuss

| Metric | When to use | Example value |
|--------|-------------|----------------|
| **Precision@k** | "How many of top-k are good?" | 0.8 = 8/10 are relevant |
| **Recall@k** | "Did we find all good items?" | 0.6 = found 6/10 good items |
| **NDCG** | "Is ranking good?" | 0.95 = excellent ranking |
| **Coverage** | "Can we recommend all items?" | 0.70 = 70% of catalog recommended |
| **Diversity** | "Are recommendations varied?" | 0.5 = somewhat diverse |

---

## 6. DEBUGGING CHECKLIST

### "My model isn't working!"

```
□ Data loaded correctly?
  └─ Check: len(df), df.head(), df.info()

□ Train/test split correct?
  └─ Check: len(train) + len(test) ≈ len(df), no overlap

□ Baseline working?
  └─ Check: Can you reproduce popularity-based RMSE ~0.94?

□ Prediction function correct?
  └─ Check: pred = function(test_user, test_item)
  └─ Check: 1 ≤ pred ≤ 5 (clipped?)

□ RMSE calculation correct?
  └─ Check: rmse = sqrt(mean_squared_error(...))
  └─ Check: Using correct actual vs predicted

□ Hyperparameters reasonable?
  └─ Check: k in range [5, 100] for CF
  └─ Check: n_factors << n_items
  └─ Check: alpha in range [0.001, 0.1]
  └─ Check: lambda in range [0.01, 1.0]

□ Convergence happening?
  └─ Check: epoch_rmse decreasing? If not → ↑ alpha
  └─ Check: oscillating? If yes → ↓ alpha

□ Overfitting happening?
  └─ Check: train_rmse << test_rmse? If yes → ↑ lambda or ↓ n_factors
```

---

## 7. STUDENT QUESTIONS & QUICK ANSWERS

**Q: Why use Bayesian averaging instead of simple mean?**
A: Prevents overfitting. Movie with 1 rating of 5★ shouldn't rank #1. Formula regresses to global mean when sample size small.

**Q: Why set diagonal to 0 in similarity matrix?**
A: User is always perfectly similar to themselves (correlation = 1), so would always be selected as neighbour, biasing predictions.

**Q: Why shuffle data in SGD?**
A: Different ordering affects convergence. Random order explores parameter space better and reduces local minima.

**Q: Why clip predictions to [1,5]?**
A: Ratings must be in valid range. Math might produce 0.3 or 5.8 which are impossible.

**Q: Why does CF need min_periods=5?**
A: Can't calculate meaningful correlation with only 1-2 co-rated items. Need sufficient overlap.

**Q: Why update biases before factors in MF?**
A: Mathematically independent in gradient, but practically, biases capture global trends first, freeing factors for patterns. Order matters for convergence.

**Q: What if test RMSE worse than training?**
A: Normal! Called overfitting. Model memorised training data. Solution: ↑ regularisation or ↓ model complexity.

**Q: How do I know my hyperparameters are good?**
A: Use validation set (separate from test). Try multiple values, pick one with best validation RMSE.

**Q: Should I normalise ratings?**
A: Not necessary for RMSE (location/scale invariant). Might help for gradient descent convergence though.

**Q: What's the difference between training on all data vs with train/test split?**
A: Train/test shows generalisation. Training on all data shows what's possible but is misleading about real performance.

---

## 8. THE BIAS-VARIANCE TRADEOFF VISUAL

```
                    MODEL COMPLEXITY
        Simple (Popularity)          Complex (MF)
             ↓                            ↓
Test Error  /‾‾\___________
           /     \         \___________
          /        \       /           \___
         /          \_____/  BIAS      /   \
        /                  \          /     \___
       /________            \________/         \________
      Underfitting         SWEET SPOT         Overfitting

GOAL: Find sweet spot where test error is minimized!
- Too much bias: Popularity (underfits)
- Too much variance: Basic MF (overfits)  
- Balanced: MF with biases (best!)

SOLUTION METHODS:
- Reduce variance: Regularisation, fewer factors, more training data
- Reduce bias: More complex model, better features
```

---

## 9. REAL SYSTEM COMPLEXITY

What Netflix/Spotify actually do (beyond scope, but good context):

```
SIMPLE (This Lab):
R ≈ U × V^T

REAL (Production):
R ≈ μ +
    b_u +                      (user bias)
    b_i +                      (item bias)
    U_u · V_i +                (learned interaction)
    W · c +                    (temporal factors)
    X · context +              (device, location, time)
    + ensemble of other models (DNN, gradient boosting, etc)
    + diversity constraints
    + exploration bonus
    + business rules
    + manual overrides
    ...
```

Even Netflix, the pioneers, need multiple models working together!

---

## 10. ONE-PAGE STUDENT SUMMARY

```
WEEK 6: RECOMMENDER SYSTEMS

GOAL: Learn different approaches to recommendation and compare them

THE FOUR METHODS:

1. POPULARITY: Everyone gets same recommendations
   - Simplest, baseline, test RMSE ~0.94
   - Pattern: calculate item averages + regularisation

2. COLLABORATIVE FILTERING: Show me what users like me liked
   - Personalised, interpretable, test RMSE ~0.92
   - Pattern: find similar users, weight their preferences
   - Problem: new users/items have no history

3. CONTENT-BASED: If you like action, watch more action
   - Works for new items, interpretable, test RMSE ~0.96
   - Pattern: use item features, not user history
   - Problem: over-specialises, needs good metadata

4. MATRIX FACTORISATION: Decompose into latent factors
   - Most accurate, scalable, test RMSE ~0.91
   - Pattern: learn U and V matrices via SGD
   - Problem: black box, hyperparameter tuning needed

KEY INSIGHTS:
✓ More complex ≠ always better (compare to baselines!)
✓ Personalisation helps but modest gains (~2-3%)
✓ Overfitting is real (train vs test performance)
✓ Regularisation is essential (biases, lambda parameter)
✓ Hyperparameters matter (k in CF, n_factors/alpha in MF)

NEXT WEEK: Fairness, filter bubbles, ethics in recommendation
```

---

## 11. QUICK FIXES FOR COMMON ISSUES

| Issue | Symptom | Fix |
|-------|---------|-----|
| High RMSE | Predictions all wrong | Check data loading, verify baseline works |
| Oscillating loss | RMSE goes up/down each epoch | Decrease learning rate (alpha) |
| No convergence | RMSE barely changes | Increase learning rate or try different init |
| Overfitting | Big train/test gap | Increase lambda, decrease n_factors |
| Memory error | Code crashes | Use smaller sample, fewer factors, less data |
| NaN predictions | result is NaN | Check for division by zero, missing values |
| Slow training | Takes forever | Use sample of data, reduce epochs, fewer factors |
| Poor neighbourhood | CF fallback always used | Check correlation min_periods, increase k |

---

## 12. QUICK REFERENCE FORMULAS

```
BAYESIAN AVERAGE:
W_i = (n_i/(n_i+m)) * μ_i + (m/(n_i+m)) * μ

PEARSON CORRELATION:
S(a,u) = Σ_i[(r_a,i - r̄_a)(r_u,i - r̄_u)] / √[Σ(r_a,i - r̄_a)² * Σ(r_u,i - r̄_u)²]

CF PREDICTION:
r̂_a,i = r̄_a + Σ[S(a,u) * (r_u,i - r̄_u)] / Σ S(a,u)

MF PREDICTION (Simple):
r̂_ui = U_u · V_i

MF PREDICTION (Full):
r̂_ui = μ + b_u + b_i + U_u · V_i

SGD UPDATE:
U ← U + α(error * V - λ * U)
V ← V + α(error * U - λ * V)

RMSE:
RMSE = √(1/n * Σ(actual - predicted)²)
```

---

**Last Updated**: March 2026
**Target Audience**: Teaching Assistants for IJC317
**Lab Duration**: ~2 hours for all exercises
