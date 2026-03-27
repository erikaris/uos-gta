# IJC317 Week 7 — Teaching Assistant Guide
### Recommender System Challenges, FATES & Societal Impact

> This guide is for TAs running the Week 7 lab session. It covers what students are expected to do, common sticking points, model answers to all discussion questions, and conceptual explanations you can use when students are confused.

---

## Session overview

| Item | Detail |
|------|--------|
| Lecture topic | Real-world RS applications, challenges (cold-start, filter bubbles), FATES, and societal impact |
| Lab topic | Simulating cold-start and data sparsity; measuring filter bubbles via Shannon entropy; fixing them with MMR re-ranking |
| Dataset | MovieLens 100k (downloaded live from GroupLens) |
| Model | Matrix Factorisation with biases, trained via SGD (built in Week 6 lab) |
| Key libraries | `pandas`, `numpy`, `sklearn`, `scipy`, `matplotlib`, `seaborn` |
| Key concepts to consolidate | RMSE, collaborative filtering, latent factors, Shannon entropy, Pareto fronts |

---

## What students did last week (Week 6)

Students built a series of recommender systems ending with a Matrix Factorisation model with biases. They evaluated these purely on RMSE. This week deliberately extends that work by asking: *is RMSE the right thing to optimise?* Make sure students bring or can re-run their Week 6 notebook, or use the `MatrixFactorisationModel` class provided in the Week 7 worksheet.

---

## Lab structure at a glance

```
Setup          → Load MovieLens 100k, split train/test, create 3 training datasets
Exercise 1     → Cold-start & sparsity: train 3 MF models, compare RMSE across datasets and user bins
Exercise 2     → Filter bubbles: implement Shannon entropy, measure genre diversity drop in recommendations
Exercise 3     → MMR: implement re-ranking that balances relevance with diversity; run statistical analysis
```

The lab is designed to be completed sequentially — later exercises depend on models and DataFrames created earlier. If a student has a broken Exercise 1, they will struggle with Exercises 2 and 3.

---

## Exercise 1 — Cold-start & data sparsity

### What students need to do

There is one main `TODO`: create `user_cs_train_df` by selecting 20 random users and stripping all but one of their training ratings. The rest of Exercise 1 (training the models, running the bin analysis) uses provided code.

**Common mistake:** Students often filter from `train_df` instead of `test_df` when choosing the 20 cold-start users. This risks selecting users who don't appear in the test set, making the cold-start evaluation meaningless. The key line is:

```python
all_test_users = test_df['user_id'].unique()   # must be from test_df, not train_df
target_users = np.random.choice(all_test_users, size=20, replace=False)
```

**Another common mistake:** Not using `replace=False` in `np.random.choice`. Without it, the same user could be selected twice, giving fewer than 20 unique cold-start users.

### Discussion questions and model answers

**Q: "What do you notice about the relative performance of the dense vs sparse models on the training set vs the test set?"**

The code now prints a table with four numbers — train RMSE and test RMSE for each model — and this comparison is where the real learning happens. Model A (dense) has a low training RMSE and a slightly higher test RMSE, which is the normal, modest generalisation gap. Model B (sparse) is the interesting case: its training RMSE may look deceptively similar to model A's, because with only 20,000 ratings there is less data to fit and the model does not overfit badly. However, its test RMSE is notably higher than model A's — the sparse model has not seen enough signal to learn reliable user preference patterns and cannot generalise. The key pedagogical point: looking at training RMSE alone would mislead students into thinking the sparse model is fine. Only by comparing both train and test does the problem become clear — the sparse model did not overfit, it simply never learned enough to begin with.

**Q: "Is the cold-start outcome as you would have expected? Why?"**

Yes. The model has only a single training rating for each of the 20 cold-start users, meaning their latent vectors (`U[u]`) are barely updated from their random initialisations. The model has almost no signal to personalise from. In contrast, "warm" users with 30+ ratings have well-trained vectors that genuinely encode their tastes. The result is that cold-start users get predictions closer to the global average than to their actual preferences — hence the higher RMSE.

**Q: "What do you notice about performance as user profile size decreases/increases?"**

RMSE decreases monotonically as profile size increases. Power users (60+ ratings) get the lowest RMSE; cold-start users (1–15 ratings) get the highest. This is the cold-start problem made concrete and quantitative. Every extra rating a user provides helps the model learn their taste more precisely.

**Q: "How could we obtain better performance for cold-start users?"**

- **Content-based filtering:** use item features (genre, director, year) rather than interaction history — doesn't require any ratings from the user
- **Onboarding survey / active learning:** ask new users to rate a small set of carefully chosen "diverse" seed items on sign-up, giving the model immediate signal
- **Demographic proxies:** use age, location, or other available data as a starting point
- **Transfer learning:** initialise a new user's vector from the average of similar users (by demographics or onboarding responses)
- **Hybrid approach:** blend content-based predictions (strong for cold users) with collaborative filtering (strong for warm users); shift the weighting as more ratings accumulate

---

## Exercise 2 — Filter bubbles and echo chambers

### What students need to do

The main `TODO` is implementing `calculate_entropy`. Everything else (the analysis loop, the scatter plot, the worst-offenders table) is provided code that students run and interpret.

**Common mistake:** Not handling the `None` / empty input case in `calculate_entropy`. If `get_genre_distribution` returns `None`, trying to call `np.sum(ps * np.log2(ps))` will raise a `TypeError`. The guard must come first:

```python
if ps is None or len(ps) == 0:
    return 0.0
```

**Another common mistake:** Using `np.log` (natural log) instead of `np.log2`. Shannon entropy is conventionally measured in *bits* using log base-2. Using the natural log gives nats instead of bits — the formula still works mathematically but the values will be different and won't match anything in the literature. The formula in the worksheet explicitly shows `log2`.

**Conceptual confusion to watch for:** Some students think the entropy of the recommendation list should always be *higher* than the history entropy (more diverse = better). Remind them that the y=x line is the ideal — we want recs to *preserve* the user's actual diversity, not necessarily maximise it. A user who only watches comedies should still get mostly comedies recommended; the problem is when a user who watches a *mix* of genres starts getting only one genre recommended.

### Discussion questions and model answers

**Q: "How would you interpret the output of the filter bubble analysis? Is there evidence our algorithm may be causing filter bubbles?"**

Yes. If the average recommendation entropy is lower than the average history entropy, the algorithm is systematically narrowing genre diversity. Most points in the scatter plot will fall *below* the y=x line, meaning the recommendation list is less genre-diverse than the user's actual viewing history. This is direct evidence of filter bubbles: the algorithm has learned users' dominant genre preference and is amplifying it at the expense of variety.

**Q: "What does the scatter plot tell you about the recommender's performance in terms of diversity?"**

Points below the y=x line mean the recommender's top-20 list is less genre-diverse than the user's history. Most points cluster below this line, confirming a systematic diversity reduction. Users with high history entropy (diverse tastes) tend to experience the biggest drop — the algorithm cannot match the breadth of their tastes because it gravitates toward their most-clicked genres. Users with already low history entropy (narrow tastes) see little change — there is not much diversity to lose.

**Q: "Where would the points lie if our system was perfect in terms of preserving diversity?"**

All points would lie exactly on the y=x diagonal line — meaning the genre diversity of every recommendation list perfectly matches the genre diversity of that user's viewing history.

**Q: "Why do you think filter bubbles are happening for the worst offenders?"**

These users have a very strong genre signal in their history — for example, they have rated 50 action movies and 2 comedies. The MF model latches onto the dominant signal (action) when computing their latent vector, and the top-20 items with the highest dot-product with that vector are almost entirely action films. The algorithm is doing exactly what it was trained to do (predict high ratings), but the side effect is a near-homogeneous recommendation list.

**Q: "Is the recommender performing poorly for these filter-bubble users?"**

No — and this is the critical point for students to grasp. RMSE is often *better* for these users than for diverse-taste users, because their preferences are consistent and predictable. The algorithm correctly anticipates that they will rate action films highly. The system is accurate but not good in a broader sense. This is the core tension in recommender system design: optimising purely for predictive accuracy can be socially harmful.

**Q: "If you were the lead developer at a streaming service, how would you modify the ranking logic?"**

Multiple valid answers:
- Apply MMR re-ranking (exactly what Exercise 3 does) to penalise genre repetition within the list
- Hard constraint: cap the number of items per genre in any top-N list (e.g. max 3 action films in a top-20)
- Inject "exploration" slots — reserve 2–3 positions in every recommendation list for items from genres not in the user's top-3, regardless of predicted rating
- Decay: reduce the weight given to highly-rated genres over time so the model is forced to explore
- Accept a small increase in RMSE in exchange for long-term user satisfaction and retention

---

## Exercise 3 — MMR re-ranking

### What students need to do

Two `TODO`s: (1) run MMR at `lambda=1.0` vs `lambda=0.5` for the worst-bubble user and compare entropies; (2) perform statistical analysis comparing standard vs MMR entropies across all users.

**The MMR function is provided** — students do not need to implement it, but they should understand what it does. A useful way to explain it verbally:

> "Instead of just picking the 20 movies with the highest predicted rating, we build the list one movie at a time. Each time we add a movie, we score all remaining candidates by how relevant they are *minus* a penalty for how similar they are to what we've already picked. This forces us to diversify as the list grows."

**Common mistake on the lambda TODO:** Students sometimes set `lambda_val=0` to get "standard" recommendations. Point out that `lambda=0` means *pure diversity* — ignore relevance entirely, which also gives bad recommendations. The correct value for "standard, no diversity" is `lambda=1.0`, which zeroes out the diversity penalty.

**Warning about runtime:** The provided all-users MMR loop is slow because `mmr_rerank` was written for clarity, not efficiency. Warn students in advance: this cell may take several minutes. They should run it and then move on to writing the statistical analysis while it runs.

### Discussion questions and model answers

**Q: "Did we manage to save the user from the filter bubble?"**

Yes, in most cases. The entropy under `lambda=0.5` should be noticeably higher than under `lambda=1.0` for the worst-bubble user. MMR has forced the algorithm to include movies from genres that were underrepresented in the standard top-20. Students can experiment with `lambda=0.3` to see even more diversity (at the cost of some relevance).

**Q: "What does the final scatter plot tell you about the effect MMR is having?"**

Orange dots (MMR) sit higher than blue dots (standard) for most users, meaning MMR has successfully increased recommendation entropy. The improvement is most visible for users who were deepest in a filter bubble (the points that were furthest below y=x). Some orange dots may still be below y=x for users with very narrow tastes — even MMR cannot fully compensate if all top-50 candidates belong to the same genre.

**Q: "Why are there still some users whose datapoint is on the wrong side of the y=x line even with MMR?"**

MMR can only diversify from the candidate pool it has been given (the top-50 items by predicted rating). If a user's taste is so focused that all 50 candidates are the same genre, MMR has nothing diverse to choose from. The fundamental constraint is the candidate set, not the re-ranking algorithm itself. A fix would be to broaden the candidate pool (e.g. top-200 instead of top-50) or to inject random items from underrepresented genres before applying MMR.

**Q: "Statistical analysis — how do you quantify the improvement?"**

The model answer involves three steps:
1. **Mean comparison:** show that average MMR entropy > average standard rec entropy
2. **Paired t-test:** confirm the difference is statistically significant (p < 0.05), not just a random fluctuation. Use paired t-test (not independent samples) because we are comparing two conditions on the *same* users.
3. **Cohen's d:** quantify the practical magnitude of the improvement — a large effect size means the difference is big enough to matter in a real product, not just statistically detectable.

---

## Key concepts — quick reference for when students are confused

### What is latent factor / latent vector?

A latent vector is a compressed representation of a user or item. Instead of saying "this user likes action, dislikes romance, and moderately enjoys comedy," the model learns a vector of numbers (e.g. 30 numbers) that implicitly encode those preferences without us telling it what the dimensions mean. The dot product between a user vector and an item vector is a measure of how well their preferences "align."

### Why does regularisation prevent overfitting?

Without regularisation, the model is free to make its latent vectors as large as it wants. It will make one user's vector enormous to perfectly fit their few ratings — but that vector won't generalise to new items. Regularisation adds a penalty for large weights, forcing the model to "spread the evidence" across more parameters and produce vectors that generalise better.

### Why is Shannon entropy the right measure for diversity?

Entropy treats genre diversity as an information-theoretic problem. A list with 20 action films has low entropy (you can predict each item's genre perfectly — no surprise, no information). A list spread across 10 genres has high entropy (each item's genre is harder to predict — more information, more surprise). This maps naturally to filter bubbles: a bubble-heavy list is low entropy; a diverse list is high entropy.

### What is the Pareto front (mentioned in the lecture)?

When optimising two competing objectives (e.g. taste accuracy vs healthiness), the Pareto front is the set of solutions where you cannot improve one objective without worsening the other. It represents the best achievable trade-offs. In the MMR context, each value of lambda traces a point along the Pareto front between diversity and relevance.

### Why use a paired t-test rather than an independent samples t-test?

Because we are comparing two measurements taken on the same users (standard entropy and MMR entropy for the same person). The paired test controls for individual differences between users — it measures whether MMR changed each user's entropy, rather than whether MMR users as a group are different from standard users as a group (which they aren't, since they're the same people).

---

## Timing guide (90-minute lab)

| Time | Activity |
|------|----------|
| 0–10 min | Setup: students run the data loading and dataset creation cells; check everyone can download MovieLens |
| 10–30 min | Exercise 1: create `user_cs_train_df`, train 3 models (this takes a few minutes per model), run evaluations |
| 30–50 min | Exercise 2: implement `calculate_entropy`, run entropy analysis and scatter plot, discuss findings |
| 50–70 min | Exercise 3: run MMR for worst-bubble user, start the all-users MMR loop (warn: slow), begin statistical analysis |
| 70–85 min | Finish statistical analysis, interpret final scatter plot |
| 85–90 min | Group discussion: "If an algorithm is 100% accurate at predicting what users want to see, is it failing society?" |

---

## Closing discussion prompt

End the session with the lecture's key provocation:

> *"If an algorithm is 100% accurate at predicting what users want to see, is it failing society?"*

Encourage students to think about:
- The difference between short-term preference satisfaction and long-term wellbeing
- Who is responsible for filter bubbles — the algorithm, the platform, or the user?
- Whether diversity should be a legal requirement (the EU AI Act touches on this)
- Whether transparency (explainable recommendations) would help users escape their own bubbles
