# Teaching Assistant Preparation Guide
## IJC317 Week 6: Building and Evaluating Recommender Systems

**Created**: March 2026  
**For**: Teaching Assistants preparing for the week 6 lab session  
**Time to Read**: 10 minutes  
**Time to Master**: 2-3 hours

---

## What You'll Find Here

This folder contains **three comprehensive documents** to prepare you as a teaching assistant:

### 1. **TA_Guide_Week6_Recommender_Systems.md** ← START HERE
**The main guide** (12,000+ words, fully structured)

- **Lecture Summary**: Complete overview of recommender systems theory
  - What are recommender systems and why they matter
  - Four main approaches explained with formulas and intuition
  - Real-world context (Netflix Prize, scale challenges)

- **Lab Overview**: What you're teaching students
  - Learning objectives
  - Dataset structure (MovieLens 100K)
  - Three main exercises

- **Code Walkthrough**: Every single line of code explained
  - Inline comments for 200+ lines of code
  - Clear explanations of what each section does
  - Why decisions were made

- **Solved Exercises**: Complete answers to all lab questions
  - Exercise 1: Popularity-based recommender (with discussion answers)
  - Exercise 2: Collaborative filtering (with performance analysis)
  - Exercise 3: Matrix factorisation basic version
  - Exercise 4: Netflix Prize full model with biases

- **Key Concepts**: Teaching points you should emphasize
  - Bias-variance tradeoff
  - Cold start problem
  - Scalability comparison
  - Evaluation metrics beyond RMSE
  - Fairness and ethics considerations

### 2. **IJC317_Week6_Complete_Solutions.py** ← RUN THIS
**Fully executable Python script** (all exercises solved, fully commented)

- Copy-paste ready code for all four exercises
- Every line has inline comments explaining what it does
- Includes data loading, training, and evaluation
- Produces actual RMSE numbers
- Can be run as Jupyter notebook cells or standalone script

**Key features**:
- Reproducible results (uses `random_state=42`)
- Prints intermediate results and explanations
- Shows expected outputs
- Includes discussion answers at end of each section

### 3. **Quick_Reference_Cheat_Sheet.md** ← USE DURING LABS
**One-page (actually multi-page) quick reference** for in-the-moment answers

- **The four methods at a glance**: Side-by-side comparison
- **Problem-solving flowchart**: "Student says X, what do I do?"
- **Hyperparameter tuning guide**: Quick tables for k, alpha, lambda
- **Common code patterns**: Copy-paste solutions
- **Common questions & answers**: Pre-prepared responses
- **Quick fixes for issues**: "My RMSE is too high! What do I do?"
- **Debugging checklist**: Systematic troubleshooting
- **All formulas**: For quick lookup

---

## 30-Second Overview

### The Four Recommender Systems (in order of complexity)

| Method | Idea | Test RMSE | Code Lines |
|--------|------|-----------|-----------|
| **Popularity** | Everyone gets same recommendations | 0.94 | ~20 |
| **Collaborative Filtering** | Show me what similar users like | 0.92 | ~50 |
| **Content-Based** | If you like action, watch more action | 0.96 | ~40 |
| **Matrix Factorisation** | Decompose into latent factors | 0.91 | ~80 |

**Winner**: Matrix factorisation (most accurate, fastest predictions)  
**Teaching Focus**: All four, with emphasis on understanding tradeoffs  
**Key Insight**: More complex ≠ always better (but MF is best here)

---

## What Students Will Learn

By the end of the lab, students should be able to:

✓ Implement and evaluate four different recommender systems  
✓ Understand the strengths and limitations of each  
✓ Apply appropriate evaluation metrics  
✓ Tune hyperparameters for better performance  
✓ Explain the bias-variance tradeoff in context of recommendations  
✓ Discuss real-world implications (filter bubbles, cold start, fairness)  

---

## Your Role as TA

### Before the Lab (30 min preparation)
1. Read this README
2. Skim the **Lecture Summary** section of the main guide
3. Understand the four methods conceptually
4. Run the **Complete Solutions** script yourself to see expected outputs
5. Read through the **Quick Reference** to understand common issues

### During the Lab
1. **Help students understand the concept first** (not just code)
2. Use the **Problem-Solving Flowchart** when students get stuck
3. Reference **Common Code Patterns** for copy-paste solutions
4. **Don't just give answers** - guide them to understand
5. Point to the **Cheat Sheet** for quick formula lookups
6. Use the **Discussion Questions** from main guide to deepen thinking

### Office Hours / Follow-up
1. Help students understand **why** one method beats another
2. Discuss real-world implications (Netflix, Spotify use of these techniques)
3. Encourage experimentation with hyperparameters
4. Suggest extensions (hybrid systems, diversity metrics)

---

## Common Misconceptions to Address

**"More complex models are always better"**  
→ No! Popularity baseline (RMSE 0.94) vs MF (RMSE 0.91) is only ~3% improvement  
→ Simpler often better for production (faster, easier to maintain, less overfitting)

**"Higher training RMSE is bad"**  
→ No! You actually WANT lower test RMSE even if training RMSE is higher  
→ Gap between train/test RMSE = overfitting (happens in MF without biases)

**"Latent factors are like genres"**  
→ Not exactly. Factors are learned patterns, not predefined categories  
→ They can capture complex combinations (e.g., "serious sci-fi for adults")

**"Correlation = causation in similarity"**  
→ No. Just because two users rate similarly doesn't mean they have the same interests  
→ Could be coincidence, or they rate based on different criteria

**"More data = always better"**  
→ Mostly yes, but noisy data can hurt (implicit feedback)  
→ Also: data quality > data quantity

---

## Expected Student Questions (Already Answered)

See the **"Quick Reference"** section 12 for pre-written answers to:
- "Why use Bayesian averaging?"
- "Why set diagonal to 0?"
- "Why shuffle data?"
- "Why clip predictions?"
- And 10+ more!

---

## Lab Exercise Breakdown

### Exercise 1: Popularity-Based (20 min)
**What students do**:
- Load MovieLens 100K dataset
- Calculate item statistics (mean, count)
- Implement Bayesian averaging formula
- Evaluate on test set

**Expected output**: RMSE ~0.94

**Common issues**:
- Don't understand why regularisation helps (show concrete example)
- Forget to handle movies not in test set (must use fallback mean)
- Try to rank by simple mean instead of weighted mean

### Exercise 2: Collaborative Filtering (40 min)
**What students do**:
- Build user × item matrix
- Calculate Pearson correlations
- Implement k-nearest neighbours prediction
- Test different k values
- Explain fallbacks

**Expected output**: RMSE ~0.92 (best k=10-30)

**Common issues**:
- Don't understand why correlation on transposed matrix
- Forget fallbacks → crashes on new users
- Over/under estimate k value
- Don't understand why neighbours improve over popularity

### Exercise 3: Matrix Factorisation Basic (30 min)
**What students do**:
- Initialise U and V matrices
- Implement SGD training loop
- Fill in prediction, error, update formulas
- Evaluate on test set

**Expected output**: Training RMSE ~0.80-0.85, Test RMSE ~1.00-1.05

**Key learning**: Overfitting! This shows why regularisation matters.

**Common issues**:
- Think test RMSE worse than training is bad (it's not! normal overfitting)
- Don't understand dot product = prediction
- Forget np.clip to [1,5]
- Learning rate wrong → no convergence or oscillation

### Exercise 4: Full Netflix Model (20 min)
**What students do**:
- Add user and item bias terms
- Update biases before factors (order matters!)
- Evaluate on test set

**Expected output**: Test RMSE ~0.91 (much better!)

**Key learning**: Biases fix overfitting. Full model is balanced.

**Common issues**:
- Forget to initialize biases to zeros
- Update factors before biases (wrong order, converges slower)
- Don't see improvement (make sure you're adding biases!)

---

## Expected Timings

| Activity | Time |
|----------|------|
| Lecture (you attended) | 50 min |
| Exercise 1 (Popularity) | 20 min |
| Exercise 2 (CF) | 40 min |
| Exercise 3 (MF Basic) | 30 min |
| Exercise 4 (MF Full) | 20 min |
| Discussion & Q&A | 20 min |
| **Total** | **~180 min (3 hours)** |

If time is short, prioritize: 1 → 2 → 4 (can skip basic MF in exercise 3)

---

## Your Success Checklist

- [ ] You've read this README (5 min)
- [ ] You've skimmed the Lecture Summary (10 min)
- [ ] You understand the 4 methods conceptually (10 min)
- [ ] You've run the Solutions script and saw it work (20 min)
- [ ] You've reviewed Common Issues section (10 min)
- [ ] You have Quick Reference printed/available (ready)
- [ ] You understand bias-variance tradeoff (key concept!)
- [ ] You can explain why each method has its formula
- [ ] You know the expected RMSE values (0.94, 0.92, 0.91, 0.96)
- [ ] You're ready to help students! ✓

---

## Key Formulas to Know (for reference)

```
BAYESIAN AVERAGE:
W_i = (n_i / (n_i + m)) * μ_i + (m / (n_i + m)) * μ

PEARSON CORRELATION:
S(a,u) = Σ[(r_a,i - r̄_a)(r_u,i - r̄_u)] / √[Σ(r_a,i - r̄_a)² * Σ(r_u,i - r̄_u)²]

CF PREDICTION:
r̂ = r̄_a + Σ[S(a,u) * (r_u,i - r̄_u)] / Σ S(a,u)

MF FULL MODEL:
r̂ = μ + b_u + b_i + U_u · V_i

SGD UPDATE:
U ← U + α(error * V - λ * U)
V ← V + α(error * U - λ * V)

RMSE:
RMSE = √(1/n * Σ(actual - predicted)²)
```

---

## Important Context for Students

**Why this matters**:
- Netflix Prize won with algorithms like this (2006-2009)
- Spotify, Amazon, YouTube all use similar approaches
- Recommendation systems are multi-billion dollar business
- Algorithms affect what information billions of people see
- Filter bubbles, bias, fairness are serious concerns

**This is applied ML, not just theory**

---

## Resources You Have

1. **Lecture PDF**: IJC317_Lecture_week_6__Recommender_Systems_Principles_and_Techniques_.pdf
   - Covers all theory from different angle
   - Good for students who want detailed references

2. **Lab Notebook**: Copy_of_IJC317_Lab_week_6__Building_and_Evaluating_Recommender_Systems_.ipynb
   - Actual notebook students are working from
   - Reference for expected output formats

3. **This folder**: 3 comprehensive guides
   - Use them! They're your support system

---

## Final Checklist Before Lab

**30 minutes before lab starts**:
- [ ] Read the Quick Reference front page
- [ ] Run the Solutions script (verify it works in your environment)
- [ ] Note down expected RMSE values
- [ ] Have the Problem-Solving Flowchart open
- [ ] Review Discussion Questions section
- [ ] Know where the Quick Reference formulas are
- [ ] Take a deep breath - you've got this! 💪

**During lab**:
- [ ] Encourage students to understand concepts first
- [ ] Reference these guides when stuck
- [ ] Ask discussion questions from main guide
- [ ] Celebrate wins (especially when they get MF working!)
- [ ] Make it fun - recommendation systems are cool!

---

## Questions While Preparing?

**Common TA questions**:

*Q: What if a student gets different RMSE values?*  
A: Normal! Different random seed, library versions, etc. within ±0.02 is fine. If very different, check they're loading/splitting data same way.

*Q: Should I give them the code?*  
A: No! Guide them to write it. But copy-paste patterns from Quick Reference are fine for syntax help. Focus on understanding.

*Q: What if they want to implement other methods?*  
A: Encourage it! Content-based is good extension. But focus on understanding the core four first.

*Q: How do I explain why MF is better if I don't fully understand latent factors?*  
A: You don't need to! You can say "it's a mathematical decomposition that captures patterns we can't see directly" - that's honest and sufficient.

*Q: They're getting NaN predictions, what do I do?*  
A: Check: division by zero (empty neighbours in CF), missing user/item IDs, or data loading error. Use the debugging checklist!

---

## You've Got Everything You Need

This preparation materials cover:
- ✓ Full theory explanation
- ✓ All code implementations
- ✓ Every exercise solution  
- ✓ Common questions and answers
- ✓ Quick reference for in-the-moment help
- ✓ Debugging checklists
- ✓ Discussion points

**You're ready to teach this lab!**

---

## Next Steps

1. **Right now**: Read the first part of TA_Guide (Lecture Summary)
2. **In 10 min**: Run the Complete Solutions script
3. **In 20 min**: Review the Quick Reference
4. **Day before lab**: Do a full walkthrough of the lab yourself
5. **Lab day**: You're ready!

---

**Good luck! You're going to do great.** 🎓

For questions, refer to:
- Theory questions → **TA_Guide Lecture Summary**
- Code questions → **Complete Solutions.py or Quick Reference**  
- Student stuck? → **Problem-Solving Flowchart in Quick Reference**

---

*Last updated: March 2026*  
*Created for: IJC317 Data Science and AI in Practice, University of Sheffield*
