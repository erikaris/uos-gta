# Lab 05 TA Session Materials - START HERE

## 📚 What You Have

I've created **3 comprehensive documents** to help you prepare for your TA session:

### 1. **Lab_05_Changes_Summary.md** ⭐ START HERE
- What changed in the updated worksheet
- YOLO8 → YOLO12 update
- SAM 1.0 → SAM 2.1 update
- Migration guide from old materials
- **Read this first (10 minutes)**

### 2. **Lab_05_Updated_TA_Guide.md** 📖 DETAILED REFERENCE
- Complete explanation of all 4 activities
- EVERY LINE OF CODE has inline comments
- Accessible language, step-by-step
- For detailed preparation and understanding
- **Read this second (30-40 minutes)**

### 3. **Lab_05_Updated_Quick_Reference.md** ⚡ QUICK LOOKUP
- Visual diagrams and flowcharts
- Code patterns you'll need
- Common issues and solutions
- FAQ with student questions
- Hyperparameter cheat sheet
- **Use during your session for quick answers**

---

## 🚀 How to Use These Documents

### Option A: Fast Prep (45 minutes)
1. Read **Changes Summary** (10 min)
2. Skim **Quick Reference** (15 min)
3. Review code comments in **Detailed Guide** (20 min)
4. You're ready!

### Option B: Thorough Prep (90 minutes)
1. Read **Changes Summary** (10 min)
2. Read **Detailed Guide** front to back (60 min)
3. Reference **Quick Reference** as needed (20 min)
4. You're very prepared!

### Option C: During Session
1. Have **Quick Reference** open for quick lookup
2. Have **Detailed Guide** for detailed explanations
3. Reference **Changes Summary** if explaining updates

---

## 🎯 Lab 05 at a Glance

**What This Lab Is About:**
Learning three powerful computer vision models beyond basic CNNs

### Four Activities:

| Activity | Model | Time | Key Learning |
|----------|-------|------|--------------|
| 1 | ViT (torchvision) | 40 min | Transfer learning, fine-tuning |
| 2 | ViT (HuggingFace) | 30 min | Different API, same concept |
| 3 | YOLO12 | 30 min | Object detection, zero-shot |
| 4 | SAM2.1 | 30 min | Segmentation, flexible prompts |

**Expected Results:**
- ViT: 95%+ accuracy (vs 71% with LeNet!)
- YOLO12: Real-time detection (30+ FPS)
- SAM2.1: Flexible segmentation with any prompt

---

## 🔍 Key Updates in New Worksheet

```
YOLO: v8 → v12 (2025 version!)
SAM: v1 → v2.1 (latest!)
Code: Simplified → Complete + Production-quality
Training: No progress bar → tqdm progress bar
```

---

## ⚠️ Most Important Things to Know

1. **Transfer Learning is Key**
   - Pre-trained on ImageNet
   - Fine-tune only last layers
   - 4× faster training!

2. **YOLO12 (NEW!)**
   - Not YOLO8 anymore
   - Same API, better results
   - Zero-shot (no training)

3. **SAM2.1 (NEW!)**
   - Latest generation
   - More efficient than SAM1.0
   - Same prompting methods

4. **Layer Freezing Works**
   - Preserve ImageNet knowledge
   - Only train last layer
   - Prevents overfitting

---

## 🎓 Teaching Strategy

### What to Emphasize:
1. "Transfer learning is the biggest breakthrough"
2. "Fine-tuning is 4× faster than training from scratch"
3. "YOLO is real-time, uses latest version"
4. "SAM works on anything - zero-shot!"

### Live Demo Ideas:
1. Show ViT training with tqdm progress bar
2. Run YOLO on your own image
3. Try SAM with different prompts
4. Compare accuracies: LeNet (71%) vs ViT (95%)

### Common Misconceptions to Address:
- "YOLO can detect anything" → Only 80 COCO classes
- "SAM needs labels" → No! Zero-shot
- "ViT is slow" → Actually fine-tunes faster than training from scratch
- "I have to train these models" → No! Load and run (except ViT needs fine-tuning)

---

## 📝 Session Outline (2 hours)

```
0:00-0:10    Overview (Lab purpose, 4 activities)
0:10-0:40    Activity 1: ViT Fine-tuning (code walkthrough)
0:40-1:00    Activity 2: ViT HuggingFace (show differences)
1:00-1:30    Activity 3: YOLO12 (live demo)
1:30-1:50    Activity 4: SAM2.1 (try different prompts)
1:50-2:00    Q&A + Summary
```

---

## ✅ Pre-Session Checklist

- [ ] Read Changes Summary (10 min)
- [ ] Skim Quick Reference (15 min)
- [ ] Review code comments in Detailed Guide
- [ ] Test running one Activity on your machine
- [ ] Prepare demo images/videos
- [ ] Know the 4 key concepts
- [ ] Practice explaining transfer learning
- [ ] Have the 3 documents open during session

---

## 🆘 Quick Reference: Where to Find Things

**Want to know about...?**

| Topic | Document | Section |
|-------|----------|---------|
| What changed? | Changes Summary | Top section |
| How does code work? | Detailed Guide | Part 3 |
| Code patterns? | Quick Reference | Code patterns section |
| Common issues? | Quick Reference | Troubleshooting |
| Models comparison? | Quick Reference | Comparison table |
| Layer freezing? | Detailed Guide | Section 1.7 |
| YOLO12 info? | Quick Reference | YOLO section |
| SAM2.1 info? | Quick Reference | SAM section |

---

## 💡 Pro Tips

1. **Have Quick Reference open** during your session
2. **Run code yourself first** - you'll understand better
3. **Use tqdm progress bar** as a cool demo feature
4. **Show the accuracy jump** - LeNet 71% → ViT 95%!
5. **Emphasize zero-shot** for YOLO and SAM
6. **Freeze layers carefully** - explain why it matters

---

## 🎬 Good Luck!

You've got this! The materials are comprehensive and ready to go.

**Key Takeaway:** 
This lab shows the power of transfer learning - using pre-trained models to achieve much better results much faster. That's the modern way to do deep learning!

---

Questions? Check the FAQ in Quick Reference! 📖
