# Lab 05 - What Changed? (Updated vs Original)

## Quick Summary of Updates

The lecturer has made several **IMPORTANT changes** to the Lab 05 worksheet. Here's what's different:

---

## MAJOR CHANGES BY ACTIVITY

### Activity 1: ViT with torchvision ✅

**What stayed the same:**
- Core concept: Fine-tune ViT on CIFAR-10
- Training/evaluation functions
- Model selection by validation accuracy
- Freezing strategy

**What changed:**
```python
# BEFORE (Original):
# - Used simplified code examples
# - Limited explanation of layers

# AFTER (Updated):
# - Full functional code
# - Better training functions with tqdm progress
# - Explicit layer freezing
# - Complete logging and results tracking
```

---

### Activity 2: ViT with HuggingFace ✅

**What stayed the same:**
- Use HuggingFace transformers library
- Same CIFAR-10 dataset
- Same training philosophy

**What changed:**
```python
# BEFORE: Incomplete code, minimal explanation
# AFTER:  Full working code, complete functions

# NEW: Layer freezing function
def free_layers(vit_model, first_n_layers=6):
    # Freeze first N blocks
    # Unfreeze last block + head
    # More detailed than before

# NEW: Updated forward pass handling
outputs = model.forward(images)
loss = criterion(outputs.logits, labels)  # Note: .logits attribute!
```

---

### Activity 3: YOLO - MAJOR UPDATE! 🚨

**BIGGEST CHANGE: Updated to YOLO12 (2025 version)**

```python
# BEFORE (Original):
# model = YOLO('yolov8m.pt')  # YOLO v8 (older)

# AFTER (Updated):
# model = YOLO('yolo12n.pt')   # YOLO v12 (2025!)
```

**What changed:**
- Model versions available: yolo12n, yolo12s, yolo12m, yolo12l, yolo12x
- Better accuracy and speed than YOLO8
- Same basic API (still zero-shot, just load and run)
- Automatic bounding box visualization still works

**What stayed the same:**
- Zero-shot inference (no training)
- Works on 80 COCO classes
- Real-time detection on images and videos
- Results object structure

---

### Activity 4: SAM - UPDATED! ✅

**What changed:**
```python
# BEFORE (Original):
# model = SAM("sam_b.pt")        # SAM 1.0

# AFTER (Updated):
# model = SAM("sam2.1_b.pt")     # SAM 2.1 (latest!)
```

**New features in SAM 2.1:**
- More efficient (faster)
- Better accuracy
- Video support (mentioned but not fully used)
- Same prompting API (bboxes, points, text)

**Code differences:**
```python
# Available SAM models:
# "sam_b.pt"      = Original SAM base
# "sam_l.pt"      = Original SAM large
# "sam2.1_b.pt"   = SAM 2.1 base (NEW!)
# "sam2.1_l.pt"   = SAM 2.1 large (NEW!)
# "sam2_b.pt"     = SAM 2 base (intermediate)
```

**What stayed the same:**
- Zero-shot segmentation concept
- Prompting methods (bbox, points, text)
- Visualization methods

---

## CODE STRUCTURE IMPROVEMENTS

### Better Training Function

**BEFORE:**
```python
# Minimal function, less clear
def train(model, data_loader):
    for images, labels in data_loader:
        # ... basic training ...
```

**AFTER:**
```python
# Complete function with tqdm progress
def train(model, data_loader, criterion, optimizer, device, epoch):
    model.train()
    
    # Initialize tracking
    loss_history = []
    y_pred, y_true = torch.Tensor(), torch.Tensor()
    
    # Progress bar with time estimate
    for images, labels in tqdm(data_loader, total=len(train_loader)):
        # ... complete training loop ...
        # ... includes logging and accuracy tracking ...
    
    # Return metrics
    return {"loss": avg_loss, "accuracy": accuracy}
```

### Better Model Selection

**BEFORE:**
- Comment: "Warning: Actually we don't do model selection in this code example"
- No proper validation split

**AFTER:**
- Explicit best model tracking
- Save best model by validation accuracy
- Clear epoch selection

```python
# AFTER (Updated):
best_model = None
best_acc = -1
best_epoch = -1

for epoch in range(NUM_EPOCHS):
    train_results = train(...)
    val_results = evaluate(...)
    
    if val_results['accuracy'] > best_acc:
        best_acc = val_results['accuracy']
        best_epoch = epoch
        torch.save(...)  # Save best
```

---

## TECHNICAL IMPROVEMENTS

### Device Handling
**BEFORE:** Assumed GPU was available  
**AFTER:** Explicit device management
```python
# AFTER (Updated):
if "cuda" in DEVICE:
    vit_model.cuda()
```

### Dataset Preprocessing
**BEFORE:** Basic transforms  
**AFTER:** More explicit, with better comments
```python
# AFTER (Updated):
transforms.Normalize(
    (0.4914, 0.4822, 0.4465),  # ImageNet means (R,G,B)
    (0.2023, 0.1994, 0.2010)   # ImageNet stds (R,G,B)
)
```

### Data Subset Function
**BEFORE:** Simple implementation  
**AFTER:** Better documented, clearer purpose
```python
def get_subset(dataset, percentage):
    """
    Take random sample of dataset
    Why: ViT training is slow!
    """
    # Better comments and structure
```

---

## NEW/EXPANDED SECTIONS

### Activity 3 Now Includes:
- ✅ Single image detection (before)
- ✅ Video detection (before)
- ✅ Extracting class names from results (before)
- 🆕 Better result investigation code
- 🆕 More detailed visualization

### Activity 4 Now Includes:
- ✅ Segment everything (before)
- ✅ Bbox prompts (before)
- ✅ Point prompts (before)
- 🆕 SAM 2.1 specific examples
- 🆕 Better visualization

---

## WHAT DIDN'T CHANGE (Core Concepts)

### Still Teaching:
1. **Transfer Learning** - Reuse pre-trained models
2. **Fine-tuning** - Train only last layers
3. **Layer Freezing** - Preserve ImageNet knowledge
4. **Zero-shot** - YOLO and SAM need no training
5. **Prompting** - Guide models with flexible inputs

### Same Results Expected:
- ViT: 95%+ accuracy on CIFAR-10
- YOLO: Detects objects in real-time
- SAM: Segments any object with prompts

---

## MIGRATION GUIDE: If You Use Old Notes

### Change 1: Model Loading
```python
# Old:
model = YOLO('yolov8m.pt')

# New:
model = YOLO('yolo12n.pt')
```

### Change 2: SAM Model
```python
# Old:
model = SAM("sam_b.pt")

# New:
model = SAM("sam2.1_b.pt")
```

### Change 3: Training Functions
```python
# Old: Simple functions
# New: Complete functions with tqdm and logging

# Everything else stays the same!
```

---

## SUMMARY TABLE: Before vs After

| Aspect | Before | After |
|--------|--------|-------|
| **ViT torchvision** | Basic | Complete with logging |
| **ViT HuggingFace** | Minimal | Full implementation |
| **YOLO** | YOLO8 | YOLO12 (2025!) |
| **SAM** | SAM 1.0 | SAM 2.1 (latest!) |
| **Training code** | Simplified | Production-quality |
| **Progress bars** | No | Yes (tqdm) |
| **Model selection** | Comment warning | Full implementation |
| **Comments** | Sparse | Comprehensive |

---

## TEACHING IMPACT: What This Means for Your TA Session

### Good News:
✅ Code is now more complete  
✅ Better for students to learn from  
✅ More realistic training example  
✅ Latest models (YOLO12, SAM2.1)  

### For You (TA):
- More to explain (good - shows modern practices)
- Better code to reference
- Can show real training in action (tqdm progress)
- Can explain model selection properly

### Topics to Emphasize:
1. **YOLO version update** - Why YOLO12 is better
2. **Progress tracking** - How tqdm helps
3. **Model selection** - Saving best model, not last model
4. **Complete workflow** - From data → training → evaluation

---

## SIDE-BY-SIDE: Key Code Changes

### Training Loop

```python
# BEFORE (Old):
for epoch in range(NUM_EPOCHS):
    train(model, train_loader)
    evaluate(model, test_loader)

# AFTER (Updated):
for epoch in range(NUM_EPOCHS):
    train_results = train(vit_model, train_loader, criterion, 
                         optimizer, DEVICE, epoch)
    val_results = evaluate(vit_model, test_loader, criterion, DEVICE)
    print(f"Epoch {epoch}, Val loss: {val_results['loss']:.4f}")
    
    if val_results['accuracy'] > best_acc:
        best_acc = val_results['accuracy']
        best_epoch = epoch
        # Save model
```

### Model Loading

```python
# BEFORE (YOLO):
model = YOLO('yolov8m.pt')  # Old version

# AFTER (YOLO):
model = YOLO('yolo12n.pt')  # New version, nano size

# BEFORE (SAM):
model = SAM("sam_b.pt")     # Original

# AFTER (SAM):
model = SAM("sam2.1_b.pt")  # Latest generation
```

---

## FOR YOUR PREPARATION: Action Items

### 1. Update Your Materials
- [x] Read old materials? Skim them for context
- [ ] Read new materials thoroughly
- [ ] Note all model version changes
- [ ] Update any demos you planned

### 2. Test the Code
- [ ] Run Activity 1 (verify it works)
- [ ] Run Activity 3 with YOLO12 (new!)
- [ ] Run Activity 4 with SAM2.1 (new!)
- [ ] Note any differences from before

### 3. Prepare Your Explanation
- [ ] Emphasize transfer learning (unchanged concept)
- [ ] Explain why fine-tuning works (unchanged)
- [ ] Show new model versions (YOLO12, SAM2.1)
- [ ] Demo the progress bar (tqdm) - cool visual!

### 4. Anticipate Questions
```
Q: Why YOLO12 instead of YOLO8?
A: Better accuracy, faster, 2025 version

Q: What changed in SAM 2.1?
A: More efficient, better masks, same API

Q: Do I need to re-learn everything?
A: No! Core concepts unchanged, just better code

Q: Will my YOLO8 code still work?
A: Yes, same API, just swap 'yolo12n.pt'
```

---

## FINAL CHECKLIST FOR YOUR SESSION

Before you teach:
- [ ] Read the NEW full guide (Lab_05_Updated_TA_Guide.md)
- [ ] Skim the NEW quick reference
- [ ] Note all inline code comments
- [ ] Test running one Activity
- [ ] Prepare demo images/videos
- [ ] Update any slides you have
- [ ] Know the version changes (YOLO8→12, SAM1→2.1)
- [ ] Understand why each change was made

You're ready! 🚀

---

End of Change Summary
