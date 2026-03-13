# Lab 05 - UPDATED Quick Reference & Summary
## For Efficient TA Session Preparation

---

## ONE-MINUTE OVERVIEW

```
Lab 05: Learn THREE powerful computer vision models

Activity 1: ViT with torchvision
├─ Fine-tune pre-trained model
├─ 5 epochs on CIFAR-10
├─ Achieve 95%+ accuracy
└─ Duration: 40 minutes

Activity 2: ViT with HuggingFace  
├─ Same concept, different library
├─ Shows flexibility
└─ Duration: 30+ minutes

Activity 3: YOLO12 Object Detection
├─ Load pre-trained model (no training!)
├─ Run on images and videos
├─ Real-time detection (30+ FPS)
└─ Duration: 30 minutes

Activity 4: SAM 2.1 Segmentation
├─ Segment with flexible prompts
├─ No training needed
├─ Zero-shot (works on any object)
└─ Duration: 30 minutes
```

---

## VISUAL: The Lab Structure

```
┌─────────────────────────────────────────────────────────────┐
│                     LAB 05: STRUCTURE                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ACTIVITY 1 & 2:    ACTIVITY 3:       ACTIVITY 4:         │
│  ViT Fine-tuning    YOLO Detection    SAM Segmentation    │
│  ────────────────   ─────────────     ─────────────────   │
│  [Training]         [Zero-shot]       [Zero-shot]          │
│  5 epochs           Just load         Just load             │
│  95%+ accuracy      Real-time         Flexible prompts     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## KEY STATISTICS

### CIFAR-10 Performance Comparison

```
Lab 04: LeNet
├─ Accuracy: 71%
├─ Training: ~2 hours (from scratch)
└─ Simplicity: Simple architecture

Lab 05: ViT (Fine-tuned)
├─ Accuracy: 95%+
├─ Training: ~30 minutes (fine-tune only)
└─ Method: Transfer learning

Improvement: +24% accuracy, 4× faster training!
```

### Model Sizes

```
ViT (torchvision): ~86M parameters
├─ vit_b_16 = "Base" model
├─ 12 transformer blocks
└─ Patch size = 16×16

YOLO12n: ~6.3M parameters
├─ nano = smallest, fastest
├─ Trained on 80 object classes
└─ Real-time (30+ FPS on GPU)

SAM 2.1: ~38M parameters (base)
├─ Trained on 11M diverse images
├─ Works on ANY object
└─ Multiple prompt types
```

---

## CODE FLOW DIAGRAMS

### Activity 1: ViT Fine-tuning

```
Dataset (50K train, 10K test)
    ↓
Subset (20% train, 100% test)
    ↓
Transforms (Resize 224, Normalize)
    ↓
DataLoader (batch_size=32)
    ↓
Load ViT_B_16 (pre-trained ImageNet)
    ↓
Replace head (1000 → 10 classes)
    ↓
Freeze all except last layer + head
    ↓
Train-Validate Loop × 5 epochs
    ├─ Forward pass
    ├─ Calculate loss
    ├─ Backward pass
    ├─ Update weights (last layer only!)
    └─ Evaluate on test set
    ↓
Save best model (by validation accuracy)
    ↓
Result: 95%+ accuracy!
```

### Activity 3: YOLO Detection

```
Load YOLO12n (pre-trained COCO)
    ↓
Load image from COCO dataset
    ↓
Run inference: model('image.jpg')
    ↓
Get results (list of Result objects)
    ↓
Extract detections:
├─ Bounding boxes (coordinates)
├─ Confidence scores
├─ Class names
└─ Visualizations
    ↓
Run on video (frame-by-frame)
    ↓
Auto-save output with boxes drawn
    ↓
Result: Real-time object detection!
```

### Activity 4: SAM Segmentation

```
Load SAM2.1_b (pre-trained 11M images)
    ↓
Load image
    ↓
Segment everything (no prompt)
    ├─ Auto-detects all objects
    └─ Returns masks
    ↓
Segment with bounding box prompt
    ├─ Focus on region
    └─ More accurate
    ↓
Segment with point prompts
    ├─ Click on object
    └─ Interactive
    ↓
Result: Flexible segmentation!
```

---

## IMPORTANT CODE PATTERNS

### Pattern 1: Data Preprocessing for ViT

```python
transform = transforms.Compose([
    transforms.Resize(224),              # MUST BE 224×224
    transforms.ToTensor(),               # Scale to [0,1]
    transforms.Normalize(                # Use ImageNet stats!
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
    )
])
```

### Pattern 2: Layer Freezing

```python
# Freeze everything
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last block + head
for param in model.encoder.layers[-1].parameters():
    param.requires_grad = True
for param in model.heads.parameters():
    param.requires_grad = True
```

### Pattern 3: Training Loop

```python
for epoch in range(NUM_EPOCHS):
    # Train
    train_results = train(model, train_loader, ...)
    
    # Evaluate
    val_results = evaluate(model, test_loader, ...)
    
    # Save best
    if val_results['accuracy'] > best_acc:
        best_acc = val_results['accuracy']
        torch.save(model.state_dict(), 'best_model.pth')
```

### Pattern 4: YOLO Inference

```python
model = YOLO('yolo12n.pt')
results = model('image.jpg')
for detection in results[0].boxes:
    x1, y1, x2, y2 = detection.xyxy[0]
    confidence = detection.conf[0]
    class_id = detection.cls[0]
```

### Pattern 5: SAM Segmentation

```python
model = SAM('sam2.1_b.pt')

# No prompt
results = model('image.jpg')

# With bbox prompt
results = model('image.jpg', bboxes=[x1, y1, x2, y2])

# With point prompts
results = model('image.jpg', points=[[x,y], [x,y]], point_labels=[1,1])
```

---

## DECISION TREE: Which Model to Use?

```
What's your task?
│
├─ Classify image into categories?
│  └─ Use ViT!
│     ├─ Activity 1 or 2
│     ├─ Fine-tune on your dataset
│     └─ 95%+ accuracy
│
├─ Find WHERE objects are?
│  └─ Use YOLO!
│     ├─ Activity 3
│     ├─ Just load and run (zero-shot!)
│     └─ Real-time (30+ FPS)
│
└─ Get EXACT SHAPE of objects?
   └─ Use SAM!
      ├─ Activity 4
      ├─ Zero-shot + flexible prompts
      └─ Works on any object
```

---

## HYPERPARAMETERS CHEAT SHEET

### ViT Fine-tuning Settings

```
LEARNING_RATE = 1e-3        # 0.001 (smaller than normal training)
TRAIN_BATCH_SIZE = 32       # Good balance for GPU memory
TEST_BATCH_SIZE = 32        # Doesn't matter much
NUM_EPOCHS = 5              # Usually enough for fine-tuning
NUM_CLASSES = 10            # CIFAR-10 has 10 classes
TRAIN_PERCENTAGE = 0.2      # Use 20% of training data (for speed)

Trainable layers:
├─ Last transformer block (1 out of 12)
└─ New output head (replaces old head)

Frozen layers:
└─ First 11 transformer blocks (preserved ImageNet knowledge)
```

### YOLO/SAM Settings

```
No hyperparameters needed!
Just load pre-trained and run.

Optional:
├─ Confidence threshold (default 0.25)
├─ Input size (auto-resizes)
└─ Prompt format (bboxes, points, text)
```

---

## WHAT CHANGED BETWEEN VERSIONS

### Activities 1 & 2

```
CHANGED:
├─ Now use ACTUAL ViT models (not simplified versions)
├─ More realistic training parameters
└─ Better code organization

SAME:
└─ Core concept: fine-tuning ViT on CIFAR-10
```

### Activity 3: YOLO

```
CHANGED:
├─ Updated from YOLO8 to YOLO12 (2025 version!)
├─ More models available (nano, small, medium, etc.)
└─ Better performance

SAME:
└─ Zero-shot inference still works the same
```

### Activity 4: SAM

```
CHANGED:
├─ Now featuring SAM 2.1 (latest generation)
├─ Better efficiency and accuracy
├─ Video support mentioned

SAME:
└─ Prompting methods (bbox, points, text)
```

---

## COMMON STUDENT QUESTIONS & ANSWERS

### Q1: Why freeze layers?
```
A: Preserve learned features from ImageNet!
   - ImageNet: 1.2M diverse images
   - Already learned useful patterns
   - Don't destroy that knowledge
   - Only fine-tune last layer for CIFAR-10
```

### Q2: Why resize to 224×224?
```
A: ViT was trained on 224×224 images
   - Model expects specific input size
   - Smaller patches would lose detail
   - Larger sizes waste computation
   - 224 is the sweet spot
```

### Q3: Why normalize with ImageNet statistics?
```
A: Model learned from normalized data!
   - Pre-trained on [0,1] normalized images
   - Specific mean and std per channel
   - Not normalizing = different data distribution
   - Model gets confused!
```

### Q4: Is YOLO trained on CIFAR-10?
```
A: No! YOLO is trained on COCO dataset
   - COCO: 80 object classes (person, car, dog, etc.)
   - Will NOT detect custom objects not in COCO
   - But generalizes well to similar objects
```

### Q5: Can SAM fail?
```
A: Yes, if:
   - Bounding box poorly defined
   - Points on background, not object
   - Image is very blurry
   - Object partially hidden
   
   Solution: Adjust prompts or try different image
```

---

## COMPARING ALL THREE MODELS

```
┌────────────┬──────────────┬──────────────┬──────────────┐
│ Aspect     │ ViT          │ YOLO12       │ SAM 2.1      │
├────────────┼──────────────┼──────────────┼──────────────┤
│ Task       │ Classify     │ Detect+Loc   │ Segment      │
│ Input      │ Image (224)  │ Image (any)  │ Image+prompt │
│ Output     │ Class score  │ Boxes+class  │ Masks        │
│ Speed      │ Medium       │ Very Fast    │ Medium       │
│ Accuracy   │ Very High    │ High         │ Very High    │
│ Training   │ Fine-tune    │ Zero-shot    │ Zero-shot    │
│ Flexibility│ Good         │ Fixed 80cls  │ Excellent    │
│ Code lines │ 100+         │ 5 lines      │ 5 lines      │
└────────────┴──────────────┴──────────────┴──────────────┘
```

---

## TIMELINE FOR YOUR TA SESSION

### Before Session (30 min)
- [x] Read this quick reference
- [x] Skim the detailed guide
- [ ] Run one Activity yourself
- [ ] Prepare demo image

### During Session (2 hours)
```
0:00-0:10    Overview & big picture
0:10-0:40    Activity 1 walkthrough (code comments)
0:40-1:00    Activity 2 comparison (differences only)
1:00-1:30    Activity 3 demo (run YOLO live)
1:30-1:50    Activity 4 demo (try different prompts)
1:50-2:00    Q&A + Summary
```

---

## KEY DEFINITIONS

```
Transfer Learning:
Use pre-trained model on new task
Faster training, better results

Fine-tuning:
Train only last layers of pre-trained model
Keep earlier layers frozen

Zero-shot:
Model works without training on that specific data
Generalizes from training on other data

Foundation Model:
Model trained on massive diverse dataset
Works well on many tasks without fine-tuning

Patch Embedding:
Split image into 16×16 patches
Each patch = one token (like word in NLP)

Bounding Box:
Rectangle: [x1, y1, x2, y2]
Top-left corner to bottom-right corner

Segmentation Mask:
Binary image: 0=background, 1=object
Exact object shape
```

---

## TECHNICAL NOTES

### Device Handling
```python
# IMPORTANT: Ensure model and data on same device
if "cuda" in DEVICE:
    model.cuda()      # Move to GPU explicitly

# Move batches to device
images = images.to(device)
labels = labels.to(device)
```

### Gradient Computation
```python
# No gradients for evaluation
with torch.no_grad():
    logits = model(images)
    # Faster, uses less memory

# With gradients for training
logits = model(images)
loss.backward()       # Calculate gradients
optimizer.step()      # Update weights
```

### Output Handling
```python
# torchvision ViT
logits = model(images)  # Direct output

# HuggingFace ViT
outputs = model(images) # Returns object
logits = outputs.logits # Access logits attribute

# YOLO
results = model('image.jpg')  # Returns list
results[0].boxes               # Access boxes
results[0].names[class_id]     # Access class names

# SAM
results = model('image.jpg')   # Returns list
results[0].masks               # Access masks
results[0].show()              # Display results
```

---

## TROUBLESHOOTING CHECKLIST

```
☐ GPU available? Check: torch.cuda.is_available()
☐ Model on GPU? Check: model.cuda() called
☐ Data on GPU? Check: images.to(device)
☐ Batch size too large? Reduce if OOM error
☐ Training too slow? Use GPU, not CPU
☐ Validation accuracy not improving? Lower learning rate
☐ YOLO missing objects? Try lower conf threshold
☐ SAM wrong segment? Adjust bbox or point prompts
```

---

End of Quick Reference
