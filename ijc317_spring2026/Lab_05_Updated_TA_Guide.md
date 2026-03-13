# Lab 05: Deep Learning for Image Segmentation & Object Detection
## Updated TA Session Preparation Guide

**Last Updated:** [Current Date]  
**Status:** Revised with latest worksheet modifications

---

# PART 1: WHAT IS THIS LAB ABOUT?

## High-Level Overview

This lab teaches you about **advanced computer vision models** that go beyond basic CNNs like LeNet from Lab 04. You'll work with three powerful pre-trained architectures:

### What We're Learning:

1. **Vision Transformer (ViT)** - A transformer-based model (similar to ChatGPT!) that uses attention instead of convolutions
   - Two implementations: torchvision and HuggingFace
   
2. **YOLO 12** - Real-time object detection (2025 version!)
   - Detects objects AND their exact locations
   - Much faster than traditional detection methods
   
3. **SAM 2.1** - Segment Anything Model (latest generation)
   - Segments exact shapes of objects
   - Zero-shot: works on any object without training
   - Multiple prompting methods: bounding box, points, text

### Real-World Performance Jump:

```
Lab 04 Results (LeNet on CIFAR-10):
├── Accuracy: 71%
└── Speed: Medium

Lab 05 Results (ViT on CIFAR-10):
├── Accuracy: 95%+ 
└── Speed: Fast
```

---

# PART 2: OUTLINE OF STEPS IN THIS LAB

## Structure: 4 Activities

### ACTIVITY 1: Fine-tune ViT Using torchvision
**Duration:** 40 minutes | **Difficulty:** Medium | **Key Focus:** Transfer Learning

```
Step 1: Setup (imports, hyperparameters, device)
        ↓
Step 2: Load CIFAR-10 dataset
        ↓
Step 3: Create data subsets (20% train, 100% test - for speed)
        ↓
Step 4: Create data loaders with batch size 32
        ↓
Step 5: Load pre-trained ViT_B_16 from ImageNet
        ↓
Step 6: Replace output head (1000 classes → 10 classes)
        ↓
Step 7: Freeze all layers except last encoder block and head
        ↓
Step 8: Define loss function (CrossEntropyLoss) and optimizer (Adam)
        ↓
Step 9: Train-validate loop for 5 epochs with tqdm progress bars
        ↓
Step 10: Save best model based on validation accuracy
```

**Key Concept:** Fine-tuning = Reuse ImageNet knowledge, adapt to CIFAR-10

---

### ACTIVITY 2: Fine-tune ViT Using HuggingFace
**Duration:** 30+ minutes | **Difficulty:** Medium | **Key Focus:** Different API, Same Results

```
Step 1: Install transformers library
        ↓
Step 2: Load CIFAR-10 dataset (same as Activity 1)
        ↓
Step 3: Create subsets (same as Activity 1)
        ↓
Step 4: Load pre-trained ViT from HuggingFace
        ↓
Step 5: Replace classification head (custom layer)
        ↓
Step 6: Freeze first 11 layers, keep last layer + head trainable
        ↓
Step 7: Define loss, optimizer, train-validate loop
        ↓
Step 8: Save best model
```

**Key Difference from Activity 1:**
- HuggingFace API returns an object with `.logits` attribute
- More control over layer freezing
- More flexible for research

---

### ACTIVITY 3: YOLO 12 Object Detection
**Duration:** 30 minutes | **Difficulty:** Easy | **Key Focus:** Pre-trained Models + Zero-shot

```
Step 1: Install ultralytics library
        ↓
Step 2: Load pre-trained YOLO12n model (nano version)
        ↓
Step 3: Download test image from COCO dataset
        ↓
Step 4: Run YOLO inference on single image
        ↓
Step 5: Visualize results (bounding boxes drawn automatically)
        ↓
Step 6: Investigate results object structure
        ↓
Step 7: Extract and print class names of detected objects
        ↓
Step 8: Run YOLO on video file
        ↓
Step 9: Visualize video results with bounding boxes
```

**Key Points:**
- No training needed! Just load and run
- YOLO automatically saves output with drawn boxes
- Works on any image size
- Real-time processing

---

### ACTIVITY 4: SAM 2.1 - Segment Anything
**Duration:** 30 minutes | **Difficulty:** Medium | **Key Focus:** Zero-shot + Prompting

```
Step 1: Load SAM 2.1 model (latest generation)
        ↓
Step 2: Segment everything in image (auto-detect all objects)
        ↓
Step 3: Visualize all segmentation masks
        ↓
Step 4: (Exercise) Try SAM 2.1 model (different version)
        ↓
Step 5: Segment with bounding box prompt (guide the model)
        ↓
Step 6: Segment with point prompts (click-based)
        ↓
Step 7: (Exercise) Try text prompt segmentation
```

**Key Concept:** Flexible prompting for interactive segmentation

---

# PART 3: DETAILED CODE EXPLANATIONS WITH INLINE COMMENTS

## ACTIVITY 1: Vision Transformer (ViT) - torchvision Implementation

### Section 1.1: Imports and Settings

```python
# ===== CORE LIBRARIES =====
import torch                                    # PyTorch: Deep learning framework
import torch.nn as nn                           # Neural network modules (Linear, Conv2d, etc.)
import torch.optim as optim                     # Optimizers (Adam, SGD, etc.)
import torchvision.transforms as transforms     # Image transformations (resize, normalize, etc.)
from torch.utils.data import DataLoader         # Batches images for training
from torchvision import datasets                # Pre-built datasets like CIFAR-10
from tqdm import tqdm                           # Progress bars with time estimates

# ===== HYPERPARAMETERS (Tunable settings) =====
RANDOM_SEED = 42                                # For reproducibility: same random numbers each run
LEARNING_RATE = 1e-3                            # 0.001 - step size for weight updates
TRAIN_BATCH_SIZE = 32                           # Process 32 images at a time during training
TEST_BATCH_SIZE = 32                            # Process 32 images at a time during evaluation
NUM_EPOCHS = 5                                  # Train for 5 complete passes through training data
DROP_RATE = 0.5                                 # Dropout: randomly disable 50% of neurons (prevents overfitting)

# ===== ARCHITECTURE SETTINGS =====
NUM_CLASSES = 10                                # CIFAR-10 has 10 object classes

# ===== DEVICE SELECTION (GPU or CPU) =====
if torch.cuda.is_available():                   # Check if GPU is available
    DEVICE = "cuda:0"                           # Use GPU (100× faster!)
else:
    DEVICE = "cpu"                              # Fallback to CPU
print("DEVICE:", DEVICE)                        # Print which device we're using

GRAYSCALE = True                                # Flag variable (not used in this code)
```

---

### Section 1.2: Load and Prepare CIFAR-10 Dataset

```python
# ===== PREPROCESSING: Transform images for ViT =====
transform_train = transforms.Compose([
    # Step 1: Resize to 224×224 (ViT expects this size, CIFAR-10 is 32×32)
    transforms.Resize(224),                     
    
    # Step 2: Convert PIL image to tensor and scale pixels to [0, 1]
    transforms.ToTensor(),                      
    
    # Step 3: Normalize using ImageNet statistics
    # These mean/std values come from the ImageNet dataset (1.2 million images)
    # Using ImageNet stats helps transfer learning (pre-trained model expects this)
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),          # R, G, B channel means
        std=(0.2023, 0.1994, 0.2010)            # R, G, B channel standard deviations
    ),
])

# ===== SAME FOR TEST DATA =====
transform_test = transforms.Compose([
    transforms.Resize(224),                     # Must match training size!
    transforms.ToTensor(),                      
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),          # MUST USE SAME normalization as training!
        std=(0.2023, 0.1994, 0.2010)
    ),
])

# ===== LOAD CIFAR-10 TRAINING SET =====
full_train_dataset = datasets.CIFAR10(
    root='./datasets',                          # Where to store/load data
    train=True,                                 # Load training set (50,000 images)
    download=True,                              # Auto-download if not already present
    transform=transform_train                   # Apply preprocessing
)

# ===== LOAD CIFAR-10 TEST SET =====
full_test_dataset = datasets.CIFAR10(
    root='./datasets',                          # Same location as training data
    train=False,                                # Load test set (10,000 images)
    download=True,                              # Auto-download if needed
    transform=transform_test                    # Apply preprocessing
)
```

---

### Section 1.3: Create Data Subsets (Because ViT is Slow!)

```python
import numpy as np                              # For random sampling
from torch.utils.data import Subset            # For creating subsets

# ===== HELPER FUNCTION: Sample Random Subset =====
def get_subset(dataset, percentage):
    """
    Purpose: Take a random sample of a dataset
    
    Args:
        dataset: Full dataset (CIFAR-10 with 50,000 or 10,000 images)
        percentage: What fraction to keep (0.2 = 20%, 0.5 = 50%, 1.0 = 100%)
    
    Returns:
        Subset object with randomly selected indices
    
    Why useful:
        ViT is slow! Full training takes hours.
        Using 20% (10,000 images) trains in ~30 minutes.
    """
    num_data = len(dataset)                     # Total images in dataset
    
    indices = np.random.choice(
        num_data,                               # Choose from 0 to num_data-1
        int(num_data * percentage),             # How many to choose (20% of 50000 = 10000)
        replace=False                           # Don't pick same image twice
    )
    
    return Subset(dataset, indices)             # Return subset with selected indices

# ===== CREATE SUBSETS FOR FASTER TRAINING =====
train_percentage = 0.2                          # Use 20% of training data (10,000 images)
train_dataset = get_subset(full_train_dataset, train_percentage)

test_percentage = 1.0                           # Use 100% of test data (10,000 images)
test_dataset = get_subset(full_test_dataset, test_percentage)

print("train_dataset size:", len(train_dataset))  # ~10,000
print("test_dataset size:", len(test_dataset))    # ~10,000
```

---

### Section 1.4: Create Data Loaders

```python
# ===== TRAINING DATA LOADER =====
train_loader = DataLoader(
    train_dataset,                              # Dataset to load from
    batch_size=TRAIN_BATCH_SIZE,                # Load 32 images per batch
    shuffle=True                                # Randomize order each epoch (helps learning)
)

# ===== TEST DATA LOADER =====
test_loader = DataLoader(
    test_dataset,                               # Dataset to load from
    batch_size=TEST_BATCH_SIZE,                 # Load 32 images per batch
    shuffle=False                               # Don't shuffle (order doesn't matter for testing)
)

# ===== CALCULATE LOADER SIZES =====
print("train_loader batches:", len(train_loader))  # 10000 / 32 ≈ 313 batches
print("test_loader batches:", len(test_loader))    # 10000 / 32 ≈ 313 batches
```

**How DataLoader Works:**
```
Full dataset (10,000 images)
    ↓
Shuffle order (if shuffle=True)
    ↓
Split into batches of 32
    ↓
Each batch: [batch_size=32, channels=3, height=224, width=224]
```

---

### Section 1.5: Load Pre-trained ViT Model from torchvision

```python
from torchvision.models import ViT_B_16_Weights, vit_b_16

# ===== LOAD PRE-TRAINED VIT MODEL =====
vit_model = vit_b_16(
    weights=ViT_B_16_Weights.IMAGENET1K_V1      # Load weights pre-trained on ImageNet
)

# What is ViT_B_16?
# - ViT = Vision Transformer (no convolutions, all attention)
# - B = Base model size (not tiny like Mobile, not huge like Large)
# - 16 = Divides image into 16×16 pixel patches
# - IMAGENET1K_V1 = Trained on 1.2 million ImageNet images

# Why pre-trained?
# - Already learned features for recognizing objects
# - Would take weeks to train from scratch
# - Transfer learning: reuse knowledge, train faster!

pretrained_vit_name = vit_b_16.__name__         # Get model name for saving ("vit_b_16")
print("Loaded model:", pretrained_vit_name)
```

---

### Section 1.6: Modify Output Head for CIFAR-10

```python
# ===== THE PROBLEM =====
# ImageNet has 1000 classes (dog breeds, cars, animals, etc.)
# CIFAR-10 has only 10 classes (airplane, car, bird, etc.)
# vit_model.heads outputs 1000 values, but we need 10!

# ===== THE SOLUTION: Replace the head =====
vit_model.heads = nn.Sequential(
    # Remove old layer (1000 outputs)
    # Add new layer that outputs 10 values
    nn.Linear(
        in_features=vit_model.heads.head.in_features,  # Input dimension (768 features)
        out_features=NUM_CLASSES                       # Output dimension (10 classes)
    )
)

# Now vit_model outputs [batch_size, 10] instead of [batch_size, 1000]!
```

**What Happens:**
```
Before: Transformer backbone (768 features) → Old head → 1000 outputs
After:  Transformer backbone (768 features) → New head → 10 outputs
                                              ^^^^^^^^
                                        This layer is NEW!
```

---

### Section 1.7: Freeze Layers (Train Only Last Layer)

```python
# ===== WHY FREEZE LAYERS? =====
# ImageNet pre-trained model already learned good features
# We don't want to destroy that knowledge
# Instead, only fine-tune the last layer for CIFAR-10

# ===== STEP 1: Freeze ALL layers =====
for param in vit_model.parameters():
    param.requires_grad = False                 # "Don't update weights during training"

# ===== STEP 2: UNFREEZE the last encoder layer =====
# Allow the last transformer block to adapt to CIFAR-10
for param in vit_model.encoder.layers[-1].parameters():
    param.requires_grad = True                  # "DO update weights"

# ===== STEP 3: UNFREEZE the output head =====
# The new head needs to be trained (it's initialized randomly)
for param in vit_model.heads.parameters():
    param.requires_grad = True                  # "DO update weights"

# ===== RESULT =====
# Total layers: 12 transformer blocks + 1 head
# Frozen: 11 blocks (don't change)
# Trainable: 1 block + 1 head (adapt to CIFAR-10)
# Speed: ~100× faster than training all layers!
```

**Layer Freezing Strategy:**
```
[Block 1] ❌ FROZEN
[Block 2] ❌ FROZEN
...
[Block 10] ❌ FROZEN
[Block 11] ❌ FROZEN
[Block 12] ✓ TRAINABLE (last chance to adapt)
[Head] ✓ TRAINABLE (new, must learn)
```

---

### Section 1.8: Training Setup

```python
# ===== MOVE MODEL TO DEVICE =====
vit_model = vit_model.to(DEVICE)                # Transfer to GPU (if available)

# ===== CREATE OPTIMIZER =====
# Adam = Adaptive Moment Estimation
# Smart optimizer that adjusts learning rates per weight
optimizer = torch.optim.Adam(
    vit_model.parameters(),                     # Which parameters to optimize
    lr=LEARNING_RATE                            # Learning rate: 0.001
)

# ===== CREATE LOSS FUNCTION =====
# CrossEntropyLoss = for multi-class classification
# Compares predicted class scores vs actual class
criterion = torch.nn.CrossEntropyLoss()
```

---

### Section 1.9: Training and Evaluation Functions

```python
# ===== HELPER FUNCTION 1: Compute Accuracy =====
def compute_accuracy(gt, preds):
    """
    Calculate what percentage of predictions are correct
    
    Args:
        gt: Ground truth labels (actual classes)
        preds: Predicted classes
    
    Returns:
        accuracy: Fraction correct (0.0 to 1.0)
    """
    with torch.no_grad():                       # Don't calculate gradients (we're not training)
        # Create boolean tensor: True where prediction matches ground truth
        correct_pred = (preds == gt).to(torch.int)  # Convert True/False to 1/0
        
        # Sum all correct predictions
        num_correct = correct_pred.sum()
        
        # Calculate accuracy
        accuracy = num_correct.item() / len(gt)  # Divide by total predictions
    
    return accuracy


# ===== HELPER FUNCTION 2: Training Function =====
def train(model, data_loader, criterion, optimizer, device, epoch):
    """
    Train model for one epoch
    
    Args:
        model: ViT model to train
        data_loader: Training data loader with batches
        criterion: Loss function (CrossEntropyLoss)
        optimizer: Optimizer (Adam)
        device: GPU or CPU
        epoch: Current epoch number (for logging)
    
    Returns:
        Dictionary with loss and accuracy for this epoch
    """
    
    # ===== SET TRAINING MODE =====
    model.train()                               # Enable dropout, batch norm updates, etc.
    
    # ===== INITIALIZE TRACKING VARIABLES =====
    loss_history = []                           # Store all loss values
    y_pred = torch.Tensor().to(device)          # Predictions (empty tensor, will grow)
    y_true = torch.Tensor().to(device)          # Ground truth labels (empty tensor)
    
    i = 0                                       # Step counter
    
    # ===== MAIN TRAINING LOOP =====
    for images, labels in tqdm(data_loader, total=len(train_loader)):
        # tqdm shows progress bar with time estimate
        
        i += 1                                  # Increment step counter
        
        # Step 1: Move data to device (GPU/CPU)
        images = images.to(device)              # [batch=32, channels=3, height=224, width=224]
        labels = labels.to(device)              # [batch=32] (one label per image)
        
        # Step 2: Forward pass (predict)
        logits = model(images)                  # ViT processes images → [batch=32, classes=10]
        # logits = raw scores (not probabilities)
        # Example: [0.5, -0.3, 0.8, -0.1, ...]
        
        # Step 3: Calculate loss (how wrong are we?)
        loss = criterion(logits, labels)        # Compare logits vs actual labels
        # Small loss = good prediction
        # Large loss = bad prediction
        
        # Track loss
        loss_history.append(loss.item())        # Save loss as Python float
        
        # Step 4: Backward pass (calculate gradients)
        optimizer.zero_grad()                   # Clear old gradients
        loss.backward()                         # Backpropagation: calculate new gradients
        
        # Step 5: Update weights (gradient descent)
        optimizer.step()                        # Update weights: w_new = w_old - lr * gradient
        
        # Step 6: Get predictions for accuracy calculation
        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=1)    # [batch=32, classes=10] with values 0-1
        # Example: [0.05, 0.1, 0.6, 0.05, ...] (sums to 1)
        
        # Get predicted class (highest probability)
        preds = torch.argmax(probs, dim=1)      # [batch=32] with values 0-9
        # Example: [5, 3, 2, 7, ...] (one class per image)
        
        # Accumulate predictions and labels
        y_pred = torch.cat([y_pred, preds])     # Append to list
        y_true = torch.cat([y_true, labels])
        
        # Step 7: Logging (every 100 steps)
        if i % 100 == 0:
            accuracy = compute_accuracy(y_true, y_pred)
            avg_loss = sum(loss_history) / len(y_pred)
            # Optional: Print progress (commented out)
            # print(f"Epoch {epoch}, Step {i}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")
    
    # ===== END OF EPOCH =====
    avg_loss = sum(loss_history) / len(y_pred)
    accuracy = compute_accuracy(y_true, y_pred)
    print(f"Epoch {epoch}: Training loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")
    
    return {"loss": avg_loss, "accuracy": accuracy}


# ===== HELPER FUNCTION 3: Evaluation Function =====
def evaluate(model, data_loader, criterion, device):
    """
    Evaluate model on test set (same as train but no weight updates)
    
    Args:
        model: ViT model to evaluate
        data_loader: Test data loader
        criterion: Loss function
        device: GPU or CPU
    
    Returns:
        Dictionary with loss and accuracy on test set
    """
    
    # ===== SET EVALUATION MODE =====
    model.eval()                                # Disable dropout, freeze batch norm
    
    # ===== INITIALIZE TRACKING VARIABLES =====
    y_pred = torch.Tensor().to(device)
    y_true = torch.Tensor().to(device)
    loss_history = []
    
    # ===== EVALUATION LOOP =====
    with torch.no_grad():                       # No gradients needed (not training)
        for images, labels in tqdm(data_loader, total=len(test_loader)):
            # Move data to device
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass only (no backward)
            logits = model(images)
            
            # Calculate loss
            loss = criterion(logits, labels)
            loss_history.append(loss.item())
            
            # Get predictions
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            # Accumulate
            y_pred = torch.cat([y_pred, preds])
            y_true = torch.cat([y_true, labels])
    
    # ===== CALCULATE FINAL METRICS =====
    avg_loss = sum(loss_history) / len(y_pred)
    accuracy = compute_accuracy(y_true, y_pred)
    
    return {"loss": avg_loss, "accuracy": accuracy}
```

---

### Section 1.10: Main Training Loop with Model Selection

```python
# ===== CREATE LOSS FUNCTION AND OPTIMIZER =====
criterion = nn.CrossEntropyLoss()              # Multi-class classification loss
optimizer = optim.Adam(vit_model.parameters(), lr=LEARNING_RATE)  # Optimizer with lr=0.001

# ===== INITIALIZE TRACKING VARIABLES =====
best_model = None                              # Will store best model weights
best_acc = -1                                  # Track best validation accuracy (starts low)
best_epoch = -1                                # Track which epoch had best accuracy
train_loss = []                                # Store loss for each epoch
val_loss = []                                  # Store validation loss for each epoch

# ===== IMPORTANT: Move model to device =====
# This resolves a common CUDA error where model and data are on different devices
if "cuda" in DEVICE:
    vit_model.cuda()                           # Explicitly move to GPU

# ===== MAIN EPOCH LOOP =====
for epoch in range(NUM_EPOCHS):                # Train for 5 epochs
    print(f"\n===== EPOCH {epoch+1}/{NUM_EPOCHS} =====")
    
    # Step 1: Train for one epoch
    train_results = train(vit_model, train_loader, criterion, optimizer, DEVICE, epoch)
    
    # Step 2: Evaluate on test set
    val_results = evaluate(vit_model, test_loader, criterion, DEVICE)
    
    # Step 3: Print results
    print(f"Epoch {epoch}, Val loss={val_results['loss']:.4f}, Val accuracy={val_results['accuracy']:.4f}")
    
    # Step 4: Model selection (save best model)
    if val_results['accuracy'] > best_acc:
        best_acc = val_results['accuracy']
        best_model = vit_model                 # Save reference to model
        best_epoch = epoch
        print(f"✓ New best accuracy: {best_acc:.4f}")
    
    # Step 5: Tracking for visualization
    train_loss.append(train_results["loss"])
    val_loss.append(val_results["loss"])

# ===== SAVE THE BEST MODEL =====
import os

# Create directory if it doesn't exist
model_dir = "/content/models/CIFAR10/"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Create full file path
save_path = model_dir + pretrained_vit_name + "_torchvision_cifar10.pkl"

# Save model checkpoint with multiple pieces of information
torch.save({
    'num_epoch': epoch + 1,                    # Total epochs trained
    'best_epoch': best_epoch + 1,              # Which epoch was best
    'model_state_dict': vit_model.state_dict(),  # Model weights (the important part)
    'optimizer_state_dict': optimizer.state_dict(),  # Optimizer state (for resuming training)
}, save_path)

print(f"\n✓ Saved best model to {save_path}")
print(f"Best accuracy: {best_acc:.4f} (epoch {best_epoch+1})")
```

---

## ACTIVITY 2: Vision Transformer - HuggingFace Implementation

### Key Differences from Activity 1:

```python
# ===== INSTALLATION =====
!pip install -q transformers datasets

# ===== IMPORTS =====
from transformers import ViTModel, ViTForImageClassification

# ===== LOAD MODEL (DIFFERENT API) =====
pretrained_vit_name = "vit-base-patch16-224"   # HuggingFace model identifier
vit_model = ViTForImageClassification.from_pretrained(
    "google/" + pretrained_vit_name              # Google's official ViT model
).to(DEVICE)

# ===== REPLACE HEAD =====
vit_model.config.classifier = 'mlp'
vit_model.config.num_labels = NUM_CLASSES
vit_model.classifier = nn.Linear(vit_model.config.hidden_size, NUM_CLASSES).to(DEVICE)

# ===== FREEZE LAYERS (DIFFERENT STRUCTURE) =====
def free_layers(vit_model, first_n_layers=6):
    """
    Purpose: Freeze first N transformer blocks, unfreeze rest
    
    Args:
        vit_model: HuggingFace ViT model
        first_n_layers: How many blocks to freeze (out of 12)
    """
    
    # Freeze embeddings (initial layer)
    for param in vit_model.vit.embeddings.parameters():
        param.requires_grad = False
    
    # Freeze first N blocks
    for i in range(first_n_layers):
        for param in vit_model.vit.encoder.layer[i].parameters():
            param.requires_grad = False
    
    print(f"First {first_n_layers} blocks frozen.")
    
    # Ensure classifier is trainable
    for param in vit_model.classifier.parameters():
        param.requires_grad = True

# Call the function to freeze first 11 blocks (keep last block + head trainable)
free_layers(vit_model, first_n_layers=11)
```

### Key Difference in Forward Pass:

```python
# ===== HUGGINGFACE OUTPUT IS DIFFERENT =====
# torchvision: outputs logits directly
# logits = model(images)

# HuggingFace: outputs an object with multiple attributes
# outputs = model.forward(images)  # Returns a ModelOutput object
# outputs.logits  # Access logits from the object

# In training function:
outputs = model.forward(images)     # Get ModelOutput object
loss = criterion(outputs.logits, labels)  # Use .logits attribute
probs = torch.softmax(outputs.logits, dim=1)  # Get probabilities from logits
```

---

## ACTIVITY 3: YOLO 12 Object Detection

### Section 3.1: Installation and Setup

```python
# ===== INSTALL ULTRALYTICS =====
!pip install ultralytics

# ===== IMPORTS =====
import numpy as np
import cv2                                      # OpenCV for image processing
import random
from google.colab.patches import cv2_imshow    # Display images in Colab

from ultralytics import YOLO                   # YOLO model from Ultralytics
```

---

### Section 3.2: Load YOLO Model

```python
# ===== LOAD PRETRAINED YOLO12 MODEL =====
model = YOLO('yolo12n.pt')                     # Load YOLO12 nano (smallest, fastest)

# What is YOLO12n?
# - YOLO = You Only Look Once (real-time object detection)
# - 12 = Version 12 (2025 release!)
# - n = nano (smallest size, ~6.3M parameters, fastest)
# Other sizes: s (small), m (medium), l (large), x (xlarge)
# 
# Pre-trained on COCO dataset (80 object classes)
# Classes: person, car, dog, cat, bicycle, etc.

print("Model type:", type(model))              # Should be: <class 'ultralytics.models.yolo.detect.DetectionModel'>
```

---

### Section 3.3: Run YOLO on Single Image

```python
# ===== DOWNLOAD TEST IMAGE FROM COCO DATASET =====
!wget http://images.cocodataset.org/val2017/000000439715.jpg -O input.jpg
# Downloads a real image from COCO dataset

# ===== LOAD AND DISPLAY IMAGE =====
import cv2
from google.colab.patches import cv2_imshow

im = cv2.imread("./input.jpg")                 # Read image using OpenCV
cv2_imshow(im)                                 # Display in Colab notebook

# ===== RUN YOLO INFERENCE =====
results = model('input.jpg', save=True)        # Run detection on image
# save=True: saves output image with bounding boxes to /content/runs/detect/predict/

# ===== DISPLAY RESULT =====
im = cv2.imread("/content/runs/detect/predict/input.jpg")  # Load output with boxes
cv2_imshow(im)                                 # Display result

# ===== INVESTIGATE RESULTS OBJECT =====
print("Type of results:", type(results))       # Should be: <class 'list'>
# results is a list because we can run inference on multiple images

print("Type of results[0]:", type(results[0])) # Each element is a Result object

# ===== ACCESS BOUNDING BOXES =====
print("Boxes type:", type(results[0].boxes))   # ultralytics.engine.results.Boxes

# Print all detected boxes
print(results[0].boxes)                         # Shows all bounding boxes with confidence

# ===== EXTRACT CLASS NAMES =====
for idx, c in enumerate(results[0].boxes.cls):
    # c is a tensor with class ID (0-79 for COCO)
    class_id = int(c.item())                   # Convert tensor to Python int
    class_name = results[0].names[class_id]    # Get class name
    print(f"Detection {idx}: {class_name}")
```

**Result Example:**
```
Detection 0: person
Detection 1: car
Detection 2: dog
Detection 3: bicycle
```

---

### Section 3.4: Run YOLO on Video

```python
# ===== YOLO ON VIDEO =====
# Note: Video file should be uploaded to Colab or available in /content/

video_name = "Moving Vehicles in Highway 1min"  # Name without extension
video_ext = ".mp4"

# ===== RUN DETECTION ON VIDEO =====
result = model('/content/' + video_name + video_ext, save=True)
# YOLO automatically:
# 1. Reads video frame-by-frame
# 2. Detects objects in each frame
# 3. Draws bounding boxes
# 4. Saves output video with detections

# Output saved to: /content/runs/detect/predict{N}/ (where N is a number)

# ===== DISPLAY OUTPUT VIDEO =====
from moviepy.editor import VideoFileClip

result_dir = "/content/runs/detect/predict6/"   # Adjust number as needed
result_ext = ".avi"

# Load video clip
clip = VideoFileClip(result_dir + video_name + result_ext)

# Display with max 200 seconds of video
clip.ipython_display(width=800, maxduration=200)
```

---

## ACTIVITY 4: SAM 2.1 - Segment Anything Model

### Section 4.1: Setup

```python
# ===== INSTALL ULTRALYTICS (if not done) =====
!pip install ultralytics

# ===== CHECK SYSTEM =====
import ultralytics
ultralytics.checks()                           # Verify installation, check GPU/CPU, display versions
```

---

### Section 4.2: Segment Everything (Auto-Detect All Objects)

```python
from ultralytics import SAM

# ===== LOAD SAM MODEL =====
# Available models:
# - "sam_b.pt" = base (fastest)
# - "sam_l.pt" = large (more accurate)
# - "sam2.1_b.pt" = SAM 2.1 base (latest generation!)
# - "sam2.1_l.pt" = SAM 2.1 large

model = SAM("sam_b.pt")                        # Load SAM base model

# Display model info
model.info()                                   # Print model statistics (parameters, speed, etc.)

# ===== SEGMENT EVERYTHING =====
# No prompt needed - SAM automatically finds all objects!
results = model("https://ultralytics.com/images/bus.jpg")

# Returns a list of Result objects (one per input image)
print("Result type:", type(results[0]))        # ultralytics.engine.results.Results

# ===== VISUALIZE RESULTS =====
results[0].show()                              # Display segmentation masks (all objects highlighted)
```

---

### Section 4.3: Segment Everything with Latest SAM 2.1

```python
from ultralytics import SAM

# ===== LOAD SAM 2.1 MODEL =====
# SAM 2.1 is the latest generation (more efficient than SAM)
model = SAM("sam2.1_b.pt")

model.info()                                   # Display model info

# ===== SEGMENT ALL OBJECTS =====
results = model("https://ultralytics.com/images/bus.jpg")

# ===== DISPLAY RESULTS =====
results[0].show()                              # Show all segmentation masks
```

---

### Section 4.4: Segment with Bounding Box Prompt

```python
from ultralytics import SAM

# ===== LOAD MODEL =====
model = SAM("sam2.1_b.pt")

# ===== DEFINE BOUNDING BOX PROMPT =====
# Format: [x1, y1, x2, y2]
# Where (x1, y1) = top-left corner
#       (x2, y2) = bottom-right corner

bbox_prompt = [
    3.8328723907470703,      # x1 (left edge)
    229.35601806640625,      # y1 (top edge)
    796.2098999023438,       # x2 (right edge)
    728.4313354492188        # y2 (bottom edge)
]

# This bounding box focuses on the bus in the image

# ===== SEGMENT WITH BBOX =====
results = model(
    "https://ultralytics.com/images/bus.jpg",
    bboxes=bbox_prompt                         # Guide SAM to focus on this region
)

# ===== VISUALIZE =====
results[0].show()                              # Shows only segmentation inside bbox
```

**Why Bounding Box Prompts?**
```
Without prompt: Segments EVERYTHING (slow, may miss details)
With bbox: Focus on specific region (faster, more accurate)
Example: Instead of "segment all objects", use "segment the bus"
```

---

### Section 4.5: Segment with Point Prompts

```python
from ultralytics import SAM

# ===== LOAD MODEL =====
model = SAM("sam2.1_b.pt")

# ===== DEFINE POINT PROMPTS =====
# Format: List of [x, y] coordinates
# x, y = pixel coordinates where user clicks

point_prompt = [
    [34, 714],              # Point 1
    [283, 634],             # Point 2
    [150, 150]              # Point 3 (on different object)
]

# ===== POINT LABELS =====
# 1 = foreground point (part of object to segment)
# 0 = background point (not part of object)

point_labels = [1, 1, 1]                       # All points are foreground

# ===== SEGMENT WITH POINTS =====
results = model(
    "https://ultralytics.com/images/bus.jpg",
    points=point_prompt,                       # User clicks
    point_labels=point_labels                  # Labels for each point
)

# ===== VISUALIZE =====
results[0].show()                              # Shows segmentation around clicked points
```

**Interactive Segmentation Flow:**
```
User clicks on object
    ↓
Points sent to SAM
    ↓
SAM identifies object around points
    ↓
Returns segmentation mask
    ↓
User can refine with more points if needed
```

---

# QUICK REFERENCE TABLE

## Models at a Glance

| Model | Task | Input | Output | Speed | Accuracy | Training |
|-------|------|-------|--------|-------|----------|----------|
| **ViT (torchvision)** | Classification | Image | Class label | Fast | 95%+ | Fine-tune |
| **ViT (HuggingFace)** | Classification | Image | Class label | Fast | 95%+ | Fine-tune |
| **YOLO12n** | Detection | Image/Video | Bboxes + classes | Real-time | High | Pre-trained |
| **SAM 2.1** | Segmentation | Image + prompt | Masks | Medium | Very high | Pre-trained |

---

## Key Differences Between the Two ViT Implementations

```
TORCHVISION:
├─ Simpler API
├─ Straightforward layer structure
├─ Good for beginners
└─ Slightly faster

HUGGINGFACE:
├─ More flexible
├─ Better for research
├─ Output is an object with multiple attributes
└─ More customization options
```

---

# COMMON ISSUES & SOLUTIONS

### Issue 1: CUDA Out of Memory

```python
# Solution: Reduce batch size
TRAIN_BATCH_SIZE = 16  # Instead of 32

# Or use smaller model
model = YOLO('yolo12n.pt')  # nano instead of medium
```

### Issue 2: Training Too Slow

```python
# Solution 1: Use GPU
DEVICE = "cuda:0"  # Not "cpu"

# Solution 2: Reduce dataset
train_percentage = 0.1  # 10% instead of 20%

# Solution 3: Reduce epochs
NUM_EPOCHS = 3  # Instead of 5
```

### Issue 3: YOLO Not Detecting Objects

```python
# Possible reasons:
# 1. Object not in COCO dataset (80 classes only)
# 2. Object too small or too far away
# 3. Confidence threshold too high

# Solution:
results = model('image.jpg', conf=0.3)  # Lower confidence threshold
# Default is 0.25
```

### Issue 4: SAM Segmentation Looks Wrong

```python
# Possible reasons:
# 1. Bounding box too large/small
# 2. Points on wrong part of object
# 3. Image too blurry

# Solution:
# Adjust bounding box coordinates
# Move points to clearer part of object
# Try different image
```

---

# TEACHING TIPS FOR YOUR TA SESSION

## What Students Often Miss:

1. **Why freeze layers?** → "Don't destroy ImageNet knowledge, only fine-tune"
2. **Why resize to 224×224?** → "ViT was pre-trained on 224×224, expects same size"
3. **Why normalize with ImageNet stats?** → "Model learned from normalized ImageNet images"
4. **YOLO is zero-shot?** → "Trained on COCO once, works on any image after"
5. **SAM is foundation model?** → "Trained on 11M images, generalizes to any object"

## Demo Ideas:

1. Show train/val loss decreasing
2. Run YOLO on your own image
3. Try SAM with different prompts
4. Compare ViT vs LeNet accuracy (95% vs 71%)

## Key Takeaways:

```
Transfer Learning = Game changer
  Pre-trained model + fine-tune >> Train from scratch

Layer Freezing = Preserve knowledge
  Freeze most layers, fine-tune last layer
  Fast training, better results

Zero-shot Models = No training needed
  YOLO, SAM work directly
  Just load and run!
```

---

End of Updated TA Session Guide
