# 👁 Diabetic Retinopathy Detection and Severity Classification

## 📖 Overview
This project implements an automated system for *detecting and classifying Diabetic Retinopathy (DR) severity* from retinal fundus images using deep learning.  
We explored and compared several *hybrid CNN–Transformer architectures, advanced **data augmentation, and **class balancing* techniques to achieve robust and interpretable results on the *APTOS 2019 Blindness Detection dataset*.

The primary objective was to identify the most effective training pipeline and model architecture for achieving high validation accuracy and stable performance across all DR severity levels.

---

## 🚀 Key Highlights

### 🧠 Hybrid Model Design
- Combined *EfficientNet-B0* (for fine-grained convolutional features) with *ResNet18* (for deep residual context).
- Feature fusion through concatenation of both networks’ outputs (1792-dim vector) followed by fully connected layers for classification.
- Achieved *up to 86% validation accuracy*, outperforming single-architecture CNNs.

### ⚖ Class Imbalance Handling
- Applied multiple strategies to handle the skewed dataset distribution:
  - *WeightedRandomSampler* for balanced mini-batches.
  - *Weighted Cross-Entropy* loss to emphasize minority classes.
  - *SMOTE* (Synthetic Minority Oversampling Technique) for feature-space oversampling.
  - *Mixup Augmentation* for soft-label regularization and smoother decision boundaries.

### 🧩 Attention-Based Lesion Fusion
- Implemented a *Lesion-Aware Fusion model* combining global and local lesion patches using *Multiple Instance Learning (MIL)*.
- Integrated *CORN loss* for ordinal regression to preserve severity order between DR grades.
- Resulted in the *lowest validation loss of 0.2293*, demonstrating strong ordinal consistency.

### 🔍 Transformer Integration
- Developed an *EfficientNet-B4 + Swin Transformer Hybrid*.
- The CNN branch captures local texture; the Swin Transformer extracts long-range dependencies.
- Achieved *82% validation accuracy* with enhanced contextual understanding.

### ⚙ Automated Hyperparameter Tuning
- Implemented *Optuna Bayesian optimization* to fine-tune learning rate, optimizer, dropout, and Mixup parameters.
- Best Optuna trial achieved *84.4%* validation accuracy (learning rate = 8.55e-5, RMSProp, dropout = 0.33, α = 0.95).

---

## 🧬 Dataset

*Source:* [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection)  
*Total Images:* 3,296 (2,930 training + 366 validation)

| Label | Severity Level | Description |
|:------|:----------------|:-------------|
| 0 | No DR | No visible signs of diabetic retinopathy |
| 1 | Mild | Presence of microaneurysms |
| 2 | Moderate | Hemorrhages and microaneurysms |
| 3 | Severe | Extensive hemorrhages and vascular abnormalities |
| 4 | Proliferative DR | New vessel formation; high risk of blindness |

---

## 🧰 Preprocessing & Augmentation

| Step | Description |
|:-----|:-------------|
| *Resize* | 224×224 pixels (global input); 384×384 for lesion-aware models |
| *Normalization* | Applied ImageNet mean and standard deviation |
| *Augmentations* | Random flip, rotation (±15°), brightness/contrast jitter, random resized crop |
| *CLAHE* | Applied for improved local contrast and vessel visibility |
| *Libraries Used* | torchvision.transforms, Albumentations |

---

## 🧠 Model Architectures and Experiments

| Exp. | Model | Description | Validation Result |
|:----:|:------|:-------------|:----------------:|
| *1* | Hybrid EfficientNet-B0 + ResNet18 | Baseline hybrid model (CrossEntropyLoss, no balancing) | *81.0%* |
| *2* | + WeightedRandomSampler | Balanced training batches | *82.24%* |
| *3* | + Weighted CE + Label Smoothing | Regularized loss, smoothed predictions | *83.06%* |
| *4* | + SMOTE + Mixup | Synthetic feature oversampling + mixed-label augmentation | *84.0%* |
| *5* | Hybrid + Mixup (α=1.0) | Best-performing configuration | *86.0%* |
| *6* | EfficientNet-B4 + Swin Transformer | Hybrid CNN + Transformer fusion | *82.0%* |
| *7* | Hyperparameter Optimization (Optuna) | Bayesian search (lr=8.55e-5, RMSProp) | *84.4%* |
| *8* | Lesion-Aware Fusion + Attention MIL + CORN Loss | Ordinal regression with lesion patches | *Val Loss = 0.2293* |

---

## 🧪 Training Setup

| Parameter | Value |
|:-----------|:------|
| *Framework* | PyTorch |
| *Optimizers* | Adam / RMSProp / AdamW |
| *Loss Functions* | CrossEntropy, Weighted CE, CORN Loss |
| *Batch Size* | 32 (8 for MIL) |
| *Epochs* | 20–30 (4 for quick tests) |
| *Learning Rate* | 1e-4 (tuned via Optuna) |
| *Device* | CUDA GPU (Colab / Kaggle environment) |

---

## 🧩 Techniques Implemented

| Technique | Purpose |
|:-----------|:---------|
| *Weighted Sampling* | Balances class representation during training |
| *Weighted CE & Label Smoothing* | Stabilizes loss and avoids overconfidence |
| *SMOTE Oversampling* | Creates synthetic minority examples |
| *Mixup Augmentation* | Combines samples and labels for smooth decision boundaries |
| *Optuna Optimization* | Automates parameter tuning via Bayesian search |
| *Attention-based MIL Fusion* | Aggregates lesion patch features with global context |
| *Evaluation Metrics* | Accuracy, Precision, Recall, F1-Score, Confusion Matrix |

---

## 📊 Results Summary

| Technique | Accuracy / Loss | Key Observation |
|:-----------|:----------------|:----------------|
| Baseline Hybrid | 81% | Initial reference model |
| Weighted Sampling | 82.24% | Improved minority recall |
| Weighted CE + Label Smoothing | 83.06% | Better stability |
| SMOTE + Mixup | 84% | Balanced training + regularization |
| Mixup Only (α=1.0) | *86%* | Best validation accuracy |
| Optuna Auto-Tuning | 84.4% | Verified model robustness |
| Lesion-Aware Fusion (MIL + CORN) | Val Loss = *0.2293* | Best ordinal consistency |

---

## 🧩 Key Observations
- *Mixup augmentation* produced the highest accuracy (86%), showing strong regularization effects.  
- *SMOTE + Mixup* improved recall for underrepresented classes.  
- *Weighted losses* helped stabilize training but yielded marginal gains.  
- *Optuna* validated that the tuned baseline (Mixup) already approached optimal performance.  
- *Attention MIL model* achieved the best ordinal consistency and interpretability.  
- *Transformer hybrid* captured spatial–contextual dependencies but underperformed slightly due to computational overhead.

---

## 🔮 Future Work
- Integrate *Vision Transformer (ViT)* and *ConvNeXt* backbones for improved representation learning.  
- Incorporate *Grad-CAM* and *Attention Visualization* for clinical interpretability.  
- Extend training to multi-dataset setups (IDRiD, Messidor) for better generalization.  
- Develop a *Streamlit/Flask web interface* for real-time DR grading and visualization.

---

## 📚 References
1. APTOS 2019 Blindness Detection Dataset (Kaggle)  
2. Tymchenko et al., ICPRAM 2020 — Multi-task CNN with Focal, MSE, and Ordinal Regression  
3. Yang et al., PLOS ONE 2024 — ViT with Masked Autoencoder pretraining  
4. Ahmed & Bhuiyan, arXiv 2025 — Class-balanced EfficientNet Framework  
5. Aftab & Akhtar, JSEA 2025 — CLAHE + SMOTE + Ensemble Fusion  

---
