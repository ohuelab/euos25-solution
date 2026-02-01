# EUOS25 Solution
<img width="741" height="418" alt="image" src="https://github.com/user-attachments/assets/735bcd5b-d32f-4a9e-be4e-4c23cc23f327" />

## Overview

This solution uses a weighted ensemble of diverse models with different molecular representations. The final submission combines predictions from 26 unique models optimized for each task.

## Models

### 1. Uni-Mol-2 (3D Molecular Representation)
- **Uni-Mol-2-84M**: 84M parameter pretrained model with Focal Loss
- **Uni-Mol-2-310M**: 310M parameter pretrained model with Focal Loss
- Uses 3D molecular conformations

### 2. Graph Neural Networks
- **ChemProp**: Standard and Focal Loss variants
- **Multitask ChemProp**: Joint training on related tasks

### 3. Transformer Models
- **Chemeleon**: SELFIES-based transformers with LightGBM/CatBoost heads
- **ChemBERTa**: Pretrained chemical language model with CatBoost

### 4. Gradient Boosting
- **LightGBM**: With Optuna hyperparameter optimization
- **CatBoost**: Various feature combinations

## Features

- **3D Conformations**: Uni-Mol-2
- **Molecular Graphs**: ChemProp
- **SMILES/SELFIES**: Transformer models
- **Molecular Descriptors**: RDKit descriptors, Morgan fingerprints (ECFP)
- **Learned Embeddings**: ChemBERTa, Uni-Mol-2

## Ensemble Strategy

**Task-specific weighted ensembles:**
Weighted-average ensemble based on CV score

**Key techniques:**
- Focal Loss for class imbalance
- Multitask learning
- 5-fold cross-validation

## Final Submission Composition

See `notebooks/final_submission_composition.ipynb` for visualization of model weights.
