# SentenceTransformer Pair Classification Trainer

This repository contains training and inference scripts for **sentence pair classification** using SentenceTransformers. It supports:
- **Binary Classification (2 Labels)** using `ContrastiveLoss`
- **Multi-Class Classification (More than 2 Labels)** using `SoftmaxLoss`

## 📌 Repository Structure

```
├── inference-for-2-labels.py      # Inference for binary classification (2 labels)
├── inference.py                   # Inference for multi-class classification (>2 labels)
├── train-using-contrastive-loss-2-labels.py  # Training script for binary classification
├── train.py                        # Training script for multi-class classification
├── README.md                       # This documentation
```

## 🚀 Getting Started

### Installation
Ensure you have Python 3.8+ and install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔧 Training

### Train for **2 Labels (Binary Classification)**
Uses `ContrastiveLoss` to train a model for sentence pair classification when there are **only two labels**.

```bash
python train-using-contrastive-loss-2-labels.py
```

### Train for **More than 2 Labels (Multi-Class Classification)**
Uses `SoftmaxLoss` to train a model when there are **more than two labels**.

```bash
python train.py
```

---

## 🧐 Inference

### Inference for **2 Labels (Binary Classification)**
Use the trained model to make predictions for **two-label** classification.

```bash
python inference-for-2-labels.py --model_path path/to/binary_model
```

### Inference for **More than 2 Labels (Multi-Class Classification)**
Use the trained model to classify sentences with **more than two labels**.

```bash
python inference.py --model_path path/to/multiclass_model
```

---

## ⚡ Model Details

| Classification Type | Loss Function      | Training Script                                     | Inference Script                     |
|--------------------|------------------|------------------------------------------------|--------------------------------|
| **Binary (2 Labels)** | ContrastiveLoss   | `train-using-contrastive-loss-2-labels.py`  | `inference-for-2-labels.py` |
| **Multi-Class (>2 Labels)** | SoftmaxLoss      | `train.py`                                 | `inference.py`               |

---

## 📌 Notes
- Ensure to provide the correct dataset format before training.
- `train-using-contrastive-loss-2-labels.py` and `train.py` require **sentence pairs** for classification.
- Modify hyperparameters like batch size, learning rate, and epochs in the scripts as needed.

---

## 🏆 Contributors
- **Your Name** - *Alpesh Sonar*

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
