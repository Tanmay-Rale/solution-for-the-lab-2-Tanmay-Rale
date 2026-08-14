# Lab 2: Traffic Sign Classification & Experiment Management
### ELEC5308 – Intelligent Information Engineering Practice
**University of Sydney | School of Electrical and Computer Engineering**

---

## Overview

In this lab you will build, train, and evaluate a **Convolutional Neural Network (CNN)**
to classify traffic signs from the [GTSRB dataset](http://benchmark.ini.rub.de/), and
track every experiment using **CometML** — a professional MLOps platform.

This lab directly connects to **Lecture 2 (Computer Vision & Representation Learning)**
and covers:
- Image preprocessing and data augmentation
- CNN architectures (ConvBlocks, ResidualBlocks, pooling)
- Evaluation metrics (accuracy, precision, recall, F1 — micro / macro / weighted)
- Confusion matrices
- Transfer learning with pretrained ResNets
- Experiment tracking with CometML

---

## 🗂 Repository Structure

```
lab2-elec5308/
│
├── elec5308/                   ← PROVIDED Python package (do NOT modify)
│   ├── __init__.py
│   ├── datasets.py             ← GTSRB loader, class registry (43 classes)
│   ├── models.py               ← ConvBlock, ResidualBlock, SimpleCNN, get_pretrained_resnet
│   ├── metrics.py              ← compute_metrics, plot_confusion_matrix
│   ├── trainer.py              ← Trainer class with CometML integration
│   └── visualisation.py       ← show_sample_images, plot_class_distribution, plot_training_history
│
├── lab2_notebook.ipynb         ← YOUR WORKING NOTEBOOK ← start here
│
├── requirements.txt
├── setup.py
└── TUTOR_GUIDE.md
```

---

## ⚡ Quick Start

### 1. Clone your repository

Your repository has been distributed via GitHub Classroom.

```bash
git clone https://github.com/elec5308/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Set up your environment

**Option A: pip (recommended for most students)**

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate          # Windows

# Install the elec5308 package and dependencies
pip install -e .
pip install -r requirements.txt
```

**Option B: conda**

```bash
conda create -n elec5308 python=3.10
conda activate elec5308
pip install -e .
pip install -r requirements.txt
```

**Option C: Google Colab** *(if you don't have a GPU)*

Open `lab2_notebook.ipynb` in Colab. The first cell installs everything automatically.

### 3. Set up CometML

1. Go to [comet.ml](https://www.comet.ml) and create a **free** account.
2. Navigate to **Settings → API Keys** → copy your key.
3. Store it as an environment variable (recommended):
   ```bash
   export COMET_API_KEY="your-key-here"
   ```
4. Replace `YOUR_COMET_USERNAME` in Section 0 of the notebook.

### 4. Open the notebook

```bash
jupyter notebook lab2_notebook.ipynb
```

Work through each section in order. Read the instructions in each cell carefully.

---

## 📋 Lab Tasks

| Task | Section | Description | Points |
|------|---------|-------------|--------|
| **Task 1** | Section 2 | Data preprocessing & DataLoaders | M1 (20) |
| **Task 2** | Section 3 | Design your CNN architecture | M2 (20) |
| **Task 3** | Section 4 | Train your model with CometML | M3 (30) |
| **Task 4** | Section 5 | Evaluate metrics & confusion matrix | M4 (20) |
| **Task 5** ⭐ | Section 6 | Transfer learning with ResNet | M5 (10) |

**Total: 100 points (10 points are bonus)**

---

## 📤 Submission

### What to submit

Your GitHub repository.

---

## 🤖 AI Usage Policy

AI tools are **permitted** for this lab. However:

- You **must** complete a proper AI Journal for every meaningful AI interaction.
- AI use is **not** graded, but the journal is checked for completion.
- Copying AI-generated code without understanding it will hurt you in the quiz and exam.

---

## 📚 Dataset: GTSRB

The **German Traffic Sign Recognition Benchmark** contains:
- **51,839 images** across **43 classes**
- Real-world photos captured from dashcam footage in Germany
- Varying lighting, weather, and perspective conditions
- Strong class imbalance (most common class has ~10× more samples than rarest)

The dataset downloads automatically on first run (~300 MB).

---

## 💻 Hardware

- **CPU** is sufficient for this lab (SimpleCNN trains in ~15 min on CPU).
- **GPU** (CUDA or Apple Silicon MPS) will speed things up significantly.
- **Google Colab** (free tier) provides a T4 GPU if you don't have local hardware.

---

## ❓ Getting Help

- **Ed Discussion** — post public questions so everyone benefits.
- **Lab sessions** — tutors are available during your scheduled lab.
- **Office hours** — see Canvas for Dr Shakiba's office hours.

---

## References

- Stallkamp, J., et al. (2011). *The German Traffic Sign Recognition Benchmark*. IJCNN.  
- He, K., et al. (2016). *Deep Residual Learning for Image Recognition*. CVPR.  
- LeCun, Y., et al. (1998). *Gradient-Based Learning Applied to Document Recognition*. IEEE.  
- Simonyan, K., & Zisserman, A. (2015). *Very Deep Convolutional Networks*. ICLR.

---

*ELEC5308 Lab 2 | University of Sydney | 2026 Semester 2*
