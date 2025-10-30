# Toy Robot: Toy Classification Using CNN

A deep learning project using **Convolutional Neural Networks (CNNs)** and **transfer learning (MobileNetV2)** to classify children's toys from a custom, self-collected dataset.  
Built in **Python (TensorFlow/Keras)** as part of my personal data science portfolio.

---

## Repository Contents

- **data/** — Sample dataset (train/validate/test image folders)  
- **images/** — Supporting images for markdown documentation  
- **scripts/** — Python scripts for model training, testing, and visualization  
  - `01_data_pipeline.py`  
  - `02_baseline_model.py`  
  - `03_dropout_augmentation.py`  
  - `04_learning_rate_reducer.py`  
  - `05_architecture_experiments.py`  
  - `06_transfer_learning_mobilenet.py`  
  - `07_gradcam_visualization.py`  
- **posts/** — Markdown post for GitHub Pages (portfolio write-up)  
  - `2025-03-07-cnn-Toy-Robot.md`  
- **models/** — Saved `.h5` model files (if stored locally)  
- **README.md** — Project overview (this file)

---

## Project Overview

Inspired by my four-year-old son’s scattered toy collection, this project explores how a robot might **recognize and sort toys into bins** using computer vision.  
Using a **self-collected dataset of five toy categories** — *Bananagrams, Brios, Cars, Duplos, and Magnatiles* — I trained several CNN architectures from scratch, then implemented **transfer learning** with **MobileNetV2**, achieving perfect test accuracy.

> “If successful and scaled, no parent will ever step on a Lego again.”

---

## Methods & Model Evolution

| **Stage** | **Technique / Change** | **Test Accuracy** |
|------------|------------------------|------------------:|
| Baseline CNN (2 conv layers) | Starting architecture | 74.7% |
| + Dropout (0.5) | Reduced overfitting | 80.0% |
| + Image Augmentation | Added variation | 74.7% |
| + Learning Rate Reducer | Smoothed convergence | 76.0% |
| + Architecture Experiment 4 | 4 conv layers, tuned filters | 80.0% |
| **Transfer Learning (MobileNetV2)** | Pretrained model, fine-tuned | **100%** |

---

## Data Overview

- **Dataset size:** ~725 images total  
- **Classes:** 5 (Bananagrams, Brios, Cars, Duplos, Magnatiles)  
- **Split:** 100 training / 30 validation / 15 test per class  
- **Image size:** 128×128 pixels, RGB normalized  
- **Data source:** all images captured with iPhone, diverse lighting/backgrounds to simulate real-world variability

---

## Key Concepts

- **Overfitting mitigation:** Dropout, image augmentation, and learning rate reduction  
- **Architecture tuning:** Iterative experiments with layers, filters, and kernels  
- **Visualization:** Grad-CAM analysis to inspect model feature detection and bias  
- **Transfer learning:** Leveraging MobileNetV2 pretrained on ImageNet for lightweight, high-accuracy inference

---

## Results

- **Baseline CNN:** Suffered overfitting despite decent training accuracy.  
- **Dropout + Augmentation:** Reduced overfitting but limited improvement.  
- **Transfer Learning:** MobileNetV2 achieved **100% test accuracy** after just 7 epochs.  
- **Grad-CAM findings:** Revealed model bias toward background context; future iterations will address data bias and sample diversity.

---

## Future Work

- Expand dataset with additional toy types and lighting conditions  
- Fine-tune MobileNetV2 layers for explainability and robustness  
- Implement Keras-Tuner for automated hyperparameter optimization  
- Integrate with a robotics system for physical toy sorting (the “pick-and-place” challenge)

---

## Tech Stack

- **Python**
- **TensorFlow / Keras**
- **NumPy, Pandas, Matplotlib**
- **OpenCV, Pillow**
- **Grad-CAM visualization**

---

## Visuals

![Toy Robot sample collage](images/collage3.png)
*Example training images from five toy classes.*

---

## About the Author

**Samuel Shaw, PhD**  
Data Scientist | Sociologist | Researcher  
Seattle, WA  
[GitHub: @SammyShaw](https://github.com/SammyShaw) | [LinkedIn](https://www.linkedin.com/in/samuelclayshaw)

---

*This repository accompanies the full write-up on my [portfolio site](https://samyshaw.github.io/posts/2025-03-07-cnn-Toy-Robot).*

