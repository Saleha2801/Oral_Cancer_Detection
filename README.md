# 🧬 Multimodal Deep Learning for AI-Assisted Early Detection of Oral Cancer

## 📌 Overview

Oral cancer, particularly Oral Squamous Cell Carcinoma (OSCC), is a major cancer affecting the oral cavity. Early detection is important because identifying abnormal lesions at an early stage can support timely clinical evaluation and treatment.

This project proposes an **AI-assisted multimodal deep learning system** for oral lesion classification by combining:

- 🖼️ Histopathological images
- 🧑‍⚕️ Clinical and demographic information

Instead of relying only on image-based analysis, the proposed system integrates both visual and clinical information to provide a more comprehensive prediction.

The system is designed to classify oral lesions into three categories:

1. **Normal**
2. **Potentially Malignant** – Leukoplakia/Dysplasia
3. **Malignant** – Oral Squamous Cell Carcinoma (OSCC)

The project also incorporates **Explainable AI (XAI)** techniques such as **Grad-CAM** and **SHAP** to provide insights into the model's predictions.

> ⚠️ **Disclaimer:** This project is an academic AI research prototype and is not intended to replace professional medical diagnosis or clinical decision-making.

---

## 🎯 Objectives

The main objectives of this project are:

- To identify relevant clinical, demographic, and histopathological features associated with oral cancer detection.
- To preprocess histopathological images and clinical data.
- To develop a CNN-based image feature extraction model using transfer learning architectures such as EfficientNet, ResNet, or DenseNet.
- To extract useful features from clinical and demographic data using a dense neural network (MLP).
- To combine image and clinical features using a multimodal feature fusion approach.
- To classify oral lesions into Normal, Potentially Malignant, and OSCC categories.
- To evaluate the performance using Accuracy, Precision, Recall, F1-Score, AUC-ROC, and Confusion Matrix.
- To use Grad-CAM for visual explanation of image-based predictions.
- To use SHAP for interpreting the contribution of clinical features.

---

## 🏗️ Proposed Architecture

```text
                         ┌──────────────────────┐
                         │   Dataset Collection │
                         └──────────┬───────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
          Histopathology Images             Clinical Metadata
                    │                               │
                    ▼                               ▼
          Image Preprocessing              Clinical Preprocessing
                    │                               │
                    ▼                               ▼
            CNN / EfficientNet /              MLP
              ResNet / DenseNet                │
                    │                           │
                    ▼                           ▼
             Image Features              Clinical Features
                    │                           │
                    └─────────────┬─────────────┘
                                  ▼
                    ┌──────────────────────────┐
                    │  Multimodal Feature      │
                    │  Fusion                   │
                    │  (Concatenation /        │
                    │   Attention-Based)       │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Deep Learning Classifier  │
                    └────────────┬─────────────┘
                                 │
                                 ▼
               ┌──────────────────────────────────┐
               │       Oral Lesion Classes        │
               │                                  │
               │  Normal                          │
               │  Potentially Malignant           │
               │  OSCC                            │
               └───────────────┬──────────────────┘
                               │
                               ▼
                    ┌──────────────────────────┐
                    │ Explainable AI           │
                    │                          │
                    │ Grad-CAM → Image Regions │
                    │ SHAP → Clinical Features│
                    └──────────────────────────┘