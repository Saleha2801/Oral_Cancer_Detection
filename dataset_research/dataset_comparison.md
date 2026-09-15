# Comprehensive Academic Dataset Research & Comparison for Oral Cancer AI

## 1. Executive Summary & Research Context
Our academic project, **“Multimodal Deep Convolutional Neural Network Pipeline for AI-Assisted Early Detection of Oral Cancer,”** utilizes a multimodal strategy comprising:
1. **Macroscopic Clinical Oral Photography** (`Clinical oral images`, 1,081 images) for surface screening.
2. **Microscopic Histopathological Biopsy** (`NDU-UFES`, 237 images) for tissue-level gold-standard diagnosis.
3. **Sociodemographic & Epidemiological Clinical Data** (`ndb-ufes.csv`, 237 patient cases) for risk stratification.

To expand the dataset ethically and scientifically, we evaluated open, peer-reviewed medical repositories (Mendeley Data, Kaggle, Zenodo, NIH/NCI GDC).

---

## 2. Evaluation of 10 Critical ML & Clinical Compatibility Criteria

| Criterion | Evaluation & Scientific Protocol |
| :--- | :--- |
| **1. Class Compatibility** | Our target classification is **Normal / Benign**, **Potentially Malignant (Leukoplakia / Dysplasia)**, and **Malignant (OSCC)**. External datasets often have only binary labels (Normal vs OSCC) and lack dysplasia severity grading. We **must not** force binary datasets into 3-class dysplasia categories. |
| **2. Label Definitions** | NDU-UFES histological labels were verified by board-certified oral pathologists using WHO histological grading criteria. Datasets without pathology-confirmed ground truth (e.g. unverified web scrapes) are excluded. |
| **3. Modality Separation** | Clinical intraoral photos (macroscopic, camera RGB, tissue surface) and histopathology slides (microscopic H&E stain, cellular architecture) **must never be combined into a single training tensor**. They are separate sensory channels processed by dedicated CNN branches. |
| **4. Patient-Level Splitting** | If patient IDs are available (as in NDU-UFES with `patient_id` and TCGA with barcodes), splitting into train, validation, and test **must be grouped by patient**. Putting patches or slides from the same patient into both train and test causes catastrophic data leakage. |
| **5. Training vs External Validation** | Direct training merging is only permissible when optical acquisition, staining, resolution, and class definitions align. When acquisition conditions differ, external datasets should be utilized as **independent external test cohorts**. |
| **6. Duplicate Patient / Image Risk** | Checked via cryptographic MD5 image hashing. Public web datasets often replicate images across platforms (e.g. Wikimedia, Kaggle mirrors). |
| **7. Staining & Optical Variations** | Microscopic slides exhibit significant laboratory-specific variations in Hematoxylin & Eosin (H&E) staining intensity, illumination, and magnification (10x, 20x, 40x, 100x, 400x). Macroscopic photos exhibit flash reflection, angle, and camera white-balance differences. |
| **8. Normalization Protocols** | Histopathology images should undergo standard ImageNet normalization or Macenko/Reinhard stain normalization. Clinical oral images require standard color normalization (`mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`). |
| **9. Paired Clinical Metadata** | Only NDU-UFES and TCGA provide true paired clinical-pathological metadata. Datasets without patient records cannot be used for clinical MLP or multimodal feature fusion. |
| **10. Licensing & Academic Compliance** | All selected datasets permit academic research under CC BY 4.0, CC0, or NIH Open Access. |

---

## 3. In-Depth Comparison of Candidate Datasets

### A. NDB-UFES (Current Base Histopathology & Clinical Cohort)
- **Source:** Mendeley Data ([DOI: 10.17632/bbmmm4wgr8](https://doi.org/10.17632/bbmmm4wgr8))
- **Publication:** Campos et al., *Data in Brief*, 2021.
- **Samples:** 237 images (OSCC: 91, Leukoplakia with dysplasia: 89, Leukoplakia without dysplasia: 57).
- **Modality:** Histopathology (10x and 40x optical microscopy).
- **Paired Clinical Data:** Yes (age group, gender, anatomical site, lesion size, tobacco use, alcohol consumption, sun exposure).
- **Patient IDs:** Yes (`patient_id` available).
- **Role in Project:** Primary base dataset for biopsy classification, clinical risk prediction, and multimodal late fusion.
- **Scientific Strengths:** Rigorous histological ground truth, patient IDs for zero-leakage splitting, paired clinical features.

### B. Histopathological Imaging Database for Oral Cancer (Rahman et al.)
- **Source:** Mendeley Data ([DOI: 10.17632/ftmp4cvtmb](https://doi.org/10.17632/ftmp4cvtmb))
- **Publication:** Rahman et al., Elsevier, 2020.
- **Samples:** 1,224 images from 230 patients (Normal: 89 at 100x, 201 at 400x; OSCC: 439 at 100x, 495 at 400x).
- **Modality:** Microscopic H&E biopsy slides (100x and 400x magnification).
- **Paired Clinical Data:** None (pure image collection).
- **Compatibility Verdict:** **PARTIALLY COMPATIBLE.**
- **RECOMMENDED ROLE:** **External Validation or Pre-Training Only.**
- **CRITICAL WARNING — DO NOT DIRECTLY MERGE WITH NDB-UFES:**
  1. *Label Mismatch:* Rahman et al. contains only **Normal** and **OSCC**. It contains **zero** cases of Leukoplakia or epithelial dysplasia. Merging it directly into NDB-UFES would artificially dilute the critical intermediate "dysplasia" class.
  2. *Magnification Divergence:* Rahman images are at 100x and 400x; NDB-UFES is predominantly at 10x and 40x.
  3. *Staining Discrepancies:* Different laboratory H&E staining protocols would introduce domain-shift shortcuts where CNNs classify stain variations rather than cell morphology.

### C. Oral Cancer Lips and Tongue Images (Kaggle / Shivam17299)
- **Source:** Kaggle ([oral-cancer-lips-and-tongue-images](https://www.kaggle.com/datasets/shivam17299/oral-cancer-lips-and-tongue-images))
- **Samples:** 1,081 clinical photos (Cancer: 587, Non-cancer: 495).
- **Modality:** Clinical intraoral photography.
- **Paired Clinical Data:** None.
- **Role in Project:** Primary clinical oral photography dataset for macroscopic lesion screening.
- **Scientific Strengths:** Balanced binary distribution of visible oral mucosal abnormalities.

### D. Annotated Oral Cavity Images Dataset (Zenodo / AIIMS New Delhi / Lin et al. 2024)
- **Source:** Zenodo ([DOI: 10.1016/j.oraloncology.2024.106946](https://doi.org/10.1016/j.oraloncology.2024.106946))
- **Publication:** Lin et al., *Oral Oncology*, 2024.
- **Samples:** 3,000 intraoral images (Healthy, Benign, OPMD, OSCC) with COCO segmentation masks and clinical parameters.
- **Modality:** Clinical intraoral photography.
- **Compatibility Verdict:** **HIGHLY COMPATIBLE FOR CLINICAL PHOTOGRAPHY.**
- **RECOMMENDED ROLE:** External validation for the clinical photography model, and future multi-class OPMD expansion.
- **Access Protocol:** Requires institutional data use request on Zenodo; not downloadable via automated scripts.

### E. GDC TCGA-HNSC (The Cancer Genome Atlas - Head and Neck)
- **Source:** NIH National Cancer Institute ([GDC Data Portal](https://portal.gdc.cancer.gov/projects/TCGA-HNSC))
- **Samples:** 528 cases (thousands of gigapixel whole-slide images).
- **Modality:** Digital Whole Slide Pathology (WSI, SVS format).
- **Paired Clinical Data:** Full genomic, TNM clinical stage, and survival data.
- **Compatibility Verdict:** Suitable for advanced external validation of OSCC whole-slide tissue patterns.

---

## 4. Synthesis & Faculty Recommendation Table

| Dataset | Modality | Best Academic Role | Direct Merge Status | Reason |
| :--- | :--- | :--- | :--- | :--- |
| **NDB-UFES** | Histology + Tabular | Base Primary Training & Multimodal Benchmarking | Current Base | Gold-standard paired clinical-pathological cohort with dysplasia grading. |
| **Kaggle Lips & Tongue** | Photography | Base Primary Training (Oral Photo Model) | Current Base | Direct visual screening of superficial oral lesions. |
| **Rahman et al.** | Histology | **External Validation Cohort** | **DO NOT MERGE** | Binary (Normal vs OSCC); lacks dysplasia; magnification/staining domain shift. |
| **AIIMS / Lin et al.** | Photography | **External Clinical Validation** | **DO NOT MERGE** | Different camera optical profiles; best kept as held-out external benchmark. |
| **TCGA-HNSC** | Histology (WSI) | **Secondary Advanced Validation** | **DO NOT MERGE** | Gigapixel whole-slide scans require pyramidal tiling; multi-site head and neck. |

---

## 5. How to Manually Add and Integrate External Datasets

To ensure the codebase is clean, reproducible, and ready to accept additional data without ad-hoc code modifications:

1. **Target Directory Structure:**
   ```text
   Oral_Cancer_Detection/
   ├── datasets/
   │   ├── external/
   │   │   ├── histopathology_external/
   │   │   │   ├── cancer/
   │   │   │   └── non_cancer/
   │   │   └── clinical_external/
   │   │       ├── cancer/
   │   │       └── non_cancer/
   ```

2. **Manual Download Steps:**
   - **For Rahman et al. Histopathology:**
     1. Visit Mendeley Data: [https://doi.org/10.17632/ftmp4cvtmb](https://doi.org/10.17632/ftmp4cvtmb)
     2. Download `OSCC.zip` and `Normal.zip`.
     3. Extract images into `datasets/external/histopathology_external/cancer` and `datasets/external/histopathology_external/non_cancer`.
   - **For Annotated Oral Cavity Photography (Lin et al.):**
     1. Request access on Zenodo: [https://doi.org/10.1016/j.oraloncology.2024.106946](https://doi.org/10.1016/j.oraloncology.2024.106946)
     2. Extract images into `datasets/external/clinical_external/`.

3. **Running the Pipeline with an External Dataset:**
   The training pipeline accepts the `--dataset` or `--data-dir` argument:
   ```bash
   python -m training.train --model ResNet50 --dataset external_histo --data-dir datasets/external/histopathology_external
   ```
