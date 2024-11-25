# Medical Image Segmentation using the EM Algorithm

This project implements an **Expectation-Maximization (EM) algorithm** for segmenting brain tissues into **Grey Matter (GM)**, **White Matter (WM)**, and **Cerebrospinal Fluid (CSF)** from MRI images. The pipeline preprocesses brain MRI scans using **T1** and **T2_FLAIR** modalities, applies clustering techniques, and evaluates the results using **Dice similarity coefficients**.

---

## **Project Overview**

### **1. Problem Definition**
- Segment brain tissues from MRI scans into GM, WM, and CSF.
- Use **T1** and **T2_FLAIR** modalities for preprocessing and clustering.
- Evaluate segmentation results with **Dice similarity coefficients**.

### **2. Methodology**
- **Preprocessing:**
  - **Normalization:** Normalize T1 and T2_FLAIR intensities for consistent intensity ranges.
- **Clustering and EM Algorithm:**
  - **KMeans Initialization:** Initialize cluster centers for GM, WM, and CSF.
  - **EM Algorithm:** Refine cluster assignments and estimate parameters (mean, variance, mixing coefficients).
  - **Cluster Sorting:** Sort clusters by mean intensities to assign consistent labels (1: CSF, 2: GM, 3: WM).
- **Evaluation:**
  - Use Dice similarity coefficient to compare predicted labels with ground truth masks.

---

## **Results**

### **Dice Scores for T1 Modality**

| Folder | Mean Dice | Std Dice | GM Dice | WM Dice | CSF Dice |
|--------|-----------|----------|---------|---------|----------|
| 1      | 0.846     | 0.029    | 0.806   | 0.862   | 0.873    |
| 2      | 0.529     | 0.232    | 0.241   | 0.539   | 0.805    |
| 3      | 0.823     | 0.038    | 0.788   | 0.824   | 0.856    |
| 4      | 0.857     | 0.021    | 0.818   | 0.863   | 0.856    |
| 5      | 0.881     | 0.012    | 0.866   | 0.893   | 0.885    |

---

### **Key Observations**
- **Grey Matter (GM):**
  - Mean Dice scores vary across folders, with Folder 5 achieving the highest score (0.866).
  - Folder 2 demonstrates significantly lower GM Dice scores (0.241).
- **White Matter (WM):**
  - WM segmentation consistently performs well across folders (0.824 - 0.893).
  - Folder 2 has the lowest WM Dice score (0.539).
- **Cerebrospinal Fluid (CSF):**
  - CSF segmentation shows high accuracy (0.805 - 0.885), with Folder 2 being slightly lower.
- **Folder 2 Challenges:**
  - The presence of lesions in T2_FLAIR images caused misclassification, leading to lower scores.

---

## **Analysis of Folder 2 Performance**
- **Issue:** T2_FLAIR lesions with higher intensity confused the algorithm, causing misclassification as WM.
- **Solution:** Using only T1 images improved Dice scores:
  - CSF: 0.853
  - GM: 0.802
  - WM: 0.774

---

## **Conclusion**
- Successfully segmented brain tissues (GM, WM, CSF) using the EM algorithm.
- Evaluated segmentation with Dice similarity coefficients, achieving high performance for WM and GM.
- Future improvements include refining CSF segmentation and addressing modality-based discrepancies (e.g., T2_FLAIR lesions).
