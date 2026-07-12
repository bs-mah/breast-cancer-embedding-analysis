# Downstream Validation of Unsupervised Multi-Omics Patient Embeddings for Breast Cancer Stratification

## Project Overview

This project evaluates whether patient embeddings learned in an unsupervised 
manner from multi-omics data are clinically useful. These embeddings were 
produced by a hierarchy of models that progressively incorporate structural 
inductive bias from protein–protein interaction (PPI) networks.

The downstream validation is performed across three complementary tasks: 
supervised classification, unsupervised clustering, and patient retrieval 
by similarity.

## Objectives

- Evaluate the linear separability of clinical labels in the learned embedding spaces across all model levels
- Assess the unsupervised geometric organization of patient embeddings and their alignment with known clinical phenotypes
- Validate embedding quality through a patient retrieval paradigm and measure clinical concordance of retrieved neighbors
- Produce a comparative evaluation across model levels (Level 0 through Level 3) to determine which level of structural   inductive bias yields the most clinically informative representations

## Embedding Levels

The four embedding levels correspond to different architectural components of the model:

- Level 0 : Simple autoencoder
- Level 1 : Denoising autoencoder
- Level 2 : PPI graph regularisation
- Level 3 : GNN encoder using the PPI graph

## Repository Structure

```bash
├── README.md
├── requirements.txt
│
├── code/
│   ├── task1_classification.py     # Supervised classification (PAM50, Survival, Stage)
│   ├── task2_clustering.py         # Unsupervised clustering analysis
│   └── task3_retrieval.py          # Patient retrieval by cosine similarity
│
├── data/
│   ├── df_level0.csv               # Patient embeddings — Level 0
│   ├── df_level1.csv               # Patient embeddings — Level 1
│   ├── df_level2.csv               # Patient embeddings — Level 2
│   ├── df_level3.csv               # Patient embeddings — Level 3
│   └── PAM50labels.csv             # PAM50 subtype labels
│
└── report/
    ├── report_task1.pdf            # Task 1 — Supervised Classification
    ├── report_task2.pdf            # Task 2 — Unsupervised Clustering
    └── report_task3.pdf            # Task 3 — Patient Retrieval
```

## Dataset

Each row corresponds to one patient. The dataset combines clinical metadata, survival outcomes, processed clinical variables, and learned patient embeddings.

### Identifier

- patient_id : Patient ID (12-character format), used for reference only

### Survival Variables

- time : Follow-up time in days
- event : Survival indicator: 1 = death, 0 = censored
- survival_binary : Binarized survival outcome (median split): 1 = high risk, 0 = low risk

### Clinical Variables

- age_at_diagnosis : Age at diagnosis (years)
- stage_pathologic_overall_num : Numeric pathological stage (may contain missing values)
- T_stage : Collapsed tumour size: T1, T2, T3, T4
- N_stage : Collapsed lymph node status: N0, N1, N2, N3
- M_stage : Metastasis status: 0 = M0, 1 = M1, x = unknown
- stage_binary : Binary stage: 0 = Early (I, II), 1 = Late (III, IV)

### Patient Embeddings

- levelX_z0 → levelX_z63 : 64-dimensional embedding vector per patient, learned using unsupervised deep learning on multi-omics data

### PAM50 Labels (PAM50labels.csv)

- patient_id : Patient ID
- PAM50_subtype : Molecular subtype: Basal, Her2, LumA, LumB, Normal


## Tasks

### Task 1 — Supervised Classification
Evaluates whether the embeddings can predict clinical labels (PAM50 subtypes, survival, pathological stage) using supervised classifiers (Logistic Regression, SVM, k-NN, Random Forest, XGBoost, AdaBoost, MLP) with 5-fold stratified cross-validation repeated over 10 random seeds.

### Task 2 — Unsupervised Clustering
Evaluates whether the embeddings naturally organise into clinically coherent groups without supervision. Five clustering algorithms are applied (K-Means, Agglomerative, Spectral, GMM, HDBSCAN) and assessed using internal metrics (Silhouette, Davies-Bouldin), external metrics (ARI, NMI vs PAM50) and statistical association tests (chi-square).

### Task 3 — Patient Retrieval
Validates embedding quality through a clinical retrieval paradigm: given a query patient, are their nearest neighbours in embedding space clinically similar? Cosine similarity is used to retrieve top-k neighbours (k = 5, 10, 20) and evaluated using PAM50 Precision@k, Survival Precision@k, and Stage Concordance.


## Requirements

Install all dependencies with:

```bash
pip install -r requirements.txt
```

**Main dependencies:**
```
numpy
pandas
scikit-learn
matplotlib
scipy
xgboost
lifelines
umap-learn
```

## Usage

Place all data files in the same directory as the scripts, then run:

```bash
# Task 1 — Supervised Classification
python code/task1_classification.py

# Task 2 — Unsupervised Clustering
python code/task2_clustering.py

# Task 3 — Patient Retrieval
python code/task3_retrieval.py
```

All results (CSV files and figures) are automatically saved in the working directory.

---

## Key Results Summary

| Task | Best Level | Best Metric |
|---|---|---|
| PAM50 Classification | Level 2 | Macro F1 = 0.43 |
| Survival Classification | Level 3 | Macro F1 = 0.71 |
| PAM50 Precision@5 (retrieval) | Level 2 | 0.345 vs baseline 0.308 |
| Survival Precision@5 (retrieval) | Level 3 | 0.820 vs baseline 0.500 |
| Clustering (Silhouette) | Level 3 HDBSCAN | 0.80 |

---

## Author

Maha Bouslimani — 2026

## Supervisors

- Dr. Hamza Zidoum - Supervisor
- Habiba El Keraby - Collaborator
