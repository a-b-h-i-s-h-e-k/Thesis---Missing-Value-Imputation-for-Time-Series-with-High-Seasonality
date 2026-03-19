# Thesis - Missing Value Imputation for Time Series with High Seasonality



This repository contains the experimental code, benchmark framework, and supplementary materials for the MSc thesis:

**“Missing Value Imputation for Time Series with High Seasonality”**  
**Abhishek Bargujar**  
Supervisor: **Prof. Anton Dignös**  
Degree: MSc in Data Science  
University: Free University of Bozen–Bolzano (unibz)  
Academic Year: 2025/2026  

---

## 📘 Thesis Abstract

Time series data collected from real-world sensors frequently suffer from missing values, which can significantly degrade the quality of downstream analysis, modeling, and forecasting. This challenge becomes substantially more complex in the presence of **high seasonality**, where strong and repeating temporal patterns coexist with phase shifts and long-range dependencies.

This thesis presents a systematic comparative study of classical, matrix-based, model-based, and pattern-based missing value imputation methods under diverse missingness scenarios. Particular emphasis is placed on **pattern-based approaches**, especially the TKCM framework and its extensions, and their ability to exploit repeating seasonal structures.

Experiments are conducted on real-world environmental sensor data from South Tyrol, evaluating both **single-point** and **continuous block missing values** using standard error metrics such as MAE and RMSE.

---

## 🌦️ Data Source

All datasets used in this thesis were obtained from the **South Tyrol Open Data Portal**.

- **Stations**: 172 total stations  
  - Valley stations  
  - Mountain stations  
  - River-gauging stations  
- **Sensors**: 12 sensor types (air temperature, humidity, wind speed, water temperature, solar radiation, etc.)
- **Sampling frequency**: 30 minutes  
- **Time range**: 2023-01-01 to 2025-01-01  

Raw data were preprocessed to ensure regular sampling, temporal alignment, and consistency across stations.

---

## ❓ Missingness Definition

In this thesis, missingness refers to the absence of observed values at specific time points in a time series. The following missingness patterns are studied:

### Single Missing Values
- A single observation removed from the series
- Evaluated at:
  - Middle of the series
  - End of the series

### Block Missing Values
- Continuous segments of missing observations
- Evaluated at:
  - Middle of the series
  - End of the series
- Block sizes:
  - 1 day
  - 1 week
  - 1 month

### Global Missingness
- Single missing values distributed across the entire series
- Multiple seasonal cycles affected
- Used to evaluate robustness under dispersed information loss

---

## 🧪 Experimental Setup

### Datasets
- Valley stations (high seasonality)
- Mountain stations (higher variability)
- Gauging stations (strong local temporal continuity)

### Evaluation Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

All methods are evaluated using an identical preprocessing pipeline, missingness injection strategy, and metric computation.

---

## 🔧 Implemented Methods

### Baseline Methods
- Zero Imputation
- Minimum Value Imputation

### Matrix-Based Methods
- SVDImpute
- SoftImpute
- CDRec

### Model-Based Methods
- DynaMMo
- TRMF

### Pattern-Based Methods
- TKCM
- Weighted TKCM
- TKCM Plus *(proposed)*
- TKCM Pro *(proposed)*

---

## ✨ Contributions of This Thesis

- Systematic survey of 56 imputation algorithms
- Selection of 22 candidate methods
- Identification of 7 methods suitable for highly seasonal and phase-shifted time series
- Construction of a seasonality-aware benchmark
- Proposal of three TKCM extensions:
  - Weighted TKCM
  - TKCM Plus
  - TKCM Pro
- Large-scale experimental validation on real-world seasonal sensor data
- Extensive parameter tuning over:
  - window size (k)
  - pattern length / reference selection (l)
  - α and β weighting parameters

---


---

## ⚙️ Requirements

All experiments were executed in Python.  
Dependencies are listed in `requirements.txt`.

### Setup

- python -m venv venv
- source venv/bin/activate
- pip install -r requirements.txt



