# DNSC6330_Assignment6 - Overview

### Basic Information

1. Person or organization developing model: Bhuwan Gupta (bhuwang@gwu.edu)
2. Model date: August, 2021-2025
3. Model version: 2.0
4. License: MIT
5. Model implementation code: [DNSC_6301_Example_Project.ipynb](https://github.com/jphall663/GWU_DNSC_6301_project/blob/main/DNSC_6301_Example_Project.ipynb) 

### Intended Use

### Intended Use

* **Business value**  
  The project demonstrates—on real HMDA data—how an **interpretable, bias‑remediated model (EBM)** can flag loans likely to be *high‑priced* (APR ≥ 150 bps) and supply the fairness documentation regulators expect. It gives executives and compliance teams a concrete template for technical review and rapid incident response, aligning with the transparency objective in Assignment 6.

* **How the model is designed to be used**  
  *Offline* within the course repository to:  
  - Batch‑score the 2020 HMDA dataset (Assignment 1)  
  - Inspect global & local explanations (Assignment 2)  
  - Audit AIR fairness and apply remediation (Assignment 3)  
  - Probe security via model‑extraction & adversarial tests (Assignment 4)  
  - Stress‑test and debug via recession and residual analysis (Assignment 5)

* **Intended users**  
  - Students in DNSC 6301/6330 learning responsible ML  
  - Academic researchers studying interpretable fair‑lending models  
  - Practitioners seeking a worked example of model‑card documentation and AIR remediation  

* **Additional‑purpose statement**  
  The model **must not** be used for real‑world mortgage decisioning or any production system without full regulatory compliance checks and human oversight; it is intended solely for educational demonstration on the 2020 HMDA training/test files.

### Training Data

* **Source of training data**  
  Processed Home Mortgage Disclosure Act (**HMDA**) dataset (2023 snapshot provided with Assignment 1).

* **How the data was divided**  
  `hmda_train_preprocessed.csv` was split into training and validation sets with **70 % training** and **30 % validation** using:

  ```python
  SEED = 12345
  np.random.seed(SEED)
  split_ratio = 0.7  # 70% / 30% train‑test split
  split = np.random.rand(len(data)) < split_ratio
  train = data[split]
  valid = data[~split]

* **Row counts**  
  * Training rows  = **112,253**  
  * Validation rows = **48,085**

###### Data Dictionary  

| Name | Modeling Role | Measurement Level | Description |
|------|---------------|-------------------|-------------|
| **term_360** | input | int (binary) | 1 = loan term is 360 months; 0 = other term |
| **conforming** | input | int (binary) | 1 = conforming loan; 0 = non‑conforming |
| **debt_to_income_ratio_missing** | input | int (binary) | 1 = borrower’s DTI missing |
| **loan_amount_std** | input | float | *Engineered* – z‑scored loan amount |
| **loan_to_value_ratio_std** | input | float | *Engineered* – z‑scored LTV ratio |
| **no_intro_rate_period_std** | input | float | *Engineered* – z‑scored flag for *no* introductory‑rate period |
| **intro_rate_period_std** | input | float | *Engineered* – z‑scored intro‑rate period length |
| **property_value_std** | input | float | *Engineered* – z‑scored property value |
| **income_std** | input | float | *Engineered* – z‑scored borrower income |
| **debt_to_income_ratio_std** | input | float | *Engineered* – z‑scored DTI ratio |
| **high_priced** | target | int (binary) | 1 = APR ≥ 150 bps above benchmark; 0 = otherwise |

<sub>*Engineered columns:* variables with the `_std` suffix are standardised (z‑scored); the `_missing` suffix marks missing values.</sub>

### Evaluation Data

* **Source of evaluation (test) data**  
  Processed HMDA dataset (2023 snapshot — `hmda_test_preprocessed.csv`).

* **Row count**  
  Test rows  = **19,831** rows.

* **Differences vs training data**  
  The test file has the identical column schema as the training set (all feature columns present); the only distinction is that the target label `high_priced` is withheld.

### Model Details

* **Input columns**  
  `income_std`, `property_value_std`, `debt_to_income_ratio_missing`, `no_intro_rate_period_std`, `conforming`

* **Target column**  
  `high_priced`

* **Model type**  
  **Explainable Boosting Machine (EBM)** — an additive, interpretable gradient‑boosting method.

* **Software**  
  Python 3.x with scikit‑learn **0.22.2.post1** (EBM via the `interpret` library).

* **Hyper‑parameters / settings**  

  ```python
  rem_params = {
      'max_bins': 128,
      'max_interaction_bins': 16,
      'interactions': 10,
      'outer_bags': 12,
      'inner_bags': 0,
      'learning_rate': 0.01,
      'validation_size': 0.5,
      'min_samples_leaf': 5,
      'max_leaves': 5,
      'n_jobs': 4,
      'early_stopping_rounds': 100,
      'random_state': 12345
  }

## Assignment-wise Results Summary

#### Assignment 1: Model Training on Explainable Models
   * Objective: Train and compare GLM, Monotonic XGBoost, and EBM models on HMDA data
   * Key Models: GLM, XGBoost (monotonic), EBM
   * Train AUCs: GLM - 0.7652 | XGBoost - 0.80008 | EBM - 0.8314
   * Validation AUCs: GLM - 0.7538 | XGBoost - 0.7916 | EBM - 0.8250
   * Observations: EBM achieved the best performance among explainable models. GLM served as baseline.

#### Assignment 2: Model Explanation and Feature Importance
   * Objective: Use global/local importance and partial dependence to compare model behaviors
   * Approach: Extract SHAP values, regression coefficients, and EBM scores
   * Visuals: Global bar plots, PDPs for key variables, local explanations at 10th/50th/90th percentiles
   * Observations: Models showed consistent directionality on top features, but differed in strength and interaction detection

#### Assignment 3: Fairness Testing and Remediation (AIR)
   * Objective: Test models for discrimination using AIR; improve without reducing AIR below 0.8
   * Initial EBM AIRs: Black vs White - 0.843 | Asian vs White - 1.109 | Female vs Male - 0.999
   * Remediated EBM AIRs: Black vs White - 0.943 | Asian vs White - 1.159 | Female vs Male - 1.021
   * Validation AUC: 0.8250 (no performance drop from original EBM)
   * Observations: Grid search helped improve fairness while retaining model performance

#### Assignment 4: Red Teaming and Adversarial Testing
   * Objective: Simulate adversarial attacks on model via model extraction and counterexamples
   * Methods: Decision tree model extraction, adversarial input generation, attack simulation
   * Outcomes: Exposed model vulnerabilities to feature flipping and response skew
   * Observations: EBM showed resilience to mild attacks but lacked protection against well-crafted inputs

#### Assignment 5: Debugging and Robustness (Final Remediated EBM)
   * Objective: Stress test EBM under recession and improve residual error patterns
   * Stress Test Result: AUC degraded under recession conditions
   * Residual Fixes: Outlier removal and reweighting improved stability
   * Final AUCs: Train - 0.7984 | Validation - 0.8005 | True Test - 0.7801
   * Final AIRs: Black vs White - 0.981 | Asian vs White - 1.110 | Female vs Male - 1.008
   * Observations: Final model achieved optimal tradeoff between fairness and performance


## Quantitative Analysis
Models were assessed primarily with AUC and AIR. See details below:

| Metric    | Train AUC | Validation AUC |
|:--------:|:---------:|:--------------:|
| AUC Score | 0.7802    | 0.7739         |
    
###### Table 1. AUC values across data partitions.
    
| Group               | Validation AIR |
|:------------------:|:--------------:|
| Black vs. White     | 0.787          |
| Asian vs. White     | 1.157          |
| Female vs. Male     | 0.958          |
    
###### Table 2. Validation AIR values for race and sex groups.
 

## Figures 

* Assignment 1

![image](https://github.com/user-attachments/assets/3bde93c3-9e85-4ba3-9d58-afdf3ff6768a)

Figure 1. Histograms data exploration.

![image](https://github.com/user-attachments/assets/5509c352-3f74-4185-851f-3f241cd33f8b)

Figure 2. Heatmaps correlations.

* Assignment 2

![image](https://github.com/user-attachments/assets/52804666-89ee-4ab0-9172-75f08d00f5f3)

Figure 3. Global feature importance.

![image](https://github.com/user-attachments/assets/491d1399-613e-4da3-a9f1-a5a56ec8a6c5)

Figure 4. Local feature importance

![image](https://github.com/user-attachments/assets/bb860789-ef80-4205-806b-b0761212c2d6)

Figure 5. Partial dependence feature

* Assignment 3

![image](https://github.com/user-attachments/assets/1d3c8db5-3369-4cb5-bd2e-1467fd8e9ce6)

Figure 6. AIR V/S AUC EBM

* Assignment 4

![image](https://github.com/user-attachments/assets/6d3b72fc-93af-4f99-993e-0434f092f6fd)

Figure 7. Simulated data

![image](https://github.com/user-attachments/assets/ca7a066e-48ea-41bf-8e6b-397fc4b3153d)

Figure 8. Stolen model

![image](https://github.com/user-attachments/assets/561077de-83e4-4331-921e-79abe7c0fe37)

Figure 9. Distributed random forest

* Assignment 5

![image](https://github.com/user-attachments/assets/95cfa5cc-cbb2-4cd2-b068-659d98dfe640)

Figure 10. Simulate recession conditions in validation data

![image](https://github.com/user-attachments/assets/71eb3c14-bbb0-40aa-ad43-e397ac389fc8)

Figure 11. Global Logloss Residuals 

## Ethical Considerations

### Potential Negative Impacts:
* The remediated EBM model may still encode indirect bias despite AIR parity improvements.
* Overreliance on fairness metrics like AIR can mask other disparities (e.g., precision, recall across groups).
* Software issues (e.g., float precision errors or binning instability) could lead to inconsistent scoring.
* Real-world risk: borrowers may be unfairly classified as high-risk during economic downturns, affecting credit access.

### Potential Uncertainties:

* Shifts in borrower behavior or policy changes (e.g., interest rate ceilings) could affect generalizability.
* Unknown interaction effects between features might behave unexpectedly under real-time deployment conditions.
* Model drift or adversarial manipulation could arise if integrated into production systems without monitoring.

### Unexpected Results:

* Remediated EBM showed slightly lower AUC but improved AIR, highlighting a tradeoff between accuracy and fairness.
* Feature no_intro_rate_period_std had unexpectedly high influence in certain EBM iterations.
