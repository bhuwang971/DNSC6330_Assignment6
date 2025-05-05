# DNSC6330_Assignment6

## RML Assignments Overview

### Basic Information

1. Person or organization developing model: Bhuwan Gupta(bhuwang@gwu.edu)
2. Model date: August, 2021-2025
3. Model version: 2.0
4. License: MIT
5. Model implementation code: [DNSC_6301_Example_Project.ipynb](https://github.com/jphall663/GWU_DNSC_6301_project/blob/main/DNSC_6301_Example_Project.ipynb) 

### Intended Use

| Rubric bullet | Statement (sourced from assignment PDFs & course outputs) |
|--------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Business value** | The project demonstrates—on real HMDA data—how an **interpretable, bias‑remediated model (EBM)** can flag loans likely to be *high‑priced* (APR ≥ 150 bps) and provide the fairness documentation regulators expect. It offers executives and compliance teams a concrete template for technical review and rapid incident response, fully aligned with the transparency goal described in the Assignment 6 brief. |
| **How the model is designed to be used** | *Offline* within the course repo to:<br>• Batch‑score the 2020 HMDA dataset (Assignment 1)<br>• Inspect global & local explanations (Assignment 2)<br>• Audit AIR fairness and apply remediation (Assignment 3)<br>• Probe security via model‑extraction & adversarial tests (Assignment 4)<br>• Stress‑test and debug via recession and residual analysis (Assignment 5) |
| **Intended users** | • Students in DNSC 6301/6330 learning responsible ML<br>• Academic researchers studying interpretable fair‑lending models<br>• Practitioners seeking a worked example of model‑card documentation and AIR remediation |
| **Additional‑purpose statement** | The model **must not** be used for real‑world mortgage decisioning or any production system without full regulatory compliance checks and human oversight; it is intended solely for educational demonstration on the 2020 HMDA training/test files. |


### Training Data
###### Data Dictionary  
*(Column meanings sourced from Assignment 1 feature list)*

| Name | Modeling Role | Measurement Level | Description |
|------|---------------|-------------------|-------------|
| **term_360** | input | int&nbsp;(binary) | 1 = mortgage term is standard 360 months; 0 = other term |
| **conforming** | input | int&nbsp;(binary) | 1 = loan meets conforming limits; 0 = non‑conforming (jumbo, HELOC, reverse‑mortgage, etc.) |
| **debt_to_income_ratio_missing** | input | int&nbsp;(binary) | 1 = borrower’s DTI ratio is missing |
| **loan_amount_std** | input | float | Standardised (z‑scored) loan amount |
| **loan_to_value_ratio_std** | input | float | Standardised LTV ratio (loan ÷ property value) |
| **no_intro_rate_period_std** | input | float | Standardised binary flag for loans with *no* introductory‑rate period |
| **intro_rate_period_std** | input | float | Standardised length (months) of introductory‑rate period |
| **property_value_std** | input | float | Standardised appraised property value |
| **income_std** | input | float | Standardised borrower income |
| **debt_to_income_ratio_std** | input | float | Standardised borrower debt‑to‑income ratio |
| **high_priced** | target | int&nbsp;(binary) | 1 = APR ≥ 150 bps above benchmark; 0 = otherwise |

<sub>Binary indicators are encoded as 0/1; continuous features were z‑scored before modelling.</sub>

   * Source of training data: Processed HMDA dataset (2023)
   * How training data was divided into training and validation data: 50% training, 25% validation, 25% test
   * Number of rows in training and validation data:
     Training rows: 112253
     Validation rows: 48085

## Test Data
   * Source of test data: Processed HMDA dataset (2023)
   * Number of rows in test data: 19831
   * State any differences in columns between training and test data: None

## Model details
   * Columns used as inputs in the final model: 'TERM_360', 'CONFORMING', 'DEBT_TO_INCOME_RATIO_MISSING', 'LOAN_AMOUNT_STD', 'LOAN_TO_VALUE_RATIO_STD', 'NO_INTRO_RATE_PERIOD_STD', 'INTRO_RATE_PERIOD_STD', 'PROPERTY_VALUE_STD', 'INCOME_STD', 'DEBT_TO_INCOME_RATIO_STD'
   * Column(s) used as target(s) in the final model: 'HIGH_PRICED'
   * Type of model: Explainable Boosting Machine (EBM)
   * Software used to implement the model: Python, scikit-learn
   * Version of the modeling software: 0.22.2.post1
   * Hyperparameters or other settings of your model : ['loan_amount_std', 'no_intro_rate_period_std', 'term_360', 'income_std', 'debt_to_income_ratio_missing', 'intro_rate_period_std', 'property_value_std']

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
