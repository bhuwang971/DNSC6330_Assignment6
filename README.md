# DNSC6330_Assignment 6 - RML Overview

### Basic Information

1. Person or organization developing model: Bhuwan Gupta (bhuwang@gwu.edu)
2. Model date: August, 2021-2025
3. Model version: 2.0
4. License: MIT
5. Model implementation code: [DNSC_6301_Example_Project.ipynb](https://github.com/jphall663/GWU_DNSC_6301_project/blob/main/DNSC_6301_Example_Project.ipynb) 

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
  Processed Home Mortgage Disclosure Act (**HMDA**) dataset (2020 snapshot provided with Assignment 1).

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

##### Data Dictionary  

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

### Assignment‑wise Results Summary

| Assignment | Objective / Key Steps | Main Findings |
|------------|----------------------|---------------|
| **1 – Model Training** | Trained & compared three explainable models on the 2020 HMDA data set: **GLM (elastic‑net)**, **monotonic XGBoost**, and **Explainable Boosting Machine (EBM)**. | **EBM** achieved the top validation **AUC = 0.8250**; GLM served as the baseline for subsequent comparisons. |
| **2 – Explanation & Importance** | Generated global & local feature‑importance plots and partial‑dependence curves for each model. | All models ranked the same top drivers (debt‑to‑income ratio, income, property value, etc.); EBM provided the clearest additive patterns. |
| **3 – Fairness & AIR Remediation** | Evaluated discrimination via Adverse‑Impact Ratio (AIR) and performed a random‑grid search plus cut‑off tuning to satisfy **AIR ≥ 0.80**. | Remediated EBM lifted **Black / White AIR from 0.768 → 0.824** at cut‑off 0.22 with < 0.3 pp AUC loss. |
| **4 – Red‑Teaming** | Executed a model‑extraction attack (one‑tree surrogate via a single API call) and crafted > 1 000 adversarial input rows. | Surrogate diagram (`stolen_dt.png`) confirms extraction risk; crafted rows can systematically push pricing predictions up or down. |
| **5 – Debugging & Robustness** | Ran recession stress‑test, residual‑outlier removal, and class re‑balancing. | Final EBM reached **Validation AUC 0.724** with **min AIR 1.008**; simulated recession dropped AUC to ≈ 0.59, signaling the need for live monitoring. |

### Quantitative Analysis

Models were assessed primarily with **AUC** (predictive performance) and **AIR** (fairness).  
Results for the best remediated EBM are shown below.

**Table 1 – AUC values across data partitions**

| Metric      | Train AUC | Validation AUC |
|-------------|-----------|----------------|
| **AUC Score** | **0.7728** | **0.7243** |

**Table 2 – Validation AIR values (four‑fifths rule)**

| Group (Minority / Reference) | Validation AIR |
|------------------------------|----------------|
| Black / White                | **0.808** |
| Asian / White                | **1.175** |
| Female / Male                | **0.952** |


###### Plots

* Assignment 1:

![image](https://github.com/user-attachments/assets/d24e0c4f-940f-4c40-922b-6fc5b2d290e6)
Figure 1: Data exploration histograms

![image](https://github.com/user-attachments/assets/6b8ed89d-c771-4911-8dd6-05f825687483)
Figure 2: Correlation heatmap

* Assignment 2:

![image](https://github.com/user-attachments/assets/e11598dd-aa17-4501-9de7-80626ce4099d)
Figure 3: Global feature importance

![image](https://github.com/user-attachments/assets/f209f57e-cd3c-4067-b4a1-070e43cd7c86)
Figure 4: Local feature importance

![image](https://github.com/user-attachments/assets/f5339d6f-1956-4303-8c9e-c15f6f564e1d)
Figure 5: Partial dependence for all features

* Assignment 3:

![image](https://github.com/user-attachments/assets/8fd78271-594b-436d-a2f0-c8eaab530569)
Figure 6: AIR v/s AUC for EBM

* Assignment 4:

![image](https://github.com/user-attachments/assets/1ce15874-c0a5-40e1-8de3-f53ae75b8146)
Figure 7: Simulated data histograms

![image](https://github.com/user-attachments/assets/d5532204-f9ba-4b53-b66b-cbdedcbc37ef)
Figure 8: Stolen model

![image](https://github.com/user-attachments/assets/5bd097fa-0a62-4d73-b190-77141b488291)
Figure 9: Variable importance for stolen model

* Assignment 5:

![image](https://github.com/user-attachments/assets/f5fe67cf-911d-4923-ba74-9ad589cc7581)
Figure 10: Simulate recession conditions in validation data plots

![image](https://github.com/user-attachments/assets/1bf09bef-9f0e-4a81-82e5-c5246322006f)
Figure 11: Global logloss residuals plot

##### Alternative models considered

| Model | Implementation & Libraries | Feature Set | Key Hyper‑parameters & Constraints |
|-------|---------------------------|-------------|------------------------------------|
| **Elastic‑Net Generalized Linear Model (GLM)** | *H2O GLM* in binomial mode (logistic regression). 10‑fold cross‑validation used for automatic λ path selection. | Same 10 numeric / binary predictors as the EBM baseline. | • `alpha` grid: {0 (lasso), 0.25, 0.5, 0.75, 1 (ridge)}<br>• `lambda_search = TRUE` with early stopping on deviance<br>• Standardized inputs; class‑balancing disabled to keep raw prevalence<br>• Maximum iterations = 1 000 with convergence tolerance 1e‑5 |
| **Monotonic XGBoost Gradient‑Boosted Trees** | *xgboost* v1.7 – tree booster with monotone constraints applied. | Same 10 predictors; directional constraints imposed where domain logic was clear (e.g., **debt‑to‑income ↑ → probability ↑**, **income_std ↑ → probability ↓**). | • `max_depth = 3` (shallow trees for transparency)<br>• `eta = 0.05`, `subsample = 0.8`, `colsample_bytree = 0.8`<br>• `nrounds` determined via 5‑fold CV with early stopping (50 rounds patience)<br>• `min_child_weight = 10`, `gamma = 0.0` to discourage spurious splits<br>• Monotone constraint vector supplied as `monotone_constraints="(0,0,0,1,0,0,-1,1,0,0)"` |

### Ethical Considerations

#### Potential Negative Impacts
* **Math / software issues**  
  - *Histogram‑bin precision* – the EBM’s `max_bins = 512` discretisation can cause round‑off drift that flips scores across the 0.22 decision cut‑off.  
  - *Random‑seed sensitivity* – the 10–25‑trial random grid search used in Assignment 3 yields materially different models when the seed changes.
  - *Software‑update risk* – future releases of `interpret` or its dependencies could change default binning or scoring logic; strict version‑pinning and automated regression tests are required.

* **Security & robustness**  
  - The red‑teaming exercise (Assignment 4) showed a one‑tree surrogate can be stolen with **one API call** (`stolen_dt.png`), enabling > 1 000 crafted inputs that steer predictions.

* **Fairness limitations**  
  - AIR parity was verified only for **race** (Black / Asian / White) and **sex** (Female / Male); other protected classes (e.g., age, geography) remain un‑audited.

* **Real‑world harm**  
  - In the recession stress‑test (Assignment 5) AUC fell from ≈ 0.72 → 0.59; borrowers—especially minority first‑time applicants—could be over‑charged or declined when macro‑conditions shift.

#### Potential Uncertainties
* **Distribution shift** – future lending rules or housing‑market shocks could push `income_std`, `property_value_std`, etc. outside the 2020 HMDA range, and the 4‑feature EBM may extrapolate poorly.  
* **Model drift & monitoring** – no online monitoring is in place; adversarial or gradual drift could silently degrade both AUC and AIR.  
* **Metric sufficiency** – AIR meeting the four‑fifths rule does **not** guarantee equal precision/recall, so hidden disparities may persist.

#### Unexpected Results During Training & Evaluation
* Raising the probability cut‑off **0.19 → 0.22** increased Black / White AIR **0.768 → 0.824** with **< 0.3 pp** AUC loss.  
* Removing ten extreme log‑loss outliers and down‑sampling the majority class *improved* validation AUC **0.7202 → 0.7243** (Assignment 5).  
* `no_intro_rate_period_std` surfaced as a highly influential feature in some grid‑search runs—an unexpected proxy requiring domain review.  
* Even a mild simulated recession (10 % loans perturbed) dropped AUC to ≈ 0.592, revealing greater sensitivity than anticipated.

> **Who, when, how?**  
> • **Who** – Prospective borrowers (especially low‑income or minority), plus lenders relying on the scores.  
> • **When** – At rate‑quote or automated underwriting, particularly during economic volatility.  
> • **How** – Through misclassification (false positives/negatives), adversarial manipulation, or silent drift leading to unfair pricing decisions.
