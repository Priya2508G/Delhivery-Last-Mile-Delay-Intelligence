# Delhivery Last-Mile Delay Intelligence Pipeline

**Program:** Artificial Intelligence and Machine Learning Certification Program – E&ICT Academy, IIT Roorkee  
**Project:** Delhivery Last-Mile Delay Intelligence Pipeline  
**Repository:** Delhivery-Last-Mile-Delay-Intelligence  
**Branch:** `main`

## 1. Project Overview

This project develops an end-to-end machine learning pipeline to improve last-mile delivery operations.

The system uses shipment and hub information to:

- Clean and prepare shipment data.
- Predict delivery duration using regression.
- Identify shipments at risk of missing their promised delivery time.
- Segment delivery hubs according to operational performance.
- Package a reusable machine learning model.
- Generate predictions and recommended actions for new shipments.

The project uses a synthetic Delhivery dataset containing 24 hubs, historical shipment records, and new shipments for prediction.

All monetary values are represented in INR.

## 2. Repository Structure

```text
Delhivery-Last-Mile-Delay-Intelligence/
│
├── data/
│   ├── hubs.csv
│   ├── raw_shipments.csv
│   ├── new_shipments.csv
│   └── modelling_table.csv
│
├── models/
│   ├── delay_model.pkl
│   ├── delay_threshold.csv
│   ├── feature_order.csv
│   └── impute_values.csv
│
├── outputs/
│   ├── Part A data cleaning and EDA outputs
│   ├── Part B regression outputs
│   ├── Part C classification outputs
│   ├── Part D hub segmentation outputs
│   └── New shipment prediction outputs
│
├── src/
│   ├── shipment_prep.py
│   └── part_a_clean_eda.py
│
├── generate_data.py
├── part_a_clean_eda.py
├── part_b_regression.py
├── part_c_classification.py
├── part_d_segments.py
├── package_model.py
├── predict_delay.py
├── requirements.txt
└── README.md
```

## 3. Business Rules

The promised delivery time depends on the service type.

| Service type | Promised delivery time |
|---|---:|
| Express | 36 hours |
| Standard | 60 hours |
| Economy | 96 hours |

A shipment is classified as delayed when:

`delivery_hours > promised_hours`

The main objective is to predict delivery duration and identify potential delays using information available at dispatch time.

## 4. How to Run

### Prerequisites

- Python 3.x
- pandas
- NumPy
- scikit-learn
- joblib

The exact package versions are specified in `requirements.txt`.

### Clone the repository

```bash
git clone https://github.com/Priya2508G/Delhivery-Last-Mile-Delay-Intelligence.git
cd Delhivery-Last-Mile-Delay-Intelligence
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Execute the pipeline

Run the following commands in order:

```bash
python generate_data.py
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
python package_model.py
python predict_delay.py
```

### Execution order

1. `generate_data.py` generates the synthetic hub and shipment datasets.
2. `part_a_clean_eda.py` cleans the data, performs exploratory analysis, and creates the modelling table.
3. `part_b_regression.py` trains and evaluates delivery-time regression models.
4. `part_c_classification.py` trains and evaluates proactive delay-classification models.
5. `part_d_segments.py` performs hub segmentation and creates operational profiles.
6. `package_model.py` packages the final classification model and supporting configuration files.
7. `predict_delay.py` loads the packaged model and generates actions for new shipments.

**Model compatibility:** The committed model pickle is intended for use with the package versions pinned in `requirements.txt`. If the environment uses different scikit-learn versions, rerun `package_model.py` to regenerate the model before prediction.

## 5. Data Findings

### 5.1 Dataset overview

| Metric | Result |
|---|---:|
| Number of hubs | 24 |
| Raw historical shipment rows | 1,224 |
| Duplicate rows removed | 24 |
| Clean historical shipment rows | 1,200 |
| New shipments | 48 |
| Delayed historical shipments | 278 |
| Historical delay rate | 23.17% |

### 5.2 Data cleaning

The data preparation process included:

- Removing 24 duplicate historical records.
- Standardising service-type spelling and casing.
- Handling missing payment-mode values using `Unknown`.
- Converting declared values into numeric INR values.
- Parsing dispatch timestamps into datetime format.
- Identifying invalid scanner weight values above 100 kg as sentinel values.
- Replacing missing numeric values using median imputation.
- Creating the promised delivery time and delay target.
- Preparing features for modelling.

The final modelling table contains 1,200 rows and 28 columns.

### 5.3 Delay rate by service type

| Service type | Shipments | Delayed shipments | Delay rate |
|---|---:|---:|---:|
| Economy | 315 | 27 | 8.57% |
| Express | 243 | 119 | 48.97% |
| Standard | 642 | 132 | 20.56% |

**Finding:** Express shipments have the highest observed delay rate at 48.97%. This indicates that Express deliveries require particular attention during dispatch planning and operational monitoring.

### 5.4 Delay rate by hub tier

| Hub tier | Shipments | Delayed shipments | Delay rate |
|---|---:|---:|---:|
| Tier 1 | 400 | 16 | 4.00% |
| Tier 2 | 400 | 86 | 21.50% |
| Tier 3 | 400 | 176 | 44.00% |

**Finding:** The delay rate increases from Tier 1 to Tier 3. Tier 3 hubs show the highest delay rate, indicating a need for additional operational review and capacity planning.

### 5.5 Delay rate by distance band

| Distance band | Shipments | Delayed shipments | Delay rate |
|---|---:|---:|---:|
| 0–200 km | 274 | 6 | 2.19% |
| 201–400 km | 563 | 77 | 13.68% |
| 401–600 km | 275 | 120 | 43.64% |
| 601+ km | 88 | 75 | 85.23% |

**Finding:** Longer-distance shipments have higher observed delay rates. The 601+ km distance band has the highest delay rate at 85.23%.

### 5.6 Correlation analysis

The three strongest correlations with `delivery_hours`, excluding the target's self-correlation, are:

| Feature | Correlation with delivery_hours |
|---|---:|
| promised_hours | +0.8372 |
| distance_km | +0.4013 |
| n_stops | +0.3247 |

**Interpretation:**

- `promised_hours` has a strong positive relationship with delivery duration.
- `distance_km` has a moderate positive relationship with delivery duration.
- `n_stops` has a positive relationship with delivery duration.

These correlations describe associations in the dataset and do not independently establish causation.

### 5.7 Declared-value transformation

The declared-value distribution is right-skewed, with a mean of approximately INR 1,533.05 and a median of INR 1,121.50.

A natural logarithm transformation was applied to create `log_declared_value`.

| Measure | Original declared value | Log-transformed value |
|---|---:|---:|
| Mean | 1,533.05 | 7.028 |
| Median | 1,121.50 | 7.022 |

The logarithmic transformation reduces the influence of very large declared values and produces a more balanced feature distribution.

## 6. Model Results – Regression

### 6.1 Objective

The regression task predicts delivery duration in hours using information available at dispatch time.

The following regression algorithms were evaluated:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

### 6.2 Validation comparison

| Maximum Depth | Training RMSE | Validation RMSE |
|---|---:|---:|
| 1 | 11.6334 | 12.1324 |
| 2 | 9.3687 | 10.4114 |
| 3 | 7.5295 | 8.7033 |
| 4 | 6.0999 | 7.3896 |
| 6 | 4.7114 | 6.2796 |
| 8 | 3.3360 | 6.8385 |
| 12 | 1.3613 | 7.4173 |
| Unlimited | 0.0000 | 7.7890 |

The training error decreases as tree depth increases. However, validation error begins increasing after depth 6, indicating overfitting. Therefore, depth 6 provides the best validation performance among the tested depths.

**Selected model: Linear Regression**

Linear Regression achieved the lowest validation RMSE among the evaluated models. Its validation R² was also strong.

### 6.3 Baseline comparison

A mean-prediction baseline was used as a reference. It predicts the training-set mean delivery duration for every shipment.

The baseline R² is approximately -0.021. A negative R² indicates that the baseline performs worse than predicting the actual mean of the evaluation targets.

The selected Linear Regression model performs substantially better than this baseline.

### 6.4 Decision Tree depth investigation

Different maximum depths were tested to understand the bias–variance trade-off.

| Maximum depth | Training RMSE | Validation RMSE |
|---|---:|---:|
| 1 | 11.6334 | 12.1324 |
| 6 | 4.7114 | 6.2796 |
| 8 | 3.3360 | 6.8385 |
| 12 | 1.3613 | 7.4173 |
| Unlimited | 0.0000 | 7.7890 |

**Interpretation:**

- Depth 1 shows high bias and underfitting because both training and validation errors are high.
- Increasing the tree depth improves training performance.
- Unlimited depth produces zero training error but a higher validation error, indicating overfitting.
- Depth 6 provides the best validation RMSE among the tested depths.

### 6.5 Gradient Boosting learning-rate investigation

| Learning rate | Validation R² |
|---|---:|
| 0.01 | 0.713 |
| 0.1 | 0.931 |
| 1.0 | 0.844 |

A learning rate of 0.01 produces slower learning within the tested boosting stages. A learning rate of 1.0 makes larger updates and gives weaker validation performance. The learning rate of 0.1 provides the best validation R² score of 0.931 and is selected for this experiment.

The lower learning rate of 0.01 may not learn sufficiently within the tested configuration, while 1.0 gives weaker validation performance.

### 6.6 Cross-validation comparison

| Model | Mean CV R² | Standard deviation |
|---|---:|---:|
| Linear Regression | 0.9261 | 0.0048 |
| Gradient Boosting | 0.9352 | 0.0044 |

Gradient Boosting achieved a slightly higher mean cross-validation R² score. However, Linear Regression achieved a slightly lower validation RMSE (4.8912 compared with 4.9165 for Gradient Boosting). Since validation RMSE is the selection criterion, Linear Regression was retained as the final regression model.

Gradient Boosting achieved a higher mean cross-validation R². However, Linear Regression achieved a slightly lower validation RMSE (4.8912 compared with 4.9165).

Linear Regression was selected based on its validation RMSE. This demonstrates that the model with the best cross-validation score is not necessarily the model with the best validation RMSE.

### 6.7 Final regression test results

| Metric | Test result |
|---|---:|
| MAE | 3.5671 |
| MSE | 20.4624 |
| RMSE | 4.5235 |
| R² | 0.9493 |

**Interpretation:**

- MAE of 3.5671 means the average absolute prediction error is approximately 3.57 hours.
- RMSE of 4.5235 measures the prediction error while giving greater weight to larger errors.
- R² of 0.9493 indicates that the model explains a substantial proportion of the variation in delivery duration in the test set.

The selected Linear Regression model provides a useful delivery-time prediction baseline for the operational pipeline.

## 7. Model Results – Classification

### 7.1 Objective

The classification model predicts whether a shipment is likely to miss its promised delivery time.

The target variable is `is_delayed`:

- `1` indicates delayed.
- `0` indicates not delayed.

The dataset was split into training, validation, and test sets using stratification.

| Dataset | Rows |
|---|---:|
| Training | 840 |
| Validation | 180 |
| Test | 180 |

### 7.2 Baseline comparison

The majority-class baseline predicts every shipment as not delayed.

| Metric | Baseline result |
|---|---:|
| Accuracy | 76.67% |
| Delayed-class precision | 0 |
| Delayed-class recall | 0 |

The validation baseline confusion matrix was:

| | Predicted not delayed | Predicted delayed |
|---|---:|---:|
| Actual not delayed | 138 | 0 |
| Actual delayed | 42 | 0 |

Although the baseline achieves 76.67% accuracy, it misses all 42 delayed validation shipments.

Therefore, accuracy alone is not sufficient for this business problem. Identifying delayed shipments is more important than simply predicting the majority class.

### 7.3 Feature importance and noise investigation

The five noise columns investigated were:

1. `label_print_batch`
2. `scanner_build_no`
3. `tape_roll_no`
4. `sticker_serial_mod`
5. `shelf_slot_no`

The combined Random Forest importance of the five noise columns was approximately 22.44%.

However, all five noise columns had F-score ranks outside the top seven (ranks 8, 12, 14, 16, and 17).
The five noise columns are label_print_batch, scanner_build_no, tape_roll_no, sticker_serial_mod, and shelf_slot_no. Together, they contribute 22.44% of the Random Forest feature importance. However, all five noise features rank outside the final seven selected features. This supports excluding them from the final classification model.

The final seven selected features were:

1. `distance_km`
2. `hub_tier`
3. `hub_daily_capacity`
4. `n_stops`
5. `promised_hours`
6. `rain_flag`
7. `sale_event_flag`

These features were selected based on the feature-selection investigation, with the five noise columns excluded from the final seven-feature configuration.

### 7.4 Feature-count investigation

Different numbers of top-ranked features were evaluated.

| Features (K) | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| 5 | 0.8611 | 0.7073 | 0.6905 | 0.6988 |
| 7 | 0.9222 | 0.8333 | 0.8333 | 0.8333 |
| 10 | 0.9000 | 0.7857 | 0.7857 | 0.7857 |
| 12 | 0.9111 | 0.8250 | 0.7857 | 0.8049 |
| 17 | 0.9056 | 0.8378 | 0.7381 | 0.7848 |

The seven-feature configuration achieved the highest F1-score among the tested feature counts.

This provides a balance between predictive performance and a simpler feature set.

### 7.5 Model and tuning comparison

| Method | Validation F1 |
|---|---:|
| Grid Search | 0.7750 |
| Randomized Search | 0.7848 |
| Default Random Forest using the same seven features | 0.8333 |

The default Random Forest with seven selected features achieved the highest validation F1-score in this comparison.

The tuning searches did not improve on this result. The simpler configuration was retained.

Since the validation set contains only 180 records, these comparisons should be interpreted cautiously. A larger dataset would provide stronger evidence of model performance.

### 7.6 Decision-threshold selection

The decision threshold converts predicted probabilities into delayed or not-delayed classifications.

| Threshold | Validation Precision | Validation Recall | Validation F1 |
|---|---:|---:|---:|
| 0.4 | 0.7400 | 0.8810 | 0.8043 |
| 0.5 | 0.7955 | 0.8333 | 0.8140 |
| 0.6 | 0.8889 | 0.7619 | 0.8205 |

The selected threshold is **0.50**.

A threshold of 0.60 increases precision but reduces recall below 0.80. A threshold of 0.50 maintains validation recall above 0.80 while providing a strong F1-score.

The selected threshold therefore prioritises identifying delayed shipments while balancing false alarms.

The threshold is saved in `models/delay_threshold.csv` for consistent predictions.

### 7.7 Final classification test results

| Metric | Test result |
|---|---:|
| Accuracy | 86.67% |
| Precision | 68.89% |
| Recall | 75.61% |
| F1-score | 72.09% |

Confusion matrix:

| | Predicted not delayed | Predicted delayed |
|---|---:|---:|
| Actual not delayed | 125 | 14 |
| Actual delayed | 10 | 31 |

The final model correctly classified 125 on-time shipments and 31 delayed shipments. It produced 14 false positives and 10 false negatives.

The recall of 75.61% means the model detected approximately 7.6 out of every 10 actually delayed shipments. The precision of 68.89% means approximately 6.9 out of every 10 shipments flagged as delayed were truly delayed. These results help the operations team identify shipments that may need proactive attention.

**Interpretation:**

- The model correctly identified 31 delayed shipments.
- It missed 10 delayed shipments.
- It generated 14 false alarms.
- It correctly classified 125 shipments as not delayed.

The test recall of 75.61% means that the model identified approximately 8 out of every 10 delayed shipments in the test set.

The test recall is lower than the validation recall of 83.33%. This shows that validation performance does not always transfer perfectly to unseen data.

The model should therefore support operational decisions rather than replace human judgement.

## 8. Operations Recommendation

### 8.1 Hub segmentation

The 24 hubs were grouped into three operational segments using clustering.

| Segment | Number of hubs | Average delivery hours | Delay rate | Average distance |
|---|---:|---:|---:|---:|
| Standard | 7 | 47.2396 | 2.57% | 188.93 km |
| Watch | 8 | 56.8137 | 15.50% | 314.18 km |
| Escalate | 9 | 64.6504 | 46.00% | 470.48 km |

### 8.2 Operational interpretation

**Standard segment**

These hubs show lower average delivery duration and a low delay rate. Continue regular monitoring and existing operating procedures.

**Watch segment**

These hubs have a moderate delay rate. Monitor dispatch schedules, shipment volume, route distance, and daily capacity.

**Escalate segment**

These hubs have the highest average delivery duration and delay rate. Prioritise them for operational investigation, capacity planning, dispatch review, and proactive shipment monitoring.

### 8.3 Why three clusters were selected

The separation ratio increased as the number of clusters increased.

| K | Separation ratio | Cluster sizes |
|---|---:|---|
| 2 | 2.8940 | 15–9 |
| 3 | 5.3930 | 9–8–7 |
| 4 | 6.3050 | 8–7–5–4 |
| 5 | 6.8866 | 8–7–4–3–2 |

Although K=4 and K=5 have higher separation ratios, they create clusters with fewer than six hubs.

K=3 was selected because it provides a practical balance between cluster separation and operational usefulness. Each group has enough hubs to support a meaningful operational playbook.

### 8.4 Hub tier mismatches

The following hubs have a mismatch between their supplied tier and their observed segment.

| Hub | City | Supplied tier | Observed segment |
|---|---|---:|---|
| HUB07 | Kolkata | Tier 1 | Watch |
| HUB13 | Nagpur | Tier 2 | Escalate |
| HUB15 | Kochi | Tier 2 | Escalate |
| HUB22 | Dehradun | Tier 3 | Watch |

These hubs should be reviewed using actual delivery performance rather than relying only on existing tier labels.

A mismatch is an investigation signal, not automatic proof that the original tier is incorrect.

### 8.5 New shipment recommendations

The prediction pipeline processed 48 new shipments.

| Recommended action | Number of shipments |
|---|---:|
| No action | 40 |
| Proactive call | 2 |
| Escalate | 6 |
| **Total** | **48** |

All 48 shipment IDs are unique, and the generated action file contains no missing values.

Recommended operational actions:

- Prioritise the six escalated shipments for immediate review.
- Contact the two proactive-call shipments to check delivery progress and possible issues.
- Continue normal monitoring for the remaining 40 shipments.

The classification test recall of 75.61% means approximately approximately 7.6 out of every 10 delayed shipments were identified in the held-out test set. However, the model missed 10 delayed test shipments, so operational monitoring must continue even when a shipment is not flagged.

## 9. Model Packaging and Reproducibility

The final classification model is packaged using `package_model.py`.

The following files are saved:

| File | Purpose |
|---|---|
| `models/delay_model.pkl` | Saved classification model |
| `models/delay_threshold.csv` | Selected decision threshold |
| `models/feature_order.csv` | Required feature order |
| `models/impute_values.csv` | Saved imputation values |

A model reload verification was performed on 180 test rows.

| Verification | Result |
|---|---:|
| Rows checked | 180 |
| Maximum absolute probability difference | 0.0000 |
| Saved classification threshold | 0.50 |

The maximum probability difference of 0.0000 confirms that the reloaded model reproduced the original model probabilities for the verification data.

The saved threshold, feature order, and imputation values help ensure that new shipment data is prepared consistently with the training data.

## 10. Production Plan

The current project is an offline prototype. The following plan describes how it could be developed into a production decision-support system.

### 10.1 Production characteristics

| Characteristic | Project implementation | Production approach |
|---|---|---|
| Accuracy | Regression and classification evaluation metrics | Recalculate MAE, RMSE, precision, recall, and F1 using newly completed shipments |
| Reliability | Missing-value handling and model reload verification | Validate incoming data and monitor prediction failures |
| Reproducibility | Pinned requirements and saved model configuration | Record package versions, model version, and training date |
| Performance | Separate cleaning, modelling, packaging, and prediction scripts | Monitor prediction runtime and batch-processing time |
| Maintainability | Modular Python scripts and CSV outputs | Maintain clear feature definitions and version-controlled code |
| Consistency | Saved feature order, imputation values, and threshold | Apply the same preprocessing and decision rules to new shipments |
| Monitorability | Shipment actions and hub segments | Track delay rates, false alarms, missed delays, and hub-level performance |

### 10.2 Monitoring quantities and alert thresholds

The following are proposed initial production monitoring thresholds. They are recommendations for a future deployment, not measured guarantees from the current synthetic dataset.

| Monitoring quantity | Proposed trigger | Recommended action |
|---|---|---|
| Missing required input values | More than 2% in a daily batch | Investigate data ingestion and source quality |
| Regression MAE | Rolling MAE exceeds 5 hours | Review model accuracy and recent delivery patterns |
| Classification recall | Falls below 70% on newly completed shipments | Investigate missed delays and consider retraining |
| Hub delay rate | Increases by more than 10 percentage points against its recent baseline | Review hub capacity, dispatch, and route conditions |
| Prediction batch failures | Any failed production batch | Investigate immediately before relying on the output |

### 10.3 Retraining schedule

- Monitor input data quality and prediction outcomes daily.
- Review model performance weekly.
- Retrain monthly when enough newly completed shipment records are available.
- Retrain sooner if performance falls below the monitoring thresholds or delivery processes change significantly.
- Evaluate a new model on a separate holdout period before replacing the existing model.
- Keep the previous model available for rollback if the new model performs poorly.

### 10.4 Sale-event monitoring

Sale-event periods should be compared with previous sale-event periods rather than compared only with ordinary weeks.

The monitoring process should compare:

- Shipment volume during sale events.
- Service-type distribution.
- Distance and parcel-weight distributions.
- Hub capacity and number of stops.
- Actual delay rates.
- Predicted delay probabilities.

This comparison helps distinguish normal sale-event workload changes from genuine model performance deterioration.

### 10.5 Data-drift monitoring

The following input features should be monitored for changes:

- `distance_km`
- `parcel_weight_kg`
- `service_type`
- `payment_mode`
- `hub_tier`
- `hub_daily_capacity`
- `rain_flag`
- `sale_event_flag`
- `n_stops`

Changes in feature distributions or in the relationship between input features and actual delivery outcomes should trigger investigation.

If data drift or performance deterioration is detected, the team should review data quality, investigate operational changes, and evaluate retraining.

## 11. Limitations

1. **Synthetic dataset:** The project uses generated shipment data. Its results do not establish actual Delhivery operational performance.
2. **Limited sample size:** The cleaned historical dataset contains 1,200 shipment records and 24 hubs. More real-world observations are required to establish generalisation.
3. **External factors:** Traffic, road closures, address quality, courier availability, and other external conditions are not fully represented.
4. **Model uncertainty:** Predictions are estimates. False positives and false negatives can affect operational decisions.
5. **Hub segmentation:** The segments are based on available synthetic features and should be validated against real operational conditions before use.
6. **Offline prototype:** The current pipeline does not connect to live shipment systems, real-time tracking, or production alert services.
7. **Model drift:** Delivery patterns can change over time, so continuous monitoring and periodic retraining are necessary.

## 12. Conclusion

The Delhivery Last-Mile Delay Intelligence Pipeline demonstrates an end-to-end machine learning workflow for last-mile delivery intelligence.

The project includes data cleaning, exploratory data analysis, delivery-time regression, proactive delay classification, hub segmentation, model packaging, and new-shipment action recommendations.

The regression model achieved a test R² of 0.9493, while the classification model achieved a test recall of 75.61%. Hub segmentation identified Standard, Watch, and Escalate operational groups.

The new-shipment prediction pipeline processed 48 shipments and generated actionable recommendations for proactive calls and escalations.

The project demonstrates how machine learning can support delivery planning and operational decision-making. Real-world deployment would require validation using actual shipment records, integration with operational systems, and continuous monitoring.