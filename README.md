# Delhivery Last-Mile Delay Intelligence Pipeline

**Program:** Artificial Intelligence and Machine Learning Certification Program – E&ICT Academy, IIT Roorkee  
**Project:** Delhivery Last-Mile Delay Intelligence Pipeline  
**Repository branch:** `main`

## 1. Project Overview

This project develops an end-to-end machine learning pipeline to predict last-mile delivery duration, identify shipments at risk of missing their promised delivery time, and group delivery hubs into operational performance segments.

The project uses a reproducible synthetic Delhivery dataset containing 24 hubs, 1,224 historical shipment records before cleaning, and 48 new shipments for prediction.

The pipeline includes:

- Data cleaning and exploratory data analysis.
- Feature engineering and data quality checks.
- Delivery-time prediction using regression.
- Proactive delay-risk classification.
- Hub segmentation using clustering.
- Model packaging and reload verification.
- New-shipment predictions and operational actions.

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
│   ├── Part A data-quality and EDA outputs
│   ├── Part B regression outputs
│   ├── Part C classification outputs
│   ├── Part D hub-segmentation outputs
│   └── new-shipment prediction outputs
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

The promised delivery time depends on the service type:

| Service type | Promise |
|---|---:|
| Express | 36 hours |
| Standard | 60 hours |
| Economy | 96 hours |

A shipment is considered delayed when:

`delivery_hours > promised_hours`

The project uses dispatch-time information to predict delivery duration and identify possible promise misses before delivery is completed.

## 4. How to Run

### Requirements

- Python 3.x
- pandas
- NumPy
- scikit-learn
- joblib

The exact package versions are pinned in `requirements.txt`.

### Installation

Clone the repository:

```bash
git clone https://github.com/Priya2508G/Delhivery-Last-Mile-Delay-Intelligence.git
cd Delhivery-Last-Mile-Delay-Intelligence
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the complete pipeline in this order:

```bash
python generate_data.py
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
python package_model.py
python predict_delay.py
```

`generate_data.py` creates the synthetic hub and shipment data. The Part A script cleans and prepares the data. Parts B, C, and D perform regression, classification, and hub segmentation. `package_model.py` saves the final model and its supporting configuration files. `predict_delay.py` uses the packaged model to generate actions for new shipments.

The committed model pickle is intended to be used with the pinned scikit-learn version. If package versions differ, rerun `package_model.py` before making predictions.

## 5. Data Findings

### Dataset size and cleaning

| Metric | Result |
|---|---:|
| Hubs | 24 |
| Raw historical shipment rows | 1,224 |
| Duplicate rows removed | 24 |
| Clean historical rows | 1,200 |
| New shipments | 48 |
| Delayed historical shipments | 278 |
| Historical delay rate | 23.17% |

The cleaning process removed duplicate records, standardised service names, handled missing payment values, converted declared values into INR, parsed dispatch timestamps, and treated invalid scanner weight values above 100 kg as sentinel values.

Missing numeric values were filled using median imputation. The resulting modelling table contains 1,200 rows and 28 columns.

### Delay rate by service type

| Service type | Shipments | Delayed | Delay rate |
|---|---:|---:|---:|
| Economy | 315 | 27 | 8.57% |
| Express | 243 | 119 | 48.97% |
| Standard | 642 | 132 | 20.56% |

Express shipments have the highest observed delay rate.

### Delay rate by hub tier

| Hub tier | Shipments | Delayed | Delay rate |
|---|---:|---:|---:|
| Tier 1 | 400 | 16 | 4.00% |
| Tier 2 | 400 | 86 | 21.50% |
| Tier 3 | 400 | 176 | 44.00% |

The delay rate increases from Tier 1 to Tier 3. This indicates that hub capacity and operating conditions should be considered when planning dispatches.

### Delay rate by distance band

| Distance band | Shipments | Delayed | Delay rate |
|---|---:|---:|---:|
| 0–200 km | 274 | 6 | 2.19% |
| 201–400 km | 563 | 77 | 13.68% |
| 401–600 km | 275 | 120 | 43.64% |
| 601+ km | 88 | 75 | 85.23% |

Longer-distance shipments have substantially higher delay rates. The 601+ km group has the highest observed delay rate.

### Correlation findings

The three strongest correlations with `delivery_hours`, excluding the target's self-correlation, are:

| Feature | Correlation |
|---|---:|
| promised_hours | +0.8372 |
| distance_km | +0.4013 |
| n_stops | +0.3247 |

These are associations, not proof of causation. The correlation between promised hours and delivery hours is particularly strong, while distance and number of stops also show positive relationships.

### Declared-value transformation

The declared-value distribution is right-skewed. The original declared-value mean is approximately INR 1,533.05 and its median is INR 1,121.50.

A natural logarithm transformation was used to create `log_declared_value`. After transformation, the mean is approximately 7.028 and the median is approximately 7.022.

The transformed values are more balanced, reducing the effect of very large declared values on modelling.

## 6. Model Results – Regression

### Objective

Predict delivery duration in hours using information available at dispatch time.

The models evaluated include:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

### Validation comparison

| Model | Validation RMSE | Validation R² |
|---|---:|---:|
| Linear Regression | 4.8912 | 0.9313 |
| Decision Tree | 6.2796 | 0.8867 |
| Random Forest | 5.4680 | 0.9141 |
| Gradient Boosting | 4.9165 | 0.9306 |

Linear Regression achieved the lowest validation RMSE among the evaluated models and was selected.

### Hyperparameter observations

Decision Tree depth was tested to understand the bias–variance trade-off. A very shallow tree has high bias and may underfit. An unrestricted tree can fit training data extremely closely but may overfit and perform less reliably on unseen data. Depth 6 provided the selected depth in the tuning investigation.

Gradient Boosting learning-rate experiments showed:

| Learning rate | Validation R² |
|---|---:|
| 0.01 | 0.713 |
| 0.1 | 0.931 |
| 1.0 | 0.844 |

A learning rate of 0.1 performed best among these tested values.

Cross-validation was also used to examine model stability. The final model choice considered validation performance and the cross-validation comparison rather than relying on training performance alone.

### Final test results

| Metric | Test result |
|---|---:|
| MAE | 3.5671 |
| MSE | 20.4624 |
| RMSE | 4.5235 |
| R² | 0.9493 |

The test R² indicates that the selected model explains a substantial proportion of the variation in delivery duration on the held-out test data. The test MAE indicates an average absolute prediction error of approximately 3.57 hours.

## 7. Model Results – Classification

### Objective

Predict whether a shipment is likely to miss its promised delivery time.

The classification task uses `is_delayed` as the target. The data was split into training, validation, and test sets using stratification.

### Baseline comparison

The majority-class baseline predicts every shipment as not delayed. It achieves approximately 76.67% accuracy because most records are not delayed.

However, this baseline identifies none of the delayed shipments. Its delayed-class recall is zero, making it unsuitable for proactive operations.

### Feature importance and noise investigation

The five noise columns investigated were:

- `label_print_batch`
- `scanner_build_no`
- `tape_roll_no`
- `sticker_serial_mod`
- `shelf_slot_no`

These noise features were excluded from the final seven-feature selection. The final selected features were:

1. `distance_km`
2. `hub_tier`
3. `hub_daily_capacity`
4. `n_stops`
5. `promised_hours`
6. `rain_flag`
7. `sale_event_flag`

The noise-feature importance was also examined to check whether the model was relying heavily on irrelevant operational identifiers.

### Feature-count and tuning investigation

Feature selection was investigated using different feature counts, including 5, 7, 10, 12, and 17 features.

The seven-feature configuration was selected to balance predictive performance and a simpler, more maintainable model.

The tuned alternatives were compared with the default Random Forest configuration. The final selection retained the seven meaningful features and used the validation results to choose the model configuration and decision threshold.

### Decision threshold

The final classification threshold is 0.50.

A threshold determines how predicted probabilities are converted into delayed/not-delayed decisions. Lowering it generally identifies more potential delays but can increase false alarms. Raising it can reduce false alarms but may miss more delayed shipments.

The threshold and feature order are saved with the packaged model so that prediction behaviour is reproducible.

### Final test results

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

The model identified 31 of the 41 delayed shipments in the test set and missed 10. It also generated 14 false alarms.

The test recall is lower than the validation recall observed during model selection. This is an important limitation: validation performance should not be assumed to transfer perfectly to unseen shipments. The model should be monitored and recalibrated as additional real delivery data becomes available.

## 8. Operations Recommendation

### Hub segmentation

The 24 hubs were grouped into three operational segments using clustering.

| Segment | Hubs | Average delivery hours | Delay rate | Average distance |
|---|---:|---:|---:|---:|
| Standard | 7 | 47.2396 | 2.57% | 188.93 km |
| Watch | 8 | 56.8137 | 15.50% | 314.18 km |
| Escalate | 9 | 64.6504 | 46.00% | 470.48 km |

The Escalate segment has the highest average delivery time and delay rate. These hubs should receive priority for operational review, capacity planning, and dispatch monitoring.

### Why three clusters were selected

The separation ratio increased as the number of clusters increased:

| K | Separation ratio | Cluster sizes |
|---|---:|---|
| 2 | 2.8940 | 15–9 |
| 3 | 5.3930 | 9–8–7 |
| 4 | 6.3050 | 8–7–5–4 |
| 5 | 6.8866 | 8–7–4–3–2 |

Although K=4 and K=5 have higher separation ratios, they create clusters with fewer than six hubs. Three clusters provide a more practical balance between separation and operational usefulness.

### Hub tier mismatches

The following hubs have a mismatch between their supplied tier and their observed segment:

| Hub | City | Supplied tier | Observed segment |
|---|---|---:|---|
| HUB07 | Kolkata | Tier 1 | Watch |
| HUB13 | Nagpur | Tier 2 | Escalate |
| HUB15 | Kochi | Tier 2 | Escalate |
| HUB22 | Dehradun | Tier 3 | Watch |

These hubs should be reviewed using actual delivery performance rather than relying only on their existing tier labels. The mismatch does not automatically mean that the supplied tier is incorrect; it identifies hubs that deserve investigation.

### New shipment actions

The new-shipment prediction pipeline processed 48 shipments.

| Action | Shipments |
|---|---:|
| No action | 40 |
| Proactive call | 2 |
| Escalate | 6 |
| Total | 48 |

All 48 shipment IDs are unique, and the generated action file contains no missing values.

The recommended operational approach is to prioritise the six escalated shipments, contact the two proactive-call shipments, and continue normal monitoring for the remaining 40 shipments.

The classification model's test recall of 75.61% means it identified about 8 of every 10 delayed test shipments. Since it also missed 10 delayed shipments in the test set, predictions should support—not replace—operational judgement.

## 9. Model Packaging and Reproducibility

The final model is packaged using `package_model.py`.

The following supporting files are saved:

- `models/delay_model.pkl`
- `models/delay_threshold.csv`
- `models/feature_order.csv`
- `models/impute_values.csv`

A reload verification was performed on 180 test rows. The maximum absolute probability difference between the original and reloaded model was 0.0000.

The saved threshold is 0.50. The feature-order and imputation files help ensure that new shipment data is prepared consistently with the training data.

## 10. Production Plan

The current project is an offline prototype. The following plan describes how it could be developed into a production decision-support system.

| Production characteristic | Project implementation | Monitoring or control |
|---|---|---|
| Accuracy | Regression and classification test metrics | Recalculate MAE, RMSE, precision, recall, and F1 using newly completed shipments |
| Reliability | Missing-value handling and model reload check | Alert if required input fields are missing or predictions fail |
| Reproducibility | Pinned `requirements.txt`, saved model and feature order | Record package versions, model version, and training date |
| Performance | Separate cleaning, modelling, packaging, and prediction scripts | Track prediction runtime and processing failures |
| Maintainability | Modular Python scripts and CSV outputs | Review feature definitions and code changes before release |
| Monitorability | Prediction actions and hub segments | Track delay rates, false alarms, missed delays, and hub-level changes |

### Monitoring thresholds

The following are proposed initial operational alert thresholds. They are monitoring rules for a future deployment, not measured guarantees from the current synthetic dataset.

1. **Data completeness:** raise an alert if more than 2% of required shipment input values are missing in a daily batch.
2. **Regression error:** review the regression model if rolling MAE exceeds 5 hours.
3. **Delay-risk performance:** review the classifier if measured delayed-shipment recall falls below 70% on newly completed shipments.
4. **Hub performance:** investigate a hub if its rolling delay rate increases by more than 10 percentage points compared with its own recent baseline.
5. **Prediction failures:** investigate immediately if any production prediction batch fails or produces duplicate shipment IDs.

### Retraining schedule

- Monitor data quality and prediction outcomes daily.
- Review model performance weekly.
- Retrain monthly when enough newly completed shipment records are available.
- Retrain sooner if performance drops below the defined monitoring thresholds or if the delivery process changes significantly.
- Validate a new model on a separate holdout period before replacing the deployed model.

### Sale-event and data-drift monitoring

The synthetic data includes a sale-event indicator. Sale-event periods should be compared with previous sale-event periods, rather than compared only with ordinary weeks.

Monitor changes in:

- Distance and parcel-weight distributions.
- Service-type and payment-mode proportions.
- Hub-tier and hub-capacity distributions.
- Rain and sale-event frequencies.
- Predicted delay probabilities and actual delay rates.

A significant change in these input distributions or in the relationship between inputs and actual delivery outcomes should trigger investigation and possible retraining.

## 11. Limitations

1. **Synthetic data:** the project uses generated shipment records. Its results do not establish actual Delhivery operational performance.
2. **Limited sample size:** the historical dataset contains 1,200 cleaned records and only 24 hubs. More real-world observations are required to establish generalisation.
3. **Unobserved external factors:** traffic, road closures, address quality, courier availability, and other external conditions are not fully represented.
4. **Model uncertainty:** predictions are estimates. False positives and false negatives can affect operational decisions.
5. **Cluster interpretation:** hub segments are based on the available synthetic features and should be reviewed against real operational context before use.
6. **Offline prototype:** the current pipeline does not connect to live shipment systems, real-time tracking, or production alert services.

## 12. Conclusion

The project demonstrates a complete offline machine learning workflow for last-mile delivery intelligence.

It cleans and prepares shipment data, predicts delivery duration, classifies potential delays, segments hubs into operational groups, packages a reproducible model, and produces action recommendations for new shipments.

The results demonstrate how data-driven insights can help prioritise high-risk shipments and hubs. Real-world deployment would require validation on actual delivery records, integration with operational systems, and continuous model monitoring.