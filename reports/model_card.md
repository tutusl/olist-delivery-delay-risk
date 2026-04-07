# Model Card: Olist Delivery Delay Risk

## Model intent

- Intended use: rank delivered-order scenarios by risk of arriving after the estimated date.
- Not intended for: customer-level profiling, seller punishment, or operational decisions without human review.

## Target

- `is_late = 1` when `order_delivered_customer_date > order_estimated_delivery_date`
- `is_late = 0` otherwise

## Data source

- Dataset: Brazilian E-Commerce Public Dataset by Olist
- Unit of analysis: order
- Training subset: delivered orders with non-null purchase, estimated delivery, and delivered customer timestamps

## Feature policy

Included feature groups:

- purchase timestamp features;
- customer state;
- order item aggregates;
- payment aggregates;
- product aggregates;
- seller-state aggregates.

Excluded because of leakage or bad timing:

- delivered timestamps;
- order status as a predictive feature;
- review fields;
- any post-delivery information.

## Evaluation

Primary metric:

- average precision

Secondary metrics:

- ROC-AUC
- balanced accuracy
- F2 score - harmonic mean of precision and recall with beta=2, weighting recall 4x more than precision. Chosen because missing a late delivery is costlier than a false alarm in an operational triage setting.
- Precision@500 - fraction of truly late orders among the 500 orders the model scores as highest risk. Reflects a realistic operational scenario where a team can only intervene on a fixed number of orders per day.

## Starter models

- Baseline: `DummyClassifier(strategy="prior")`
- Stronger model: `RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample")`

## First run observations

- Training date: April 7, 2026
- Dataset: Brazilian E-Commerce Public Dataset by Olist (Kaggle, CC-BY-NC-SA-4.0)
- Modeling subset: 96,470 delivered orders with valid purchase, estimated delivery, and actual delivery timestamps
- Observed late-delivery rate: 8.11%
- Train/test split: 80/20 stratified holdout, random_state=42

### Observed metrics

| Metric | DummyClassifier | RandomForestClassifier |
|---|---|---|
| Average Precision | 0.081 | 0.305 |
| ROC-AUC | 0.500 | 0.782 |
| Balanced Accuracy | 0.500 | 0.610 |
| F2 Score | 0.000 | 0.273 |
| Precision@500 | 0.084 | 0.488 |

### Current best model

RandomForestClassifier. It achieves roughly 3.8x the baseline average precision and nearly 49% precision in the top-500 riskiest orders. The F2 score of 0.27 indicates that recall on late deliveries is still limited - the model is better at ranking than at catching all late orders.

## Leakage policy

The following columns are excluded from the feature set because they contain post-outcome or post-purchase information:

- `order_approved_at`
- `order_delivered_carrier_date`
- `order_delivered_customer_date`
- `order_estimated_delivery_date` (used to derive the target, then dropped)
- `order_status`
- `review_score`, `review_comment_title`, `review_comment_message`

A runtime guard (`validate_no_leakage`) raises an error if any of these columns appear in the final feature matrix.

## Known limitations

- The project uses a public dataset and may not reflect current marketplace behavior.
- Delivery risk depends on logistics variables (carrier capacity, weather, route distance) that are not captured in this dataset.
- The target is based on estimated delivery dates, which are themselves part of the business process - a generous estimate reduces the late rate mechanically.
- Some useful operational features may be unavailable at purchase time and are intentionally excluded.
- The model has not been calibrated; predicted probabilities should not be interpreted as true delay likelihoods without further work.
- No hyperparameter tuning or cross-validation has been performed yet - these are first-pass results.

## Fairness and risk notes

- Geographic patterns may reflect marketplace or logistics inequalities rather than seller or customer quality.
- The model should be used for analysis and prioritization, not for automated punitive actions.
- State-level features could encode socioeconomic proxies; downstream use should consider fairness audits.

## Reproducibility

- Environment: `requirements.txt`
- Main training entry point: `python -m src.models.train`
- Notebook entry point: `notebooks/01_eda.ipynb`
- Results file: `reports/training_summary.csv`
