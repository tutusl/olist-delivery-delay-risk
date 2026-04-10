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

## Models

- Baseline: `DummyClassifier(strategy="prior")`
- Random Forest: `RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample")`
- LightGBM: `LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31, scale_pos_weight=10)`

## Iteration 1 — random stratified split

- Training date: April 7, 2026
- Dataset: Brazilian E-Commerce Public Dataset by Olist (Kaggle, CC-BY-NC-SA-4.0)
- Modeling subset: 96,470 delivered orders
- Observed late-delivery rate: 8.11%
- Train/test split: 80/20 stratified holdout, random_state=42

| Metric | DummyClassifier | RandomForestClassifier |
|---|---|---|
| Average Precision | 0.081 | 0.305 |
| ROC-AUC | 0.500 | 0.782 |
| Balanced Accuracy | 0.500 | 0.610 |
| F2 Score | 0.000 | 0.273 |
| Precision@500 | 0.084 | 0.488 |

These results were later found to be overly optimistic due to temporal leakage in the random split.

## Iteration 2 — chronological split

- Training date: April 10, 2026
- Split method: chronological (train on earlier 80%, test on later 20%)
- Late-delivery rate in test period: ~5.3%

| Metric | DummyClassifier | RandomForestClassifier | LGBMClassifier |
|---|---|---|---|
| Average Precision | 0.053 | 0.086 | 0.079 |
| ROC-AUC | 0.500 | 0.669 | 0.630 |
| Balanced Accuracy | 0.500 | 0.505 | 0.527 |
| F2 Score | 0.000 | 0.019 | 0.120 |
| Precision@500 | 0.050 | 0.106 | 0.102 |

### Current best model

RandomForestClassifier by average precision (0.086). The improvement over baseline is modest (~1.6x), reflecting the difficulty of predicting delays out-of-time with only purchase-time features. LightGBM achieves a higher F2 score (0.120 vs 0.019), suggesting it finds a better precision-recall trade-off at its default threshold.

### Feature importance (permutation, Random Forest)

| Feature | Importance |
|---|---|
| estimated_delivery_days | 0.036 |
| primary_seller_state | 0.007 |
| avg_product_volume_cm3 | 0.002 |
| payment_type_mode | 0.002 |
| customer_state | 0.001 |

The estimated delivery window is the dominant predictor — orders with tighter promised delivery windows are far more likely to arrive late.

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
- No hyperparameter tuning or cross-validation has been performed yet.
- The chronological split shows that late-delivery patterns shift over time, limiting out-of-time predictive power with purchase-time features alone.

## Fairness and risk notes

- Geographic patterns may reflect marketplace or logistics inequalities rather than seller or customer quality.
- The model should be used for analysis and prioritization, not for automated punitive actions.
- State-level features could encode socioeconomic proxies; downstream use should consider fairness audits.

## Reproducibility

- Environment: `requirements.txt`
- Main training entry point: `python -m src.models.train`
- Notebook entry point: `notebooks/01_eda.ipynb`
- Results file: `reports/training_summary.csv`
- Feature importance: `reports/feature_importance.csv`
