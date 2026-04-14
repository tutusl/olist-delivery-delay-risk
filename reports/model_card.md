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
- HistGradientBoosting: `HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_leaf_nodes=31, class_weight="balanced")`

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

## Iteration 2 — chronological split, tuned

- Training date: April 10, 2026
- Split method: chronological (train on earlier 80%, test on later 20%)
- Late-delivery rate in test period: ~5.3%
- Tuning: `RandomizedSearchCV` with `TimeSeriesSplit` (4 folds, 12 iterations per model)

### Holdout metrics (tuned models)

| Metric | DummyClassifier | RandomForest (tuned) | HistGradientBoosting (tuned) |
|---|---|---|---|
| Average Precision | 0.053 | 0.067 | 0.082 |
| ROC-AUC | 0.500 | 0.544 | 0.663 |
| Balanced Accuracy | 0.500 | 0.518 | 0.545 |
| F2 Score | 0.000 | 0.121 | 0.165 |
| Precision@500 | 0.050 | 0.094 | 0.102 |

### Current best model

HistGradientBoostingClassifier (tuned). It leads on every holdout metric. The improvement over baseline is modest (~1.6x AP) but consistent, reflecting the difficulty of predicting delays out-of-time with only purchase-time features.

### Walk-forward tuning

| Model | CV Avg Precision | Train AP | Selected params |
|---|---|---|---|
| Random Forest | 0.185 | 0.553 | max_depth=12, max_features=sqrt, min_samples_leaf=10, n_estimators=300 |
| HistGradientBoosting | 0.168 | 0.246 | l2_reg=1.0, lr=0.05, max_iter=600, max_leaf_nodes=15, min_samples_leaf=50 |

Random Forest scored higher in CV but lower on the final holdout, with a large train-CV gap indicating overfitting. The HistGradientBoosting configuration generalizes better.

### Calibration

| Variant | Avg Precision | ROC-AUC | Precision@500 | Brier Score |
|---|---|---|---|---|
| Uncalibrated | 0.082 | 0.663 | 0.102 | 0.139 |
| Calibrated (sigmoid) | 0.172 | 0.796 | 0.236 | 0.049 |

Sigmoid calibration doubles average precision and cuts the Brier score by 3x. Precision@500 improves from 10% to 24%. The calibrated probabilities are all below 0.5, so F2 at the default threshold drops to 0.0 — acceptable for a ranking use case where the model surfaces high-risk orders rather than hard-classifying.

### Feature importance (permutation, HistGradientBoosting)

| Feature | Importance |
|---|---|
| estimated_delivery_days | 0.035 |
| primary_seller_state | 0.004 |
| payment_type_mode | 0.002 |
| avg_product_volume_cm3 | 0.001 |
| purchase_day_of_week | 0.001 |
| freight_to_price_ratio | 0.001 |

The estimated delivery window is the dominant predictor — orders with tighter promised delivery windows are far more likely to arrive late.

### Error analysis

Performance varies substantially across test-set slices:

| Slice | N | Late rate | Avg Precision |
|---|---|---|---|
| delivery_window=0-10d | 2,901 | 19.9% | 0.195 |
| customer_state=SP | 8,921 | 7.4% | 0.135 |
| payment_type=boleto | 3,526 | 7.0% | 0.107 |
| payment_type=credit_card | 14,753 | 4.8% | 0.076 |
| delivery_window=10-20d | 6,407 | 4.2% | 0.067 |
| delivery_window=30d+ | 4,043 | 0.8% | 0.039 |

The model is most useful on short-window orders (0-10 days), where 1 in 5 deliveries is late and average precision reaches 0.20. This is the segment where operational intervention would have the most impact. Longer delivery windows have near-zero late rates, leaving little room for the model to add value.

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
- Uncalibrated probabilities should not be interpreted as true delay likelihoods; the sigmoid-calibrated variant (Brier 0.049) is better suited for probability estimates.
- The chronological split shows that late-delivery patterns shift over time, limiting out-of-time predictive power with purchase-time features alone.

## Fairness and risk notes

- Geographic patterns may reflect marketplace or logistics inequalities rather than seller or customer quality.
- The model should be used for analysis and prioritization, not for automated punitive actions.
- State-level features could encode socioeconomic proxies; downstream use should consider fairness audits.

## Reproducibility

- Environment: `requirements.txt`
- Main training entry point: `python -m src.models.train`
- Notebook entry points: `notebooks/01_data_understanding.ipynb` (grain, joins, missingness, funnel), `notebooks/02_eda.ipynb` (modeling-oriented EDA with temporal drift section), `notebooks/03_route_eda.ipynb` (seller-customer corridor analysis), and `notebooks/04_cohort_eda.ipynb` (seller/product cohorts and missingness-as-signal check)
- Results file: `reports/training_summary.csv`
- Feature importance: `reports/feature_importance.csv`
- Error analysis: `reports/error_analysis.csv`
- Tuning summary: `reports/tuning_summary.csv`
- Calibration comparison: `reports/calibration_summary.csv`
