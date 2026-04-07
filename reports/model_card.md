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

## Starter models

- Baseline: `DummyClassifier`
- Stronger model: `RandomForestClassifier`

## Known limitations

- The project uses a public dataset and may not reflect current marketplace behavior.
- Delivery risk depends on logistics variables that may not be fully captured here.
- The target is based on estimated delivery dates, which are themselves part of the business process.
- Some useful operational features may be unavailable at purchase time and are intentionally excluded.

## Fairness and risk notes

- Geographic patterns may reflect marketplace or logistics inequalities rather than seller or customer quality.
- The model should be used for analysis and prioritization, not for automated punitive actions.

## Reproducibility

- Environment: `requirements.txt`
- Main training entry point: `python -m src.models.train`
- Notebook entry point: `notebooks/01_eda.ipynb`

## To fill after the first run

- final training date;
- dataset version reference;
- observed class balance;
- best model and metrics;
- key failure modes found during error analysis.
