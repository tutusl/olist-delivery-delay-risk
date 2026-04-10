# Olist Delivery Delay Risk

Predict late deliveries in the Olist marketplace and explain which purchase-time signals are associated with higher delivery risk.

## Why this project exists

This repository is a first portfolio-ready data science project. It focuses on a realistic e-commerce problem, keeps the scope intentionally small, and emphasizes the skills most data science interviews care about:

- data cleaning and table joins;
- exploratory analysis with business questions;
- baseline versus stronger model comparison;
- leakage prevention;
- reproducibility and documentation.

## Business question

Which factors are associated with late deliveries, and can we flag higher-risk orders using only information that is available around purchase time?

## Dataset

Source: [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

This project uses the following CSV files from the dataset:

- `olist_customers_dataset.csv`
- `olist_orders_dataset.csv`
- `olist_order_items_dataset.csv`
- `olist_order_payments_dataset.csv`
- `olist_products_dataset.csv`
- `olist_sellers_dataset.csv`

Place those files inside `data/raw/`.

The raw dataset is not committed to GitHub on purpose. The repository only stores code, documentation, and lightweight artifacts.

## Target definition

The target is `is_late`:

- `1` if `order_delivered_customer_date > order_estimated_delivery_date`
- `0` otherwise

Only delivered orders with the required timestamps are used for modeling.

## Primary metric

The primary metric is **average precision**, because late deliveries are not evenly distributed and this project is framed as a ranking problem: we want to surface higher-risk orders early.

Secondary metrics:

- ROC-AUC
- balanced accuracy
- F2 score - weighs recall 4x more than precision, reflecting that missing a late delivery is costlier than a false alarm
- Precision@500 - precision among the top 500 highest-risk predictions, simulating a fixed-capacity intervention scenario

## Leakage policy

The model must not use post-outcome information such as:

- `order_delivered_customer_date`
- `order_delivered_carrier_date`
- `order_status`
- review-related information

The feature set is intentionally limited to purchase-time and checkout-time signals, plus order-level aggregates derived from products, sellers, and payments.

## Repository structure

```text
olist-delivery-delay-risk/
|-- README.md
|-- PLAN.md
|-- TASKS.md
|-- requirements.txt
|-- .gitignore
|-- notebooks/
|   `-- 01_eda.ipynb
|-- src/
|   |-- data/
|   |-- features/
|   `-- models/
|-- reports/
|   `-- model_card.md
|-- tests/
`-- data/
```

## Setup

### 1. Create a virtual environment

PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If `python` is not available in your PATH, call your local Python executable directly.

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Download and place the data

Download the Olist dataset from Kaggle and move the required CSV files to:

```text
data/raw/
```

### 4. Run tests

```powershell
python -m pytest
```

### 5. Open the notebook

```powershell
jupyter notebook notebooks/01_eda.ipynb
```

### 6. Train the starter models

```powershell
python -m src.models.train
```

## What the starter pipeline does

1. Loads the required Olist tables.
2. Aggregates order item, payment, product, and seller signals at the order level.
3. Builds a modeling frame for delivered orders only.
4. Creates purchase-time features.
5. Splits the data chronologically (earlier orders train, later orders test).
6. Trains:
   - a `DummyClassifier` baseline;
   - a `RandomForestClassifier`;
   - a `HistGradientBoostingClassifier`.
7. Computes permutation importance for the best model.
8. Runs error analysis across delivery-window, state, and payment-type slices.
9. Saves reports to `reports/training_summary.csv`, `reports/feature_importance.csv`, and `reports/error_analysis.csv`.

## Initial EDA questions

The notebook is structured around three questions:

1. How common are late deliveries in the delivered-order subset?
2. Which purchase-time patterns seem associated with late deliveries?
3. Which customer, payment, and seller/product signals look promising before modeling?

## First iteration results (random split)

The first end-to-end run used 96,470 delivered orders with a random stratified 80/20 split. The late-delivery base rate is **8.11%**.

| Metric | DummyClassifier | RandomForestClassifier |
|---|---|---|
| Average Precision | 0.081 | 0.305 |
| ROC-AUC | 0.500 | 0.782 |
| Balanced Accuracy | 0.500 | 0.610 |
| F2 Score | 0.000 | 0.273 |
| Precision@500 | 0.084 | 0.488 |

The Random Forest picks up real signal — average precision is roughly 3.8x the baseline — but these numbers turned out to be overly optimistic (see iteration 2).

## Second iteration results (chronological split)

Iteration 2 switched to a **chronological train/test split** (train on earlier 80% of orders, test on later 20%), added `HistGradientBoostingClassifier`, and computed permutation importance plus error analysis. The late-delivery base rate in the test period is **5.3%**.

| Metric | DummyClassifier | RandomForestClassifier | HistGradientBoosting |
|---|---|---|---|
| Average Precision | 0.053 | 0.086 | 0.083 |
| ROC-AUC | 0.500 | 0.669 | 0.683 |
| Balanced Accuracy | 0.500 | 0.505 | 0.522 |
| F2 Score | 0.000 | 0.019 | 0.115 |
| Precision@500 | 0.050 | 0.106 | 0.076 |

Random Forest leads on average precision and Precision@500 (better ranking). HistGradientBoosting has a higher ROC-AUC and F2 score, meaning it catches more late orders at its default threshold. Neither model dominates across all metrics — the choice depends on whether the use case values ranking or recall more.

### Why the drop from iteration 1?

The random split leaked temporal patterns: orders from the same time period appeared in both train and test, inflating metrics. The chronological split is a more honest evaluation — it simulates deploying the model today to predict tomorrow's delays. The large drop shows that late-delivery patterns shift over time, making this a harder problem than the first iteration suggested.

### Feature importance (permutation, Random Forest)

| Feature | Importance |
|---|---|
| estimated_delivery_days | 0.036 |
| primary_seller_state | 0.007 |
| avg_product_volume_cm3 | 0.002 |
| payment_type_mode | 0.002 |
| customer_state | 0.001 |

The estimated delivery window dominates — tighter promised windows carry much higher risk of arriving late. Geographic features (seller state, customer state) are the next most important signals.

### Error analysis

The model performs unevenly across slices of the test set:

| Slice | N | Late rate | Avg Precision |
|---|---|---|---|
| delivery_window=0-10d | 2,901 | 19.9% | 0.221 |
| customer_state=SP | 8,921 | 7.4% | 0.157 |
| payment_type=boleto | 3,526 | 7.0% | 0.110 |
| payment_type=credit_card | 14,753 | 4.8% | 0.080 |
| delivery_window=10-20d | 6,407 | 4.2% | 0.073 |
| delivery_window=30d+ | 4,043 | 0.8% | 0.049 |

Orders with short delivery windows (0-10 days) have the highest late rate (20%) and the model's best average precision (0.22) — exactly where intervention would matter most. Longer windows have such low late rates that there is little signal to exploit. Sao Paulo (SP) is the strongest state-level slice, likely because it is both the largest market and a logistics hub with more variance in delivery performance.

### Key EDA findings

1. Late-delivery rates vary significantly by customer state, suggesting geographic and logistics patterns matter.
2. Orders with shorter estimated delivery windows are more likely to arrive late, meaning tighter promises carry more risk.
3. Payment type shows some association with delay rates, possibly reflecting different buyer profiles or order types.

## Current status

This repository has completed two modeling iterations:

- project structure and starter pipeline code;
- test coverage for schema, target creation, missing values, and leakage guardrails;
- first EDA notebook with five plots covering target balance, state-level rates, delivery windows, payment types, and price distributions;
- first training run with baseline and Random Forest (random split);
- second training run with chronological split, HistGradientBoosting, permutation importance, and error analysis.

### What the iterations showed

The switch from random to chronological splitting was the single biggest methodological improvement. It revealed that the first iteration's 0.305 average precision was unreliable and that the real out-of-time signal is modest. This is a common and important lesson: always evaluate time-dependent problems with time-aware splits.

The error analysis shows the model adds the most value on short-window orders — the highest-risk segment — which is a practical positive even though overall metrics are modest.

## Suggested next steps

- Hyperparameter tuning with cross-validated search on chronological folds.
- Calibration: predicted probabilities are not currently reliable as true delay likelihoods.
- Additional portfolio projects: `olist-funnel-and-seller-conversion` (Product/Analytics) or `olist-review-score-prediction` (second modeling project).
