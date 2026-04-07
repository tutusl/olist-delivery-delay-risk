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
pytest
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
5. Trains:
   - a `DummyClassifier` baseline;
   - a `RandomForestClassifier` as the first stronger model.
6. Saves a summary report to `reports/training_summary.csv`.

## Initial EDA questions

The notebook is structured around three questions:

1. How common are late deliveries in the delivered-order subset?
2. Which purchase-time patterns seem associated with late deliveries?
3. Which customer, payment, and seller/product signals look promising before modeling?

## Current status

This repository already includes:

- project structure;
- starter pipeline code;
- test coverage for schema, target creation, missing values, and leakage guardrails;
- a notebook template for the first round of EDA;
- a model card template.

What is still expected from the first project iteration:

- load the real Olist files into `data/raw/`;
- run the notebook and training script;
- write the first concrete findings in this README and in the model card;
- optionally pin this repo on your GitHub profile after the first full run.

## Suggested next steps

After this first repo is working end-to-end, the next portfolio options are:

- `olist-funnel-and-seller-conversion` for Product/Analytics;
- `olist-review-score-prediction` for a second modeling project.
