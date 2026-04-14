# Task Checklist

## Repo foundation

- [x] Create the repository structure
- [x] Add starter documentation
- [x] Add a notebook template
- [x] Add starter training code
- [x] Add guardrail tests

## Data setup

- [x] Download the Olist dataset from Kaggle
- [x] Place the required CSV files inside `data/raw/`
- [x] Confirm the expected filenames match the repository loader

## First analysis pass

- [x] Run `pytest`
- [x] Open `notebooks/02_eda.ipynb`
- [x] Answer the three EDA questions in the notebook
- [x] Write 4 to 6 useful charts

## Data understanding pass

- [x] Scaffold `notebooks/01_data_understanding.ipynb`
- [x] Document table grain and declared primary keys
- [x] Validate join cardinality against expected 1:1 / 1:N shapes
- [x] Report missingness for all tables
- [x] Duplicate checks beyond declared keys
- [x] Order-status funnel to the delivered-only subset

## Temporal EDA pass

- [x] Extend `notebooks/02_eda.ipynb` with a temporal-patterns section
- [x] Late rate by purchase month, ISO week, weekday, and hour
- [x] Estimated delivery window by month
- [x] Written explanation for the random-vs-chronological gap

## Route-level EDA pass

- [x] Scaffold `notebooks/03_route_eda.ipynb`
- [x] Seller-state x customer-state late-rate heatmap with volume masking
- [x] Riskiest and safest corridors above a minimum-volume threshold
- [x] Recorded the geolocation-distance follow-up as a design note
- [ ] Implement haversine distance feature in `src/features/engineering.py`

## Cohort EDA pass

- [x] Scaffold `notebooks/04_cohort_eda.ipynb`
- [x] Late rate by seller volume bucket (with primary-seller mode join)
- [x] Late rate by item-count and product-weight quintile buckets
- [x] Late rate by product category with minimum-volume threshold
- [x] Missingness-as-signal check on `missing_product_metadata_share`
- [ ] Promote confirmed cohort signals to `src/features/engineering.py` (target-encoded category, log seller volume)

## First modeling pass

- [x] Run `python -m src.models.train`
- [x] Compare the baseline with the tree-based model
- [x] Record metrics in `reports/model_card.md`
- [x] Update the README with real findings

## Publish

- [x] Initialize git if needed
- [x] Commit the scaffold
- [x] Push to the personal GitHub repository
- [ ] Pin the repository on the GitHub profile

## Second modeling iteration

- [x] Switch to chronological train/test split
- [x] Add gradient boosting model (HistGradientBoostingClassifier)
- [x] Tune hyperparameters with cross-validated search
- [x] Add feature importance / explainability analysis
- [x] Add calibration comparison (sigmoid calibration with Brier score)
- [x] Add slice-level error analysis (delivery window, state, payment type)
- [x] Update README and model_card.md with iteration 2 results
