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
- [x] Open `notebooks/01_eda.ipynb`
- [x] Answer the three EDA questions in the notebook
- [x] Write 4 to 6 useful charts

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
- [ ] Tune hyperparameters with cross-validated search
- [x] Add feature importance / explainability analysis
- [x] Update README and model_card.md with iteration 2 results
