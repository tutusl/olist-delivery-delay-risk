# Task Checklist

## Repo foundation

- [x] Create the repository structure
- [x] Add starter documentation
- [x] Add a notebook template
- [x] Add starter training code
- [x] Add guardrail tests

## Data setup

- [ ] Download the Olist dataset from Kaggle
- [ ] Place the required CSV files inside `data/raw/`
- [ ] Confirm the expected filenames match the repository loader

## First analysis pass

- [ ] Run `pytest`
- [ ] Open `notebooks/01_eda.ipynb`
- [ ] Answer the three EDA questions in the notebook
- [ ] Write 4 to 6 useful charts

## First modeling pass

- [ ] Run `python -m src.models.train`
- [ ] Compare the baseline with the tree-based model
- [ ] Record metrics in `reports/model_card.md`
- [ ] Update the README with real findings

## Publish

- [ ] Initialize git if needed
- [ ] Commit the scaffold
- [ ] Push to the personal GitHub repository
- [ ] Pin the repository on the GitHub profile
