# Project Plan

## Goal

Build a portfolio-ready data science project that predicts late deliveries in the Olist marketplace using only purchase-time signals.

## Scope

- Use a subset of the Olist public e-commerce dataset.
- Join multiple tables into an order-level modeling frame.
- Perform EDA in a notebook.
- Train one baseline and one stronger model.
- Document results, risks, and limitations.

## Intended audience

- recruiters reviewing a first serious DS portfolio project;
- hiring managers looking for evidence of clean thinking and practical ML fundamentals;
- the repository owner, as a reproducible study artifact.

## Success criteria

- the repository is understandable in under two minutes;
- a new environment can install dependencies and run the pipeline;
- the model uses a clear target and avoids leakage;
- the project contains a business-facing explanation, not only code.

## Modeling defaults

- target: late delivery flag;
- primary metric: average precision;
- baseline: `DummyClassifier`;
- stronger model: `RandomForestClassifier`;
- test split: stratified holdout.

## Explicit non-goals

- no Docker in v1;
- no Airflow, dbt, Great Expectations, or MLflow in v1;
- no attempt to build a production-grade service;
- no deep learning.
