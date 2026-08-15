# MLFromScratch - A Machine Learning Model library recreating widely used ML algorithms

**Important note:** This project is a work in Progress as of now, with the overall goal of solidifying my knowledge of ML on a core level. Expect progress to be gradual as this is a learning experience above all else :)

## Overview
All core algorithms & model classes can be found in the ML class
Imports should follow the form: from ML import (model)
The names of the model classes to be used in import statements are as follows:
- LinearRegression
- LogisticRegression
- KNN
- ClassificationTree

Each model has their own fit() and predict() method and can be used like a typical scikit-learn model

Jupyter notebooks used to benchmark my algorithms against scikit-learn can be found in the benchmarks folder. The classification models are tested on a kaggle dataset regarding if tumors are cancerous or not, with 33 features and 529 rows.

## Current Features 
- Base model superclass
- Modules(WIP)
- Linear regression model, including testing & benchmarking
- Logistic regression model, with pytest testing & benchmark notebook
- KNN, with benchmarking/testing/visualization notebook
- Classification Tree, including benchmarking

## Roadmap / Todo (models to be implemented)
- Regression tree
- Random forest
- K-means 
- PCA



## Installation

Clone the repository, then install in editable mode:

```bash
git clone https://github.com/PaulJMartinelli/MLFromScratch
cd machine-learning-from-scratch
pip install -e .
```

## Requirements

Install dependencies: 

```bash
pip install -r requirements.txt
```
