# Machine Learning Algorithms

> Seven core ML algorithms implemented and explained in Jupyter notebooks, plus model benchmarks and end-to-end case studies on real datasets.

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-notebooks-F37626?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

A working reference for the algorithms that make up the foundation of supervised and
unsupervised learning. Each algorithm gets its own folder with worked examples, and the
repository builds from single-algorithm notebooks up to full modelling workflows.

Three layers:

1. **`algorithms/`** — one folder per algorithm, from an introduction through applied examples
2. **`benchmarks/`** — head-to-head comparison of many models on the same task
3. **`case-studies/`** — complete workflows on real datasets, from raw data to evaluation

## Algorithms

| # | Algorithm | Type | Covered |
| --- | --- | --- | --- |
| 01 | Linear Regression | Regression | Introduction, two worked examples, three applied projects |
| 02 | Polynomial Regression | Regression | Fitting non-linear relationships with polynomial features |
| 03 | K-Nearest Neighbours | Both | Three exercises across classification and regression |
| 04 | K-Means | Clustering | Unsupervised grouping, centroid convergence |
| 05 | Decision Trees | Classification | Tree construction, visualisation, the play-tennis example |
| 06 | Random Forest | Ensemble | Bagging over decision trees |
| 07 | Logistic Regression | Classification | Applied to sonar rock-vs-mine classification |

### Linear Regression Projects

- **E-Commerce** — predicting annual customer spend from session and membership data
- **Delaney Solubility** — regression on molecular descriptors to predict compound solubility
- **Olympic Medal Count** — forecasting national medal totals from historical data

## Benchmarks

`benchmarks/` runs the same five model families over one dataset so their behaviour can be
compared directly, each wrapped in a `StandardScaler` pipeline:

| | Classification | Regression |
| --- | --- | --- |
| **Dataset** | Breast Cancer Wisconsin | California Housing |
| **Models** | Logistic Regression, KNN, SVC, Decision Tree, Random Forest | Linear Regression, KNN, SVR, Decision Tree, Random Forest |
| **Metrics** | Accuracy, confusion matrix, classification report | MAE, MSE, R² |
| **Validation** | `cross_val_score` | `cross_validate` with `KFold` |

Both datasets load directly from `sklearn.datasets`, so the notebooks run with no local data.

## Case Studies

| Notebook | Dataset | Focus |
| --- | --- | --- |
| Bank Marketing | Portuguese bank campaign | Categorical encoding and term-deposit prediction with XGBoost |
| KDD (parts 1–2) | Titanic, Real Estate, Teams | The full Knowledge Discovery pipeline: cleaning, transformation, class imbalance handling with SMOTE, mining, evaluation |
| KDDL HW1 | Tabular | Applied KDD coursework |

## Requirements

- Python 3.8 or newer
- Jupyter (Notebook or Lab)

## Installation

```bash
git clone https://github.com/0bada11/machine-learning-algorithms.git
cd machine-learning-algorithms
pip install -r requirements.txt
```

## Usage

```bash
jupyter notebook
```

Then open any notebook. Datasets are stored alongside the notebooks that use them, so they
run without further setup. Notebooks using `sklearn.datasets` loaders (iris, breast cancer,
California housing) fetch their data automatically.

## Project Structure

```
machine-learning-algorithms/
├── algorithms/
│   ├── 01-linear-regression/
│   │   └── projects/          # E-Commerce, Delaney Solubility, Olympic Medals
│   ├── 02-polynomial-regression/
│   ├── 03-k-nearest-neighbors/
│   ├── 04-k-means-clustering/
│   ├── 05-decision-trees/
│   ├── 06-random-forest/
│   └── 07-logistic-regression/
├── benchmarks/
│   ├── model-benchmark-classification.ipynb
│   └── model-benchmark-regression.ipynb
├── case-studies/               # Bank Marketing, KDD, Titanic, Real Estate
├── requirements.txt
└── LICENSE
```

## Related Repositories

[Python Fundamentals](https://github.com/0bada11/python-fundamentals) — the NumPy, pandas,
and Matplotlib groundwork these notebooks build on ·
[AI Topics Notes](https://github.com/0bada11/ai-topics-notes) — written notes on
transformers, RAG, and neural networks

## License

Released under the [MIT License](LICENSE). Datasets included here are public and remain
under their own respective licences.
