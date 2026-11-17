# Radar and Sonar Echo Classification

**PH1050 – Final Group Project – Problem 8 - Group 9**

## Project Overview

This project investigates whether a metal object can be distinguished from a rock using the energy pattern of the sonar echo reflected by the object.

Each sonar return is represented by 60 energy measurements (frequency bins). We analyse these measurements statistically and use machine-learning models to classify the returned echoes into two categories: **Metal** and **Rock**.

## Project Workflow

Our project follows four main stages:

1. **Data Collection and Preparation** – We prepared a synthetic sonar echo dataset and used the UCI Sonar dataset as a real-data comparison.
2. **Data Analysis** – We calculated the mean energy of each return, compared the two classes using Welch's t-test and a 95% confidence interval, and plotted the average energy profiles.
3. **Model Training** – We implemented a k-nearest neighbours (kNN) classifier and a decision-tree classifier. For kNN, the optimal value of k was selected using stratified 5-fold cross-validation.
4. **Model Testing** – We evaluated the models using cross-validation and train-test splitting. We also investigated the effect of increasing noise on classification using the synthetic dataset.

## Datasets

We used two datasets:

### 1. Synthetic Dataset

File: `synthetic_echo_returns_400_formatted.csv`

- Total samples: 400
- Metal cylinder samples: 200
- Rock samples: 200
- Features: 60 energy-bin measurements per sample.

The dataset contains the following columns:

- `energy_bin_01` to `energy_bin_60`: Echo energy measurements.
- `label`: Class identifier (`M` or `R`).
- `class_name`: Descriptive class name (`metal_cylinder` or `rock`).

Here, `M` represents a metal cylinder and `R` represents a rock.

### 2. Real Dataset – UCI Sonar

File: `uci_sonar_clean_reformatted.csv`

- Total samples: 208
- Metal cylinder samples: 111
- Rock samples: 97
- Features: 60 energy-bin measurements per sample.

The dataset contains:

- `return_id`: Identifier for each sonar return.
- `class`: Target class, containing `metal_cylinder` or `rock`.
- `energy_bin_01` to `energy_bin_60`: Echo energy measurements.

The `return_id` column is an identifier and is not used as a model feature.

**Original dataset source:** [UCI Machine Learning Repository – Connectionist Bench (Sonar, Mines vs. Rocks)](https://doi.org/10.24432/C5T01Q).

## Part 1: Data Analysis

For each sonar return, we calculated the mean of its 60 energy measurements.

We then compared the mean energy of metal and rock samples using:

- Welch's two-sample t-test.
- 95% confidence interval for the difference in means.
- Average energy profile plots for both classes.

### Statistical Results

| Dataset | Metal Mean | Rock Mean | Welch's t-statistic | p-value |
|---|---:|---:|---:|---:|
| Synthetic | 0.3281 | 0.1843 | 37.4805 | 3.09 × 10⁻¹¹⁵ |
| UCI Sonar | 0.2915 | 0.2697 | 4.1353 | 5.32 × 10⁻⁵ |

The 95% confidence intervals for the difference (Metal − Rock) were:

- Synthetic dataset: [0.1362, 0.1513]
- UCI Sonar dataset: [0.0114, 0.0323]

Both datasets show a higher average echo energy for metal samples compared to rock samples. The statistical results indicate a significant difference between the two classes.

## Part 2: Machine Learning

### k-Nearest Neighbours (kNN)

We implemented a kNN classifier using a pipeline consisting of:

- **StandardScaler:** To standardise the input features.
- **KNeighborsClassifier:** To classify sonar returns based on their nearest neighbours.

Different values of k were evaluated using stratified 5-fold cross-validation, and the optimal value was selected based on the cross-validation performance.

A majority-class classifier was also used as a baseline for comparison.

### Decision Tree

We also implemented a decision-tree classifier to classify the sonar returns into metal and rock categories.

The model was trained using the energy-bin measurements as input features and the corresponding material labels as the target. The maximum depth of the decision tree was controlled to limit model complexity.

### Model Testing and Evaluation

The trained models were evaluated using cross-validation and train-test splitting.

We also performed a noise analysis on the synthetic dataset by introducing different levels of Gaussian noise into the energy measurements and observing the effect on classification.

These experiments allowed us to study the performance and reliability of the machine-learning models under different data conditions.

## Conclusion

This project investigates the possibility of distinguishing metal objects from rocks using sonar echo-energy measurements.

The statistical analysis shows that metal samples have a higher average echo energy than rock samples in both datasets. We further explored machine-learning approaches, including k-nearest neighbours and decision trees, to automate the classification process.

The project combines physical interpretation, statistical analysis, and data-driven modelling to understand how sonar echo patterns can be used for material classification.

## Team Contributions

| Responsibility | Team Members |
|---|---|
| Data Collection | Santhosh S, Vinodh |
| Data Analysis | Aaradhya Jain, Dhiwin |
| Model Making | Dushyanth Rangarajan, Siddharth Tarimala |
| Model Testing | Pavan |

## Project Files

| File | Description |
|---|---|
| `synthetic_echo_returns_400_formatted.csv` | Synthetic sonar echo dataset |
| `uci_sonar_clean_reformatted.csv` | Real UCI Sonar dataset |
| `Data_Analysis_Problem_8_group_9.ipynb` | Statistical analysis of synthetic data |
| `Data_Analysis_for_Real_Data.ipynb` | Statistical analysis of real data |
| `Part3_kNN (1).ipynb` | kNN training, cross-validation and testing |
| `Decision_tree_endsemproject.ipynb` | Decision-tree classification |

## Tools and Libraries Used

- Python
- Jupyter Notebook / Google Colab
- NumPy
- Pandas
- SciPy
- Matplotlib
- Seaborn
- Scikit-learn

### Installation

Install the required Python libraries using:

```bash
pip install numpy pandas scipy matplotlib seaborn scikit-learn ucimlrepo
```

Open the notebooks in Jupyter Notebook or Google Colab. Ensure the corresponding CSV files are available in the working directory and update file paths wherever necessary.

---

**Course:** PH1050 – Foundation of Computational Physics  
**Project:** Final Group Project – Problem 8  
**Institution:** Indian Institute of Technology Madras
