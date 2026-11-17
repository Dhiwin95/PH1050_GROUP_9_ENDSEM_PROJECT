# Radar and Sonar Echo Classification

**PH1050 – Foundation of Computational Physics | Final Group Project – Problem 8 | IIT Madras**

## Project overview

This project investigates whether a metal object can be distinguished from a rock using the energy pattern of the sonar echo reflected by the object. Each sonar return is represented by **60 energy measurements**, one for each frequency bin. We compare the two classes using statistical analysis and train machine-learning models to classify an echo as **metal** or **rock**.

## Project workflow

Our work follows four main stages:

1. **Data collection and preparation** – We prepared a synthetic echo dataset and used the UCI Sonar dataset as a second, real-data comparison. The data were arranged as 60 energy-bin features and a class label for each return.
2. **Data analysis** – For every return, we calculated the mean across its 60 energy bins. We compared the class means using Welch's two-sample t-test and a 95% confidence interval, and plotted the average energy profile of each class.
3. **Model training** – We implemented a k-nearest neighbours (kNN) classifier using feature standardisation. The value of `k` was selected using stratified 5-fold cross-validation and compared with a majority-class baseline. A decision-tree experiment is also included in a separate notebook.
4. **Model testing** – We evaluated the kNN model using cross-validation and a stratified 80/20 train-test split. We also tested how added Gaussian noise affects classification on the synthetic dataset.

## Datasets

| Dataset | Samples | Metal | Rock | Features per sample |
|---|---:|---:|---:|---:|
| Synthetic echo returns | 400 | 200 | 200 | 60 energy bins |
| UCI Sonar (Mines vs. Rocks) | 208 | 111 | 97 | 60 energy attributes |

- **Synthetic dataset:** `synthetic_echo_returns_400_formatted.csv`. The class labels are `metal_cylinder` / `rock` and `M` / `R`.
- **Real dataset:** `uci_sonar_clean_reformatted.csv`. This is the UCI Connectionist Bench (Sonar, Mines vs. Rocks) dataset. In the original labels, `M` refers to a metal cylinder (mine) and `R` refers to a rock. Each record has 60 energy values in the range 0 to 1. Dataset source: [UCI Machine Learning Repository – Connectionist Bench (Sonar, Mines vs. Rocks)](https://doi.org/10.24432/C5T01Q).

The uploaded files contain the prepared synthetic CSV, but not the script that generated it. For reproducibility, the exact synthetic echo-generation method and noise settings should also be documented in the repository if that script is maintained separately.

## Statistical analysis

For each sample, we computed the mean of its 60 energy values. We then compared the metal and rock groups using Welch's t-test and calculated a 95% confidence interval for the difference in means (**metal − rock**).

| Dataset | Mean energy: metal | Mean energy: rock | Welch's t-statistic | p-value | 95% CI for metal − rock |
|---|---:|---:|---:|---:|---:|
| Synthetic | 0.3281 | 0.1843 | 37.4805 | 3.09 × 10⁻¹¹⁵ | [0.1362, 0.1513] |
| UCI Sonar | 0.2915 | 0.2697 | 4.1353 | 5.32 × 10⁻⁵ | [0.0114, 0.0323] |

For both datasets, the mean energy is higher for the metal class. The small p-values and confidence intervals that do not include zero indicate a statistically significant difference in the sample means.

## Machine-learning models and results

### k-nearest neighbours (kNN)

The kNN workflow uses a `StandardScaler` followed by `KNeighborsClassifier`. We tested odd values of `k` from 1 to 31 using stratified 5-fold cross-validation. A majority-class `DummyClassifier` provides the baseline. The notebook also uses nested cross-validation to estimate performance while accounting for the selection of `k`.

| Dataset | Selected `k` | Best 5-fold CV accuracy | Nested 5-fold CV accuracy |
|---|---:|---:|---:|
| Synthetic | 21 | 100.00% | 100.00% |
| UCI Sonar | 1 | 84.65% | 83.18% |

These kNN values were reproduced by running the workflow in the uploaded notebook on the supplied CSV files; the notebook itself did not contain saved outputs for these result cells.

On the UCI dataset, the additional stratified 80/20 hold-out test gave **92.86% accuracy on 42 test samples**. Because this test set is small, the cross-validation results are more informative than the single hold-out score.

The synthetic dataset remains highly separable at low added-noise levels. In the noise-sweep experiment, the best 5-fold CV accuracy decreased from 100% without added noise to approximately 89.0% at noise standard deviation 0.4, 77.8% at 0.6, and 69.3% at 0.8.

### Decision tree

A separate notebook contains a decision-tree experiment using a maximum depth of 5 and an 80/20 stratified train-test split. Its saved data preview and feature/target column selection do not clearly match the supplied CSV structure. The notebook should be corrected to select the 60 `energy_bin_` columns as input features and the class-label column as the target before its accuracy or classification report is included as a verified result.

## Conclusion

The statistical analysis shows that the metal class has a higher average returned energy than the rock class in both the synthetic and UCI datasets. The kNN classifier separates the synthetic samples very well and achieves about **83% nested cross-validation accuracy** on the UCI Sonar data. The synthetic noise experiment also shows that classification becomes more difficult as the noise level increases. These results suggest that the echo-energy profile contains useful information for distinguishing the two classes, while performance on real data is less perfect than on the synthetic data.

## Team contributions

| Responsibility | Team members |
|---|---|
| Data collection | Santhosh S, Vinodh |
| Data analysis | Aaradhya Jain, Dhiwin |
| Model making | Dushyanth Rangarajan, Siddharth Tarimala |
| Model testing | Pavan |

## Project files

The project materials include the following notebooks and datasets:

- `Data_Analysis_Problem_8_group_9.ipynb` – statistical analysis of the synthetic dataset.
- `Data_Analysis_for_Real_Data.ipynb` – statistical analysis of the UCI Sonar dataset.
- `Part3_kNN (1).ipynb` – kNN model selection, cross-validation, hold-out testing, and noise-sweep analysis.
- `Decision_tree_endsemproject.ipynb` – decision-tree experiment.
- `synthetic_echo_returns_400_formatted.csv` – synthetic echo dataset.
- `uci_sonar_clean_reformatted.csv` – cleaned UCI Sonar dataset.

## Tools and libraries

Python, Jupyter Notebook / Google Colab, NumPy, pandas, SciPy, Matplotlib, Seaborn, and scikit-learn.

To install the main Python libraries:

```bash
pip install numpy pandas scipy matplotlib seaborn scikit-learn ucimlrepo
```
