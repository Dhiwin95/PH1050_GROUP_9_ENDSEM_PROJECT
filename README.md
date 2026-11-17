# Radar and Sonar Echoes: Metal Object or Rock?

**PH1050 – Foundation of Computational Physics · Final Group Project (Problem 8) · IIT Madras**

Can we tell a metal object from a rock by the echo it sends back? We test this with a physics model, a statistical comparison, and a classifier, on both a simulated dataset and the real UCI Sonar dataset.

## Team

| Role | Members |
|------|---------|
| Data Collection | Santhosh S, Vinodh |
| Data Analysis | Aaradhya Jain, Dhiwin |
| Model Making | Dushyanth Rangarajan, Siddharth Tarimala |
| Model Testing | Pavan |

## Data

Each return is a row of **60 energy bins** (`energy_bin_01` … `energy_bin_60`) with a class label.

| File | Source | Size |
|------|--------|------|
| `data/synthetic_echo_returns_400_formatted.csv` | Simulated: pulse echo scaled by each material's reflection coefficient, plus noise | 200 metal + 200 rock |
| `data/uci_sonar_clean_reformatted.csv` | UCI Sonar (Mines vs Rocks), cleaned and reformatted | 111 metal + 97 rock |

Extra Gaussian noise (clipped to [0, 1]) is added to the synthetic data for the higher-noise tests in Part 3.

## What we did

- **Part 1 – Physics model:** echo timing (`d = vt/2`), reflection/transmission coefficients (`R + T = 1`), and the steel vs granite prediction. *(Add the notebook link here.)*
- **Part 2 – Statistics:** per-return mean energy, Welch t-test, 95% CI for the difference (metal − rock), and average profile plots.
- **Part 3 – Classification:** kNN (with scaling) and a decision tree, with `k` / depth chosen by 5-fold cross-validation and compared with a majority-class baseline. Repeated on the real UCI data and at higher noise levels.

## Results

**Part 2: Welch t-test (metal − rock)**

| Dataset | Mean (rock → metal) | t | p-value | 95% CI of difference |
|---------|--------------------|---|---------|----------------------|
| Synthetic | – | 37.48 | 3.1e-115 | [0.1362, 0.1513] |
| UCI Sonar | 0.2697 → 0.2915 | 4.14 | 5.3e-05 | [0.0114, 0.0323] |

Metal returns have a significantly higher mean energy in both datasets, as predicted.

**Part 3: 5-fold CV accuracy (mean ± std)**

| Dataset | Baseline | Best kNN | Best decision tree |
|---------|----------|----------|--------------------|
| Synthetic | 0.500 | 1.000 ± 0.000 | 0.990 ± 0.010 (depth 3) |
| Synthetic + noise 0.2 | 0.500 | 0.983 ± 0.014 (k = 5) | 0.847 ± 0.053 (depth 6) |
| Synthetic + noise 0.4 | 0.500 | 0.890 ± 0.027 (k = 25) | 0.708 ± 0.034 (depth 5) |
| Synthetic + noise 0.8 | 0.500 | 0.693 ± 0.019 (k = 31) | 0.645 ± 0.031 (depth 3) |
| UCI Sonar (real) | 0.534 | 0.846 ± 0.042 (k = 1) | 0.745 ± 0.059 (depth 1) |

As noise increases, the best `k` grows (5 → 25 → 31): more neighbours are needed to average out the noise. kNN beats the decision tree throughout.

## Repository structure

```
├── README.md
├── requirements.txt
├── data/
│   ├── synthetic_echo_returns_400_formatted.csv
│   └── uci_sonar_clean_reformatted.csv
├── notebooks/
│   ├── Data_Analysis_Problem_8_group_9.ipynb     # Part 2, synthetic data
│   ├── Data_Analysis_for_Real_Data.ipynb         # Part 2, UCI Sonar
│   ├── Part3_kNN.ipynb                           # Part 3, kNN + CV + noise sweep
│   └── Decision_tree_endsemproject.ipynb         # Part 3, decision tree
└── figures/
```

## How to run

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
jupyter notebook notebooks/
```

`requirements.txt`: `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `jupyter`

The notebooks were written in Google Colab; update the data paths at the top of each notebook to point to the `data/` folder.
