# 🔊 Radar and Sonar Echoes: Can We Tell a Metal Object from a Rock by Its Echo?

**PH1050 – Foundation of Computational Physics | Final Group Project (Problem 8)**
Instructor: Dr. Ayana Ghosh · IIT Madras

> A computational study that combines **wave physics** (echo timing and acoustic reflection), **statistics** (Welch's t-test, confidence intervals) and **machine learning** (kNN / decision tree) to answer one question: *can an echo alone tell a metal object from a rock?*

---

## 📌 Table of Contents
1. [Project Overview](#-project-overview)
2. [Team and Roles](#-team-and-roles)
3. [Repository Structure](#-repository-structure)
4. [Data](#-data)
5. [Methodology](#-methodology)
6. [Results](#-results)
7. [Installation and Usage](#-installation-and-usage)
8. [Running the Tests](#-running-the-tests)
9. [Rubric Mapping](#-rubric-mapping)
10. [References](#-references)

---

## 🎯 Project Overview

Sonar and radar systems send out a pulse and listen for the echo. Two things come back:

- **When** the echo arrives, which tells us the distance to the target.
- **How strong** the echo is, which depends on the material of the target.

This project investigates both, in three parts:

| Part | Goal |
|------|------|
| **Part 1 – Physics model** | Simulate a noisy echo, recover the target distance from the peak, and study how the error scales with the sampling step Δt. Derive and numerically verify the reflection and transmission coefficients, then predict whether steel or granite reflects more strongly in water. |
| **Part 2 – Data and statistics** | Compute the mean energy of each return, compare the two classes with a **Welch t-test** and a **confidence interval** for the difference, and plot the average profile of each class. Check whether the data agree with the physics prediction. |
| **Part 3 – Data-driven model** | Classify metal vs rock with **kNN** (or a decision tree). Pick `k` (or depth) by **5-fold cross-validation**, report accuracy ± spread against a majority-class baseline, and repeat at a higher noise level or on a second dataset. |

---

## 👥 Team and Roles

| Role | Team Members | Responsibilities |
|------|--------------|------------------|
| **Data Collection** | Santhosh S, Vinodh | Sourcing / generating the dataset (200 returns per class, 60 energy bins), adding realistic noise and an unmodelled effect, documenting provenance |
| **Data Analysis** | Aaradhya Jain, Dhiwin | Per-return means, Welch t-test, confidence interval, average-profile plots, comparison with the physics prediction |
| **Model Making** | Dushyanth Rangarajan, Siddharth Tarimala | Echo-timing simulation, reflection/transmission physics, kNN / decision-tree classifier, cross-validation |
| **Model Testing** | Pavan | Unit tests, verification of limiting cases, baseline comparison, second dataset / higher-noise robustness check |

> *Detailed who-did-what table is included in the final report (`report/`).*

---

## 📁 Repository Structure

```
radar-sonar-echoes/
│
├── README.md                  # You are here
├── requirements.txt           # Python dependencies
├── .gitignore
│
├── data/
│   ├── raw/                   # Original / downloaded data (if real data used)
│   └── generated/             # Simulated datasets (200 returns/class x 60 bins)
│
├── src/
│   ├── data_generation.py     # Simulated sonar returns + noise model
│   ├── echo_timing.py         # Part 1: pulse echo, peak detection, distance recovery
│   ├── reflection.py          # Part 1: R, T, impedance, material table
│   ├── statistics_analysis.py # Part 2: means, Welch t-test, CI, profile plots
│   ├── classifier.py          # Part 3: kNN / decision tree + 5-fold CV
│   └── utils.py               # Shared helpers (plotting, seeding, I/O)
│
├── notebooks/
│   └── main_analysis.ipynb    # End-to-end walkthrough
│
├── tests/
│   ├── test_echo_timing.py    # Distance recovered within one sampling step
│   ├── test_reflection.py     # R + T = 1, R = 0 when Z1 = Z2, R -> 1 when Z2 >> Z1
│   └── test_statistics.py     # Welch test / CI sanity checks
│
├── figures/                   # All generated plots
│
└── report/
    ├── report.pdf             # 6–8 page report with who-did-what table
    └── slides.pdf             # 12-minute talk
```

> Adjust file names to match what your team actually commits.

---

## 📊 Data

> ✏️ **Team: fill in the option you used and delete the other.**

**Option A – Real dataset:** *UCI Sonar (Mines vs Rocks)* – 208 sonar returns, 60 energy values each, labelled metal cylinder (M) vs rock (R). Source: <https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks>

**Option B – Simulated dataset:**
- 200 returns per class (metal, rock), each binned into **60 energy values**.
- Echo amplitude is set by the reflection coefficient of each material.
- Gaussian noise is added to every bin.
- An **unmodelled effect** (e.g. an additional effect such as multipath echoes, frequency-dependent attenuation, or a varying noise floor) is included in the data but *not* in our model, so that the comparison is a genuine test and not a self-fulfilling one.
- Generated with a fixed random seed for reproducibility (`src/data_generation.py`).

---

## 🔬 Methodology

### Part 1 – Physics Model

**Echo timing.** A pulse travels at speed `v` to a target at distance `d` and back:

```
t = 2d / v          →     d = v·t / 2
```

The noisy return is sampled every Δt, the echo is located from its peak, and `d` is recovered (v = 1500 m/s in water). The error in `d` is plotted against Δt on a **log–log plot** to study its scaling. The recovered distance should lie within one sampling step.

**Reflection and transmission.** At a boundary between media with acoustic impedance `Z = ρv`:

```
R = ((Z₂ − Z₁) / (Z₂ + Z₁))²          T = 4 Z₁ Z₂ / (Z₁ + Z₂)²
```

Numerical checks:
- ✅ `R + T = 1`
- ✅ `R = 0` when `Z₁ = Z₂`
- ✅ `R → 1` when `Z₂ ≫ Z₁`

**Material prediction.** Using typical density and sound speed for water, steel and granite (see the table below), we predict which of metal and rock returns the stronger echo.

| Material | ρ (kg/m³) | v (m/s) | Z = ρv (MRayl) |
|----------|-----------|---------|----------------|
| Water    | ~1000     | ~1500   | ~1.5           |
| Steel    | ~7850     | ~5900   | ~46            |
| Granite  | ~2700     | ~5800   | ~16            |

*Typical textbook values. Cite your exact source in the report.*

With these values, the water–steel boundary gives R ≈ 0.88 and the water–granite boundary gives R ≈ 0.68, so **steel is predicted to reflect more strongly**.

### Part 2 – Statistics
- Mean energy computed for each return (row).
- **Welch's t-test** (unequal variances) between the metal and rock classes.
- **95 % confidence interval** for the difference in means.
- Average profile (mean across returns, per energy bin) plotted for each class.
- Conclusion: does the data agree with the Part 1 prediction?

### Part 3 – Classification
- Classifier: **kNN** (or decision tree).
- Hyperparameter (`k` or tree depth) chosen by **5-fold cross-validation**.
- Reported as accuracy ± standard deviation across folds.
- **Baseline:** always predict the majority class.
- Repeated on a **second dataset or a higher noise level**, comparing the best `k`.

---

## 📈 Results

> ✏️ **Fill in once the analysis is complete. Add the plots from `figures/`.**

| Quantity | Value |
|----------|-------|
| Distance recovery error (within one Δt?) | _TBD_ |
| R + T (numerical check) | _TBD_ |
| Predicted stronger reflector | _TBD_ |
| Welch t-statistic / p-value | _TBD_ |
| 95 % CI for mean difference | _TBD_ |
| Best k (or depth) from 5-fold CV | _TBD_ |
| CV accuracy (mean ± std) | _TBD_ |
| Majority-class baseline accuracy | _TBD_ |
| Best k on second dataset / higher noise | _TBD_ |

**Key figures (add after generating):**

```markdown
![Error vs dt](figures/error_vs_dt.png)
![Average class profiles](figures/average_profiles.png)
![CV accuracy vs k](figures/cv_accuracy_vs_k.png)
```

---

## ⚙️ Installation and Usage

**1. Clone the repository**
```bash
git clone https://github.com/<your-username>/radar-sonar-echoes.git
cd radar-sonar-echoes
```

**2. (Recommended) Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the analysis**
```bash
python src/data_generation.py       # generate the dataset (if simulated)
python src/echo_timing.py           # Part 1: echo timing + error vs dt
python src/reflection.py            # Part 1: reflection / transmission checks
python src/statistics_analysis.py   # Part 2: Welch t-test, CI, profiles
python src/classifier.py            # Part 3: kNN + cross-validation
```
Or open `notebooks/main_analysis.ipynb` for the full walkthrough.

**Suggested `requirements.txt`**
```
numpy
scipy
pandas
matplotlib
scikit-learn
pytest
jupyter
```

---

## 🧪 Running the Tests

```bash
pytest tests/ -v
```

The tests check that:
- the recovered distance is within one sampling step,
- `R + T = 1` and both limiting cases hold numerically,
- the statistical routines behave correctly on known inputs.

---

## 📝 Rubric Mapping

| Rubric line | Requirement | Where |
|-------------|-------------|-------|
| B2 | Distance recovered within one sampling step | `src/echo_timing.py`, `tests/test_echo_timing.py` |
| B3 | R + T = 1 and both limiting cases checked numerically | `src/reflection.py`, `tests/test_reflection.py` |
| B4 | Log–log plot of error vs Δt | `figures/error_vs_dt.png` |
| B5 | Predicted stronger reflector vs the data's mean return | `src/reflection.py`, `src/statistics_analysis.py` |
| C4 | Welch t-test and CI read correctly | `src/statistics_analysis.py`, report |
| D6 | Second dataset or noise level, with a conclusion | `src/classifier.py`, report |

---

## 📚 References

- Dua, D. and Graff, C. *UCI Machine Learning Repository* (Sonar, Mines vs. Rocks dataset), University of California, Irvine.
- Kinsler, L. E. et al. *Fundamentals of Acoustics*, Wiley (acoustic impedance, reflection and transmission).
- Material property tables (ρ, v) — *cite your source here*.
- Course material, PH1050 – Foundation of Computational Physics, IIT Madras.

---

## 📄 License

This project was developed for academic purposes as part of PH1050 at IIT Madras.
