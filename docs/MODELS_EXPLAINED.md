# The five models, scaling, ensembles and optimisation: explained

A study guide for the machine-learning part of Exoplanet Atlas: what each model is,
how we trained and tested it, the numbers we got, where the code lives, and the
concepts around it (feature scaling, normalisation, standardisation, missing values,
ensembles, tuning, leakage, metrics). File references are relative to the
repository root; line numbers are as of October 2026.

---

## 1. The machine-learning problem in one paragraph

Every star becomes **one row of 56 numbers** (the features: dip depth, dip shape,
second-eclipse strength, how the star varies, its temperature and size, ...) and
**one answer** (the label): PLANET, EB (eclipsing binary), VARIABLE or NOISE. We
have **400 labelled stars, 100 per class**. A classifier learns the mapping
*56 numbers -> class* and must output a probability for each of the 4 classes.

- Inputs `X`: a table of 400 rows x 56 columns (`data/processed/features.parquet`)
- Answers `y`: 400 labels, from public catalogs (never from our own model)
- Task type: **supervised, multi-class (4 classes) classification on tabular data**
- Main metric: **macro-F1** (explained in section 3)

---

## 2. The process every model goes through (the same for all 5)

This is what makes the comparison fair: same data, same folds, same metrics.

### Step 1: load the table
`src/triage/train.py`, `load_dataset()` (line 55)
- keeps only stars whose pipeline run succeeded (400 of 400)
- keeps the 56 model features; drops a feature that is empty for every star
  (`ruwe` was dropped)
- turns infinities into missing values (line 62)
- builds `X` (features), `y` (labels as 0-3), `groups` (the TIC ID of each star)

### Step 2: split into 5 folds, grouped by star
`src/triage/train.py`, `cv_folds()` (line 100), using `StratifiedGroupKFold`
- **5-fold cross-validation**: the 400 stars are cut into 5 groups of 80. Each model
  is trained 5 times, each time on 320 stars, and tested on the 80 it did not see.
- **Stratified**: every fold keeps the 25/25/25/25 class balance.
- **Grouped by TIC ID**: one star can never be on both sides of a split (the code
  asserts this). Without grouping, a star could leak into its own test.
- After 5 rounds, **every star has been predicted exactly once by a model that never
  saw it**. These are the *out-of-fold* (OOF) predictions.
- Seed 42 makes the split identical every run, for every model.

### Step 3: handle missing values and scale (only for models that need it)
33 of the 56 features have gaps (738 of 22,400 cells), e.g. `duration_ratio_expected`
is missing for 115 stars (no catalog density), `logg` for 104.
- XGBoost: handles gaps itself (section 4.1).
- Random forest: gaps filled with the median.
- Logistic regression, SVM, KNN: gaps filled with the median **and** features
  standardised (section 6).

### Step 4: balance the classes
`src/triage/train.py`, `_fit()` (line 90): `compute_sample_weight("balanced", y)`
gives each class the same total weight. Our classes are already equal (100 each), so
this changes nothing today, but keeps training fair if class sizes ever differ. The
other models get the same effect with `class_weight="balanced"`.

### Step 5: train, predict, score
`src/triage/train.py`, `cv_predict()` (line 113) and `metrics()` (line 185): for
each fold, train on 320, predict probabilities for 80, compute the metrics; then
report **mean ± standard deviation over the 5 folds**. The ± is important: it shows
how much the score moves when the stars change.

### Step 6 (XGBoost only, the deployed model): calibrate, then fit on all 400
- Calibration (section 9.3): `fit_calibration()` (line 139), `cross_calibrate()` (line 158)
- Final model trained on all 400 stars and saved as `models/triage_xgb.json`
  (`train()`, line 277)

**Commands**
```bash
python scripts/train_triage.py         # XGBoost, random forest, logistic regression, soft-voting ensembles
python scripts/compare_classifiers.py  # + SVM and KNN, all metrics, confusion matrices
python scripts/save_evaluation.py      # writes models/model_evaluation.json for the app
```

---

## 3. The metrics, in plain words (with our numbers)

Take one class, say PLANET:

| Metric | Question it answers | Formula |
|---|---|---|
| **Precision** | Of the stars we *called* planet, how many really are? | TP / (TP + FP) |
| **Recall** | Of the *real* planets, how many did we find? | TP / (TP + FN) |
| **F1** | One number balancing both | 2 x P x R / (P + R) |
| **Macro-F1** | The average F1 of the 4 classes, each counted equally | mean(F1_planet, F1_EB, F1_var, F1_noise) |
| **Accuracy** | Share of all stars classified correctly | correct / total |
| **PR-AUC** | How well the probabilities *rank* each class (area under the precision-recall curve; 1 = perfect ranking) | averaged over the 4 classes |
| **Log loss** | How good the *probabilities* are; punishes confident wrong answers heavily; lower is better | -mean(log p_true_class) |
| **Confusion matrix** | Which classes get mixed up with which | rows = true class, columns = predicted |
| **Planets kept at 5% EB leakage** | If we set the planet threshold so only 5% of binaries slip through as "planet", how many real planets survive? | (see `planet_recall_at_eb_rate()`, line 167) |

TP = true positive (planet called planet), FP = false positive (non-planet called
planet), FN = false negative (planet called something else).

**Why macro-F1 is the main metric:** all four classes matter equally. Accuracy can
look fine while one class is ignored; macro-F1 cannot.

---

## 4. The five models

Results below are all on the same 400 stars and 5 folds (mean ± spread).

| Model | Accuracy | Precision | Recall | **Macro-F1** | PR-AUC | Log loss | Train (5 folds) |
|---|---|---|---|---|---|---|---|
| **XGBoost** (deployed, calibrated) | 0.87 ± 0.05 | 0.88 | 0.87 | **0.87 ± 0.05** | 0.92 | **0.43** | 2.5 s |
| Random forest | 0.84 ± 0.03 | 0.85 | 0.84 | 0.84 ± 0.03 | 0.90 | 0.55 | 1.5 s |
| SVM (RBF kernel) | 0.80 ± 0.04 | 0.81 | 0.80 | 0.80 ± 0.04 | 0.87 | 0.54 | 0.1 s |
| Logistic regression | 0.78 ± 0.04 | 0.79 | 0.78 | 0.79 ± 0.04 | 0.84 | 0.73 | < 0.1 s |
| K-nearest neighbours | 0.72 ± 0.08 | 0.73 | 0.72 | 0.71 ± 0.08 | 0.82 | 1.06 | < 0.1 s |

F1 per class:

| Model | Planet | Binary | Variable | Noise |
|---|---|---|---|---|
| XGBoost | 0.87 | 0.94 | 0.87 | 0.81 |
| Random forest | 0.83 | 0.92 | 0.84 | 0.77 |
| SVM | 0.81 | 0.85 | 0.80 | 0.72 |
| Logistic regression | 0.81 | 0.88 | 0.78 | 0.68 |
| KNN | 0.76 | 0.79 | 0.71 | 0.58 |

Noise is the hardest class for every model: faint planets and quiet variables look
like noise. That is a property of the data, not of one model.

### 4.1 XGBoost (eXtreme Gradient Boosting): the deployed model

**What it is.** An ensemble of small decision trees built *one after another*
(**boosting**). A decision tree asks yes/no questions about features ("is the dip
V-shaped? is there a second eclipse?") and ends in a leaf with a score. Tree 1 makes
a rough guess; tree 2 is trained to correct tree 1's errors; tree 3 corrects what is
still wrong, and so on. The final answer adds up all 400 trees, then a softmax turns
the sums into 4 probabilities.

**How it learns.** "Gradient boosting": each new tree is fitted to the *gradient of
the loss* (the direction in which the predictions are wrong), and added with a small
step (the learning rate). Regularisation (L2 penalty on leaf values, minimum evidence
per leaf) stops trees from memorising single stars.

**Settings used** (`src/triage/train.py`, `make_xgb()`, line 67)

| Setting | Value | Meaning | Why this value |
|---|---|---|---|
| `objective` | `multi:softprob` | 4-class probabilities | multi-class task |
| `n_estimators` | 400 | number of trees | enough to learn, small steps |
| `max_depth` | 4 | questions per tree | shallow = less overfitting on 400 stars |
| `learning_rate` | 0.05 | weight of each new tree | small steps generalise better |
| `subsample` | 0.8 | each tree sees 80% of stars | randomness = less overfitting |
| `colsample_bytree` | 0.8 | each tree sees 80% of features | same |
| `min_child_weight` | 2 | minimum evidence in a leaf | no leaf for a single star |
| `reg_lambda` | 1.0 | L2 penalty on leaf values | smoother model |
| `tree_method` | `hist` | fast histogram splits | speed |
| sample weights | balanced | each class counts equally | fairness |

**Preprocessing.** None. Trees compare one feature to a threshold, so the scale of a
feature does not matter, and **missing values are handled natively**: at each split
XGBoost learns which side missing values should go to.

**Extra steps (deployed model only).**
- Probability calibration with Platt scaling (section 9.3)
- Exact per-star explanations with SHAP (section 9.4): `src/triage/model.py`,
  `contributions()` (line 88)

**Results.** Macro-F1 **0.87 ± 0.05**, best on every class; best log loss (0.43,
most trustworthy probabilities); keeps 88% of planets at 5% binary leakage; never
calls a real planet a binary. Confusion matrix (out of 100 per class):

| true \ predicted | Planet | EB | Variable | Noise |
|---|---|---|---|---|
| Planet | **87** | 0 | 4 | 9 |
| EB | 3 | **91** | 4 | 2 |
| Variable | 1 | 0 | **87** | 12 |
| Noise | 7 | 3 | 7 | **83** |

**Why it wins here.** Small table, mixed-scale features, missing values, and
interactions between features (e.g. "deep dip AND big star" means binary).
Boosted trees are the standard winners for this kind of data.

**Where the code is.** Built in `make_xgb()` (`src/triage/train.py:67`), trained in
`train()` (line 277), saved to `models/triage_xgb.json` + `models/triage_meta.json`,
loaded and used by `src/triage/model.py` (`TriageModel.predict()`, line 104).

### 4.2 Random forest: the bagging baseline

**What it is.** Also many decision trees, but built *independently* (**bagging**,
"bootstrap aggregating"): each tree trains on a random sample of stars drawn with
replacement and, at each split, considers only a random subset of features. Then
the trees **vote** (their probabilities are averaged). Randomness makes the trees
different; averaging cancels their individual mistakes.

**Settings** (`src/triage/train.py`, `make_baselines()`, line 77): 500 trees,
`min_samples_leaf=2`, `class_weight="balanced_subsample"`, default
`max_features="sqrt"` (each split looks at about 7 of the 56 features).

**Preprocessing.** Missing values filled with the median plus a 0/1 "was missing"
column (`SimpleImputer(strategy="median", add_indicator=True)`, line 83). **No
scaling** (trees do not need it).

**Results.** 0.84 ± 0.03: second best and the most stable across folds (smallest
spread). It misses more planets than XGBoost (16 planets called noise vs 9).

**XGBoost vs random forest in one line:** bagging reduces *variance* (instability)
by averaging independent trees; boosting reduces *bias* (systematic error) by
correcting mistakes step by step. Here the remaining error is more bias than
variance, so boosting wins.

### 4.3 Logistic regression: the linear baseline

**What it is.** The simplest probabilistic classifier. For each class it computes a
weighted sum of the features (z = w1 x feature1 + w2 x feature2 + ... + b) and turns
the 4 sums into probabilities with the softmax function. It draws **straight
(linear) boundaries** between classes.

**Settings** (`make_baselines()`, line 81): `C=0.5` (C is the *inverse* of
regularisation strength: smaller C = weights kept smaller = simpler model),
`class_weight="balanced"`, `max_iter=5000` (enough iterations for the optimiser to
converge).

**Preprocessing.** Median imputation + missing indicator **and standardisation**
(`StandardScaler`, line 80). It *needs* scaling: the weights are compared across
features and the optimiser converges badly when one feature is in the thousands and
another between 0 and 1.

**Results.** 0.79 ± 0.04. Good on planets (0.81) and binaries (0.88), weak on
variables and noise (0.68): separating those needs combinations of features (curved
boundaries) that a linear model cannot draw. Its log loss (0.73) is the second worst,
after KNN: its probabilities are less trustworthy than the tree models'.

**Why keep it.** As a sanity baseline: it shows how much the non-linear models add
(0.79 -> 0.87).

### 4.4 Support vector machine (SVM, RBF kernel): extra model 1

**What it is.** An SVM looks for the boundary with the **widest margin**: the largest
gap between classes, decided only by the hardest examples near the border (the
"support vectors"). With the **RBF kernel** it can draw curved boundaries: the
kernel measures similarity between two stars as exp(-gamma x distance^2), which
implicitly maps the data into a space where a straight boundary works. Four classes
are handled by training one-vs-one SVMs internally.

**Settings** (`scripts/compare_classifiers.py`, `make_svm()`, line 49)

| Setting | Value | Meaning |
|---|---|---|
| `kernel` | `"rbf"` | curved boundaries |
| `C` | 3.0 | penalty for training stars on the wrong side (higher = tighter fit) |
| `gamma` | `"scale"` | kernel width = 1 / (56 x variance of X), set from the data |
| `class_weight` | `"balanced"` | equal weight per class |
| `probability` | `True` | adds internal Platt scaling (5-fold) so it outputs probabilities |

**Preprocessing.** Median imputation + missing indicator + **standardisation**.
Essential: the RBF kernel uses distances, and unscaled, `depth_ppm` (spread
~100,000) would completely drown `ingress_to_total_ratio` (spread ~0.1).

**Results.** 0.80 ± 0.04. Better than logistic regression (curved boundaries help),
worse than the trees. Interesting detail: it keeps **91%** of planets at 5% binary
leakage (better ranking of planets than XGBoost's 88%), but its hard class decisions
are weaker, especially for binaries (0.85) and noise (0.72).

**Why it loses to trees here.** It treats all 56 features as one geometry (every
feature contributes to the distance), cannot use missing values directly, and is
sensitive to outliers and to the choice of C and gamma.

### 4.5 K-nearest neighbours (KNN): extra model 2

**What it is.** No training at all ("lazy learner"). To classify a new star, find the
**k most similar training stars** (smallest Euclidean distance in the 56-feature
space) and let them vote; closer neighbours count more.

**Settings** (`scripts/compare_classifiers.py`, `make_knn()`, line 58): `k = 9`
(about 3% of a 320-star training fold), `weights="distance"` (a neighbour's vote is
weighted by 1/distance), Euclidean distance (sklearn default).

**Preprocessing.** Median imputation + missing indicator + **standardisation**. KNN
is the model most dependent on scaling: distance *is* the model.

**Results.** 0.71 ± 0.08, the weakest and least stable, with the worst log loss
(1.06). It calls **32 noise stars "planet"** and has noise F1 0.58.

**Why it is weakest: the curse of dimensionality.** In 56 dimensions, distances
between points become similar to each other, so "nearest" means less; and every
feature counts equally in the distance, including the weak ones. Trees, by contrast,
pick the useful features automatically.

**Why include it.** It is the simplest possible model, so it shows clearly why
feature choice and model structure matter.

### 4.6 The ranking and the takeaway

**XGBoost (0.87) > random forest (0.84) > SVM (0.80) > logistic regression (0.79) > KNN (0.71)**

Tree ensembles win on small tabular data with mixed scales, missing values and
feature interactions; distance- and line-based models need scaling and still trail.
XGBoost is kept for its accuracy, its calibrated probabilities (best log loss) and
its exact SHAP explanations.

---

## 5. Missing values and imputation

**What.** "Missing" means a feature could not be measured: no catalog radius, no
second eclipse to measure, too few transits. In our table: 738 missing cells, 33 of
56 features affected; most often `duration_ratio_expected` (115 stars), `logg` (104),
`contratio` (91), the secondary-eclipse features (~50).

**What we did.**
- XGBoost: kept as missing (NaN). It learns the best direction for missing values at
  every split; often "missing" itself carries information (no second eclipse found).
- Other models: `SimpleImputer(strategy="median", add_indicator=True)`
  - fills each gap with that feature's **median** (robust to outliers, unlike the mean)
  - adds a 0/1 column per gappy feature saying "this value was missing", so the
    information "missing" is not lost
- Infinite values (from dividing by zero) are first turned into missing
  (`load_dataset()`, `src/triage/train.py:62`).

**Where.** `src/triage/train.py:80` and `:83` (baselines), and
`scripts/compare_classifiers.py:53` and `:61` (SVM, KNN).

---

## 6. Feature scaling, standardisation and normalisation (in depth)

### 6.1 The problem scaling solves

Our features live on wildly different scales (measured on the 400 stars):

| Feature | Min | Median | Max | Spread (std) |
|---|---|---|---|---|
| `teff` (star temperature, K) | 3,334 | 6,073 | 14,600 | 1,755 |
| `depth_ppm` (dip depth) | 19 | 5,214 | 1,152,000 | 104,800 |
| `rms_ppm` (scatter) | 209 | 2,084 | 337,700 | 54,000 |
| `period_d` (days) | 0.50 | 3.16 | 14.9 | 4.7 |
| `ingress_to_total_ratio` (dip shape, 0 = U, 0.5 = V) | 0.017 | 0.31 | 0.50 | 0.12 |
| `skew` | -11.6 | -0.08 | 2.6 | 1.9 |

A model that uses **distances** (KNN, SVM) or **weighted sums with a penalty**
(logistic regression, SVM) would treat `depth_ppm` as about a million times more
important than the dip shape, just because its numbers are bigger. Scaling puts every
feature on a comparable scale so the model, not the unit, decides what matters.

### 6.2 Standardisation (z-score scaling): what we use

For each feature: **z = (x - mean) / standard deviation**

After it, every feature has mean 0 and standard deviation 1. A value of z = +2 means
"2 standard deviations above average", whatever the unit. Example from our data: the
first star's temperature 6,073 K becomes z = -0.13 (just below the average star).

- In code: `sklearn.preprocessing.StandardScaler`
- Used by: logistic regression (`src/triage/train.py:80`), SVM
  (`scripts/compare_classifiers.py:53`), KNN (`scripts/compare_classifiers.py:61`)
- Order inside the pipeline: impute -> standardise -> model
  (`make_pipeline(SimpleImputer(...), StandardScaler(), model)`)

### 6.3 Normalisation (min-max scaling): the alternative

For each feature: **x' = (x - min) / (max - min)**, which squeezes values into [0, 1].

- Simple and keeps zeros, but **very sensitive to outliers**: with `depth_ppm` the
  maximum is 1,152,000 while the median is 5,214, so almost every star would be
  squashed near 0.
- In code it would be `sklearn.preprocessing.MinMaxScaler`. **We do not use it** for
  that reason; standardisation is the better default for features with long tails.
- (A third option, `RobustScaler`, uses the median and interquartile range and is the
  most outlier-proof; standardisation was enough here.)

### 6.4 Who needs scaling, who does not

| Model | Needs scaling? | Why |
|---|---|---|
| XGBoost, random forest | **No** | each split compares one feature to a threshold; changing the unit moves the threshold, not the result |
| Logistic regression | **Yes** | penalised weighted sum; the optimiser converges badly on mixed scales |
| SVM (RBF) | **Yes** | kernel uses distances |
| KNN | **Yes, critically** | distance is the whole model |

### 6.5 The leakage trap, and how the code avoids it

The scaler's mean and standard deviation must be computed **from the training stars
only**, then applied to the test stars. Computing them on all 400 stars first would
let information about the test stars leak into training. Because the scaler sits
*inside* the sklearn `Pipeline`, it is re-fitted on the 320 training stars of each
fold automatically, and only applied (`transform`) to the 80 test stars. The same is
true for the imputer's medians.

### 6.6 Other kinds of "normalisation" in this project (do not confuse them)

| Name | What it does | Where |
|---|---|---|
| **Flux normalisation** (astronomy) | Divides each TESS sector's brightness by its median, so every sector sits at 1.0 (raw levels differ between sectors) | `src/preprocess/preprocessor.py`, `clean_sector()`, lines 109 and 113 |
| **Detrending** | Divides the light curve by a smooth trend (biweight filter), so only short dips remain: flat = flux / trend | `src/preprocess/preprocessor.py`, `detrend()`, line 153 |
| **Probability renormalisation** | After calibration, rescales the 4 probabilities so they sum to 1 | `src/triage/model.py:48`; `_normalise()` in `src/triage/train.py:180` |
| **Confusion-matrix normalisation** | Divides each row by its total, so it shows percentages per true class | `metrics()`, `src/triage/train.py:185` |
| **Per-dip features** | SNR divided by sqrt(number of dips), so a star seen for 10 sectors is judged like one seen for 2 | `src/vetting/transit_features.py`, `per_transit_features()` |

None of these is *feature scaling* for the model; they are signal-processing or
probability steps. The only model-side scaling is the `StandardScaler` above.

---

## 7. Ensembles: bagging, boosting, voting, stacking

**An ensemble combines several models so their individual mistakes cancel.** It only
helps when members are (a) reasonably accurate and (b) wrong on *different* stars.

### 7.1 The types

| Type | How it combines | Example here |
|---|---|---|
| **Bagging** | Many models on random samples of the data, averaged; reduces variance | Random forest (500 trees) |
| **Boosting** | Models built in sequence, each fixing the last one's errors; reduces bias | XGBoost (400 trees) |
| **Hard voting** | Each model names one class; the **majority** wins (ties: we let XGBoost decide) | XGBoost + RF + SVM |
| **Soft voting** | **Average the probabilities**, pick the highest | XGBoost + RF |
| **Weighted soft voting** | Weighted average: P = w1 x P_XGB + w2 x P_RF + w3 x P_SVM, weights sum to 1 | weights chosen by nested CV |
| **Seed bagging** | Same model trained with different random seeds, averaged | 10 x XGBoost |
| **Stacking** | A small "meta-model" learns how to combine the members' outputs | not adopted (overfits on 400 stars) |

Hard vs soft: soft voting uses *how sure* each model is (a 51% vote and a 99% vote
count differently); hard voting throws that away. Soft voting is usually better when
probabilities are reasonable.

### 7.2 What we measured (same 5 folds)

| Method | Macro-F1 | vs XGBoost |
|---|---|---|
| XGBoost alone | **0.871 ± 0.048** | |
| Soft voting XGBoost + RF | 0.866 | -0.005 |
| Soft voting XGBoost + RF + LR | 0.844 | -0.027 |
| Weighted soft voting XGBoost / RF / SVM (nested weights) | 0.864 | -0.008 |
| Seed-bagged XGBoost (10 seeds) | 0.866 | -0.005 |
| Hard voting XGBoost + RF + SVM | 0.852 | -0.019 |
| Hard voting, all 5 models | 0.832 | -0.040 |

### 7.3 Why none of them helped (the evidence)

- **The models share their mistakes.** XGBoost makes 52 errors out of 400; random
  forest also gets 44 of those wrong, SVM 40, logistic regression 41, KNN 47.
- **A hard vote of XGBoost + RF + SVM can fix only 4 XGBoost errors** (both others
  right) **but breaks 12 correct answers** (both others wrong): net -8.
- **40 of XGBoost's 52 errors involve the noise class.** All models read the same 56
  numbers, so they fail on the same hard stars: a data limit, not a model limit.
- **Weighted voting's "best" weights jump around** from fold to fold (from 0.7 / 0.3 / 0
  to 0.1 / 0.9 / 0 for XGBoost / RF / SVM): with about 100 stars per inner validation
  fold, the weight search fits noise.
- Adding weaker members dilutes the strongest one.
- XGBoost alone keeps exact SHAP explanations; an average of different models does not.

**Ensembles that could help** would add *new information*: a neural network on the
raw folded light curve combined with XGBoost on the features (multi-view), a
two-stage "signal vs noise" specialist, or averaging predictions over different
sector pairs of the same star.

### 7.4 Where the code is

- Soft voting (2 ensembles): `src/triage/train.py`, lines 319-322
- Hard voting: `scripts/compare_ensembles.py`, `hard_vote()` (line 58)
- Weighted soft voting: `blend()` (line 69) + the nested weight search in `main()`
  (inner `StratifiedGroupKFold`, line 94)
- Seed bagging: `scripts/compare_ensembles.py`, `main()` (`SEEDS = range(10)`)
- Results: `outputs/ensemble_comparison.md`

---

## 8. Optimisation

### 8.1 Hyperparameter tuning (`scripts/tune_xgboost.py`)

- **Hyperparameters** are settings chosen *before* training (tree depth, number of
  trees, learning rate, ...). **Tuning** tries many settings and keeps the best.
- **Method: RandomizedSearchCV**, 40 random combinations of 8 settings (a full grid
  would be 390,625 combinations; random search covers the space far more cheaply).
- **Nested cross-validation**: inside each outer training fold (320 stars) the 40
  settings are compared with an inner 3-fold grouped CV; the winner is then tested
  once on the 80 untouched stars. 5 x 40 x 3 + 120 = 720 model fits, under a minute.
- **Result: 0.839 ± 0.036 tuned vs 0.871 ± 0.048 hand-chosen.** The "best" settings
  differed in every fold (depth 2, 3, 4, 4, 5): the search was fitting noise. The
  hand-chosen settings (already conservative for small data) are kept.
- Code: `search()` (line 68) builds the RandomizedSearchCV with the inner grouped
  folds; `SPACE` (top of the file) lists the ranges.

### 8.2 Feature selection (`scripts/select_features.py`)

- Question: do we need all 56 features? Unhelpful features can let a small dataset
  learn chance patterns.
- Two methods, both on training stars only: **SHAP ranking** (keep the top k by
  importance, `shap_ranking()`, line 53) and **recursive feature elimination**
  (repeatedly drop the 4 weakest, `rfe_ranking()`, line 62).
- k chosen by inner CV (nested), line 98.
- **Result:** 40 features already score 0.86; 20 features 0.84-0.85; 10 features
  0.81; nested estimate with selection 0.868 vs 0.866 with all 56 (no real gain).
  All features kept. Shown as a chart in the app.

### 8.3 Optimisations already built in

- Probability calibration (section 9.3)
- Planet threshold chosen so only 5% of binaries pass (keeps 88% of planets)
- Speed: parallel downloads, batched catalog queries, 10-minute binning before the
  period search, a capped period grid

### 8.4 The decision rule we used

Adopt a change only if it beats the current model by **more than the fold-to-fold
spread (about ± 0.05)**, measured with **nested** cross-validation. Every experiment
above failed that test; only **more data** (100 -> 400 stars, 0.76 -> 0.87) clearly
helped.

---

## 9. Other things you should know

### 9.1 Data leakage
Any information about the test stars reaching training. Guarded against by: grouped
folds (a star is never on both sides), pipelines that fit the imputer and scaler on
training stars only, cross-validated calibration, nested CV for tuning, demo stars
excluded from training, and labels from catalogs, never from the model.

### 9.2 Overfitting, bias and variance
- **Overfitting**: memorising the 320 training stars instead of learning patterns;
  shows up as good training scores and poor test scores.
- **Bias** = systematic error (model too simple, e.g. logistic regression's straight
  lines). **Variance** = instability (model changes a lot with the data, e.g. KNN's
  ± 0.08).
- Our defences: shallow trees, small learning rate, sampling, L2 regularisation,
  balanced weights, and honest evaluation.

### 9.3 Probability calibration (Platt scaling)
Raw XGBoost probabilities are over-confident. For each class a sigmoid (S-shaped
curve) is fitted that maps raw scores to honest probabilities, so "70% planet" means
about 7 in 10 such stars are planets. It is fitted on out-of-fold predictions and
itself cross-validated. Code: `fit_calibration()` (`src/triage/train.py:139`, the
curve is a 1-feature logistic regression), applied by `apply_calibration()`
(`src/triage/model.py:37`). The app's 50% "needs review" threshold relies on it.

### 9.4 SHAP (explainability)
Splits one prediction into contributions from each feature (Shapley values from game
theory; exact for tree models). Per star: XGBoost's built-in TreeSHAP
(`src/triage/model.py:88`, `pred_contribs=True`), turned into sentences by
`ui/explain.py`. Globally: `shap.TreeExplainer` in `src/triage/train.py:247` and
`scripts/save_evaluation.py:31`. Top features: skew, dip shape (U vs V), star
temperature, duration-vs-expected ratio, autocorrelation strength.

### 9.5 Reproducibility
Seed 42 for folds and models; pinned package versions in `requirements.txt`; every
experiment is a script that writes its results to `outputs/`; the label file's hash is
stored with the model.

### 9.6 Why not deep learning?
A neural network on raw light curves needs thousands of labelled examples; with 400
it would overfit and could not be explained star by star. It is the natural next step
once injected synthetic transits provide more training examples.

---

## 10. Where everything is: quick map

| Topic | File | Function / line |
|---|---|---|
| Load features, handle infinities | `src/triage/train.py` | `load_dataset()` 55, line 62 |
| XGBoost settings | `src/triage/train.py` | `make_xgb()` 67 |
| Random forest, logistic regression, imputer, scaler | `src/triage/train.py` | `make_baselines()` 77 |
| Balanced weights | `src/triage/train.py` | `_fit()` 90 |
| 5 grouped folds | `src/triage/train.py` | `cv_folds()` 100 |
| Out-of-fold predictions | `src/triage/train.py` | `cv_predict()` 113 |
| Metrics | `src/triage/train.py` | `metrics()` 185 |
| Calibration | `src/triage/train.py` | `fit_calibration()` 139, `cross_calibrate()` 158 |
| Soft-voting ensembles | `src/triage/train.py` | lines 319-322 |
| Training run | `src/triage/train.py` / `scripts/train_triage.py` | `train()` 277 |
| SVM, KNN | `scripts/compare_classifiers.py` | `make_svm()` 49, `make_knn()` 58 |
| Hard / weighted voting, seed bagging | `scripts/compare_ensembles.py` | `hard_vote()` 58, `blend()` 69 |
| Hyperparameter tuning | `scripts/tune_xgboost.py` | `SPACE`, `search()` 68 |
| Feature selection | `scripts/select_features.py` | `shap_ranking()` 53, `rfe_ranking()` 62 |
| Prediction, calibration at run time, SHAP per star | `src/triage/model.py` | `predict()` 104, `apply_calibration()` 37, `contributions()` 88 |
| Flux normalisation, detrending | `src/preprocess/preprocessor.py` | `clean_sector()` 87, `detrend()` 130 |
| App's model comparison table | `app.py` (Model evaluation section) reads `models/model_evaluation.json` | built by `scripts/save_evaluation.py` |

Result files: `outputs/triage_results.json`, `outputs/classifier_comparison.md`,
`outputs/ensemble_comparison.md`, `outputs/xgb_tuning.md`,
`outputs/feature_selection.md`, `models/model_evaluation.json`.

---

## 11. Viva quick answers

- **Which model and why?** Calibrated XGBoost: best macro-F1 (0.87), best log loss,
  handles missing values, exact SHAP explanations.
- **Why do SVM and KNN need scaling but XGBoost not?** They use distances / penalised
  weighted sums; trees only compare one feature to a threshold.
- **Standardisation vs normalisation?** z = (x - mean) / std vs (x - min) / (max - min);
  we standardise because our features have extreme outliers that would squash a
  min-max scale.
- **Where is scaling done, and why inside a pipeline?** `StandardScaler` in the
  pipelines (`train.py:80`, `compare_classifiers.py:53/61`); inside a pipeline it is
  fitted on training stars only, so no leakage.
- **Hard vs soft voting?** Majority of predicted classes vs average of probabilities;
  soft keeps each model's confidence.
- **Why did ensembles not help?** The members make the same mistakes (44 of 52
  shared with random forest); a hard vote fixes 4 errors and breaks 12.
- **Why did tuning not help?** About 100 stars per inner validation fold; the search
  fits noise (best settings change every fold); nested CV exposed it.
- **Why nested cross-validation?** If the same stars choose the settings and report
  the score, the score is optimistic.
- **What improved the model?** More data: 100 -> 400 stars, macro-F1 0.76 -> 0.87.
