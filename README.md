# Bank Marketing — Term Deposit Subscription Prediction

Exploratory analysis and classification model for a Portuguese bank's direct-marketing
campaign (UCI *Bank Marketing* dataset, course version: 11,302 contacts, 10 feature
columns): who subscribes to a term deposit, and how well the outcome can be predicted.

**Headline finding:** Ranking customers on pre-call data only, Logistic Regression
reaches **AUC = 0.71**. Calling the top 20% of the ranked list would reach about
**45% of subscribers**, about **2.2× better than calling at random** (lift 2.2).

- **Why 0.90 is only an upper bound:** with call duration (`duration`) included, AUC
  rises to 0.90. Duration is known only after the call has ended, so it cannot be used
  to decide whom to call. The 0.90 result is reported separately as an upper bound,
  not as the model's real performance.
- **Why accuracy is misleading:** only 12.2% of customers subscribed, so a model that
  always predicts "no" already scores about 88% accuracy. Random Forest has the highest
  accuracy (0.89) but misses two-thirds of subscribers (recall 0.34).

The pre-call figures (test set) come from a follow-up analysis that is not yet public.
The notebooks in this repo contain the model with `duration`, i.e. the upper bound.

---

## Notebooks — read in this order

| # | Notebook | What it does | Key output |
|---|---|---|---|
| 00 | `00_EDA.ipynb` | Distributions, dtypes, missing values, class balance | `cleaned_bank_data.csv` (11,260 rows after de-duplication) |
| 03 | `03_Classification.ipynb` | Term-deposit subscription prediction | Logistic Regression with `duration` (upper bound): AUC = 0.90, F1 = 0.53, Recall = 0.80 |

**Stack:** Python · pandas · scikit-learn · matplotlib/seaborn · Google Colab

---

## How to run

Both notebooks load data with a plain `pd.read_csv()` — no upload step, no
API key, nothing to configure. `00_EDA.ipynb` reads `bank_marketing.csv` and
writes `cleaned_bank_data.csv`; `03_Classification.ipynb` reads that output
directly. Open either in Jupyter or Colab and run top to bottom; both CSVs
are already committed in this repo.

---

## Data

`bank_marketing.csv` — a course-specific variant of the UCI *Bank Marketing* dataset
([Moro, Laureano & Cortez, 2011](http://hdl.handle.net/1822/14838)). The full dataset
has 17 possible variables; each student group was assigned a different random subset
of both variables and observations, so this file has only 10 of the 17: `age`,
`marital`, `education`, `housing`, `loan`, `contact`, `month`, `duration`, `pdays`,
`TermDeposit` (target). Variables that are often prominent in other public analyses of
this dataset — `job`, `balance`, `campaign`, `previous`, `poutcome` — were never part
of this group's data, not dropped by choice.

`cleaned_bank_data.csv` is produced by `00_EDA.ipynb` and committed so
`03_Classification.ipynb` runs standalone.

---

## Known limitations

- **`duration` is leaky.** It records how long the call lasted, which is only known
  *after* the contact has happened — so a model using it cannot be deployed to
  decide who to call. `03_Classification.ipynb` keeps it as a feature deliberately,
  for benchmarking consistency with the standard UCI treatment of this dataset, and
  says so explicitly — the reported AUC/F1/Recall should be read as an upper bound on
  what's achievable with post-call information, not as a deployable pre-call model.
- **Class imbalance:** only 12.2% of customers subscribed. Plain accuracy is
  misleading here (a model that always predicts "no" would already score ~88%), so
  AUC, F1, and Recall are reported instead, and are what the model comparison above
  rests on.

---

## Business takeaway

- **Don't pick the model on accuracy alone.** Random Forest's headline accuracy
  (0.89) looks better than Logistic Regression's, but its recall (0.34) means it
  misses two out of three actual subscribers — for a call-prioritization tool, that's
  the wrong trade-off. Logistic Regression's recall of 0.80 catches far more of the
  customers actually worth calling.
- **The model is a retrospective scoring tool, not a pre-call targeting tool.** Its
  strongest signal (`duration`) only exists after the call has already happened, so
  it's useful for analyzing calls that were made, not for deciding who to call next.
  Rebuilt without `duration` (see headline), the pre-call model reaches AUC 0.71 and
  lift 2.2 in the top 20%.

---

## Team and my contribution

Group project (3 students) for the DMA course at WU Vienna.

- The exploratory data analysis (EDA) was done jointly.
- One teammate did the regression, another teammate did the segmentation (clustering),
  and I built the classification model.
- I checked `duration` for data leakage in the data and reported the 0.90 result
  separately as an upper bound.

This repo contains only the EDA notebook and my classification notebook.
