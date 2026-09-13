# Bank Marketing — Campaign Response Prediction & Customer Segmentation

End-to-end analysis of a Portuguese bank's direct-marketing campaign
(UCI *Bank Marketing* dataset, course version: 11,302 contacts, 10 feature columns):
who subscribes to a term deposit, which customer groups exist, and how well the
outcome can be predicted.

**Headline finding:** Logistic Regression predicts term-deposit subscription well
(AUC = 0.90), but the more actionable result is from clustering: one customer segment
(37% of the base) converts at 20.2%, nearly 5x the rate of another segment (28% of the
base) that converts at just 4.3% and has almost never been contacted before —
prioritizing the first group over the second would cut wasted contacts substantially.

---

## Notebooks — read in this order

| # | Notebook | What it does | Key output |
|---|---|---|---|
| 00 | `00_EDA.ipynb` | Distributions, dtypes, missing values, class balance | `cleaned_bank_data.csv` (11,260 rows after de-duplication) |
| 01 | `01_Clustering.ipynb` | Customer segmentation, k = 3 (K-Means) | 3 segments: Warm Responders, Repeat-Contact Skeptics, Cold Prospects (below) |
| 02 | `02_Regression.ipynb` | Predicts `duration` (call length in seconds) | Linear Regression best: R² = 0.18, RMSE = 216s |
| 03 | `03_Classification.ipynb` | Term-deposit subscription prediction | Logistic Regression best: AUC = 0.90, F1 = 0.53 |

**Stack:** Python · pandas · scikit-learn · matplotlib/seaborn · Google Colab

---

## How to run

These notebooks were written in **Google Colab** and use `files.upload()` to load
the CSV interactively.

- **In Colab:** open the notebook, run the first cell, and upload
  `bank_marketing.csv` (or `cleaned_bank_data.csv` for notebooks 01–03) when prompted.
- **Locally (Jupyter):** replace the `files.upload()` cell with
  ```python
  df = pd.read_csv("bank_marketing.csv")
  ```
  Both CSVs are committed in this repo, so no download is needed.

---

## Data

`bank_marketing.csv` — UCI Machine Learning Repository, Moro et al. (2014),
course-specific subset. `cleaned_bank_data.csv` is produced by `00_EDA.ipynb`
and committed so notebooks 01–03 run standalone.

Columns: `age`, `marital`, `education`, `housing`, `loan`, `contact`, `month`,
`duration`, `pdays`, `TermDeposit` (target).

---

## Segments (from `01_Clustering.ipynb`)

K-Means with k=3, chosen on interpretability grounds — silhouette score keeps rising
up to k=10, but beyond k=3 it starts carving out micro-segments of under 50 customers,
too small to be actionable.

| Segment | Size | Subscription rate | Defining trait |
|---|---|---|---|
| Warm Responders | 4,186 (37.2%) | 20.2% | Oldest on average (43), moderate prior-contact rate (19%), mostly no housing loan |
| Repeat-Contact Skeptics | 3,880 (34.5%) | 10.2% | Youngest (39), contacted before most often (33%) yet converts only at half the rate of Warm Responders |
| Cold Prospects | 3,194 (28.4%) | 4.3% | Almost never contacted before (0.3%), contact channel mostly unlogged/unknown |

---

## Known limitations

- **`duration` is leaky.** It records how long the call lasted, which is only known
  *after* the contact has happened — so a model using it cannot be deployed to
  decide who to call. Notebook 03 keeps it as a feature deliberately, for
  benchmarking consistency with the standard UCI treatment of this dataset, and says
  so explicitly — the reported AUC/F1 should be read as an upper bound on what's
  achievable with post-call information, not as a deployable pre-call model.
- **Class imbalance:** only 12.2% of customers subscribed. Plain accuracy is
  misleading here (a model that always predicts "no" would already score ~88%), so
  AUC and F1 are reported instead, and are what the "best model" claims above rest on.
- **Weak cluster separation.** The K-Means silhouette score at k=3 is only 0.10
  (0 = no structure, 1 = perfectly separated) — the three segments are real and
  behaviourally distinct (subscription rate ranges 4.3%–20.2%), but they overlap
  substantially in feature space rather than forming tight, well-isolated groups.
  Read the segment boundaries as directional, not hard cutoffs.

---

## Business takeaway

- **Reallocate outreach toward Warm Responders.** They convert at 20.2% versus 4.3%
  for Cold Prospects — calling five Cold Prospects yields roughly one subscriber where
  calling five Warm Responders yields one; shifting call volume toward the first group
  cuts the number of contacts needed per subscriber substantially.
- **Repeat-Contact Skeptics are being over-called for what they return.** They already
  receive the most repeat contact (33% previously contacted, the highest of the three
  segments) but convert at half the rate of Warm Responders — the current campaign is
  spending disproportionate effort here for below-average return.
- **The classification model (AUC 0.90) is useful for ranking within a call list, not
  for justifying who gets called at all** — its main predictive signal (`duration`)
  is only available *after* the call happens (see limitations below), so it scores
  campaigns retrospectively rather than targeting them prospectively.

---

## Team & contribution

Course team project (Group 7).
**My contribution:** all of the exploratory data analysis (`00_EDA.ipynb`) and the
classification work (`03_Classification.ipynb`).
