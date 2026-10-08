# 🧠 Early Stroke Risk Estimator

An interactive ML app that estimates a person's stroke risk from basic health and lifestyle inputs, and shows which factors drove the estimate.

> **Educational demo only. Not a medical device or a diagnosis.**

**Live app:** https://early-stroke-prediction-medical.streamlit.app/

## How it works
- **Data:** Kaggle Healthcare Stroke Dataset (5,109 records, ~4.9% strokes), in `data/`.
- **Model:** logistic regression pipeline (median imputation, scaling, one-hot encoding). It's deliberately not class-reweighted, so probabilities stay calibrated to the real base rate.
- **Performance:** ~0.84 cross-validated ROC AUC.
- **Explainability:** per-feature log-odds contributions for every prediction.
- The model trains in a couple of seconds on first load, so there's no pickle file to go out of sync with library versions.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (free)
1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), click **New app**, pick this repo, branch `main`, main file `app.py`.
3. Deploy.

## Structure
| File | Purpose |
|---|---|
| `app.py` | Streamlit UI |
| `model.py` | data loading, training, risk banding, explanations |
| `data/` | dataset |
