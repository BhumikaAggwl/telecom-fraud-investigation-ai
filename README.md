```markdown
# Intelligent Telecom Fraud Investigation System
### XGBoost + LangChain Multi-Agent AI

## Overview
A fraud detection and investigation pipeline for telecom call data. An XGBoost
classifier flags suspicious activity, and a LangChain multi-agent system
(Risk Assessment → Investigation → Report Generation) automates analyst-style
investigation for flagged cases — mimicking how a fraud analytics team would
triage and document cases in production.

This project was built independently to explore problems adjacent to work I
did during my AI internship at Vodafone. The internal codebase and data are
confidential; this repository uses a synthetic telecom dataset built to
reflect realistic, overlapping (non-trivially-separable) fraud patterns.

## Architecture
```
Incoming CSV batch
      │
      ▼
Schema Validation → Cleaning
      │
      ▼
XGBoost Fraud Classifier
      │
      ▼
Business Rule Threshold (high-risk / uncertain → agents)
      │
      ▼
LangChain Multi-Agent Pipeline
  Risk Agent → Investigation Agent → Report Agent
      │
      ▼
Investigation Report (.md) + Batch Summary (.csv) + Logs
```

## Why this design
- **XGBoost, not deep learning**: tabular fraud data with engineered features
  performs better with tree ensembles; simpler to tune, explain, and deploy.
- **LLM agents for reasoning, not classification**: LLMs are weak structured-data
  classifiers. The ML model predicts; the agents interpret, investigate, and
  document — playing to each technology's strength.
- **Selective agent routing**: only high-risk/uncertain cases are sent to the
  LLM, keeping inference cost and latency low — a real cost-control pattern
  used in production fraud systems.
- **Class-weighted XGBoost over plain logistic regression baseline**: the
  baseline had strong precision (92.9%) but very poor recall (10.6%) — it
  played it safe and missed most real fraud. XGBoost with `scale_pos_weight`
  raised recall to 37.9% and nearly doubled F1-score (0.19 → 0.37), at the
  cost of some precision — a deliberate tradeoff favoring catching more fraud
  over avoiding false positives, appropriate for a first-pass triage system
  where flagged cases get human/agent review before action.

## Dataset
Synthetic telecom call records (10,000 rows, ~8% fraud rate) generated with
deliberately overlapping class distributions and 12% label noise to avoid an
unrealistically easy classification task. Features include call duration,
call frequency, night-call ratio, SIM age, recharge amount, unique contacts,
location/device changes, and in/out call ratio.

## Model Performance

| Metric | Baseline (Logistic Regression) | XGBoost (class-weighted) |
|---|---|---|
| Accuracy | 0.8335 | 0.7615 |
| Precision | 0.9286 | 0.3608 |
| Recall | 0.1057 | 0.3794 |
| F1 Score | 0.1898 | 0.3699 |
| ROC AUC | 0.6455 | 0.6246 |

The baseline model's high accuracy is misleading — it comes from predicting
"not fraud" most of the time on an imbalanced dataset. XGBoost trades some
accuracy/precision for dramatically better fraud recall, which is the metric
that matters most in fraud detection.

## Multi-Agent Pipeline
1. **Risk Assessment Agent** — converts fraud probability + features into a
   Low/Medium/High risk level with reasoning.
2. **Investigation Agent** — infers likely fraud type (SIM box, robocalling,
   spam campaign, etc.) and recommends investigation steps.
3. **Report Agent** — synthesizes both into a structured markdown
   investigation report (fraud score, risk level, evidence, recommended
   actions, final recommendation).

## Batch Processing
Drop a CSV into `data/incoming_calls/`. `batch_processor.py` will:
1. Detect the new file
2. Validate schema
3. Clean and impute missing values
4. Run XGBoost prediction
5. Apply business-rule thresholds
6. Route only high-risk/uncertain cases to the agent pipeline
7. Generate investigation reports (`data/reports/`)
8. Write a running batch summary (`data/reports/batch_summary.csv`)
9. Log every step (`logs/pipeline.log`)
10. Archive the processed file

## Project Structure
```
Fraud-Detection-AI/
├── data/
│   ├── incoming_calls/     # drop new batch CSVs here
│   ├── processed/          # cleaned training dataset
│   └── reports/            # generated investigation reports + summary
├── models/                 # trained model + metrics
├── agents/                 # risk, investigation, report agents
├── ml/                     # train, predict, baseline scripts
├── utils/                  # data generation, validation, logging
├── main.py                 # single-run demo pipeline
├── batch_processor.py      # automated batch pipeline
└── requirements.txt
```

## Tech Stack
Python, XGBoost, scikit-learn, LangChain, Groq (Llama 3.1 8B Instant), pandas, joblib

## Limitations & Future Improvements
- Synthetic dataset — real-world deployment would need validation against
  actual telecom call detail records (CDRs).
- Threshold values (0.4 / 0.6) are illustrative business rules, not tuned
  against a real cost-of-error analysis.
- No SHAP/feature-importance explainability layer yet — a natural next step
  for model transparency.
- Agents currently run sequentially; could be parallelized for latency.
- No persistent database — reports/summaries are file-based for simplicity.
```

Say **"done"** — and honestly, at this point the project is complete and genuinely solid. Push it to GitHub whenever you're ready.