# Vehicle Insurance Fraud Detection & AI Risk Monitoring Agent

**Group 9 | Final Project | Streamlit + Python + XGBoost + Generative AI**

**Live application:** https://miaomiao-1225-group-9-app-gutumw.streamlit.app/  
**Source repository:** https://github.com/Miaomiao-1225/Group-9

## 1. Project overview

Insurance fraud can impose financial losses and increase the cost of legitimate claims. This project demonstrates how historical vehicle-insurance claim data can be used to explore fraud patterns, compare machine-learning methods, inspect saved case-level model scores, and generate an AI-assisted risk monitoring report.

The application joins four components:

1. **Data and feature engineering:** clean the historical claims dataset and transform the original fields for model training.
2. **Machine learning:** compare XGBoost models using five class-imbalance handling methods.
3. **Interactive Streamlit dashboard:** explore the data, features, model results, and saved test cases.
4. **AI fraud monitoring agent:** execute Python analytics tools, optionally let a language model select those tools, and synthesize a report based on tool outputs.

The application is a **historical analytical and educational demonstration**, not a production fraud adjudication system, a real-time transaction monitor, or an investment portfolio monitor. A high model score is not proof of fraudulent behavior and should not trigger automatic claim denial.

## 2. Dataset and project scope

The current repository includes `data/vehicle_fraud.csv`. The deployed dataset and saved metadata have the following characteristics:

| Item | Value |
| --- | ---: |
| Claim records | 15,420 |
| Labeled fraud cases | 923 |
| Observed fraud prevalence | 5.99% |
| Original train split | 10,794 |
| Original test split | 4,626 |
| Engineered fields | 39 |
| Encoded model inputs | 120 |

The target column is `FraudFound` (`Yes`/`No`). Examples of claim-related variables include policy type, fault, vehicle category, vehicle make, accident area, and previous claims. The repository's raw-data provenance and permissions should be verified by the group before external redistribution; this README does not assert a source or license that has not been documented.

**Important:** The historical dataset includes labels. The observed fraud prevalence describes the data; it is not a forecast of fraud in future insurance claims.

## 3. Modeling approach

`train_original.py` implements the original cleaning, feature engineering, preprocessing, resampling, and XGBoost workflow. The dashboard intentionally preserves this original workflow and the saved outputs instead of silently replacing the models.

The five compared sampling methods are:

- SMOTE
- Edited Nearest Neighbors (ENN)
- SMOTE-Tomek
- ADASYN
- SMOTE-ENN

The application displays ROC curves, confusion matrices, feature-gain rankings, and the original reported **AUC**, **Recall**, **G-mean**, and **KS** metrics. The original model-selection metadata identifies **ADASYN** as having the highest reported original test AUC (**0.8653**). This must be interpreted alongside the validation limitations below; it is **not** an independently validated generalization claim.

The Top 10 ranking aggregates XGBoost split gain over encoded categories that belong to the same engineered field. **Only the chart is restricted to the Top 10; the original trained models retain their complete feature inputs.** Feature importance is neither a causal effect nor the percentage of fraud explained.

## 4. Streamlit application pages

| Page | What it does |
| --- | --- |
| Dataset overview | Displays claim counts, fraud prevalence, class distribution, and a sample of the dataset. |
| Top 10 features | Displays feature-gain rankings, mappings, and downloadable results. |
| Original model comparison | Compares five sampling methods using saved metrics, ROC curves, a selectable confusion matrix, and sampling summaries. |
| Test case explorer | Displays an existing held-out test row, its true label, and its saved score for a chosen model. |
| AI Risk Assistant | Opens the Week 11 fraud monitoring agent for analytical tool execution and report generation. |
| Validation & sharing | Documents the reproduced workflow and acknowledged limitations. |

The **Test case explorer** does **not** train a model or score newly submitted claims. Its scores come from the saved original test run and are not calibrated real-world probabilities.

## 5. AI fraud monitoring agent

The agent is implemented in `fraud_agent.py` and receives the dataset, saved model metadata, and repository files from `app.py`.

### Analytical tools

| Python tool | Purpose |
| --- | --- |
| `portfolio_overview` | Counts historical claims, fraud labels, and observed prevalence. The name is a legacy tool label, not an investment portfolio function. |
| `fraud_segments` | Compares observed fraud rates across selected business-relevant claim categories, with a minimum group size of 30. |
| `model_metrics` | Reads and reports the saved five-model performance summary, with warnings about interpretation. |
| `top_features` | Retrieves saved feature-gain rankings. |
| `highest_scored_cases` | Lists highest saved test-set model scores for analyst review; these are not calibrated probabilities. |

### Execution modes

**Efficient mode (default):** Python executes the five analytical tools, then sends a compact summary of the calculated evidence to the configured large language model for a written report. This normally requires one LLM request per run.

**Advanced mode (optional):** The LLM first selects analytical tools from a strict allowlist, Python executes them, and the LLM then synthesizes the results. This normally requires two LLM requests per run and is subject to provider availability and rate limits. If AI calls fail, Python-generated fallback findings remain available, and the UI surfaces a diagnostic rather than claiming AI success.

### Output

The generated report includes an executive summary, quantitative risk findings, proposed priorities for **human review**, limitations, and an auditable JSON record of executed Python tool outputs. Use **Download Risk Monitoring Report (.md)** to save the Markdown report.

The language model is instructed not to invent metrics or equate suspicious claims with proven fraud. Still, its narrative requires human verification against the underlying tool outputs.

## 6. How to run locally

Prerequisites: Python compatible with the versions in `requirements.txt`, and access to the files included in this repository.

```bash
git clone https://github.com/Miaomiao-1225/Group-9.git
cd Group-9
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux (alternative)
source .venv/bin/activate
```

Then install dependencies and start Streamlit:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard uses precomputed artifacts in `results/`; retraining is **not required** merely to launch the app. Keep `data/vehicle_fraud.csv`, `results/metadata.json`, and the other saved result files in place.

### AI credentials

The AI integration uses OpenRouter through the OpenAI-compatible API client. Configure `OPENROUTER_API_KEY` and optionally `OPENROUTER_MODEL` in **Streamlit Community Cloud → App settings → Secrets**, or in your local environment. Example Secrets **template** (never commit a real key):

```toml
OPENROUTER_API_KEY = "YOUR_PRIVATE_OPENROUTER_KEY"
OPENROUTER_MODEL = "YOUR_AVAILABLE_OPENROUTER_MODEL_ID"
```

The model must be accessible to your OpenRouter account. Availability, free-tier quotas, and provider rate limits can change. Do not store real keys in public source files or screenshots. The project can still run Python analytics without a functioning AI endpoint, but such fallback output should **not** be presented as an AI-generated report.

## 7. Testing and reproducibility

The repository provides `verify.py`, intended to verify saved preprocessing/model output consistency and selected Streamlit pages. To run it locally, after installing dependencies:

```bash
python verify.py
```

The repository also includes a legacy `unittest` suite in `test/test_pipeline.py`; these tests depend on an older input filename and may be skipped when that file is not present. Skipped tests do not establish that a pipeline has passed.

**Live manual checks for the final presentation:**

1. Open **Dataset overview** and check the dataset count and fraud rate.
2. Open **Top 10 features** and verify that the table and chart load.
3. Open **Original model comparison**, switch the selected method, and inspect the confusion matrix.
4. Open **Test case explorer**, select a test case, and check its saved model score.
5. Open **AI Risk Assistant**, run the default mode, verify that Python tools execute, and download the report.
6. When API availability allows, turn on **Advanced** and confirm that `LLM-selected Python tools` and `LLM synthesized from Python tool results` appear.
7. Review the generated report against the displayed tool results and check how the app behaves when the AI provider returns an error.

**Evidence observed during an October 2026 live walkthrough:** the dashboard was accessible and an Advanced agent run displayed successful LLM tool selection, five Python tool executions, and LLM synthesis. This confirms that those functions ran in that session. It does **not** replace a documented, repeatable end-to-end test of every page, file, or environment.

## 8. Known limitations and responsible use

- **Data leakage in original workflow:** preprocessing was fitted before the original train/test split.
- **Test-informed selection:** the original test set was used for early stopping and to select the reported best sampling method. The displayed test metrics are therefore **not independent, untouched holdout estimates**.
- **Original KS behavior:** the retained KS calculation can separate tied scores.
- **Sampling imbalance:** performance depends on the decision threshold and the trade-off between finding fraud cases and incorrectly flagging genuine claims. Accuracy or AUC alone is insufficient to assess operational suitability.
- **Uncalibrated scores:** saved model scores must not be interpreted directly as real-world probabilities.
- **Association, not causation:** segment fraud rates and split-gain feature rankings cannot establish that an attribute causes fraud.
- **Fairness and privacy:** sensitive attributes and potentially correlated proxies require careful governance. This demonstration does not establish legal compliance or a comprehensive fairness audit.
- **LLM reliability:** generated text can omit nuance or make mistakes; downstream conclusions must be checked against Python outputs. Free hosted models may return empty content, rate-limit errors, or intermittent failures.
- **Business deployment gap:** this project does not ingest live claims, offer claim-management integrations, or support automated claim decisions.

To strengthen the evaluation for a future production-oriented version, use train-only preprocessing, a separate validation set for model selection, a final untouched test set, probability calibration where appropriate, threshold-sensitive metrics, robustness/fairness checks, and operational monitoring.

## 9. Repository guide

| File / directory | Description |
| --- | --- |
| `app.py` | Streamlit application entry point and dashboard pages. |
| `fraud_agent.py` | AI agent, allowed Python analytics tools, API integration, and report generation. |
| `train_original.py` | Original training and five resampling-method comparison workflow. |
| `original_code.py` | Earlier original functions retained for comparison. |
| `pipeline.py`, `main.py` | Additional pipeline implementation and launcher; not the main Streamlit entry point. |
| `verify.py` | Intended saved-output consistency checks and Streamlit page smoke tests. |
| `test/test_pipeline.py` | Legacy unit tests for pipeline components. |
| `data/vehicle_fraud.csv` | Historical vehicle-insurance claims dataset. |
| `results/` | Saved metrics, model files, selected test predictions, feature rankings, and metadata. |
| `requirements.txt` | Python dependencies for the application. |

## 10. Final presentation suggestion

A concise demonstration can follow this order: **business problem → historical dataset and imbalance → XGBoost method comparison → dashboard exploration → Advanced AI tool selection → Python evidence → AI-written monitoring report → limitations and human oversight**. The instructor's required scenario and presentation format should take precedence if they differ from this demonstration sequence.
