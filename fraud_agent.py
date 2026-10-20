"""Week 11 AI fraud-monitoring agent for Group 9's existing Streamlit app.

Keeps the original trained models, dataset and dashboard unchanged. Provides
safe Python analytics tools, optional LLM planning/synthesis via OpenRouter,
and a downloadable evidence-grounded monitoring report.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st


TOOL_NAMES = (
    "portfolio_overview",
    "fraud_segments",
    "model_metrics",
    "top_features",
    "highest_scored_cases",
)

TOOL_DESCRIPTION = {
    "portfolio_overview": "Compute the dataset size, class balance and observed fraud rate.",
    "fraud_segments": "Find observed claim segments with comparatively high fraud rates.",
    "model_metrics": "Read original five-model performance metrics and limitations.",
    "top_features": "Read the saved Top 10 engineered-feature ranking.",
    "highest_scored_cases": "Find test-set cases with the highest saved model scores.",
}


SYSTEM_INSTRUCTIONS = """You are an insurance-fraud ANALYTICAL assistant.
Use ONLY numeric evidence returned by the available Python analytical tools.
The model scores are uncalibrated scores, NOT actual probabilities of fraud.
Observed group fraud rates are descriptive correlations, NOT causation.
A suspicious claim is NOT proof of wrongdoing. Do not recommend automatic claim denial.
Do not invent missing numbers, test results, fairness findings or data sources.
When a file is in Git merge conflict, state that metric is unavailable.
There is preprocessing leakage and the original test set was used for early
stopping/model selection: the original metrics are not independent generalization estimates.
No legal, personal financial, or claim-settlement advice.
Avoid using protected personal characteristics for targeting people.
Keep the English report concise, specific, and suitable for a classroom demo.
"""


def _setting(name: str, default: str = "") -> str:
    """Read cloud Secrets or local environment, without hardcoded credentials."""
    try:
        secret = st.secrets.get(name)
        if secret:
            return str(secret)
    except (FileNotFoundError, KeyError, AttributeError):
        pass
    return os.getenv(name, default)


def _load_result(root: Path, relative: str) -> tuple[pd.DataFrame | None, str | None]:
    """Reject unmerged CSVs; silently choosing one side would misreport results."""
    path = root / relative
    if not path.is_file():
        return None, f"Missing result file: {relative}"
    try:
        raw = path.read_text(encoding="utf-8-sig")
        if any(marker in raw for marker in ("<<<<<<< HEAD", "=======\n", ">>>>>>> ")):
            return None, f"Git merge-conflict markers detected in {relative}. Resolve and validate this file first."
        return pd.read_csv(path), None
    except Exception as exc:
        return None, f"Cannot load {relative}: {type(exc).__name__}"


def portfolio_overview(data: pd.DataFrame, root: Path, meta: dict) -> dict:
    if "FraudFound" not in data.columns:
        return {"status": "unavailable", "reason": "FraudFound column is missing"}
    labels = data["FraudFound"].astype(str).str.strip().str.lower()
    fraud = int((labels == "yes").sum())
    total = len(data)
    return {
        "status": "ok",
        "records": total,
        "observed_fraud_cases": fraud,
        "observed_nonfraud_cases": int(total - fraud),
        "observed_fraud_rate_pct": round(100 * fraud / total, 2) if total else None,
        "engineered_field_count": meta.get("engineered_fields"),
        "encoded_model_inputs": meta.get("encoded_fields"),
        "meaning": "Observed label prevalence in the supplied historical dataset, not a live fraud forecast.",
    }


def fraud_segments(data: pd.DataFrame, root: Path, meta: dict) -> dict:
    if "FraudFound" not in data.columns:
        return {"status": "unavailable", "reason": "FraudFound column is missing"}
    target = data["FraudFound"].astype(str).str.strip().str.lower().eq("yes").astype(int)
    baseline = float(target.mean()) if len(data) else 0.0
    results = []
    # Avoid personal protected characteristics; choose business process variables.
    for field in ["BasePolicy", "Fault", "AccidentArea", "VehicleCategory", "PolicyType", "Make"]:
        if field not in data.columns:
            continue
        frame = pd.DataFrame({"segment": data[field].fillna("Unknown").astype(str), "is_fraud": target})
        aggregate = frame.groupby("segment", dropna=False).is_fraud.agg(["size", "sum", "mean"])
        aggregate = aggregate[aggregate["size"] >= 30].sort_values("mean", ascending=False).head(3)
        for segment, row in aggregate.iterrows():
            results.append({
                "field": field,
                "segment": str(segment)[:80],
                "claims": int(row["size"]),
                "labeled_fraud": int(row["sum"]),
                "observed_fraud_rate_pct": round(float(row["mean"]) * 100, 2),
                "relative_to_dataset": round(float(row["mean"]) / baseline, 2) if baseline else None,
            })
    results.sort(key=lambda x: (x["observed_fraud_rate_pct"], x["claims"]), reverse=True)
    return {
        "status": "ok",
        "min_claims_per_segment": 30,
        "reference_fraud_rate_pct": round(100 * baseline, 2),
        "segments": results[:12],
        "interpretation": "Historical associations only; group rates do not establish causality or justify adverse individual decisions.",
    }


def model_metrics(data: pd.DataFrame, root: Path, meta: dict) -> dict:
    frame, error = _load_result(root, "results/original_metrics.csv")
    if error:
        return {"status": "unavailable", "reason": error}
    needed = {"Method", "AUC", "Recall", "G_mean", "KS"}
    if not needed.issubset(frame.columns):
        return {"status": "unavailable", "reason": "Model metrics columns are invalid"}
    rows = []
    for _, record in frame.iterrows():
        try:
            rows.append({"method": str(record.Method), "AUC": round(float(record.AUC), 4),
                         "recall": round(float(record.Recall), 4), "G_mean": round(float(record.G_mean), 4),
                         "KS": round(float(record.KS), 4)})
        except (ValueError, TypeError):
            return {"status": "unavailable", "reason": "Invalid numeric values in original_metrics.csv"}
    return {
        "status": "ok", "models": rows,
        "caution": "Original preprocessing occurred before splitting and original test labels were used for early stopping / model choice; NOT an independent test estimate.",
    }


def top_features(data: pd.DataFrame, root: Path, meta: dict) -> dict:
    frame, error = _load_result(root, "results/top10_features.csv")
    if error:
        return {"status": "unavailable", "reason": error}
    if not {"Feature", "Gain_share"}.issubset(frame.columns):
        return {"status": "unavailable", "reason": "Top 10 features columns are invalid"}
    features = []
    for _, record in frame.head(10).iterrows():
        try:
            features.append({"feature": str(record.Feature), "gain_share_pct": round(100 * float(record.Gain_share), 2)})
        except (ValueError, TypeError):
            return {"status": "unavailable", "reason": "Invalid gain values"}
    return {"status": "ok", "features": features,
            "caution": "Split-gain importance is not causal and is not a fraction of fraud explained."}


def highest_scored_cases(data: pd.DataFrame, root: Path, meta: dict) -> dict:
    frame, error = _load_result(root, "results/test_predictions.csv")
    if error:
        return {"status": "unavailable", "reason": error}
    method = meta.get("best_method_by_original_test_auc", "ADASYN")
    if not {"Row_index", "Actual", method}.issubset(frame.columns):
        return {"status": "unavailable", "reason": "Test prediction columns are missing"}
    try:
        scores = pd.to_numeric(frame[method], errors="coerce")
        if not scores.notna().all() or not scores.between(0, 1).all():
            return {"status": "unavailable", "reason": "Invalid saved test scores"}
        ranked = frame.assign(score=scores).sort_values("score", ascending=False).head(10)
        cases = [{"dataset_row_index": int(row.Row_index), "score": round(float(row.score), 4),
                  "actual_label": str(row.Actual)} for _, row in ranked.iterrows()]
    except (ValueError, TypeError):
        return {"status": "unavailable", "reason": "Could not rank saved scores"}
    return {"status": "ok", "selected_model": method, "highest_scored_test_cases": cases,
            "caution": "These are historical test cases, not live transactions. Score is not a calibrated probability."}


ANALYTICS = {
    "portfolio_overview": portfolio_overview,
    "fraud_segments": fraud_segments,
    "model_metrics": model_metrics,
    "top_features": top_features,
    "highest_scored_cases": highest_scored_cases,
}


def _client():
    key = _setting("OPENROUTER_API_KEY")
    if not key:
        return None
    from openai import OpenAI
    return OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1", timeout=40, max_retries=1)


def _answer(client, messages, max_tokens=550) -> str:
    response = client.chat.completions.create(
        model=_setting("OPENROUTER_MODEL", "openrouter/free"),
        messages=messages,
        temperature=0.2,
        max_tokens=max_tokens,
        extra_headers={
            "HTTP-Referer": "https://miaomiao-1225-group-9-app-gutumw.streamlit.app",
            "X-OpenRouter-Title": "Group9-Fraud-Monitor",
        },
    )
    return (response.choices[0].message.content or "").strip()


def _choose_tools(client, question: str) -> tuple[list[str], str]:
    """LLM tool planning with an allowlist; safe fallback if provider can't answer."""
    default = list(TOOL_NAMES)
    if not client:
        return default, "Rule-based fallback (API key unavailable)"
    try:
        plan_prompt = (
            "Choose Python analytical tools for this user monitoring task. "
            "Return ONLY a JSON object like {\"tools\":[\"portfolio_overview\",\"fraud_segments\"]}. "
            "Allowed tools and descriptions: " + json.dumps(TOOL_DESCRIPTION) +
            "\nUser task: " + question[:600]
        )
        reply = _answer(client, [{"role": "system", "content": "You are an analytics tool planner. Return JSON, nothing else."},
                                 {"role": "user", "content": plan_prompt}], max_tokens=200)
        match = re.search(r"\{[\s\S]*\}", reply)
        plan = json.loads(match.group(0)) if match else {}
        selected = [t for t in plan.get("tools", []) if t in ANALYTICS]
        selected = list(dict.fromkeys(selected))
        # Always include an overall control total for grounding and comparison.
        if "portfolio_overview" not in selected:
            selected.insert(0, "portfolio_overview")
        return selected or default, "LLM-selected Python tools"
    except Exception:
        return default, "Rule-based fallback (tool planning failed)"


def _fallback_summary(outputs: dict) -> str:
    overview = outputs.get("portfolio_overview", {})
    parts = ["### Risk findings based on actual Python calculations"]
    if overview.get("status") == "ok":
        parts.append(f"- Dataset: {overview['records']:,} claims; {overview['observed_fraud_cases']:,} fraud labels "
                     f"({overview['observed_fraud_rate_pct']:.2f}% observed fraud rate).")
    segments = outputs.get("fraud_segments", {})
    if segments.get("status") == "ok" and segments.get("segments"):
        first = segments["segments"][0]
        parts.append(f"- Highest displayed historical segment: {first['field']} = {first['segment']} "
                     f"({first['observed_fraud_rate_pct']:.2f}% across {first['claims']} cases; descriptive, not causal).")
    scores = outputs.get("highest_scored_cases", {})
    if scores.get("status") == "ok" and scores.get("highest_scored_test_cases"):
        first = scores["highest_scored_test_cases"][0]
        parts.append(f"- Highest saved {scores['selected_model']} test score: {first['score']:.4f}, "
                     f"dataset row {first['dataset_row_index']} (not a calibrated probability).")
    bad = [x["reason"] for x in outputs.values() if x.get("status") != "ok"]
    if bad:
        parts.append("- Not available because: " + "; ".join(bad))
    parts.append("- Recommendations: check the flagged case segments and data integrity with a human reviewer; "
                 "resolve merge conflicts before comparing model metrics. Do not automatically reject claims.")
    parts.append("- Limitations: preprocessing leakage and test-informed model selection make original model "
                 "performance unsuitable as a fully independent validation estimate.")
    return "\n".join(parts)


def _run_monitor(question: str, data: pd.DataFrame, root: Path, meta: dict):
    client = _client()
    selected, plan_mode = _choose_tools(client, question)
    outputs = {name: ANALYTICS[name](data, root, meta) for name in selected}
    narrative = _fallback_summary(outputs)
    synthesis_mode = "Python-generated summary (no AI connection)"
    if client:
        try:
            response = _answer(client, [
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": (
                    "Produce a succinct risk monitoring report in Markdown. Provide: executive summary, "
                    "quantitative findings, specific actions, and important limitations. "
                    "Say explicitly when numeric metrics are unavailable. Do not invent values. "
                    "\nUser's task: " + question[:1200] +
                    "\nPython tool outputs (trusted calculations):\n" + json.dumps(outputs, ensure_ascii=False, default=str)
                )},
            ], max_tokens=950)
            if response:
                narrative, synthesis_mode = response, "LLM synthesized from Python tool results"
        except Exception as exc:
            synthesis_mode = f"Python summary (AI synthesis failed: {type(exc).__name__})"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    report = (
        "# Group 9 — Vehicle Insurance Fraud Risk Monitoring Report\n\n"
        f"Generated: {timestamp}\n\nTask: {question}\n\n"
        f"Tool selection: {plan_mode}\n\nTools executed: {', '.join(selected)}\n\n"
        + narrative + "\n\n## Auditable analytical evidence (JSON)\n\n```json\n"
        + json.dumps(outputs, indent=2, ensure_ascii=False, default=str) + "\n```\n"
    )
    return selected, plan_mode, outputs, narrative, synthesis_mode, report


def render_fraud_agent(meta: dict, root: Path, data: pd.DataFrame):
    st.subheader("🤖 AI Fraud Monitoring Agent — Week 11")
    st.write("Select analytical tools, run actual Python calculations, flag risks, and produce a downloadable report.")
    st.caption("This monitors the historical vehicle-insurance fraud dataset; it is not an investment portfolio monitor.")
    if not _setting("OPENROUTER_API_KEY"):
        st.warning("OPENROUTER_API_KEY is missing. Python analytical tools still work, but AI planning and written synthesis are disabled. Add the key to Streamlit Cloud → App Settings → Secrets.")
    question = st.text_area("Monitoring objective", value=(
        "Analyze historical vehicle insurance fraud risks. Compare claim segments, "
        "review available model evidence, and propose human-review priorities."
    ), height=95)
    if st.button("▶ Run Fraud Risk Agent", type="primary", disabled=not question.strip()):
        with st.spinner("Executing analytical tools and generating report..."):
            st.session_state.group9_agent_result = _run_monitor(question.strip(), data, root, meta)
    if "group9_agent_result" not in st.session_state:
        st.info("Click 'Run Fraud Risk Agent' to start. This will not retrain or modify your existing models.")
        return
    selected, plan_mode, outputs, narrative, synthesis_mode, report = st.session_state.group9_agent_result
    st.success("Agent analysis completed")
    st.write("**Tool selection:**", plan_mode)
    st.write("**Python tools executed:**", ", ".join(selected))
    st.write("**Narrative mode:**", synthesis_mode)
    overview = outputs.get("portfolio_overview", {})
    if overview.get("status") == "ok":
        a, b, c = st.columns(3)
        a.metric("Historical claims", f"{overview['records']:,}")
        b.metric("Labeled fraud cases", f"{overview['observed_fraud_cases']:,}")
        c.metric("Observed fraud rate", f"{overview['observed_fraud_rate_pct']:.2f}%")
    problems = [record["reason"] for record in outputs.values() if record.get("status") != "ok"]
    for issue in problems:
        st.warning(issue)
    st.markdown(narrative)
    with st.expander("Show actual Python tool outputs / Auditable evidence"):
        st.json(outputs)
    st.download_button("⬇ Download Risk Monitoring Report (.md)", report,
                       file_name="group9_fraud_risk_monitoring_report.md", mime="text/markdown")
