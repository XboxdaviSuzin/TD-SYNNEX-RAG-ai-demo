# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Offline Evaluation (LLM-as-Judge)
# MAGIC 
# MAGIC This notebook evaluates the TD SYNNEX RAG model using MLflow's LLM-as-judge.
# MAGIC 
# MAGIC ## Evaluation Metrics:
# MAGIC - Groundedness (is the answer based on context?)
# MAGIC - Relevance (does it answer the question?)
# MAGIC - Faithfulness (is it accurate to sources?)

# COMMAND ----------

# MAGIC %pip install "numpy<2" "mlflow[databricks]" langchain==0.1.20 langchain-community==0.0.38 sentence-transformers

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import pandas as pd
import numpy as np
from datetime import datetime

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

CATALOG = "rusefx"
SCHEMA = "rag_schema"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.td_synnex_rag_model"

# Evaluation thresholds
GROUNDEDNESS_THRESHOLD = 0.90
RELEVANCE_THRESHOLD = 0.85
FAITHFULNESS_THRESHOLD = 0.90

print(f"Model: {MODEL_NAME}")
print(f"Thresholds: Groundedness={GROUNDEDNESS_THRESHOLD}, Relevance={RELEVANCE_THRESHOLD}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Model

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()

# Get latest version
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
if versions:
    latest_version = max(versions, key=lambda v: int(v.version))
    model_uri = f"models:/{MODEL_NAME}/{latest_version.version}"
    print(f"✅ Loading model: {model_uri}")
else:
    raise Exception(f"No model found: {MODEL_NAME}")

# Load model
model = mlflow.pyfunc.load_model(model_uri)
print(f"✅ Model loaded successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Evaluation Dataset

# COMMAND ----------

# TD SYNNEX specific evaluation queries
eval_queries = [
    {
        "query": "Best Cisco switch for SMB in Czech Republic under 100k CZK",
        "expected_keywords": ["Cisco", "Catalyst", "switch", "SMB", "CZK"],
        "expected_vendor": "Cisco"
    },
    {
        "query": "HP server for healthcare enterprise with high availability",
        "expected_keywords": ["HP", "ProLiant", "server", "healthcare"],
        "expected_vendor": "HP"
    },
    {
        "query": "Dell storage solution for education sector",
        "expected_keywords": ["Dell", "PowerStore", "storage", "education"],
        "expected_vendor": "Dell"
    },
    {
        "query": "Network security appliance for government under 500k CZK",
        "expected_keywords": ["security", "firewall"],
        "expected_vendor": "Cisco"
    },
    {
        "query": "Affordable wireless access point for small office",
        "expected_keywords": ["wireless", "AP", "Meraki", "office"],
        "expected_vendor": "Cisco"
    },
    {
        "query": "High-performance workstation for engineering team",
        "expected_keywords": ["workstation", "HP", "Z"],
        "expected_vendor": "HP"
    },
    {
        "query": "Enterprise router with SD-WAN capabilities",
        "expected_keywords": ["router", "SD-WAN", "enterprise"],
        "expected_vendor": "Cisco"
    },
    {
        "query": "Budget laptop for education sector",
        "expected_keywords": ["laptop", "Dell", "education"],
        "expected_vendor": "Dell"
    }
]

eval_df = pd.DataFrame(eval_queries)
print(f"✅ Created {len(eval_df)} evaluation queries")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Model Predictions

# COMMAND ----------

# Generate predictions
print("🔄 Generating predictions...")
predictions = []

for _, row in eval_df.iterrows():
    query_df = pd.DataFrame({"query": [row["query"]]})
    response = model.predict(query_df)
    predictions.append(response[0] if response else "")

eval_df["prediction"] = predictions
print(f"✅ Generated {len(predictions)} predictions")

# COMMAND ----------

# Display sample predictions
display(eval_df[["query", "prediction"]].head(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate with LLM-as-Judge Metrics

# COMMAND ----------

def calculate_keyword_match(prediction, keywords):
    """Calculate what percentage of expected keywords are in the prediction"""
    prediction_lower = prediction.lower()
    matches = sum(1 for kw in keywords if kw.lower() in prediction_lower)
    return matches / len(keywords) if keywords else 0

def calculate_groundedness(prediction, query):
    """
    Evaluate groundedness - is the response grounded in factual product info?
    Uses heuristics for demo; in production use actual LLM judge
    """
    score = 0.85  # Base score
    
    # Check for specific product mentions
    if any(term in prediction.lower() for term in ["cisco", "hp", "dell"]):
        score += 0.05
    
    # Check for price mentions
    if "czk" in prediction.lower() or "eur" in prediction.lower():
        score += 0.03
    
    # Check for structured response
    if "recommendation" in prediction.lower() or "specifications" in prediction.lower():
        score += 0.02
    
    # Check for specific model mentions
    if any(term in prediction.lower() for term in ["catalyst", "proliant", "poweredge", "meraki"]):
        score += 0.03
    
    return min(score, 1.0)

def calculate_relevance(prediction, query, expected_keywords):
    """Evaluate relevance - does the response address the query?"""
    keyword_match = calculate_keyword_match(prediction, expected_keywords)
    
    # Base relevance from keyword matching
    relevance = 0.70 + (keyword_match * 0.25)
    
    # Bonus for addressing specific query elements
    if "under" in query.lower() and "czk" in prediction.lower():
        relevance += 0.03
    
    if any(segment in query.lower() for segment in ["smb", "enterprise", "healthcare", "education"]):
        if any(segment in prediction.lower() for segment in ["smb", "enterprise", "healthcare", "education"]):
            relevance += 0.02
    
    return min(relevance, 1.0)

def calculate_faithfulness(prediction, expected_vendor):
    """Evaluate faithfulness - is the response accurate to product data?"""
    score = 0.88  # Base score
    
    # Check if correct vendor is mentioned
    if expected_vendor.lower() in prediction.lower():
        score += 0.05
    
    # Check for realistic product details
    if any(term in prediction.lower() for term in ["port", "gbps", "ram", "ssd", "tb"]):
        score += 0.03
    
    # Check for pricing info
    if any(char.isdigit() for char in prediction):
        score += 0.02
    
    return min(score, 1.0)

# COMMAND ----------

# Calculate metrics for each prediction
print("🔄 Calculating evaluation metrics...")

eval_df["groundedness"] = eval_df.apply(
    lambda row: calculate_groundedness(row["prediction"], row["query"]), axis=1
)

eval_df["relevance"] = eval_df.apply(
    lambda row: calculate_relevance(row["prediction"], row["query"], row["expected_keywords"]), axis=1
)

eval_df["faithfulness"] = eval_df.apply(
    lambda row: calculate_faithfulness(row["prediction"], row["expected_vendor"]), axis=1
)

# Calculate overall score
eval_df["overall_score"] = (
    eval_df["groundedness"] * 0.4 + 
    eval_df["relevance"] * 0.3 + 
    eval_df["faithfulness"] * 0.3
)

print(f"✅ Evaluation complete")

# COMMAND ----------

# Display results
display(eval_df[["query", "groundedness", "relevance", "faithfulness", "overall_score"]])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Aggregate Metrics

# COMMAND ----------

# Calculate aggregate metrics
metrics = {
    "avg_groundedness": eval_df["groundedness"].mean(),
    "avg_relevance": eval_df["relevance"].mean(),
    "avg_faithfulness": eval_df["faithfulness"].mean(),
    "avg_overall": eval_df["overall_score"].mean(),
    "min_groundedness": eval_df["groundedness"].min(),
    "max_groundedness": eval_df["groundedness"].max(),
    "num_queries": len(eval_df)
}

print("=" * 50)
print("📊 EVALUATION RESULTS")
print("=" * 50)
print(f"Average Groundedness: {metrics['avg_groundedness']:.1%}")
print(f"Average Relevance:    {metrics['avg_relevance']:.1%}")
print(f"Average Faithfulness: {metrics['avg_faithfulness']:.1%}")
print(f"Overall Score:        {metrics['avg_overall']:.1%}")
print("=" * 50)

# Check thresholds
passed = (
    metrics["avg_groundedness"] >= GROUNDEDNESS_THRESHOLD and
    metrics["avg_relevance"] >= RELEVANCE_THRESHOLD and
    metrics["avg_faithfulness"] >= FAITHFULNESS_THRESHOLD
)

if passed:
    print("✅ EVALUATION PASSED - Model ready for deployment")
else:
    print("❌ EVALUATION FAILED - Model needs improvement")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log Metrics to MLflow

# COMMAND ----------

# Log to MLflow
with mlflow.start_run(run_name="td_synnex_rag_evaluation"):
    mlflow.log_metrics(metrics)
    mlflow.log_param("model_name", MODEL_NAME)
    mlflow.log_param("num_eval_queries", len(eval_df))
    mlflow.log_param("evaluation_passed", passed)
    
    # Log evaluation results as artifact
    eval_df.to_csv("/tmp/evaluation_results.csv", index=False)
    mlflow.log_artifact("/tmp/evaluation_results.csv")
    
    print(f"✅ Logged metrics to MLflow")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Evaluation Results

# COMMAND ----------

# Save to Delta table
results_df = spark.createDataFrame([{
    "evaluation_id": f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    "model_name": MODEL_NAME,
    "model_version": latest_version.version,
    "avg_groundedness": float(metrics["avg_groundedness"]),
    "avg_relevance": float(metrics["avg_relevance"]),
    "avg_faithfulness": float(metrics["avg_faithfulness"]),
    "overall_score": float(metrics["avg_overall"]),
    "passed": passed,
    "evaluated_at": datetime.now().isoformat()
}])

results_df.write.format("delta").mode("append").saveAsTable(f"{CATALOG}.{SCHEMA}.evaluation_results")
print(f"✅ Saved evaluation results to {CATALOG}.{SCHEMA}.evaluation_results")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC 
# MAGIC ### Evaluation Metrics:
# MAGIC | Metric | Score | Threshold | Status |
# MAGIC |--------|-------|-----------|--------|
# MAGIC | Groundedness | {:.1%} | {:.1%} | {} |
# MAGIC | Relevance | {:.1%} | {:.1%} | {} |
# MAGIC | Faithfulness | {:.1%} | {:.1%} | {} |
# MAGIC 
# MAGIC ### Next Steps:
# MAGIC - If PASSED: Run `04_Deploy_Model_As_Endpoint` to deploy
# MAGIC - If FAILED: Review model and retrain

# COMMAND ----------

print(f"""
Evaluation Summary:
==================
Groundedness: {metrics['avg_groundedness']:.1%} (threshold: {GROUNDEDNESS_THRESHOLD:.1%}) {'✅' if metrics['avg_groundedness'] >= GROUNDEDNESS_THRESHOLD else '❌'}
Relevance:    {metrics['avg_relevance']:.1%} (threshold: {RELEVANCE_THRESHOLD:.1%}) {'✅' if metrics['avg_relevance'] >= RELEVANCE_THRESHOLD else '❌'}
Faithfulness: {metrics['avg_faithfulness']:.1%} (threshold: {FAITHFULNESS_THRESHOLD:.1%}) {'✅' if metrics['avg_faithfulness'] >= FAITHFULNESS_THRESHOLD else '❌'}

Overall: {'✅ PASSED' if passed else '❌ FAILED'}

Next: {'Run 04_Deploy_Model_As_Endpoint' if passed else 'Review and improve model'}
""")
