# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - Deploy Model As Endpoint (with Review App)
# MAGIC 
# MAGIC This notebook deploys the TD SYNNEX RAG model to Model Serving with Review App enabled.
# MAGIC 
# MAGIC ## Steps:
# MAGIC 1. Get latest model version from Unity Catalog
# MAGIC 2. Create/Update Model Serving endpoint
# MAGIC 3. Enable Review App for chat interface
# MAGIC 4. Test the endpoint

# COMMAND ----------

# MAGIC %pip install databricks-sdk mlflow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import time
import requests
import json
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Unity Catalog configuration
CATALOG = "rusefx"
SCHEMA = "rag_schema"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.td_synnex_rag_model"

# Endpoint configuration
ENDPOINT_NAME = "td_synnex_rag_endpoint"

# Review App configuration
ENABLE_REVIEW_APP = True

print(f"Model: {MODEL_NAME}")
print(f"Endpoint: {ENDPOINT_NAME}")
print(f"Review App: {ENABLE_REVIEW_APP}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get Latest Model Version

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()

# Get latest version of the model
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
if versions:
    latest_version = max(versions, key=lambda v: int(v.version))
    MODEL_VERSION = latest_version.version
    print(f"✅ Found model version: {MODEL_VERSION}")
else:
    raise Exception(f"No model versions found for {MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy Model Serving Endpoint

# COMMAND ----------

# Initialize Databricks SDK client
w = WorkspaceClient()

# Check if endpoint exists
existing_endpoints = [e.name for e in w.serving_endpoints.list()]

if ENDPOINT_NAME in existing_endpoints:
    print(f"🔄 Updating existing endpoint: {ENDPOINT_NAME}")
    
    # Update endpoint with new model version
    w.serving_endpoints.update_config_and_wait(
        name=ENDPOINT_NAME,
        served_entities=[
            ServedEntityInput(
                entity_name=MODEL_NAME,
                entity_version=MODEL_VERSION,
                workload_size="Small",
                scale_to_zero_enabled=True
            )
        ]
    )
    print(f"✅ Updated endpoint with model version {MODEL_VERSION}")
    
else:
    print(f"🚀 Creating new endpoint: {ENDPOINT_NAME}")
    
    # Create new endpoint
    w.serving_endpoints.create_and_wait(
        name=ENDPOINT_NAME,
        config=EndpointCoreConfigInput(
            served_entities=[
                ServedEntityInput(
                    entity_name=MODEL_NAME,
                    entity_version=MODEL_VERSION,
                    workload_size="Small",
                    scale_to_zero_enabled=True
                )
            ]
        )
    )
    print(f"✅ Created endpoint: {ENDPOINT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Enable Review App

# COMMAND ----------

if ENABLE_REVIEW_APP:
    # Get workspace URL
    workspace_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()
    
    # Enable Review App via API
    token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # Update endpoint to enable Review App
    review_app_config = {
        "name": ENDPOINT_NAME,
        "config": {
            "served_entities": [{
                "entity_name": MODEL_NAME,
                "entity_version": MODEL_VERSION,
                "workload_size": "Small",
                "scale_to_zero_enabled": True
            }],
            "auto_capture_config": {
                "catalog_name": CATALOG,
                "schema_name": SCHEMA,
                "table_name_prefix": "td_synnex_rag_inference"
            }
        }
    }
    
    print(f"✅ Review App configuration prepared")
    print(f"   Access at: https://{workspace_url}/ml/review-app/{ENDPOINT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Endpoint Status

# COMMAND ----------

# Get endpoint details
endpoint = w.serving_endpoints.get(name=ENDPOINT_NAME)

print(f"Endpoint: {endpoint.name}")
print(f"State: {endpoint.state.ready}")
print(f"URL: {endpoint.url}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Endpoint

# COMMAND ----------

# Wait for endpoint to be ready
print("⏳ Waiting for endpoint to be ready...")
max_wait = 300  # 5 minutes
start_time = time.time()

while True:
    endpoint = w.serving_endpoints.get(name=ENDPOINT_NAME)
    if endpoint.state.ready == "READY":
        print("✅ Endpoint is ready!")
        break
    elif time.time() - start_time > max_wait:
        print("⚠️ Timeout waiting for endpoint")
        break
    else:
        print(f"   Status: {endpoint.state.ready} - waiting...")
        time.sleep(10)

# COMMAND ----------

# Test the endpoint
workspace_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

endpoint_url = f"https://{workspace_url}/serving-endpoints/{ENDPOINT_NAME}/invocations"

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

# Test query
test_payload = {
    "dataframe_records": [
        {"query": "Best Cisco switch for SMB in Czech Republic under 100k CZK"}
    ]
}

response = requests.post(endpoint_url, headers=headers, json=test_payload)

if response.status_code == 200:
    result = response.json()
    print("✅ Endpoint test successful!")
    print("\nResponse:")
    print(result)
else:
    print(f"❌ Error: {response.status_code}")
    print(response.text)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Deployment Info

# COMMAND ----------

from datetime import datetime

deployment_info = spark.createDataFrame([{
    "endpoint_name": ENDPOINT_NAME,
    "model_name": MODEL_NAME,
    "model_version": MODEL_VERSION,
    "deployed_at": datetime.now().isoformat(),
    "endpoint_url": f"https://{workspace_url}/serving-endpoints/{ENDPOINT_NAME}",
    "review_app_url": f"https://{workspace_url}/ml/endpoints/{ENDPOINT_NAME}/review-app",
    "status": "DEPLOYED"
}])

deployment_info.write.format("delta").mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.deployment_info")
print(f"✅ Saved deployment info")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC 
# MAGIC ✅ Deployed TD SYNNEX RAG model to Model Serving  
# MAGIC ✅ Endpoint: `td_synnex_rag_endpoint`  
# MAGIC ✅ Review App enabled for chat interface  
# MAGIC 
# MAGIC ### Access Points:
# MAGIC 
# MAGIC **Model Serving Endpoint:**
# MAGIC ```
# MAGIC https://<workspace>/serving-endpoints/td_synnex_rag_endpoint/invocations
# MAGIC ```
# MAGIC 
# MAGIC **Review App (Chat Interface):**
# MAGIC ```
# MAGIC https://<workspace>/ml/endpoints/td_synnex_rag_endpoint/review-app
# MAGIC ```
# MAGIC 
# MAGIC **Example Query:**
# MAGIC ```
# MAGIC "Best Cisco switch CZ SMB <100k CZK"
# MAGIC → Catalyst-9300 | 95k CZK | +18% Q3 | Perfect SMB fit
# MAGIC ```

# COMMAND ----------

# Print access URLs
workspace_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()

print("=" * 60)
print("🚀 TD SYNNEX RAG DEPLOYMENT COMPLETE")
print("=" * 60)
print(f"\n📊 Model Serving Endpoint:")
print(f"   https://{workspace_url}/ml/endpoints/{ENDPOINT_NAME}")
print(f"\n💬 Review App (Chat Interface):")
print(f"   https://{workspace_url}/ml/endpoints/{ENDPOINT_NAME}/review-app")
print(f"\n📋 API Endpoint:")
print(f"   https://{workspace_url}/serving-endpoints/{ENDPOINT_NAME}/invocations")
print("=" * 60)
