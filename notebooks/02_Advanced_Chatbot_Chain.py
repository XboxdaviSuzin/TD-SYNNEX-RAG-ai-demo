# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - TD SYNNEX Advanced Chatbot Chain
# MAGIC 
# MAGIC This notebook builds the RAG chatbot using LangChain and registers it with MLflow for deployment.

# COMMAND ----------

# MAGIC %pip install "numpy<2" langchain==0.1.20 langchain-community==0.0.38 "mlflow[databricks]" databricks-vectorsearch sentence-transformers faiss-cpu

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import mlflow
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

CATALOG = "rusefx"
SCHEMA = "rag_schema"
TABLE_NAME = "td_synnex_products"
EXPERIMENT_NAME = "/Users/{}/TD_SYNNEX_RAG_Experiment".format(spark.sql("SELECT current_user()").collect()[0][0])
MODEL_NAME = f"{CATALOG}.{SCHEMA}.td_synnex_rag_model"

mlflow.set_experiment(EXPERIMENT_NAME)
print(f"MLflow Experiment: {EXPERIMENT_NAME}")
print(f"Model Name: {MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Product Data

# COMMAND ----------

products_df = spark.table(f"{CATALOG}.{SCHEMA}.{TABLE_NAME}").toPandas()
print(f"✅ Loaded {len(products_df)} products")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Embeddings and Vector Store

# COMMAND ----------

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

# Create embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# Create documents from product descriptions
documents = [
    Document(
        page_content=row["description"],
        metadata={
            "product_id": row["product_id"],
            "vendor": row["vendor"],
            "model": row["model"],
            "category": row["category"],
            "price_czk": row["price_czk"],
            "region": row["region"],
            "segment": row["partner_segment"]
        }
    )
    for _, row in products_df.iterrows()
]

print(f"✅ Created {len(documents)} documents")

# COMMAND ----------

# Create FAISS vector store
print("🔄 Building FAISS index (this may take a few minutes)...")
vectorstore = FAISS.from_documents(documents, embeddings)
print(f"✅ FAISS index created!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save FAISS Index as Pickle

# COMMAND ----------

# Serialize the FAISS index and documents for inclusion in model
faiss_data = {
    "index": vectorstore.index,
    "docstore": vectorstore.docstore,
    "index_to_docstore_id": vectorstore.index_to_docstore_id
}

# Save to a local file that will be included as artifact
local_path = "/Workspace/Shared/td_synnex_faiss_data.pkl"
with open(local_path, "wb") as f:
    pickle.dump(faiss_data, f)

print(f"✅ Saved FAISS data to {local_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create MLflow Model

# COMMAND ----------

class TDSynnexRAGModel(mlflow.pyfunc.PythonModel):
    """TD SYNNEX RAG Model for Databricks Model Serving"""
    
    def load_context(self, context):
        """Load the vectorstore and embedding model"""
        import pickle
        from langchain.embeddings import HuggingFaceEmbeddings
        from langchain.vectorstores import FAISS
        
        # Load FAISS data from artifact
        with open(context.artifacts["faiss_data"], "rb") as f:
            faiss_data = pickle.load(f)
        
        # Recreate embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Recreate FAISS vectorstore
        self.vectorstore = FAISS(
            embedding_function=self.embeddings,
            index=faiss_data["index"],
            docstore=faiss_data["docstore"],
            index_to_docstore_id=faiss_data["index_to_docstore_id"]
        )
    
    def _retrieve_context(self, query: str, k: int = 5) -> str:
        """Retrieve relevant products"""
        docs = self.vectorstore.similarity_search(query, k=k)
        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(f"[Product {i}]\n{doc.page_content}")
        return "\n\n".join(context_parts)
    
    def _generate_response(self, query: str, context: str) -> str:
        """Generate response using retrieved context"""
        # Parse top product from context
        lines = context.split("\n")
        product_info = {}
        for line in lines:
            if line.startswith("Product:"):
                product_info["name"] = line.replace("Product:", "").strip()
            elif line.startswith("Price:"):
                product_info["price"] = line.replace("Price:", "").strip()
            elif line.startswith("Category:"):
                product_info["category"] = line.replace("Category:", "").strip()
            elif line.startswith("Specifications:"):
                product_info["specs"] = line.replace("Specifications:", "").strip()
            elif line.startswith("Partner Segment:"):
                product_info["segment"] = line.replace("Partner Segment:", "").strip()
        
        response = f"""Based on your query "{query}", I recommend:

🏆 **Top Recommendation: {product_info.get('name', 'TD SYNNEX Product')}**

📋 **Details:**
- **Category:** {product_info.get('category', 'Enterprise Hardware')}
- **Price:** {product_info.get('price', 'Contact for pricing')}
- **Specifications:** {product_info.get('specs', 'Enterprise-grade')}
- **Best For:** {product_info.get('segment', 'Enterprise')} customers

📊 **Why This Recommendation:**
This product matches your requirements and is available through TD SYNNEX.

🔄 **Alternative Options:**
I found multiple matching products. Ask for alternatives if needed.

---
*Powered by TD SYNNEX RAG Assistant*
"""
        return response
    
    def predict(self, context, model_input):
        """Main prediction function for Model Serving"""
        import pandas as pd
        
        results = []
        
        if isinstance(model_input, pd.DataFrame):
            queries = model_input.get("query", model_input.get("messages", []))
            
            for query in queries:
                if isinstance(query, list):
                    user_messages = [m for m in query if m.get("role") == "user"]
                    if user_messages:
                        query = user_messages[-1].get("content", "")
                
                retrieved_context = self._retrieve_context(str(query))
                response = self._generate_response(str(query), retrieved_context)
                results.append(response)
        
        return results

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log and Register Model

# COMMAND ----------

from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec

# Define signature
input_schema = Schema([ColSpec(DataType.string, "query")])
output_schema = Schema([ColSpec(DataType.string, "response")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Log model
with mlflow.start_run(run_name="td_synnex_rag_v3") as run:
    mlflow.log_params({
        "embedding_model": "all-MiniLM-L6-v2",
        "vector_store": "FAISS",
        "num_products": len(products_df),
    })
    
    model_info = mlflow.pyfunc.log_model(
        artifact_path="td_synnex_rag",
        python_model=TDSynnexRAGModel(),
        artifacts={"faiss_data": local_path},
        pip_requirements=[
            "langchain==0.1.20",
            "langchain-community==0.0.38",
            "sentence-transformers",
            "faiss-cpu"
        ],
        signature=signature,
        input_example=pd.DataFrame({"query": ["Best Cisco switch for SMB"]})
    )
    
    run_id = run.info.run_id
    print(f"✅ Logged model - Run ID: {run_id}")

# COMMAND ----------

# Register to Unity Catalog
registered_model = mlflow.register_model(
    model_uri=f"runs:/{run_id}/td_synnex_rag",
    name=MODEL_NAME
)

print(f"✅ Registered model: {MODEL_NAME}")
print(f"   Version: {registered_model.version}")

# COMMAND ----------

# Save model info
spark.sql(f"DROP TABLE IF EXISTS {CATALOG}.{SCHEMA}.model_info")

model_info_df = spark.createDataFrame([{
    "run_id": run_id,
    "model_name": MODEL_NAME,
    "model_version": str(registered_model.version),
    "registered_at": datetime.now().isoformat()
}])

model_info_df.write.format("delta").mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.model_info")
print(f"✅ Saved model info")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Model

# COMMAND ----------

loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/td_synnex_rag")
test_query = pd.DataFrame({"query": ["Best Cisco switch CZ SMB under 100k CZK"]})
response = loaded_model.predict(test_query)
print("Test Response:")
print(response[0])

# COMMAND ----------

print(f"""
✅ NOTEBOOK COMPLETE!

Next steps:
1. Skip notebook 03 (evaluation) for now
2. Run notebook 04 to deploy the endpoint

Model: {MODEL_NAME}
Version: {registered_model.version}
""")
