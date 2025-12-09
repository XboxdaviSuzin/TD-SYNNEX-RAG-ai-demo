# Databricks notebook source
# MAGIC %md
# MAGIC # 📊 Create Logging Tables for RAG System
# MAGIC 
# MAGIC This notebook creates the Delta tables required for:
# MAGIC - **User Feedback Logging** (RLHF-style feedback loop)
# MAGIC - **Document Upload Logging** (Audit trail for uploaded documents)
# MAGIC 
# MAGIC **Tables Created:**
# MAGIC 1. `td_synnex_catalog.rag_feedback` - Stores thumbs up/down feedback
# MAGIC 2. `td_synnex_catalog.uploaded_documents` - Stores document upload metadata

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Set Catalog and Schema

# COMMAND ----------

# Use your catalog (change if needed)
CATALOG = "td_synnex_catalog"
SCHEMA = "default"

# Create catalog if it doesn't exist
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"USE CATALOG {CATALOG}")

# Create schema if it doesn't exist
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
spark.sql(f"USE SCHEMA {SCHEMA}")

print(f"✅ Using catalog: {CATALOG}, schema: {SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Create Feedback Table
# MAGIC 
# MAGIC This table stores user feedback (thumbs up/down) for RLHF-style training.

# COMMAND ----------

# Create rag_feedback table
spark.sql("""
CREATE TABLE IF NOT EXISTS rag_feedback (
    response_id STRING COMMENT 'Unique identifier for the response',
    query STRING COMMENT 'User query',
    answer STRING COMMENT 'AI response (truncated to 500 chars)',
    feedback STRING COMMENT 'Feedback type: up or down',
    timestamp STRING COMMENT 'ISO timestamp of feedback',
    user_session STRING COMMENT 'User session identifier',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP() COMMENT 'Record creation time'
)
USING DELTA
COMMENT 'Stores user feedback for RAG responses - used for RLHF-style training'
TBLPROPERTIES (
    'delta.enableChangeDataFeed' = 'true',
    'delta.autoOptimize.optimizeWrite' = 'true'
)
""")

print("✅ Created table: rag_feedback")

# Show table schema
display(spark.sql("DESCRIBE TABLE rag_feedback"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Create Uploaded Documents Table
# MAGIC 
# MAGIC This table stores metadata for all uploaded documents (PDF, TXT, Text, Links).

# COMMAND ----------

# Create uploaded_documents table
spark.sql("""
CREATE TABLE IF NOT EXISTS uploaded_documents (
    doc_id STRING COMMENT 'Unique document identifier',
    doc_type STRING COMMENT 'Document type: PDF, TXT, Text, or Link',
    doc_name STRING COMMENT 'Original file name or URL',
    content_preview STRING COMMENT 'First 500 characters of content',
    chunks_count INT COMMENT 'Number of chunks created',
    timestamp STRING COMMENT 'ISO timestamp of upload',
    user_session STRING COMMENT 'User session identifier',
    status STRING COMMENT 'Processing status: processed, failed, pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP() COMMENT 'Record creation time'
)
USING DELTA
COMMENT 'Stores metadata for uploaded documents - audit trail for RAG Q&A'
TBLPROPERTIES (
    'delta.enableChangeDataFeed' = 'true',
    'delta.autoOptimize.optimizeWrite' = 'true'
)
""")

print("✅ Created table: uploaded_documents")

# Show table schema
display(spark.sql("DESCRIBE TABLE uploaded_documents"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Create Query Logs Table (Optional)
# MAGIC 
# MAGIC This table stores all user queries for analytics.

# COMMAND ----------

# Create query_logs table
spark.sql("""
CREATE TABLE IF NOT EXISTS query_logs (
    query_id STRING COMMENT 'Unique query identifier',
    query STRING COMMENT 'User query text',
    source STRING COMMENT 'Source: product_chat or rag_qa',
    response_preview STRING COMMENT 'First 200 characters of response',
    timestamp STRING COMMENT 'ISO timestamp of query',
    user_session STRING COMMENT 'User session identifier',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP() COMMENT 'Record creation time'
)
USING DELTA
COMMENT 'Stores all user queries for analytics and retraining'
TBLPROPERTIES (
    'delta.enableChangeDataFeed' = 'true',
    'delta.autoOptimize.optimizeWrite' = 'true'
)
""")

print("✅ Created table: query_logs")

# Show table schema
display(spark.sql("DESCRIBE TABLE query_logs"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Insert Sample Data (for Testing)

# COMMAND ----------

# Insert sample feedback
spark.sql("""
INSERT INTO rag_feedback (response_id, query, answer, feedback, timestamp, user_session)
VALUES 
    ('sample_001', 'What Cisco switches support PoE+?', 'The Catalyst 9300 series supports PoE+...', 'up', '2025-12-09T14:32:00', 'demo_session'),
    ('sample_002', 'Compare Dell vs HPE servers', 'Dell PowerEdge offers better iDRAC...', 'up', '2025-12-09T14:35:00', 'demo_session'),
    ('sample_003', 'Show HP laptops with 32GB RAM', 'HP EliteBook 860 G10 has 32GB RAM...', 'down', '2025-12-09T14:38:00', 'demo_session')
""")

print("✅ Inserted sample feedback data")

# Insert sample document
spark.sql("""
INSERT INTO uploaded_documents (doc_id, doc_type, doc_name, content_preview, chunks_count, timestamp, user_session, status)
VALUES 
    ('doc_sample_001', 'PDF', 'cisco_product_guide.pdf', 'Cisco Catalyst 9300 Series Data Sheet...', 15, '2025-12-09T14:30:00', 'demo_session', 'processed'),
    ('doc_sample_002', 'Link', 'https://nvidia.com/earnings', 'NVIDIA Q3 FY2024 Financial Results...', 8, '2025-12-09T14:45:00', 'demo_session', 'processed')
""")

print("✅ Inserted sample document data")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Verify Tables

# COMMAND ----------

print("📊 RAG Feedback Table:")
display(spark.sql("SELECT * FROM rag_feedback ORDER BY created_at DESC LIMIT 10"))

# COMMAND ----------

print("📄 Uploaded Documents Table:")
display(spark.sql("SELECT * FROM uploaded_documents ORDER BY created_at DESC LIMIT 10"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Grant Permissions (Optional)
# MAGIC 
# MAGIC If you need to grant access to other users:

# COMMAND ----------

# Uncomment and modify as needed:
# spark.sql("GRANT SELECT, INSERT ON TABLE rag_feedback TO `users`")
# spark.sql("GRANT SELECT, INSERT ON TABLE uploaded_documents TO `users`")
# spark.sql("GRANT SELECT, INSERT ON TABLE query_logs TO `users`")

print("ℹ️ Uncomment the permission grants above if needed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Summary
# MAGIC 
# MAGIC Tables created successfully:
# MAGIC 
# MAGIC | Table | Purpose |
# MAGIC |-------|---------|
# MAGIC | `rag_feedback` | RLHF-style user feedback (👍/👎) |
# MAGIC | `uploaded_documents` | Document upload audit trail |
# MAGIC | `query_logs` | Query analytics and retraining |
# MAGIC 
# MAGIC **Next Steps:**
# MAGIC 1. Run the Streamlit app: `streamlit run app.py`
# MAGIC 2. Upload a document in RAG Q&A tab
# MAGIC 3. Click 👍 or 👎 on responses
# MAGIC 4. Check this notebook to see logged data

# COMMAND ----------

# Quick stats
print("📈 Quick Stats:")
print(f"   Feedback records: {spark.sql('SELECT COUNT(*) FROM rag_feedback').collect()[0][0]}")
print(f"   Document records: {spark.sql('SELECT COUNT(*) FROM uploaded_documents').collect()[0][0]}")
print(f"   Positive feedback: {spark.sql(\"SELECT COUNT(*) FROM rag_feedback WHERE feedback = 'up'\").collect()[0][0]}")
print(f"   Negative feedback: {spark.sql(\"SELECT COUNT(*) FROM rag_feedback WHERE feedback = 'down'\").collect()[0][0]}")
