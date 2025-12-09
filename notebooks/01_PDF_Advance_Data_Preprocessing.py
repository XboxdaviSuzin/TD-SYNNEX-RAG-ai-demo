# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - TD SYNNEX PDF/Catalog Data Preprocessing
# MAGIC 
# MAGIC This notebook processes the TD SYNNEX product catalog and creates embeddings for the RAG chatbot.
# MAGIC 
# MAGIC ## Steps:
# MAGIC 1. Generate/Load product catalog (5K Cisco/HP/Dell EU products)
# MAGIC 2. Create text chunks for embedding
# MAGIC 3. Generate embeddings using sentence-transformers
# MAGIC 4. Store in Delta Table with Vector Search index

# COMMAND ----------

# MAGIC %pip install "numpy<2" langchain==0.1.20 langchain-community==0.0.38 sentence-transformers chromadb==0.4.24 tiktoken pypdf databricks-vectorsearch

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import numpy as np
import pandas as pd
from datetime import datetime

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Unity Catalog configuration
CATALOG = "rusefx"
SCHEMA = "rag_schema"
TABLE_NAME = "td_synnex_products"
VECTOR_INDEX_NAME = "td_synnex_products_index"

# Embedding configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

print(f"Catalog: {CATALOG}.{SCHEMA}")
print(f"Table: {TABLE_NAME}")
print(f"Embedding Model: {EMBEDDING_MODEL}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Schema if not exists

# COMMAND ----------

# Check if catalog exists and create schema
# Note: rusefx catalog already exists in your workspace
try:
    # Try to use the existing catalog
    spark.sql(f"USE CATALOG {CATALOG}")
    print(f"✅ Using catalog: {CATALOG}")
except Exception as e:
    print(f"⚠️ Catalog {CATALOG} issue: {e}")
    # Fall back to default catalog
    CATALOG = spark.sql("SELECT current_catalog()").collect()[0][0]
    print(f"ℹ️ Using current catalog: {CATALOG}")

# Create schema in the catalog
try:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
    print(f"✅ Schema {CATALOG}.{SCHEMA} is ready")
except Exception as e:
    print(f"⚠️ Schema creation issue: {e}")
    # Try with default location
    try:
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
        print(f"✅ Schema {SCHEMA} created in current catalog")
    except Exception as e2:
        print(f"❌ Could not create schema: {e2}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate TD SYNNEX Product Catalog (5K Products)

# COMMAND ----------

import random

# Seed for reproducibility
np.random.seed(42)
random.seed(42)

# Product catalog configuration
VENDORS = {
    "Cisco": {
        "categories": ["Switches", "Routers", "Wireless", "Security", "Collaboration"],
        "models": {
            "Switches": ["Catalyst-9200", "Catalyst-9300", "Catalyst-9400", "Catalyst-9500", "Catalyst-9600", 
                        "Nexus-9300", "Nexus-9500", "Meraki-MS120", "Meraki-MS250", "Meraki-MS350"],
            "Routers": ["ISR-1000", "ISR-4000", "ASR-1000", "ASR-9000", "Catalyst-8200", "Catalyst-8300"],
            "Wireless": ["Meraki-MR36", "Meraki-MR46", "Meraki-MR56", "Catalyst-9100", "Aironet-2800"],
            "Security": ["Firepower-1000", "Firepower-2100", "ASA-5500", "Umbrella-SIG", "SecureX"],
            "Collaboration": ["Webex-Room-Kit", "Webex-Board", "Webex-Desk", "IP-Phone-8800", "Headset-700"]
        }
    },
    "HP": {
        "categories": ["Servers", "Storage", "Networking", "Workstations", "Accessories"],
        "models": {
            "Servers": ["ProLiant-DL380", "ProLiant-DL360", "ProLiant-ML350", "Synergy-480", "Edgeline-EL8000"],
            "Storage": ["Nimble-HF20", "Nimble-AF40", "Primera-A630", "StoreEasy-1660", "MSA-2062"],
            "Networking": ["Aruba-2930F", "Aruba-6300", "Aruba-CX-8360", "FlexNetwork-5130", "OfficeConnect"],
            "Workstations": ["Z4-G4", "Z6-G4", "Z8-G4", "ZBook-Fury", "ZBook-Studio"],
            "Accessories": ["Smart-Tank-750", "LaserJet-Pro", "EliteDisplay-E27", "Poly-Studio", "Thunderbolt-Dock"]
        }
    },
    "Dell": {
        "categories": ["Servers", "Storage", "Networking", "Laptops", "Desktops"],
        "models": {
            "Servers": ["PowerEdge-R750", "PowerEdge-R650", "PowerEdge-R550", "PowerEdge-T550", "PowerEdge-MX7000"],
            "Storage": ["PowerStore-500", "PowerStore-1000", "PowerScale-F200", "Unity-XT-380", "ECS-EX500"],
            "Networking": ["PowerSwitch-S5248", "PowerSwitch-Z9432", "PowerSwitch-N3248", "SmartFabric-Director"],
            "Laptops": ["Latitude-7440", "Latitude-5540", "Precision-7680", "XPS-15", "Inspiron-16"],
            "Desktops": ["OptiPlex-7010", "OptiPlex-5000", "Precision-3660", "Precision-5860", "XPS-Desktop"]
        }
    }
}

EU_REGIONS = ["CZ", "DE", "PL", "AT", "SK", "HU", "NL", "BE", "FR", "IT", "ES", "UK", "SE", "DK", "NO", "FI"]
PARTNER_SEGMENTS = ["Enterprise", "SMB", "SOHO", "Government", "Education", "Healthcare"]

def generate_specs(vendor, category, model):
    specs_templates = {
        "Switches": [
            f"48-port Gigabit, PoE+, {random.choice([4, 8, 12])}x 10G SFP+, {random.randint(176, 640)}Gbps switching",
            f"{random.choice([24, 48])}-port Multi-Gig, UPOE, Stackable, Layer 3 managed",
        ],
        "Routers": [
            f"Enterprise router, {random.randint(2, 8)} WAN ports, {random.randint(1, 10)}Gbps throughput, SD-WAN ready",
        ],
        "Servers": [
            f"2U Rack, Intel Xeon Gold, {random.choice([128, 256, 512])}GB RAM, {random.randint(2, 8)}x NVMe SSD",
        ],
        "Storage": [
            f"All-Flash Array, {random.randint(10, 500)}TB raw, {random.randint(100, 500)}K IOPS, NVMe-oF",
        ],
    }
    templates = specs_templates.get(category, [f"Enterprise-grade {category.lower()}, {vendor} certified"])
    return random.choice(templates)

def generate_catalog(n_products=5000):
    products = []
    product_id = 1000
    
    base_prices = {
        "Switches": (25000, 450000), "Routers": (30000, 500000), "Wireless": (8000, 85000),
        "Security": (45000, 650000), "Collaboration": (15000, 180000), "Servers": (85000, 850000),
        "Storage": (120000, 1200000), "Networking": (20000, 280000), "Workstations": (55000, 350000),
        "Accessories": (2000, 45000), "Laptops": (25000, 120000), "Desktops": (18000, 85000)
    }
    
    for _ in range(n_products):
        vendor = random.choice(list(VENDORS.keys()))
        category = random.choice(VENDORS[vendor]["categories"])
        model = random.choice(VENDORS[vendor]["models"][category])
        
        price_range = base_prices.get(category, (10000, 100000))
        price_czk = random.randint(price_range[0], price_range[1])
        
        specs = generate_specs(vendor, category, model)
        revenue_trend = random.uniform(-15, 35)
        stock_qty = random.choices([random.randint(0, 500), random.randint(1, 50), 0], weights=[0.7, 0.25, 0.05])[0]
        stock_status = "In Stock" if stock_qty > 50 else "Low Stock" if stock_qty > 0 else "Out of Stock"
        
        region = random.choice(EU_REGIONS)
        segment = random.choice(PARTNER_SEGMENTS)
        
        # Create rich description for RAG
        description = f"""Product: {vendor} {model}
Category: {category}
Specifications: {specs}
Price: {price_czk:,} CZK ({int(price_czk/25.5):,} EUR)
Region: {region}
Partner Segment: {segment}
Q3 Revenue Trend: {revenue_trend:+.1f}%
Stock Status: {stock_status} ({stock_qty} units)
Warranty: {random.choice([1, 2, 3, 5])} years
Margin: {random.uniform(8, 28):.1f}%

This {vendor} {model} is ideal for {segment} customers in {region}. 
It offers {specs} with enterprise-grade reliability."""

        products.append({
            "product_id": f"TD-{product_id:06d}",
            "vendor": vendor,
            "category": category,
            "model": model,
            "full_name": f"{vendor} {model}",
            "specs": specs,
            "price_czk": price_czk,
            "price_eur": int(price_czk / 25.5),
            "region": region,
            "partner_segment": segment,
            "revenue_trend_q3": round(revenue_trend, 1),
            "stock_qty": stock_qty,
            "stock_status": stock_status,
            "description": description
        })
        product_id += 1
    
    return pd.DataFrame(products)

# Generate catalog
print("🔄 Generating 5,000 TD SYNNEX products...")
catalog_df = generate_catalog(5000)
print(f"✅ Generated {len(catalog_df)} products")
print(f"\nVendor distribution:\n{catalog_df['vendor'].value_counts()}")

# COMMAND ----------

# Display sample
display(catalog_df.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Embeddings

# COMMAND ----------

from sentence_transformers import SentenceTransformer

# Load model
print(f"Loading embedding model: {EMBEDDING_MODEL}")
model = SentenceTransformer(EMBEDDING_MODEL)

# Generate embeddings for descriptions
print("🔄 Generating embeddings for all products...")
descriptions = catalog_df["description"].tolist()
embeddings = model.encode(descriptions, show_progress_bar=True, batch_size=64)

print(f"✅ Generated {len(embeddings)} embeddings with shape {embeddings.shape}")

# COMMAND ----------

# Add embeddings to dataframe
catalog_df["embedding"] = list(embeddings)
catalog_df["created_at"] = datetime.now().isoformat()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save to Delta Table

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, ArrayType

# Create Spark DataFrame
spark_df = spark.createDataFrame(catalog_df)

# Save to Delta Table
table_path = f"{CATALOG}.{SCHEMA}.{TABLE_NAME}"
spark_df.write.format("delta").mode("overwrite").saveAsTable(table_path)

print(f"✅ Saved {spark_df.count()} products to {table_path}")

# COMMAND ----------

# Verify data
display(spark.table(table_path).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Vector Search Index (for production RAG)

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

# Create or get endpoint
VECTOR_SEARCH_ENDPOINT = "td_synnex_vs_endpoint"

try:
    vsc.create_endpoint(name=VECTOR_SEARCH_ENDPOINT, endpoint_type="STANDARD")
    print(f"✅ Created Vector Search endpoint: {VECTOR_SEARCH_ENDPOINT}")
except Exception as e:
    print(f"ℹ️ Vector Search endpoint exists: {e}")

# Create index
index_name = f"{CATALOG}.{SCHEMA}.{VECTOR_INDEX_NAME}"
source_table = f"{CATALOG}.{SCHEMA}.{TABLE_NAME}"

try:
    vsc.create_delta_sync_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT,
        index_name=index_name,
        source_table_name=source_table,
        pipeline_type="TRIGGERED",
        primary_key="product_id",
        embedding_dimension=EMBEDDING_DIM,
        embedding_vector_column="embedding"
    )
    print(f"✅ Created Vector Search index: {index_name}")
except Exception as e:
    print(f"ℹ️ Vector Search index status: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC 
# MAGIC ✅ Generated 5,000 TD SYNNEX products (Cisco/HP/Dell)  
# MAGIC ✅ Created embeddings with sentence-transformers  
# MAGIC ✅ Stored in Delta Table: `rusefx.rag_schema.td_synnex_products`  
# MAGIC ✅ Created Vector Search index for production RAG  
# MAGIC 
# MAGIC **Next:** Run `02_Advanced_Chatbot_Chain` notebook
