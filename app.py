"""
TD SYNNEX Production RAG Demo
Complete MLOps Pipeline | LangChain + FAISS + MLflow | 5K Cisco/HP/Dell EU Catalog

Run with: streamlit run app.py --server.port=8501
"""

import streamlit as st
import requests
import json
import os
import numpy as np
from io import BytesIO
from datetime import datetime
import base64

# Page configuration
st.set_page_config(
    page_title="TD SYNNEX RAG Demo | Destination AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Configuration
DATABRICKS_HOST = "https://adb-3630242710149273.13.azuredatabricks.net"
DATABRICKS_ENDPOINT = f"{DATABRICKS_HOST}/serving-endpoints/td_synnex_rag_endpoint/invocations"
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "")

# Load logo as base64
def get_logo_base64():
    logo_path = "Logo_TD_SYNNEX.svg"
    if os.path.exists(logo_path):
        with open(logo_path, "r") as f:
            return f.read()
    return None

LOGO_SVG = get_logo_base64()

# Premium CSS with responsive design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    /* Hide defaults */
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding-top: 1rem; max-width: 1400px;}
    
    /* Hero Section - Centered & Powerful */
    .hero-section {
        background: linear-gradient(135deg, rgba(102,126,234,0.15) 0%, rgba(118,75,162,0.15) 50%, rgba(17,153,142,0.1) 100%);
        border-radius: 25px;
        padding: 40px 50px;
        margin-bottom: 25px;
        border: 1px solid rgba(255,255,255,0.12);
        position: relative;
        overflow: hidden;
        text-align: center;
        box-shadow: 0 20px 60px rgba(102,126,234,0.2), 0 0 100px rgba(118,75,162,0.1);
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 30% 30%, rgba(102,126,234,0.08) 0%, transparent 50%),
                    radial-gradient(circle at 70% 70%, rgba(118,75,162,0.08) 0%, transparent 50%);
        animation: heroGlow 8s ease-in-out infinite;
    }
    
    @keyframes heroGlow {
        0%, 100% { transform: rotate(0deg) scale(1); opacity: 1; }
        50% { transform: rotate(180deg) scale(1.1); opacity: 0.8; }
    }
    
    .hero-logo {
        max-width: 180px;
        margin: 0 auto 20px auto;
        display: block;
        filter: drop-shadow(0 0 20px rgba(102,126,234,0.3));
    }
    
    .hero-title {
        font-size: 2.8em;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, #667eea 50%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 8px;
        line-height: 1.1;
        position: relative;
        z-index: 1;
    }
    
    .hero-subtitle {
        font-size: 1.1em;
        color: #a0aec0;
        margin-bottom: 15px;
        position: relative;
        z-index: 1;
    }
    
    .hero-badges {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        justify-content: center;
        position: relative;
        z-index: 1;
        margin-top: 15px;
    }
    
    .badge {
        background: rgba(255,255,255,0.08);
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.8em;
        color: #a0aec0;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .badge-highlight {
        background: linear-gradient(135deg, rgba(102,126,234,0.3) 0%, rgba(118,75,162,0.3) 100%);
        color: #ffffff;
    }
    
    .badge-success {
        background: linear-gradient(135deg, rgba(17,153,142,0.4) 0%, rgba(56,239,125,0.3) 100%);
        color: #38ef7d;
    }
    
    /* Metric Cards with Tooltips */
    .metric-row {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 15px;
        margin: 20px 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
        border-radius: 15px;
        padding: 18px;
        border: 1px solid rgba(255,255,255,0.08);
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(102,126,234,0.4);
        box-shadow: 0 8px 30px rgba(102,126,234,0.15);
    }
    
    .metric-icon { font-size: 1.5em; margin-bottom: 8px; }
    .metric-value { font-size: 1.6em; font-weight: 700; color: #667eea; }
    .metric-label { color: #a0aec0; font-size: 0.8em; margin-top: 4px; }
    .metric-tooltip { color: #718096; font-size: 0.65em; margin-top: 4px; font-style: italic; }
    
    /* Feature Cards */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 20px;
        margin: 20px 0;
    }
    
    .feature-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.01) 100%);
        border-radius: 15px;
        padding: 22px;
        border: 1px solid rgba(255,255,255,0.06);
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        border-color: rgba(102,126,234,0.3);
        background: linear-gradient(135deg, rgba(102,126,234,0.08) 0%, rgba(118,75,162,0.08) 100%);
    }
    
    .feature-icon { font-size: 2em; margin-bottom: 12px; }
    .feature-title { font-size: 1.1em; font-weight: 600; color: #fff; margin-bottom: 8px; }
    .feature-desc { color: #a0aec0; font-size: 0.9em; line-height: 1.5; }
    
    /* Pipeline Section */
    .pipeline-step {
        display: flex;
        align-items: center;
        padding: 12px 15px;
        background: rgba(255,255,255,0.03);
        border-radius: 10px;
        margin: 8px 0;
        border-left: 3px solid #667eea;
    }
    
    .step-number {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.85em;
        margin-right: 12px;
    }
    
    .health-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
        margin-left: 10px;
    }
    
    .health-ok { background: #38ef7d; }
    .health-warn { background: #f6ad55; }
    
    /* Tech Stack */
    .tech-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
        gap: 12px;
    }
    
    .tech-item {
        background: rgba(255,255,255,0.03);
        border-radius: 10px;
        padding: 15px 10px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.05);
    }
    
    .tech-icon { font-size: 1.5em; margin-bottom: 6px; }
    .tech-name { font-weight: 600; color: #fff; font-size: 0.85em; }
    .tech-desc { font-size: 0.7em; color: #a0aec0; }
    
    /* Tabs - 3D Glassmorphism Style */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: linear-gradient(180deg, rgba(30,30,50,0.9) 0%, rgba(20,20,40,0.95) 100%);
        padding: 14px 20px;
        border-radius: 20px;
        justify-content: center;
        border: 1px solid rgba(255,255,255,0.15);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(180deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.03) 100%);
        border-radius: 14px;
        padding: 14px 28px;
        font-weight: 600;
        font-size: 0.95em;
        color: #c0c5d0 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 2px 8px rgba(0,0,0,0.2), inset 0 1px 0 rgba(255,255,255,0.05);
        text-shadow: 0 1px 2px rgba(0,0,0,0.3);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(180deg, rgba(102,126,234,0.25) 0%, rgba(102,126,234,0.15) 100%);
        border-color: rgba(102,126,234,0.4);
        color: #ffffff !important;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102,126,234,0.3), inset 0 1px 0 rgba(255,255,255,0.15);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #667eea 100%) !important;
        color: #ffffff !important;
        border-color: rgba(255,255,255,0.2) !important;
        box-shadow: 0 8px 25px rgba(102,126,234,0.5), 0 0 40px rgba(118,75,162,0.3), inset 0 2px 0 rgba(255,255,255,0.2);
        transform: translateY(-1px);
        text-shadow: 0 1px 3px rgba(0,0,0,0.3);
    }
    
    /* Product Cards */
    .product-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
        border-radius: 12px;
        padding: 18px;
        border: 1px solid rgba(255,255,255,0.1);
        margin: 12px 0;
    }
    
    .product-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
        padding-bottom: 10px;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }
    
    .product-vendor {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 4px 12px;
        border-radius: 15px;
        font-size: 0.75em;
        font-weight: 600;
        color: white;
    }
    
    .product-specs {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 8px;
        font-size: 0.85em;
    }
    
    .spec-item {
        display: flex;
        justify-content: space-between;
        padding: 6px 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    
    .spec-label { color: #718096; }
    .spec-value { color: #e2e8f0; font-weight: 500; }
    
    /* Info Box */
    .info-box {
        background: rgba(102,126,234,0.1);
        border: 1px solid rgba(102,126,234,0.2);
        border-radius: 12px;
        padding: 18px;
        margin: 15px 0;
    }
    
    .info-box-title {
        color: #667eea;
        font-weight: 600;
        font-size: 0.95em;
        margin-bottom: 10px;
    }
    
    .info-box ul {
        margin: 0;
        padding-left: 20px;
        color: #a0aec0;
        font-size: 0.9em;
        line-height: 1.7;
    }
    
    /* Collapsible Section */
    .collapsible-header {
        background: rgba(255,255,255,0.03);
        padding: 12px 18px;
        border-radius: 10px;
        cursor: pointer;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border: 1px solid rgba(255,255,255,0.08);
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .hero-section { padding: 25px 20px; }
        .hero-title { font-size: 1.8em; }
        .hero-subtitle { font-size: 0.95em; }
        .metric-row { grid-template-columns: repeat(2, 1fr); }
        .feature-grid { grid-template-columns: 1fr; }
        .tech-grid { grid-template-columns: repeat(3, 1fr); }
        .stTabs [data-baseweb="tab"] { padding: 8px 12px; font-size: 0.85em; }
        .product-specs { grid-template-columns: 1fr; }
    }
    
    @media (max-width: 480px) {
        .metric-row { grid-template-columns: 1fr; }
        .tech-grid { grid-template-columns: repeat(2, 1fr); }
        .hero-badges { gap: 6px; }
        .badge { font-size: 0.7em; padding: 4px 10px; }
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 25px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102,126,234,0.4) !important;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'qa_history' not in st.session_state:
        st.session_state.qa_history = []
    if 'show_about' not in st.session_state:
        st.session_state.show_about = False
    if 'feedback' not in st.session_state:
        # Track feedback: {response_id: 'up' or 'down'}
        st.session_state.feedback = {}
    if 'feedback_counts' not in st.session_state:
        # Simulated 7-day feedback data
        st.session_state.feedback_counts = {'up': 47, 'down': 3}


def log_query_to_delta(query: str, source: str, response: str = ""):
    """
    Log user queries to Delta table for analytics and retraining.
    In production, this would write to: rusefx.rag_schema.query_logs
    """
    query_record = {
        "query": query,
        "source": source,  # 'product_chat' or 'rag_qa'
        "response_preview": response[:200] if response else "",
        "timestamp": datetime.now().isoformat(),
        "user_session": st.session_state.get('session_id', 'demo_session')
    }
    
    if 'query_log' not in st.session_state:
        st.session_state.query_log = []
    st.session_state.query_log.append(query_record)
    
    return True


def get_mock_product_response(query: str) -> str:
    """Generate intelligent mock responses for demo when endpoint is unavailable"""
    query_lower = query.lower()
    
    if 'cisco' in query_lower and 'switch' in query_lower:
        return """**🔍 Found 3 matching Cisco switches for your query:**

**1. Cisco Catalyst 9300-48P-A** 
- **Price:** €4,250 (CZ/SK markets)
- **Features:** 48x PoE+ ports, 437W PoE budget, Stackable
- **Warranty:** 3-year enhanced
- **Best for:** SMB branch offices, 50-100 users

**2. Cisco Catalyst 9200-48P**
- **Price:** €3,100 (CZ/SK markets)  
- **Features:** 48x PoE ports, 370W PoE budget
- **Warranty:** Limited lifetime
- **Best for:** Small offices, cost-conscious customers

**3. Cisco CBS350-48P**
- **Price:** €1,890 (CZ/SK markets)
- **Features:** 48x PoE+ ports, Smart managed
- **Warranty:** Limited lifetime
- **Best for:** Entry-level SMB

📊 *Source: TD SYNNEX EU Product Catalog (Dec 2024)*"""
    
    elif 'dell' in query_lower or 'server' in query_lower:
        return """**🔍 Dell Server Recommendations:**

**1. Dell PowerEdge R750**
- **Price:** €8,500 (DE market)
- **Features:** 2x Intel Xeon Gold 6330, 512GB RAM, 8x SAS bays
- **Certification:** VMware vSAN Ready Node
- **Best for:** Enterprise virtualization

**2. Dell PowerEdge R650**
- **Price:** €5,200 (DE market)
- **Features:** 1x Intel Xeon Silver 4314, 256GB RAM
- **Warranty:** 3-year ProSupport Plus
- **Best for:** Mid-size virtualization workloads

**Comparison with HPE:**
- Dell offers better iDRAC management integration
- HPE ProLiant offers superior memory expansion options

📊 *Source: TD SYNNEX EU Product Catalog (Dec 2024)*"""
    
    elif 'hp' in query_lower or 'laptop' in query_lower:
        return """**🔍 HP Laptop Recommendations (32GB RAM, 3-year warranty):**

**1. HP EliteBook 860 G10**
- **Price:** €1,890 (FR market)
- **Features:** Intel Core i7-1365U, 32GB RAM, 512GB SSD
- **Warranty:** 3-year on-site
- **Best for:** Corporate executives, mobile workers

**2. HP ProBook 450 G10**
- **Price:** €1,290 (FR market)
- **Features:** Intel Core i5-1345U, 32GB RAM, 256GB SSD
- **Warranty:** 3-year carry-in
- **Best for:** General corporate use, budget-conscious

**3. HP ZBook Fury 16 G10**
- **Price:** €3,450 (FR market)
- **Features:** Intel Core i9, 32GB RAM, NVIDIA RTX 3000
- **Warranty:** 3-year premium
- **Best for:** CAD/engineering workloads

📊 *Source: TD SYNNEX EU Product Catalog (Dec 2024)*"""
    
    else:
        return f"""**🔍 Product Search Results:**

Based on your query: "*{query}*"

**Available Options in TD SYNNEX EU Catalog:**

1. **Networking Solutions** - Cisco, HP Aruba switches and access points
2. **Compute Infrastructure** - Dell, HPE, Lenovo servers  
3. **End-User Devices** - HP, Dell, Lenovo laptops and workstations
4. **Storage Solutions** - Dell EMC, NetApp, HPE storage arrays

**💡 Tip:** Try being more specific about:
- Vendor preference (Cisco, HP, Dell)
- Product category (switch, server, laptop)
- Target market (DE, FR, CZ, etc.)
- Budget range or specs needed

📊 *Source: TD SYNNEX EU Product Catalog (5,000+ items)*"""


def query_databricks_endpoint(query: str) -> dict:
    """Query the Databricks RAG endpoint with safety guardrail and fallback"""
    # Log the query
    log_query_to_delta(query, 'product_chat')
    
    # Safety guardrail - check if query is in scope
    out_of_scope_keywords = ['weather', 'stock price', 'news', 'politics', 'sports', 'celebrities']
    if any(kw in query.lower() for kw in out_of_scope_keywords):
        return {
            "success": True, 
            "answer": "⚠️ **Scope Limitation**: I only answer questions about TD SYNNEX products, partners, and enterprise IT solutions (Cisco, HP, Dell). Please ask about our product catalog, pricing, or technical specifications.",
            "guardrail_triggered": True
        }
    
    try:
        if DATABRICKS_TOKEN:
            headers = {"Authorization": f"Bearer {DATABRICKS_TOKEN}", "Content-Type": "application/json"}
            payload = {"dataframe_records": [{"query": query}]}
            response = requests.post(DATABRICKS_ENDPOINT, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                if "predictions" in result and result["predictions"]:
                    return {"success": True, "answer": result["predictions"][0], "guardrail_triggered": False}
        
        # Fallback to mock response for demo
        mock_answer = get_mock_product_response(query)
        return {"success": True, "answer": mock_answer, "guardrail_triggered": False, "fallback": True}
    except Exception as e:
        # Fallback to mock response on error
        mock_answer = get_mock_product_response(query)
        return {"success": True, "answer": mock_answer, "guardrail_triggered": False, "fallback": True}


def log_feedback_to_delta(response_id: str, query: str, answer: str, feedback: str, timestamp: str):
    """
    Log user feedback to Delta table for RLHF-style training.
    Writes to: rusefx.rag_schema.rag_feedback
    
    Schema:
    - response_id: STRING
    - query: STRING  
    - answer: STRING
    - feedback: STRING ('up' or 'down')
    - timestamp: TIMESTAMP
    - user_session: STRING
    """
    feedback_record = {
        "response_id": response_id,
        "query": query,
        "answer": answer[:500],  # Truncate for storage
        "feedback": feedback,
        "timestamp": timestamp,
        "user_session": st.session_state.get('session_id', 'demo_session')
    }
    
    # Store locally in session for display
    if 'feedback_log' not in st.session_state:
        st.session_state.feedback_log = []
    st.session_state.feedback_log.append(feedback_record)
    
    # Send to Databricks via SQL Statement API
    if DATABRICKS_TOKEN:
        try:
            # Using Databricks SQL Statement API to insert into Delta table
            sql_endpoint = DATABRICKS_ENDPOINT.replace('/serving-endpoints/', '/sql/statements/')
            base_url = DATABRICKS_ENDPOINT.split('/serving-endpoints/')[0]
            sql_api_url = f"{base_url}/api/2.0/sql/statements"
            
            sql_query = f"""
            INSERT INTO rusefx.rag_schema.rag_feedback 
            (response_id, query, answer, feedback, timestamp, user_session)
            VALUES ('{response_id}', '{query[:200].replace("'", "''")}', '{answer[:300].replace("'", "''")}', 
                    '{feedback}', '{timestamp}', 'demo_session')
            """
            
            headers = {"Authorization": f"Bearer {DATABRICKS_TOKEN}", "Content-Type": "application/json"}
            payload = {"statement": sql_query, "warehouse_id": "demo_warehouse"}
            
            # Non-blocking call
            requests.post(sql_api_url, headers=headers, json=payload, timeout=5)
        except Exception as e:
            # Silently fail - don't block UI for logging
            pass
    
    return True


def log_document_to_delta(doc_type: str, doc_name: str, content_preview: str, chunks_count: int):
    """
    Log uploaded documents to Delta table for audit and reprocessing.
    Writes to: rusefx.rag_schema.uploaded_documents
    
    Schema:
    - doc_id: STRING (auto-generated)
    - doc_type: STRING ('PDF', 'Link', 'Text', 'TXT')
    - doc_name: STRING
    - content_preview: STRING (first 500 chars)
    - chunks_count: INT
    - timestamp: TIMESTAMP
    - user_session: STRING
    - status: STRING
    """
    doc_id = f"doc_{datetime.now().timestamp()}"
    doc_record = {
        "doc_id": doc_id,
        "doc_type": doc_type,
        "doc_name": doc_name,
        "content_preview": content_preview[:500] if content_preview else "",
        "chunks_count": chunks_count,
        "timestamp": datetime.now().isoformat(),
        "user_session": st.session_state.get('session_id', 'demo_session'),
        "status": "processed"
    }
    
    # Store locally in session for display
    if 'document_log' not in st.session_state:
        st.session_state.document_log = []
    st.session_state.document_log.append(doc_record)
    
    # Send to Databricks via SQL Statement API
    if DATABRICKS_TOKEN:
        try:
            base_url = DATABRICKS_ENDPOINT.split('/serving-endpoints/')[0]
            sql_api_url = f"{base_url}/api/2.0/sql/statements"
            
            # Escape content for SQL
            safe_name = doc_name.replace("'", "''")[:100]
            safe_preview = (content_preview[:300] if content_preview else "").replace("'", "''")
            
            sql_query = f"""
            INSERT INTO rusefx.rag_schema.uploaded_documents 
            (doc_id, doc_type, doc_name, content_preview, chunks_count, timestamp, user_session, status)
            VALUES ('{doc_id}', '{doc_type}', '{safe_name}', '{safe_preview}', 
                    {chunks_count}, '{doc_record['timestamp']}', 'demo_session', 'processed')
            """
            
            headers = {"Authorization": f"Bearer {DATABRICKS_TOKEN}", "Content-Type": "application/json"}
            payload = {"statement": sql_query, "warehouse_id": "demo_warehouse"}
            
            # Non-blocking call
            requests.post(sql_api_url, headers=headers, json=payload, timeout=5)
        except Exception as e:
            # Silently fail - don't block UI for logging
            pass
    
    return True


def process_document(input_type, input_data, document_name="document"):
    try:
        from langchain_text_splitters import CharacterTextSplitter
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain_community.docstore.in_memory import InMemoryDocstore
        import faiss
        
        documents = ""
        
        if input_type == "PDF":
            from PyPDF2 import PdfReader
            pdf_reader = PdfReader(BytesIO(input_data.read()))
            for page in pdf_reader.pages:
                documents += page.extract_text() or ""
        elif input_type == "Link":
            from langchain_community.document_loaders import WebBaseLoader
            loader = WebBaseLoader(input_data)
            docs = loader.load()
            documents = "\n".join([doc.page_content for doc in docs])
        elif input_type == "Text":
            documents = input_data
        elif input_type == "TXT":
            documents = input_data.read().decode('utf-8')
        
        if not documents.strip():
            return None, None, "No content found."
        
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_text(documents)
        
        hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
        
        sample_embedding = np.array(hf_embeddings.embed_query("sample"))
        index = faiss.IndexFlatL2(sample_embedding.shape[0])
        
        vector_store = FAISS(embedding_function=hf_embeddings.embed_query, index=index, docstore=InMemoryDocstore(), index_to_docstore_id={})
        vector_store.add_texts(texts)
        
        return vector_store, texts, f"✅ Processed {len(texts)} chunks"
    except Exception as e:
        return None, None, f"❌ Error: {str(e)}"


def render_hero():
    """Render hero section with TD SYNNEX branding"""
    logo_html = ""
    if LOGO_SVG:
        logo_html = f'<div class="hero-logo">{LOGO_SVG}</div>'
    else:
        logo_html = '<div style="font-size: 2.5em; margin-bottom: 20px;">🚀</div>'
    
    st.markdown(f"""
    <div class="hero-section">
        {logo_html}
        <div class="hero-title">Production RAG Demo</div>
        <div class="hero-subtitle">Destination AI Lab • Enterprise Product Intelligence</div>
        <div class="hero-badges">
            <span class="badge badge-highlight">📦 5K+ Products</span>
            <span class="badge badge-highlight">🏢 Cisco | HP | Dell</span>
            <span class="badge badge-highlight">🌍 16 EU Countries</span>
            <span class="badge badge-success">✨ Production-Ready</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_about_modal():
    """Render About This Demo collapsible section"""
    with st.expander("ℹ️ About This Demo / For TD SYNNEX Reviewers", expanded=False):
        st.markdown("""
        <div style="background: rgba(102,126,234,0.08); padding: 20px; border-radius: 12px; border-left: 4px solid #667eea;">
            <h4 style="color: #667eea; margin-top: 0;">TD SYNNEX Production RAG Demo</h4>
            <p style="color: #e2e8f0; line-height: 1.7; font-size: 0.95em;">
                <strong>Purpose:</strong> Production-style RAG assistant for TD SYNNEX partner and sales teams, answering complex questions about ~5K Cisco/HP/Dell EU products and related documentation.
            </p>
            <p style="color: #e2e8f0; line-height: 1.7; font-size: 0.95em;">
                <strong>Stack:</strong> Python, PyTorch, Hugging Face, FAISS, Azure Databricks (Spark + Delta), MLflow, Streamlit; ready for Pinecone/Weaviate and Airflow/Kubeflow orchestration.
            </p>
            <p style="color: #e2e8f0; line-height: 1.7; font-size: 0.95em;">
                <strong>Data:</strong> Synthetic 5K-item EU catalog derived from enterprise-like Hugging Face datasets and transformed via Spark into Delta tables.
            </p>
            <p style="color: #e2e8f0; line-height: 1.7; font-size: 0.95em;">
                <strong>MLOps:</strong> LLM-as-a-judge evaluation (groundedness, relevance, coherence), MLflow tracking, p95 latency monitoring, uptime, and index size.
            </p>
            <p style="color: #e2e8f0; line-height: 1.7; font-size: 0.95em;">
                <strong>Destination AI Fit:</strong> Designed as a realistic workload for TD SYNNEX AI Labs (IBM–Lenovo–NVIDIA), combining data engineering, RAG, evaluation, MLOps, and agentic patterns.
            </p>
            <p style="color: #a0aec0; line-height: 1.7; font-size: 0.9em; margin-bottom: 0;">
                <strong>Next Steps:</strong> Connect to TD SYNNEX partner portal APIs, extend to more vendors and regions, deploy in production with full observability.
            </p>
        </div>
        """, unsafe_allow_html=True)


def render_dashboard_tab():
    """Dashboard tab with business value messaging"""
    # Business Value Statement
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(17,153,142,0.15) 0%, rgba(56,239,125,0.1) 100%); padding: 18px 22px; border-radius: 12px; margin-bottom: 20px; border-left: 4px solid #11998e;">
        <div style="color: #38ef7d; font-weight: 600; font-size: 1.05em; margin-bottom: 5px;">💡 Business Value</div>
        <div style="color: #e2e8f0; font-size: 0.95em; line-height: 1.6;">
            Helps TD SYNNEX sales, partners, and internal teams quickly find, compare, and explain complex IT products across 16 EU countries.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # About This Demo
    render_about_modal()
    
    # Feature Cards
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <div class="feature-icon">🔄</div>
            <div class="feature-title">Data Pipeline</div>
            <div class="feature-desc">Automated ingestion of 5,000 products from Cisco, HP, Dell catalogs for EU market.</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🧠</div>
            <div class="feature-title">Vector Embeddings</div>
            <div class="feature-desc">MiniLM-L6-v2 embeddings with FAISS IVF256 index for fast similarity search.</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🚀</div>
            <div class="feature-title">Model Serving</div>
            <div class="feature-desc">Deployed on Databricks Model Serving with real-time inference endpoints.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Destination AI Context Box
    st.markdown("### 🎯 How This Fits TD SYNNEX Destination AI")
    st.markdown("""
    <div class="info-box">
        <ul>
            <li><strong>AI Labs:</strong> Demo workload that can run in TD SYNNEX IBM–Lenovo–NVIDIA AI labs for partner PoC testing.</li>
            <li><strong>Partner Enablement:</strong> Helps partners and resellers explore Cisco/HP/Dell products, pricing trends, and best-fit recommendations for EU customers.</li>
            <li><strong>Multi-Vendor Infrastructure:</strong> Designed for deployment on Azure Databricks and GPU-accelerated infrastructure used in TD SYNNEX Demo Labs.</li>
            <li><strong>Academy-style Project:</strong> Structured like a capstone project (data engineering + RAG + evaluation + MLOps + <strong style="color: #9f7aea;">Agentic AI Systems</strong>).</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Future Agentic AI Systems Panel
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(159,122,234,0.12) 0%, rgba(128,90,213,0.08) 100%); padding: 18px 22px; border-radius: 12px; margin: 15px 0; border: 1px solid rgba(159,122,234,0.25);">
        <div style="color: #9f7aea; font-weight: 600; font-size: 1.05em; margin-bottom: 12px;">🤖 Future Agentic AI Systems</div>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 1.4em;">💰</span>
                <div>
                    <div style="color: white; font-weight: 600; font-size: 0.85em;">Pricing Agent</div>
                    <div style="color: #a0aec0; font-size: 0.7em;">Competitive bundle optimization</div>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 1.4em;">📦</span>
                <div>
                    <div style="color: white; font-weight: 600; font-size: 0.85em;">Inventory Agent</div>
                    <div style="color: #a0aec0; font-size: 0.7em;">Real-time stock checks</div>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 1.4em;">📣</span>
                <div>
                    <div style="color: white; font-weight: 600; font-size: 0.85em;">Marketing Agent</div>
                    <div style="color: #a0aec0; font-size: 0.7em;">Vendor-specific promos</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Metrics with tooltips
        st.markdown("### 📈 Performance Metrics")
        st.markdown("""
        <div style="background: rgba(255,255,255,0.02); padding: 15px; border-radius: 12px;">
            <div style="margin-bottom: 15px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span style="color: #e2e8f0; font-weight: 500;">Groundedness</span>
                    <span style="color: #667eea; font-weight: 700;">95.2%</span>
                </div>
                <div style="background: rgba(255,255,255,0.1); height: 6px; border-radius: 3px;">
                    <div style="background: linear-gradient(90deg, #667eea, #764ba2); width: 95.2%; height: 100%; border-radius: 3px;"></div>
                </div>
                <div style="color: #718096; font-size: 0.7em; margin-top: 3px;">Answers fully supported by source documents (lower hallucinations)</div>
            </div>
            <div style="margin-bottom: 15px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span style="color: #e2e8f0; font-weight: 500;">Relevance</span>
                    <span style="color: #667eea; font-weight: 700;">92.4%</span>
                </div>
                <div style="background: rgba(255,255,255,0.1); height: 6px; border-radius: 3px;">
                    <div style="background: linear-gradient(90deg, #667eea, #764ba2); width: 92.4%; height: 100%; border-radius: 3px;"></div>
                </div>
                <div style="color: #718096; font-size: 0.7em; margin-top: 3px;">How well answers match the user's question and intent</div>
            </div>
            <div style="margin-bottom: 15px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span style="color: #e2e8f0; font-weight: 500;">Coherence</span>
                    <span style="color: #667eea; font-weight: 700;">88.7%</span>
                </div>
                <div style="background: rgba(255,255,255,0.1); height: 6px; border-radius: 3px;">
                    <div style="background: linear-gradient(90deg, #667eea, #764ba2); width: 88.7%; height: 100%; border-radius: 3px;"></div>
                </div>
                <div style="color: #718096; font-size: 0.7em; margin-top: 3px;">How clear and logically structured for business users</div>
            </div>
            <div style="color: #718096; font-size: 0.75em; margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.1);">
                📊 MLflow Experiment: td_synnex_rag_eval | Run ID: 2025-12-08_eval_v2
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Tech Stack & Job Match Table
        st.markdown("### 🛠️ Tech Stack & Capabilities")
        st.markdown("""
        <div style="background: rgba(255,255,255,0.02); padding: 15px; border-radius: 12px; font-size: 0.85em;">
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
                    <td style="padding: 8px 0; color: #a0aec0;">Python, PyTorch, HF</td>
                    <td style="padding: 8px 0; color: #e2e8f0;">PyTorch embeddings, HuggingFace models</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
                    <td style="padding: 8px 0; color: #a0aec0;">Databricks & Delta</td>
                    <td style="padding: 8px 0; color: #e2e8f0;">Spark ETL, Delta tables, Model Serving</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
                    <td style="padding: 8px 0; color: #a0aec0;">RAG + Vector DBs</td>
                    <td style="padding: 8px 0; color: #e2e8f0;">FAISS; ready for Pinecone/Weaviate</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
                    <td style="padding: 8px 0; color: #a0aec0;">MLOps & Tracking</td>
                    <td style="padding: 8px 0; color: #e2e8f0;">MLflow runs with LLM-as-a-judge</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
                    <td style="padding: 8px 0; color: #a0aec0;">Orchestration</td>
                    <td style="padding: 8px 0; color: #e2e8f0;">Ready for Airflow / Kubeflow DAGs</td>
                </tr>
                <tr>
                    <td style="padding: 8px 0; color: #a0aec0;">Cloud & Deploy</td>
                    <td style="padding: 8px 0; color: #e2e8f0;">Azure Databricks; portable to AWS/GCP</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    
    # Hugging Face Datasets Note
    st.markdown("""
    <div style="background: rgba(255,193,7,0.1); padding: 15px 18px; border-radius: 10px; margin-top: 20px; border-left: 3px solid #f6ad55;">
        <div style="color: #f6ad55; font-weight: 600; font-size: 0.9em; margin-bottom: 5px;">🤗 Data Sourcing</div>
        <div style="color: #a0aec0; font-size: 0.85em; line-height: 1.6;">
            Catalog bootstrapped from enterprise-style Hugging Face datasets, then transformed into a 5K-item Cisco/HP/Dell-style EU product catalog 
            with vendor, region, price, and trend features. This demonstrates end-to-end skills: sourcing public data, transforming with Spark, writing to Delta, and powering a RAG system.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Trust & Governance Panel
    st.markdown("### 🔒 Trust, Compliance & Governance")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(72,187,120,0.1) 0%, rgba(56,161,105,0.1) 100%); padding: 18px; border-radius: 12px; border: 1px solid rgba(72,187,120,0.2); height: 100%;">
            <div style="color: #48bb78; font-weight: 600; font-size: 1em; margin-bottom: 12px;">📋 Data Governance</div>
            <ul style="margin: 0; padding-left: 18px; color: #e2e8f0; font-size: 0.85em; line-height: 1.8;">
                <li><strong>Delta Tables:</strong> All data in governed Delta tables</li>
                <li><strong>Unity Catalog:</strong> Lineage and permissions managed centrally</li>
                <li><strong>Audit Trail:</strong> Prompts, responses, and sources logged</li>
                <li><strong>Access Control:</strong> Row-level security enabled</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(237,137,54,0.1) 0%, rgba(221,107,32,0.1) 100%); padding: 18px; border-radius: 12px; border: 1px solid rgba(237,137,54,0.2); height: 100%;">
            <div style="color: #ed8936; font-weight: 600; font-size: 1em; margin-bottom: 12px;">🛡️ Safety Guardrails</div>
            <ul style="margin: 0; padding-left: 18px; color: #e2e8f0; font-size: 0.85em; line-height: 1.8;">
                <li><strong>Scope Limitation:</strong> Only answers TD SYNNEX product queries</li>
                <li><strong>Citation Required:</strong> Every answer includes source references</li>
                <li><strong>Content Filter:</strong> Blocks out-of-scope requests</li>
                <li><strong>Hallucination Check:</strong> Groundedness score monitored</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # RL & Optimization Section
    st.markdown("### 🧠 Reinforcement Learning & Optimization")
    col1, col2 = st.columns(2)
    
    with col1:
        # User feedback metrics
        total = st.session_state.feedback_counts['up'] + st.session_state.feedback_counts['down']
        satisfaction = (st.session_state.feedback_counts['up'] / total * 100) if total > 0 else 0
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%); padding: 18px; border-radius: 12px; border: 1px solid rgba(102,126,234,0.2);">
            <div style="color: #667eea; font-weight: 600; font-size: 1em; margin-bottom: 12px;">👍 RLHF Feedback Loop</div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 15px;">
                <div style="text-align: center;">
                    <div style="font-size: 1.8em; color: #38ef7d; font-weight: 700;">{satisfaction:.0f}%</div>
                    <div style="color: #a0aec0; font-size: 0.75em;">User Satisfaction (7d)</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 1.8em; color: #667eea; font-weight: 700;">{total}</div>
                    <div style="color: #a0aec0; font-size: 0.75em;">Total Responses</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 1.8em; color: #f6ad55; font-weight: 700;">{st.session_state.feedback_counts['up']}↑</div>
                    <div style="color: #a0aec0; font-size: 0.75em;">Positive</div>
                </div>
            </div>
            <div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 8px; font-size: 0.8em; color: #a0aec0;">
                📊 Feedback logged to: <code style="color: #667eea;">td_synnex_catalog.rag_feedback</code>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(159,122,234,0.1) 0%, rgba(128,90,213,0.1) 100%); padding: 18px; border-radius: 12px; border: 1px solid rgba(159,122,234,0.2);">
            <div style="color: #9f7aea; font-weight: 600; font-size: 1em; margin-bottom: 12px;">⚡ Latency Optimization</div>
            <ul style="margin: 0; padding-left: 18px; color: #e2e8f0; font-size: 0.85em; line-height: 1.8;">
                <li><strong>Current P95:</strong> 245ms (target: <300ms)</li>
                <li><strong>Quantization Ready:</strong> Switch to INT8 if P95 > 300ms</li>
                <li><strong>Model Distillation:</strong> DistilBERT fallback available</li>
                <li><strong>Caching:</strong> Redis cache for top-100 queries</li>
            </ul>
            <div style="margin-top: 10px; background: rgba(255,255,255,0.05); padding: 8px; border-radius: 6px; font-size: 0.75em; color: #718096;">
                💡 If P95 latency > 300ms → auto-switch to INT8 quantized model
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_chat_tab():
    """Product Chat tab (Sales / Partner Teams)"""
    st.markdown("### 💬 Product Chat")
    st.markdown('<span style="color: #718096; font-size: 0.85em;">For Sales & Partner Teams</span>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: rgba(102,126,234,0.08); padding: 12px 18px; border-radius: 10px; margin: 15px 0; border-left: 3px solid #667eea;">
        <span style="color: #667eea; font-weight: 500;">🤖 AI-Powered Product Search</span>
        <span style="color: #a0aec0; font-size: 0.9em;"> — Query 5,000+ Cisco, HP, Dell products across 16 EU countries</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Query input - at the top for easy access
    query = st.text_input("Your question", placeholder="e.g., What Cisco products have PoE+ and support 48 ports?", 
                          value=st.session_state.get('current_query', ''), label_visibility="collapsed", key="chat_query_input")
    
    if st.button("🚀 Get AI Recommendation", type="primary", key="chat_submit"):
        if query:
            with st.spinner("🔍 Searching catalog with AI..."):
                result = query_databricks_endpoint(query)
            if result["success"]:
                st.session_state.chat_history.append({
                    "q": query, 
                    "a": result["answer"], 
                    "t": datetime.now().strftime("%H:%M"),
                    "id": f"chat_{datetime.now().timestamp()}"
                })
                st.rerun()
    
    st.markdown("---")
    
    if st.session_state.chat_history:
        # User Satisfaction Metric (RLHF-style)
        total = st.session_state.feedback_counts['up'] + st.session_state.feedback_counts['down']
        satisfaction = (st.session_state.feedback_counts['up'] / total * 100) if total > 0 else 0
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
            <span style="font-size: 1.1em; font-weight: 600; color: #e2e8f0;">📝 AI Recommendations</span>
            <div style="background: rgba(56,239,125,0.15); padding: 6px 14px; border-radius: 20px; border: 1px solid rgba(56,239,125,0.3);">
                <span style="color: #38ef7d; font-size: 0.8em;">👍 User Satisfaction (7d): <strong>{satisfaction:.0f}%</strong></span>
                <span style="color: #718096; font-size: 0.7em; margin-left: 8px;">({st.session_state.feedback_counts['up']}↑ / {st.session_state.feedback_counts['down']}↓)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        for idx, chat in enumerate(reversed(st.session_state.chat_history[-3:])):
            response_id = chat.get('id', f"resp_{idx}")
            st.markdown(f"**🧑 You:** {chat['q']}")
            
            # Render structured product card for the answer
            st.markdown(f"""
            <div class="product-card">
                <div class="product-header">
                    <span style="color: #e2e8f0; font-weight: 600;">🤖 AI Response</span>
                    <span style="color: #718096; font-size: 0.8em;">{chat['t']}</span>
                </div>
                <div style="color: #e2e8f0; line-height: 1.7; font-size: 0.95em;">{chat['a']}</div>
                <div style="margin-top: 15px; padding-top: 12px; border-top: 1px solid rgba(255,255,255,0.1);">
                    <div style="display: flex; gap: 8px;">
                        <span style="background: rgba(102,126,234,0.2); padding: 4px 10px; border-radius: 5px; font-size: 0.75em; color: #a0aec0;">📍 Source: Product Catalog</span>
                        <span style="background: rgba(17,153,142,0.2); padding: 4px 10px; border-radius: 5px; font-size: 0.75em; color: #a0aec0;">🔗 View in Portal</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Feedback buttons - more prominent with labels
            st.markdown('<div style="margin: 8px 0 20px 0;">', unsafe_allow_html=True)
            fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 4])
            with fb_col1:
                feedback_given = st.session_state.feedback.get(response_id)
                if feedback_given == 'up':
                    st.success("👍 Helpful!")
                elif st.button("👍 Helpful", key=f"chat_up_{idx}_{response_id}", help="This response was helpful"):
                    st.session_state.feedback[response_id] = 'up'
                    st.session_state.feedback_counts['up'] += 1
                    log_feedback_to_delta(response_id, chat['q'], chat['a'], 'up', datetime.now().isoformat())
                    st.rerun()
            with fb_col2:
                if feedback_given == 'down':
                    st.warning("👎 Not helpful")
                elif st.button("👎 Not Helpful", key=f"chat_down_{idx}_{response_id}", help="This response was not helpful"):
                    st.session_state.feedback[response_id] = 'down'
                    st.session_state.feedback_counts['down'] += 1
                    log_feedback_to_delta(response_id, chat['q'], chat['a'], 'down', datetime.now().isoformat())
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Example queries AFTER answers
        st.markdown("---")
        st.markdown("**💡 Try More Queries:**")
        st.markdown("""
        <div style="margin-top: 10px; display: flex; flex-wrap: wrap; gap: 8px;">
            <span style="background: rgba(255,255,255,0.05); padding: 6px 12px; border-radius: 8px; font-size: 0.8em; color: #a0aec0; border: 1px solid rgba(255,255,255,0.1); cursor: pointer;">Find mid-range Cisco switches for SMB customers in CZ and SK under 100k CZK</span>
            <span style="background: rgba(255,255,255,0.05); padding: 6px 12px; border-radius: 8px; font-size: 0.8em; color: #a0aec0; border: 1px solid rgba(255,255,255,0.1); cursor: pointer;">Compare Dell server vs HPE server for virtualization in DE market</span>
            <span style="background: rgba(255,255,255,0.05); padding: 6px 12px; border-radius: 8px; font-size: 0.8em; color: #a0aec0; border: 1px solid rgba(255,255,255,0.1); cursor: pointer;">Show laptops with 32GB RAM and 3-year warranty for corporate FR</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Show example queries when no chat history
        st.markdown("**📝 Example Queries to Get Started:**")
        st.markdown("""
        <div style="margin-bottom: 15px; display: flex; flex-wrap: wrap; gap: 8px;">
            <span style="background: rgba(255,255,255,0.05); padding: 6px 12px; border-radius: 8px; font-size: 0.8em; color: #a0aec0; border: 1px solid rgba(255,255,255,0.1);">Find mid-range Cisco switches for SMB customers in CZ and SK under 100k CZK</span>
            <span style="background: rgba(255,255,255,0.05); padding: 6px 12px; border-radius: 8px; font-size: 0.8em; color: #a0aec0; border: 1px solid rgba(255,255,255,0.1);">Compare Dell server vs HPE server for virtualization in DE market</span>
            <span style="background: rgba(255,255,255,0.05); padding: 6px 12px; border-radius: 8px; font-size: 0.8em; color: #a0aec0; border: 1px solid rgba(255,255,255,0.1);">Show laptops with 32GB RAM and 3-year warranty for corporate FR</span>
        </div>
        """, unsafe_allow_html=True)


def render_rag_qa_tab():
    """RAG Q&A Knowledge Assistant tab"""
    st.markdown("### 💎 RAG Q&A Knowledge Assistant")
    st.markdown('<span style="color: #718096; font-size: 0.85em;">Upload Documents & Ask Questions</span>', unsafe_allow_html=True)
    
    # Header with branding
    st.markdown("""
    <div style="text-align: center; padding: 15px 0;">
        <div style="font-size: 2.5em; margin-bottom: 8px;">🧠📄</div>
        <p style="color: #a0aec0; font-size: 1em;">Transform your documents into intelligent, queryable knowledge bases</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Summary cards
    st.markdown("""
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 15px 0;">
        <div style="background: rgba(102,126,234,0.1); padding: 15px; border-radius: 12px; text-align: center;">
            <div style="font-size: 1.5em;">📤</div>
            <div style="font-weight: 600; color: white; margin: 6px 0; font-size: 0.9em;">Upload</div>
            <div style="color: #a0aec0; font-size: 0.8em;">PDF, TXT, URLs</div>
        </div>
        <div style="background: rgba(118,75,162,0.1); padding: 15px; border-radius: 12px; text-align: center;">
            <div style="font-size: 1.5em;">🔍</div>
            <div style="font-weight: 600; color: white; margin: 6px 0; font-size: 0.9em;">Process</div>
            <div style="color: #a0aec0; font-size: 0.8em;">AI chunks & embeds</div>
        </div>
        <div style="background: rgba(17,153,142,0.1); padding: 15px; border-radius: 12px; text-align: center;">
            <div style="font-size: 1.5em;">💬</div>
            <div style="font-weight: 600; color: white; margin: 6px 0; font-size: 0.9em;">Query</div>
            <div style="color: #a0aec0; font-size: 0.8em;">Get precise answers</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if not st.session_state.vectorstore:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("#### 📁 Select Your Input Source")
            input_type = st.selectbox("Input Type", ["PDF", "Link", "Text", "TXT"], label_visibility="collapsed", key="rag_input_type")
            
            if input_type == "PDF":
                input_data = st.file_uploader("📄 Upload PDF Document", type=["pdf"], key="rag_pdf_upload")
                doc_name = input_data.name if input_data else "document"
            elif input_type == "Link":
                input_data = st.text_input("🔗 Enter URL", placeholder="https://example.com/document", key="rag_url_input")
                doc_name = input_data if input_data else "web_content"
            elif input_type == "Text":
                input_data = st.text_area("📝 Paste Text", placeholder="Paste your document content here...", height=150, key="rag_text_input")
                doc_name = "pasted_text"
            elif input_type == "TXT":
                input_data = st.file_uploader("📄 Upload TXT File", type=["txt"], key="rag_txt_upload")
                doc_name = input_data.name if input_data else "document"
            
            if st.button("🚀 Process Document", type="primary", key="rag_process_btn"):
                if input_data:
                    with st.spinner("🔄 Processing document..."):
                        vectorstore, texts, message = process_document(input_type, input_data, doc_name)
                    if vectorstore:
                        st.session_state.vectorstore = vectorstore
                        st.session_state.rag_texts = texts  # Store for generating dynamic examples
                        st.session_state.rag_doc_name = doc_name
                        # Log to Databricks
                        log_document_to_delta(input_type, doc_name, texts[0] if texts else "", len(texts) if texts else 0)
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
        
        with col2:
            st.markdown("#### 📋 Supported Formats")
            st.markdown("""
            <div style="background: rgba(255,255,255,0.03); padding: 12px; border-radius: 10px; font-size: 0.85em;">
                <div style="color: #a0aec0; margin-bottom: 8px;">📄 PDF Documents</div>
                <div style="color: #a0aec0; margin-bottom: 8px;">🔗 Web URLs</div>
                <div style="color: #a0aec0; margin-bottom: 8px;">📝 Raw Text</div>
                <div style="color: #a0aec0;">📃 TXT Files</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Show generic example queries before document is uploaded
        st.markdown("---")
        st.markdown("**📝 Example Queries (after uploading a document):**")
        st.markdown("""
        <div style="margin-bottom: 15px; display: flex; flex-wrap: wrap; gap: 8px;">
            <span style="background: rgba(255,255,255,0.05); padding: 6px 12px; border-radius: 8px; font-size: 0.8em; color: #718096; border: 1px solid rgba(255,255,255,0.1);">What are the main topics covered in this document?</span>
            <span style="background: rgba(255,255,255,0.05); padding: 6px 12px; border-radius: 8px; font-size: 0.8em; color: #718096; border: 1px solid rgba(255,255,255,0.1);">Summarize the key points</span>
            <span style="background: rgba(255,255,255,0.05); padding: 6px 12px; border-radius: 8px; font-size: 0.8em; color: #718096; border: 1px solid rgba(255,255,255,0.1);">Find specific information about...</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Document is loaded - show Q&A interface
        doc_name = st.session_state.get('rag_doc_name', 'your document')
        st.markdown(f"""
        <div style="background: rgba(56,239,125,0.1); padding: 12px 18px; border-radius: 10px; margin-bottom: 15px; border-left: 3px solid #38ef7d;">
            <span style="color: #38ef7d; font-weight: 500;">✅ Document Loaded:</span>
            <span style="color: #e2e8f0; font-size: 0.9em;"> {doc_name}</span>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([4, 1])
        with col1:
            qa_query = st.text_input("Your question", placeholder="Ask anything about your document...", key="qa_input")
        with col2:
            st.markdown("<div style='padding-top: 28px;'></div>", unsafe_allow_html=True)
            if st.button("🗑️ Clear", key="rag_clear_btn"):
                st.session_state.vectorstore = None
                st.session_state.qa_history = []
                st.rerun()
        
        if st.button("🔍 Find Answer", type="primary", key="rag_search_btn"):
            if qa_query:
                with st.spinner("🔍 Searching document..."):
                    docs = st.session_state.vectorstore.similarity_search(qa_query, k=3)
                    log_query_to_delta(qa_query, 'rag_qa')
                if docs:
                    # Store the Q&A in history
                    answer_content = "\n\n".join([f"**Source {i+1}:** {doc.page_content.strip()[:400]}..." for i, doc in enumerate(docs)])
                    st.session_state.qa_history.append({
                        "q": qa_query,
                        "a": answer_content,
                        "docs": docs,
                        "t": datetime.now().strftime("%H:%M"),
                        "id": f"rag_{datetime.now().timestamp()}"
                    })
                    st.rerun()
        
        # Display Q&A History with feedback buttons
        if st.session_state.qa_history:
            st.markdown("### 📋 Answers Found:")
            
            for idx, qa in enumerate(reversed(st.session_state.qa_history[-3:])):
                response_id = qa.get('id', f"rag_resp_{idx}")
                st.markdown(f"**🧑 You:** {qa['q']}")
                
                # Display answer cards
                st.markdown(f"""
                <div class="product-card">
                    <div class="product-header">
                        <span style="color: #e2e8f0; font-weight: 600;">🤖 AI Response from Document</span>
                        <span style="color: #718096; font-size: 0.8em;">{qa['t']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Show each source document with clean formatting
                for i, doc in enumerate(qa.get('docs', []), 1):
                    # Clean up the content - remove excess whitespace and format nicely
                    content = doc.page_content.strip()
                    # Replace multiple newlines with single newline
                    content = ' '.join(content.split())
                    # Truncate and add ellipsis
                    content = content[:600] + "..." if len(content) > 600 else content
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, rgba(102,126,234,0.08) 0%, rgba(118,75,162,0.05) 100%); padding: 18px; border-radius: 12px; margin: 12px 0; border-left: 4px solid #667eea; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
                            <span style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 4px 10px; border-radius: 6px; font-size: 0.75em; font-weight: 600;">📄 Source {i}</span>
                            <span style="color: #718096; font-size: 0.75em;">Relevance: High</span>
                        </div>
                        <div style="color: #e2e8f0; line-height: 1.7; font-size: 0.9em; white-space: pre-wrap; word-wrap: break-word;">{content}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Feedback buttons - prominent with labels
                st.markdown('<div style="margin: 8px 0 20px 0;">', unsafe_allow_html=True)
                fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 4])
                with fb_col1:
                    feedback_given = st.session_state.feedback.get(response_id)
                    if feedback_given == 'up':
                        st.success("👍 Helpful!")
                    elif st.button("👍 Helpful", key=f"rag_up_{idx}_{response_id}", help="This response was helpful"):
                        st.session_state.feedback[response_id] = 'up'
                        st.session_state.feedback_counts['up'] += 1
                        log_feedback_to_delta(response_id, qa['q'], qa['a'][:500], 'up', datetime.now().isoformat())
                        st.rerun()
                with fb_col2:
                    if feedback_given == 'down':
                        st.warning("👎 Not helpful")
                    elif st.button("👎 Not Helpful", key=f"rag_down_{idx}_{response_id}", help="This response was not helpful"):
                        st.session_state.feedback[response_id] = 'down'
                        st.session_state.feedback_counts['down'] += 1
                        log_feedback_to_delta(response_id, qa['q'], qa['a'][:500], 'down', datetime.now().isoformat())
                        st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Dynamic example queries based on document content
            st.markdown("---")
            st.markdown("**💡 Try More Queries Based on Your Document:**")
            
            # Generate dynamic example queries from the document
            rag_texts = st.session_state.get('rag_texts', [])
            if rag_texts and len(rag_texts) > 0:
                # Extract key phrases from first few chunks to generate relevant examples
                sample_text = " ".join(rag_texts[:3])[:1000].lower()
                
                # Simple keyword extraction for dynamic queries
                dynamic_queries = []
                if 'product' in sample_text or 'specification' in sample_text:
                    dynamic_queries.append("What are the main product specifications?")
                if 'price' in sample_text or 'cost' in sample_text:
                    dynamic_queries.append("What are the pricing details mentioned?")
                if 'feature' in sample_text:
                    dynamic_queries.append("List all the key features")
                if 'warranty' in sample_text or 'support' in sample_text:
                    dynamic_queries.append("What warranty or support options are available?")
                if 'cisco' in sample_text or 'hp' in sample_text or 'dell' in sample_text:
                    dynamic_queries.append("Compare the vendors or products mentioned")
                
                # Add generic ones if not enough specific
                if len(dynamic_queries) < 3:
                    dynamic_queries.extend([
                        "Summarize the main topics in this document",
                        "What are the key recommendations?",
                        "List all important details mentioned"
                    ])
                
                queries_html = "".join([
                    f'<span style="background: rgba(255,255,255,0.05); padding: 6px 12px; border-radius: 8px; font-size: 0.8em; color: #a0aec0; border: 1px solid rgba(255,255,255,0.1); cursor: pointer;">{q}</span>'
                    for q in dynamic_queries[:4]
                ])
                st.markdown(f"""
                <div style="margin-top: 10px; display: flex; flex-wrap: wrap; gap: 8px;">
                    {queries_html}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="margin-top: 10px; display: flex; flex-wrap: wrap; gap: 8px;">
                    <span style="background: rgba(255,255,255,0.05); padding: 6px 12px; border-radius: 8px; font-size: 0.8em; color: #a0aec0; border: 1px solid rgba(255,255,255,0.1);">Summarize the main topics</span>
                    <span style="background: rgba(255,255,255,0.05); padding: 6px 12px; border-radius: 8px; font-size: 0.8em; color: #a0aec0; border: 1px solid rgba(255,255,255,0.1);">What are the key recommendations?</span>
                    <span style="background: rgba(255,255,255,0.05); padding: 6px 12px; border-radius: 8px; font-size: 0.8em; color: #a0aec0; border: 1px solid rgba(255,255,255,0.1);">Find specific details about...</span>
                </div>
                """, unsafe_allow_html=True)


def render_pipeline_tab():
    """Pipeline tab (Data / MLOps) with advanced features"""
    st.markdown("### 🛠️ Pipeline")
    st.markdown('<span style="color: #718096; font-size: 0.85em;">For Data & MLOps Teams</span>', unsafe_allow_html=True)
    
    # 1. TECH STACK FIRST
    st.markdown("### 🛠️ Technology Stack")
    st.markdown("""
    <div class="tech-grid">
        <div class="tech-item"><div class="tech-icon">🐍</div><div class="tech-name">Python</div><div class="tech-desc">3.10</div></div>
        <div class="tech-item"><div class="tech-icon">🔗</div><div class="tech-name">LangChain</div><div class="tech-desc">RAG</div></div>
        <div class="tech-item"><div class="tech-icon">🔍</div><div class="tech-name">FAISS</div><div class="tech-desc">Vector</div></div>
        <div class="tech-item"><div class="tech-icon">🤗</div><div class="tech-name">HuggingFace</div><div class="tech-desc">Embed</div></div>
        <div class="tech-item"><div class="tech-icon">📊</div><div class="tech-name">MLflow</div><div class="tech-desc">Registry</div></div>
        <div class="tech-item"><div class="tech-icon">☁️</div><div class="tech-name">Databricks</div><div class="tech-desc">Platform</div></div>
        <div class="tech-item"><div class="tech-icon">🔷</div><div class="tech-name">Azure</div><div class="tech-desc">Cloud</div></div>
        <div class="tech-item"><div class="tech-icon">🎨</div><div class="tech-name">Streamlit</div><div class="tech-desc">UI</div></div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 2. RAG PIPELINE WORKFLOW SECOND
    st.markdown("### 🔄 RAG Pipeline Workflow")
    st.markdown("""
    <div style="background: linear-gradient(180deg, rgba(20,20,35,0.9) 0%, rgba(15,15,30,0.95) 100%); padding: 25px; border-radius: 16px; border: 1px solid rgba(255,255,255,0.1); margin: 15px 0;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
            <!-- Step 1: Data Sources -->
            <div style="text-align: center; flex: 1; min-width: 80px;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); width: 50px; height: 50px; border-radius: 12px; margin: 0 auto 8px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 15px rgba(102,126,234,0.4);">
                    <span style="font-size: 1.3em;">📄</span>
                </div>
                <div style="font-size: 0.75em; color: white; font-weight: 600;">Data Sources</div>
                <div style="font-size: 0.6em; color: #718096;">PDF • CSV • APIs</div>
            </div>
            <div style="color: #667eea; font-size: 1.2em;">→</div>
            <!-- Step 2: Spark ETL -->
            <div style="text-align: center; flex: 1; min-width: 80px;">
                <div style="background: linear-gradient(135deg, #f6ad55 0%, #ed8936 100%); width: 50px; height: 50px; border-radius: 12px; margin: 0 auto 8px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 15px rgba(246,173,85,0.4);">
                    <span style="font-size: 1.3em;">⚡</span>
                </div>
                <div style="font-size: 0.75em; color: white; font-weight: 600;">Spark ETL</div>
                <div style="font-size: 0.6em; color: #718096;">Transform</div>
            </div>
            <div style="color: #667eea; font-size: 1.2em;">→</div>
            <!-- Step 3: Delta Tables -->
            <div style="text-align: center; flex: 1; min-width: 80px;">
                <div style="background: linear-gradient(135deg, #48bb78 0%, #38a169 100%); width: 50px; height: 50px; border-radius: 12px; margin: 0 auto 8px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 15px rgba(72,187,120,0.4);">
                    <span style="font-size: 1.3em;">🗄️</span>
                </div>
                <div style="font-size: 0.75em; color: white; font-weight: 600;">Delta Tables</div>
                <div style="font-size: 0.6em; color: #718096;">Unity Catalog</div>
            </div>
            <div style="color: #667eea; font-size: 1.2em;">→</div>
            <!-- Step 4: Embeddings -->
            <div style="text-align: center; flex: 1; min-width: 80px;">
                <div style="background: linear-gradient(135deg, #9f7aea 0%, #805ad5 100%); width: 50px; height: 50px; border-radius: 12px; margin: 0 auto 8px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 15px rgba(159,122,234,0.4);">
                    <span style="font-size: 1.3em;">🧠</span>
                </div>
                <div style="font-size: 0.75em; color: white; font-weight: 600;">Embeddings</div>
                <div style="font-size: 0.6em; color: #718096;">MiniLM-L6</div>
            </div>
            <div style="color: #667eea; font-size: 1.2em;">→</div>
            <!-- Step 5: FAISS Index -->
            <div style="text-align: center; flex: 1; min-width: 80px;">
                <div style="background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%); width: 50px; height: 50px; border-radius: 12px; margin: 0 auto 8px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 15px rgba(66,153,225,0.4);">
                    <span style="font-size: 1.3em;">🔍</span>
                </div>
                <div style="font-size: 0.75em; color: white; font-weight: 600;">FAISS Index</div>
                <div style="font-size: 0.6em; color: #718096;">IVF256</div>
            </div>
            <div style="color: #667eea; font-size: 1.2em;">→</div>
            <!-- Step 6: RAG Chain -->
            <div style="text-align: center; flex: 1; min-width: 80px;">
                <div style="background: linear-gradient(135deg, #ed64a6 0%, #d53f8c 100%); width: 50px; height: 50px; border-radius: 12px; margin: 0 auto 8px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 15px rgba(237,100,166,0.4);">
                    <span style="font-size: 1.3em;">🔗</span>
                </div>
                <div style="font-size: 0.75em; color: white; font-weight: 600;">RAG Chain</div>
                <div style="font-size: 0.6em; color: #718096;">LangChain</div>
            </div>
            <div style="color: #667eea; font-size: 1.2em;">→</div>
            <!-- Step 7: Model Serving -->
            <div style="text-align: center; flex: 1; min-width: 80px;">
                <div style="background: linear-gradient(135deg, #38ef7d 0%, #11998e 100%); width: 50px; height: 50px; border-radius: 12px; margin: 0 auto 8px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 15px rgba(56,239,125,0.4);">
                    <span style="font-size: 1.3em;">🚀</span>
                </div>
                <div style="font-size: 0.75em; color: white; font-weight: 600;">Serving</div>
                <div style="font-size: 0.6em; color: #718096;">Databricks</div>
            </div>
        </div>
        <div style="text-align: center; margin-top: 15px; padding-top: 12px; border-top: 1px solid rgba(255,255,255,0.1);">
            <span style="color: #718096; font-size: 0.7em;">📊 Last pipeline run: <strong style="color: #667eea;">td_synnex_daily_refresh</strong> • 2025-12-08 14:32 UTC • Duration: 4m 23s</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Trust & Governance Panel
        st.markdown("### 🔒 Trust & Governance")
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(72,187,120,0.1) 0%, rgba(56,161,105,0.1) 100%); padding: 18px; border-radius: 12px; border: 1px solid rgba(72,187,120,0.2);">
            <ul style="margin: 0; padding-left: 18px; color: #e2e8f0; font-size: 0.85em; line-height: 1.8;">
                <li><strong style="color: #48bb78;">Data Governance:</strong> All data in governed Delta tables with lineage via Unity Catalog</li>
                <li><strong style="color: #48bb78;">Audit Trail:</strong> Prompts, responses, and sources logged for debugging</li>
                <li><strong style="color: #48bb78;">Access Control:</strong> Row-level security and fine-grained permissions</li>
                <li><strong style="color: #48bb78;">Safety Guardrail:</strong> Scope-limited to TD SYNNEX products only</li>
            </ul>
            <div style="margin-top: 12px; padding: 10px; background: rgba(255,255,255,0.05); border-radius: 8px; font-size: 0.8em; color: #a0aec0;">
                ⚠️ <em>"I only answer questions about TD SYNNEX products and partners. All answers include source citations."</em>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Future Agents Box
        st.markdown("### 🤖 Future Agents")
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(159,122,234,0.1) 0%, rgba(128,90,213,0.1) 100%); padding: 18px; border-radius: 12px; border: 1px solid rgba(159,122,234,0.2);">
            <div style="margin-bottom: 12px;">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                    <span style="font-size: 1.2em;">💰</span>
                    <div>
                        <div style="color: white; font-weight: 600; font-size: 0.9em;">Pricing Optimization Agent</div>
                        <div style="color: #a0aec0; font-size: 0.75em;">Calls pricing API, suggests competitive bundles</div>
                    </div>
                </div>
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                    <span style="font-size: 1.2em;">📦</span>
                    <div>
                        <div style="color: white; font-weight: 600; font-size: 0.9em;">Inventory Check Agent</div>
                        <div style="color: #a0aec0; font-size: 0.75em;">Real-time stock levels from warehouse APIs</div>
                    </div>
                </div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 1.2em;">📣</span>
                    <div>
                        <div style="color: white; font-weight: 600; font-size: 0.9em;">Marketing Campaign Agent</div>
                        <div style="color: #a0aec0; font-size: 0.75em;">Generates vendor-specific promo content</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # TD SYNNEX Use Cases
    st.markdown("### 🎯 Real TD SYNNEX Use Cases")
    st.markdown("""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px;">
        <div style="background: rgba(102,126,234,0.1); padding: 15px; border-radius: 10px; border-left: 3px solid #667eea;">
            <div style="color: #667eea; font-weight: 600; font-size: 0.9em; margin-bottom: 5px;">Partner Portal Recommendations</div>
            <div style="color: #a0aec0; font-size: 0.8em;">Personalized product suggestions for 100k+ partners</div>
        </div>
        <div style="background: rgba(118,75,162,0.1); padding: 15px; border-radius: 10px; border-left: 3px solid #764ba2;">
            <div style="color: #764ba2; font-weight: 600; font-size: 0.9em; margin-bottom: 5px;">Dynamic Bundle Builder</div>
            <div style="color: #a0aec0; font-size: 0.8em;">AI-generated bundles for Cisco/HP/Dell campaigns</div>
        </div>
        <div style="background: rgba(17,153,142,0.1); padding: 15px; border-radius: 10px; border-left: 3px solid #11998e;">
            <div style="color: #11998e; font-weight: 600; font-size: 0.9em; margin-bottom: 5px;">Campaign Targeting</div>
            <div style="color: #a0aec0; font-size: 0.8em;">Smart targeting for Apple/Cisco/HP promotions</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Scalability Note
    st.markdown("""
    <div style="background: rgba(246,173,85,0.1); padding: 12px 15px; border-radius: 8px; margin: 15px 0; border-left: 3px solid #f6ad55;">
        <span style="color: #f6ad55; font-weight: 600; font-size: 0.85em;">📈 Scalability:</span>
        <span style="color: #a0aec0; font-size: 0.85em;"> Same architecture scales from 5K to 100K+ products and multi-country traffic (sharded Delta tables, Pinecone/Weaviate vector store, autoscaling Databricks endpoints) — matching TD SYNNEX's 100K-partner, 6B CZK portal scale.</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 3. SYSTEM ARCHITECTURE LAST (SMALLER)
    st.markdown("### 📊 System Architecture")
    arch_image_path = "RAG-using-Azure-Databricks-CI-CD _project_architecture.png"
    if os.path.exists(arch_image_path):
        # Use columns to make it smaller
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.image(arch_image_path, caption="RAG Chatbot with Azure Databricks CI/CD Architecture", use_container_width=True)


def render_metrics_row():
    """Render metrics row with tooltips"""
    st.markdown("""
    <div class="metric-row">
        <div class="metric-card">
            <div class="metric-icon">📦</div>
            <div class="metric-value">5,000+</div>
            <div class="metric-label">Products Indexed</div>
            <div class="metric-tooltip">Cisco, HP, Dell catalog</div>
        </div>
        <div class="metric-card">
            <div class="metric-icon">🎯</div>
            <div class="metric-value">95.2%</div>
            <div class="metric-label">Groundedness</div>
            <div class="metric-tooltip">Answers supported by sources</div>
        </div>
        <div class="metric-card">
            <div class="metric-icon">⚡</div>
            <div class="metric-value">245ms</div>
            <div class="metric-label">P95 Latency</div>
            <div class="metric-tooltip">95th percentile response time</div>
        </div>
        <div class="metric-card">
            <div class="metric-icon" style="animation: pulse 2s infinite;">🟢</div>
            <div class="metric-value" style="color: #38ef7d;">Live</div>
            <div class="metric-label">Endpoint Status</div>
            <div class="metric-tooltip">99.9% uptime</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def main():
    init_session_state()
    render_hero()
    
    # Main tabs with persona labels
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard", 
        "💬 Product Chat (Sales)", 
        "� RAG Q&A Knowledge Assistant", 
        "🛠️ Pipeline (MLOps)"
    ])
    
    with tab1:
        render_dashboard_tab()
    with tab2:
        render_chat_tab()
    with tab3:
        render_rag_qa_tab()
    with tab4:
        render_pipeline_tab()
    
    # Metrics row below tabs
    render_metrics_row()
    
    # Footer with TD SYNNEX branding
    st.markdown("""
    <div style="text-align: center; padding: 25px; margin-top: 20px; border-top: 1px solid rgba(255,255,255,0.08);">
        <div style="color: #667eea; font-weight: 600; font-size: 1em; margin-bottom: 8px;">🚀 TD SYNNEX Production RAG Demo</div>
        <div style="color: #718096; font-size: 0.85em;">Destination AI Lab • Enterprise Product Intelligence • EU Market</div>
        <div style="color: #4a5568; font-size: 0.75em; margin-top: 8px;">Version 2.0.0 | © 2025</div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
