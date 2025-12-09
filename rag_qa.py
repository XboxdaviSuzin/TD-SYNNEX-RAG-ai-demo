"""
TD SYNNEX Advanced RAG Q&A System
Upload documents (PDF, DOCX, TXT) or paste URLs and get AI-powered answers

Based on: https://github.com/balupeddireddy08/Advanced-RAG-Q-A-System
Integrated with TD SYNNEX RAG Demo

Run with: streamlit run rag_qa.py --server.port=8504
"""

import streamlit as st
import os
import numpy as np
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="TD SYNNEX RAG Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    .upload-box {
        background: rgba(255,255,255,0.05);
        border: 2px dashed rgba(102,126,234,0.5);
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
    }
    
    .qa-card {
        background: linear-gradient(135deg, rgba(102,126,234,0.2) 0%, rgba(118,75,162,0.2) 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        border-left: 4px solid #667eea;
    }
    
    .answer-box {
        background: rgba(17, 153, 142, 0.2);
        border-radius: 15px;
        padding: 20px;
        border-left: 4px solid #11998e;
        margin: 15px 0;
    }
    
    .source-chip {
        background: rgba(102,126,234,0.3);
        padding: 5px 12px;
        border-radius: 15px;
        display: inline-block;
        margin: 3px;
        font-size: 0.85em;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state"""
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'qa_history' not in st.session_state:
        st.session_state.qa_history = []
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
    if 'saved_to_databricks' not in st.session_state:
        st.session_state.saved_to_databricks = False


# Databricks Configuration
DATABRICKS_HOST = "https://adb-3630242710149273.13.azuredatabricks.net"
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "")
CATALOG = "rusefx"
SCHEMA = "rag_schema"


def save_to_databricks(document_name: str, document_type: str, chunks: list, embeddings_list: list = None):
    """Save processed documents to Databricks Delta Table."""
    import requests
    import json
    from datetime import datetime
    
    if not DATABRICKS_TOKEN:
        return False, "Databricks token not configured"
    
    try:
        headers = {
            "Authorization": f"Bearer {DATABRICKS_TOKEN}",
            "Content-Type": "application/json"
        }
        
        # First, get available SQL warehouses
        warehouses_url = f"{DATABRICKS_HOST}/api/2.0/sql/warehouses"
        wh_response = requests.get(warehouses_url, headers=headers, timeout=30)
        
        warehouse_id = None
        if wh_response.status_code == 200:
            warehouses = wh_response.json().get("warehouses", [])
            # Find a running warehouse or use first available
            for wh in warehouses:
                if wh.get("state") == "RUNNING":
                    warehouse_id = wh.get("id")
                    break
            if not warehouse_id and warehouses:
                warehouse_id = warehouses[0].get("id")
        
        if not warehouse_id:
            # Fallback: save to DBFS as JSON instead
            return save_to_dbfs(document_name, document_type, chunks)
        
        # Create table and insert using SQL Statements API
        sql_endpoint = f"{DATABRICKS_HOST}/api/2.0/sql/statements"
        
        # Create table
        create_sql = f"""
            CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.uploaded_documents (
                document_id STRING,
                document_name STRING,
                document_type STRING,
                chunk_id INT,
                chunk_text STRING,
                uploaded_at STRING
            ) USING DELTA
        """
        
        payload = {
            "warehouse_id": warehouse_id,
            "statement": create_sql,
            "wait_timeout": "30s"
        }
        
        response = requests.post(sql_endpoint, headers=headers, json=payload, timeout=60)
        
        # Insert data
        timestamp = datetime.now().isoformat()
        doc_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        for i, chunk in enumerate(chunks[:20]):  # Limit to 20 chunks for demo
            escaped_text = chunk.replace("'", "''").replace("\n", " ")[:500]
            
            insert_sql = f"""
                INSERT INTO {CATALOG}.{SCHEMA}.uploaded_documents 
                VALUES ('{doc_id}', '{document_name}', '{document_type}', {i}, '{escaped_text}', '{timestamp}')
            """
            
            payload = {
                "warehouse_id": warehouse_id,
                "statement": insert_sql,
                "wait_timeout": "10s"
            }
            requests.post(sql_endpoint, headers=headers, json=payload, timeout=30)
        
        return True, f"✅ Saved {min(len(chunks), 20)} chunks to Databricks ({CATALOG}.{SCHEMA}.uploaded_documents)"
        
    except Exception as e:
        return False, f"❌ Databricks save failed: {str(e)}"


def save_to_dbfs(document_name: str, document_type: str, chunks: list):
    """Fallback: Save to DBFS as JSON if no SQL warehouse available."""
    import requests
    import json
    from datetime import datetime
    import base64
    
    try:
        headers = {
            "Authorization": f"Bearer {DATABRICKS_TOKEN}",
            "Content-Type": "application/json"
        }
        
        # Prepare data
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        doc_id = f"doc_{timestamp}"
        
        data = {
            "document_id": doc_id,
            "document_name": document_name,
            "document_type": document_type,
            "chunks": chunks[:50],
            "uploaded_at": datetime.now().isoformat()
        }
        
        # Save to DBFS
        dbfs_path = f"/FileStore/rag_documents/{doc_id}.json"
        content = base64.b64encode(json.dumps(data, indent=2).encode()).decode()
        
        # Create directory
        requests.post(
            f"{DATABRICKS_HOST}/api/2.0/dbfs/mkdirs",
            headers=headers,
            json={"path": "/FileStore/rag_documents"}
        )
        
        # Write file
        response = requests.post(
            f"{DATABRICKS_HOST}/api/2.0/dbfs/put",
            headers=headers,
            json={"path": dbfs_path, "contents": content, "overwrite": True}
        )
        
        if response.status_code == 200:
            return True, f"✅ Saved to DBFS: {dbfs_path}"
        else:
            return False, f"DBFS save failed: {response.text}"
            
    except Exception as e:
        return False, f"DBFS save failed: {str(e)}"



def process_input(input_type, input_data, document_name="uploaded_document"):
    """Process different input types and return a vectorstore."""
    try:
        from langchain_text_splitters import CharacterTextSplitter
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain_community.docstore.in_memory import InMemoryDocstore
        import faiss
        
        documents = ""
        
        if input_type == "Link":
            from langchain_community.document_loaders import WebBaseLoader
            loader = WebBaseLoader(input_data)
            docs = loader.load()
            documents = "\n".join([doc.page_content for doc in docs])
            
        elif input_type == "PDF":
            from PyPDF2 import PdfReader
            pdf_reader = PdfReader(BytesIO(input_data.read()))
            for page in pdf_reader.pages:
                documents += page.extract_text() or ""
                
        elif input_type == "Text":
            documents = input_data
            
        elif input_type == "DOCX":
            from docx import Document
            doc = Document(BytesIO(input_data.read()))
            documents = "\n".join([para.text for para in doc.paragraphs])
            
        elif input_type == "TXT":
            documents = input_data.read().decode('utf-8')
        
        if not documents.strip():
            return None, None, "No text content found in the input."
        
        # Split text into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_text(documents)
        
        # Create embeddings
        hf_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Create FAISS vector store
        sample_embedding = np.array(hf_embeddings.embed_query("sample"))
        dimension = sample_embedding.shape[0]
        index = faiss.IndexFlatL2(dimension)
        
        vector_store = FAISS(
            embedding_function=hf_embeddings.embed_query,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        vector_store.add_texts(texts)
        
        return vector_store, texts, f"✅ Processed {len(texts)} text chunks successfully!"
        
    except Exception as e:
        return None, None, f"❌ Error: {str(e)}"



def answer_question(vectorstore, query):
    """Answer a question using the vectorstore."""
    try:
        # Simple similarity search and response
        docs = vectorstore.similarity_search(query, k=3)
        
        if not docs:
            return "No relevant information found."
        
        # Format response cleanly
        response = "### 📋 Answer\n\n"
        
        for i, doc in enumerate(docs, 1):
            # Clean up the text
            content = doc.page_content.strip()
            # Limit each chunk for readability
            if len(content) > 500:
                content = content[:500] + "..."
            
            response += f"**Source {i}:**\n"
            response += f"> {content}\n\n"
        
        response += "---\n"
        response += f"*Found {len(docs)} relevant sections from your documents.*"
        
        return response
        
    except Exception as e:
        return f"Error answering question: {str(e)}"


def render_sidebar():
    """Render sidebar"""
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <h1 style="color: #667eea;">📄 RAG Q&A</h1>
            <p style="color: #a0aec0;">Document Intelligence</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Status
        st.markdown("### 📊 Status")
        if st.session_state.vectorstore:
            st.success("✅ Documents Loaded")
        else:
            st.info("📤 Upload documents to start")
        
        st.markdown("---")
        
        # Supported formats
        st.markdown("### 📁 Supported Formats")
        st.markdown("""
        - 📄 **PDF** - PDF documents
        - 📝 **DOCX** - Word documents
        - 📃 **TXT** - Text files
        - 🔗 **URL** - Web pages
        - ✏️ **Text** - Paste text directly
        """)
        
        st.markdown("---")
        
        # Clear button
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.vectorstore = None
            st.session_state.qa_history = []
            st.session_state.documents_loaded = False
            st.rerun()
        
        # Attribution
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; font-size: 0.8em; color: #a0aec0;">
            Based on <a href="https://github.com/balupeddireddy08/Advanced-RAG-Q-A-System" target="_blank">Advanced-RAG-Q-A-System</a>
        </div>
        """, unsafe_allow_html=True)


def render_main():
    """Render main content"""
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1>🔍 Advanced RAG Q&A System</h1>
        <p style="color: #a0aec0; font-size: 1.1em;">
            Upload documents • Ask questions • Get AI-powered answers
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Document Upload Section
    if not st.session_state.vectorstore:
        st.markdown("### 📤 Step 1: Upload Your Documents")
        
        input_type = st.selectbox(
            "Select Input Type",
            ["PDF", "DOCX", "TXT", "Link", "Text"],
            help="Choose the type of input you want to process"
        )
        
        input_data = None
        doc_name = "uploaded_document"
        
        if input_type == "PDF":
            input_data = st.file_uploader("Upload a PDF file", type=["pdf"])
            if input_data:
                doc_name = input_data.name
        elif input_type == "DOCX":
            input_data = st.file_uploader("Upload a Word document", type=["docx", "doc"])
            if input_data:
                doc_name = input_data.name
        elif input_type == "TXT":
            input_data = st.file_uploader("Upload a text file", type=["txt"])
            if input_data:
                doc_name = input_data.name
        elif input_type == "Link":
            input_data = st.text_input("Enter URL", placeholder="https://example.com/article")
            doc_name = "web_page"
        elif input_type == "Text":
            input_data = st.text_area("Paste your text", height=200, placeholder="Paste your text content here...")
            doc_name = "pasted_text"
        
        # Databricks save option
        save_to_db = st.checkbox("💾 Save to Databricks (Unity Catalog)", value=True, 
                                  help="Store document chunks in Databricks Delta Table for persistence")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Process Documents", type="primary", use_container_width=True):
                if input_data:
                    with st.spinner("📊 Processing documents and creating vector index..."):
                        vectorstore, chunks, message = process_input(input_type, input_data, doc_name)
                        
                    if vectorstore:
                        st.session_state.vectorstore = vectorstore
                        st.session_state.chunks = chunks
                        st.success(message)
                        
                        # Save to Databricks if enabled
                        if save_to_db and chunks:
                            with st.spinner("☁️ Saving to Databricks..."):
                                success, db_message = save_to_databricks(doc_name, input_type, chunks)
                            if success:
                                st.success(db_message)
                                st.session_state.saved_to_databricks = True
                            else:
                                st.warning(db_message)
                        
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.warning("Please provide input data first.")
    
    # Q&A Section
    else:
        st.markdown("### ✅ Documents Loaded Successfully!")
        st.info("Your documents have been processed and indexed. You can now ask questions.")
        
        st.markdown("---")
        st.markdown("### 💬 Step 2: Ask Questions")
        
        query = st.text_input(
            "Your question",
            placeholder="What is the main topic of the document?",
            label_visibility="collapsed"
        )
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔍 Get Answer", type="primary", use_container_width=True):
                if query:
                    with st.spinner("🔍 Searching documents..."):
                        answer = answer_question(st.session_state.vectorstore, query)
                    
                    st.session_state.qa_history.append({
                        "question": query,
                        "answer": answer
                    })
        
        # Display Q&A history
        if st.session_state.qa_history:
            st.markdown("---")
            st.markdown("### 📝 Q&A History")
            
            for i, qa in enumerate(reversed(st.session_state.qa_history)):
                with st.expander(f"❓ {qa['question'][:50]}...", expanded=(i == 0)):
                    st.markdown(f"**Question:** {qa['question']}")
                    st.markdown("---")
                    st.markdown(f"**Answer:**")
                    st.markdown(qa['answer'])
    
    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 30px; color: #a0aec0; font-size: 0.9em;">
        Powered by LangChain + FAISS + HuggingFace Embeddings
    </div>
    """, unsafe_allow_html=True)


def main():
    init_session_state()
    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()
