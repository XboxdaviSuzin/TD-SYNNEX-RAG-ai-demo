"""
TD SYNNEX Partner Chat Assistant
Conversational AI interface powered by RAG

Run with: streamlit run chat_app.py --server.port=8503
"""

import streamlit as st
import requests
import json
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="TD SYNNEX Partner Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Databricks Endpoint Configuration
DATABRICKS_ENDPOINT = "https://adb-3630242710149273.13.azuredatabricks.net/serving-endpoints/td_synnex_rag_endpoint/invocations"
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "")

# Custom CSS for chat interface
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a5f 0%, #1a1a2e 100%);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Chat container */
    .chat-container {
        max-width: 900px;
        margin: 0 auto;
        padding: 20px;
    }
    
    /* Message bubbles */
    .assistant-message {
        background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
        border-radius: 15px 15px 15px 5px;
        padding: 15px 20px;
        margin: 10px 0;
        border-left: 4px solid #667eea;
        color: #e2e8f0;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px 15px 5px 15px;
        padding: 15px 20px;
        margin: 10px 0;
        color: white;
        text-align: right;
        margin-left: auto;
        max-width: 80%;
    }
    
    /* Avatar styling */
    .avatar-container {
        text-align: center;
        padding: 20px;
    }
    
    .avatar-img {
        width: 150px;
        height: 150px;
        border-radius: 50%;
        border: 4px solid #667eea;
        object-fit: cover;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 25px;
        padding: 15px 20px;
        color: white;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 30px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
    }
    
    /* Code block */
    .sql-block {
        background: #1e1e1e;
        border-radius: 10px;
        padding: 15px;
        font-family: monospace;
        color: #e2e8f0;
        overflow-x: auto;
    }
    
    /* Welcome card */
    .welcome-card {
        background: linear-gradient(135deg, rgba(102,126,234,0.2) 0%, rgba(118,75,162,0.2) 100%);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state for chat history"""
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hi, I'm your TD SYNNEX Partner Assistant! 👋 How can I help you today? Ask me about Cisco, HP, or Dell products.",
                "timestamp": datetime.now().strftime("%H:%M")
            }
        ]


def query_databricks_endpoint(query: str) -> dict:
    """Query the real Databricks RAG endpoint"""
    try:
        headers = {
            "Authorization": f"Bearer {DATABRICKS_TOKEN}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "dataframe_records": [{"query": query}]
        }
        
        response = requests.post(DATABRICKS_ENDPOINT, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if "predictions" in result and result["predictions"]:
                return {"success": True, "response": result["predictions"][0]}
        
        return {"success": False, "error": f"Error: {response.status_code}"}
    
    except Exception as e:
        return {"success": False, "error": str(e)}


def render_sidebar():
    """Render the sidebar with avatar and info"""
    with st.sidebar:
        # Logo
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h1 style="color: #667eea; margin: 0;">TD SYNNEX</h1>
            <p style="color: #a0aec0; font-size: 0.9em;">Partner Assistant</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Avatar placeholder (using emoji as placeholder)
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <div style="width: 150px; height: 150px; border-radius: 50%; 
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        margin: 0 auto; display: flex; align-items: center; justify-content: center;
                        font-size: 80px; border: 4px solid rgba(255,255,255,0.2);">
                🤖
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center; padding: 10px 0;">
            <p style="color: #e2e8f0;">Welcome! I'm your TD SYNNEX virtual assistant powered by RAG/LLM technology.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Hi, I'm your TD SYNNEX Partner Assistant! 👋 How can I help you today?",
                    "timestamp": datetime.now().strftime("%H:%M")
                }
            ]
            st.rerun()
        
        st.markdown("---")
        
        # Expandable sections
        with st.expander("📋 My scope"):
            st.markdown("""
            - Cisco/HP/Dell product catalog
            - 5,000 EU products
            - Pricing in CZK & EUR
            - Partner segment info
            - Q3 revenue trends
            """)
        
        with st.expander("⚠️ My limitations"):
            st.markdown("""
            - I work with catalog data only
            - No real-time inventory
            - No order placement
            - Demo mode only
            """)
        
        with st.expander("📖 User Guide"):
            st.markdown("""
            **Example queries:**
            - "Best Cisco switch for SMB"
            - "HP server for healthcare"
            - "Dell storage under 500k CZK"
            - "Wireless solution for office"
            """)


def render_chat():
    """Render the main chat interface"""
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Chat header
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h2 style="color: #e2e8f0;">💬 Partner Product Assistant</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "assistant":
            col1, col2 = st.columns([1, 12])
            with col1:
                st.markdown("🤖")
            with col2:
                st.markdown(f"""
                <div class="assistant-message">
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
                
                # Show SQL/Source if available
                if "sql" in message:
                    with st.expander("📊 Show SQL"):
                        st.code(message["sql"], language="sql")
        else:
            st.markdown(f"""
            <div class="user-message">
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_input():
    """Render the chat input"""
    st.markdown("<br>" * 2, unsafe_allow_html=True)
    
    col1, col2 = st.columns([6, 1])
    
    with col1:
        user_input = st.text_input(
            "Type your question...",
            key="user_input",
            placeholder="Type in your question...",
            label_visibility="collapsed"
        )
    
    with col2:
        send_clicked = st.button("➤", type="primary", use_container_width=True)
    
    if send_clicked and user_input:
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().strftime("%H:%M")
        })
        
        # Get response from Databricks
        with st.spinner("Thinking..."):
            result = query_databricks_endpoint(user_input)
        
        if result["success"]:
            # Generate mock SQL for demo
            mock_sql = f"""SELECT DISTINCT product.product_id, product.vendor, product.model, product.price_czk
FROM td_synnex.rag_schema.td_synnex_products product
WHERE product.description LIKE '%{user_input.split()[0]}%'
ORDER BY product.price_czk ASC
LIMIT 5;"""
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["response"],
                "timestamp": datetime.now().strftime("%H:%M"),
                "sql": mock_sql
            })
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"I apologize, but I encountered an error: {result.get('error', 'Unknown error')}. Please try again.",
                "timestamp": datetime.now().strftime("%H:%M")
            })
        
        st.rerun()


def main():
    """Main function"""
    init_session_state()
    render_sidebar()
    render_chat()
    render_input()
    
    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 20px; color: #a0aec0; font-size: 0.8em;">
        Powered by TD SYNNEX RAG | Databricks + LangChain + FAISS
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
