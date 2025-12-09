# RAG Chatbot with Azure Databricks CI/CD - Project Summary

## 📋 Version: 2.0.0

**Release Date:** 2025-12-09  
**Status:** ✅ Complete & Production Ready

---

## 🎯 Project Overview

This project implements a **Retrieval-Augmented Generation (RAG) Chatbot** using Azure Databricks with a complete CI/CD pipeline. It serves as a base template for building production-ready MLOps pipelines.

### Key Features

- ✅ RAG-based Q&A chatbot using LangChain
- ✅ MLflow integration for experiment tracking and model registry
- ✅ Unity Catalog for data governance
- ✅ Delta Tables for data storage
- ✅ GitHub Actions CI/CD pipeline
- ✅ Azure Service Principal authentication
- ✅ Serverless and cluster compute support

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        GitHub Repository                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Push to   │──│  GitHub     │──│   Databricks Workflows  │  │
│  │   main      │  │   Actions   │  │   (Staging/Production)  │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Azure Databricks Workspace                    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    RAG Chatbot Pipeline                      ││
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐ ││
│  │  │   PDF    │──│  Build   │──│ Evaluate │──│   Deploy     │ ││
│  │  │ Process  │  │ Chatbot  │  │  Model   │  │  Endpoint    │ ││
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────┘ ││
│  └─────────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Unity Catalog: rusefx.rag_schema                           ││
│  │  ├── pdf_chunks (embeddings)                                ││
│  │  ├── model_info (MLflow run IDs)                            ││
│  │  ├── evaluation_results                                      ││
│  │  └── deployment_info                                         ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
RAG-using-Azure-Databricks-CI-CD/
├── .github/
│   └── workflows/
│       ├── deploy-cicd.yml              # Manual CI/CD configuration
│       ├── LLMOps-bundle-ci.yml         # CI validation on PRs
│       ├── LLMOps-bundle-cd-staging.yml # Deploy to staging (main branch)
│       ├── LLMOps-bundle-cd-prod.yml    # Deploy to production (release branch)
│       └── LLMOps-run-tests.yml         # Unit & integration tests
├── notebooks/
│   ├── 01_PDF_Advance_Data_Preprocessing.py  # Process PDFs → embeddings
│   ├── 02_Advanced_Chatbot_Chain.py          # Build RAG chatbot + MLflow
│   ├── 03_Offline_Evaluation.py              # Evaluate model quality
│   └── 04_Deploy_Model_As_Endpoint.py        # Deploy to Model Serving
├── utils/
│   ├── __init__.py
│   └── config.py                        # Secure configuration management
├── .env.example                         # Environment variables template
├── .gitignore                           # Protects sensitive files
├── SECURITY.md                          # Security best practices
├── PROJECT_SUMMARY.md                   # This file
└── README.md                            # Original project README
```

---

## 🔧 Configuration Required

### 1. Azure Resources

| Resource | Purpose |
|----------|---------|
| Azure Resource Group | Container for all resources |
| Azure Storage (ADLS Gen2) | Metastore storage |
| Azure Databricks Workspace | ML platform |

### 2. GitHub Secrets

| Secret Name | Description |
|-------------|-------------|
| `DATABRICKS_HOST` | Databricks workspace URL |
| `DATABRICKS_TOKEN` | Personal access token |
| `STAGING_AZURE_SP_APPLICATION_ID` | Staging service principal |
| `STAGING_AZURE_SP_CLIENT_SECRET` | Staging SP secret |
| `STAGING_AZURE_SP_TENANT_ID` | Azure tenant ID |
| `PROD_AZURE_SP_APPLICATION_ID` | Production service principal |
| `PROD_AZURE_SP_CLIENT_SECRET` | Production SP secret |
| `PROD_AZURE_SP_TENANT_ID` | Azure tenant ID |

### 3. Local Environment (.env)

```bash
DATABRICKS_HOST=https://your-workspace.azuredatabricks.net
DATABRICKS_TOKEN=your-token
DATABRICKS_CLUSTER_ID=your-cluster-id
```

---

## 🚀 Quick Start Guide

### Step 1: Clone the Repository

```bash
git clone https://github.com/sridharankaliyamoorthy/RAG-using-Azure-Databricks-CI-CD-Project017.git
cd RAG-using-Azure-Databricks-CI-CD-Project017
```

### Step 2: Set Up Environment

```bash
# Copy example env file
cp .env.example .env

# Edit with your values
nano .env
```

### Step 3: Configure Databricks CLI

```bash
# Install Databricks CLI
brew install databricks/tap/databricks

# Configure
databricks configure --token
```

### Step 4: Upload Notebooks

```bash
databricks workspace mkdirs /Workspace/Shared/RAG_Chatbot
databricks workspace import /Workspace/Shared/RAG_Chatbot/01_PDF_Advance_Data_Preprocessing --file notebooks/01_PDF_Advance_Data_Preprocessing.py --language PYTHON
# Repeat for all notebooks...
```

### Step 5: Run the Pipeline

1. Open Databricks workspace
2. Navigate to `Shared/RAG_Chatbot`
3. Run notebooks in order: 01 → 02 → 03 → 04

---

## 📊 Pipeline Stages

### Notebook 01: PDF Data Preprocessing

- Loads PDF documents (or sample data for demo)
- Splits text into chunks
- Generates embeddings
- Saves to Delta Table: `rusefx.rag_schema.pdf_chunks`

### Notebook 02: Advanced Chatbot Chain

- Loads embeddings from Delta Table
- Creates vector store for retrieval
- Builds RAG chatbot with LangChain
- Registers model with MLflow
- Saves model info to Delta Table

### Notebook 03: Offline Evaluation

- Loads trained model
- Runs evaluation metrics (word overlap, key phrase coverage)
- Logs metrics to MLflow
- Records pass/fail status

### Notebook 04: Deploy Model As Endpoint

- Verifies evaluation passed
- Registers model in Unity Catalog
- Creates/updates Model Serving endpoint
- Saves deployment info

---

## 🔄 CI/CD Workflow

### Triggers

| Branch | Action | Workflow |
|--------|--------|----------|
| `main` | Push | Deploy to Staging |
| `release` | Push | Deploy to Production |
| Any | Pull Request | Run CI Validation |

### Flow

```
Developer Push → GitHub Actions → Validate → Deploy to Staging → (Manual) → Deploy to Production
```

---

## 📦 Dependencies

### Python Packages (Databricks)

```
numpy<2
langchain==0.1.20
langchain-community==0.0.38
openai==1.30.0
chromadb==0.4.24
mlflow
tiktoken
pypdf
```

### CLI Tools

- Azure CLI (`az`)
- Databricks CLI (`databricks`)
- Git

---

## 🔐 Security Best Practices

1. **Never commit secrets** - Use `.env` files and `.gitignore`
2. **Use Azure Key Vault** - For production secrets
3. **Use Databricks Secret Scopes** - For notebook secrets
4. **Rotate tokens regularly** - PATs should expire
5. **Use Service Principals** - For CI/CD authentication

---

## 🧪 Testing

### Run Local Tests

```bash
# Install test dependencies
pip install pytest

# Run tests
pytest tests/
```

### Run in CI

Tests run automatically on Pull Requests via `LLMOps-run-tests.yml`.

---

## 📈 Monitoring

### MLflow Experiment

- Location: `/Users/{username}/RAG_Chatbot_Experiment`
- Tracks: Parameters, metrics, model artifacts

### Delta Tables

- `rusefx.rag_schema.evaluation_results` - Historical evaluations
- `rusefx.rag_schema.deployment_info` - Deployment history

---

## 🔮 Future Enhancements

- [ ] Add real PDF document processing
- [ ] Integrate OpenAI GPT-4 for responses
- [ ] Add Streamlit/Gradio UI
- [ ] Implement A/B testing for models
- [ ] Add cost monitoring with Azure Cost Management
- [ ] Add monitoring with Azure Application Insights

---

## 📞 Support

For issues or questions, open a GitHub issue or refer to:

- [Databricks Documentation](https://docs.databricks.com/)
- [LangChain Documentation](https://python.langchain.com/)
- [MLflow Documentation](https://mlflow.org/docs/latest/)

---

## 📄 License

This project is licensed under the MIT License.

---

**Created:** 2025-12-09  
**Version:** 1.0.0  
**Author:** sridharankaliyamoorthy
