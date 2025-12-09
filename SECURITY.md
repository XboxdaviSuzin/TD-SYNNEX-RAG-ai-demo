# 🔐 Security Best Practices Guide

This document outlines security best practices for managing secrets and sensitive data in this project.

---

## 1. Environment Variables (.env)

### Setup
Store all secrets in a local `.env` file that is **never committed to Git**.

```bash
# Create your .env file from the template
cp .env.example .env

# Set proper permissions (read/write only for owner)
chmod 600 .env
```

### Python Usage
```python
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access secrets
databricks_token = os.getenv('DATABRICKS_TOKEN')
databricks_host = os.getenv('DATABRICKS_HOST')

# With default values
cluster_id = os.getenv('DATABRICKS_CLUSTER_ID', 'default-cluster')
```

### Install python-dotenv
```bash
pip install python-dotenv
```

---

## 2. .gitignore Protection

The `.gitignore` file in this project prevents sensitive files from being committed:

```gitignore
# Already configured in this project
.env
.env.local
.env.*.local
*.env
.databrickscfg
secrets/
credentials/
*.pem
*.key
*.secret
*.tfvars
```

### Verify Protection
```bash
# Check if .env is being ignored
git status --ignored | grep .env
```

---

## 3. Azure Key Vault (Production)

For production deployments, use Azure Key Vault to store secrets.

### Setup Azure Key Vault
```bash
# Create a Key Vault
az keyvault create \
  --name "your-keyvault-name" \
  --resource-group "your-resource-group" \
  --location "westeurope"

# Store a secret
az keyvault secret set \
  --vault-name "your-keyvault-name" \
  --name "DATABRICKS-TOKEN" \
  --value "your-token-value"

# Retrieve a secret
az keyvault secret show \
  --vault-name "your-keyvault-name" \
  --name "DATABRICKS-TOKEN" \
  --query "value" -o tsv
```

### Python Integration
```python
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

# Initialize client
credential = DefaultAzureCredential()
vault_url = "https://your-keyvault-name.vault.azure.net"
client = SecretClient(vault_url=vault_url, credential=credential)

# Retrieve secrets
databricks_token = client.get_secret("DATABRICKS-TOKEN").value
```

### Required Package
```bash
pip install azure-identity azure-keyvault-secrets
```

---

## 4. Databricks Secrets (In Notebooks)

Use Databricks Secret Scopes for secrets within Databricks notebooks.

### Create Secret Scope (CLI)
```bash
# Create a scope backed by Azure Key Vault
databricks secrets create-scope \
  --scope "my-scope" \
  --scope-backend-type AZURE_KEYVAULT \
  --resource-id "/subscriptions/.../resourceGroups/.../providers/Microsoft.KeyVault/vaults/your-keyvault" \
  --dns-name "https://your-keyvault.vault.azure.net/"
```

### Use in Notebooks
```python
# Access secrets in Databricks notebooks
api_key = dbutils.secrets.get(scope="my-scope", key="API-KEY")
```

---

## 5. GitHub Secrets (CI/CD)

For GitHub Actions workflows, use GitHub Secrets.

### Add Secrets to GitHub
1. Go to your repository → **Settings** → **Secrets and variables** → **Actions**
2. Click **"New repository secret"**
3. Add each secret:
   - `DATABRICKS_TOKEN`
   - `PROD_AZURE_SP_APPLICATION_ID`
   - `PROD_AZURE_SP_CLIENT_SECRET`
   - `PROD_AZURE_SP_TENANT_ID`
   - etc.

### Use in Workflows
```yaml
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Configure Databricks CLI
        env:
          DATABRICKS_TOKEN: ${{ secrets.DATABRICKS_TOKEN }}
          DATABRICKS_HOST: ${{ secrets.DATABRICKS_HOST }}
        run: |
          databricks configure --token <<< "$DATABRICKS_TOKEN"
```

---

## 6. No Front-End Exposure

### ❌ NEVER do this:
```javascript
// WRONG - Exposes API key in browser
const apiKey = "sk-abc123...";
fetch('https://api.openai.com/v1/chat', {
  headers: { 'Authorization': `Bearer ${apiKey}` }
});
```

### ✅ Use a Backend Proxy:
```python
# Backend API (Flask example)
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

@app.route('/api/chat', methods=['POST'])
def chat():
    # API key is only on the server
    api_key = os.getenv('OPENAI_API_KEY')
    
    # Make request to OpenAI from backend
    response = requests.post(
        'https://api.openai.com/v1/chat/completions',
        headers={'Authorization': f'Bearer {api_key}'},
        json=request.json
    )
    return jsonify(response.json())
```

---

## 7. Security Checklist

| Item | Status | Action |
|------|--------|--------|
| ✅ `.gitignore` configured | Done | Protects sensitive files |
| ✅ `.env` file created | Done | Local secrets storage |
| ✅ File permissions set | Done | `chmod 600 .env` |
| ⬜ GitHub Secrets | Pending | Add to repository settings |
| ⬜ Azure Key Vault | Pending | Set up for production |
| ⬜ Databricks Secret Scope | Pending | Link to Key Vault |
| ⬜ Regenerate exposed token | **URGENT** | Revoke old token |

---

## 8. Emergency: If Secrets Are Exposed

If you accidentally expose a secret:

1. **Immediately revoke/rotate** the exposed credential
2. **Check git history** for any committed secrets:
   ```bash
   git log --all --full-history -- "*.env"
   ```
3. **Remove from history** if committed:
   ```bash
   git filter-branch --force --index-filter \
     'git rm --cached --ignore-unmatch .env' \
     --prune-empty --tag-name-filter cat -- --all
   ```
4. **Force push** (coordinate with team):
   ```bash
   git push origin --force --all
   ```
5. **Notify team members** to re-clone the repository

---

## 9. Quick Reference

| Environment | Secret Storage Method |
|-------------|----------------------|
| Local Development | `.env` file |
| CI/CD (GitHub Actions) | GitHub Secrets |
| Production (Azure) | Azure Key Vault |
| Databricks Notebooks | Databricks Secret Scopes |

---

**Remember:** Treat secrets like passwords. Never share them, rotate them regularly, and use the principle of least privilege.
