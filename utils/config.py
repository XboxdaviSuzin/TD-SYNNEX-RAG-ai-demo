"""
Secure Configuration Management Utility

This module provides secure access to environment variables and secrets.
It supports multiple secret backends:
- Local .env files (development)
- Azure Key Vault (production)
- Databricks Secret Scopes (notebooks)

Usage:
    from utils.config import get_secret, get_config

    # Get a secret
    token = get_secret('DATABRICKS_TOKEN')
    
    # Get configuration
    config = get_config()
    host = config.databricks_host
"""

import os
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Application configuration loaded from environment."""
    databricks_host: str
    databricks_token: str
    databricks_cluster_id: str
    
    # Azure Service Principal (Production)
    prod_azure_sp_app_id: Optional[str] = None
    prod_azure_sp_client_secret: Optional[str] = None
    prod_azure_sp_tenant_id: Optional[str] = None
    
    # Azure Service Principal (Staging)
    staging_azure_sp_app_id: Optional[str] = None
    staging_azure_sp_client_secret: Optional[str] = None
    staging_azure_sp_tenant_id: Optional[str] = None
    
    # Email
    email_username: Optional[str] = None
    email_password: Optional[str] = None
    
    # GitHub
    workflow_token: Optional[str] = None


def load_dotenv_file(dotenv_path: str = ".env") -> None:
    """
    Load environment variables from .env file.
    Falls back gracefully if file doesn't exist.
    """
    try:
        from dotenv import load_dotenv
        
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path)
            logger.info(f"Loaded environment from {dotenv_path}")
        else:
            logger.warning(f".env file not found at {dotenv_path}")
    except ImportError:
        logger.warning("python-dotenv not installed. Install with: pip install python-dotenv")


def get_secret(key: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """
    Get a secret from environment variables.
    
    Args:
        key: The environment variable name
        default: Default value if not found
        required: If True, raises error when not found
    
    Returns:
        The secret value or default
    
    Raises:
        ValueError: If required=True and secret not found
    """
    # First, try to load from .env if not already in environment
    if key not in os.environ:
        load_dotenv_file()
    
    value = os.getenv(key, default)
    
    if required and not value:
        raise ValueError(f"Required secret '{key}' not found in environment")
    
    if value and key.lower().endswith(('token', 'secret', 'password', 'key')):
        logger.debug(f"Loaded secret: {key} = ***REDACTED***")
    else:
        logger.debug(f"Loaded config: {key} = {value}")
    
    return value


def get_secret_from_keyvault(
    vault_url: str,
    secret_name: str
) -> Optional[str]:
    """
    Get a secret from Azure Key Vault.
    
    Args:
        vault_url: The Key Vault URL (e.g., https://my-vault.vault.azure.net)
        secret_name: Name of the secret in Key Vault
    
    Returns:
        The secret value or None
    """
    try:
        from azure.identity import DefaultAzureCredential
        from azure.keyvault.secrets import SecretClient
        
        credential = DefaultAzureCredential()
        client = SecretClient(vault_url=vault_url, credential=credential)
        
        secret = client.get_secret(secret_name)
        logger.info(f"Retrieved secret '{secret_name}' from Key Vault")
        return secret.value
        
    except ImportError:
        logger.error("Azure SDK not installed. Install with: pip install azure-identity azure-keyvault-secrets")
        return None
    except Exception as e:
        logger.error(f"Failed to retrieve secret from Key Vault: {e}")
        return None


def get_secret_from_databricks(scope: str, key: str) -> Optional[str]:
    """
    Get a secret from Databricks Secret Scope.
    Only works when running in a Databricks notebook/job.
    
    Args:
        scope: The secret scope name
        key: The secret key name
    
    Returns:
        The secret value or None
    """
    try:
        # dbutils is only available in Databricks runtime
        from pyspark.dbutils import DBUtils
        from pyspark.sql import SparkSession
        
        spark = SparkSession.builder.getOrCreate()
        dbutils = DBUtils(spark)
        
        value = dbutils.secrets.get(scope=scope, key=key)
        logger.info(f"Retrieved secret '{key}' from Databricks scope '{scope}'")
        return value
        
    except ImportError:
        logger.debug("Not running in Databricks environment")
        return None
    except Exception as e:
        logger.error(f"Failed to retrieve secret from Databricks: {e}")
        return None


@lru_cache(maxsize=1)
def get_config() -> Config:
    """
    Load and return application configuration.
    Uses caching to avoid reloading on every call.
    
    Returns:
        Config object with all settings
    """
    # Ensure .env is loaded
    load_dotenv_file()
    
    return Config(
        databricks_host=get_secret('DATABRICKS_HOST', required=True),
        databricks_token=get_secret('DATABRICKS_TOKEN', required=True),
        databricks_cluster_id=get_secret('DATABRICKS_CLUSTER_ID', ''),
        
        prod_azure_sp_app_id=get_secret('PROD_AZURE_SP_APPLICATION_ID'),
        prod_azure_sp_client_secret=get_secret('PROD_AZURE_SP_CLIENT_SECRET'),
        prod_azure_sp_tenant_id=get_secret('PROD_AZURE_SP_TENANT_ID'),
        
        staging_azure_sp_app_id=get_secret('STAGING_AZURE_SP_APPLICATION_ID'),
        staging_azure_sp_client_secret=get_secret('STAGING_AZURE_SP_CLIENT_SECRET'),
        staging_azure_sp_tenant_id=get_secret('STAGING_AZURE_SP_TENANT_ID'),
        
        email_username=get_secret('EMAIL_USERNAME'),
        email_password=get_secret('EMAIL_PASSWORD'),
        
        workflow_token=get_secret('WORKFLOW_TOKEN'),
    )


def validate_config() -> Dict[str, bool]:
    """
    Validate that all required configuration is present.
    
    Returns:
        Dict with validation results for each config item
    """
    load_dotenv_file()
    
    required_keys = [
        'DATABRICKS_HOST',
        'DATABRICKS_TOKEN',
    ]
    
    optional_keys = [
        'DATABRICKS_CLUSTER_ID',
        'PROD_AZURE_SP_APPLICATION_ID',
        'PROD_AZURE_SP_CLIENT_SECRET',
        'PROD_AZURE_SP_TENANT_ID',
        'STAGING_AZURE_SP_APPLICATION_ID',
        'STAGING_AZURE_SP_CLIENT_SECRET',
        'STAGING_AZURE_SP_TENANT_ID',
        'EMAIL_USERNAME',
        'EMAIL_PASSWORD',
        'WORKFLOW_TOKEN',
    ]
    
    results = {}
    
    print("\n🔐 Configuration Validation")
    print("=" * 50)
    
    print("\n📋 Required Configuration:")
    for key in required_keys:
        value = os.getenv(key)
        is_set = bool(value)
        results[key] = is_set
        status = "✅" if is_set else "❌"
        print(f"  {status} {key}")
    
    print("\n📋 Optional Configuration:")
    for key in optional_keys:
        value = os.getenv(key)
        is_set = bool(value)
        results[key] = is_set
        status = "✅" if is_set else "⬜"
        print(f"  {status} {key}")
    
    print("=" * 50)
    
    required_ok = all(results[k] for k in required_keys)
    if required_ok:
        print("✅ All required configuration is present!")
    else:
        print("❌ Missing required configuration. Check your .env file.")
    
    return results


# For testing
if __name__ == "__main__":
    validate_config()
