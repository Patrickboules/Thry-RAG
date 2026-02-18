import os
import logging
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validateEnv():
    env_variables = [
        "DATABASE_URL",
        "GROQ_API_KEY",
        "ALLOWED_ORIGINS",
        "THRY_API_KEY",
        "HUGGINGFACE_TOKEN"
    ]
    missing = [v for v in env_variables if not os.getenv(v)]

    if missing:
        # Fixed the join logic here to show the actual variable names
        error_msg = f"❌ Deployment Failed: Missing Env Vars: {', '.join(missing)}"
        logger.error(error_msg)
        raise ImportError(error_msg)

    logger.info("✅ Environment validated.")

def get_uuid():
    """
    Generate a unique UUID
    """
    return uuid.uuid7()