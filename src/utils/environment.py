import os

from dotenv import load_dotenv


REQUIRED_ENVIRONMENT_VARIABLES = ["WANDB_API_KEY", "HF_TOKEN"]


def validate_environment_variables() -> None:
    """
    Validate that all required environment variables for Agent API operation are set.
    This function performs a fail-fast check at startup, ensuring that the application
    does not start in a misconfigured state if any essential variables are missing
    or empty.
    """
    for var in REQUIRED_ENVIRONMENT_VARIABLES:
        value = os.getenv(var)
        if not value:
            raise EnvironmentError(f"Critical environment variable {var} is missing!")
        if len(value.strip()) == 0:
            raise EnvironmentError(f"Critical environment variable {var} is empty!")


def init_environment() -> None:
    """Load environment variables from .env file and validate them."""
    load_dotenv()
    validate_environment_variables()
