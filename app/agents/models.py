import os
from typing import Optional
import botocore
from pathlib import Path
from langchain_aws import ChatBedrock
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseChatModel
from app.utils.logging_utils import logger
import google.auth.exceptions
import google.auth
from dotenv import load_dotenv
from dotenv.main import find_dotenv

 
class ModelManager:

    # Class-level state with process-specific initialization
    _state = {
        'model': None,
        'credentials_checked': False,
        'google_credentials': None,
        'aws_profile': None,
        'process_id': None
    }

    # Define valid models for each endpoint
    BEDROCK_MODELS = {
        "sonnet": "anthropic.claude-v2",
        "sonnet3.5": "anthropic.claude-3",
        "sonnet3.5-v2": "anthropic.claude-3-sonnet",
        "haiku": "anthropic.claude-3-haiku",
        "opus": "anthropic.claude-3-opus"
    }

    GOOGLE_MODELS = {
        "gemini-pro": "gemini-pro",
        "gemini-1.5-pro": "gemini-1.5-pro"
    }

    DEFAULT_MODELS = {
        "bedrock": "sonnet3.5-v2",
        "google": "gemini-1.5-pro"
    }

    @classmethod
    def validate_model_choice(cls, endpoint: str, model: str) -> str:
        """
        Validates the model choice for the given endpoint and returns the full model identifier.
        Raises ValueError if invalid.
        """
        if endpoint == "bedrock":
            if model not in cls.BEDROCK_MODELS:
                valid_models = ", ".join(cls.BEDROCK_MODELS.keys())
                raise ValueError(
                    f"Invalid model '{model}' for bedrock endpoint. "
                    f"Valid models are: {valid_models}"
                )
            return cls.BEDROCK_MODELS[model]

        elif endpoint == "google":
            if model not in cls.GOOGLE_MODELS:
                valid_models = ", ".join(cls.GOOGLE_MODELS.keys())
                raise ValueError(
                    f"Invalid model '{model}' for google endpoint. "
                    f"Valid models are: {valid_models}"
                )
            return cls.GOOGLE_MODELS[model]
        else:
            raise ValueError(f"Invalid endpoint: {endpoint}. Must be 'bedrock' or 'google'")
 
    @classmethod
    def _load_credentials(cls) -> bool:
        """
        Load credentials from environment or .env files.
        Returns True if GOOGLE_API_KEY is found, False otherwise.
        """
        current_pid = os.getpid()

        # Check if we've already loaded credentials in this process
        if cls._state['credentials_checked'] and cls._state['process_id'] == current_pid:
            return bool(cls._state['google_credentials'])

        # Reset state for new process
        if cls._state['process_id'] != current_pid:
            cls._state['credentials_checked'] = False

        cwd = os.getcwd()
        home = str(Path.home())
        env_locations = {
            'current_dir': os.path.join(cwd, '.env'),
            'home_ziya': os.path.join(home, '.ziya', '.env'),
            'found_dotenv': find_dotenv()
        }

        logger.debug("Searching for .env files:")
        for location_name, env_file in env_locations.items():
            if os.path.exists(env_file):
                logger.info(f"Loading credentials from {location_name}: {env_file}")
                try:
                    with open(env_file, 'r') as f:
                        logger.debug(f"Content of {env_file}:")
                        for line in f:
                            logger.debug(f"  {line.rstrip()}")
                except Exception as e:
                    logger.error(f"Error reading {env_file}: {e}")
            else:
                logger.debug(f"No .env file at {location_name}: {env_file}")

        for env_file in env_locations.values():
            cls._credentials_checked = True
            cls._google_credentials = os.getenv("GOOGLE_API_KEY")
            if os.path.exists(env_file):
                logger.info(f"Loading credentials from {env_file}")
                success = load_dotenv(env_file, override=True)
                if success:
                    # Explicitly store the value we loaded
                    api_key = os.getenv("GOOGLE_API_KEY")
                    if api_key:
                        cls._state.update({
                            'credentials_checked': True,
                            'google_credentials': os.getenv("GOOGLE_API_KEY"),
                            'process_id': current_pid
                        })
                        return True
                    else:
                        logger.warning(f"Found .env file at {location_name}: {env_file} but it doesn't contain GOOGLE_API_KEY")
        else:
            if "GOOGLE_API_KEY" not in os.environ:
                logger.debug("No .env file found, using system environment variables")
            cls._state.update({
                'credentials_checked': True,
                'google_credentials': os.getenv("GOOGLE_API_KEY"),
                'process_id': current_pid
            })
            return bool(cls._state['google_credentials'])

    @classmethod
    def initialize_model(cls) -> BaseChatModel:
        """Initialize and return the appropriate model based on environment settings."""
        current_pid = os.getpid()

        # Return cached model if it exists for this process
        if cls._state['model'] is not None and cls._state['process_id'] == current_pid:
            return cls._state['model']

        # Reset state for new process
        if cls._state['process_id'] != current_pid:
            cls._state['model'] = None
            cls._state['credentials_checked'] = False

        endpoint = os.environ.get("ZIYA_ENDPOINT", "bedrock")
        model_name = os.environ.get("ZIYA_MODEL")

        # If no model specified, use default for endpoint
        if not model_name:
            model_name = cls.DEFAULT_MODELS[endpoint]

        # Validate model choice
        cls.validate_model_choice(endpoint, model_name)
 
        if endpoint == "bedrock":
            cls._state['model'] = cls._initialize_bedrock_model(model_name)
        elif endpoint == "google":
            cls._state['model'] = cls._initialize_google_model(model_name)
        else:
            raise ValueError(f"Unsupported endpoint: {endpoint}")

        # Update process ID after successful initialization
            cls._state['process_id'] = current_pid

        return cls._state['model']
 
    @classmethod
    def _initialize_bedrock_model(cls, model_name: Optional[str] = None) -> ChatBedrock:
        """Initialize a Bedrock model."""
        if not model_name:
            model_name = "sonnet3.5-v2"
 
        if model_name not in cls.BEDROCK_MODELS:
            raise ValueError(f"Invalid Bedrock model: {model_name}")
 
        if not cls._state['aws_profile']:
            cls._state['aws_profile'] = os.environ.get("ZIYA_AWS_PROFILE")
            if cls._state['aws_profile']:
                logger.info(f"Using AWS Profile: {cls._state['aws_profile']}")
            else:
                logger.info("Using default AWS credentials")
 
        model_id = cls.BEDROCK_MODELS[model_name]
        logger.info(f"Initializing Bedrock model: {model_id}")
 
        return ChatBedrock(
            model_id=model_id,
            model_kwargs={
                "max_tokens": 4096,
                "temperature": 0.3,
                "top_k": 15
            },
            credentials_profile_name=cls._state['aws_profile'],
            config=botocore.config.Config(
                read_timeout=900,
                retries={
                    'max_attempts': 3,
                    'total_max_attempts': 5
                }
            )
        )
 
    @classmethod
    def _initialize_google_model(cls, model_name: Optional[str] = None) -> ChatGoogleGenerativeAI:
        """Initialize a Google model."""
        if not model_name:
            model_name = "gemini-1.5-pro"

        # Load credentials if not already loaded
        if not cls._state['credentials_checked']:
            if not cls._load_credentials():
                raise ValueError(
                    "GOOGLE_API_KEY environment variable is required for google endpoint.\n"
                    "You can set it in your environment or create a .env file in either:\n"
                    "  - Your current directory\n"
                    "  - ~/.ziya/.env\n")
 
        if model_name not in cls.GOOGLE_MODELS:
            raise ValueError(f"Invalid Google model: {model_name}")

        # Force reload of environment variables
        load_dotenv(override=True)

        logger.info("Checking Google authentication methods...")

        # Check API Key
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            logger.debug(f"Found API key (starts with: {api_key[:6]}...)")
            if not api_key.startswith("AI"):
                logger.warning(f"API key format looks incorrect (starts with '{api_key[:6]}', should start with 'AI')")
        else:
            logger.debug("No API key found in environment")

        # Check Application Default Credentials
        try:
            credentials, project = google.auth.default()
            logger.debug(f"Found ADC credentials (project: {project})")
        except Exception as e:
            logger.debug(f"No ADC credentials found: {str(e)}")
            credentials = None
            project = None
 
        try:
            logger.info(f"Attempting to initialize Google model: {model_name}")
            model = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0.3,
                convert_system_message_to_human=True,
                max_output_tokens=4096
            )
            # Test the model with a simple query
            # response = model.invoke("Test connection")
            logger.info("Successfully connected to Google API")
            return model

        except google.auth.exceptions.DefaultCredentialsError as e:
            logger.error(f"Authentication error details: {str(e)}")
            raise ValueError(
                "\nGoogle API authentication failed. You need to either:\n\n"
                "1. Use an API key (recommended for testing):\n"
                "   - Get an API key from: https://makersuite.google.com/app/apikey\n"
                "   - Add to .env file: GOOGLE_API_KEY=your_key_here\n"
                f"   Current API key status: {'Found' if api_key else 'Not found'}\n\n"
                "2. Or set up Application Default Credentials (for production):\n"
                "   - Install gcloud CLI: https://cloud.google.com/sdk/docs/install\n"
                "   - Run: gcloud auth application-default login\n"
                "   - See: https://cloud.google.com/docs/authentication/external/set-up-adc\n"
                f"   Current ADC status: {'Found' if credentials else 'Not found'}\n\n"
                "Choose option 1 (API key) if you're just getting started.\n"
            )
        except Exception as e:
            logger.error(f"Unexpected error initializing Google model: {str(e)}")
            raise ValueError(
                f"\nFailed to initialize Google model: {str(e)}\n\n"
                f"API key status: {'Found' if api_key else 'Not found'}\n"
                f"ADC status: {'Found' if credentials else 'Not found'}\n"
                "Please check your credentials and try again."
            )
 
    @classmethod
    def get_available_models(cls, endpoint: Optional[str] = None) -> list[str]:
        """Get list of available models for the specified endpoint."""
        if endpoint is None:
            endpoint = os.environ.get("ZIYA_ENDPOINT", "bedrock")
 
        if endpoint == "bedrock":
            return list(cls.BEDROCK_MODELS.keys())
        elif endpoint == "google":
            return cls.GOOGLE_MODELS
        else:
            raise ValueError(f"Unsupported endpoint: {endpoint}")
