"""
Custom exceptions for the application.
"""
import os
import tempfile
import atexit
import shutil

# Create a unique directory in the system temp directory for this process
_PROCESS_ID = os.getpid()
_TEMP_DIR = os.path.join(tempfile.gettempdir(), f"ziya_error_flags_{_PROCESS_ID}")

# Ensure the temp directory exists
os.makedirs(_TEMP_DIR, exist_ok=True)

# Register cleanup function to remove the temp directory on exit
def _cleanup_temp_dir():
    try:
        if os.path.exists(_TEMP_DIR):
            shutil.rmtree(_TEMP_DIR)
    except:
        pass

atexit.register(_cleanup_temp_dir)

class ThrottlingException(Exception):
    """
    Exception raised when AWS Bedrock returns a throttling error.
    This is a custom exception that can be caught and handled properly.
    """
    def __init__(self, message="Too many requests to AWS Bedrock. Please wait a moment before trying again."):
        self.message = message
        super().__init__(self.message)
        
    def __str__(self):
        return self.message
        
class ExpiredTokenException(Exception):
    """
    Exception raised when AWS credentials have expired.
    """
    def __init__(self, message="AWS credentials have expired. Please refresh your credentials."):
        self.message = message
        super().__init__(self.message)
        
    def __str__(self):
        return self.message

class ValidationError(Exception):
    """Exception raised when input validation fails."""
    pass
        
class KnownCredentialException(SystemExit):
    """
    Exception raised for known credential issues that should be displayed without traceback.
    This exception is used to provide clean error messages for expected authentication failures.
    
    This inherits from SystemExit to prevent traceback from being printed when uncaught.
    """
    # Reset the class variable on module import
    _error_displayed = False
    
    def __init__(self, message):
        self.message = message
        
        # Always print the message - we need to see credential errors
        print("\n" + "=" * 80)
        print(message)
        print("=" * 80 + "\n")
        
        # Mark that we've displayed the error
        KnownCredentialException._error_displayed = True
            
        super().__init__(1)  # Exit code 1
        
    def __str__(self):
        return self.message
