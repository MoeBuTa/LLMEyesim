class LLMError(Exception):
    """Base exception for llm-related errors"""

    pass


class InvalidLLMType(LLMError):
    """Raised when an invalid llm type is specified"""

    pass


class ConfigurationError(LLMError):
    """Raised when there's an error in llm configuration"""

    pass
