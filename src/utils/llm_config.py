import os
from typing import Literal, Optional
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

LLMProvider = Literal["groq", "anthropic", "openai"]

def get_llm(
    provider: LLMProvider = "groq",
    model: Optional[str] = None,
    temperature: float = 0,
    **kwargs
) -> BaseChatModel:
    """
    Get a configured LLM instance based on the specified provider.
    
    Args:
        provider: The LLM provider to use ('groq', 'anthropic', or 'openai')
        model: The model name to use. If None, uses default for the provider
        temperature: The temperature parameter for the LLM
        **kwargs: Additional arguments to pass to the LLM constructor
        
    Returns:
        BaseChatModel: Configured LLM instance
    """
    # Set default models if not specified
    if model is None:
        model = {
            "groq": "llama-3.3-70b-versatile",
            "anthropic": "claude-3-opus-20240229",
            "openai": "gpt-4-turbo-preview"
        }.get(provider, "llama-3.3-70b-versatile")
    
    # Get API key from environment
    api_key_env = {
        "groq": "GROQ_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY"
    }.get(provider)
    
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(f"{api_key_env} environment variable not set")
    
    # Create the appropriate LLM instance
    if provider == "groq":
        return ChatGroq(
            model=model,
            temperature=temperature,
            api_key=api_key,
            **kwargs
        )
    elif provider == "anthropic":
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            api_key=api_key,
            **kwargs
        )
    elif provider == "openai":
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def get_default_llm() -> Optional[BaseChatModel]:
    """
    Get the default LLM instance. Returns None if no API key is configured.
    This is a lazy-loading alternative to avoid module-level failures.
    """
    # Try providers in order of preference
    providers = ["groq", "anthropic", "openai"]
    api_keys = {
        "groq": "GROQ_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY", 
        "openai": "OPENAI_API_KEY"
    }
    
    for provider in providers:
        if os.getenv(api_keys[provider]):
            try:
                return get_llm(provider=provider)
            except Exception:
                continue
    
    return None


# Lazy-loaded default LLM (use get_default_llm() instead of this directly)
_default_llm = None

def default_llm() -> BaseChatModel:
    """Get default LLM instance with lazy loading."""
    global _default_llm
    if _default_llm is None:
        _default_llm = get_default_llm()
        if _default_llm is None:
            raise ValueError(
                "No LLM API key configured. Please set one of: "
                "GROQ_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY"
            )
    return _default_llm