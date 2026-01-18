"""Factory functions for creating LangChain v1 LLM instances."""

from functools import lru_cache
from langchain_openai import ChatOpenAI

from ..config import get_settings


@lru_cache(maxsize=4)
def create_chat_model(temperature: float = 0.0, streaming: bool = True) -> ChatOpenAI:
  """Create a LangChain v1 ChatOpenAI instance.

  Args:
  temperature: Model temperature (default: 0.0 for deterministic outputs).
  streaming: Enable streaming mode for token-by-token output (default: True).

  Returns:
    Configured ChatOpenAI instance.
  """
  settings = get_settings()
  return ChatOpenAI(
    model=settings.openai_model_name,
    api_key=settings.openai_api_key,
    temperature=temperature,
    streaming=streaming,
  )