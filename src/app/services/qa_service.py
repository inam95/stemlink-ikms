"""Service layer for handling QA requests.

This module provides a simple interface for the FastAPI layer to interact
with the multi-agent RAG pipeline without depending directly on LangGraph
or agent implementation details.
"""

from typing import AsyncGenerator, Dict, Any

from ..core.agents import run_qa_flow, stream_qa_flow

def answer_question(question: str) -> Dict[str, Any]:
  """Run the multi-agent QA flow for a given question.

  Args:
    question: User's natural language question about the vector databases paper.

  Returns:
    Dictionary containing at least `answer` and `context` keys.
  """
  return run_qa_flow(question)

async def stream_answer(question: str) -> AsyncGenerator[str, None]:
  """Stream the multi-agent QA flow for a given question, yielding tokens.

  Args:
    question: User's natural language question about the vector databases paper.

  Yields:
    String tokens from the verification agent's output.
  """
  async for token in stream_qa_flow(question):
    yield token
