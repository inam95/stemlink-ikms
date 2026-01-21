"""Agent implementations for the multi-agent RAG flow.

This module defines four LangChain agents (Planning, Retrieval, Summarization,
Verification) and thin node functions that LangGraph uses to invoke them.
"""

import json
from typing import List

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ..llm import create_chat_model
from .tools import retrieval_tool
from .prompts import PLANNING_SYSTEM_PROMPT, RETRIEVAL_SYSTEM_PROMPT, SUMMARIZATION_SYSTEM_PROMPT, VERIFICATION_SYSTEM_PROMPT
from .state import QAState

def _extract_last_ai_content(messages: List[object]) -> str:
  """Extract the content of the last AIMessage in a messages list."""
  for msg in reversed(messages):
    if isinstance(msg, AIMessage):
      return str(msg.content)
  return ""

# Define agents at module level for reuse
planning_agent = create_agent(
  model=create_chat_model(),
  tools=[],
  system_prompt=PLANNING_SYSTEM_PROMPT,
)

retrieval_agent = create_agent(
  model=create_chat_model(),
  tools=[retrieval_tool],
  system_prompt=RETRIEVAL_SYSTEM_PROMPT,
)

summarization_agent = create_agent(
  model=create_chat_model(),
  tools=[],
  system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
)

verification_agent = create_agent(
  model=create_chat_model(),
  tools=[],
  system_prompt=VERIFICATION_SYSTEM_PROMPT,
)

def planning_node(state: QAState) -> QAState:
  """Planning Agent node: analyzes question and generates search plan.

  This node:
  - Sends the user's question to the Planning Agent.
  - Agent analyzes complexity and decomposes into sub-questions.
  - Extracts structured plan (JSON) from the response.
  - Stores plan and sub_questions in state.
  """

  question = state["question"]

  result = planning_agent.invoke({"messages": [HumanMessage(content=question)]})
  messages = result.get("messages", [])

  # Extract the last AI message content
  plan_output = _extract_last_ai_content(messages)

  # Parse the JSON output
  plan = None
  sub_questions = None

  try:
    parsed = json.loads(plan_output)
    plan = parsed.get("plan", "")
    sub_questions = parsed.get("sub_questions", [])
  except json.JSONDecodeError:
    # Fallback: if JSON parsing fails, use original question as single sub-question
    plan = "Unable to parse plan. Using original question for retrieval."
    sub_questions = [question]

  return {
    "plan": plan,
    "sub_questions": sub_questions,
  }

def retrieval_node(state: QAState) -> QAState:
  """Retrieval Agent node: gathers context from vector store.

  This node:
  - Checks if sub_questions exist from planning phase.
  - If yes: executes retrieval for each sub-question and aggregates results.
  - If no: falls back to single retrieval with original question.
  - Deduplicates chunks to avoid redundant context.
  - Stores the consolidated context string in `state["context"]`.
  """

  question = state["question"]
  sub_questions = state.get("sub_questions", [])

  all_contexts = []
  seen_chunks = set()  # Track unique chunks by content hash

  # Use sub-questions if available, otherwise use original question
  queries = sub_questions if sub_questions else [question]

  for query in queries:
    result = retrieval_agent.invoke({"messages": [HumanMessage(content=query)]})
    messages = result.get("messages", [])

    # Extract context from ToolMessage
    for msg in reversed(messages):
      if isinstance(msg, ToolMessage):
        chunk_content = str(msg.content)

        # Simple deduplication by content hash
        content_hash = hash(chunk_content)
        if content_hash not in seen_chunks:
          seen_chunks.add(content_hash)
          all_contexts.append(chunk_content)
        break

  # Combine all unique contexts
  context = "\n\n---\n\n".join(all_contexts) if all_contexts else ""

  return {
    "context": context,
  }

def summarization_node(state: QAState) -> QAState:
  """Summarization Agent node: generates draft answer from context.

  This node:
  - Sends question + context to the Summarization Agent.
  - Agent responds with a draft answer grounded only in the context.
  - Stores the draft answer in `state["draft_answer"]`.
  """

  question = state["question"]
  context = state.get("context")

  user_content = f"Question: {question}\n\nContext:\n{context}"

  result = summarization_agent.invoke(
    {"messages": [HumanMessage(content=user_content)]}
  )
  messages = result.get("messages", [])
  draft_answer = _extract_last_ai_content(messages)

  return {
    "draft_answer": draft_answer,
  }

def verification_node(state: QAState) -> QAState:
  """Verification Agent node: verifies and corrects the draft answer.

  This node:
  - Sends question + context + draft_answer to the Verification Agent.
  - Agent checks for hallucinations and unsupported claims.
  - Stores the final verified answer in `state["answer"]`.

  Note: For streaming support, this node calls the LLM directly instead of
  using the agent wrapper, as agent invoke() doesn't support token streaming.
  """
  question = state["question"]
  context = state.get("context", "")
  draft_answer = state.get("draft_answer", "")

  user_content = f"""Question: {question}
    Context:
    {context}

    Draft Answer:
    {draft_answer}

    Please verify and correct the draft answer, removing any unsupported claims."""


  messages = [
    {"role": "system", "content": VERIFICATION_SYSTEM_PROMPT},
    {"role": "user", "content": user_content}
  ]

  llm = create_chat_model()
  response = llm.invoke(messages)
  answer = response.content

  return {
    "answer": answer,
  }