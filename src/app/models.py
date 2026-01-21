from pydantic import BaseModel


class QuestionRequest(BaseModel):
  """Request body for the `/qa` endpoint.

  The PRD specifies a single field named `question` that contains
  the user's natural language question about the vector databases paper.
  """

  question: str


class QAResponse(BaseModel):
  """Response body for the `/qa` endpoint.

  From the API consumer's perspective we expose the final verified answer,
  context snippets, and the query planning metadata (plan and sub-questions).
  Internal draft answers remain inside the agent pipeline.
  """

  answer: str
  context: str
  plan: str | None = None
  sub_questions: list[str] | None = None