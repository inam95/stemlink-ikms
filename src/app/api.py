from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

from .services.indexing_service import index_pdf_file
from .models import QAResponse, QuestionRequest
from .services.qa_service import answer_question

app = FastAPI(
  title="IKMS (Information Knowledge Management System)",
  description=(
    "Demo API for asking questions about a vector database paper."
    "The `/qa` endpoint currently returns placeholder responses and "
    "will be wired to a multi-agent RAG pipeline in later user stories."
  ),
  version="0.1.0",
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(
  request: Request, exc: Exception
) -> JSONResponse:
  """Catch-all handler for unexpected errors.

  FastAPI will still handle `HTTPException` instances and validation errors
  separately; this is only for truly unexpected failures so API consumers
  get a consistent 500 response body.
  """

  if isinstance(exc, HTTPException):
    raise exc

  return JSONResponse(
    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    content={
      "error": "Internal Server Error",
      "message": str(exc),
    },
  )

@app.post("/qa", response_model=QAResponse, status_code=status.HTTP_200_OK)
async def qa_endpoint(payload: QuestionRequest) -> QAResponse:
  """Submit a question about the vector databases paper.

  US-001 requirements:
  - Accept POST requests at `/qa` with JSON body containing a `question` field
  - Validate the request format and return 400 for invalid requests
  - Return 200 with `answer`, `draft_answer`, and `context` fields
  - Delegate to the multi-agent RAG service layer for processing
  """

  question = payload.question.strip()
  if not question:
    raise HTTPException(
      status_code=status.HTTP_400_BAD_REQUEST,
      detail="`question` must be a non-empty string.",
    )

  result = answer_question(question)
  result = {
    "answer": result.get("answer", ""),
    "context": result.get("context", ""),
  }

  return QAResponse(
    answer=result.get("answer", ""),
    context=result.get("context", ""),
  )


@app.post("/index-pdf", status_code=status.HTTP_200_OK)
async def index_pdf(file: UploadFile = File(...)) -> dict:
  """Upload a PDF and index it into the vector database.

  This endpoint:
  - Accepts a PDF file upload
  - Saves it to the local `data/uploads/` directory
  - Uses PyPDFLoader to load the document into LangChain `Document` objects
  - Indexes those documents into the configured Pinecone vector store
  """

  if file.content_type not in ("application/pdf",):
    raise HTTPException(
      status_code=status.HTTP_400_BAD_REQUEST,
      detail="Only PDF files are supported.",
    )

  upload_dir = Path("data/uploads")
  upload_dir.mkdir(parents=True, exist_ok=True)

  file_path = upload_dir / file.filename
  contents = await file.read()
  file_path.write_bytes(contents)

  chunks_indexed = index_pdf_file(file_path)

  return {
    "filename": file.filename,
    "chunks_indexed": chunks_indexed,
    "message": "PDF indexed successfully.",
  }