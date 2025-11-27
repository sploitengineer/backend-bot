import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from google.cloud import discoveryengine_v1 as discoveryengine
from google.api_core.client_options import ClientOptions

# --- Configuration ---
PROJECT_ID = "testcase-69"
LOCATION = "us-central1"
SEARCH_LOCATION = "global"

# --- Knowledge Vault IDs ---
DATA_STORE_MAP = {
    "DRIVING_LICENSE": "driving-store_1764007089201",
    "HOME_REGISTRATION": "housing-store_1764008078871",
    "ELECTRICITY": "electricity-store_1764008116062",
    "GENERAL": "driving-store_1764007089201",  # fallback
}

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)

app = FastAPI(title="Public Services Query Resolver")

# --- CORS (needed for Flutter web) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # for POC, wide-open; you can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Models ---
class QueryRequest(BaseModel):
    user_query: str
    language: str = "English"
    tone_mode: str = "standard"         # "standard" | "eli5" (or others later)
    enable_voice_response: bool = False # reserved for future

class Citation(BaseModel):
    document_name: str
    page_number: int
    snippet: str
    source_link: str

class AgentResponse(BaseModel):
    answer_text: str
    detected_intent: str
    citations: List[Citation] = []
    audio_base64: Optional[str] = None  # reserved for future TTS


# --- Prompts & Personas ---

# 🔧 Main fix: make the assistant answer fully, not just point to docs.
SYSTEM_PROMPT_STANDARD = """
You are "Nova", an expert assistant for Indian public services (driving licence,
home registration, electricity and related topics).

Your goals:
1. Give a clear, concise, DIRECT answer to the citizen's question.
2. Use the provided document context as your primary source of truth.
3. When the context has gaps, still give the best possible practical guidance,
   and be transparent about any missing details.

VERY IMPORTANT INSTRUCTIONS:
- DO NOT tell the user to "go read the documents" or "check the PDF".
- DO NOT answer with only references like "(see page 7)".
- DO NOT say "the context does not contain further details" without first
  summarising what IS available.
- Instead, integrate the relevant information into a helpful explanation.

STYLE:
- Use plain, friendly language.
- When describing procedures, prefer step-by-step bullet points.
- When useful, give a short example.
- You may mention sources briefly, e.g. "This is based on Clause 9 of the Motor
  Vehicles Act", but keep the main answer standalone.

OUTPUT FORMAT:
- Write only the final answer to the citizen, in natural paragraphs and lists.
- Do NOT repeat the raw snippets or the entire context.
- Do NOT output JSON.
"""

SYSTEM_PROMPT_ELI5 = """
You are "Nova", a very patient teacher explaining Indian public services to a
beginner. Imagine you are talking to a 12-year-old who is new to government
processes.

STYLE:
- Use very simple, friendly language.
- Explain concepts with small examples and analogies.
- Break procedures into short, clearly numbered steps.
- Avoid legal jargon; if you must mention a legal term, explain it.

IMPORTANT:
- Still follow the same rules about using the document context as your main
  source of truth.
- Do NOT tell the user to read the documents themselves.
- Your answer should stand alone and be easy to follow.
"""

def get_system_instruction(mode: str, language: str) -> str:
    base = SYSTEM_PROMPT_ELI5 if mode.lower() == "eli5" else SYSTEM_PROMPT_STANDARD
    return (
        f"{base}\n\n"
        f"LANGUAGE REQUIREMENT:\n"
        f"- Answer entirely in {language}. Translate any legal or technical terms "
        f"into {language} as needed.\n"
    )


# --- Core Logic ---

def determine_intent(query: str) -> str:
    """
    Very small Gemini call that just classifies into our 4 categories.
    """
    model = GenerativeModel("gemini-2.5-flash")
    prompt = f"""
    Classify this user query into one of these categories ONLY:
    - DRIVING_LICENSE
    - HOME_REGISTRATION
    - ELECTRICITY
    - GENERAL

    Return ONLY the category name, nothing else.

    Query: {query}
    """
    response = model.generate_content(prompt)
    intent = response.text.strip().upper()

    if intent not in DATA_STORE_MAP:
        return "GENERAL"
    return intent


def search_knowledge_base(query: str, category: str) -> List[Citation]:
    target_store_id = DATA_STORE_MAP.get(category, DATA_STORE_MAP["GENERAL"])
    print(f"🤖 Routing query to Knowledge Vault: {target_store_id} (Category: {category})")

    client_options = (
        ClientOptions(api_endpoint=f"{SEARCH_LOCATION}-discoveryengine.googleapis.com")
        if SEARCH_LOCATION != "global"
        else None
    )
    client = discoveryengine.SearchServiceClient(client_options=client_options)

    serving_config = client.serving_config_path(
        project=PROJECT_ID,
        location=SEARCH_LOCATION,
        data_store=target_store_id,
        serving_config="default_search",
    )

    content_search_spec = discoveryengine.SearchRequest.ContentSearchSpec(
        snippet_spec=discoveryengine.SearchRequest.ContentSearchSpec.SnippetSpec(
            return_snippet=True
        ),
        extractive_content_spec=discoveryengine.SearchRequest.ContentSearchSpec.ExtractiveContentSpec(
            max_extractive_answer_count=1,
        ),
    )

    request = discoveryengine.SearchRequest(
        serving_config=serving_config,
        query=query,
        page_size=3,
        content_search_spec=content_search_spec,
        query_expansion_spec=discoveryengine.SearchRequest.QueryExpansionSpec(
            condition=discoveryengine.SearchRequest.QueryExpansionSpec.Condition.AUTO,
        ),
    )

    try:
        response = client.search(request)
    except Exception as e:
        print(f"Error searching knowledge base: {e}")
        return []

    citations: List[Citation] = []

    for result in response.results:
        doc_data = result.document.derived_struct_data
        link = doc_data.get("link", "")
        snippet = ""
        if doc_data.get("snippets"):
            snippet = doc_data.get("snippets")[0].get("snippet", "")

        page_num = 1
        if doc_data.get("extractive_answers"):
            page_content = doc_data.get("extractive_answers")[0].get("pageNumber", "1")
            try:
                page_num = int(page_content)
            except Exception:
                pass

        citations.append(
            Citation(
                document_name=result.document.name.split("/")[-1],
                page_number=page_num,
                snippet=snippet,
                source_link=link,
            )
        )

    return citations


# --- API Endpoints ---

@app.post("/api/query", response_model=AgentResponse)
async def process_query(request: QueryRequest):
    """
    Main entry point used by your Flutter app.
    - Detects intent (which vault to use)
    - Retrieves top snippets from Discovery Engine
    - Asks Gemini to write a full, user-friendly answer
    - Returns answer + intent + citations (for your UI)
    """
    try:
        intent = determine_intent(request.user_query)
    except Exception as e:
        print(f"Error determining intent: {e}")
        intent = "GENERAL"

    relevant_docs = search_knowledge_base(request.user_query, intent)

    # Build a compact context for the model: we only need a few key snippets.
    if relevant_docs:
        context_lines = []
        for d in relevant_docs:
            # Short, structured context line
            clean_snippet = d.snippet.replace("\n", " ").strip()
            context_lines.append(
                f"[Doc: {d.document_name} | Page {d.page_number}] {clean_snippet}"
            )
        context_str = "\n".join(context_lines)
    else:
        context_str = (
            "No specific documents were retrieved. "
            "Use your general knowledge about Indian public services, but avoid making up specific law clause numbers."
        )

    # Main generative call (you can tune temperature/max_output_tokens here)
    model = GenerativeModel("gemini-2.5-pro")
    system_instruction = get_system_instruction(request.tone_mode, request.language)

    full_prompt = f"""
{system_instruction}

USER QUESTION:
{request.user_query}

CONTEXT FROM OFFICIAL DOCUMENTS (category: {intent}):
{context_str}

TASK:
- Use the context above as your primary reference.
- Give a direct, practical answer to the citizen.
- Explain the process in clear steps (1, 2, 3...) when relevant.
- Include typical timelines, key eligibility rules, and important documents
  if they can be reasonably inferred from the context.
- If something is truly not in the context and you cannot safely infer it,
  say so briefly but still provide any high-level guidance you can.

REMEMBER:
- Do NOT ask the user to read the documents themselves.
- Do NOT just list document names or page numbers.
- Your answer must be useful on its own.
"""

    try:
        response = model.generate_content(
            full_prompt,
            generation_config=GenerationConfig(
                temperature=0.3,          # more factual / less rambly
                max_output_tokens=1024,
                top_p=0.9,
                top_k=40,
            ),
        )
    except Exception as e:
        print(f"Gemini generation error: {e}")
        raise HTTPException(status_code=500, detail="LLM generation failed")

    answer_text = response.text or "Sorry, I could not generate an answer at this time."

    return AgentResponse(
        answer_text=answer_text.strip(),
        detected_intent=intent,
        citations=relevant_docs,
    )


@app.get("/api/sync-offline-knowledge")
async def get_offline_db():
    """
    Placeholder endpoint for your mobile offline FAQ sync.
    """
    return {
        "version": "1.0.5",
        "download_url": "https://storage.googleapis.com/your-bucket/offline_faqs.sqlite",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
