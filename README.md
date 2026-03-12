# Thry-RAG - AI Investment Education Chatbot

Thry is an AI-powered chatbot that answers questions about investing using content from the **Thndr Learn** educational curriculum. It retrieves relevant information from the curriculum, re-ranks it for precision, and uses an LLM to generate clear, grounded responses — all while remembering the full conversation history.

---

## How It Works

### 1. PDF Ingestion (One-Time Setup)

Before the chatbot can answer anything, the Thndr Learn PDF is processed and stored:

- The PDF is loaded and split into overlapping chunks (500 tokens each, 100 overlap)
- Each chunk is converted into a vector embedding using a HuggingFace sentence transformer model
- The embeddings are stored in a **PostgreSQL vector database (PGVector)**

This only runs once. After that, the knowledge base is ready for retrieval.

---

### 2. Answering a Question

When a user sends a message to `/chat`, this is what happens under the hood:

```
User Message
     │
     ▼
 LangGraph Agent
     │
     ├── Decides to call the Retriever Tool
     │         │
     │         ├── Searches PGVector for top 10 similar chunks
     │         │
     │         └── Sends them to LangSearch Reranker → returns top 3
     │
     └── LLM (Qwen3-32B via Groq) generates a response
              │
              ▼
         Response returned to user
```

**Step-by-step:**

1. The user's message arrives at the FastAPI backend
2. The **LangGraph agent** receives the message along with the full conversation history (retrieved from PostgreSQL)
3. The agent calls the **retriever tool**, which searches the vector database for the 10 most semantically similar chunks to the user's query
4. Those 10 chunks are passed to a **reranker API**, which narrows them down to the 3 most relevant ones
5. The top 3 chunks are handed back to the LLM as context
6. The **LLM** generates a response grounded in the curriculum content
7. The response and updated conversation history are saved back to PostgreSQL
8. The answer is returned to the user

---

### 3. Memory & Sessions

Each user gets a unique session via a browser cookie. Each chat thread within a session has its own ID. These are combined and hashed to create a **thread ID** that the LangGraph checkpointer uses to persist and retrieve conversation history from PostgreSQL — so Thry remembers what was said earlier in the conversation.

---

### 4. Backend & Safety

The FastAPI server handles all requests with several production-grade safeguards:

- **Rate limiting** — max 5 requests/minute per user, 60/minute globally (via Upstash Redis)
- **Concurrency control** — max 8 simultaneous agent calls via asyncio Semaphore
- **Timeout handling** — requests that take longer than 50 seconds return a clean 504 error
- **Health check** endpoint at `/health` that verifies database connectivity

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Qwen3-32B via Groq API |
| Agent Framework | LangGraph + LangChain |
| Vector Database | PostgreSQL + PGVector |
| Embeddings | HuggingFace (`all-MiniLM-L6-v2`) |
| Reranker | LangSearch Reranker API |
| Conversation Memory | AsyncPostgresSaver (LangGraph) |
| API Framework | FastAPI |
| Rate Limiting | Upstash Redis |
| PDF Processing | PyPDF + RecursiveCharacterTextSplitter |

---

## Project Structure

```
├── index.py          # FastAPI app, routes, rate limiting, session management
├── AiAgent.py        # LangGraph agent graph definition and runner
├── llm.py            # LLM initialization and tool binding
├── tools.py          # Retriever + reranker tool
├── database.py       # PostgreSQL, PGVector, and connection pool setup
├── Embeddings.py     # PDF ingestion and embedding pipeline
├── domain.py         # Request validation models
└── config.py         # Environment validation and utilities
```
