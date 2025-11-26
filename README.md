## Procurement Assistant

This repository contains an AI-powered procurement assistant that converts natural-language questions into MongoDB queries over the California Department of General Services purchase order dataset. The system uses a LangGraph workflow to orchestrate question classification, query generation, and response formatting.

### Quickstart


#### Docker setup

1. **Set environment variables**
   Create a `.env` file in the root directory and fill in the following values:
   - `PRIMARY_LLM_API_KEY` → your LLM API key (defaults to Grok/xAI).
   - Also fill the Langsmith API Key if you want to use langsmith for observability and debugging. (note: you will need to have an account )

2. **Start all services with Docker Compose**
   ```bash
   docker compose up --build
   ```
   This will start MongoDB, the FastAPI backend, and Streamlit UI. it might take a while the first time it is setup, especially during requirements.txt setup. 


### Data Setup

   1. **Extract the data archive**
      Download the data archive and extract its contents into the `data/` directory. The archive should contain:
      - `PURCHASE ORDER DATA EXTRACT 2012-2015_0.csv` (procurement dataset)
      - Reference documents (.docx and .pdf files) for RAG functionality

   2. **Verify data placement**
      Ensure your data files are placed in the `data/` directory:
      - `data/PURCHASE ORDER DATA EXTRACT 2012-2015_0.csv` (procurement dataset)
      - Reference document files (.docx and .pdf) which will be used for RAG when users ask questions related to that content.

3. **Load the dataset**
   ```bash
   docker compose exec app sh -c "cd /app && PYTHONPATH=/app python -m src.shared.data_loader --csv 'data/PURCHASE ORDER DATA EXTRACT 2012-2015_0.csv'"
   ```
   **Note**: The `PYTHONPATH=/app` is required for Python to find the src modules within the container.

4. **Build the reference vector store**
   ```bash
   docker compose exec app python -m data_scripts.build_reference_store --docs-dir data/
   ```
   This indexes any PDF/DOCX guidance stored under `data/` (e.g., the acquisition method memo or data dictionary) into Chroma so the agent can cite them later.

5. **Access the application**
   - **Streamlit UI**: http://localhost:8501


### Architecture

The application follows a modular, layered architecture with clear separation of concerns:

#### Core Components
- **Agent System** (`src/agent/`): LangGraph workflow orchestration that converts natural language to MongoDB queries with conversation history support
- **Services** (`src/services/`): LLM integration, question classification, query generation, and response formatting
- **Data Layer** (`src/data/`): MongoDB connection and Chroma vector storage for reference documents
- **Configuration** (`src/config/`): Settings management, constants, and LLM configurations
- **Interfaces** (`src/interfaces/`): FastAPI backend and Streamlit web interface
- **Shared Utilities** (`src/shared/`): Data loading, telemetry, and observability

#### File Structure

```
src/
├── __init__.py              # Main package interface exposing chat()
├── agent/                   # Core agent functionality
│   ├── __init__.py
│   ├── agent.py            # Main chat interface with conversation history
│   ├── types.py            # Type definitions (AgentState, etc.)
│   └── workflow.py         # LangGraph workflow orchestration
├── services/                # Backend services
│   ├── __init__.py
│   ├── llm_service/         # LLM and AI functionality
│   │   ├── __init__.py
│   │   ├── classification.py # Question categorization (routing logic)
│   │   └── llm.py           # LLM initialization & utilities
│   └── query_service/       # Query generation & processing
│       ├── __init__.py
│       ├── query_generation.py # Query creation, validation & retry logic
│       └── response_formatting.py # Natural language response formatting
├── data/                    # Data layer operations
│   ├── __init__.py
│   ├── mongodb.py           # MongoDB connection & query execution
│   └── vector_store.py      # Chroma vector storage for reference documents
├── config/                  # Configuration & constants
│   ├── __init__.py
│   ├── config.py           # Settings management (env vars, etc.)
│   └── constants.py        # Prompts, enums, schema definitions
├── interfaces/              # User interfaces
│   ├── __init__.py
│   ├── api_server.py      # FastAPI backend server
│   └── web_ui.py           # Streamlit web interface
└── shared/                  # Shared utilities
    ├── __init__.py
    ├── data_loader.py      # CSV → MongoDB ingestion
    └── telemetry.py        # Logging & tracing (LangSmith integration)
```

#### Key Design Patterns

- **Modular Architecture**: Each folder contains focused, single-responsibility modules
- **Error Handling**: Comprehensive validation and retry mechanisms for query generation
- **Observability**: Built-in telemetry with LangSmith tracing for debugging
- **Testability**: Modular design enables isolated unit testing

#### Data Flow

1. **User Input** → `src/interfaces/api_server.py` (FastAPI endpoint) or `src/interfaces/web_ui.py` (Streamlit UI)
2. **Agent Orchestration** → `src/agent/agent.py` (manages conversation history and calls LangGraph workflow)
3. **Question Classification** → `src/services/llm_service/classification.py` (routes to query generation, reference lookup, or out-of-scope)
4. **Query Generation** → `src/services/query_service/query_generation.py` (LLM creates MongoDB aggregation pipeline)
5. **Database Execution** → `src/data/mongodb.py` (executes aggregation pipeline against MongoDB)
6. **Response Formatting** → `src/services/query_service/response_formatting.py` (converts results to natural language)
7. **Final Response** → User via API/UI with updated conversation history

### Docker Services

- **mongodb**: MongoDB 7.0 database with authentication
- **app**: FastAPI backend service with health checks
- **streamlit**: Streamlit UI service

### Docker Commands

```bash
# Start all services
docker compose up

# Start in background
docker compose up -d

# View logs
docker compose logs -f

# Stop services
docker compose down

# Remove volumes (will delete data)
docker compose down -v

# Load data into running container
docker compose exec app sh -c "cd /app && PYTHONPATH=/app python -m src.shared.data_loader --csv 'data/PURCHASE ORDER DATA EXTRACT 2012-2015_0.csv'"
```

### Data Documentation & Validation

- `docs/data_schema.md` — canonical schema reference covering every column, data type, and known data-quality caveats.
- `python data_scripts/validate_mongodb_schema.py --expected-count 919734 --sample-size 500` — quick health check to confirm MongoDB ingestion completeness and field coverage.

### Reference Document Retrieval

- `data_scripts/build_reference_store.py` walks all PDF/DOCX files inside `REFERENCE_DOCS_DIR` (defaults to `data/`) and chunks them into a persistent Chroma DB located at `VECTOR_STORE_DIR`.
- Runtime questions classified as `database_info` or `acquisition_methods` are answered directly from the retrieved passages instead of running MongoDB queries.
- Customize ingestion via environment variables or pass `--docs-dir` / `--persist-dir` flags when running the script (e.g., `python -m data_scripts.build_reference_store --docs-dir .` if your guidance lives in the repo root).
- Re-run the script whenever you add or edit documentation so embeddings stay in sync.


### LangSmith debugging (optional)

LangSmith tracing is wired into the agent workflow so you can inspect every prompt, retry, and aggregation pipeline that gets executed.

1. [Create a LangSmith account](https://smith.langchain.com/) and an API key.
2. Copy `example.env` to `.env` (if you have not already) and set:
   - `LANGCHAIN_TRACING_V2=true`
   - `LANGCHAIN_API_KEY=<your key>`
   - `LANGCHAIN_PROJECT=procurement-agent-debug` (or any project name you prefer)
3. Start the app as usual with `docker compose up`.

Each `/chat` request now shows up in LangSmith with nested runs for:
- `procurement_chat` (top-level FastAPI request)
- `analyze_question` with per-attempt metadata showing the MongoDB prompt, model output, and validation errors
- `execute_mongodb_query` tool calls (captures the final pipeline)
- `format_response` (LLM that narrates the final answer)

Use these traces to debug malformed aggregation pipelines, replay prompts, and compare different prompt/model settings safely.


