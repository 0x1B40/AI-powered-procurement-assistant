## Procurement Assistant Prototype

This repository contains a prototype AI agent that converts natural-language procurement questions into MongoDB queries over the California Department of General Services purchase order dataset.

### Quickstart

#### Option 1: Docker (Recommended)

1. **Set environment variables**
   Copy `.env.example` to `.env` and fill in the following values:
   - `OPENAI_API_KEY` → your OpenAI API key.

2. **Start all services with Docker Compose**
   ```bash
   docker-compose up
   ```
   This will start MongoDB, the FastAPI backend, and Streamlit UI.

3. **Load the dataset**
   ```bash
   docker-compose exec app python -m src.data_loader --csv "data/PURCHASE ORDER DATA EXTRACT.csv"
   ```

4. **Build the reference vector store**
   ```bash
   docker-compose exec app python -m scripts.build_reference_store --docs-dir data/
   ```
   This indexes any PDF/DOCX guidance stored under `data/` (e.g., the acquisition method memo or data dictionary) into Chroma so the agent can cite them later.

5. **Access the application**
   - **Streamlit UI**: http://localhost:8501
   - **FastAPI backend**: http://localhost:8000
   - **API docs**: http://localhost:8000/docs


### Architecture

The application follows a modular, layered architecture with clear separation of concerns:

#### Core Components
- **Agent System** (`src/agent/`): LangChain + LangGraph orchestration that converts natural language to MongoDB queries
- **Services** (`src/services/`): LLM integration, query processing, and response formatting
- **Data Layer** (`src/data/`): MongoDB connection and vector storage
- **Configuration** (`src/config/`): Settings management and constants
- **Interfaces** (`src/interfaces/`): FastAPI backend and Streamlit interface
- **Shared Utilities** (`src/shared/`): Data loading and telemetry

#### File Structure

```
src/
├── __init__.py              # Main package interface exposing chat()
├── agent/                   # Core agent functionality
│   ├── __init__.py
│   ├── agent.py            # Main chat interface (65 lines)
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
│   └── vector_store.py      # Vector storage for reference documents
├── config/                  # Configuration & constants
│   ├── __init__.py
│   ├── config.py           # Settings management (env vars, etc.)
│   └── constants.py        # Prompts, enums, schema definitions
├── interfaces/              # User interfaces
│   ├── __init__.py
│   ├── api_server.py      # FastAPI backend server
│   └── web_ui.py           # Streamlit web interface
└── shared/                  # Shared utilities
│   ├── __init__.py
│   ├── data_loader.py      # CSV → MongoDB ingestion
│   ├── telemetry.py        # Logging & tracing (LangSmith integration)
│   └── vector_store.py     # Chroma vector DB for reference docs
└── api/                     # API & UI components
    ├── __init__.py
    ├── server.py           # FastAPI backend (/chat, /health endpoints)
    └── ui.py               # Streamlit chat interface
```

#### Key Design Patterns

- **Modular Architecture**: Each folder contains focused, single-responsibility modules
- **Dependency Injection**: Clean separation between LLM, database, and business logic
- **Error Handling**: Comprehensive validation and retry mechanisms for query generation
- **Observability**: Built-in telemetry with LangSmith tracing for debugging
- **Testability**: Modular design enables isolated unit testing

#### Data Flow

1. **User Input** → `src/interfaces/api_server.py` (FastAPI endpoint)
2. **Question Classification** → `src/services/llm_service/classification.py` (routes to appropriate handler)
3. **Query Generation** → `src/services/query_service/query_generation.py` (LLM creates MongoDB pipeline)
4. **Database Execution** → `src/data/mongodb.py` (executes aggregation pipeline)
5. **Response Formatting** → `src/query/response_formatting.py` (converts results to natural language)
6. **Final Response** → User via API/UI

### Docker Services

- **mongodb**: MongoDB 7.0 database with authentication
- **app**: FastAPI backend service with health checks
- **streamlit**: Streamlit UI service

### Docker Commands

```bash
# Start all services
docker-compose up

# Start in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Remove volumes (will delete data)
docker-compose down -v

# Load data into running container
docker-compose exec app python -m src.data_loader --csv "data/PURCHASE ORDER DATA EXTRACT.csv"
```

### Data Setup

For Docker usage, place your data files in the `data/` directory:
- `data/PURCHASE ORDER DATA EXTRACT.csv` (156MB dataset)
- Other documentation files can remain in the root or data directory

### Data Documentation & Validation

- `docs/data_schema.md` — canonical schema reference covering every column, data type, and known data-quality caveats.
- `python data_scripts/validate_mongodb_schema.py --expected-count 919734 --sample-size 500` — quick health check to confirm MongoDB ingestion completeness and field coverage.

### Reference Document Retrieval

- `data_scripts/build_reference_store.py` walks all PDF/DOCX files inside `REFERENCE_DOCS_DIR` (defaults to `data/`) and chunks them into a persistent Chroma DB located at `VECTOR_STORE_DIR`.
- Runtime questions classified as `database_info` or `acquisition_methods` are answered directly from the retrieved passages instead of running MongoDB queries.
- Customize ingestion via environment variables or pass `--docs-dir` / `--persist-dir` flags when running the script (e.g., `python -m scripts.build_reference_store --docs-dir .` if your guidance lives in the repo root).
- Re-run the script whenever you add or edit documentation so embeddings stay in sync.

### Testing Queries

Sample prompts you can use once the system is running:

- "How many purchase orders were created in Q3 2014?"
- "Which quarter saw the highest total spend?"
- "List the top 5 most frequently ordered line items."
- "What's the average order amount for technology-related purchases in 2013?"

### LangSmith debugging (optional)

LangSmith tracing is wired into the MongoDB agent so you can inspect every prompt, retry, and aggregation pipeline that gets executed.

1. [Create a LangSmith account](https://smith.langchain.com/) and an API key.
2. Copy `.env.example` to `.env` (if you have not already) and set:
   - `LANGCHAIN_TRACING_V2=true`
   - `LANGCHAIN_API_KEY=<your key>`
   - `LANGCHAIN_PROJECT=procurement-agent-debug` (or any project name you prefer)
3. Start the app as usual with `docker-compose up`.

Each `/chat` request now shows up in LangSmith with nested runs for:
- `procurement_chat` (top-level FastAPI request)
- `analyze_question` with per-attempt metadata showing the MongoDB prompt, model output, and validation errors
- `execute_mongodb_query` tool calls (captures the final pipeline)
- `format_response` (LLM that narrates the final answer)

Use these traces to debug malformed aggregation pipelines, replay prompts, and compare different prompt/model settings safely.

### Deliverables Checklist

- [x] Data loader and MongoDB schema definition.
- [x] AI agent that translates chat → MongoDB queries (LangChain + LangGraph implementation).
- [x] Conversational interface (API + Streamlit).
- [x] Docker containerization with docker-compose.
- [x] Unit tests for key components.
- [x] Exploratory data analysis notebook.
- [ ] Video walkthrough (record after functionality is verified).

