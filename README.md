## Procurement Assistant Prototype

This repository contains a prototype AI agent that converts natural-language procurement questions into MongoDB queries over the California Department of General Services purchase order dataset.

### Quickstart

#### Option 1: Docker (Recommended)

1. **Set environment variables**
   Copy `.env.example` to `.env` and fill in the following values:
   - `OPENAI_API_KEY` → your OpenAI API key.

2. **Start all services with Docker Compose**
   ```bash
   docker-compose up --build
   ```
   This will start MongoDB, the FastAPI backend, and Streamlit UI.

3. **Load the dataset**
   ```bash
   docker-compose exec app python -m src.data_loader --csv "data/PURCHASE ORDER DATA EXTRACT.csv"
   ```

4. **Access the application**
   - **Streamlit UI**: http://localhost:8501
   - **FastAPI backend**: http://localhost:8000
   - **API docs**: http://localhost:8000/docs

#### Option 2: Local Development (Virtual Environment)

1. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Unix/Mac:
   source .venv/bin/activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set environment variables**
   Copy `.env.example` to `.env` and fill in the following values:
   - `MONGODB_URI` → connection string to your MongoDB instance (e.g., `mongodb://localhost:27017`).
   - `MONGODB_DB` → database name to load the purchases collection into.
   - `OPENAI_API_KEY` (or another provider key if you swap LLMs).

4. **Run the application (Easy way)**
   ```bash
   python run_local.py
   ```
   This will start the FastAPI server with auto-reload.

5. **Or run manually**
   ```bash
   # Load data (optional - requires MongoDB running)
   python -m src.data_loader --csv "PURCHASE ORDER DATA EXTRACT.csv"

   # Backend (in one terminal)
   uvicorn src.server:app --reload

   # Frontend (in another terminal)
   streamlit run src/ui.py
   ```

6. **Run tests**
   ```bash
   python -m pytest tests/ -v
   # Or on Windows:
   activate.bat test
   ```

### Windows Convenience Scripts

For Windows users, you can use the `activate.bat` script:

```cmd
# Activate virtual environment
activate.bat

# Run tests
activate.bat test

# Start server
activate.bat run

# Start FastAPI server only
activate.bat server

# Start Streamlit UI only
activate.bat ui
```

### Architecture

- `src/data_loader.py` — incremental CSV → MongoDB ingestion with schema normalization.
- `src/agent.py` — LangChain + LangGraph agent that converts natural language to MongoDB queries and executes them.
- `src/server.py` — FastAPI app exposing `/chat` and `/health`.
- `src/ui.py` — Streamlit chat client that calls the backend.
- `notebooks/eda.ipynb` — exploratory data analysis notebook.

### Docker Services

- **mongodb**: MongoDB 7.0 database with authentication
- **app**: FastAPI backend service with health checks
- **streamlit**: Streamlit UI service

### Docker Commands

```bash
# Start all services
docker-compose up --build

# Start in background
docker-compose up -d --build

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
- `python scripts/validate_mongodb_schema.py --expected-count 919734 --sample-size 500` — quick health check to confirm MongoDB ingestion completeness and field coverage.

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
3. Start the app as usual (`docker-compose up` or `python run_local.py`).

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

