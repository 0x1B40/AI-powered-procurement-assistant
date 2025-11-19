## Procurement Assistant Prototype

This repository contains a prototype AI agent that converts natural-language procurement questions into MongoDB queries over the California Department of General Services purchase order dataset.

### Quickstart

1. **Install dependencies**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. **Set environment variables**
   Copy `.env.example` to `.env` and fill in the following values:
   - `MONGODB_URI` → connection string to your MongoDB instance.
   - `MONGODB_DB` → database name to load the purchases collection into.
   - `OPENAI_API_KEY` (or another provider key if you swap LLMs).
3. **Load the dataset**
   ```bash
   python -m src.data_loader --csv "PURCHASE ORDER DATA EXTRACT.csv"
   ```
   The loader streams the 156 MB CSV in chunks and writes to MongoDB.
4. **Run the backend**
   ```bash
   uvicorn src.server:app --reload
   ```
5. **Start the chat UI**
   ```bash
   streamlit run src/ui.py
   ```

### Architecture

- `src/data_loader.py` — incremental CSV → MongoDB ingestion with schema normalization.
- `src/agent.py` — LangChain agent that grounds an LLM with a MongoDB Aggregation Evaluation Tool.
- `src/server.py` — FastAPI app exposing `/chat` and `/health`.
- `src/ui.py` — Streamlit chat client that calls the backend.
- `notebooks/eda.ipynb` — optional exploratory notebook stub.

### Testing Queries

Sample prompts you can use once the system is running:

- “How many purchase orders were created in Q3 2014?”
- “Which quarter saw the highest total spend?”
- “List the top 5 most frequently ordered line items.”
- “What’s the average order amount for technology-related purchases in 2013?”

### Deliverables Checklist

- [x] Data loader and MongoDB schema definition.
- [x] LLM agent that translates chat → Mongo pipeline.
- [x] Conversational interface (API + Streamlit).
- [ ] Video walkthrough (record after functionality is verified).

