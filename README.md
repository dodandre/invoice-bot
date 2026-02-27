# Invoice Bot — Memory-Powered SQL Assistant

Streamlit app that answers natural-language questions against SAP ERP data (invoices, sales orders, delivery, credit risk, etc.). It uses OpenAI/LangChain to pick tables, generate SQL, and run queries. Supports document flow tracing (order → delivery → invoice → accounting), credit risk checks, and multi-provider insights (ChatGPT, Claude, Gemini, Perplexity).

## Stack

- **Python 3** — Streamlit, OpenAI, LangChain, pyodbc, pandas, Altair
- **Backend** — SQL Server (SAP ERP tables: VBRK, VBRP, VBAK, VBAP, LIKP, LIPS, VBFA, KNKK, BSAD, etc.)
- **Optional** — Kafka, PDF context, voice input

## Setup

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/Invoice_bot.git
cd Invoice_bot
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate
pip install -r requirements.txt
```

### 2. ODBC driver

Install **ODBC Driver 17 for SQL Server** (or 18) so pyodbc can connect to your SQL Server / SAP database.

### 3. Configuration

**Do not commit `config.py`** — it holds database and API credentials.

- Copy the example config and add your values:
  ```bash
  copy config.example.py config.py   # Windows
  # cp config.example.py config.py   # macOS/Linux
  ```
- Edit `config.py` and set:
  - `OPENAI_API_KEY` (required)
  - `SQL_SERVER`, `SQL_DATABASE`, `SQL_USERNAME`, `SQL_PASSWORD` for your SAP ERP database
  - Optional: `ANTHROPIC_API_KEY`, `GOOGLE_GEMINI_API_KEY`, `PERPLEXITY_API_KEY` for extra insights

You can also use environment variables; the example config reads from `os.environ` where possible.

### 4. Run

```bash
streamlit run main.py
```

Open the URL shown in the terminal (e.g. http://localhost:8501).

## Features

- **Natural language → SQL**: Ask questions in plain English; the app selects tables, builds joins, and runs queries.
- **Document flow**: Trace order number, delivery number, invoice number, and accounting document with item-level breakdown (VBAP, LIPS, VBRP) and links (VBFA, BSAD/BSID).
- **Credit risk**: e.g. “Does invoice number 90035998 have high credit risk?” — links VBRK to KNKK/KNA1.
- **Query results**: Table + charts and optional insights from multiple AI providers.

## Project layout

- `main.py` — Streamlit UI, Trace sidebar, query flow
- `functions.py` — Table selection, SQL generation, `json_to_sql`, trace, document flow, run_sql
- `config.py` — **Local only** (not in repo): credentials, table descriptions, CONN_STR
- `config.example.py` — Template for `config.py`
- `table_mapping/` — JSON column mappings per SAP table
- `requirements.txt` — Python dependencies

## Pushing to GitHub

1. Ensure `.gitignore` is in place (it excludes `config.py`, `chatbot/`, `__pycache__/`, etc.).
2. Initialize and push:
   ```bash
   git init
   git add .
   git status   # confirm config.py is not listed
   git commit -m "Initial commit: Invoice bot SQL assistant"
   git remote add origin https://github.com/YOUR_USERNAME/Invoice_bot.git
   git branch -M main
   git push -u origin main
   ```
3. If `config.py` was ever committed, remove it from history and add to `.gitignore`, then force-push (or use a new repo).

## License

Use and modify as needed for your organization.
