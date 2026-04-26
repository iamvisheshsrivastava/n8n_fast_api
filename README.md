# n8n + FastAPI CSV Workflow

This repo is a small CSV workflow project built around three pieces:

- a FastAPI backend for merging CSVs, inferring column types, and cleaning data
- a Streamlit dashboard for viewing the latest results
- n8n workflows for automating the API calls

The project is Docker-based, so the backend, Streamlit app, n8n, and a simple NGINX homepage can run together with Docker Compose.

## What It Does

- Merge two CSV files by one or more columns
- Infer basic column types from CSV text
- Run LLM-based cleaning instructions through Together.ai
- Run manual cleaning steps like imputation, scaling, normalization, outlier removal, and one-hot encoding
- Store the latest inference or cleaning result in memory so the Streamlit app can display it
- Provide sample n8n workflow files that can be imported into n8n

## Project Structure

```text
n8n_fast_api/
├─ backend/
│  ├─ app/
│  │  ├─ __init__.py
│  │  └─ main.py              # FastAPI routes and cleaning logic
│  └─ requirements.txt
├─ frontend/
│  ├─ streamlit_app.py        # Streamlit dashboard
│  └─ requirements.txt
├─ docker/
│  ├─ backend.Dockerfile
│  └─ frontend.Dockerfile
├─ infra/
│  └─ nginx/
│     ├─ index.production.html
│     └─ nginx.conf.template
├─ workflows/
│  ├─ workflow.json
│  ├─ WorkflowFileUpload.json
│  └─ Working_Flow.json
├─ docker-compose.yml
├─ .env.prod
└─ README.md
```

## Main API Routes

FastAPI lives in `backend/app/main.py`.

| Route | Method | Purpose |
| --- | --- | --- |
| `/merge` | `POST` | Merge two CSV files using file paths available to the backend container |
| `/mergefileupload` | `POST` | Upload two CSV files and merge them |
| `/inference` | `POST` | Infer simple column types from CSV text |
| `/last_inference` | `GET` | Return the last inference result |
| `/LLMCleaning` | `POST` | Clean CSV text using a natural language instruction |
| `/last_cleaning` | `GET` | Return the last LLM cleaning result |
| `/manual_cleaning` | `POST` | Apply deterministic cleaning steps from JSON parameters |
| `/last_manual_cleaning` | `GET` | Return the last manual cleaning result |

## Run with Docker Compose

Create a local `.env` file first:

```bash
cp .env.prod .env
```

Update the values as needed:

```ini
PUBLIC_BASE_URL=http://localhost
TOGETHER_API_KEY=your_together_api_key
FASTAPI_ROOT_PATH=/fastapi
```

Then start the stack:

```bash
docker compose up -d --build
```

Default local URLs:

- FastAPI docs: `http://localhost:8000/docs`
- Streamlit: `http://localhost:8501`
- n8n: `http://localhost:5678`
- NGINX homepage: `http://localhost:8090`

## Run Parts Manually

Backend:

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Frontend:

```bash
cd frontend
pip install -r requirements.txt
streamlit run streamlit_app.py --server.port 8501
```

The Streamlit app reads the backend URL from `FASTAPI_BASE_URL`. If it is not set, it uses:

```text
http://127.0.0.1:8000
```

## Example Requests

Upload and merge two CSV files:

```bash
curl -X POST http://localhost:8000/mergefileupload \
  -F "file1=@first.csv" \
  -F "file2=@second.csv" \
  -F "on=id" \
  -F "how=left" \
  -o merged.csv
```

Infer columns:

```bash
curl -X POST http://localhost:8000/inference \
  --data-urlencode "csv_text=$(cat merged.csv)"
```

Run LLM cleaning:

```bash
curl -X POST http://localhost:8000/LLMCleaning \
  --data-urlencode "csv_text=$(cat merged.csv)" \
  --data-urlencode "instruction=drop rows with missing prices and standardize numeric columns" \
  -o cleaned.csv
```

## n8n Workflows

The `workflows/` folder contains sample n8n workflow exports. Import them from the n8n UI:

```text
Menu -> Import from file
```

After importing, check the API URLs inside the workflow nodes and update them for your machine or VM.

## Notes

- LLM cleaning executes generated Python code, so do not expose that endpoint publicly without adding proper sandboxing and authentication.
- The NGINX config in this repo is currently a simple static homepage, not a full reverse proxy for all services.
- Docker Compose mounts `./data` into the backend container at `/data`.
- The app stores latest results in memory. Restarting the backend clears them.

## License

MIT
