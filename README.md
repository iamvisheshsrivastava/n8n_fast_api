# Production-Ready CSV Processing & ML Inference Service (FastAPI · Docker · n8n)   

## Executive Summary
End-to-end, containerized data processing and ML inference service built with FastAPI and Docker.
Supports CSV merging, column type inference, deterministic and LLM-assisted cleaning, and workflow automation via n8n.
Designed for local development and VM-based deployment with reverse proxying and clear API boundaries.

## Deployment
The stack has been deployed on a Linux VM using Docker Compose and NGINX.
Cloud-specific steps (Azure VM) are documented below, but the setup is cloud-agnostic.

---

## Key Features
- FastAPI backend for CSV merging, inference, and cleaning pipelines
- Deterministic preprocessing and optional LLM-assisted transformations
- n8n workflows for automation and integration
- Streamlit UI for inspection and visualization
- Fully containerized with Docker Compose and NGINX reverse proxy

---

## 🗂️ Repository Structure

```
N8N_FAST_API/
├─ backend/
│  ├─ app/
│  │  ├─ __init__.py
│  │  └─ main.py                    # FastAPI backend (root_path="/fastapi")
│  └─ requirements.txt              # Backend dependencies
├─ frontend/
│  ├─ streamlit_app.py              # Streamlit app (dashboard)
│  └─ requirements.txt              # Frontend dependencies
├─ docker/
│  ├─ backend.Dockerfile            # Image for the FastAPI app
│  └─ frontend.Dockerfile           # Image for the Streamlit UI
├─ infra/
│  └─ nginx/
│     ├─ index.local.html
│     ├─ index.production.html      # Landing page with links to Streamlit / FastAPI / n8n
│     └─ nginx.conf.template        # NGINX config (env-templated)
├─ workflows/
│  ├─ workflow.json
│  ├─ WorkflowFileUpload.json
│  └─ Working_Flow.json
├─ .dockerignore
├─ .env.prod                        # Environment variables for docker-compose (create your own)
├─ .gitignore
├─ docker-compose.yml               # Orchestrates FastAPI, Streamlit, n8n, and NGINX
└─ README.md
```

**Note on paths:** FastAPI is created with `FastAPI(title="CSV Merge API", root_path="/fastapi")`. Routes are mounted at `/` (e.g., `/merge`, `/inference`), while OpenAPI is served under the `root_path` when behind a reverse proxy. Locally you can still use `http://localhost:8000/docs` while NGINX can expose `/fastapi/docs` externally.

---

## 🧱 Architecture

* **FastAPI (`backend/app/main.py`)** — business logic (merge, inference, LLM cleaning, manual cleaning) and cached `last_*` endpoints.
* **Streamlit (`frontend/streamlit_app.py`)** — 4 tabs: Column Inference, LLM Cleaning, Visualization, Manual Cleaning. Connects to the FastAPI service (defaults to `FASTAPI_BASE_URL=http://127.0.0.1:8000`).
* **n8n** — optional workflow engine for forms and automation (import any workflow from 'workflows' folder).
* **NGINX** — reverse proxy and a static App Hub (see `infra/nginx/index.production.html`).

---

## API Overview
The FastAPI service exposes clear, stateless endpoints designed for automation and integration into data workflows.

## 🧩 API Endpoints (FastAPI)

**Base app:** `app = FastAPI(title="CSV Merge API", root_path="/fastapi")`

### 1) Merge two CSVs by key(s)

**`POST /merge`**
Form fields:

* `file1_path` *(str, required)*: path to first CSV on the server/container.
* `file2_path` *(str, required)*: path to second CSV on the server/container.
* `on` *(str, required)*: comma‑separated join keys (e.g., `id` or `id,date`).
* `how` *(str, optional)*: `inner|left|right|outer` (default `inner`).

**`POST /mergefileupload`**
Form fields:

* `file1` *(UploadFile, required)*
* `file2` *(UploadFile, required)*
* `on` *(str, required)* — join keys
* `how` *(str, default `inner`)*

**Response:** streamed `merged.csv` file.

### 2) Column type inference

**`POST /inference`**
Form fields:

* `csv_text` *(str, required)* — entire CSV as text

Logic: pandas dtype checks (int/float/bool/datetime). Otherwise **categorical** if unique ratio < 5%, else **string**.

**Response (JSON):** `{ "columns": { "col": "type", ... }, "rows": <int> }`

**`GET /last_inference`** — returns the last inference result (404 if none).

### 3) LLM‑powered cleaning

**`POST /LLMCleaning`**
Form fields:

* `csv_text` *(str, required)* — entire CSV as text
* `instruction` *(str, required)* — natural‑language cleaning instruction

Flow: strict prompt → Together API → expects JSON `{ "code": "<python>" }` → `exec` on a copy of `df`.

**Response:** streamed `cleaned.csv`. Also caches: instruction, executed code, and cleaned CSV.

**`GET /last_cleaning`** — returns last LLM cleaning metadata + CSV (404 if none).

### 4) Manual cleaning (deterministic)

**`POST /manual_cleaning`**
Body (JSON):

```json
{
  "data": "<CSV text>",
  "params": {
    "Select Preprocessing Step(s)": [
      "Remove Columns with Excessive NaNs",
      "Remove Rows with Excessive NaNs",
      "Impute Missing Values",
      "Remove Outliers",
      "Scale",
      "Normalize",
      "Binarize",
      "One-Hot Encoding"
    ],
    "NaN Threshold for Column Removal": 0.5,
    "NaN Threshold for Row Removal": 0.5,
    "Impute: strategy": "mean",
    "Impute: fill_value (used when strategy=constant)": 0,
    "Remove Outliers: iqr_multiplier": 1.5,
    "Scale: min_value": 0.0,
    "Scale: max_value": 1.0,
    "Binarize: threshold": 0.0
  }
}
```

**Response:** streamed `manual_cleaned.csv` and cached metadata (row/column deltas, logs of applied steps).

**`GET /last_manual_cleaning`** — returns last manual cleaning metadata + CSV (404 if none).

---

## 🧪 cURL Examples

**Upload‑based merge**

```bash
curl -X POST http://localhost:8000/mergefileupload \
  -F "file1=@/path/to/a.csv" \
  -F "file2=@/path/to/b.csv" \
  -F "on=id" \
  -F "how=left" \
  -o merged.csv
```

**Inference (send CSV content as form field)**

```bash
curl -X POST http://localhost:8000/inference \
  --data-urlencode "csv_text=$(cat merged.csv)"
```

**LLM Cleaning**

```bash
curl -X POST http://localhost:8000/LLMCleaning \
  --data-urlencode "csv_text=$(cat merged.csv)" \
  --data-urlencode "instruction=drop rows with null price, convert date to datetime, and standardize numeric columns"
```

**Manual Cleaning**

```bash
curl -X POST http://localhost:8000/manual_cleaning \
  -H "Content-Type: application/json" \
  -d @params.json \
  -o manual_cleaned.csv
```

---

## 🖥️ Streamlit App (`frontend/streamlit_app.py`)

Pages: Column Inference, LLM Cleaning, Visualization, Manual Cleaning.

`FASTAPI_BASE_URL` defaults to `http://127.0.0.1:8000`. If you run behind NGINX with a different host, set this env var in `.env` or `.env.prod`.

Visualizations include bar charts, box plots, word clouds, maps, heatmaps, and media previews. A URL summarizer is available via the LLM.

Run locally (after installing requirements):

```bash
streamlit run frontend/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

---

## 🐳 Run with Docker Compose (Local)

Create and fill `.env.prod` (example):

```ini
# General
PYTHONUNBUFFERED=1

# Public base URL for the landing page and client links
PUBLIC_BASE_URL=http://localhost   # set to http://<YOUR_IP> in Azure

# Together API (LLM)
TOGETHER_API_KEY=sk_your_key_here

# Optional: n8n basic auth
N8N_BASIC_AUTH_ACTIVE=true
N8N_BASIC_AUTH_USER=admin
N8N_BASIC_AUTH_PASSWORD=strongpassword

# Optional: FastAPI root path if you proxy it behind /fastapi
FASTAPI_ROOT_PATH=/fastapi
```

Build and start:

```bash
docker compose up -d --build
```

Open services:

* FastAPI docs: [http://localhost:8000/docs](http://localhost:8000/docs) (or via NGINX: `/fastapi/docs` if configured)
* Streamlit: [http://localhost:8501/](http://localhost:8501/)
* n8n: [http://localhost:5678/](http://localhost:5678/)
* Landing page: [http://localhost/](http://localhost/) (if NGINX exposes port 80)

**NLTK data:** If you see errors for `punkt/stopwords`, add to your Dockerfile:

```dockerfile
RUN python -m nltk.downloader punkt stopwords
```

---

## ☁️ Deploying on **Azure VM** (Docker Compose + NGINX)

> These steps document how this repository is hosted on **Microsoft Azure**. Adjust for your own IP/domain. No AWS services are used in this deployment.

### 1) Create a VM

* Image: **Ubuntu 22.04 LTS**
* Size: `Standard_B2s_v2` (works fine for small workloads)
* Public IP: **Enabled** (consider reserving a static IP)
* Disk: Standard SSD is OK
* Auth: SSH recommended

### 2) Open inbound ports (Network Security Group)

Allow at minimum:

* **80/TCP** — NGINX landing page & proxied routes
* **8000/TCP** — (optional) direct FastAPI access for debugging
* **8501/TCP** — (optional) direct Streamlit access
* **5678/TCP** — n8n UI (protect with basic auth)

> For production, prefer exposing only **80/443** and proxy everything through NGINX.

### 3) Install Docker & docker compose

```bash
# as root or with sudo
apt-get update -y
apt-get install -y ca-certificates curl gnupg
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo $VERSION_CODENAME) stable" | \
  tee /etc/apt/sources.list.d/docker.list > /dev/null

apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
usermod -aG docker $USER   # log out/in to apply
```

### 4) Clone & configure

```bash
# clone your repo
cd ~
git clone https://github.com/iamvisheshsrivastava/n8n_fast_api.git
cd n8n_fast_api

# create prod env file
cp .env.prod .env
# then edit .env and set at minimum:
# PUBLIC_BASE_URL=http://<YOUR_PUBLIC_IP>
# TOGETHER_API_KEY=...
# FASTAPI_ROOT_PATH=/fastapi
# N8N_BASIC_AUTH_* (recommended)
```

### 5) (Optional) NGINX config

This repo ships `infra/nginx/nginx.conf.template` that maps `/fastapi/` → `fastapi:8000` and exposes a static landing page. Ensure these key blocks exist:

```nginx
location /fastapi/ {
    proxy_pass http://fastapi:8000/;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}

location / {
    root   /usr/share/nginx/html;   # serves index.production.html
    index  index.html index.htm;
}
```

The landing page reads `PUBLIC_BASE_URL` and renders links to **Streamlit**, **FastAPI docs** (under `/fastapi/docs`), and **n8n**.

### 6) Start the stack

```bash
docker compose up -d --build
```

### 7) Test (replace with your IP or domain)

* Landing page: `http://<YOUR_PUBLIC_IP>/`
* FastAPI docs (proxied): `http://<YOUR_PUBLIC_IP>/fastapi/docs`
* Streamlit: `http://<YOUR_PUBLIC_IP>:8501/` *(or add an NGINX location to proxy this under `/ui`)*
* n8n: `http://<YOUR_PUBLIC_IP>:5678/`

> Tip: point a domain at your IP and add TLS (Caddy or NGINX + Let’s Encrypt). In production, terminate TLS and proxy all UIs under friendly paths (e.g., `/ui`, `/n8n`).

---

## Security Notes
This project intentionally demonstrates both deterministic and LLM-driven data transformations.
LLM-based execution is powerful and must be sandboxed or restricted in production environments.

---

## 🧷 n8n Workflow

Import one of the files in `workflows/` into n8n (**Menu → Import from file**). Connect it to the `/manual_cleaning` endpoint or any custom nodes you need.

---

## 📦 Requirements (key libs)

* Backend: fastapi, uvicorn, pandas, numpy, scikit-learn, imbalanced-learn, nltk, tldextract, geopy, together
* Frontend: streamlit, pandas, requests, python-dotenv, beautifulsoup4, pillow, wordcloud, plotly, seaborn, matplotlib, together

Install backend dependencies:

```bash
pip install -r backend/requirements.txt
```

Install frontend dependencies:

```bash
pip install -r frontend/requirements.txt
```

---

## 🛣️ Roadmap / TODO

* Persist results to object storage (e.g., **Azure Blob**; S3 also possible but **not used** here).
* Add auth (JWT/API key) around FastAPI endpoints.
* Add unit tests and CI.
* Improve type inference rules & confidence scores.
* Safer LLM exec with a restricted sandbox.
* Add endpoint tests for `backend/app/main.py`.
* Add Streamlit UI smoke tests for `frontend/streamlit_app.py`.
* Add NGINX locations to proxy Streamlit (`/ui`) and n8n (`/n8n`) through port 80/443 only.

---

## 🙋 Troubleshooting

* **CORS / proxy issues:** Check `infra/nginx/nginx.conf.template` and `FASTAPI_ROOT_PATH`. If Swagger UI cannot call the API behind a prefix, verify `root_path` matches the upstream location.
* **NLTK missing resources:** Install/download `punkt` & `stopwords`.
* **Large CSVs in forms:** For `/inference` and `/LLMCleaning`, you’re sending the whole CSV as a form field. Prefer upload endpoints for very large files.
* **Model/API errors:** Ensure `TOGETHER_API_KEY` is set and the chosen model ID exists.

---

## 📄 License

MIT.

---

## 🙏 Acknowledgments

* FastAPI
* Streamlit
* n8n
* Together.ai (Kimi‑K2 Instruct model used in examples)
