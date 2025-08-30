# CSV Cleaning + Inference Stack (FastAPI · Streamlit · n8n · NGINX) — **Azure‑Hosted**

A small, containerized toolkit for CSV merging, column type inference, data cleaning (LLM‑powered + manual rules), and visualization. The stack exposes a **FastAPI** backend, a **Streamlit** frontend, an **n8n** automation workflow, and an **NGINX** landing page that links everything together.

> **Live deployment:** This project is currently hosted on a **Microsoft Azure** VM (Ubuntu 22.04) with Docker Compose and NGINX. The README includes Azure‑specific steps. (No AWS is used for hosting; any S3 mentions are optional roadmap items only.)

---

## ✨ Features

* **CSV Merge API:** join two CSVs by one or more keys via file upload or file path.
* **Column Inference:** fast heuristics (pandas dtypes + low‑cardinality rules) to label columns.
* **LLM Cleaning:** send natural‑language instructions; the API requests code from an LLM and executes it on the DataFrame (⚠️ powerful but dangerous, see *Security*).
* **Manual Cleaning:** deterministic pipeline driven by form parameters (remove NaNs, impute, outlier removal, scaling, normalization, binarization, one‑hot encoding, etc.).
* **Visualization Hub (Streamlit):** inspect inferred types, show charts per column, preview image/video/document URLs, summarize webpages with the LLM, and download cleaned datasets.
* **n8n Workflow:** example automation (importable from `workflow.json`).
* **NGINX Landing Page:** simple index page linking to Streamlit, FastAPI docs, and n8n.

---

## 🗂️ Repository Structure

```
N8N_FAST_API/
├─ nginx/
│  └─ index.production.html         # Landing page with links to Streamlit / FastAPI / n8n
├─ .dockerignore
├─ .env.prod                        # Environment variables for docker-compose (create your own)
├─ .gitignore
├─ app.py                           # FastAPI backend (root_path="/fastapi")
├─ docker-compose.yml               # Orchestrates FastAPI, Streamlit, n8n, and NGINX
├─ fastapi.Dockerfile               # Image for the FastAPI app
├─ nginx.conf.template              # NGINX config (env‑templated)
├─ requirements.txt                 # Python dependencies for both services
├─ streamlit.Dockerfile             # Image for the Streamlit UI (ui.py)
├─ ui.py                            # Streamlit app (dashboard)
└─ workflow.json                    # n8n workflow export (importable)
```

**Note on paths:** FastAPI is created with `FastAPI(title="CSV Merge API", root_path="/fastapi")`. Routes are mounted at `/` (e.g., `/merge`, `/inference`), while OpenAPI is served under the `root_path` when behind a reverse proxy. Locally you can still use `http://localhost:8000/docs` while NGINX can expose `/fastapi/docs` externally.

---

## 🧱 Architecture

* **FastAPI (`app.py`)** — business logic (merge, inference, LLM cleaning, manual cleaning) and cached `last_*` endpoints.
* **Streamlit (`ui.py`)** — 4 tabs: Column Inference, LLM Cleaning, Visualization, Manual Cleaning. Connects to the FastAPI service (defaults to `BASE_URL=http://127.0.0.1:8000`).
* **n8n** — optional workflow engine for forms and automation (import any workflow from 'workflows' folder).
* **NGINX** — reverse proxy and a static App Hub (see `nginx/index.production.html`).

---

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

## 🖥️ Streamlit App (`ui.py`)

Pages: Column Inference, LLM Cleaning, Visualization, Manual Cleaning.

`BASE_URL` defaults to `http://127.0.0.1:8000`. If you run behind NGINX with a different host, set an env var (see `.env.prod`) or read from `PUBLIC_BASE_URL` and propagate it into `ui.py`.

Visualizations include bar charts, box plots, word clouds, maps, heatmaps, and media previews. A URL summarizer is available via the LLM.

Run locally (after installing requirements):

```bash
streamlit run ui.py --server.port 8501 --server.address 0.0.0.0
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

This repo ships `nginx.conf.template` that maps `/fastapi/` → `fastapi:8000` and exposes a static landing page. Ensure these key blocks exist:

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

## 🔐 Security Notes

* **LLM code execution:** `/LLMCleaning` executes arbitrary Python returned by the model via `exec`. Only enable with trusted keys and users. Consider sandboxing, timeouts, resource limits, and whitelisting allowed modules. Log the returned code and keep audit trails.
* **File paths:** For `/merge`, file paths should point to safe, mounted locations. Prefer `/mergefileupload` for user input.
* **Auth:** Protect **n8n** (basic auth env vars) and consider JWT/API keys for FastAPI. Hide direct service ports behind NGINX where possible.

---

## 🧷 n8n Workflow

Import `workflow.json` into n8n (**Menu → Import from file**). Connect it to the `/manual_cleaning` endpoint or any custom nodes you need.

---

## 📦 Requirements (key libs)

* fastapi, uvicorn
* pandas, numpy, scikit‑learn, imbalanced‑learn
* nltk, tldextract, geopy
* requests, python‑dotenv, beautifulsoup4, pillow
* wordcloud, plotly, seaborn, matplotlib
* streamlit
* together (Python SDK)

Install all via:

```bash
pip install -r requirements.txt
```

---

## 🛣️ Roadmap / TODO

* Persist results to object storage (e.g., **Azure Blob**; S3 also possible but **not used** here).
* Add auth (JWT/API key) around FastAPI endpoints.
* Add unit tests and CI.
* Improve type inference rules & confidence scores.
* Safer LLM exec with a restricted sandbox.
* Make `BASE_URL` configurable via environment variable in `ui.py`.
* Add NGINX locations to proxy Streamlit (`/ui`) and n8n (`/n8n`) through port 80/443 only.

---

## 🙋 Troubleshooting

* **CORS / proxy issues:** Check `nginx.conf.template` and `FASTAPI_ROOT_PATH`. If Swagger UI can’t call the API behind a prefix, verify `root_path` matches the upstream location.
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
