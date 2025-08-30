# CSV Cleaning + Inference Stack (FastAPI · Streamlit · n8n · NGINX)

A small, containerized toolkit for **CSV merging, column type inference, data cleaning (LLM‑powered + manual rules), and visualization**. The stack exposes a **FastAPI** backend, a **Streamlit** frontend, an **n8n** automation workflow, and an **NGINX** landing page that links everything together.

---

## ✨ Features

* **CSV Merge API**: join two CSVs by one or more keys via file upload or file path.
* **Column Inference**: fast heuristics (pandas dtypes + low‑cardinality rules) to label columns.
* **LLM Cleaning**: send natural‑language instructions; the API requests code from an LLM and executes it on the DataFrame (⚠️ *powerful but dangerous, see Security*).
* **Manual Cleaning**: deterministic pipeline driven by form parameters (remove NaNs, impute, outlier removal, scaling, normalization, binarization, one‑hot encoding, etc.).
* **Visualization Hub (Streamlit)**: inspect inferred types, show charts per column, preview image/video/document URLs, summarize webpages with the LLM, and download cleaned datasets.
* **n8n Workflow**: example automation (importable from `workflow.json`).
* **NGINX Landing Page**: simple index page linking to Streamlit, FastAPI docs, and n8n.

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

> **Note on paths**: FastAPI is created with `FastAPI(title="CSV Merge API", root_path="/fastapi")`. Routes are **mounted at `/`** (e.g., `/merge`, `/inference`), but OpenAPI **serves under the root\_path** when behind a reverse proxy. Locally you can still use `http://localhost:8000/docs` while NGINX can expose `/fastapi/docs` externally.

---

## 🧱 Architecture

* **FastAPI** (`app.py`): business logic (merge, inference, LLM cleaning, manual cleaning) and cached "last\_\*" endpoints.
* **Streamlit** (`ui.py`): 4 tabs — Column Inference, LLM Cleaning, Visualization, Manual Cleaning. Connects to the FastAPI service (`BASE_URL=http://127.0.0.1:8000` by default).
* **n8n**: optional workflow engine for forms and automation (import `workflow.json`).
* **NGINX**: optional reverse proxy and a static **App Hub** (see `nginx/index.production.html`).

---

## 🧩 Endpoints (FastAPI)

Base app: `app = FastAPI(title="CSV Merge API", root_path="/fastapi")`

### 1) Merge two CSVs by key(s)

**POST** `/merge`

* **Form fields**

  * `file1_path` (str, required): Path to first CSV on the server/container.
  * `file2_path` (str, required): Path to second CSV on the server/container.
  * `on` (str, required): Comma‑separated join keys, e.g., `id` or `id,date`.
  * `how` (str, optional): `inner|left|right|outer` (default `inner`).
* **Response**: streamed `merged.csv` file.

**POST** `/mergefileupload`

* **Form fields**

  * `file1` (UploadFile, required)
  * `file2` (UploadFile, required)
  * `on` (str, required): join keys
  * `how` (str, default `inner`)
* **Response**: streamed `merged.csv` file.

### 2) Column type inference

**POST** `/inference`

* **Form fields**

  * `csv_text` (str, required): Entire CSV **as text**.
* **Logic**: pandas dtype checks (int/float/bool/datetime). Otherwise `categorical` if unique ratio < 5%, else `string`.
* **Response (JSON)**: `{ "columns": { "col": "type", ... }, "rows": <int> }`

**GET** `/last_inference`

* Returns the last inference result (404 if none).

### 3) LLM‑powered cleaning

**POST** `/LLMCleaning`

* **Form fields**

  * `csv_text` (str, required): Entire CSV as text
  * `instruction` (str, required): Natural‑language cleaning instruction
* **Flow**: strict prompt → Together API → expects JSON `{ "code": "<python>" }` → `exec` on a copy of `df`.
* **Response**: streamed `cleaned.csv`. Also caches: instruction, executed code, and cleaned CSV.

**GET** `/last_cleaning`

* Returns last LLM cleaning metadata + CSV (404 if none).

### 4) Manual cleaning (deterministic)

**POST** `/manual_cleaning`

* **Body (JSON)**

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
* **Response**: streamed `manual_cleaned.csv` and cached metadata (row/column deltas, logs of applied steps).

**GET** `/last_manual_cleaning`

* Returns last manual cleaning metadata + CSV (404 if none).

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

**Inference** (send CSV content as form field)

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

* **Pages**: *Column Inference*, *LLM Cleaning*, *Visualization*, *Manual Cleaning*.
* **BASE\_URL** defaults to `http://127.0.0.1:8000`. If you run behind NGINX with a different host, update this constant or set an env var and read it in `ui.py`.
* Visualizations include bar charts, box plots, word clouds, maps, heatmaps, and media previews. A URL summarizer is also available via the LLM.

Run locally (after installing requirements):

```bash
streamlit run ui.py --server.port 8501 --server.address 0.0.0.0
```

---

## 🐳 Run with Docker Compose

1. Create and fill **`.env.prod`** (example):

```
# General
PYTHONUNBUFFERED=1

# Together API (LLM)
TOGETHER_API_KEY=sk_your_key_here

# Optional: n8n basic auth
N8N_BASIC_AUTH_ACTIVE=true
N8N_BASIC_AUTH_USER=admin
N8N_BASIC_AUTH_PASSWORD=strongpassword

# Optional: root path if you proxy FastAPI
FASTAPI_ROOT_PATH=/fastapi
```

2. Build and start:

```bash
docker compose up -d --build
```

3. Open services:

* **FastAPI docs**: [http://localhost:8000/docs](http://localhost:8000/docs)  *(or via NGINX: `/fastapi/docs` if configured)*
* **Streamlit**:  [http://localhost:8501/](http://localhost:8501/)
* **n8n**:       [http://localhost:5678/](http://localhost:5678/)
* **Landing page**: [http://localhost/](http://localhost/)  *(if NGINX exposes port 80)*

> **NLTK data**: If you see errors for `punkt`/`stopwords`, add to your Dockerfile: `RUN python -m nltk.downloader punkt stopwords`.

---

## 🔧 Local Development (without Docker)

```bash
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000  # then open http://localhost:8000/docs
```

> The app is instantiated with `root_path="/fastapi"` for proxy friendliness; routes are still at `/...`. You can keep it as‑is locally.

Run Streamlit in another shell:

```bash
streamlit run ui.py
```

---

## 🔐 Security Footnotes

* **LLM code execution**: `/LLMCleaning` executes arbitrary Python returned by the model inside the process via `exec`. **Only enable with trusted keys and users**. Consider sandboxing, timeouts, & whitelisting allowed modules. Log the returned code (`executed_code`) and keep audit trails.
* **File paths** on `/merge` should point to safe, mounted locations. Prefer `/mergefileupload` for user input.

---

## 🧷 n8n Workflow

* Import `workflow.json` into n8n (menu → *Import from file*). Connect it to the `/manual_cleaning` endpoint or any custom nodes you need.

---

## 📦 Requirements (key libs)

* FastAPI, Uvicorn
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

* [ ] Persist results to object storage (S3/Azure Blob) instead of in‑memory cache.
* [ ] Add auth (JWT/API key) around FastAPI endpoints.
* [ ] Add unit tests and CI.
* [ ] Improve type inference rules & confidence scores.
* [ ] Safer LLM exec with a restricted sandbox.
* [ ] Make `BASE_URL` configurable via environment variable in `ui.py`.

---

## 📄 License

MIT (update if you prefer a different license).

---

## 🙏 Acknowledgments

* [FastAPI](https://fastapi.tiangolo.com/)
* [Streamlit](https://streamlit.io/)
* [n8n](https://n8n.io/)
* [Together.ai](https://www.together.ai/) (Kimi‑K2 Instruct model used in examples)

---

## 🙋 Troubleshooting

* **CORS / reverse proxy issues**: Check `nginx.conf.template` and `root_path`. If swagger UI can’t call the API behind a prefix, verify that `root_path` matches the upstream location.
* **NLTK missing resources**: Install/download `punkt` & `stopwords`.
* **Large CSVs in forms**: For `/inference` and `/LLMCleaning`, you’re sending the whole CSV as a **form field**. Prefer upload endpoints for very large files.
* **Model/API errors**: Ensure `TOGETHER_API_KEY` is set and the chosen model ID exists.
