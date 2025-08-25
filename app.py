from fastapi import FastAPI, Form, Body, HTTPException
from fastapi.responses import StreamingResponse
from fastapi import UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
from io import BytesIO
from dotenv import load_dotenv
import numpy as np
import re
import datetime
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Binarizer
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import tldextract
from geopy.distance import geodesic
import json
from together import Together
from typing import Tuple
import os
from typing import Any, Dict, Tuple, List
from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Binarizer
from io import StringIO, BytesIO


load_dotenv()

app = FastAPI(title="CSV Merge API", root_path="/fastapi")

@app.post("/merge")
async def merge_csv(
    file1_path: str = Form(...),
    file2_path: str = Form(...),
    on: str = Form(...),            
    how: str = Form("inner")       
):
    how = how.lower().strip()
    if how not in {"inner", "left", "right", "outer"}:
        raise HTTPException(400, f"Invalid join type: {how}")

    join_cols = [c.strip() for c in on.split(",") if c.strip()]
    if not join_cols:
        raise HTTPException(400, "Provide at least one join column via 'on'")

    try:
        df1 = pd.read_csv(file1_path)
        df2 = pd.read_csv(file2_path)
    except Exception as e:
        raise HTTPException(400, f"Error reading CSVs: {e}")

    try:
        df_merged = df1.merge(df2, on=join_cols, how=how, suffixes=("", "_2"))
    except Exception as e:
        raise HTTPException(400, f"Merge failed: {e}")

    out = BytesIO()
    df_merged.to_csv(out, index=False)
    out.seek(0)

    return StreamingResponse(
        out,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="merged.csv"'}
    )

@app.post("/mergefileupload")
async def mergefileupload(
    file1: UploadFile,
    file2: UploadFile,
    on: str = Form(...),            
    how: str = Form("inner")       
):
    how = how.lower().strip()
    if how not in {"inner", "left", "right", "outer"}:
        raise HTTPException(400, f"Invalid join type: {how}")

    join_cols = [c.strip() for c in on.split(",") if c.strip()]
    if not join_cols:
        raise HTTPException(400, "Provide at least one join column via 'on'")

    try:
        df1 = pd.read_csv(file1.file)
        df2 = pd.read_csv(file2.file)
    except Exception as e:
        raise HTTPException(400, f"Error reading CSVs: {e}")

    try:
        df_merged = df1.merge(df2, on=join_cols, how=how, suffixes=("", "_2"))
    except Exception as e:
        raise HTTPException(400, f"Merge failed: {e}")

    out = BytesIO()
    df_merged.to_csv(out, index=False)
    out.seek(0)

    return StreamingResponse(
        out,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="merged.csv"'}
    )


last_inference_result = None
last_cleaning_result = None


@app.post("/inference")
async def infer_columns(csv_text: str = Form(...)):
    """
    Infer column types from a merged CSV string and cache the result.
    """
    global last_inference_result
    try:
        df = pd.read_csv(BytesIO(csv_text.encode("utf-8")))

        col_types = {}
        for col in df.columns:
            series = df[col]

            if pd.api.types.is_integer_dtype(series):
                inferred = "integer"
            elif pd.api.types.is_float_dtype(series):
                inferred = "float"
            elif pd.api.types.is_bool_dtype(series):
                inferred = "boolean"
            elif pd.api.types.is_datetime64_any_dtype(series):
                inferred = "datetime"
            else:
                if series.nunique() < (0.05 * len(series)):
                    inferred = "categorical"
                else:
                    inferred = "string"

            col_types[col] = inferred

        last_inference_result = {
            "columns": col_types,
            "rows": int(len(df))
        }

        return JSONResponse(last_inference_result)

    except Exception as e:
        raise HTTPException(400, f"Could not infer columns: {e}")


@app.get("/last_inference")
async def get_last_inference():
    """
    Return the last cached inference result.
    """
    if not last_inference_result:
        raise HTTPException(404, "No inference has been run yet")
    return JSONResponse(last_inference_result)


@app.post("/LLMCleaning")
async def llm_clean_csv(
    csv_text: str = Form(...),
    instruction: str = Form(...),
):
    """
    Clean dataset via LLM based on user instruction and cache result.
    """
    global last_cleaning_result
    try:
        df = pd.read_csv(BytesIO(csv_text.encode("utf-8")))

        cleaned_df, code_str = custom_cleaning_via_llm(instruction, df)

        cleaned_csv = cleaned_df.to_csv(index=False)

        last_cleaning_result = {
            "instruction": instruction,
            "executed_code": code_str,
            "cleaned_csv": cleaned_csv
        }

        out = BytesIO()
        out.write(cleaned_csv.encode("utf-8"))
        out.seek(0)

        return StreamingResponse(
            out,
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="cleaned.csv"'},
        )

    except Exception as e:
        raise HTTPException(400, f"Cleaning failed: {e}")


@app.get("/last_cleaning")
async def get_last_cleaning():
    """
    Return the last cached cleaning result.
    """
    if not last_cleaning_result:
        raise HTTPException(404, "No cleaning has been run yet")
    return JSONResponse(last_cleaning_result)



def custom_cleaning_via_llm(user_instruction: str, df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Calls the LLM with a strict prompt, parses returned code from JSON, executes it on df.

    Args:
        user_instruction (str): User's natural language cleaning instruction.
        df (pd.DataFrame): The DataFrame to modify.

    Returns:
        Tuple[pd.DataFrame, str]: Cleaned DataFrame and executed code.
    """
    sample_df = df.head(5)   
    formatted_df = sample_df.to_csv(index=False)

    #formatted_df = df.to_csv(index=False)
    prompt = f"""
You are a Python data cleaning assistant.

Your task is to generate Python code that modifies the `df` DataFrame *in-place* based on the user's instruction. You have access to the full dataset below in CSV format. 

Make sure the code:
- Assumes that a pandas DataFrame named `df` already exists.
- Can handle all rows of the dataset, not just a sample.
- Does not include any explanations, comments, or markdown.
- Only returns a JSON object of the form: {{ "code": "<your_python_code>" }}

USER INSTRUCTION:
{user_instruction}

FULL DATAFRAME (CSV FORMAT):
{formatted_df}

PROMPT LENGTH (characters): {len(user_instruction) + len(formatted_df)}
"""

    try:
        llm_response = call_llm(prompt)
        code_data = json.loads(llm_response)
        code_str = code_data.get("code", "")

        if not code_str:
            raise ValueError("No 'code' key found in LLM response.")

        global_vars = {
            "pd": pd,
            "np": np,
            "re": re,
            "datetime": datetime,
            "SimpleImputer": SimpleImputer,
            "StandardScaler": StandardScaler,
            "MinMaxScaler": MinMaxScaler,
            "Binarizer": Binarizer,
            "SMOTE": SMOTE,
            "RandomOverSampler": RandomOverSampler,
            "RandomUnderSampler": RandomUnderSampler,
            "nltk": nltk,
            "geopy": __import__("geopy"),  
            "geodesic": geodesic,
            #"stop_words": english_stops,
            "stops": set(stopwords.words("english")),
            "word_tokenize": word_tokenize,
            "transformers": __import__("transformers"),
            "tldextract": tldextract
        }

        local_vars = {"df": df.copy()}

        exec(code_str, global_vars, local_vars)

        return local_vars["df"], code_str

    except Exception as e:
        raise RuntimeError(f"Failed to apply LLM cleaning: {e}")



def call_llm(prompt: str, temperature=0.3, max_tokens=700) -> str:
    client = Together()
    response = client.chat.completions.create(
        model="moonshotai/Kimi-K2-Instruct",
        #model="Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()

# last_manual_cleaning_result = {}

# # ---------------- Core cleaning using Form-node params ----------------
# def manual_cleaning_fn_from_form(df: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, list]:
#     """
#     Apply preprocessing steps driven by n8n Form node output.
#     Returns (cleaned_df, logs)
#     """
#     cleaned = df.copy()
#     logs = []

#     # helpers
#     def get_val(key, default=None):
#         return params.get(key, default)

#     def to_num(value, default=None, cast=float):
#         if value in (None, "", "null"):
#             return default
#         try:
#             return cast(value)
#         except Exception:
#             try:
#                 return cast(str(value).strip())
#             except Exception:
#                 return default

#     # steps selected (array of labels)
#     steps = params.get("Select Preprocessing Step(s)", [])
#     if not isinstance(steps, list):
#         steps = [steps]
#     steps = set(steps)

#     # ---------------- Order: remove NaN-heavy cols/rows -> impute -> outliers -> scale/normalize -> binarize -> OHE ----------------

#     # 1) Remove Columns with Excessive NaNs
#     if "Remove Columns with Excessive NaNs" in steps:
#         thr = to_num(get_val("NaN Threshold for Column Removal"), 0.5, float)
#         if thr is not None:
#             frac = cleaned.isna().mean()
#             to_drop = frac[frac > thr].index.tolist()
#             if to_drop:
#                 cleaned.drop(columns=to_drop, inplace=True)
#             logs.append({"step": "Remove Columns with Excessive NaNs", "threshold": thr, "dropped_columns": to_drop})

#     # 2) Remove Rows with Excessive NaNs
#     if "Remove Rows with Excessive NaNs" in steps:
#         thr = to_num(get_val("NaN Threshold for Row Removal"), 0.5, float)
#         if thr is not None:
#             row_frac = cleaned.isna().mean(axis=1)
#             before = len(cleaned)
#             cleaned = cleaned.loc[row_frac <= thr].reset_index(drop=True)
#             logs.append({"step": "Remove Rows with Excessive NaNs", "threshold": thr, "rows_removed": before - len(cleaned)})

#     # 3) Impute Missing Values
#     if "Impute Missing Values" in steps:
#         strategy = get_val("Impute: strategy", "mean") or "mean"
#         fill_value = to_num(get_val("Impute: fill_value (used when strategy=constant)"), 0, float)

#         num_cols = cleaned.select_dtypes(include=[np.number]).columns
#         cat_cols = cleaned.select_dtypes(exclude=[np.number]).columns

#         if len(num_cols) > 0:
#             if strategy == "constant":
#                 imp_num = SimpleImputer(strategy="constant", fill_value=fill_value)
#             elif strategy in {"mean", "median", "most_frequent"}:
#                 imp_num = SimpleImputer(strategy=strategy)
#             else:
#                 imp_num = SimpleImputer(strategy="mean")
#             cleaned[num_cols] = imp_num.fit_transform(cleaned[num_cols])

#         if len(cat_cols) > 0:
#             strat_cat = "most_frequent" if strategy != "constant" else "constant"
#             imp_cat = SimpleImputer(strategy=strat_cat, fill_value=str(fill_value))
#             cleaned[cat_cols] = imp_cat.fit_transform(cleaned[cat_cols])

#         logs.append({"step": "Impute Missing Values", "strategy": strategy, "fill_value": fill_value})

#     # 4) Remove Outliers (IQR)
#     if "Remove Outliers" in steps:
#         mult = to_num(get_val("Remove Outliers: iqr_multiplier"), 1.5, float)
#         num_cols = cleaned.select_dtypes(include=[np.number]).columns
#         if len(num_cols) > 0 and mult is not None:
#             q1 = cleaned[num_cols].quantile(0.25)
#             q3 = cleaned[num_cols].quantile(0.75)
#             iqr = q3 - q1
#             lower = q1 - mult * iqr
#             upper = q3 + mult * iqr

#             mask = pd.Series(True, index=cleaned.index)
#             for col in num_cols:
#                 # keep NaNs for now; imputation was already done above
#                 col_mask = cleaned[col].between(lower[col], upper[col]) | cleaned[col].isna()
#                 mask &= col_mask

#             before = len(cleaned)
#             cleaned = cleaned.loc[mask].reset_index(drop=True)
#             logs.append({"step": "Remove Outliers", "iqr_multiplier": mult, "rows_removed": before - len(cleaned)})

#     # 5) Scale (MinMax)
#     if "Scale" in steps:
#         min_v = to_num(get_val("Scale: min_value"), 0.0, float)
#         max_v = to_num(get_val("Scale: max_value"), 1.0, float)
#         if max_v is not None and min_v is not None and max_v > min_v:
#             num_cols = cleaned.select_dtypes(include=[np.number]).columns
#             if len(num_cols) > 0:
#                 scaler = MinMaxScaler(feature_range=(min_v, max_v))
#                 cleaned[num_cols] = scaler.fit_transform(cleaned[num_cols])
#             logs.append({"step": "Scale", "min_value": min_v, "max_value": max_v})

#     # 6) Normalize (Standardize)
#     if "Normalize" in steps:
#         num_cols = cleaned.select_dtypes(include=[np.number]).columns
#         if len(num_cols) > 0:
#             scaler = StandardScaler()
#             cleaned[num_cols] = scaler.fit_transform(cleaned[num_cols])
#         logs.append({"step": "Normalize"})

#     # 7) Binarize
#     if "Binarize" in steps:
#         thr = to_num(get_val("Binarize: threshold"), 0.0, float)
#         num_cols = cleaned.select_dtypes(include=[np.number]).columns
#         if len(num_cols) > 0:
#             binarizer = Binarizer(threshold=thr)
#             cleaned[num_cols] = binarizer.fit_transform(cleaned[num_cols])
#             cleaned[num_cols] = cleaned[num_cols].astype(int)
#         logs.append({"step": "Binarize", "threshold": thr})

#     # 8) One-Hot Encoding
#     if "One-Hot Encoding" in steps:
#         cat_cols = cleaned.select_dtypes(include=["object", "category", "bool"]).columns
#         if len(cat_cols) > 0:
#             cleaned = pd.get_dummies(cleaned, columns=list(cat_cols), drop_first=False, dtype=int)
#         logs.append({"step": "One-Hot Encoding"})

#     return cleaned, logs


# # ---------------- FastAPI endpoint ----------------
# @app.post("/manual_cleaning")
# async def manual_cleaning(payload: dict = Body(...)):
#     global last_manual_cleaning_result
#     try:
#         # ✅ Parse incoming JSON
#         csv_text = payload.get("data")
#         params = payload.get("params")

#         if not csv_text or not params:
#             raise HTTPException(400, "Missing 'data' or 'params' in request body.")

#         df = pd.read_csv(BytesIO(csv_text.encode("utf-8")))

#         # ✅ Run your existing cleaning logic
#         cleaned_df, logs = manual_cleaning_fn_from_form(df, params)

#         # ✅ Serialize result
#         cleaned_csv = cleaned_df.to_csv(index=False)
#         last_manual_cleaning_result = {
#             "instruction": params.get("Instruction", ""),
#             "applied_steps": logs,
#             "params": params,
#             "executed_code": "manual_cleaning_fn_from_form(df, params)",
#             "row_count_before": len(df),
#             "row_count_after": len(cleaned_df),
#             "column_count_before": df.shape[1],
#             "column_count_after": cleaned_df.shape[1],
#             "cleaned_csv_preview": cleaned_csv[:1000],
#         }

#         out = BytesIO(cleaned_csv.encode("utf-8"))
#         out.seek(0)
#         return StreamingResponse(
#             out,
#             media_type="text/csv",
#             headers={"Content-Disposition": 'attachment; filename="manual_cleaned.csv"'},
#         )

#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(400, f"Manual cleaning failed: {e}")


# @app.get("/last_manual_cleaning")
# async def get_last_manual_cleaning():
#     if not last_manual_cleaning_result:
#         raise HTTPException(404, "No manual cleaning has been run yet")
#     return JSONResponse(last_manual_cleaning_result)

last_manual_cleaning_result: Dict[str, Any] = {}

def to_num(value, default=None, cast=float):
    if value in (None, "", "null"):
        return default
    try:
        return cast(value)
    except Exception:
        try:
            return cast(str(value).strip())
        except Exception:
            return default

def coerce_value(v):
    if v is None:
        return None
    if isinstance(v, (int, float, bool, list, dict)):
        return v
    if isinstance(v, str):
        s = v.strip()
        if s == "":
            return None
        if s.lower() in ("true", "false"):
            return s.lower() == "true"
        try:
            return float(s) if "." in s else int(s)
        except Exception:
            return s
    return v

def coerce_params(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: coerce_value(v) for k, v in d.items()}

def manual_cleaning_fn_from_form(df: pd.DataFrame, params: dict) -> Tuple[pd.DataFrame, List[dict]]:
    """
    Apply preprocessing steps based on the n8n Form node output.
    Always returns (cleaned_df, logs).
    """
    cleaned = df.copy()
    logs: List[dict] = []

    def get_val(key, default=None):
        return params.get(key, default)

    def to_num(value, default=None, cast=float):
        if value in (None, "", "null"):
            return default
        try:
            return cast(value)
        except Exception:
            try:
                return cast(str(value).strip())
            except Exception:
                return default

    steps = params.get("Select Preprocessing Step(s)", [])
    if not isinstance(steps, list):
        steps = [steps]
    steps = set(steps)

    # 1) Remove Columns with Excessive NaNs
    if "Remove Columns with Excessive NaNs" in steps:
        thr = to_num(get_val("NaN Threshold for Column Removal"), 0.5, float)
        if thr is not None:
            frac = cleaned.isna().mean()
            to_drop = frac[frac > thr].index.tolist()
            if to_drop:
                cleaned.drop(columns=to_drop, inplace=True)
            logs.append({"step": "Remove Columns with Excessive NaNs", "threshold": thr, "dropped_columns": to_drop})

    # 2) Remove Rows with Excessive NaNs
    if "Remove Rows with Excessive NaNs" in steps:
        thr = to_num(get_val("NaN Threshold for Row Removal"), 0.5, float)
        if thr is not None:
            row_frac = cleaned.isna().mean(axis=1)
            before = len(cleaned)
            cleaned = cleaned.loc[row_frac <= thr].reset_index(drop=True)
            logs.append({"step": "Remove Rows with Excessive NaNs", "threshold": thr, "rows_removed": before - len(cleaned)})

    # 3) Impute Missing Values
    if "Impute Missing Values" in steps:
        strategy = get_val("Impute: strategy", "mean") or "mean"
        fill_value = to_num(get_val("Impute: fill_value (used when strategy=constant)"), 0, float)

        num_cols = cleaned.select_dtypes(include=[np.number]).columns
        cat_cols = cleaned.select_dtypes(exclude=[np.number]).columns

        if len(num_cols) > 0:
            if strategy == "constant":
                imp_num = SimpleImputer(strategy="constant", fill_value=fill_value)
            elif strategy in {"mean", "median", "most_frequent"}:
                imp_num = SimpleImputer(strategy=strategy)
            else:
                imp_num = SimpleImputer(strategy="mean")
            cleaned[num_cols] = imp_num.fit_transform(cleaned[num_cols])

        if len(cat_cols) > 0:
            strat_cat = "most_frequent" if strategy != "constant" else "constant"
            imp_cat = SimpleImputer(strategy=strat_cat, fill_value=str(fill_value))
            cleaned[cat_cols] = imp_cat.fit_transform(cleaned[cat_cols])

        logs.append({"step": "Impute Missing Values", "strategy": strategy, "fill_value": fill_value})

    # 4) Remove Outliers (IQR)
    if "Remove Outliers" in steps:
        mult = to_num(get_val("Remove Outliers: iqr_multiplier"), 1.5, float)
        num_cols = cleaned.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0 and mult is not None:
            q1 = cleaned[num_cols].quantile(0.25)
            q3 = cleaned[num_cols].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - mult * iqr
            upper = q3 + mult * iqr

            mask = pd.Series(True, index=cleaned.index)
            for col in num_cols:
                col_mask = cleaned[col].between(lower[col], upper[col]) | cleaned[col].isna()
                mask &= col_mask

            before = len(cleaned)
            cleaned = cleaned.loc[mask].reset_index(drop=True)
            logs.append({"step": "Remove Outliers", "iqr_multiplier": mult, "rows_removed": before - len(cleaned)})

    # 5) Scale (MinMax)
    if "Scale" in steps:
        min_v = to_num(get_val("Scale: min_value"), 0.0, float)
        max_v = to_num(get_val("Scale: max_value"), 1.0, float)
        if max_v is not None and min_v is not None and max_v > min_v:
            num_cols = cleaned.select_dtypes(include=[np.number]).columns
            if len(num_cols) > 0:
                scaler = MinMaxScaler(feature_range=(min_v, max_v))
                cleaned[num_cols] = scaler.fit_transform(cleaned[num_cols])
        logs.append({"step": "Scale", "min_value": min_v, "max_value": max_v})

    # 6) Normalize (Standardize)
    if "Normalize" in steps:
        num_cols = cleaned.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            scaler = StandardScaler()
            cleaned[num_cols] = scaler.fit_transform(cleaned[num_cols])
        logs.append({"step": "Normalize"})

    # 7) Binarize
    if "Binarize" in steps:
        thr = to_num(get_val("Binarize: threshold"), 0.0, float)
        num_cols = cleaned.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            binarizer = Binarizer(threshold=thr)
            cleaned[num_cols] = binarizer.fit_transform(cleaned[num_cols]).astype(int)
        logs.append({"step": "Binarize", "threshold": thr})

    # 8) One-Hot Encoding
    if "One-Hot Encoding" in steps:
        cat_cols = cleaned.select_dtypes(include=["object", "category", "bool"]).columns
        if len(cat_cols) > 0:
            cleaned = pd.get_dummies(cleaned, columns=list(cat_cols), drop_first=False, dtype=int)
        logs.append({"step": "One-Hot Encoding"})

    return cleaned, logs


@app.post("/manual_cleaning")
async def manual_cleaning(payload: Dict[str, Any] = Body(...)):
    global last_manual_cleaning_result
    try:
        csv_text = payload.get("data")
        raw_params = payload.get("params")

        if not isinstance(csv_text, str) or csv_text.strip() == "":
            raise HTTPException(400, "Missing 'data' (CSV text).")
        if not isinstance(raw_params, dict):
            raise HTTPException(400, "Missing or invalid 'params' (object).")

        params = coerce_params(raw_params)

        try:
            df = pd.read_csv(StringIO(csv_text))
        except Exception as e:
            raise HTTPException(400, f"Could not parse CSV: {e}")

        cleaned_df, logs = manual_cleaning_fn_from_form(df, params)

        cleaned_csv = cleaned_df.to_csv(index=False)
        last_manual_cleaning_result = {
            "applied_steps": logs,
            "params": params,
            "row_count_before": int(len(df)),
            "row_count_after": int(len(cleaned_df)),
            "column_count_before": int(df.shape[1]),
            "column_count_after": int(cleaned_df.shape[1]),
            "cleaned_csv_preview": cleaned_csv[:1000],
        }

        out = BytesIO(cleaned_csv.encode("utf-8"))
        out.seek(0)
        return StreamingResponse(
            out,
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="manual_cleaned.csv"'}
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Manual cleaning failed: {e}")

@app.get("/last_manual_cleaning")
async def last_manual_cleaning():
    if not last_manual_cleaning_result:
        raise HTTPException(404, "No manual cleaning has been run yet")
    return JSONResponse(last_manual_cleaning_result)
