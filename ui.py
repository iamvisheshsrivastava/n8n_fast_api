import streamlit as st
import pandas as pd
import requests
from io import StringIO
from dotenv import load_dotenv
from PIL import Image
from bs4 import BeautifulSoup
from wordcloud import WordCloud
import plotly.express as px      
import seaborn as sns             
import matplotlib.pyplot as plt   
from PIL import Image             
from io import BytesIO          
from together import Together

################################## Functions ##########################################

def render_image_urls(urls):
    st.markdown("### 🖼️ Image Previews")
    for url in urls:
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            st.image(img, caption=url, use_column_width=True)
        except:
            st.warning(f"Could not load image: {url}")

def render_file_links(urls):
    st.markdown("### 📎 File Links")
    for url in urls:
        st.markdown(f"[📥 Download File]({url})", unsafe_allow_html=True)

def render_video_urls(urls):
    st.markdown("### 🎬 Video Previews")
    for url in urls:
        embed_url = None

        if "youtube.com/embed/" in url:
            embed_url = url

        elif "youtube.com/watch?v=" in url:
            video_id = url.split("watch?v=")[-1].split("&")[0]
            embed_url = f"https://www.youtube.com/embed/{video_id}"

        elif "youtu.be/" in url:
            video_id = url.split("youtu.be/")[-1].split("?")[0]
            embed_url = f"https://www.youtube.com/embed/{video_id}"

        elif url.lower().endswith((".mp4", ".webm", ".mov")):
            embed_url = url

        if embed_url:
            st.video(embed_url)
            st.caption(f"[🔗 Watch on YouTube]({url})")
        else:
            st.warning(f"⚠️ Unsupported video format: {url}")

def fetch_and_summarize_url(url: str, instruction: str, llm_func) -> str:
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(separator=' ', strip=True)[:1500]

        full_prompt = (
            f"You are a helpful assistant. The user wants to perform the following task on this web page:\n\n"
            f"URL: {url}\n\n"
            f"Extracted Content:\n{text}\n\n"
            f"Instruction: {instruction}\n\n"
            f"Please analyze or summarize accordingly."
        )

        return llm_func(full_prompt)
    except Exception as e:
        return f"❌ Failed to fetch or process the URL: {e}"



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

def render_column_visualization(df: pd.DataFrame, col: str, inferred_type: str):
    with st.expander(f"Visualization for `{col}` ({inferred_type})", expanded=True):
        try:
            if inferred_type == "categorical":
                top_k = df[col].value_counts().nlargest(10).reset_index()
                top_k.columns = [col, 'count']
                fig = px.bar(top_k, x=col, y='count')
                st.plotly_chart(fig)

            elif inferred_type == "Text":
                text_data = " ".join(df[col].dropna().astype(str)).lower()
                if text_data.strip():
                    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_data)
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.imshow(wordcloud, interpolation="bilinear")
                    ax.axis("off")
                    st.pyplot(fig)
                else:
                    st.warning("Text column is empty or invalid for wordcloud.")

            elif inferred_type == "Numerical":
                fig = px.box(df, y=col)
                st.plotly_chart(fig)
                if df.select_dtypes(include="number").shape[1] > 1:
                    st.markdown("**Correlation Heatmap**")
                    corr = df.select_dtypes(include="number").corr()
                    fig, ax = plt.subplots()
                    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
                    st.pyplot(fig)

            elif inferred_type == "Datetime":
                df[col] = pd.to_datetime(df[col], errors='coerce')
                timeline = df.groupby(df[col].dt.date).size().reset_index(name='Count')
                fig = px.line(timeline, x=col, y='Count')
                st.plotly_chart(fig)

            elif inferred_type == "GPS Coordinates":
                df[['lat', 'lon']] = df[col].str.split(",", expand=True).astype(float)
                st.map(df[['lat', 'lon']])

            elif inferred_type in ["Boolean", "Ordinal", "Percentage", "Currency"]:
                data = df[col].dropna()
                if inferred_type in ["Currency", "Percentage"]:
                    data = data.replace(r'[^\d\.]', '', regex=True).astype(float)
                fig = px.histogram(data, x=col)
                st.plotly_chart(fig)

            elif inferred_type == "Color Code":
                unique_colors = df[col].dropna().unique()[:20]
                st.markdown("**Color Swatches**")
                st.write("".join([
                    f"<div style='display:inline-block;width:50px;height:20px;background-color:{color};margin:2px;border:1px solid #000'></div>"
                    for color in unique_colors]), unsafe_allow_html=True)

            elif inferred_type in ["Email Address", "Phone Number"]:
                top_vals = df[col].value_counts().nlargest(10).reset_index()
                top_vals.columns = [col, 'count']
                fig = px.bar(top_vals, x=col, y='count')
                st.plotly_chart(fig)

            elif inferred_type == "Identifier / ID":
                st.info("Column identified as unique ID – typically not useful for visualization.")

            elif inferred_type in ["Null-heavy", "Constant / Low Variance"]:
                st.warning(f"Column classified as `{inferred_type}` – not suitable for meaningful visualization.")

            elif inferred_type in ["Image URL", "Video URL", "Document URL", "General URL", "File Path"]:
                urls = df[col].dropna().unique().tolist()

                if inferred_type == "Image URL":
                    with st.container():
                        st.markdown("<div style='max-height:400px; overflow-y:auto;'>", unsafe_allow_html=True)
                        render_image_urls(urls)
                        st.markdown("</div>", unsafe_allow_html=True)

                elif inferred_type == "Video URL":
                    urls = df[col].dropna().unique().tolist()
                    render_video_urls(urls)

                elif inferred_type == "General URL":
                    st.markdown("### 🌐 General URL Processor")

                    selected_url = st.selectbox("Select a URL to summarize", urls, key=f"{col}_url_select")
                    user_prompt = st.text_area("💬 What would you like to know or extract?", placeholder="e.g., Summarize the article, extract key points...", key=f"{col}_prompt_input")

                    if st.button("Analyze Webpage", key=f"{col}_analyze_button"):
                        if selected_url and user_prompt.strip():
                            with st.spinner("⏳ Fetching and analyzing..."):
                                result = fetch_and_summarize_url(selected_url, user_prompt, call_llm)

                            st.success("Done")
                            st.markdown("### Result:")
                            st.write(result)
                        else:
                            st.warning("⚠️ Please select a URL and provide an instruction.")


                elif inferred_type in ["Document URL", "File Path"]:
                    with st.container():
                        st.markdown("<div style='max-height:400px; overflow-y:auto;'>", unsafe_allow_html=True)
                        render_file_links(urls)
                        st.markdown("</div>", unsafe_allow_html=True)

                else:
                    st.write(urls) 

        except Exception as e:
            st.error(str(f"Error visualizing `{col}`: {str(e)}"))

##########################################################################################

load_dotenv()
st.set_page_config(page_title="CSV Workflow Dashboard", layout="wide")
st.title("📊 CSV Workflow Dashboard")

page = st.sidebar.radio("Choose a view", ["Column Inference", "LLM Cleaning", "Visualization", "Manual Cleaning"])

BASE_URL = "http://127.0.0.1:8000"  

# ---------------- Column Inference ----------------
if page == "Column Inference":
    st.subheader("🔍 Inferred Column Types")

    if st.button("Load Last Inference"):
        response = requests.get(f"{BASE_URL}/last_inference")
        if response.status_code == 200:
            result = response.json()
            df = pd.DataFrame(result["columns"].items(), columns=["Column", "Inferred Type"])
            st.table(df)
            st.success(f"✅ Total Rows: {result['rows']}")
        else:
            st.error(f"❌ API Error: {response.text}")


# ---------------- LLM Cleaning ----------------
elif page == "LLM Cleaning":
    st.subheader("🧹 LLM-Powered Data Cleaning")

    if st.button("Load Last Cleaning"):
        response = requests.get(f"{BASE_URL}/last_cleaning")
        if response.status_code == 200:
            result = response.json()
            cleaned_csv = result["cleaned_csv"]
            cleaned_df = pd.read_csv(StringIO(cleaned_csv))

            st.success("✅ Last Cleaning Result Loaded")
            st.write(f"**Instruction:** {result['instruction']}")
            st.code(result["executed_code"], language="python")
            st.dataframe(cleaned_df)

            st.download_button(
                label="⬇️ Download Last Cleaned CSV",
                data=cleaned_csv,
                file_name="last_cleaned.csv",
                mime="text/csv"
            )
        else:
            st.error(f"❌ API Error: {response.text}")

elif page == "Visualization":
    st.subheader("📊 Column Visualizations")

    clean_resp = requests.get(f"{BASE_URL}/last_cleaning")
    infer_resp = requests.get(f"{BASE_URL}/last_inference")

    if clean_resp.status_code == 200 and infer_resp.status_code == 200:
        clean_data = clean_resp.json()
        infer_data = infer_resp.json()

        cleaned_csv = clean_data["cleaned_csv"]
        df = pd.read_csv(StringIO(cleaned_csv))

        col_types = infer_data["columns"]

        selected_col = st.selectbox("Select a column", list(col_types.keys()))
        inferred_type = col_types[selected_col]

        inferred_type_norm = inferred_type.strip().lower()

        render_column_visualization(df, selected_col, inferred_type_norm)

    else:
        st.error("❌ No cached inference/cleaning available. Run those steps first.")

# ---------------- Manual Cleaning ----------------
elif page == "Manual Cleaning":
    st.subheader("🧹 Manual Data Cleaning")

    if st.button("Load Last Manual Cleaning"):
        response = requests.get(f"{BASE_URL}/last_manual_cleaning")
        if response.status_code == 200:
            result = response.json()
            cleaned_csv = result["cleaned_csv"]
            cleaned_df = pd.read_csv(StringIO(cleaned_csv))

            st.success("✅ Last Manual Cleaning Result Loaded")
            st.write(f"**Instruction:** {result['instruction']}")
            st.code(result["executed_code"], language="python")
            st.dataframe(cleaned_df)

            st.download_button(
                label="⬇️ Download Last Manually Cleaned CSV",
                data=cleaned_csv,
                file_name="last_manual_cleaned.csv",
                mime="text/csv"
            )
        else:
            st.error(f"❌ API Error: {response.text}")


