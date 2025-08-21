from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/")
def home():
    return HTMLResponse("""
    <html>
    <head>
        <style>
            body {
                margin: 0;
                font-family: Arial, sans-serif;
            }
            .sidebar {
                position: fixed;
                top: 0;
                left: 0;
                width: 200px;
                height: 100%;
                background: #f4f4f4;
                padding: 20px;
                box-shadow: 2px 0 5px rgba(0,0,0,0.1);
            }
            .sidebar h2 {
                font-size: 18px;
            }
            .sidebar a {
                display: block;
                margin: 10px 0;
                color: #333;
                text-decoration: none;
            }
            .sidebar a:hover {
                text-decoration: underline;
            }
            .content {
                margin-left: 220px; /* leave space for sidebar */
                padding: 10px;
                height: 100vh;
            }
            iframe {
                width: 100%;
                height: 100%;
                border: none;
            }
        </style>
        <script>
            function loadPage(page) {
                document.getElementById('contentFrame').src = page;
            }
        </script>
    </head>
    <body>
        <div class="sidebar">
            <h2>My Application</h2>
            <a href="#" onclick="loadPage('http://localhost:5678')">Workflows (n8n)</a>
            <a href="#" onclick="loadPage('http://localhost:8501')">Visualize (Streamlit)</a>
        </div>
        <div class="content">
            <iframe id="contentFrame" src=""></iframe>
        </div>
    </body>
    </html>
    """)
