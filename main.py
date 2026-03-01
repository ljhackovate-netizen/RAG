import os
import logging
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

from api.routes import clients, ingest, generate, drive
from core.vector_store import vector_store_manager

app = FastAPI(title="CGN Client Brain", version="2.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.include_router(generate.router)
app.include_router(ingest.router)
app.include_router(drive.router)
app.include_router(clients.router)

if os.getenv("ENABLE_QA", "false").lower() == "true":
    from api.routes import qa
    app.include_router(qa.router)
    logging.getLogger(__name__).info("Q&A feature ENABLED")


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    client_ids = vector_store_manager.list_clients()
    client_data = [
        {"id": c, "stats": vector_store_manager.stats(c)}
        for c in client_ids
    ]
    return templates.TemplateResponse("index.html", {
        "request": request,
        "clients": client_data,
    })


@app.get("/ingest/", response_class=HTMLResponse)
async def ingest_ui(request: Request):
    return templates.TemplateResponse("ingest.html", {"request": request})


@app.get("/health")
async def health():
    return {"status": "ok", "service": "CGN Client Brain"}
