from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from pages.api import chat, delete_namespace, ingest, ingest_url

app = FastAPI()

app.include_router(chat.router, prefix="/api/chat")
app.include_router(delete_namespace.router, prefix="/api/delete-namespace")
app.include_router(ingest.router, prefix="/api/ingest")
app.include_router(ingest_url.router, prefix="/api/ingest-url")

app.mount("/static", StaticFiles(directory="static"), name="static")
