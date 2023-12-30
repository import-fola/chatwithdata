from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# from fastapi.staticfiles import StaticFiles

from pages.api import chat, delete_namespace, ingest, ingest_url

app = FastAPI()

# Set up CORS middleware options
origins = [
    "http://localhost:3000",  # Local frontend address
    "http://127.0.0.1:3000",  # Local frontend address
    # "https://yourfrontenddomain.com",  # Add your production frontend domain here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Specifies the origins that are permitted to make requests
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


app.include_router(chat.router, prefix="/api/chat")
app.include_router(delete_namespace.router, prefix="/api/delete-namespace")
app.include_router(ingest.router, prefix="/api/ingest")
app.include_router(ingest_url.router, prefix="/api/ingest-url")

# app.mount("/static", StaticFiles(directory="static"), name="static")
