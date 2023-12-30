from fastapi import APIRouter, UploadFile, Form, HTTPException
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from utils.extractTextFromFiles import extract_text_from_file
from utils.pinecone_client import create_pinecone_index, CreatePineconeIndexArgs
from config.pinecone import PINECONE_NAME_SPACE
from typing import List
from pydantic import BaseModel

router = APIRouter()

class FileUpload(BaseModel):
    file: List[bytes] = None
    filename: str
    filetype: str

@router.post("/")
async def handler(
    files: UploadFile = Form(...), 
    openai_api_key: str = Form(...), 
    pinecone_environment: str = Form(...), 
    pinecone_index: str = Form(...), 
    pinecone_api_key: str = Form(...)
):
    # Check file size
    if files.size > 10e6:  # 10 MB size limit
        raise HTTPException(status_code=400, detail="File size exceeds limit")
    
    try:
        # Extract text from file
        raw_docs = extract_text_from_file(
            file_data = await files.read(),
            filetype = files.content_type,
            filename = files.filename
        )
        # print("raw_docs run successfully")
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
        )
        docs = text_splitter.split_documents(raw_docs)
        # print("text_splitter run successfully")

        # Create embeddings
        embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key,
        )
        # print("embeddings created")

        # Create Pinecone index
        index = await create_pinecone_index(CreatePineconeIndexArgs(
            pineconeApiKey=pinecone_api_key,
            pineconeEnvironment=pinecone_environment,
            pineconeIndexName=pinecone_index,
        ))
        # print("Pinecone index created")

        # Store documents in Pinecone
        Pinecone.from_documents(docs, embeddings, index_name=pinecone_index)
        # print("Docs stored in Pinecone index")

    except Exception as e:
        # Catch any exceptions, log them, and re-raise as a 500 error
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during processing")

    return {"message": "successfully ingested"}
