from fastapi import APIRouter, UploadFile, Form, HTTPException
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.pinecone import PineconeStore
from utils.extractTextFromFiles import extract_text_from_file
from utils.pinecone_client import create_pinecone_index
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
    file: UploadFile = Form(...), 
    openai_api_key: str = Form(...), 
    pinecone_environment: str = Form(...), 
    pinecone_index: str = Form(...), 
    pinecone_api_key: str = Form(...)
):
    try:
        # Check file size
        if await file.length() > 1e6: # adjust size limit as needed
            raise HTTPException(status_code=400, detail="File size exceeds limit")
        
        # Extract text from file
        raw_docs = await extract_text_from_file({
            'fileData': await file.read(),
            'filetype': file.content_type,
            'filename': file.filename
        })

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter({
            'chunkSize': 1000,
            'chunkOverlap': 200,
        })
        docs = await text_splitter.split_documents(raw_docs)
        
        # Create embeddings
        embeddings = OpenAIEmbeddings({
            'openAIApiKey': openai_api_key,
        })
        
        # Create Pinecone index
        index = await create_pinecone_index({
            'pineconeApiKey': pinecone_api_key,
            'pineconeEnvironment': pinecone_environment,
            'pineconeIndexName': pinecone_index,
        })
        
        # Store documents in Pinecone
        await PineconeStore.from_documents(docs, embeddings, {
            'pineconeIndex': index,
            'namespace': PINECONE_NAME_SPACE,
            'textKey': 'text',
        })

    except Exception as e:
        # Catch any exceptions, log them, and re-raise as a 500 error
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during processing")

    return {"message": "successfully ingested"}
