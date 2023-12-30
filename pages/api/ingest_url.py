from fastapi import APIRouter, Form, HTTPException
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.pinecone import Pinecone
from utils.extractTextFromWebsiteUrl import extract_text_from_website_url
from utils.pinecone_client import create_pinecone_index
from config.pinecone import PINECONE_NAME_SPACE
from typing import List
from pydantic import BaseModel

router = APIRouter()

@router.post("/")
async def handler(
    url: str = Form(...), 
    openai_api_key: str = Form(...), 
    pinecone_environment: str = Form(...), 
    pinecone_index: str = Form(...), 
    pinecone_api_key: str = Form(...)
):
    try:
        # Extract text from website URL
        raw_docs = await extract_text_from_website_url(url)
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter({
            'chunkSize': 500,
            'chunkOverlap': 20,
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
        await Pinecone.from_documents(docs, embeddings, {
            'pineconeIndex': index,
            'namespace': PINECONE_NAME_SPACE,
            'textKey': 'text',
        })

    except Exception as e:
        # Catch any exceptions, log them, and re-raise as a 500 error
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during processing")

    return {"rawDocs": raw_docs}
