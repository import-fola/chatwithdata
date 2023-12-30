from fastapi import FastAPI, Request, HTTPException, APIRouter
import aiohttp
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from schema import ChatData, Credentials
from utils.makechain import make_chain
from utils.pinecone_client import create_pinecone_index, CreatePineconeIndexArgs
from config.pinecone import PINECONE_NAME_SPACE


router = APIRouter()

@router.post("/")
async def chat_handler(chat_data: ChatData):
    sanitized_question = chat_data.question.strip().replace("\n", " ")

    openai_api_key = chat_data.credentials.openaiApiKey 
    pinecone_environment = chat_data.credentials.pineconeEnvironment 
    pinecone_index = chat_data.credentials.pineconeIndex
    pinecone_api_key = chat_data.credentials.pineconeApiKey

    try:
        index = await create_pinecone_index(CreatePineconeIndexArgs(
            pineconeApiKey=pinecone_api_key,
            pineconeEnvironment=pinecone_environment,
            pineconeIndexName=pinecone_index,
        ))

        embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key,
        )

        # create vectorstore
        vector_store = Pinecone.from_existing_index(
            index_name=pinecone_index,
            embedding=embeddings,
        )

        # create chain
        chain = make_chain(vector_store, openai_api_key)

        # ask a question
        response = await chain.call(
            {
                "question": sanitized_question,
                "chat_history": chat_data.history or [],
            }
        )

        return response

    except Exception as error:
        print("error", error)
        raise HTTPException(status_code=500, detail="Something went wrong")
