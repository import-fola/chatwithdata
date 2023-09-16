from fastapi import FastAPI, Request, HTTPException
import aiohttp
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import PineconeStore
from schema import ChatData, Credentials
from utils.makechain import make_chain
from utils.pinecone_client import create_pinecone_index
from config.pinecone import PINECONE_NAME_SPACE


app = FastAPI()


@app.post("/chat")
async def chat_handler(chat_data: ChatData):
    sanitized_question = chat_data.question.strip().replace("\n", " ")

    try:
        index = await create_pinecone_index(
            pineconeApiKey=chat_data.credentials.pineconeApiKey,
            pineconeEnvironment=chat_data.credentials.pineconeEnvironment,
            pineconeIndexName=chat_data.credentials.pineconeIndex,
        )

        # create vectorstore
        vector_store = await PineconeStore.from_existing_index(
            OpenAIEmbeddings(
                openAIApiKey=chat_data.credentials.openaiApiKey,
            ),
            {
                "pineconeIndex": index,
                "textKey": "text",
                "namespace": PINECONE_NAME_SPACE,
            },
        )

        # create chain
        chain = make_chain(vector_store, chat_data.credentials.openaiApiKey)

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
