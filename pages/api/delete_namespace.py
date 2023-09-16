from fastapi import APIRouter, HTTPException
from schema import DeleteRequest, DeleteOperationRequest, Body, Credentials
from utils.pinecone_client import create_pinecone_index
from config.pinecone import PINECONE_NAME_SPACE

router = APIRouter()

@router.post("/")
async def handler(body: Body):
    try:
        # Initialize Pinecone using the utility function
        index = await create_pinecone_index({
            'pineconeApiKey': body.credentials.pineconeApiKey,
            'pineconeEnvironment': body.credentials.pineconeEnvironment,
            'pineconeIndexName': body.credentials.pineconeIndex,
        })
        
        # Prepare data for deletion
        delete_data = body.deleteRequest

        # Perform the deletion using Pinecone
        delete_response = index.delete(
            ids=delete_data.ids,
            delete_all=delete_data.deleteAll,
            namespace=delete_data.namespace or PINECONE_NAME_SPACE,
            filter=delete_data.filter
        )
        
        return {"message": "delete successful", "response": delete_response}
    except Exception as e:
        print(f"error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
