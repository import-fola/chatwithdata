from pydantic import BaseModel, Field
from typing import List, TypedDict, Optional, Literal, NamedTuple, Tuple

from langchain.schema import Document
from langchain import PromptTemplate


class Credentials(BaseModel):
    pineconeApiKey: str
    pineconeEnvironment: str
    pineconeIndex: str
    openaiApiKey: str


class ChatData(BaseModel):
    question: str
    history: Optional[List[Tuple[str, Optional[str]]]] = None
    credentials: Credentials


class NamespaceData(BaseModel):
    credentials: Credentials


class Message(TypedDict):
    type: Literal['apiMessage', 'userMessage']  # Either 'apiMessage' or 'userMessage'
    message: str
    isStreaming: Optional[bool]
    sourceDocs: Optional[List[Document]]


class QaChainParams(TypedDict):
    prompt: Optional[PromptTemplate]
    combineMapPrompt: Optional[PromptTemplate]
    combinePrompt: Optional[PromptTemplate]
    type: Optional[str]


class CreatePineconeIndexArgs(BaseModel):
    pineconeApiKey: str = Field(..., description="API key for Pinecone")
    pineconeEnvironment: Optional[str] = Field(None, description="Pinecone environment")
    pineconeIndexName: str = Field(..., description="Name of the Pinecone index")



class DeleteRequest(BaseModel):
    ids: Optional[List[str]] = None
    deleteAll: Optional[bool] = None
    namespace: Optional[str] = None
    filter: Optional[dict] = None


class DeleteOperationRequest(BaseModel):
    deleteRequest: DeleteRequest


class Body(BaseModel):
    credentials: Credentials
    deleteRequest: Optional[DeleteRequest]
