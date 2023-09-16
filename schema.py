from pydantic import BaseModel
from typing import List, TypedDict, Optional, Literal, NamedTuple

from langchain.schema import Document
from langchain import PromptTemplate


class Credentials(BaseModel):
    pineconeApiKey: str
    pineconeEnvironment: str
    pineconeIndex: str
    openaiApiKey: str


class ChatData(BaseModel):
    question: str
    history: Optional[List[str]] = None
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


class CreatePineconeIndexArgs(NamedTuple):
    pineconeApiKey: str
    pineconeEnvironment: Optional[str]
    pineconeIndexName: str


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
