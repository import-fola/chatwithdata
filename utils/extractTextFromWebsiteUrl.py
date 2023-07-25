from langchain.schema import Document
from langchain.document_loaders import WebBaseLoader

async def extract_text_from_website_url(url):
    loader = WebBaseLoader(url)
    docs = await loader.load()
    return docs
