import asyncio
from langchain.schema import Document
from customPDFLoader import customPDFLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import Docx2txtLoader 
from langchain.document_loaders import JSONLoader
from langchain.document_loaders import CSVLoader

async def extract_text_from_file(file_data, filetype, filename):
    print('extractTextFromFile', filetype, filename)

    if not filetype:
        raise ValueError('Unknown file type')

    text = ""
    metadata = {}
    result = []

    if filetype == 'application/pdf':
        pdf_docs = await customPDFLoader(file_data, filename)
        result = pdf_docs
    elif filetype == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        docx_loader = Docx2txtLoader(file_data)
        docx_doc = await docx_loader.load()
        result = docx_doc
    elif filetype == 'application/json':
        json_loader = JSONLoader(file_data)
        json_doc = await json_loader.load()
        result = json_doc
    elif filetype in ['text/markdown', 'text/csv']:
        csv_loader = CSVLoader(file_data)
        csv_doc = await csv_loader.load()
        result = csv_doc
    elif filetype == 'text/plain':
        text_loader = TextLoader(file_data)
        text_doc = await text_loader.load()
        result = text_doc
    else:
        raise ValueError(f'Unsupported file type: {filetype}')

    return result
