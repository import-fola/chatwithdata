from langchain.schema import Document
from PyPDF2 import PdfReader
import io

def custom_pdf_loader(raw, filename=''):
    pdf_reader = PdfReader(io.BytesIO(raw))
    page_content = ''
    for page in pdf_reader.pages:
        page_content += page.extract_text()
    return [Document(page_content=page_content, 
                     metadata={'source': filename, 'pdf_numpages': len(pdf_reader.pages)})]
