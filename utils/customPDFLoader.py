from langchain.schema import Document
from PyPDF2 import PdfFileReader
import io

async def custom_pdf_loader(raw, filename=''):
    pdf_reader = PdfFileReader(io.BytesIO(raw))
    page_content = ''
    for page in range(pdf_reader.getNumPages()):
        page_content += pdf_reader.getPage(page).extractText()
    return [Document(page_content=page_content, 
                     metadata={'source': filename, 'pdf_numpages': pdf_reader.getNumPages()})]
