import os

# if 'PINECONE_INDEX_NAME' not in os.environ:
#     raise ValueError('Missing Pinecone index name in .env file')

PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', '')

# your "ingestion" will use this namespace
PINECONE_NAME_SPACE = 'pdf-starter' #namespace is optional for your vectors
