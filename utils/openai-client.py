import os
from langchain.llms import OpenAI

if 'OPENAI_API_KEY' not in os.environ:
    raise ValueError('Missing OpenAI Credentials')

openai = OpenAI(temperature=0)
