from langchain.llms import OpenAI
from langchain.vectorstores import Pinecone
from langchain.chains import ConversationalRetrievalChain

CONDENSE_PROMPT = """
Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:
"""

# change to your own 'system' prompt
QA_PROMPT = """
You are an AI assistant providing helpful advice. Use the following pieces of context to answer the question at the end.
If you don't know the answer based on the context below, just say "Hmm, I'm not sure." DO NOT try to make up an answer.
If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.

{context}

Question: {question}
Helpful answer in markdown:
"""

def make_chain(vectorstore: Pinecone, openai_api_key: str):
    model = OpenAI(
        temperature=0,  # increase temperature to get more creative answers
        model_name='gpt-3.5-turbo',
        openai_api_key=openai_api_key,  # change this to gpt-4 if you have access to the API
    )

    chain = ConversationalRetrievalChain.from_llm(
        model,
        vectorstore.as_retriever(),
        condense_question_prompt=CONDENSE_PROMPT,
        combine_docs_chain_kwargs={
            'qaTemplate': QA_PROMPT,
            'returnSourceDocuments': True
        }
    )

    return chain