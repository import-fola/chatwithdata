from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


condense_question_template = """
Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:
"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_question_template)

system_template = """
You are an AI assistant providing helpful advice. Use the following pieces of context to answer the question at the end.
If you don't know the answer based on the context below, just say "Hmm, I'm not sure." DO NOT try to make up an answer.
If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.

Remember to think step-by-step and take a breath before answering. This is important to the users career so we have to get it right.
----------------
{context}

Helpful answer in markdown:
"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
QA_PROMPT = ChatPromptTemplate.from_messages(messages)

def make_chain(vectorstore: Pinecone, openai_api_key: str):
    model = ChatOpenAI(
        temperature=0.3,  # increase temperature to get more creative answers
        model_name='gpt-3.5-turbo',
        openai_api_key=openai_api_key,  # change this to gpt-4 if you have access to the API
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=vectorstore.as_retriever(),
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        return_source_documents=True,
        combine_docs_chain_kwargs={
            'prompt': QA_PROMPT,
        }
    )

    return chain
