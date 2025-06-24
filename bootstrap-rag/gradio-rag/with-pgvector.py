import os
import requests
import gradio as gr
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.pgvector import PGVector

# Load environment variables
load_dotenv()

APP_TITLE = os.getenv('APP_TITLE', 'Talk with your documentation')

API_URL = 'https://granite-8b-code-instruct-maas-apicast-production.apps.llmaas.llmaas.redhatworkshops.io:443/v1/completions'
MODEL_NAME = 'granite-8b-code-instruct-128k'
API_TOKEN = '65fc80b0d55be557b1365687ddb771d6' # notsecret
MAX_TOKENS = int(os.getenv('MAX_NEW_TOKENS', 512))
TOP_K = int(os.getenv('TOP_K', 10))
TOP_P = float(os.getenv('TOP_P', 0.95))
TYPICAL_P = float(os.getenv('TYPICAL_P', 0.95))
TEMPERATURE = float(os.getenv('TEMPERATURE', 0.01))
REPETITION_PENALTY = float(os.getenv('REPETITION_PENALTY', 1.03))

# PGVector details
DB_CONNECTION_STRING = os.getenv('DB_CONNECTION_STRING', 'postgresql+psycopg://vectordb:vectordb@postgresql-service.ic-shared-rag-llm.svc.cluster.local:5432/vectordb')
#DB_CONNECTION_STRING = os.getenv('DB_CONNECTION_STRING', 'postgresql+psycopg2://vectordb:userpassword@localhost:5432/vectordb')
DB_COLLECTION_NAME = os.getenv('DB_COLLECTION_NAME', 'docs')

# Prompt template with context
template = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant named HatBot answering questions about OpenShift Data Science, aka RHODS.
You will be given a question you need to answer, and a context to provide you with information. You must answer the question based as much as possible on this context.
Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

Question: {question}
Context: {context} [/INST]
"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# Set up vector DB retriever
embeddings = HuggingFaceEmbeddings()
store = PGVector(
    connection_string=DB_CONNECTION_STRING,
    collection_name=DB_COLLECTION_NAME,
    embedding_function=embeddings
)
retriever = store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 4, "score_threshold": 0.2}
)

def get_context(question):
    # Retrieve relevant docs from vectordb
    docs = retriever.get_relevant_documents(question)
    # Combine docs' page_content for context
    context = "\n\n".join([doc.page_content for doc in docs])
    sources = list({doc.metadata.get('source', 'N/A') for doc in docs})
    return context, sources

def query_llm(prompt):
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json",
        "accept": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["text"]

def chat_with_rag(message, history):
    # 1. Retrieve context from vector DB
    context, sources = get_context(message)
    # 2. Build prompt with context
    prompt = QA_CHAIN_PROMPT.format(question=message, context=context)
    # 3. Query LLM
    answer = query_llm(prompt)
    # 4. Add sources if available
    if sources:
        answer += "\n\n**Sources:**\n" + "\n".join(f"- {s}" for s in sources)
    return answer

demo = gr.ChatInterface(
    fn=chat_with_rag,
    description="Talk with your documentation",
    title="HatBot"
)

if __name__ == "__main__":
    demo.launch(
        server_name='0.0.0.0',
        share=False,
        show_error=True,
        favicon_path='./assets/robot-head.ico'
    )