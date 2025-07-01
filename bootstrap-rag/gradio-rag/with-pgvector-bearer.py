import os
import random
import time
from collections.abc import Generator
from queue import Empty, Queue
from threading import Thread
from typing import Optional

import gradio as gr
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQA
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.callbacks import BaseCallbackHandler

load_dotenv()

APP_TITLE = os.getenv('APP_TITLE', 'Talk with your documentation')

INFERENCE_SERVER_URL = 'https://granite-8b-code-instruct-maas-apicast-production.apps.llmaas.llmaas.redhatworkshops.io:443/v1/completions'
MODEL_NAME = 'granite-7b-instruct'
HUGGINGFACE_API_TOKEN = 'CHANGEME'
MAX_NEW_TOKENS = int(os.getenv('MAX_NEW_TOKENS', 512))
TOP_K = int(os.getenv('TOP_K', 10))
TOP_P = float(os.getenv('TOP_P', 0.95))
TYPICAL_P = float(os.getenv('TYPICAL_P', 0.95))
TEMPERATURE = float(os.getenv('TEMPERATURE', 0.01))
REPETITION_PENALTY = float(os.getenv('REPETITION_PENALTY', 1.03))

# Streaming implementation
class QueueCallback(BaseCallbackHandler):
    """Callback handler for streaming LLM responses to a queue."""

    def __init__(self, q):
        super().__init__()  # Initialize parent class
        self.q = q

    def on_llm_new_token(self, token: str, **kwargs: any) -> None:
        self.q.put(token)

    def on_llm_end(self, *args, **kwargs: any) -> None:
        return self.q.empty()

def stream(input_text) -> Generator:
    # Create a Queue
    job_done = object()

    # Create a function to call - this will run in a thread
    def task():
        # Modified to bypass vector store
        mock_response = "This is a mock response for debugging (PGVector disabled)"
        q.put(mock_response)
        q.put(job_done)

    # Create a thread and start the function
    t = Thread(target=task)
    t.start()

    content = ""

    # Get each new token from the queue and yield for our generator
    while True:
        try:
            next_token = q.get(True, timeout=1)
            if next_token is job_done:
                break
            if isinstance(next_token, str):
                content += next_token
                yield next_token, content
        except Empty:
            continue

# A Queue is needed for Streaming implementation
q = Queue()

# LLM - Fixed implementation
llm_endpoint = HuggingFaceEndpoint(
    endpoint_url=INFERENCE_SERVER_URL,
    huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
    max_new_tokens=MAX_NEW_TOKENS,
    top_k=TOP_K,
    top_p=TOP_P,
    typical_p=TYPICAL_P,
    temperature=TEMPERATURE,
    repetition_penalty=REPETITION_PENALTY,
    streaming=True,
    verbose=False,
)

# Wrap the endpoint with ChatHuggingFace
llm = ChatHuggingFace(
    llm=llm_endpoint,
    callbacks=CallbackManager([QueueCallback(q)]),
)

# Prompt
template = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant named HatBot answering questions about OpenShift Data Science, aka RHODS. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
<</SYS>>

Question: {question}
[/INST]
"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# Modified to work without vector store - Alternative approach
def ask_llm_direct(message, history):
    """Direct LLM call without RetrievalQA"""
    try:
        formatted_prompt = template.format(question=message)
        # Use invoke instead of RetrievalQA
        response = llm.invoke(formatted_prompt)
        
        if hasattr(response, 'content'):
            return response.content
        else:
            return str(response)
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio implementation
def ask_llm(message, history):
    # Use the direct approach instead of RetrievalQA
    response = ask_llm_direct(message, history)
    
    # Simulate streaming for better UX
    for i in range(0, len(response), 10):
        chunk = response[:i+10]
        yield chunk
        time.sleep(0.05)  # Small delay for streaming effect

with gr.Blocks(title="HatBot", css="footer {visibility: hidden}") as demo:
    chatbot = gr.Chatbot(
        show_label=False,
        avatar_images=(None, 'assets/robot-head.svg'),
        render=False
    )
    gr.ChatInterface(
        ask_llm,
        chatbot=chatbot,
        clear_btn=None,
        retry_btn=None,
        undo_btn=None,
        stop_btn=None,
        description=APP_TITLE
    )

if __name__ == "__main__":
    demo.queue().launch(
        server_name='0.0.0.0',
        share=False,
        show_error=True,  # Added for debugging
        favicon_path='./assets/robot-head.ico'
    )