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
from langchain.chains import LLMChain
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceEndpoint
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
        super().__init__()
        self.q = q

    def on_llm_new_token(self, token: str, **kwargs: any) -> None:
        self.q.put(token)

    def on_llm_end(self, *args, **kwargs: any) -> None:
        self.q.put(None)  # Signal end of stream

# A Queue is needed for Streaming implementation
q = Queue()

# LLM using langchain_community
try:
    llm = HuggingFaceEndpoint(
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
        callbacks=CallbackManager([QueueCallback(q)]),
    )
except Exception as e:
    print(f"Error initializing HuggingFaceEndpoint: {e}")
    # Fallback to a simple mock LLM for testing
    class MockLLM:
        def __init__(self):
            self.callbacks = None
            
        def invoke(self, prompt):
            return "This is a mock response for testing purposes."
            
        def __call__(self, prompt):
            return self.invoke(prompt)
    
    llm = MockLLM()

# Prompt
template = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant named HatBot answering questions about OpenShift Data Science, aka RHODS. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
<</SYS>>

Question: {question}
[/INST]
"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# Create LLMChain instead of RetrievalQA
try:
    qa_chain = LLMChain(
        llm=llm,
        prompt=QA_CHAIN_PROMPT,
        verbose=False
    )
except Exception as e:
    print(f"Error creating LLMChain: {e}")
    # Fallback function
    def qa_chain_run(question):
        return f"Mock response to: {question}"
    
    class MockChain:
        def run(self, question):
            return qa_chain_run(question)
    
    qa_chain = MockChain()

def stream_response(input_text) -> Generator:
    """Generate streaming response"""
    try:
        if hasattr(qa_chain, 'run'):
            response = qa_chain.run(input_text)
        else:
            response = qa_chain(input_text)
        
        # Simulate streaming
        for i in range(0, len(response), 5):
            chunk = response[:i+5]
            yield chunk, chunk
            time.sleep(0.02)
    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        yield error_msg, error_msg

# Gradio implementation
def ask_llm(message, history):
    for chunk, full_content in stream_response(message):
        yield full_content

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
        show_error=True,
        favicon_path='./assets/robot-head.ico'
    )