import os
import time
from collections.abc import Generator
from queue import Empty, Queue
from threading import Thread
from typing import Optional

import gradio as gr
from dotenv import load_dotenv
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import requests
import json

load_dotenv()

APP_TITLE = os.getenv('APP_TITLE', 'Talk with your documentation')

INFERENCE_SERVER_URL = 'https://granite-8b-code-instruct-maas-apicast-production.apps.llmaas.llmaas.redhatworkshops.io:443/v1/completions'
MODEL_NAME = 'granite-7b-instruct'
HUGGINGFACE_API_TOKEN = os.getenv('HUGGINGFACE_API_TOKEN', 'CHANGEME')
MAX_NEW_TOKENS = int(os.getenv('MAX_NEW_TOKENS', 512))
TOP_K = int(os.getenv('TOP_K', 10))
TOP_P = float(os.getenv('TOP_P', 0.95))
TYPICAL_P = float(os.getenv('TYPICAL_P', 0.95))
TEMPERATURE = float(os.getenv('TEMPERATURE', 0.01))
REPETITION_PENALTY = float(os.getenv('REPETITION_PENALTY', 1.03))

# Custom LLM class that implements Runnable
class CustomHuggingFaceLLM:
    """Custom LLM implementation for HuggingFace endpoints"""
    
    def __init__(self, endpoint_url, api_token, **kwargs):
        self.endpoint_url = endpoint_url
        self.api_token = api_token
        self.max_new_tokens = kwargs.get('max_new_tokens', 512)
        self.temperature = kwargs.get('temperature', 0.01)
        self.top_p = kwargs.get('top_p', 0.95)
        self.repetition_penalty = kwargs.get('repetition_penalty', 1.03)
        
    def invoke(self, prompt):
        """Invoke the LLM with a prompt"""
        try:
            headers = {
                'Authorization': f'Bearer {self.api_token}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'prompt': prompt,
                'max_tokens': self.max_new_tokens,
                'temperature': self.temperature,
                'top_p': self.top_p,
                'repetition_penalty': self.repetition_penalty,
                'stop': ['</s>', '[/INST]']
            }
            
            response = requests.post(
                self.endpoint_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    return result['choices'][0].get('text', '').strip()
                else:
                    return "No response generated"
            else:
                return f"API Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Error calling LLM: {str(e)}"
    
    def __call__(self, prompt):
        return self.invoke(prompt)

# Initialize LLM
try:
    llm = CustomHuggingFaceLLM(
        endpoint_url=INFERENCE_SERVER_URL,
        api_token=HUGGINGFACE_API_TOKEN,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY
    )
    print("LLM initialized successfully")
except Exception as e:
    print(f"Error initializing LLM: {e}")
    
    # Fallback mock LLM
    class MockLLM:
        def invoke(self, prompt):
            return f"Mock response to your question. (Original LLM failed to initialize: {str(e)})"
        
        def __call__(self, prompt):
            return self.invoke(prompt)
    
    llm = MockLLM()

# Prompt template
template = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant named HatBot answering questions about OpenShift Data Science, aka RHODS. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
<</SYS>>

Question: {question}
[/INST]
"""

prompt = PromptTemplate.from_template(template)

# Create a modern chain using RunnableSequence (new LangChain pattern)
def format_prompt(question):
    return template.format(question=question)

def generate_response(question):
    """Generate response using the modern Runnable pattern"""
    try:
        formatted_prompt = format_prompt(question)
        response = llm.invoke(formatted_prompt)
        return response
    except Exception as e:
        return f"Error generating response: {str(e)}"

def stream_response(message):
    """Generate streaming response for better UX"""
    try:
        response = generate_response(message)
        
        # Simulate streaming by yielding chunks
        words = response.split()
        current_response = ""
        
        for word in words:
            current_response += word + " "
            yield current_response.strip()
            time.sleep(0.05)  # Small delay for streaming effect
            
    except Exception as e:
        yield f"Error: {str(e)}"

def chat_fn(message, history):
    """Chat function for Gradio ChatInterface"""
    # Convert history to the expected format if needed
    for response_chunk in stream_response(message):
        yield response_chunk

# Modern Gradio interface with updated parameters
with gr.Blocks(title="HatBot", css="footer {visibility: hidden}") as demo:
    # Updated Chatbot with proper type parameter
    chatbot = gr.Chatbot(
        show_label=False,
        type="messages",  # Updated to use modern message format
        avatar_images=(None, 'assets/robot-head.svg') if os.path.exists('assets/robot-head.svg') else None,
    )
    
    # Updated ChatInterface without deprecated parameters
    chat_interface = gr.ChatInterface(
        fn=chat_fn,
        chatbot=chatbot,
        description=APP_TITLE,
        examples=[
            "What is OpenShift Data Science?",
            "How do I deploy a model in RHODS?",
            "What are the key features of OpenShift Data Science?"
        ],
        title="HatBot - OpenShift Data Science Assistant"
    )

if __name__ == "__main__":
    print("Starting HatBot...")
    print(f"Using API endpoint: {INFERENCE_SERVER_URL}")
    print(f"API Token configured: {'Yes' if HUGGINGFACE_API_TOKEN != 'CHANGEME' else 'No - Please set HUGGINGFACE_API_TOKEN'}")
    
    demo.queue().launch(
        server_name='0.0.0.0',
        share=False,
        show_error=True,
        favicon_path='./assets/robot-head.ico' if os.path.exists('./assets/robot-head.ico') else None,
        inbrowser=False,
        server_port=7860
    )