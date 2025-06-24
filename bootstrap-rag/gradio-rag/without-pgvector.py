import os
import requests
import gradio as gr
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.language_models.llms import LLM  # Adjust import if needed for your LangChain version
from typing import Optional, List, Mapping, Any

# Load environment variables
load_dotenv()

APP_TITLE = os.getenv('APP_TITLE', 'Talk with your documentation')

API_URL = 'https://granite-8b-code-instruct-maas-apicast-production.apps.llmaas.llmaas.redhatworkshops.io:443/v1/completions'
MODEL_NAME = 'granite-8b-code-instruct-128k'
API_TOKEN = '65fc80b0d55be557b1365687ddb771d6'
MAX_TOKENS = int(os.getenv('MAX_NEW_TOKENS', 512))
TOP_K = int(os.getenv('TOP_K', 10))
TOP_P = float(os.getenv('TOP_P', 0.95))
TYPICAL_P = float(os.getenv('TYPICAL_P', 0.95))
TEMPERATURE = float(os.getenv('TEMPERATURE', 0.01))
REPETITION_PENALTY = float(os.getenv('REPETITION_PENALTY', 1.03))

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

def chat_interface(message, history):
    answer = query_llm(message)
    return answer

demo = gr.ChatInterface(
    fn=chat_interface,
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