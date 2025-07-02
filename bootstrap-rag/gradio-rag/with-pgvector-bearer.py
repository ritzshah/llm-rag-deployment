import os
import random
import time
import requests
import json
from collections.abc import Generator
from queue import Empty, Queue
from threading import Thread
from typing import Optional, Any, List

import gradio as gr
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQA
# Updated imports for stable LangChain versions
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.schema import LLMResult, Generation
from langchain.vectorstores.pgvector import PGVector

load_dotenv()

# Parameters
APP_TITLE = os.getenv('APP_TITLE', 'Talk with your documentation')

INFERENCE_SERVER_URL = os.getenv('INFERENCE_SERVER_URL', 'https://granite-8b-code-instruct-maas-apicast-production.apps.llmaas.llmaas.redhatworkshops.io:443/v1/completions')
MODEL_NAME = os.getenv('MODEL_NAME', 'granite-8b-code-instruct-128k')
MAX_NEW_TOKENS = int(os.getenv('MAX_NEW_TOKENS', 512))
TOP_K = int(os.getenv('TOP_K', 10))
TOP_P = float(os.getenv('TOP_P', 0.95))
TYPICAL_P = float(os.getenv('TYPICAL_P', 0.95))
TEMPERATURE = float(os.getenv('TEMPERATURE', 0.01))
REPETITION_PENALTY = float(os.getenv('REPETITION_PENALTY', 1.03))

# Add Bearer Token parameter
BEARER_TOKEN = os.getenv('BEARER_TOKEN')

DB_CONNECTION_STRING = os.getenv('DB_CONNECTION_STRING', 'postgresql+psycopg://vectordb:vectordb@localhost:5432/vectordb')
DB_COLLECTION_NAME = os.getenv('DB_COLLECTION_NAME', 'langchain_pg_collection')

# Add debug flag
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'

# Validate required environment variables
if not DB_CONNECTION_STRING:
    raise ValueError("DB_CONNECTION_STRING environment variable is required")
if not DB_COLLECTION_NAME:
    raise ValueError("DB_COLLECTION_NAME environment variable is required")
if not BEARER_TOKEN:
    raise ValueError("BEARER_TOKEN environment variable is required")
if not INFERENCE_SERVER_URL:
    raise ValueError("INFERENCE_SERVER_URL environment variable is required")
if not MODEL_NAME:
    raise ValueError("MODEL_NAME environment variable is required")

# Streaming implementation
class QueueCallback(BaseCallbackHandler):
    """Callback handler for streaming LLM responses to a queue."""

    def __init__(self, q):
        self.q = q

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.q.put(token)

    def on_llm_end(self, *args, **kwargs: Any) -> None:
        return self.q.empty()

def remove_source_duplicates(input_list):
    unique_list = []
    for item in input_list:
        if item.metadata['source'] not in unique_list:
            unique_list.append(item.metadata['source'])
    return unique_list

def answer_general_question(question):
    """Handle general questions without RAG context"""
    general_template = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Answer the question to the best of your ability using your general knowledge.
Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If you don't know the answer to a question, please say so rather than making something up.
<</SYS>>

Question: {question} [/INST]
"""
    
    prompt = general_template.format(question=question)
    try:
        if DEBUG:
            print(f"DEBUG: Calling general question with prompt length: {len(prompt)}")
        response = llm._call(prompt)
        if DEBUG:
            print(f"DEBUG: General question response length: {len(response) if response else 0}")
        return response
    except Exception as e:
        print(f"ERROR in answer_general_question: {str(e)}")
        return f"I apologize, but I encountered an error while trying to answer your question: {str(e)}"

def stream(input_text) -> Generator:
    # Create a Queue
    job_done = object()

    # Create a function to call - this will run in a thread
    def task():
        try:
            if DEBUG:
                print(f"DEBUG: Processing query: {input_text}")
            
            resp = qa_chain({"query": input_text})
            
            if DEBUG:
                print(f"DEBUG: QA chain response keys: {resp.keys() if resp else 'None'}")
                if 'result' in resp:
                    print(f"DEBUG: Result length: {len(resp['result']) if resp['result'] else 0}")
                    print(f"DEBUG: Result preview: {resp['result'][:200] if resp['result'] else 'Empty result'}")
                if 'source_documents' in resp:
                    print(f"DEBUG: Found {len(resp['source_documents'])} source documents")
            
            # First, put the actual answer from the LLM
            if 'result' in resp and resp['result']:
                # Split the response into tokens for streaming effect
                answer = resp['result'].strip()  # Remove any leading/trailing whitespace
                if answer:  # Only proceed if we have actual content
                    words = answer.split()
                    for word in words:
                        q.put(word + " ")
                        time.sleep(0.05)  # Small delay for streaming effect
                else:
                    if DEBUG:
                        print("DEBUG: Result field exists but is empty after stripping")
                    q.put("I found relevant information but couldn't generate a response. ")
            else:
                if DEBUG:
                    print("DEBUG: No result field or result is empty")
                q.put("I found relevant sources but couldn't generate a detailed response. ")
            
            # Then add sources if available
            if 'source_documents' in resp and resp['source_documents']:
                sources = remove_source_duplicates(resp['source_documents'])
                if len(sources) != 0:
                    q.put("\n\n*Sources:* \n")
                    for source in sources:
                        q.put("* " + str(source) + "\n")
                else:
                    q.put("\n\n*Note: No specific sources found for this query.*\n")
            else:
                q.put("\n\n*Note: No specific sources found for this query.*\n")
                
        except Exception as e:
            if DEBUG:
                print(f"DEBUG: Exception in task: {str(e)}")
            error_msg = str(e)
            if "No relevant docs were retrieved" in error_msg:
                # If no docs retrieved, try to answer without RAG context
                try:
                    if DEBUG:
                        print("DEBUG: Attempting general question fallback")
                    general_response = answer_general_question(input_text)
                    if general_response:
                        words = general_response.split()
                        for word in words:
                            q.put(word + " ")
                            time.sleep(0.05)
                    q.put("\n\n*Note: I couldn't find specific documentation for this query, so I answered based on my general knowledge.*\n")
                except Exception as inner_e:
                    if DEBUG:
                        print(f"DEBUG: Exception in general question fallback: {str(inner_e)}")
                    q.put(f"\nError generating response: {str(inner_e)}\n")
            else:
                q.put(f"\nError: {error_msg}\n")
        finally:
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
                return  # Use return instead of break to properly end generator
            if isinstance(next_token, str):
                content += next_token
                yield next_token, content
        except Empty:
            continue

# A Queue is needed for Streaming implementation
q = Queue()

############################
# LLM chain implementation #
############################

# Document store: Use default HuggingFace embeddings (384 dimensions)
# Explicitly specify the default model to avoid deprecation warning
embeddings = HuggingFaceEmbeddings()

# Updated PGVector from langchain.vectorstores with error handling
try:
    store = PGVector(
        connection_string=DB_CONNECTION_STRING,
        collection_name=DB_COLLECTION_NAME,
        embedding_function=embeddings,
    )
    # Test the connection
    _ = store.similarity_search("test", k=1)
    print("✓ Database connection successful")
except Exception as e:
    print(f"✗ Database connection failed: {str(e)}")
    raise

# Custom LLM class for Bearer Token authentication
class CustomEndpointLLM(BaseLLM):
    endpoint_url: str
    api_key: str
    model_name: str
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    callbacks: Optional[List] = None
    streaming: bool = True

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None, **kwargs):
        """Generate completions for multiple prompts."""
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop=stop, **kwargs)
            generations.append([Generation(text=text)])
        
        return LLMResult(generations=generations)

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': self.model_name,
            'prompt': prompt,
            'max_tokens': self.max_new_tokens,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'stream': False  # For non-streaming calls
        }
        
        if stop:
            data['stop'] = stop
            
        try:
            if DEBUG:
                print(f"DEBUG: Making LLM request to {self.endpoint_url}")
                print(f"DEBUG: Request data keys: {data.keys()}")
                print(f"DEBUG: Prompt length: {len(prompt)}")
                
            response = requests.post(self.endpoint_url, headers=headers, json=data, timeout=60)  # Increase timeout
            
            if DEBUG:
                print(f"DEBUG: Response status code: {response.status_code}")
                print(f"DEBUG: Response headers: {dict(response.headers)}")
            
            response.raise_for_status()
            
            result = response.json()
            
            if DEBUG:
                print(f"DEBUG: Response JSON keys: {result.keys() if result else 'None'}")
            
            if 'choices' in result and len(result['choices']) > 0:
                generated_text = result['choices'][0].get('text', '')
                if DEBUG:
                    print(f"DEBUG: Generated text length: {len(generated_text)}")
                    print(f"DEBUG: Generated text preview: {generated_text[:200] if generated_text else 'Empty'}")
                return generated_text
            else:
                if DEBUG:
                    print("DEBUG: No choices in response or empty choices")
                    print(f"DEBUG: Full response: {result}")
                return ""
        except requests.exceptions.RequestException as e:
            error_msg = f"Request error calling LLM: {str(e)}"
            if DEBUG:
                print(f"DEBUG: {error_msg}")
            return error_msg
        except json.JSONDecodeError as e:
            error_msg = f"JSON decode error: {str(e)}"
            if DEBUG:
                print(f"DEBUG: {error_msg}")
                print(f"DEBUG: Response text: {response.text[:500] if response else 'No response'}")
            return error_msg
        except Exception as e:
            error_msg = f"Unexpected error calling LLM: {str(e)}"
            if DEBUG:
                print(f"DEBUG: {error_msg}")
            return error_msg

    @property
    def _llm_type(self) -> str:
        return "custom_endpoint"

    def _stream(self, prompt: str, stop: Optional[List[str]] = None, **kwargs):
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': self.model_name,
            'prompt': prompt,
            'max_tokens': self.max_new_tokens,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'stream': True
        }
        
        if stop:
            data['stop'] = stop
            
        try:
            response = requests.post(self.endpoint_url, headers=headers, json=data, stream=True, timeout=60)
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        line = line[6:]
                        if line.strip() == '[DONE]':
                            break
                        try:
                            chunk = json.loads(line)
                            if 'choices' in chunk and len(chunk['choices']) > 0:
                                token = chunk['choices'][0].get('text', '')
                                if token and self.callbacks:
                                    for callback in self.callbacks:
                                        callback.on_llm_new_token(token)
                                yield token
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            if self.callbacks:
                for callback in self.callbacks:
                    callback.on_llm_new_token(f"Error in streaming: {str(e)}")

# LLM with Bearer Token authentication - Updated to use OpenAI
llm = CustomEndpointLLM(
    endpoint_url=INFERENCE_SERVER_URL,
    api_key=BEARER_TOKEN,  # Bearer token authentication
    model_name=MODEL_NAME,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    top_k=TOP_K,
    repetition_penalty=REPETITION_PENALTY,
    streaming=True,
    callbacks=[QueueCallback(q)]
)

# Prompt
template="""<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant named HatBot answering questions about OpenShift Data Science, aka RHODS.
You will be given a question you need to answer, and a context to provide you with information. You must answer the question based as much as possible on this context.
Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.

If the context is not relevant to the question, you can use your general knowledge to provide a helpful answer, but please indicate that the answer is based on general knowledge rather than the specific documentation.
<</SYS>>

Question: {question}
Context: {context} [/INST]
"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 4, "score_threshold": 0.5}  # Higher threshold to filter out irrelevant results
    ),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    return_source_documents=True
)

# Fixed Gradio implementation
def ask_llm(message, history):
    """Fixed generator function for Gradio compatibility."""
    try:
        has_response = False
        for next_token, full_content in stream(message):
            has_response = True
            yield full_content
        # Ensure we always yield something, even if stream is empty
        if not has_response:
            yield "I apologize, but I couldn't generate a response. Please try rephrasing your question."
    except Exception as e:
        yield f"Error: {str(e)}"

# Updated Gradio interface for better compatibility
with gr.Blocks(title="Red Hat Demo Platform ChatBot", css="footer {visibility: hidden}") as demo:
    gr.Markdown(f"# {APP_TITLE}")
    
    # Use ChatInterface with built-in chatbot - improved for version compatibility
    try:
        gr.ChatInterface(
            ask_llm,
            title=None,  # We already have a title above
            description=None,
            retry_btn=None,
            undo_btn=None,
            clear_btn="Clear",
        )
    except TypeError:
        # Fallback for older gradio versions
        gr.ChatInterface(
            ask_llm,
            title=None,
            description=None
        )

if __name__ == "__main__":
    demo.queue().launch(
        server_name='0.0.0.0',
        share=False,
        favicon_path='./assets/robot-head.ico'
    )