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
SHOW_DEBUG_UI = os.getenv('SHOW_DEBUG_UI', 'False').lower() == 'true'

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

def detailed_retrieval_debug(input_text, debug_info=None):
    """Perform detailed debugging of the retrieval process"""
    if debug_info is None:
        debug_info = {}
    
    try:
        # Step 1: Test embedding generation
        if DEBUG or SHOW_DEBUG_UI:
            print(f"\n=== EMBEDDING DEBUG ===")
            print(f"Query: {input_text}")
            
        query_embedding = embeddings.embed_query(input_text)
        debug_info['query_embedding_length'] = len(query_embedding) if query_embedding else 0
        debug_info['query_embedding_sample'] = query_embedding[:5] if query_embedding else []
        
        if DEBUG or SHOW_DEBUG_UI:
            print(f"Query embedding dimension: {len(query_embedding) if query_embedding else 0}")
            print(f"Query embedding sample (first 5): {query_embedding[:5] if query_embedding else []}")
        
        # Step 2: Test document retrieval with different similarity thresholds
        retriever = store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 4, "score_threshold": 0.5}
        )
        
        docs = retriever.get_relevant_documents(input_text)
        debug_info['retrieved_docs_count'] = len(docs)
        debug_info['retrieved_docs'] = []
        
        if DEBUG or SHOW_DEBUG_UI:
            print(f"\n=== RETRIEVAL DEBUG ===")
            print(f"Retrieved {len(docs)} documents with threshold 0.5")
        
        # Also try with a lower threshold to see more results
        retriever_low = store.as_retriever(
            search_type="similarity_score_threshold", 
            search_kwargs={"k": 10, "score_threshold": 0.1}
        )
        docs_low = retriever_low.get_relevant_documents(input_text)
        debug_info['retrieved_docs_low_threshold'] = len(docs_low)
        
        if DEBUG or SHOW_DEBUG_UI:
            print(f"Retrieved {len(docs_low)} documents with lower threshold 0.1")
        
        for i, doc in enumerate(docs):
            doc_info = {
                'index': i,
                'content_preview': doc.page_content[:200] if doc.page_content else "No content",
                'metadata': doc.metadata,
                'content_length': len(doc.page_content) if doc.page_content else 0
            }
            debug_info['retrieved_docs'].append(doc_info)
            
            if DEBUG or SHOW_DEBUG_UI:
                print(f"\nDocument {i+1}:")
                print(f"  Source: {doc.metadata.get('source', 'Unknown')}")
                print(f"  Content length: {len(doc.page_content) if doc.page_content else 0}")
                print(f"  Content preview: {doc.page_content[:200] if doc.page_content else 'No content'}...")
                if hasattr(doc, 'metadata') and 'score' in doc.metadata:
                    print(f"  Similarity score: {doc.metadata['score']}")
        
        # Step 3: Test context formation
        context = "\n\n".join([doc.page_content for doc in docs if doc.page_content])
        debug_info['context_length'] = len(context)
        debug_info['context_preview'] = context[:500] if context else "No context"
        
        if DEBUG or SHOW_DEBUG_UI:
            print(f"\n=== CONTEXT DEBUG ===")
            print(f"Combined context length: {len(context)}")
            print(f"Context preview: {context[:500]}...")
        
        return docs, context, debug_info
        
    except Exception as e:
        error_msg = f"Error in detailed retrieval debug: {str(e)}"
        debug_info['retrieval_error'] = error_msg
        if DEBUG or SHOW_DEBUG_UI:
            print(f"ERROR: {error_msg}")
        return [], "", debug_info

def test_llm_directly(prompt, debug_info=None):
    """Test LLM directly with a prompt"""
    if debug_info is None:
        debug_info = {}
        
    try:
        if DEBUG or SHOW_DEBUG_UI:
            print(f"\n=== LLM DIRECT TEST ===")
            print(f"Prompt length: {len(prompt)}")
            print(f"Prompt preview: {prompt[:300]}...")
        
        response = llm._call(prompt)
        debug_info['llm_response_length'] = len(response) if response else 0
        debug_info['llm_response_preview'] = response[:200] if response else "No response"
        debug_info['llm_response_full'] = response
        
        if DEBUG or SHOW_DEBUG_UI:
            print(f"LLM response length: {len(response) if response else 0}")
            print(f"LLM response: {response}")
        
        return response, debug_info
        
    except Exception as e:
        error_msg = f"Error in direct LLM test: {str(e)}"
        debug_info['llm_error'] = error_msg
        if DEBUG or SHOW_DEBUG_UI:
            print(f"ERROR: {error_msg}")
        return "", debug_info

def stream(input_text) -> Generator:
    # Create a Queue
    job_done = object()
    debug_info = {}

    # Create a function to call - this will run in a thread
    def task():
        try:
            if DEBUG or SHOW_DEBUG_UI:
                print(f"\n{'='*50}")
                print(f"PROCESSING QUERY: {input_text}")
                print(f"{'='*50}")
            
            # Step 1: Detailed retrieval debugging
            docs, context, debug_info_retrieval = detailed_retrieval_debug(input_text, debug_info)
            debug_info.update(debug_info_retrieval)
            
            # Step 2: Manual prompt construction and LLM testing
            if context:
                manual_prompt = template.format(question=input_text, context=context)
                llm_response, debug_info_llm = test_llm_directly(manual_prompt, debug_info)
                debug_info.update(debug_info_llm)
                
                if llm_response and llm_response.strip():
                    if DEBUG or SHOW_DEBUG_UI:
                        print(f"\n=== SUCCESS: Got LLM response directly ===")
                    # Use the direct LLM response
                    answer = llm_response.strip()
                    words = answer.split()
                    for word in words:
                        q.put(word + " ")
                        time.sleep(0.05)
                else:
                    if DEBUG or SHOW_DEBUG_UI:
                        print(f"\n=== ISSUE: Empty LLM response ===")
                    q.put("I found relevant information but the LLM didn't generate a response. This might be a model configuration issue. ")
            else:
                if DEBUG or SHOW_DEBUG_UI:
                    print(f"\n=== ISSUE: No context retrieved ===")
                q.put("I found relevant sources but couldn't extract meaningful context. ")
            
            # Step 3: Also try the original QA chain for comparison
            if DEBUG or SHOW_DEBUG_UI:
                print(f"\n=== TESTING ORIGINAL QA CHAIN ===")
            
            try:
                resp = qa_chain({"query": input_text})
                debug_info['qa_chain_keys'] = list(resp.keys()) if resp else []
                
                if DEBUG or SHOW_DEBUG_UI:
                    print(f"QA chain response keys: {resp.keys() if resp else 'None'}")
                    if 'result' in resp:
                        print(f"QA chain result length: {len(resp['result']) if resp['result'] else 0}")
                        print(f"QA chain result: {resp['result'][:200] if resp['result'] else 'Empty result'}")
                    if 'source_documents' in resp:
                        print(f"QA chain found {len(resp['source_documents'])} source documents")
                
                # Compare with our manual approach
                if 'result' in resp and resp['result'] and resp['result'].strip():
                    qa_answer = resp['result'].strip()
                    if qa_answer != llm_response:
                        if DEBUG or SHOW_DEBUG_UI:
                            print(f"\n=== DIFFERENCE DETECTED ===")
                            print(f"Manual LLM response length: {len(llm_response)}")
                            print(f"QA chain response length: {len(qa_answer)}")
                
            except Exception as qa_e:
                if DEBUG or SHOW_DEBUG_UI:
                    print(f"QA chain error: {str(qa_e)}")
                debug_info['qa_chain_error'] = str(qa_e)
            
            # Add debug information to output if enabled
            if SHOW_DEBUG_UI and debug_info:
                q.put("\n\n" + "="*50 + "\n")
                q.put("🔍 **DEBUG INFORMATION**\n\n")
                
                if 'query_embedding_length' in debug_info:
                    q.put(f"**Query Embedding:** {debug_info['query_embedding_length']} dimensions\n")
                
                if 'retrieved_docs_count' in debug_info:
                    q.put(f"**Retrieved Documents:** {debug_info['retrieved_docs_count']} (threshold 0.5), {debug_info.get('retrieved_docs_low_threshold', 0)} (threshold 0.1)\n")
                
                if 'context_length' in debug_info:
                    q.put(f"**Context Length:** {debug_info['context_length']} characters\n")
                
                if 'llm_response_length' in debug_info:
                    q.put(f"**LLM Response Length:** {debug_info['llm_response_length']} characters\n")
                
                if 'retrieval_error' in debug_info:
                    q.put(f"**Retrieval Error:** {debug_info['retrieval_error']}\n")
                
                if 'llm_error' in debug_info:
                    q.put(f"**LLM Error:** {debug_info['llm_error']}\n")
                
                if 'qa_chain_error' in debug_info:
                    q.put(f"**QA Chain Error:** {debug_info['qa_chain_error']}\n")
                
                # Show retrieved documents
                if 'retrieved_docs' in debug_info and debug_info['retrieved_docs']:
                    q.put("\n**Retrieved Documents:**\n")
                    for doc_info in debug_info['retrieved_docs']:
                        q.put(f"- **Doc {doc_info['index']+1}** ({doc_info['content_length']} chars): {doc_info['content_preview']}...\n")
                        q.put(f"  Source: {doc_info['metadata'].get('source', 'Unknown')}\n")
                
                q.put("\n" + "="*50 + "\n\n")
            
            # Then add sources if available
            if docs:
                sources = remove_source_duplicates(docs)
                if len(sources) != 0:
                    q.put("\n\n*Sources:* \n")
                    for source in sources:
                        q.put("* " + str(source) + "\n")
                else:
                    q.put("\n\n*Note: No specific sources found for this query.*\n")
            else:
                q.put("\n\n*Note: No specific sources found for this query.*\n")
                
        except Exception as e:
            if DEBUG or SHOW_DEBUG_UI:
                print(f"DEBUG: Exception in task: {str(e)}")
            error_msg = str(e)
            if "No relevant docs were retrieved" in error_msg:
                # If no docs retrieved, try to answer without RAG context
                try:
                    if DEBUG or SHOW_DEBUG_UI:
                        print("DEBUG: Attempting general question fallback")
                    general_response = answer_general_question(input_text)
                    if general_response:
                        words = general_response.split()
                        for word in words:
                            q.put(word + " ")
                            time.sleep(0.05)
                    q.put("\n\n*Note: I couldn't find specific documentation for this query, so I answered based on my general knowledge.*\n")
                except Exception as inner_e:
                    if DEBUG or SHOW_DEBUG_UI:
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
            if DEBUG or SHOW_DEBUG_UI:
                print(f"DEBUG: Making LLM request to {self.endpoint_url}")
                print(f"DEBUG: Request data keys: {data.keys()}")
                print(f"DEBUG: Prompt length: {len(prompt)}")
                print(f"DEBUG: Prompt preview: {prompt[:200]}...")
                
            # Try with extended timeout and retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    timeout = 120 if attempt == 0 else 180  # Increase timeout on retries
                    response = requests.post(
                        self.endpoint_url, 
                        headers=headers, 
                        json=data, 
                        timeout=timeout
                    )
                    
                    if DEBUG or SHOW_DEBUG_UI:
                        print(f"DEBUG: Attempt {attempt + 1} - Response status code: {response.status_code}")
                        print(f"DEBUG: Response headers: {dict(response.headers)}")
                    
                    # Check for specific error status codes
                    if response.status_code == 401:
                        return "ERROR: Authentication failed - Invalid bearer token"
                    elif response.status_code == 403:
                        return "ERROR: Access forbidden - Check bearer token permissions"
                    elif response.status_code == 404:
                        return "ERROR: Model endpoint not found - Check INFERENCE_SERVER_URL"
                    elif response.status_code == 429:
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt  # Exponential backoff
                            if DEBUG or SHOW_DEBUG_UI:
                                print(f"DEBUG: Rate limited, waiting {wait_time}s before retry")
                            time.sleep(wait_time)
                            continue
                        else:
                            return "ERROR: Rate limit exceeded - Try again later"
                    elif response.status_code >= 500:
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt
                            if DEBUG or SHOW_DEBUG_UI:
                                print(f"DEBUG: Server error {response.status_code}, waiting {wait_time}s before retry")
                            time.sleep(wait_time)
                            continue
                        else:
                            return f"ERROR: Server error {response.status_code} - Model service unavailable"
                    
                    response.raise_for_status()
                    break  # Success, exit retry loop
                    
                except requests.exceptions.Timeout:
                    if attempt < max_retries - 1:
                        if DEBUG or SHOW_DEBUG_UI:
                            print(f"DEBUG: Timeout on attempt {attempt + 1}, retrying...")
                        continue
                    else:
                        return "ERROR: Request timeout - Model service is not responding"
                except requests.exceptions.ConnectionError:
                    if attempt < max_retries - 1:
                        if DEBUG or SHOW_DEBUG_UI:
                            print(f"DEBUG: Connection error on attempt {attempt + 1}, retrying...")
                        time.sleep(1)
                        continue
                    else:
                        return "ERROR: Connection failed - Cannot reach model service"
            
            # Parse response
            try:
                result = response.json()
            except json.JSONDecodeError as e:
                if DEBUG or SHOW_DEBUG_UI:
                    print(f"DEBUG: JSON decode error: {str(e)}")
                    print(f"DEBUG: Response text: {response.text[:500]}")
                return f"ERROR: Invalid JSON response from model service"
            
            if DEBUG or SHOW_DEBUG_UI:
                print(f"DEBUG: Response JSON keys: {result.keys() if result else 'None'}")
            
            # Extract generated text
            if 'choices' in result and len(result['choices']) > 0:
                generated_text = result['choices'][0].get('text', '')
                if DEBUG or SHOW_DEBUG_UI:
                    print(f"DEBUG: Generated text length: {len(generated_text)}")
                    print(f"DEBUG: Generated text preview: {generated_text[:200] if generated_text else 'Empty'}")
                
                # Check if response is actually empty or just whitespace
                if not generated_text or not generated_text.strip():
                    if DEBUG or SHOW_DEBUG_UI:
                        print(f"DEBUG: Empty or whitespace-only response from LLM")
                    return "ERROR: Model returned empty response - Check model configuration"
                
                return generated_text.strip()
            else:
                if DEBUG or SHOW_DEBUG_UI:
                    print("DEBUG: No choices in response or empty choices")
                    print(f"DEBUG: Full response: {result}")
                
                # Check for error in response
                if 'error' in result:
                    error_detail = result['error']
                    if isinstance(error_detail, dict):
                        error_msg = error_detail.get('message', str(error_detail))
                    else:
                        error_msg = str(error_detail)
                    return f"ERROR: Model error - {error_msg}"
                
                return "ERROR: No response choices returned from model"
                
        except requests.exceptions.RequestException as e:
            error_msg = f"Request error calling LLM: {str(e)}"
            if DEBUG or SHOW_DEBUG_UI:
                print(f"DEBUG: {error_msg}")
            return f"ERROR: {error_msg}"
        except Exception as e:
            error_msg = f"Unexpected error calling LLM: {str(e)}"
            if DEBUG or SHOW_DEBUG_UI:
                print(f"DEBUG: {error_msg}")
            return f"ERROR: {error_msg}"

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

# Enhanced prompt template with explicit instructions
template="""<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant named HatBot answering questions about OpenShift Data Science, aka RHODS.

IMPORTANT INSTRUCTIONS:
1. You MUST provide a detailed, comprehensive answer to the user's question
2. Use the provided context to answer the question as accurately as possible
3. If the context contains relevant information, explain it thoroughly and clearly
4. Always provide a complete response, not just a summary or brief answer
5. Structure your response with clear explanations and examples when possible

RESPONSE FORMAT:
- Start with a direct answer to the question
- Provide detailed explanation using the context
- Include specific details, steps, or examples from the context
- End with any additional helpful information

Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.

If the context is not relevant to the question, you can use your general knowledge to provide a helpful answer, but please indicate that the answer is based on general knowledge rather than the specific documentation.

If you don't know the answer, say so clearly rather than making something up.
<</SYS>>

Question: {question}

Context Information:
{context}

Please provide a detailed and comprehensive answer based on the above context: [/INST]
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

# Enhanced Gradio interface with debug controls
with gr.Blocks(title="Red Hat Demo Platform ChatBot", css="footer {visibility: hidden}") as demo:
    gr.Markdown(f"# {APP_TITLE}")
    
    with gr.Row():
        with gr.Column(scale=3):
            # Main chat interface
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
        
        with gr.Column(scale=1):
            gr.Markdown("### 🔧 Debug Controls")
            
            with gr.Group():
                gr.Markdown("**Environment Variables:**")
                debug_enabled = gr.Checkbox(
                    label="Enable Console Debug Logs", 
                    value=DEBUG,
                    info="Shows detailed logs in server console"
                )
                show_debug_ui = gr.Checkbox(
                    label="Show Debug Info in Chat", 
                    value=SHOW_DEBUG_UI,
                    info="Shows debug information in chat responses"
                )
                
                def update_debug_flags(console_debug, ui_debug):
                    global DEBUG, SHOW_DEBUG_UI
                    DEBUG = console_debug
                    SHOW_DEBUG_UI = ui_debug
                    return f"Debug flags updated: Console={console_debug}, UI={ui_debug}"
                
                update_btn = gr.Button("Update Debug Settings")
                debug_status = gr.Textbox(label="Status", interactive=False)
                
                update_btn.click(
                    fn=update_debug_flags,
                    inputs=[debug_enabled, show_debug_ui],
                    outputs=[debug_status]
                )
            
            with gr.Group():
                gr.Markdown("**Test Components:**")
                test_query = gr.Textbox(
                    label="Test Query", 
                    placeholder="Enter a test question...",
                    lines=2
                )
                
                def test_retrieval_only(query):
                    if not query:
                        return "Please enter a test query"
                    
                    try:
                        docs, context, debug_info = detailed_retrieval_debug(query)
                        
                        result = f"**Retrieval Test Results:**\n\n"
                        result += f"• Documents found: {len(docs)}\n"
                        result += f"• Context length: {len(context)} chars\n"
                        result += f"• Embedding dims: {debug_info.get('query_embedding_length', 'N/A')}\n\n"
                        
                        if docs:
                            result += "**Retrieved Documents:**\n"
                            for i, doc in enumerate(docs[:3]):  # Show first 3
                                result += f"{i+1}. {doc.metadata.get('source', 'Unknown')}\n"
                                result += f"   {doc.page_content[:100]}...\n\n"
                        
                        if 'retrieval_error' in debug_info:
                            result += f"**Error:** {debug_info['retrieval_error']}\n"
                        
                        return result
                    except Exception as e:
                        return f"Error testing retrieval: {str(e)}"
                
                def test_llm_only(query):
                    if not query:
                        return "Please enter a test query"
                    
                    try:
                        # Simple test prompt
                        test_prompt = f"<s>[INST] {query} [/INST]"
                        response, debug_info = test_llm_directly(test_prompt)
                        
                        result = f"**LLM Test Results:**\n\n"
                        result += f"• Response length: {debug_info.get('llm_response_length', 0)} chars\n"
                        result += f"• Response preview: {debug_info.get('llm_response_preview', 'No response')}\n\n"
                        
                        if 'llm_error' in debug_info:
                            result += f"**Error:** {debug_info['llm_error']}\n"
                        elif response:
                            result += f"**Full Response:**\n{response}\n"
                        
                        return result
                    except Exception as e:
                        return f"Error testing LLM: {str(e)}"
                
                test_retrieval_btn = gr.Button("Test Retrieval Only")
                test_llm_btn = gr.Button("Test LLM Only")
                test_results = gr.Textbox(
                    label="Test Results", 
                    lines=10, 
                    interactive=False
                )
                
                test_retrieval_btn.click(
                    fn=test_retrieval_only,
                    inputs=[test_query],
                    outputs=[test_results]
                )
                
                test_llm_btn.click(
                    fn=test_llm_only,
                    inputs=[test_query],
                    outputs=[test_results]
                )
            
            with gr.Group():
                gr.Markdown("**System Info:**")
                system_info = gr.HTML(f"""
                <small>
                <b>Model:</b> {MODEL_NAME}<br>
                <b>Max Tokens:</b> {MAX_NEW_TOKENS}<br>
                <b>Temperature:</b> {TEMPERATURE}<br>
                <b>DB Collection:</b> {DB_COLLECTION_NAME}<br>
                <b>Endpoint:</b> {INFERENCE_SERVER_URL[:50]}...<br>
                <b>Bearer Token:</b> {'✓ Set' if BEARER_TOKEN else '✗ Missing'}
                </small>
                """)
            
            with gr.Group():
                gr.Markdown("**Troubleshooting Steps:**")
                troubleshooting = gr.HTML("""
                <small>
                <ol>
                <li><b>Check Bearer Token:</b> Ensure BEARER_TOKEN is valid</li>
                <li><b>Test LLM:</b> Use "Test LLM Only" button</li>
                <li><b>Test Retrieval:</b> Use "Test Retrieval Only" button</li>
                <li><b>Check Logs:</b> Enable console debug logs</li>
                <li><b>Try Lower Threshold:</b> Reduce similarity threshold</li>
                <li><b>Check Model:</b> Verify model name and endpoint</li>
                </ol>
                </small>
                """)

if __name__ == "__main__":
    demo.queue().launch(
        server_name='0.0.0.0',
        share=False,
        favicon_path='./assets/robot-head.ico'
    )