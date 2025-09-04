#!/usr/bin/env python3
"""
Enhanced LLM Usage Module with Configuration and MCP Integration
Replaces the original llm_usage.py with configurable parameters and external data sources
"""

import os
import json
from typing import Dict, List, Optional, Any
from dataclasses import asdict

from langchain.llms import HuggingFaceTextGenInference
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.evaluation import load_evaluator
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

# Import our MCP integration
from mcp_integration import MCPManager, MCPDocument

# Default configuration values (fallback)
DEFAULT_CONFIG = {
    "server_url": "http://llm.ic-shared-llm.svc.cluster.local:3000",
    "api_token": "",
    "model_name": "default",
    "max_new_tokens": 512,
    "temperature": 0.01,
    "top_k": 10,
    "top_p": 0.95,
    "typical_p": 0.95,
    "repetition_penalty": 1.03,
    "streaming": True
}

class ConfigurableLLMManager:
    """Manages LLM configuration and provides inference capabilities"""
    
    def __init__(self, config_file: str = "configs/llm_config.json"):
        self.config_file = config_file
        self.config = self._load_config()
        self.llm = None
        self.mcp_manager = MCPManager()
        self._initialize_llm()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load LLM configuration from file or use defaults"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                return config
        except Exception as e:
            print(f"Warning: Failed to load config from {self.config_file}: {e}")
        
        # Load from environment variables if file not available
        return self._load_from_env()
    
    def _load_from_env(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        config = DEFAULT_CONFIG.copy()
        
        # Map environment variables to config keys
        env_mapping = {
            "INFERENCE_SERVER_URL": "server_url",
            "API_TOKEN": "api_token", 
            "MODEL_NAME": "model_name",
            "MAX_NEW_TOKENS": "max_new_tokens",
            "TEMPERATURE": "temperature",
            "TOP_K": "top_k",
            "TOP_P": "top_p",
            "TYPICAL_P": "typical_p",
            "REPETITION_PENALTY": "repetition_penalty",
            "STREAMING": "streaming"
        }
        
        for env_var, config_key in env_mapping.items():
            env_value = os.getenv(env_var)
            if env_value:
                # Convert to appropriate type
                if config_key in ["max_new_tokens", "top_k"]:
                    config[config_key] = int(env_value)
                elif config_key in ["temperature", "top_p", "typical_p", "repetition_penalty"]:
                    config[config_key] = float(env_value)
                elif config_key == "streaming":
                    config[config_key] = env_value.lower() == "true"
                else:
                    config[config_key] = env_value
        
        return config
    
    def _initialize_llm(self):
        """Initialize the LLM with current configuration"""
        try:
            # Check if it's an OpenAI-compatible API or HuggingFace TGI
            if "openai" in self.config["server_url"].lower() or "v1" in self.config["server_url"]:
                # Use ChatOpenAI for OpenAI-compatible APIs
                self.llm = ChatOpenAI(
                    model=self.config["model_name"],
                    api_key=self.config["api_token"],
                    base_url=self.config["server_url"],
                    temperature=self.config["temperature"],
                    max_tokens=self.config["max_new_tokens"]
                )
            else:
                # Use HuggingFace TGI for other endpoints
                self.llm = HuggingFaceTextGenInference(
                    inference_server_url=self.config["server_url"],
                    max_new_tokens=self.config["max_new_tokens"],
                    top_k=self.config["top_k"],
                    top_p=self.config["top_p"],
                    typical_p=self.config["typical_p"],
                    temperature=self.config["temperature"],
                    repetition_penalty=self.config["repetition_penalty"],
                    streaming=self.config["streaming"],
                    verbose=False,
                )
                
                # Add token authentication if provided
                if self.config["api_token"]:
                    self.llm.headers = {"Authorization": f"Bearer {self.config['api_token']}"}
            
            print(f"✅ LLM initialized with endpoint: {self.config['server_url']}")
        
        except Exception as e:
            print(f"❌ Failed to initialize LLM: {e}")
            raise
    
    def get_llm(self):
        """Get the configured LLM instance"""
        return self.llm
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update configuration and reinitialize LLM"""
        self.config.update(new_config)
        self._initialize_llm()
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return self.config.copy()

class EnhancedRAGPipeline:
    """Enhanced RAG pipeline with MCP integration"""
    
    def __init__(self, llm_manager: ConfigurableLLMManager):
        self.llm_manager = llm_manager
        self.mcp_manager = MCPManager()
        self.embeddings = HuggingFaceEmbeddings()
        self.output_parser = StrOutputParser()
    
    def create_rag_chain_with_mcp(self, 
                                  template: str, 
                                  use_mcp: bool = True,
                                  mcp_query: Optional[str] = None,
                                  mcp_limit_per_source: int = 3):
        """Create a RAG chain that incorporates MCP sources"""
        
        def format_docs(docs):
            """Format documents for the prompt"""
            return "\n\n".join(doc.page_content for doc in docs)
        
        def get_context_with_mcp(query: str) -> str:
            """Get context from both traditional sources and MCP"""
            context_parts = []
            
            if use_mcp and self.mcp_manager.list_connectors():
                # Get documents from MCP sources
                search_query = mcp_query or query
                mcp_docs = self.mcp_manager.search_all_sources(search_query, mcp_limit_per_source)
                
                if mcp_docs:
                    mcp_context = "\n\n=== External Sources ===\n\n"
                    for doc in mcp_docs:
                        mcp_context += f"**{doc.title}** (from {doc.source}):\n{doc.content[:500]}...\n\n"
                    context_parts.append(mcp_context)
            
            return "\n\n".join(context_parts) if context_parts else "No external context available."
        
        # Create the prompt template
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Create the chain
        rag_chain = (
            {
                "context": lambda x: get_context_with_mcp(x),
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm_manager.get_llm()
            | self.output_parser
        )
        
        return rag_chain

# Legacy compatibility functions
def load_llm_config() -> Dict[str, Any]:
    """Load LLM configuration (legacy compatibility)"""
    manager = ConfigurableLLMManager()
    return manager.get_config()

def infer_with_template(input_text: str, template: str, use_mcp: bool = True) -> str:
    """Enhanced version of the original infer_with_template function"""
    try:
        # Initialize LLM manager
        llm_manager = ConfigurableLLMManager()
        
        # Create enhanced RAG pipeline
        rag_pipeline = EnhancedRAGPipeline(llm_manager)
        
        # Create the RAG chain
        rag_chain = rag_pipeline.create_rag_chain_with_mcp(
            template=template,
            use_mcp=use_mcp,
            mcp_query=input_text
        )
        
        # Run inference
        response = rag_chain.invoke(input_text)
        return response
        
    except Exception as e:
        print(f"Error in infer_with_template: {e}")
        # Fallback to simple LLM inference without MCP
        return _fallback_inference(input_text, template)

def _fallback_inference(input_text: str, template: str) -> str:
    """Fallback inference without MCP integration"""
    try:
        llm_manager = ConfigurableLLMManager()
        llm = llm_manager.get_llm()
        
        # Simple template formatting
        prompt = PromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()
        
        return chain.invoke({"input_text": input_text})
    
    except Exception as e:
        print(f"Fallback inference failed: {e}")
        return f"Error: Unable to generate response. {str(e)}"

def similarity_metric(predicted_text: str, reference_text: str) -> float:
    """Calculate similarity between predicted and reference text"""
    try:
        embedding_model = HuggingFaceEmbeddings()
        evaluator = load_evaluator("embedding_distance", embeddings=embedding_model)
        distance_score = evaluator.evaluate_strings(prediction=predicted_text, reference=reference_text)
        return 1 - distance_score["score"]
    except Exception as e:
        print(f"Error in similarity calculation: {e}")
        return 0.0

def enhanced_infer_with_context(input_text: str, 
                               template: str,
                               external_sources: bool = True,
                               github_repos: Optional[List[str]] = None,
                               notion_pages: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Enhanced inference with detailed context from multiple sources
    
    Args:
        input_text: The input query/text
        template: The prompt template
        external_sources: Whether to use external MCP sources
        github_repos: Specific GitHub repositories to search
        notion_pages: Specific Notion pages to search
    
    Returns:
        Dictionary with response and metadata
    """
    try:
        llm_manager = ConfigurableLLMManager()
        mcp_manager = MCPManager()
        
        # Collect context from various sources
        context_info = {
            "mcp_sources_used": [],
            "external_documents": [],
            "total_context_length": 0
        }
        
        # Get MCP context if enabled
        if external_sources:
            mcp_docs = mcp_manager.search_all_sources(input_text, limit_per_source=2)
            
            for doc in mcp_docs:
                context_info["external_documents"].append({
                    "title": doc.title,
                    "source": doc.source,
                    "type": doc.doc_type,
                    "length": len(doc.content)
                })
                context_info["mcp_sources_used"].append(doc.source)
                context_info["total_context_length"] += len(doc.content)
        
        # Create enhanced pipeline
        rag_pipeline = EnhancedRAGPipeline(llm_manager)
        rag_chain = rag_pipeline.create_rag_chain_with_mcp(
            template=template,
            use_mcp=external_sources,
            mcp_query=input_text
        )
        
        # Generate response
        response = rag_chain.invoke(input_text)
        
        return {
            "response": response,
            "input_text": input_text,
            "template_used": template,
            "llm_config": llm_manager.get_config(),
            "context_info": context_info,
            "success": True
        }
    
    except Exception as e:
        return {
            "response": f"Error generating response: {str(e)}",
            "input_text": input_text,
            "success": False,
            "error": str(e)
        }

# Testing and utility functions
def test_mcp_integration():
    """Test MCP integration functionality"""
    print("🧪 Testing MCP Integration...")
    
    mcp_manager = MCPManager()
    
    # Test connections
    print("\n📡 Testing MCP connections:")
    connection_results = mcp_manager.test_all_connections()
    for name, status in connection_results.items():
        print(f"  {name}: {'✅ Connected' if status else '❌ Failed'}")
    
    # Test document retrieval
    print("\n📚 Testing document retrieval:")
    docs = mcp_manager.retrieve_all_documents(limit_per_source=2)
    print(f"Retrieved {len(docs)} documents total")
    
    for doc in docs[:3]:  # Show first 3
        print(f"  - {doc.title} ({doc.doc_type}) from {doc.source}")
    
    return len(docs) > 0

def test_enhanced_inference():
    """Test the enhanced inference capabilities"""
    print("\n🚀 Testing Enhanced Inference...")
    
    test_query = "What are the main features and benefits?"
    test_template = """
    Based on the following context, please answer the question comprehensively:
    
    Context: {context}
    
    Question: {question}
    
    Please provide a detailed answer based on the available information.
    """
    
    result = enhanced_infer_with_context(
        input_text=test_query,
        template=test_template,
        external_sources=True
    )
    
    if result["success"]:
        print(f"✅ Response generated successfully")
        print(f"📊 Used {len(result['context_info']['mcp_sources_used'])} external sources")
        print(f"📝 Total context length: {result['context_info']['total_context_length']} characters")
        print(f"🔤 Response length: {len(result['response'])} characters")
    else:
        print(f"❌ Inference failed: {result['error']}")
    
    return result["success"]

def main():
    """Main function for testing the enhanced LLM usage"""
    print("🚀 Enhanced LLM Usage with MCP Integration")
    print("=" * 50)
    
    # Test LLM configuration
    print("⚙️ Testing LLM Configuration...")
    try:
        llm_manager = ConfigurableLLMManager()
        config = llm_manager.get_config()
        print(f"✅ LLM configured with endpoint: {config['server_url']}")
    except Exception as e:
        print(f"❌ LLM configuration failed: {e}")
        return
    
    # Test MCP integration
    mcp_success = test_mcp_integration()
    
    # Test enhanced inference
    inference_success = test_enhanced_inference()
    
    print("\n📋 Summary:")
    print(f"  LLM Configuration: ✅ Success")
    print(f"  MCP Integration: {'✅ Success' if mcp_success else '⚠️ Limited'}")
    print(f"  Enhanced Inference: {'✅ Success' if inference_success else '❌ Failed'}")
    
    if mcp_success and inference_success:
        print("\n🎉 All systems operational! Ready for enhanced LLM pipeline.")
    else:
        print("\n⚠️ Some features may be limited. Check your configuration.")

if __name__ == "__main__":
    main() 