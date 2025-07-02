#!/usr/bin/env python3
"""
Standalone component tester for RAG application
Helps identify which component is causing issues
"""

import os
import sys
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
BEARER_TOKEN = os.getenv('BEARER_TOKEN')
INFERENCE_SERVER_URL = os.getenv('INFERENCE_SERVER_URL')
MODEL_NAME = os.getenv('MODEL_NAME', 'granite-8b-code-instruct-128k')
DB_CONNECTION_STRING = os.getenv('DB_CONNECTION_STRING')
DB_COLLECTION_NAME = os.getenv('DB_COLLECTION_NAME', 'langchain_pg_collection')

def test_environment():
    """Test environment variables"""
    print("=" * 50)
    print("TESTING ENVIRONMENT VARIABLES")
    print("=" * 50)
    
    required_vars = {
        'BEARER_TOKEN': BEARER_TOKEN,
        'INFERENCE_SERVER_URL': INFERENCE_SERVER_URL,
        'MODEL_NAME': MODEL_NAME,
        'DB_CONNECTION_STRING': DB_CONNECTION_STRING,
        'DB_COLLECTION_NAME': DB_COLLECTION_NAME
    }
    
    all_good = True
    for var_name, var_value in required_vars.items():
        if var_value:
            print(f"✓ {var_name}: {'*' * min(len(str(var_value)), 20)}... (set)")
        else:
            print(f"✗ {var_name}: NOT SET")
            all_good = False
    
    print(f"\nEnvironment Status: {'✓ PASS' if all_good else '✗ FAIL'}")
    return all_good

def test_llm_endpoint():
    """Test LLM endpoint directly"""
    print("\n" + "=" * 50)
    print("TESTING LLM ENDPOINT")
    print("=" * 50)
    
    if not BEARER_TOKEN or not INFERENCE_SERVER_URL:
        print("✗ Missing BEARER_TOKEN or INFERENCE_SERVER_URL")
        return False
    
    headers = {
        'Authorization': f'Bearer {BEARER_TOKEN}',
        'Content-Type': 'application/json'
    }
    
    data = {
        'model': MODEL_NAME,
        'prompt': 'What is OpenShift? Please provide a brief answer.',
        'max_tokens': 100,
        'temperature': 0.1,
        'stream': False
    }
    
    try:
        print(f"Making request to: {INFERENCE_SERVER_URL}")
        print(f"Model: {MODEL_NAME}")
        print(f"Prompt: {data['prompt']}")
        
        response = requests.post(INFERENCE_SERVER_URL, headers=headers, json=data, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 401:
            print("✗ Authentication failed - Invalid bearer token")
            return False
        elif response.status_code == 403:
            print("✗ Access forbidden - Check bearer token permissions")
            return False
        elif response.status_code == 404:
            print("✗ Model endpoint not found - Check INFERENCE_SERVER_URL")
            return False
        elif response.status_code == 429:
            print("✗ Rate limit exceeded")
            return False
        elif response.status_code >= 500:
            print(f"✗ Server error {response.status_code}")
            return False
        
        response.raise_for_status()
        
        try:
            result = response.json()
            print(f"Response keys: {result.keys()}")
            
            if 'choices' in result and len(result['choices']) > 0:
                generated_text = result['choices'][0].get('text', '')
                print(f"Generated text length: {len(generated_text)}")
                print(f"Generated text: {generated_text}")
                
                if generated_text and generated_text.strip():
                    print("✓ LLM endpoint working correctly")
                    return True
                else:
                    print("✗ LLM returned empty response")
                    return False
            else:
                print("✗ No choices in response")
                print(f"Full response: {result}")
                return False
                
        except json.JSONDecodeError as e:
            print(f"✗ JSON decode error: {e}")
            print(f"Response text: {response.text[:500]}")
            return False
            
    except requests.exceptions.Timeout:
        print("✗ Request timeout")
        return False
    except requests.exceptions.ConnectionError:
        print("✗ Connection failed")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_database_connection():
    """Test database connection"""
    print("\n" + "=" * 50)
    print("TESTING DATABASE CONNECTION")
    print("=" * 50)
    
    if not DB_CONNECTION_STRING:
        print("✗ DB_CONNECTION_STRING not set")
        return False
    
    try:
        # Try to import required libraries
        from langchain.embeddings import HuggingFaceEmbeddings
        from langchain.vectorstores.pgvector import PGVector
        
        print(f"Connection string: {DB_CONNECTION_STRING[:50]}...")
        print(f"Collection name: {DB_COLLECTION_NAME}")
        
        # Test embeddings
        print("Testing embeddings...")
        embeddings = HuggingFaceEmbeddings()
        test_embedding = embeddings.embed_query("test query")
        print(f"✓ Embeddings working - dimension: {len(test_embedding)}")
        
        # Test database connection
        print("Testing database connection...")
        store = PGVector(
            connection_string=DB_CONNECTION_STRING,
            collection_name=DB_COLLECTION_NAME,
            embedding_function=embeddings,
        )
        
        # Test retrieval
        print("Testing document retrieval...")
        docs = store.similarity_search("test", k=1)
        print(f"✓ Database connection working - found {len(docs)} test documents")
        
        return True
        
    except ImportError as e:
        print(f"✗ Missing required library: {e}")
        return False
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False

def test_retrieval():
    """Test document retrieval with a real query"""
    print("\n" + "=" * 50)
    print("TESTING DOCUMENT RETRIEVAL")
    print("=" * 50)
    
    try:
        from langchain.embeddings import HuggingFaceEmbeddings
        from langchain.vectorstores.pgvector import PGVector
        
        embeddings = HuggingFaceEmbeddings()
        store = PGVector(
            connection_string=DB_CONNECTION_STRING,
            collection_name=DB_COLLECTION_NAME,
            embedding_function=embeddings,
        )
        
        test_queries = [
            "What is OpenShift?",
            "How to deploy applications?",
            "Container orchestration"
        ]
        
        for query in test_queries:
            print(f"\nTesting query: '{query}'")
            
            # Test with high threshold
            retriever = store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 4, "score_threshold": 0.5}
            )
            docs = retriever.get_relevant_documents(query)
            print(f"  High threshold (0.5): {len(docs)} documents")
            
            # Test with low threshold
            retriever_low = store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 10, "score_threshold": 0.1}
            )
            docs_low = retriever_low.get_relevant_documents(query)
            print(f"  Low threshold (0.1): {len(docs_low)} documents")
            
            if docs:
                print(f"  Sample document: {docs[0].page_content[:100]}...")
                print(f"  Source: {docs[0].metadata.get('source', 'Unknown')}")
        
        print("✓ Document retrieval working")
        return True
        
    except Exception as e:
        print(f"✗ Retrieval test failed: {e}")
        return False

def test_full_rag_pipeline():
    """Test the complete RAG pipeline"""
    print("\n" + "=" * 50)
    print("TESTING FULL RAG PIPELINE")
    print("=" * 50)
    
    try:
        from langchain.embeddings import HuggingFaceEmbeddings
        from langchain.vectorstores.pgvector import PGVector
        
        # Setup
        embeddings = HuggingFaceEmbeddings()
        store = PGVector(
            connection_string=DB_CONNECTION_STRING,
            collection_name=DB_COLLECTION_NAME,
            embedding_function=embeddings,
        )
        
        test_query = "What is OpenShift?"
        print(f"Testing with query: '{test_query}'")
        
        # Step 1: Retrieve documents
        retriever = store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 4, "score_threshold": 0.5}
        )
        docs = retriever.get_relevant_documents(test_query)
        print(f"Step 1 - Retrieved {len(docs)} documents")
        
        if not docs:
            print("✗ No documents retrieved - cannot test full pipeline")
            return False
        
        # Step 2: Build context
        context = "\n\n".join([doc.page_content for doc in docs if doc.page_content])
        print(f"Step 2 - Built context ({len(context)} characters)")
        
        # Step 3: Build prompt
        template = """<s>[INST] Based on the following context, please answer the question:

Context: {context}

Question: {question}

Please provide a detailed answer based on the context above. [/INST]"""
        
        prompt = template.format(context=context, question=test_query)
        print(f"Step 3 - Built prompt ({len(prompt)} characters)")
        
        # Step 4: Call LLM
        headers = {
            'Authorization': f'Bearer {BEARER_TOKEN}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': MODEL_NAME,
            'prompt': prompt,
            'max_tokens': 200,
            'temperature': 0.1,
            'stream': False
        }
        
        response = requests.post(INFERENCE_SERVER_URL, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        if 'choices' in result and len(result['choices']) > 0:
            generated_text = result['choices'][0].get('text', '')
            print(f"Step 4 - Got LLM response ({len(generated_text)} characters)")
            print(f"Response preview: {generated_text[:200]}...")
            
            if generated_text and generated_text.strip():
                print("✓ Full RAG pipeline working!")
                return True
            else:
                print("✗ LLM returned empty response in pipeline")
                return False
        else:
            print("✗ No response from LLM in pipeline")
            return False
            
    except Exception as e:
        print(f"✗ Full pipeline test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("RAG COMPONENT TESTER")
    print("This script tests each component of the RAG application independently")
    print()
    
    tests = [
        ("Environment Variables", test_environment),
        ("LLM Endpoint", test_llm_endpoint),
        ("Database Connection", test_database_connection),
        ("Document Retrieval", test_retrieval),
        ("Full RAG Pipeline", test_full_rag_pipeline)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Your RAG application should be working.")
    else:
        print("❌ Some tests failed. Check the detailed output above.")
        print("\nRecommended actions:")
        
        if not results.get("Environment Variables"):
            print("- Set missing environment variables in .env file")
        if not results.get("LLM Endpoint"):
            print("- Check BEARER_TOKEN and INFERENCE_SERVER_URL")
        if not results.get("Database Connection"):
            print("- Verify database is running and connection string is correct")
        if not results.get("Document Retrieval"):
            print("- Check if documents are properly indexed in the database")
        if not results.get("Full RAG Pipeline"):
            print("- Review individual component failures above")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 