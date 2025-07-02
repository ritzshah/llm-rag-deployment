# RAG Application Debugging Guide

## Overview
This guide helps you troubleshoot issues with the RAG (Retrieval-Augmented Generation) application, particularly when documents are retrieved but the LLM doesn't generate detailed responses.

## Quick Start

### Environment Variables
Set these environment variables in your `.env` file:

```bash
# Required
BEARER_TOKEN=your_bearer_token_here
INFERENCE_SERVER_URL=https://your-model-endpoint.com/v1/completions
MODEL_NAME=granite-8b-code-instruct-128k
DB_CONNECTION_STRING=postgresql+psycopg://vectordb:vectordb@localhost:5432/vectordb
DB_COLLECTION_NAME=langchain_pg_collection

# Optional - Debugging
DEBUG=true
SHOW_DEBUG_UI=true

# Optional - Model Parameters
MAX_NEW_TOKENS=512
TEMPERATURE=0.01
TOP_P=0.95
TOP_K=10
REPETITION_PENALTY=1.03
```

### Installation
```bash
pip install -r requirements-clean.txt
```

### Running with Debug Mode
```bash
DEBUG=true SHOW_DEBUG_UI=true python with-pgvector-bearer.py
```

## Common Issues & Solutions

### 1. Documents Retrieved but No LLM Response

**Symptoms:**
- You see "Sources:" listed
- But no actual answer from the LLM
- May see "I found relevant information but couldn't generate a response"

**Debugging Steps:**

1. **Enable Debug Mode:**
   ```bash
   export DEBUG=true
   export SHOW_DEBUG_UI=true
   ```

2. **Check the Debug Panel:**
   - Use the debug controls in the right sidebar
   - Click "Test LLM Only" to test the model directly
   - Click "Test Retrieval Only" to test document retrieval

3. **Check Console Logs:**
   Look for these debug messages:
   ```
   === EMBEDDING DEBUG ===
   Query embedding dimension: 384
   
   === RETRIEVAL DEBUG ===
   Retrieved X documents with threshold 0.5
   
   === LLM DIRECT TEST ===
   LLM response length: 0
   ```

4. **Common Root Causes:**

   **a) Invalid Bearer Token:**
   ```
   ERROR: Authentication failed - Invalid bearer token
   ```
   - **Solution:** Check your `BEARER_TOKEN` in the environment variables
   - Verify the token is valid and not expired

   **b) Wrong Model Endpoint:**
   ```
   ERROR: Model endpoint not found - Check INFERENCE_SERVER_URL
   ```
   - **Solution:** Verify the `INFERENCE_SERVER_URL` is correct
   - Ensure the endpoint supports the `/v1/completions` API

   **c) Model Configuration Issues:**
   ```
   ERROR: Model returned empty response - Check model configuration
   ```
   - **Solution:** Try reducing `MAX_NEW_TOKENS` or adjusting `TEMPERATURE`
   - Check if the model name is correct

   **d) Rate Limiting:**
   ```
   ERROR: Rate limit exceeded - Try again later
   ```
   - **Solution:** Wait a few minutes and try again
   - Consider reducing request frequency

### 2. No Documents Retrieved

**Symptoms:**
- "No specific sources found for this query"
- Empty retrieval results

**Debugging Steps:**

1. **Test with Lower Similarity Threshold:**
   - The app tests with both 0.5 and 0.1 thresholds
   - Check debug output for both results

2. **Check Database Connection:**
   ```bash
   # Test database connectivity
   psql "$DB_CONNECTION_STRING" -c "SELECT COUNT(*) FROM langchain_pg_collection;"
   ```

3. **Verify Embeddings:**
   - Check if query embedding has 384 dimensions
   - Ensure the embedding model is working

### 3. Context Too Long or Too Short

**Symptoms:**
- Context length shown in debug info is 0 or very large
- LLM response is truncated or irrelevant

**Solutions:**
- Adjust `MAX_NEW_TOKENS` (try 256, 512, 1024)
- Modify retrieval parameters in the code:
  ```python
  search_kwargs={"k": 4, "score_threshold": 0.5}
  ```

## Debug Features

### 1. Console Debug Logs
Enable with `DEBUG=true`:
- Shows detailed request/response information
- Displays embedding dimensions and similarity scores
- Logs all API calls and responses

### 2. UI Debug Panel
Enable with `SHOW_DEBUG_UI=true`:
- Shows debug information in chat responses
- Provides component testing buttons
- Displays system configuration

### 3. Test Functions

**Test LLM Only:**
- Tests the LLM with a simple prompt
- Bypasses retrieval completely
- Helps isolate LLM issues

**Test Retrieval Only:**
- Tests document retrieval without LLM
- Shows similarity scores and document content
- Helps isolate retrieval issues

## Advanced Debugging

### 1. Manual Testing

Test the LLM endpoint directly:
```bash
curl -X POST "$INFERENCE_SERVER_URL" \
  -H "Authorization: Bearer $BEARER_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "granite-8b-code-instruct-128k",
    "prompt": "What is OpenShift?",
    "max_tokens": 100,
    "temperature": 0.1
  }'
```

### 2. Database Queries

Check document storage:
```sql
-- Count documents
SELECT COUNT(*) FROM langchain_pg_collection;

-- Check document samples
SELECT document, metadata FROM langchain_pg_collection LIMIT 5;

-- Test similarity search
SELECT document, metadata, embedding <-> '[0.1,0.2,...]'::vector as distance 
FROM langchain_pg_collection 
ORDER BY distance LIMIT 5;
```

### 3. Code Modifications

**Adjust Similarity Threshold:**
```python
retriever = store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 6, "score_threshold": 0.3}  # Lower threshold
)
```

**Modify Prompt Template:**
```python
template = """[INST] 
Answer this question based on the context provided.
Question: {question}
Context: {context}
[/INST]"""
```

## Monitoring

### Key Metrics to Watch:
- **Query Embedding Length:** Should be 384 for default HuggingFace embeddings
- **Retrieved Documents:** Should be > 0 for relevant queries
- **Context Length:** Should be reasonable (not 0, not too large)
- **LLM Response Length:** Should be > 0 for successful responses
- **Response Time:** Should be < 30 seconds typically

### Log Analysis:
Look for these patterns in logs:
- `SUCCESS: Got LLM response directly` - Good
- `ISSUE: Empty LLM response` - LLM problem
- `ISSUE: No context retrieved` - Retrieval problem
- `ERROR:` prefixed messages - Check the specific error

## Getting Help

If you're still having issues:

1. Enable both `DEBUG=true` and `SHOW_DEBUG_UI=true`
2. Try a simple question like "What is OpenShift?"
3. Use the test buttons in the debug panel
4. Check the console logs for specific error messages
5. Share the debug output when asking for help

## Common Environment Variable Issues

```bash
# Check if variables are set
echo $BEARER_TOKEN
echo $INFERENCE_SERVER_URL
echo $DB_CONNECTION_STRING

# Set them if missing
export BEARER_TOKEN="your_token_here"
export INFERENCE_SERVER_URL="https://your-endpoint.com/v1/completions"
export DB_CONNECTION_STRING="postgresql+psycopg://user:pass@host:port/db"
```

Remember: The most common issue is authentication - make sure your bearer token is valid and has the correct permissions for the model endpoint. 