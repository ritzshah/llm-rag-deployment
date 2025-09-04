# Enhanced LLM Pipeline with UI Configuration & MCP Integration

This enhanced version of the LLM pipeline provides a user-friendly web interface for configuring inference endpoints and integrates with external data sources through MCP (Model Context Protocol) connections.

## 🚀 Key Features

### 🤖 Configurable LLM Endpoints
- **Web UI Configuration**: Configure inference server URLs and API tokens through a Streamlit interface
- **Multiple LLM Support**: Compatible with HuggingFace Text Generation Inference and OpenAI-compatible APIs
- **Real-time Testing**: Test connections and validate configurations before use
- **Parameter Tuning**: Adjust temperature, top-k, top-p, and other generation parameters

### 🔌 MCP (Model Context Protocol) Integration
- **External Data Sources**: Connect to GitHub repositories, Notion workspaces, file systems, and web sources
- **Automated Context Enrichment**: Automatically retrieve relevant context from external sources during inference
- **Multiple Connector Support**: Support for multiple simultaneous data source connections
- **Source Testing**: Validate connections to external services

### 📊 Enhanced Testing & Quality Assurance
- **Comprehensive Test Suite**: Multiple test types including basic, enhanced, and MCP-specific tests
- **Quality Metrics**: Similarity scoring and context-aware quality assessment
- **Detailed Reporting**: Rich test results with metadata and performance statistics

## 🛠️ Installation

### 1. Install Dependencies

```bash
pip install -r enhanced_requirements.txt
```

### 2. Directory Structure

The enhanced pipeline creates this structure:
```
llm-rag-deployment/examples/pipelines/
├── config_ui.py                    # Web UI for configuration
├── mcp_integration.py              # MCP connector implementations
├── enhanced_llm_usage.py           # Enhanced LLM interface
├── enhanced_test_response_quality.py  # Comprehensive testing
├── enhanced_requirements.txt       # Dependencies
├── configs/                       # Configuration storage
│   ├── llm_config.json           # LLM configuration
│   └── mcp_servers.json          # MCP server configurations
└── ENHANCED_README.md             # This file
```

## 🚀 Quick Start

### 1. Launch the Configuration UI

```bash
streamlit run config_ui.py
```

This opens a web interface at `http://localhost:8501` where you can:

- **Configure LLM Settings**: Set inference server URL, API token, and generation parameters
- **Add MCP Servers**: Connect to GitHub, Notion, file systems, or web sources
- **Test Connections**: Validate all configurations before use
- **Export Configurations**: Generate environment variables or YAML files

### 2. Configure Your LLM Endpoint

In the **🤖 LLM Config** section:

1. **Inference Server URL**: Enter your LLM endpoint (e.g., `http://your-llm-server:3000`)
2. **API Token**: Provide authentication token if required
3. **Model Parameters**: Adjust temperature, tokens, etc.
4. **Test Connection**: Click to verify your settings

Example configurations:
- **HuggingFace TGI**: `http://llm.ic-shared-llm.svc.cluster.local:3000`
- **OpenAI Compatible**: `https://api.openai.com/v1`
- **Local Ollama**: `http://localhost:11434/v1`

### 3. Add External Data Sources

In the **🔌 MCP Servers** section:

#### GitHub Repository
- **Server Type**: GitHub
- **API Token**: Your GitHub personal access token
- **Workspace ID**: Repository in format `owner/repo`

#### Notion Workspace
- **Server Type**: Notion
- **API Token**: Your Notion integration token
- **Workspace ID**: Notion workspace/database ID

#### File System
- **Server Type**: File System
- **Workspace ID**: Local directory path

#### Web Scraper
- **Server Type**: Web
- **Endpoint URL**: Target website URL

### 4. Run Enhanced Tests

```bash
python enhanced_test_response_quality.py
```

This runs a comprehensive test suite including:
- Configuration validation
- MCP source testing
- Basic response quality
- Enhanced response quality with external context

## 📋 Usage Examples

### Basic Usage (Backward Compatible)

```python
from enhanced_llm_usage import infer_with_template, similarity_metric

# Simple inference (loads config automatically)
response = infer_with_template(
    input_text="What are the benefits of prompt engineering?",
    template="Answer this question: {question}"
)
print(response)
```

### Enhanced Usage with MCP

```python
from enhanced_llm_usage import enhanced_infer_with_context

# Enhanced inference with external sources
result = enhanced_infer_with_context(
    input_text="Explain RAG pipelines",
    template="""
    Based on the context from external sources: {context}
    
    Question: {question}
    
    Provide a comprehensive answer using both your knowledge and the external context.
    """,
    external_sources=True
)

print(f"Response: {result['response']}")
print(f"Sources used: {len(result['context_info']['mcp_sources_used'])}")
```

### Manual Configuration

```python
from enhanced_llm_usage import ConfigurableLLMManager

# Create manager with custom config
llm_manager = ConfigurableLLMManager()

# Update configuration
llm_manager.update_config({
    "server_url": "http://your-custom-endpoint:8000",
    "api_token": "your-token",
    "temperature": 0.7
})

# Use the configured LLM
llm = llm_manager.get_llm()
```

## 🔧 Configuration Options

### LLM Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `server_url` | LLM inference endpoint | `http://llm.ic-shared-llm.svc.cluster.local:3000` |
| `api_token` | Authentication token | `""` |
| `model_name` | Model identifier | `"default"` |
| `max_new_tokens` | Maximum tokens to generate | `512` |
| `temperature` | Randomness control (0-2) | `0.01` |
| `top_k` | Top-k sampling | `10` |
| `top_p` | Nucleus sampling | `0.95` |
| `typical_p` | Typical sampling | `0.95` |
| `repetition_penalty` | Repetition penalty | `1.03` |
| `streaming` | Enable streaming | `true` |

### Environment Variables

You can also configure via environment variables:

```bash
export INFERENCE_SERVER_URL="http://your-llm-server:3000"
export API_TOKEN="your-token"
export MODEL_NAME="your-model"
export TEMPERATURE="0.7"
export MAX_NEW_TOKENS="1024"
```

## 🔌 MCP Connector Details

### GitHub Connector

**Features:**
- Repository content retrieval
- Code search across files
- README and documentation prioritization
- Branch specification support

**Configuration:**
```json
{
  "name": "My GitHub Repo",
  "server_type": "github",
  "endpoint_url": "https://api.github.com",
  "api_token": "ghp_your_token",
  "workspace_id": "owner/repository",
  "additional_params": {
    "branch": "main",
    "include_private": false
  }
}
```

### Notion Connector

**Features:**
- Page content extraction
- Search across workspace
- Block content parsing
- Structured data retrieval

**Required Permissions:**
- Read content
- Search workspace

### File System Connector

**Features:**
- Local file access
- Document type detection
- Recursive directory scanning
- Content filtering

**Supported File Types:**
- Markdown (`.md`)
- Text files (`.txt`, `.rst`)
- Code files (`.py`, `.js`, `.ts`)
- Configuration (`.yaml`, `.json`)

### Web Connector

**Features:**
- Web page scraping
- Content extraction
- Configurable crawl depth
- robots.txt respect

## 🧪 Testing & Quality Assurance

### Test Types

1. **Configuration Test**: Validates LLM and MCP configurations
2. **MCP Sources Test**: Tests connectivity and document retrieval from all sources
3. **Basic Quality Test**: Tests LLM response quality without external context
4. **Enhanced Quality Test**: Tests response quality with MCP-enhanced context

### Quality Metrics

- **Similarity Score**: Semantic similarity to expected response
- **Context Bonus**: Additional scoring for external context usage
- **Source Coverage**: Number of external sources contributing to response

### Running Tests

```bash
# Full test suite
python enhanced_test_response_quality.py

# Test individual components
python enhanced_llm_usage.py  # Test LLM and MCP integration
python mcp_integration.py     # Test MCP connectors only
```

## 🔒 Security Considerations

### API Token Security
- Store tokens in configuration files (not committed to version control)
- Use environment variables for production deployments
- Regularly rotate API tokens

### Network Security
- Use HTTPS endpoints when possible
- Configure firewalls for internal LLM endpoints
- Validate SSL certificates

### Data Privacy
- Review external source access permissions
- Consider data retention policies for retrieved content
- Implement audit logging for sensitive operations

## 🚨 Troubleshooting

### Common Issues

**LLM Connection Failed**
```bash
❌ Failed to initialize LLM: Connection timeout
```
- Check if the inference server is running
- Verify the endpoint URL format
- Test network connectivity
- Validate API token if required

**MCP Source Connection Failed**
```bash
❌ GitHub connection test failed: 401 Unauthorized
```
- Verify API token permissions
- Check token expiration
- Ensure correct repository access rights

**Import Errors**
```bash
ModuleNotFoundError: No module named 'streamlit'
```
- Install dependencies: `pip install -r enhanced_requirements.txt`
- Check Python version compatibility (3.8+)

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 Advanced Usage

### Custom MCP Connectors

Create custom connectors by extending `MCPBaseConnector`:

```python
from mcp_integration import MCPBaseConnector, MCPDocument

class CustomConnector(MCPBaseConnector):
    def test_connection(self) -> bool:
        # Implement connection test
        pass
    
    def retrieve_documents(self, query=None, limit=10):
        # Implement document retrieval
        pass
    
    def search_documents(self, query, limit=10):
        # Implement search functionality
        pass
```

### Pipeline Integration

Integrate with existing pipelines:

```python
from enhanced_llm_usage import EnhancedRAGPipeline, ConfigurableLLMManager

# Use in Elyra pipeline
llm_manager = ConfigurableLLMManager("configs/llm_config.json")
rag_pipeline = EnhancedRAGPipeline(llm_manager)

# Create custom chain
custom_chain = rag_pipeline.create_rag_chain_with_mcp(
    template=your_template,
    use_mcp=True,
    mcp_limit_per_source=5
)
```

## 🤝 Contributing

### Adding New MCP Connectors

1. Implement the `MCPBaseConnector` interface
2. Add connector registration in `MCPManager`
3. Update the UI template options in `config_ui.py`
4. Add tests for the new connector

### Improving the UI

1. Extend the Streamlit interface in `config_ui.py`
2. Add new configuration sections
3. Implement additional export formats

## 📄 License

This enhanced pipeline maintains the same license as the original project.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the test output for specific error messages
3. Verify all dependencies are correctly installed
4. Ensure proper network connectivity to external services

---

**Note**: This enhanced pipeline is backward compatible with the original `test_response_quality.py` script, so existing workflows will continue to function while gaining access to the new capabilities. 