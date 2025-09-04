#!/usr/bin/env python3
"""
MCP (Model Context Protocol) Integration Module
Provides functionality to connect and retrieve data from external sources
"""

import json
import os
import requests
import base64
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MCPDocument:
    """Represents a document retrieved from MCP sources"""
    content: str
    title: str
    source: str
    metadata: Dict[str, Any]
    timestamp: datetime
    doc_type: str  # 'markdown', 'code', 'text', etc.

class MCPBaseConnector(ABC):
    """Base class for MCP connectors"""
    
    def __init__(self, server_config: Dict[str, Any]):
        self.config = server_config
        self.name = server_config.get('name', 'Unknown')
        self.server_type = server_config.get('server_type', '')
        self.api_token = server_config.get('api_token', '')
        self.endpoint_url = server_config.get('endpoint_url', '')
        self.workspace_id = server_config.get('workspace_id', '')
        self.additional_params = server_config.get('additional_params', {})
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Test connection to the MCP server"""
        pass
    
    @abstractmethod
    def retrieve_documents(self, query: Optional[str] = None, limit: int = 10) -> List[MCPDocument]:
        """Retrieve documents from the MCP server"""
        pass
    
    @abstractmethod
    def search_documents(self, query: str, limit: int = 10) -> List[MCPDocument]:
        """Search for documents in the MCP server"""
        pass

class GitHubMCPConnector(MCPBaseConnector):
    """MCP connector for GitHub repositories"""
    
    def __init__(self, server_config: Dict[str, Any]):
        super().__init__(server_config)
        self.base_url = "https://api.github.com"
        self.headers = {
            "Authorization": f"token {self.api_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        self.default_branch = self.additional_params.get('branch', 'main')
    
    def test_connection(self) -> bool:
        """Test GitHub API connection"""
        try:
            response = requests.get(f"{self.base_url}/user", headers=self.headers, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"GitHub connection test failed: {e}")
            return False
    
    def _get_repo_contents(self, repo: str, path: str = "", ref: str = None) -> List[Dict]:
        """Get repository contents"""
        url = f"{self.base_url}/repos/{repo}/contents/{path}"
        params = {"ref": ref or self.default_branch}
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get repo contents: {e}")
            return []
    
    def _get_file_content(self, repo: str, path: str, ref: str = None) -> str:
        """Get file content from repository"""
        url = f"{self.base_url}/repos/{repo}/contents/{path}"
        params = {"ref": ref or self.default_branch}
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            file_data = response.json()
            
            if file_data.get('encoding') == 'base64':
                content = base64.b64decode(file_data['content']).decode('utf-8')
                return content
        except Exception as e:
            logger.error(f"Failed to get file content: {e}")
        return ""
    
    def retrieve_documents(self, query: Optional[str] = None, limit: int = 10) -> List[MCPDocument]:
        """Retrieve documents from GitHub repository"""
        documents = []
        
        if not self.workspace_id:
            logger.warning("No repository specified")
            return documents
        
        try:
            # Get repository README and documentation files
            repo_contents = self._get_repo_contents(self.workspace_id)
            
            # Priority files to include
            priority_files = ['README.md', 'README.rst', 'README.txt', 'CONTRIBUTING.md', 'LICENSE']
            doc_extensions = ['.md', '.rst', '.txt', '.py', '.js', '.ts', '.yaml', '.yml', '.json']
            
            file_count = 0
            for item in repo_contents:
                if file_count >= limit:
                    break
                
                if item['type'] == 'file':
                    filename = item['name']
                    file_path = item['path']
                    
                    # Include priority files or files with documentation extensions
                    should_include = (
                        filename in priority_files or
                        any(filename.endswith(ext) for ext in doc_extensions) or
                        (query and query.lower() in filename.lower())
                    )
                    
                    if should_include:
                        content = self._get_file_content(self.workspace_id, file_path)
                        if content:
                            # Determine document type
                            doc_type = 'markdown' if filename.endswith('.md') else 'code' if any(filename.endswith(ext) for ext in ['.py', '.js', '.ts']) else 'text'
                            
                            doc = MCPDocument(
                                content=content,
                                title=filename,
                                source=f"github://{self.workspace_id}/{file_path}",
                                metadata={
                                    'repository': self.workspace_id,
                                    'file_path': file_path,
                                    'size': item.get('size', 0),
                                    'sha': item.get('sha', ''),
                                    'download_url': item.get('download_url', '')
                                },
                                timestamp=datetime.now(),
                                doc_type=doc_type
                            )
                            documents.append(doc)
                            file_count += 1
        
        except Exception as e:
            logger.error(f"Failed to retrieve GitHub documents: {e}")
        
        return documents
    
    def search_documents(self, query: str, limit: int = 10) -> List[MCPDocument]:
        """Search for documents in GitHub repository"""
        try:
            # Use GitHub search API
            search_url = f"{self.base_url}/search/code"
            params = {
                'q': f"{query} repo:{self.workspace_id}",
                'per_page': limit
            }
            
            response = requests.get(search_url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            search_results = response.json()
            
            documents = []
            for item in search_results.get('items', []):
                content = self._get_file_content(self.workspace_id, item['path'])
                if content:
                    doc = MCPDocument(
                        content=content,
                        title=item['name'],
                        source=f"github://{self.workspace_id}/{item['path']}",
                        metadata={
                            'repository': self.workspace_id,
                            'file_path': item['path'],
                            'score': item.get('score', 0),
                            'html_url': item.get('html_url', '')
                        },
                        timestamp=datetime.now(),
                        doc_type='code' if any(item['name'].endswith(ext) for ext in ['.py', '.js', '.ts']) else 'text'
                    )
                    documents.append(doc)
            
            return documents
        
        except Exception as e:
            logger.error(f"GitHub search failed: {e}")
            return []

class NotionMCPConnector(MCPBaseConnector):
    """MCP connector for Notion workspace"""
    
    def __init__(self, server_config: Dict[str, Any]):
        super().__init__(server_config)
        self.base_url = "https://api.notion.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json"
        }
        self.page_size = self.additional_params.get('page_size', 100)
    
    def test_connection(self) -> bool:
        """Test Notion API connection"""
        try:
            response = requests.get(f"{self.base_url}/users", headers=self.headers, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Notion connection test failed: {e}")
            return False
    
    def _get_page_content(self, page_id: str) -> str:
        """Get page content from Notion"""
        try:
            # Get page blocks
            url = f"{self.base_url}/blocks/{page_id}/children"
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            blocks = response.json().get('results', [])
            content_parts = []
            
            for block in blocks:
                block_type = block.get('type', '')
                if block_type in ['paragraph', 'heading_1', 'heading_2', 'heading_3']:
                    text_content = self._extract_text_from_block(block)
                    if text_content:
                        content_parts.append(text_content)
            
            return '\n\n'.join(content_parts)
        
        except Exception as e:
            logger.error(f"Failed to get Notion page content: {e}")
            return ""
    
    def _extract_text_from_block(self, block: Dict) -> str:
        """Extract text content from a Notion block"""
        block_type = block.get('type', '')
        if block_type in ['paragraph', 'heading_1', 'heading_2', 'heading_3']:
            rich_text = block.get(block_type, {}).get('rich_text', [])
            return ''.join([text.get('plain_text', '') for text in rich_text])
        return ""
    
    def retrieve_documents(self, query: Optional[str] = None, limit: int = 10) -> List[MCPDocument]:
        """Retrieve documents from Notion workspace"""
        documents = []
        
        try:
            # Search for pages in the workspace
            search_url = f"{self.base_url}/search"
            payload = {
                "page_size": min(limit, self.page_size),
                "filter": {"property": "object", "value": "page"}
            }
            
            if query:
                payload["query"] = query
            
            response = requests.post(search_url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            
            results = response.json().get('results', [])
            
            for page in results:
                page_id = page.get('id', '')
                
                # Get page title
                title_property = page.get('properties', {}).get('title', {})
                title = ''
                if title_property.get('title'):
                    title = ''.join([t.get('plain_text', '') for t in title_property['title']])
                
                # Get page content
                content = self._get_page_content(page_id)
                
                if content:
                    doc = MCPDocument(
                        content=content,
                        title=title or 'Untitled',
                        source=f"notion://{page_id}",
                        metadata={
                            'page_id': page_id,
                            'url': page.get('url', ''),
                            'created_time': page.get('created_time', ''),
                            'last_edited_time': page.get('last_edited_time', '')
                        },
                        timestamp=datetime.now(),
                        doc_type='markdown'
                    )
                    documents.append(doc)
        
        except Exception as e:
            logger.error(f"Failed to retrieve Notion documents: {e}")
        
        return documents
    
    def search_documents(self, query: str, limit: int = 10) -> List[MCPDocument]:
        """Search for documents in Notion workspace"""
        return self.retrieve_documents(query=query, limit=limit)

class FileSystemMCPConnector(MCPBaseConnector):
    """MCP connector for local file system"""
    
    def __init__(self, server_config: Dict[str, Any]):
        super().__init__(server_config)
        self.root_path = self.workspace_id
    
    def test_connection(self) -> bool:
        """Test file system access"""
        try:
            return os.path.exists(self.root_path) and os.path.isdir(self.root_path)
        except Exception as e:
            logger.error(f"File system connection test failed: {e}")
            return False
    
    def retrieve_documents(self, query: Optional[str] = None, limit: int = 10) -> List[MCPDocument]:
        """Retrieve documents from file system"""
        documents = []
        
        if not os.path.exists(self.root_path):
            logger.warning(f"Path does not exist: {self.root_path}")
            return documents
        
        try:
            doc_extensions = ['.md', '.txt', '.rst', '.py', '.js', '.ts', '.json', '.yaml', '.yml']
            file_count = 0
            
            for root, dirs, files in os.walk(self.root_path):
                if file_count >= limit:
                    break
                
                for filename in files:
                    if file_count >= limit:
                        break
                    
                    file_path = os.path.join(root, filename)
                    
                    # Check if file matches query or has valid extension
                    should_include = (
                        any(filename.endswith(ext) for ext in doc_extensions) and
                        (not query or query.lower() in filename.lower())
                    )
                    
                    if should_include:
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            
                            # Determine document type
                            doc_type = 'markdown' if filename.endswith('.md') else 'code' if any(filename.endswith(ext) for ext in ['.py', '.js', '.ts']) else 'text'
                            
                            doc = MCPDocument(
                                content=content,
                                title=filename,
                                source=f"file://{file_path}",
                                metadata={
                                    'file_path': file_path,
                                    'size': os.path.getsize(file_path),
                                    'modified_time': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                                },
                                timestamp=datetime.now(),
                                doc_type=doc_type
                            )
                            documents.append(doc)
                            file_count += 1
                        
                        except Exception as e:
                            logger.warning(f"Failed to read file {file_path}: {e}")
        
        except Exception as e:
            logger.error(f"Failed to retrieve file system documents: {e}")
        
        return documents
    
    def search_documents(self, query: str, limit: int = 10) -> List[MCPDocument]:
        """Search for documents in file system"""
        return self.retrieve_documents(query=query, limit=limit)

class MCPManager:
    """Manages multiple MCP connections and provides unified interface"""
    
    def __init__(self, config_file: str = "configs/mcp_servers.json"):
        self.config_file = config_file
        self.connectors: Dict[str, MCPBaseConnector] = {}
        self._load_configurations()
    
    def _load_configurations(self):
        """Load MCP server configurations"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    configs = json.load(f)
                
                for config in configs:
                    self.add_connector(config)
        except Exception as e:
            logger.error(f"Failed to load MCP configurations: {e}")
    
    def add_connector(self, server_config: Dict[str, Any]):
        """Add a new MCP connector"""
        server_type = server_config.get('server_type', '')
        server_name = server_config.get('name', '')
        
        try:
            if server_type == 'github':
                connector = GitHubMCPConnector(server_config)
            elif server_type == 'notion':
                connector = NotionMCPConnector(server_config)
            elif server_type == 'filesystem':
                connector = FileSystemMCPConnector(server_config)
            else:
                logger.warning(f"Unsupported server type: {server_type}")
                return
            
            self.connectors[server_name] = connector
            logger.info(f"Added MCP connector: {server_name} ({server_type})")
        
        except Exception as e:
            logger.error(f"Failed to add MCP connector {server_name}: {e}")
    
    def test_all_connections(self) -> Dict[str, bool]:
        """Test all MCP connections"""
        results = {}
        for name, connector in self.connectors.items():
            try:
                results[name] = connector.test_connection()
            except Exception as e:
                logger.error(f"Connection test failed for {name}: {e}")
                results[name] = False
        return results
    
    def retrieve_all_documents(self, query: Optional[str] = None, limit_per_source: int = 5) -> List[MCPDocument]:
        """Retrieve documents from all configured MCP sources"""
        all_documents = []
        
        for name, connector in self.connectors.items():
            try:
                documents = connector.retrieve_documents(query=query, limit=limit_per_source)
                all_documents.extend(documents)
                logger.info(f"Retrieved {len(documents)} documents from {name}")
            except Exception as e:
                logger.error(f"Failed to retrieve documents from {name}: {e}")
        
        return all_documents
    
    def search_all_sources(self, query: str, limit_per_source: int = 5) -> List[MCPDocument]:
        """Search across all MCP sources"""
        all_documents = []
        
        for name, connector in self.connectors.items():
            try:
                documents = connector.search_documents(query=query, limit=limit_per_source)
                all_documents.extend(documents)
                logger.info(f"Found {len(documents)} documents in {name} for query: {query}")
            except Exception as e:
                logger.error(f"Search failed in {name}: {e}")
        
        return all_documents
    
    def get_connector(self, name: str) -> Optional[MCPBaseConnector]:
        """Get a specific MCP connector by name"""
        return self.connectors.get(name)
    
    def list_connectors(self) -> List[str]:
        """List all configured connector names"""
        return list(self.connectors.keys())

# Example usage and testing functions
def example_usage():
    """Example of how to use the MCP integration"""
    
    # Initialize MCP manager
    mcp_manager = MCPManager()
    
    # Test all connections
    print("Testing MCP connections...")
    connection_results = mcp_manager.test_all_connections()
    for name, status in connection_results.items():
        print(f"  {name}: {'✅ Connected' if status else '❌ Failed'}")
    
    # Retrieve documents from all sources
    print("\nRetrieving documents from all sources...")
    documents = mcp_manager.retrieve_all_documents(limit_per_source=3)
    
    print(f"Retrieved {len(documents)} total documents:")
    for doc in documents[:5]:  # Show first 5
        print(f"  - {doc.title} ({doc.doc_type}) from {doc.source}")
    
    # Search across all sources
    print("\nSearching for 'API' across all sources...")
    search_results = mcp_manager.search_all_sources("API", limit_per_source=2)
    
    print(f"Found {len(search_results)} documents:")
    for doc in search_results:
        print(f"  - {doc.title} from {doc.source}")

if __name__ == "__main__":
    example_usage() 