#!/usr/bin/env python3
"""
Enhanced LLM Configuration UI with MCP Integration
Provides a web interface for configuring LLM endpoints and external data sources
"""

import streamlit as st
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import requests
from dataclasses import dataclass, asdict
import yaml

@dataclass
class LLMConfig:
    """Configuration for LLM inference server"""
    server_url: str
    api_token: str
    model_name: str = "default"
    max_new_tokens: int = 512
    temperature: float = 0.01
    top_k: int = 10
    top_p: float = 0.95
    typical_p: float = 0.95
    repetition_penalty: float = 1.03
    streaming: bool = True

@dataclass
class MCPServerConfig:
    """Configuration for MCP (Model Context Protocol) servers"""
    name: str
    server_type: str  # github, notion, filesystem, web, etc.
    endpoint_url: str = ""
    api_token: str = ""
    workspace_id: str = ""
    additional_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_params is None:
            self.additional_params = {}

class ConfigurationManager:
    """Manages saving and loading configurations"""
    
    CONFIG_DIR = "configs"
    LLM_CONFIG_FILE = "llm_config.json"
    MCP_CONFIG_FILE = "mcp_servers.json"
    
    def __init__(self):
        os.makedirs(self.CONFIG_DIR, exist_ok=True)
    
    def save_llm_config(self, config: LLMConfig) -> bool:
        """Save LLM configuration to file"""
        try:
            config_path = os.path.join(self.CONFIG_DIR, self.LLM_CONFIG_FILE)
            with open(config_path, 'w') as f:
                json.dump(asdict(config), f, indent=2)
            return True
        except Exception as e:
            st.error(f"Failed to save LLM config: {e}")
            return False
    
    def load_llm_config(self) -> Optional[LLMConfig]:
        """Load LLM configuration from file"""
        try:
            config_path = os.path.join(self.CONFIG_DIR, self.LLM_CONFIG_FILE)
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    data = json.load(f)
                return LLMConfig(**data)
        except Exception as e:
            st.error(f"Failed to load LLM config: {e}")
        return None
    
    def save_mcp_servers(self, servers: List[MCPServerConfig]) -> bool:
        """Save MCP server configurations"""
        try:
            config_path = os.path.join(self.CONFIG_DIR, self.MCP_CONFIG_FILE)
            with open(config_path, 'w') as f:
                json.dump([asdict(server) for server in servers], f, indent=2)
            return True
        except Exception as e:
            st.error(f"Failed to save MCP configs: {e}")
            return False
    
    def load_mcp_servers(self) -> List[MCPServerConfig]:
        """Load MCP server configurations"""
        try:
            config_path = os.path.join(self.CONFIG_DIR, self.MCP_CONFIG_FILE)
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    data = json.load(f)
                return [MCPServerConfig(**server) for server in data]
        except Exception as e:
            st.error(f"Failed to load MCP configs: {e}")
        return []

def test_llm_connection(config: LLMConfig) -> bool:
    """Test connection to LLM inference server"""
    try:
        # Basic health check - adjust based on your API
        test_url = f"{config.server_url.rstrip('/')}/health"
        headers = {"Authorization": f"Bearer {config.api_token}"} if config.api_token else {}
        
        response = requests.get(test_url, headers=headers, timeout=10)
        return response.status_code == 200
    except Exception as e:
        st.error(f"Connection test failed: {e}")
        return False

def test_mcp_connection(server: MCPServerConfig) -> bool:
    """Test connection to MCP server"""
    try:
        if server.server_type == "github":
            # Test GitHub API access
            headers = {"Authorization": f"token {server.api_token}"} if server.api_token else {}
            response = requests.get("https://api.github.com/user", headers=headers, timeout=10)
            return response.status_code == 200
        elif server.server_type == "notion":
            # Test Notion API access
            headers = {
                "Authorization": f"Bearer {server.api_token}",
                "Notion-Version": "2022-06-28"
            } if server.api_token else {}
            response = requests.get("https://api.notion.com/v1/users", headers=headers, timeout=10)
            return response.status_code == 200
        # Add more server types as needed
        return True
    except Exception as e:
        st.error(f"MCP connection test failed: {e}")
        return False

def render_llm_config_section():
    """Render LLM configuration section"""
    st.header("🤖 LLM Inference Configuration")
    
    config_manager = ConfigurationManager()
    
    # Load existing config or create new
    existing_config = config_manager.load_llm_config()
    
    with st.form("llm_config_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            server_url = st.text_input(
                "Inference Server URL",
                value=existing_config.server_url if existing_config else "http://llm.ic-shared-llm.svc.cluster.local:3000",
                help="URL of your LLM inference server"
            )
            
            api_token = st.text_input(
                "API Token",
                value=existing_config.api_token if existing_config else "",
                type="password",
                help="Authentication token for the inference server"
            )
            
            model_name = st.text_input(
                "Model Name",
                value=existing_config.model_name if existing_config else "default",
                help="Name or identifier of the model to use"
            )
        
        with col2:
            max_new_tokens = st.number_input(
                "Max New Tokens",
                min_value=1,
                max_value=4096,
                value=existing_config.max_new_tokens if existing_config else 512
            )
            
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=existing_config.temperature if existing_config else 0.01,
                step=0.01,
                help="Controls randomness in generation"
            )
            
            top_k = st.number_input(
                "Top K",
                min_value=1,
                max_value=100,
                value=existing_config.top_k if existing_config else 10
            )
            
            top_p = st.slider(
                "Top P",
                min_value=0.0,
                max_value=1.0,
                value=existing_config.top_p if existing_config else 0.95,
                step=0.01
            )
        
        col3, col4 = st.columns(2)
        with col3:
            typical_p = st.slider(
                "Typical P",
                min_value=0.0,
                max_value=1.0,
                value=existing_config.typical_p if existing_config else 0.95,
                step=0.01
            )
        
        with col4:
            repetition_penalty = st.slider(
                "Repetition Penalty",
                min_value=1.0,
                max_value=2.0,
                value=existing_config.repetition_penalty if existing_config else 1.03,
                step=0.01
            )
        
        streaming = st.checkbox(
            "Enable Streaming",
            value=existing_config.streaming if existing_config else True
        )
        
        submitted = st.form_submit_button("Save LLM Configuration")
        
        if submitted:
            new_config = LLMConfig(
                server_url=server_url,
                api_token=api_token,
                model_name=model_name,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                typical_p=typical_p,
                repetition_penalty=repetition_penalty,
                streaming=streaming
            )
            
            if config_manager.save_llm_config(new_config):
                st.success("✅ LLM configuration saved successfully!")
                
                # Test connection
                with st.spinner("Testing connection..."):
                    if test_llm_connection(new_config):
                        st.success("🔗 Connection test successful!")
                    else:
                        st.warning("⚠️ Connection test failed. Please verify your settings.")

def render_mcp_config_section():
    """Render MCP (Model Context Protocol) configuration section"""
    st.header("🔌 MCP Server Configuration")
    st.markdown("Configure external data sources like GitHub, Notion, etc.")
    
    config_manager = ConfigurationManager()
    
    # Load existing MCP servers
    if 'mcp_servers' not in st.session_state:
        st.session_state.mcp_servers = config_manager.load_mcp_servers()
    
    # Server type templates
    server_templates = {
        "github": {
            "name": "GitHub",
            "endpoint_url": "https://api.github.com",
            "required_fields": ["api_token"],
            "optional_fields": ["workspace_id"]
        },
        "notion": {
            "name": "Notion",
            "endpoint_url": "https://api.notion.com/v1",
            "required_fields": ["api_token", "workspace_id"],
            "optional_fields": []
        },
        "filesystem": {
            "name": "File System",
            "endpoint_url": "file://",
            "required_fields": ["workspace_id"],
            "optional_fields": []
        },
        "web": {
            "name": "Web Scraper",
            "endpoint_url": "",
            "required_fields": ["endpoint_url"],
            "optional_fields": ["api_token"]
        }
    }
    
    # Add new MCP server form
    with st.expander("➕ Add New MCP Server", expanded=False):
        with st.form("add_mcp_server"):
            col1, col2 = st.columns(2)
            
            with col1:
                server_type = st.selectbox(
                    "Server Type",
                    options=list(server_templates.keys()),
                    format_func=lambda x: server_templates[x]["name"]
                )
                
                server_name = st.text_input(
                    "Server Name",
                    value=f"My {server_templates[server_type]['name']} Server"
                )
            
            with col2:
                endpoint_url = st.text_input(
                    "Endpoint URL",
                    value=server_templates[server_type]["endpoint_url"]
                )
                
                api_token = st.text_input(
                    "API Token",
                    type="password",
                    help="Authentication token (if required)"
                )
            
            workspace_id = st.text_input(
                "Workspace/Repository ID",
                help="GitHub repo, Notion workspace, or file path"
            )
            
            # Additional parameters
            st.subheader("Additional Parameters")
            additional_params = {}
            
            if server_type == "github":
                additional_params["branch"] = st.text_input("Default Branch", value="main")
                additional_params["include_private"] = st.checkbox("Include Private Repos")
            elif server_type == "notion":
                additional_params["page_size"] = st.number_input("Page Size", value=100, min_value=1, max_value=1000)
            elif server_type == "web":
                additional_params["max_depth"] = st.number_input("Max Crawl Depth", value=2, min_value=1, max_value=10)
                additional_params["respect_robots_txt"] = st.checkbox("Respect robots.txt", value=True)
            
            if st.form_submit_button("Add MCP Server"):
                new_server = MCPServerConfig(
                    name=server_name,
                    server_type=server_type,
                    endpoint_url=endpoint_url,
                    api_token=api_token,
                    workspace_id=workspace_id,
                    additional_params=additional_params
                )
                
                st.session_state.mcp_servers.append(new_server)
                st.success(f"✅ Added {server_name}")
                st.rerun()
    
    # Display existing servers
    if st.session_state.mcp_servers:
        st.subheader("📋 Configured MCP Servers")
        
        for i, server in enumerate(st.session_state.mcp_servers):
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
                
                with col1:
                    st.write(f"**{server.name}**")
                    st.caption(f"Type: {server.server_type} | Workspace: {server.workspace_id}")
                
                with col2:
                    st.write(f"Endpoint: {server.endpoint_url}")
                
                with col3:
                    if st.button("🧪 Test", key=f"test_{i}"):
                        with st.spinner("Testing..."):
                            if test_mcp_connection(server):
                                st.success("✅ Connected")
                            else:
                                st.error("❌ Failed")
                
                with col4:
                    if st.button("🗑️ Remove", key=f"remove_{i}"):
                        st.session_state.mcp_servers.pop(i)
                        st.rerun()
                
                st.divider()
        
        # Save all servers
        if st.button("💾 Save All MCP Configurations"):
            if config_manager.save_mcp_servers(st.session_state.mcp_servers):
                st.success("✅ All MCP configurations saved!")
    else:
        st.info("No MCP servers configured. Add one above to get started.")

def render_export_section():
    """Render configuration export section"""
    st.header("📤 Export Configuration")
    
    config_manager = ConfigurationManager()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📋 Export as Environment Variables"):
            llm_config = config_manager.load_llm_config()
            mcp_servers = config_manager.load_mcp_servers()
            
            if llm_config:
                env_vars = f"""# LLM Configuration
export INFERENCE_SERVER_URL="{llm_config.server_url}"
export API_TOKEN="{llm_config.api_token}"
export MODEL_NAME="{llm_config.model_name}"
export MAX_NEW_TOKENS={llm_config.max_new_tokens}
export TEMPERATURE={llm_config.temperature}
export TOP_K={llm_config.top_k}
export TOP_P={llm_config.top_p}
export TYPICAL_P={llm_config.typical_p}
export REPETITION_PENALTY={llm_config.repetition_penalty}
export STREAMING={str(llm_config.streaming).lower()}

# MCP Servers
export MCP_SERVERS_COUNT={len(mcp_servers)}
"""
                
                for i, server in enumerate(mcp_servers):
                    env_vars += f"""
export MCP_SERVER_{i}_NAME="{server.name}"
export MCP_SERVER_{i}_TYPE="{server.server_type}"
export MCP_SERVER_{i}_URL="{server.endpoint_url}"
export MCP_SERVER_{i}_TOKEN="{server.api_token}"
export MCP_SERVER_{i}_WORKSPACE="{server.workspace_id}"
"""
                
                st.text_area("Environment Variables", env_vars, height=300)
    
    with col2:
        if st.button("📄 Export as YAML"):
            llm_config = config_manager.load_llm_config()
            mcp_servers = config_manager.load_mcp_servers()
            
            config_dict = {
                "llm": asdict(llm_config) if llm_config else {},
                "mcp_servers": [asdict(server) for server in mcp_servers]
            }
            
            yaml_content = yaml.dump(config_dict, default_flow_style=False, indent=2)
            st.text_area("YAML Configuration", yaml_content, height=300)

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="LLM & MCP Configuration",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🚀 Enhanced LLM Pipeline Configuration")
    st.markdown("Configure your LLM inference endpoints and external data sources (MCP)")
    
    # Sidebar navigation
    with st.sidebar:
        st.header("Navigation")
        section = st.radio(
            "Choose Section",
            ["🤖 LLM Config", "🔌 MCP Servers", "📤 Export", "ℹ️ About"]
        )
    
    # Main content based on selection
    if section == "🤖 LLM Config":
        render_llm_config_section()
    elif section == "🔌 MCP Servers":
        render_mcp_config_section()
    elif section == "📤 Export":
        render_export_section()
    elif section == "ℹ️ About":
        st.header("About This Application")
        st.markdown("""
        This application provides a user-friendly interface for configuring:
        
        **🤖 LLM Inference Servers:**
        - Configure endpoint URLs and authentication
        - Adjust generation parameters (temperature, top-k, etc.)
        - Test connections in real-time
        
        **🔌 MCP (Model Context Protocol) Servers:**
        - Connect to external data sources like GitHub, Notion
        - Configure authentication and workspace access
        - Test connections and manage multiple sources
        
        **📤 Export Options:**
        - Generate environment variables for deployment
        - Export configurations as YAML files
        
        All configurations are saved locally and can be used by your LLM pipeline scripts.
        """)
        
        st.info("💡 **Tip:** Start by configuring your LLM server, then add MCP servers for external data access.")

if __name__ == "__main__":
    main() 