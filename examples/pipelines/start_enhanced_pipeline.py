#!/usr/bin/env python3
"""
Enhanced LLM Pipeline Startup Script
Quick start script for the enhanced LLM pipeline with UI configuration and MCP integration
"""

import os
import sys
import subprocess
import json
from pathlib import Path
import argparse

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'streamlit',
        'langchain',
        'requests',
        'pyyaml'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("📦 Install them with: pip install -r enhanced_requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def create_default_configs():
    """Create default configuration files if they don't exist"""
    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)
    
    # Create default LLM config
    llm_config_file = configs_dir / "llm_config.json"
    if not llm_config_file.exists():
        default_llm_config = {
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
        
        with open(llm_config_file, 'w') as f:
            json.dump(default_llm_config, f, indent=2)
        print(f"📝 Created default LLM configuration: {llm_config_file}")
    
    # Create empty MCP config
    mcp_config_file = configs_dir / "mcp_servers.json"
    if not mcp_config_file.exists():
        with open(mcp_config_file, 'w') as f:
            json.dump([], f, indent=2)
        print(f"📝 Created empty MCP configuration: {mcp_config_file}")

def create_sample_files():
    """Create sample input files if they don't exist"""
    # Sample input text
    if not Path("example_text.txt").exists():
        sample_text = """A car insurance claim has been submitted by John Smith. On October 15, 2023, at 2:30 PM, his Honda Accord was involved in an accident with a Ford Escape at the intersection of Elm Street and Maple Avenue near Smith Park in Springfield, Illinois. The other driver ran a red light and hit the front passenger side of John's vehicle. Both vehicles sustained significant damage. John was not seriously injured and has photos of the scene, witness information, and the other driver's insurance details. He is requesting claim processing under his policy for vehicle damages."""
        
        with open("example_text.txt", 'w') as f:
            f.write(sample_text)
        print("📄 Created sample example_text.txt")
    
    # Sample template
    if not Path("summary_template.txt").exists():
        sample_template = """Based on the following context and external sources, please provide a comprehensive summary of the insurance claim:

Context: {context}

Claim Details: {question}

Please summarize the key information including the incident details, parties involved, damages, and any supporting documentation mentioned. Be thorough and accurate."""
        
        with open("summary_template.txt", 'w') as f:
            f.write(sample_template)
        print("📄 Created sample summary_template.txt")

def launch_config_ui():
    """Launch the Streamlit configuration UI"""
    print("🚀 Launching Configuration UI...")
    print("🌐 The UI will open at: http://localhost:8501")
    print("💡 Use the UI to configure your LLM endpoint and MCP sources")
    print()
    
    try:
        subprocess.run(["streamlit", "run", "config_ui.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch Streamlit UI: {e}")
        return False
    except FileNotFoundError:
        print("❌ Streamlit not found. Install it with: pip install streamlit")
        return False
    
    return True

def run_tests():
    """Run the enhanced test suite"""
    print("🧪 Running Enhanced Test Suite...")
    print("=" * 50)
    
    try:
        subprocess.run([sys.executable, "enhanced_test_response_quality.py"], check=True)
        print("✅ Test suite completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Test suite failed: {e}")
        return False
    except FileNotFoundError:
        print("❌ Test file not found: enhanced_test_response_quality.py")
        return False

def run_basic_test():
    """Run a basic compatibility test"""
    print("🔄 Running Basic Compatibility Test...")
    
    try:
        # Import the enhanced modules
        from enhanced_llm_usage import test_mcp_integration, ConfigurableLLMManager
        
        # Test LLM configuration
        print("⚙️ Testing LLM configuration...")
        llm_manager = ConfigurableLLMManager()
        config = llm_manager.get_config()
        print(f"📡 LLM Endpoint: {config['server_url']}")
        
        # Test MCP integration
        print("🔌 Testing MCP integration...")
        test_mcp_integration()
        
        print("✅ Basic test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        print("💡 This might be normal if you haven't configured endpoints yet")
        return False

def print_usage_guide():
    """Print a usage guide"""
    print("\n" + "="*60)
    print("📚 USAGE GUIDE")
    print("="*60)
    print()
    print("🚀 Quick Start:")
    print("1. python start_enhanced_pipeline.py --setup    # Set up configurations")
    print("2. python start_enhanced_pipeline.py --ui       # Launch configuration UI")
    print("3. Configure your LLM endpoint and MCP sources in the UI")
    print("4. python start_enhanced_pipeline.py --test     # Run tests")
    print()
    print("📋 Available Commands:")
    print("  --setup     Create default configuration files")
    print("  --ui        Launch Streamlit configuration UI")
    print("  --test      Run comprehensive test suite")
    print("  --basic     Run basic compatibility test")
    print("  --deps      Check dependencies")
    print("  --guide     Show this usage guide")
    print()
    print("🔧 Manual Usage:")
    print("# Import in your code")
    print("from enhanced_llm_usage import infer_with_template")
    print("response = infer_with_template('Your question', 'Template: {question}')")
    print()
    print("📖 For detailed documentation, see: ENHANCED_README.md")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Enhanced LLM Pipeline Startup Script")
    parser.add_argument("--setup", action="store_true", help="Set up default configurations")
    parser.add_argument("--ui", action="store_true", help="Launch configuration UI")
    parser.add_argument("--test", action="store_true", help="Run comprehensive test suite")
    parser.add_argument("--basic", action="store_true", help="Run basic compatibility test")
    parser.add_argument("--deps", action="store_true", help="Check dependencies")
    parser.add_argument("--guide", action="store_true", help="Show usage guide")
    
    args = parser.parse_args()
    
    print("🚀 Enhanced LLM Pipeline Startup")
    print("=" * 40)
    
    # If no arguments, show guide
    if not any(vars(args).values()):
        print("💡 No command specified. Here's how to get started:")
        print_usage_guide()
        return
    
    success = True
    
    if args.deps:
        success &= check_dependencies()
    
    if args.setup:
        print("\n⚙️ Setting up configurations...")
        create_default_configs()
        create_sample_files()
        print("✅ Setup completed successfully!")
        print("💡 Next step: Run with --ui to configure your endpoints")
    
    if args.basic:
        print("\n🔄 Running basic test...")
        success &= run_basic_test()
    
    if args.test:
        print("\n🧪 Running comprehensive tests...")
        success &= run_tests()
    
    if args.ui:
        print("\n🎨 Launching Configuration UI...")
        if not check_dependencies():
            return
        success &= launch_config_ui()
    
    if args.guide:
        print_usage_guide()
    
    print("\n" + "="*40)
    if success:
        print("🎉 All operations completed successfully!")
    else:
        print("⚠️ Some operations encountered issues. Check the output above.")
        print("💡 For help, run: python start_enhanced_pipeline.py --guide")

if __name__ == "__main__":
    main() 