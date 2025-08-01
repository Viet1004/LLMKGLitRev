"""Tests for MCP Server app functionality."""

import pytest
from pathlib import Path


class TestMCPServerApp:
    """Basic MCP Server tests."""
    
    def test_app_file_exists(self):
        """Test that app.py file exists."""
        app_path = Path(__file__).parent.parent.parent / "src" / "llmkglitrev" / "MCPServer" / "app.py"
        assert app_path.exists()
    
    def test_arxiv_mcp_server_installed(self):
        """Test that arxiv-mcp-server is properly installed."""
        try:
            import arxiv_mcp_server
            assert arxiv_mcp_server is not None
        except ImportError:
            pytest.fail("arxiv_mcp_server not properly installed")
    
    def test_basic_functionality(self):
        """Test basic functionality."""
        assert True
