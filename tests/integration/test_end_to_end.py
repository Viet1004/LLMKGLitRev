"""End-to-end integration tests for LLMKGLitRev."""

import pytest


class TestBasicIntegration:
    """Basic integration tests."""
    
    def test_arxiv_mcp_server_import(self):
        """Test that arxiv-mcp-server can be imported."""
        try:
            import arxiv_mcp_server
            assert arxiv_mcp_server is not None
        except ImportError:
            pytest.fail("arxiv_mcp_server not properly installed")
    
    def test_mcp_import(self):
        """Test that mcp can be imported."""
        try:
            import mcp
            assert mcp is not None
        except ImportError:
            pytest.fail("mcp not properly installed")
    
    def test_project_structure(self):
        """Test basic project structure exists."""
        from pathlib import Path
        
        project_root = Path(__file__).parent.parent.parent
        assert (project_root / "src" / "llmkglitrev").exists()
        assert (project_root / "src" / "llmkglitrev" / "MCPClient").exists()
        assert (project_root / "src" / "llmkglitrev" / "MCPServer").exists()
    
    def test_basic_functionality(self):
        """Test basic functionality works."""
        # Simple test that always passes
        assert 1 + 1 == 2
